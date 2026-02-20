const dom = {
  modelSelect: document.getElementById("modelSelect"),
  thresholdInput: document.getElementById("thresholdInput"),
  textInput: document.getElementById("textInput"),
  predictTextBtn: document.getElementById("predictTextBtn"),
  textResult: document.getElementById("textResult"),
  fileInput: document.getElementById("fileInput"),
  dropZone: document.getElementById("dropZone"),
  textColumnInput: document.getElementById("textColumnInput"),
  predictFileBtn: document.getElementById("predictFileBtn"),
  previewBody: document.querySelector("#previewTable tbody"),
  downloadLink: document.getElementById("downloadLink"),
  statusText: document.getElementById("statusText"),
  tabTextBtn: document.getElementById("tabTextBtn"),
  tabFileBtn: document.getElementById("tabFileBtn"),
  tabText: document.getElementById("tabText"),
  tabFile: document.getElementById("tabFile"),
  themeToggle: document.getElementById("themeToggle"),
  historyToggle: document.getElementById("historyToggle"),
  historyPanel: document.getElementById("historyPanel"),
  historyOverlay: document.getElementById("historyOverlay"),
  historyClose: document.getElementById("historyClose"),
  historyList: document.getElementById("historyList"),
  historyDetail: document.getElementById("historyDetail"),
};

let models = [];
const HISTORY_KEY = "spamham_predict_history";

function getApiBase() {
  return window.location.origin;
}

function setStatus(message, isError = false) {
  dom.statusText.textContent = message;
  dom.statusText.style.color = isError ? "#c4342a" : "";
}

function setBusy(button, busy) {
  button.disabled = busy;
  if (busy) {
    button.dataset.originalText = button.textContent;
    button.textContent = "Đang xử lý...";
    return;
  }
  button.textContent = button.dataset.originalText || button.textContent;
}

function currentThreshold() {
  const value = Number(dom.thresholdInput.value);
  if (Number.isFinite(value) && value >= 0 && value <= 1) {
    return value;
  }
  return null;
}

function currentModelId() {
  return dom.modelSelect.value;
}

function truncate(text, maxLen = 40) {
  if (!text) {
    return "";
  }
  return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
}

function renderTextResult(result) {
  const scoreText = result.score == null ? "N/A" : result.score.toFixed(4);
  dom.textResult.innerHTML = `
    <p><strong>Nhãn:</strong> ${result.label.toUpperCase()}</p>
    <p><strong>Điểm spam:</strong> ${scoreText}</p>
    <p><strong>Ngưỡng dùng:</strong> ${result.threshold_used}</p>
    <p><strong>Model:</strong> ${result.model_id}</p>
  `;
  dom.textResult.classList.remove("hidden");
}

function renderPreview(rows) {
  dom.previewBody.innerHTML = "";
  for (const row of rows) {
    const tr = document.createElement("tr");
    const score = row.score == null ? "N/A" : Number(row.score).toFixed(4);
    tr.innerHTML = `
      <td>${row.row_id}</td>
      <td>${row.text}</td>
      <td>${row.label}</td>
      <td>${score}</td>
    `;
    dom.previewBody.appendChild(tr);
  }
}

async function fetchModels() {
  const res = await fetch(`${getApiBase()}/models`);
  if (!res.ok) {
    throw new Error("Không lấy được danh sách model.");
  }
  const data = await res.json();
  models = data.models || [];
  if (!models.length) {
    throw new Error("Backend chưa có model trong registry.");
  }

  dom.modelSelect.innerHTML = "";
  for (const model of models) {
    const opt = document.createElement("option");
    opt.value = model.model_id;
    opt.textContent = model.display_name;
    opt.dataset.defaultThreshold = model.default_threshold;
    dom.modelSelect.appendChild(opt);
  }
  dom.thresholdInput.value = String(models[0].default_threshold);
}

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(items) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
}

function addHistory(entry) {
  const items = loadHistory();
  items.unshift(entry);
  saveHistory(items.slice(0, 30));
}

function renderHistoryDetail(item) {
  if (!item) {
    dom.historyDetail.innerHTML = "<p>Chọn một mục để xem chi tiết.</p>";
    return;
  }
  const scoreText = item.score == null ? "N/A" : Number(item.score).toFixed(4);
  let html = `
    <h4>Chi tiết</h4>
    <p><strong>Thời gian:</strong> ${item.time}</p>
    <p><strong>Model:</strong> ${item.model_id}</p>
    <p><strong>Ngưỡng:</strong> ${item.threshold_used}</p>
  `;
  if (item.type === "text") {
    html += `
      <p><strong>Kết quả:</strong> ${item.label.toUpperCase()}</p>
      <p><strong>Điểm spam:</strong> ${scoreText}</p>
      <p><strong>Văn bản:</strong> ${item.text}</p>
    `;
  } else {
    html += `
      <p><strong>File:</strong> ${item.file_name}</p>
      <p><strong>Tổng dòng:</strong> ${item.total_rows}</p>
    `;
    if (item.preview && item.preview.length) {
      html += "<p><strong>Mẫu kết quả:</strong></p><ul>";
      for (const row of item.preview) {
        const rowScore = row.score == null ? "N/A" : Number(row.score).toFixed(4);
        html += `<li>${row.label.toUpperCase()} (${rowScore}) - ${row.text}</li>`;
      }
      html += "</ul>";
    }
  }
  dom.historyDetail.innerHTML = html;
}

function renderHistoryList() {
  const items = loadHistory();
  dom.historyList.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.className = "history-item";
    li.textContent = "Chưa có lịch sử gửi dữ liệu.";
    dom.historyList.appendChild(li);
    renderHistoryDetail(null);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.className = "history-item";
    li.innerHTML = `
      <div class="meta">${item.time} · ${item.type === "file" ? "File" : "Văn bản"}</div>
      <div class="title">${item.title}</div>
    `;
    li.addEventListener("click", () => renderHistoryDetail(item));
    dom.historyList.appendChild(li);
  }
  renderHistoryDetail(items[0]);
}

function toggleHistory(open) {
  if (open) {
    dom.historyPanel.classList.remove("hidden");
    dom.historyOverlay.classList.remove("hidden");
    renderHistoryList();
  } else {
    dom.historyPanel.classList.add("hidden");
    dom.historyOverlay.classList.add("hidden");
  }
}

async function onPredictText() {
  const text = dom.textInput.value.trim();
  if (!text) {
    setStatus("Vui lòng nhập văn bản cần dự đoán.", true);
    return;
  }

  setBusy(dom.predictTextBtn, true);
  setStatus("Đang dự đoán văn bản...");
  try {
    const payload = {
      model_id: currentModelId(),
      text,
      threshold: currentThreshold(),
    };
    const res = await fetch(`${getApiBase()}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Lỗi dự đoán.");
    }
    renderTextResult(data);
    setStatus("Dự đoán thành công.");
    const now = new Date().toLocaleString("vi-VN");
    addHistory({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      type: "text",
      time: now,
      title: `${data.label.toUpperCase()} · ${truncate(text)}`,
      model_id: data.model_id,
      threshold_used: data.threshold_used,
      label: data.label,
      score: data.score,
      text,
    });
    renderHistoryList();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(dom.predictTextBtn, false);
  }
}

async function onPredictFile() {
  const file = dom.fileInput.files?.[0];
  if (!file) {
    setStatus("Vui lòng chọn file .txt/.csv/.xlsx.", true);
    return;
  }

  setBusy(dom.predictFileBtn, true);
  setStatus("Đang xử lý file batch...");
  dom.downloadLink.classList.add("hidden");

  try {
    const form = new FormData();
    form.append("file", file);
    form.append("model_id", currentModelId());
    const threshold = currentThreshold();
    if (threshold != null) {
      form.append("threshold", String(threshold));
    }
    const textColumn = dom.textColumnInput.value.trim();
    if (textColumn) {
      form.append("text_column", textColumn);
    }
    form.append("preview_limit", "20");

    const res = await fetch(`${getApiBase()}/predict-file`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Lỗi xử lý file.");
    }

    renderPreview(data.preview || []);
    dom.downloadLink.href = `${getApiBase()}${data.download_url}`;
    dom.downloadLink.classList.remove("hidden");
    setStatus(`Đã xử lý ${data.total_rows} dòng. Cột dùng: ${data.text_column_used}`);

    const now = new Date().toLocaleString("vi-VN");
    addHistory({
      id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
      type: "file",
      time: now,
      title: `${file.name} · ${data.total_rows} dòng`,
      model_id: data.model_id,
      threshold_used:
        currentThreshold() ??
        (data.preview?.length ? data.preview[0].threshold_used : 0.5),
      file_name: file.name,
      total_rows: data.total_rows,
      preview: (data.preview || []).slice(0, 3).map((row) => ({
        text: row.text,
        label: row.label,
        score: row.score,
      })),
    });
    renderHistoryList();
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    setBusy(dom.predictFileBtn, false);
  }
}

function switchTab(tab) {
  const isText = tab === "text";
  dom.tabTextBtn.classList.toggle("active", isText);
  dom.tabFileBtn.classList.toggle("active", !isText);
  dom.tabText.classList.toggle("active", isText);
  dom.tabFile.classList.toggle("active", !isText);
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  localStorage.setItem("spamham_theme", theme);
}

function initTheme() {
  const saved = localStorage.getItem("spamham_theme") || "light";
  applyTheme(saved);
}

function bindEvents() {
  dom.predictTextBtn.addEventListener("click", onPredictText);
  dom.predictFileBtn.addEventListener("click", onPredictFile);
  dom.tabTextBtn.addEventListener("click", () => switchTab("text"));
  dom.tabFileBtn.addEventListener("click", () => switchTab("file"));
  dom.modelSelect.addEventListener("change", () => {
    const selected = models.find((m) => m.model_id === currentModelId());
    if (selected) {
      dom.thresholdInput.value = String(selected.default_threshold);
    }
  });
  dom.themeToggle.addEventListener("click", () => {
    const nextTheme = document.body.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
  });
  dom.historyToggle.addEventListener("click", () => toggleHistory(true));
  dom.historyClose.addEventListener("click", () => toggleHistory(false));
  dom.historyOverlay.addEventListener("click", () => toggleHistory(false));
  dom.dropZone.addEventListener("click", () => dom.fileInput.click());
  dom.dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dom.dropZone.classList.add("dragging");
  });
  dom.dropZone.addEventListener("dragleave", () => {
    dom.dropZone.classList.remove("dragging");
  });
  dom.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dom.dropZone.classList.remove("dragging");
    const files = event.dataTransfer?.files;
    if (!files || !files.length) {
      return;
    }
    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    dom.fileInput.files = transfer.files;
    setStatus(`Đã chọn file: ${files[0].name}`);
  });
  dom.fileInput.addEventListener("change", () => {
    const file = dom.fileInput.files?.[0];
    if (file) {
      setStatus(`Đã chọn file: ${file.name}`);
    }
  });
}

async function bootstrap() {
  initTheme();
  bindEvents();
  setStatus("Đang kết nối backend...");
  try {
    await fetchModels();
    setStatus("Sẵn sàng dự đoán.");
    renderHistoryList();
  } catch (error) {
    setStatus(error.message, true);
  }
}

bootstrap();

