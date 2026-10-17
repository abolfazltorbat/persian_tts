"use strict";
(() => {
  let currentJob = null, running = false, revealed = false, order = [], ratings = {};
  let configData = null, referenceVoice = null;
  const criteria = {pronunciation: "تلفظ فارسی", intelligibility: "وضوح و کامل بودن متن", naturalness: "طبیعی بودن و لحن", similarity: "شباهت به صدای مرجع"};
  const modelName = m => m === "pocket" ? "Pocket TTS Farsi v2" : "Qwen3-TTS";
  const esc = escapeHtml;

  async function api(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "ورودی نامعتبر یا خطای سرور");
    return data;
  }
  function updateModel() {
    const model = $("modelSelect").value;
    $("qwenWarning").hidden = model === "pocket";
    $("qwenReference").hidden = model === "pocket";
    document.querySelector(".ph-tools").hidden = model !== "pocket";
    $("blindMode").disabled = model !== "compare" || running;
    const qwen = configData?.models.find(m => m.id === "qwen");
    $("modelStatus").textContent = model === "pocket" ? "مدل فعلی فارسی · ONNX روی CPU" : qwen?.ready
      ? "Qwen نصب شده است؛ اولین اجرا شامل بارگذاری مدل است."
      : "Qwen آماده نیست. راهنمای نصب: docs/comparison.md — در مقایسه، نتیجهٔ Pocket همچنان نمایش داده می‌شود.";
  }
  $("modelSelect").onchange = updateModel;
  api("/api/config").then(c => {
    configData = c;
    for (const item of c.cases) {
      const option = document.createElement("option"); option.value = item.id; option.textContent = item.name;
      $("caseSelect").append(option);
    }
    updateModel();
  }).catch(e => { $("modelStatus").textContent = e.message; });
  $("caseSelect").onchange = () => {
    const sample = configData?.cases.find(c => c.id === $("caseSelect").value);
    if (sample) { $("txt").value = sample.text; count(); }
  };
  $("txt").addEventListener("input", () => { $("caseSelect").value = "custom"; });
  $("chips").addEventListener("click", () => { $("caseSelect").value = "custom"; });
  function updateReference() {
    if (!selVoice || selVoice === referenceVoice) return;
    referenceVoice = selVoice;
    $("referencePlayer").src = `/api/reference/${encodeURIComponent(selVoice)}`;
    $("referenceText").value = ""; // a transcript belongs to one specific clip
  }
  window.addEventListener("voicechange", updateReference);
  updateReference();
  $("referencePlayer").addEventListener("play", pauseOthers);
  function pauseOthers(event) {
    document.querySelectorAll("audio").forEach(a => { if (a !== event.target) a.pause(); });
  }
  function setBusy(value) {
    running = value;
    document.querySelectorAll(".grid input,.grid textarea,.grid select,.grid button").forEach(e => { e.disabled = value; });
    $("drop").style.pointerEvents = value ? "none" : "";
    $("go").classList.toggle("busy", value);
    $("revealModels").disabled = value;
    $("exportComparison").disabled = value || !currentJob;
    updateModel();
  }
  function cardTitle(model) {
    return revealed ? modelName(model) : `نمونهٔ ${order.indexOf(model) === 0 ? "A" : "B"}`;
  }
  function renderResult(result) {
    if (document.getElementById(`card-${result.model}`)) return;
    const card = document.createElement("article");
    card.className = "comparison-card"; card.id = `card-${result.model}`;
    card.style.order = order.indexOf(result.model);
    card.innerHTML = `<h3>${esc(cardTitle(result.model))}</h3>`;
    if (result.error) {
      const p = document.createElement("p"); p.className = "error"; p.textContent = result.error; card.append(p);
    } else {
      const player = document.createElement("audio"); player.controls = true; player.preload = "none";
      player.src = result.audio_url; player.addEventListener("play", pauseOthers); card.append(player);
      const link = document.createElement("a"); link.href = result.audio_url;
      link.download = `persian-${result.id}.wav`; link.textContent = "دانلود WAV"; card.append(link);
      const metrics = document.createElement("p"); metrics.className = "model-metrics"; metrics.hidden = !revealed;
      metrics.textContent = `${result.model_name} · ${result.device}\nتولید: ${fa(result.generation_seconds)} ثانیه · بارگذاری: ${fa(result.load_seconds)} ثانیه\nطول صوت: ${fa(result.duration)} ثانیه · RTF: ${fa(result.rtf)} (کمتر، سریع‌تر)\n${(result.warnings || []).join("\n")}`;
      card.append(metrics);
      ratings[result.model] = ratings[result.model] || {};
      for (const [key, label] of Object.entries(criteria)) {
        const row = document.createElement("label"); row.className = "rating-row";
        const caption = document.createElement("span"); caption.textContent = label; row.append(caption);
        const select = document.createElement("select"); select.setAttribute("aria-label", `${label} ${cardTitle(result.model)}`);
        select.innerHTML = '<option value="">بدون امتیاز</option>' + [1,2,3,4,5].map(n => `<option value="${n}">${fa(n)}</option>`).join("");
        select.value = ratings[result.model][key] || "";
        select.onchange = () => { ratings[result.model][key] = select.value ? +select.value : null; saveReport(); };
        row.append(select); card.append(row);
      }
      const notes = document.createElement("textarea"); notes.placeholder = "واژه‌های اشتباه، حذف‌شده یا تغییر لهجه…";
      notes.setAttribute("aria-label", "یادداشت ارزیابی"); notes.maxLength = 2000;
      notes.value = ratings[result.model].notes || "";
      notes.oninput = () => { ratings[result.model].notes = notes.value; saveReport(); }; card.append(notes);
    }
    $("comparisonCards").append(card);
  }
  function report() {
    return {...currentJob, ratings, blind_order: order, names_revealed: revealed,
      rating_scale: {min: 1, max: 5, source: "human listening", criteria},
      rated_at: new Date().toISOString()};
  }
  function saveReport() {
    if (!currentJob) return;
    try { localStorage.setItem("parsigo-last-comparison", JSON.stringify(report())); } catch (_) { /* download still works */ }
  }
  $("revealModels").onclick = () => {
    revealed = true;
    for (const model of order) {
      const card = document.getElementById(`card-${model}`);
      if (!card) continue;
      card.querySelector("h3").textContent = modelName(model);
      const metrics = card.querySelector(".model-metrics"); if (metrics) metrics.hidden = false;
    }
    $("revealModels").disabled = true; saveReport();
  };
  $("exportComparison").onclick = () => {
    if (!currentJob) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(report(), null, 2)], {type: "application/json"}));
    const a = document.createElement("a"); a.href = url; a.download = `persian-comparison-${currentJob.id}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  // Replace the original synchronous handler; existing API consumers remain supported.
  $("go").onclick = async () => {
    if (running) return;
    const text = $("txt").value.trim();
    if (!text) return toast("متنی بنویسید");
    if (!$("seed").checkValidity() || !$("seed").value) return toast("بذر باید یک عدد صحیح بین صفر و ۴۲۹۴۹۶۷۲۹۵ باشد");
    const model = $("modelSelect").value;
    const request = {text, voice: selVoice, model, mode, pace: +paceEl.value,
      seed: +$("seed").value, prepare: $("prepareFa").checked,
      ref_text: $("referenceText").value.trim(), case_id: $("caseSelect").value};
    order = model === "compare" ? (Math.random() < .5 ? ["pocket", "qwen"] : ["qwen", "pocket"]) : [model];
    revealed = model !== "compare" || !$("blindMode").checked;
    ratings = {}; currentJob = null;
    document.querySelectorAll("audio").forEach(a => a.pause());
    $("comparisonCards").replaceChildren(); $("comparison").hidden = false;
    $("result").classList.remove("show"); $("preparedText").textContent = "";
    $("jobStatus").textContent = "در حال شروع…";
    setBusy(true);
    try {
      const job = await api("/api/jobs", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(request)});
      sessionStorage.setItem("parsigo-job", job.id);
      await poll(job.id);
    } catch (e) { $("jobStatus").textContent = e.message; toast(e.message); }
    finally { setBusy(false); $("revealModels").disabled = revealed; }
  };
  async function poll(id) {
    let errors = 0;
    while (true) {
      try { currentJob = await api(`/api/jobs/${id}`); errors = 0; }
      catch (e) { if (++errors >= 4) throw e; await new Promise(r => setTimeout(r, 2000)); continue; }
      $("preparedText").textContent = currentJob.prepared_text;
      $("jobStatus").textContent = revealed ? currentJob.progress : "در حال آماده‌سازی هر دو نمونه؛ نام‌ها پنهان هستند…";
      // Avoid revealing the sequential model order through early playback.
      if (revealed || currentJob.status !== "running") currentJob.results.forEach(renderResult);
      if (currentJob.status !== "running") {
        $("jobStatus").textContent = currentJob.status === "complete" ? "آمادهٔ شنیدن و امتیازدهی" : "پردازش پایان یافت؛ خطای مدل را بررسی کنید";
        sessionStorage.removeItem("parsigo-job"); saveReport(); break;
      }
      await new Promise(r => setTimeout(r, 1500));
    }
  }
  // Refreshing the page must not lose a long-running CPU job.
  const pending = sessionStorage.getItem("parsigo-job");
  if (pending) {
    api(`/api/jobs/${pending}`).then(async job => {
      currentJob = job; order = job.request.model === "compare" ? ["pocket", "qwen"] : [job.request.model];
      revealed = true; $("modelSelect").value = job.request.model;
      $("comparison").hidden = false; setBusy(true);
      try { await poll(pending); } finally { setBusy(false); $("revealModels").disabled = true; }
    }).catch(e => { sessionStorage.removeItem("parsigo-job"); toast(e.message); });
  } else {
    try {
      const saved = JSON.parse(localStorage.getItem("parsigo-last-comparison") || "null");
      if (saved?.id && Array.isArray(saved.results) && Array.isArray(saved.blind_order)) {
        currentJob = saved; ratings = saved.ratings || {}; order = saved.blind_order;
        revealed = Boolean(saved.names_revealed);
        $("comparison").hidden = false; $("preparedText").textContent = saved.prepared_text;
        saved.results.forEach(renderResult);
        $("jobStatus").textContent = "آخرین ارزیابی ذخیره‌شده؛ صداها فقط تا زمان نگهداری در سرور قابل پخش‌اند";
        $("exportComparison").disabled = false; $("revealModels").disabled = revealed;
      }
    } catch (_) { /* unavailable or invalid local storage must not break the app */ }
  }
})();
