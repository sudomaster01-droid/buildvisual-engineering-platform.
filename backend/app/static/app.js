/* BuildVisual single-page client.
 *
 * Design rules this file follows:
 *  - no external dependencies (works offline, inside a sandboxed frame);
 *  - every state shown is the server's state — no invented percentages;
 *  - the active job is re-fetched on load, so a refresh never loses work;
 *  - one Generate Video button: the three inputs are alternatives, not a wizard.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = {
    capabilities: null,
    csrf: null,
    activeJobId: null,
    video: null,
    pollTimer: null,
    recording: null,
    audioBlob: null,
    startTime: null,
    lastStage: null,
    projectId: null,
  };

  // ---------------------------------------------------------------- helpers
  function toast(message, ms = 3800) {
    const el = $("toast");
    el.textContent = message;
    el.hidden = false;
    clearTimeout(el._t);
    el._t = setTimeout(() => { el.hidden = true; }, ms);
  }

  function cookie(name) {
    return document.cookie.split("; ").reduce((acc, part) => {
      const [k, ...rest] = part.split("=");
      return k === name ? decodeURIComponent(rest.join("=")) : acc;
    }, "");
  }

  async function api(path, options = {}) {
    const opts = { credentials: "same-origin", ...options };
    opts.headers = { ...(options.headers || {}) };
    if (opts.body && !(opts.body instanceof FormData)) {
      opts.headers["Content-Type"] = "application/json";
    }
    const method = (opts.method || "GET").toUpperCase();
    if (!["GET", "HEAD"].includes(method)) {
      const token = state.csrf || cookie("bv_csrf");
      if (token) opts.headers["X-CSRF-Token"] = token;
    }
    const res = await fetch(path, opts);
    let payload = null;
    const text = await res.text();
    if (text) { try { payload = JSON.parse(text); } catch { payload = { raw: text }; } }
    if (!res.ok) {
      const err = new Error((payload && payload.error && payload.error.message) || `Request failed (${res.status})`);
      err.status = res.status;
      err.code = payload && payload.error && payload.error.code;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  function show(el, visible) { if (el) el.hidden = !visible; }

  function humanDuration(seconds) {
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    return m ? `${m}m ${s % 60}s` : `${s}s`;
  }

  function formatBytes(bytes) {
    if (!bytes) return "";
    const mb = bytes / (1024 * 1024);
    return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
  }

  // ------------------------------------------------------------ capabilities
  async function loadCapabilities() {
    try {
      const caps = await api("/api/capabilities");
      state.capabilities = caps;
      const providerExternal = caps.video_provider_external;
      $("capabilityNote").innerHTML =
        `<strong>${escapeHtml(caps.video_note)}</strong>` +
        ` Drawings: ${caps.accepted_drawing_formats.join(", ")}. Voice notes up to ` +
        `${Math.round(caps.max_audio_seconds / 60)} min. ` +
        `Films are ${caps.max_shots} scenes of about ${caps.shot_seconds}s.`;
      $("drawingHint").textContent =
        `${caps.accepted_drawing_formats.join(", ")} · up to ${caps.max_upload_mb} MB`;
      $("footProvider").textContent =
        `${caps.video_provider_external ? "AI video" : "Model render"} · ${caps.video_provider}`;

      const limits = caps.limitations || [];
      if (!caps.drawing_interpretation) {
        $("drawingHint").textContent =
          "Reading drawings needs a vision model that is not configured here — use Speak or Type.";
      }
      if (!caps.transcription) {
        $("recordState").textContent =
          "Transcription is not configured on this deployment — please type the requirements.";
        $("recordBtn").disabled = true;
      }
      if (limits.length) {
        $("capabilityNote").title = limits.join("\n");
      }
    } catch (err) {
      $("capabilityNote").textContent =
        "Capabilities could not be loaded. The server may be starting up.";
    }
  }

  function escapeHtml(value) {
    return String(value || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  // ------------------------------------------------------------------- auth
  // One email field, one password field, one button. Every failure has its own
  // explanation, and a network problem is never reported as a wrong password.
  const authState = { busy: false, setupRequired: false };

  function setAuthError(message, { help = null } = {}) {
    const errEl = $("authError");
    const helpEl = $("authHelp");
    if (message) {
      errEl.textContent = message;
      errEl.hidden = false;
    } else {
      errEl.hidden = true;
    }
    if (help) {
      helpEl.innerHTML = help;
      helpEl.hidden = false;
    } else {
      helpEl.hidden = true;
    }
  }

  function setAuthBusy(busy, label) {
    authState.busy = busy;
    const btn = $("loginBtn");
    btn.classList.toggle("loading", busy);
    btn.disabled = busy;
    btn.setAttribute("aria-busy", String(busy));
    $("loginBtnLabel").textContent = busy ? (label || "Signing in…") : "Login";
    $("email").disabled = busy;
    $("password").disabled = busy;
  }

  function showAuthModal(visible) {
    const modal = $("authModal");
    show(modal, visible);
    if (visible) {
      // Preserve whatever address is already typed, otherwise focus sensibly.
      setTimeout(() => {
        const email = $("email");
        if (email.value) $("password").focus();
        else email.focus();
      }, 60);
    }
  }

  function showSetupNotice(status) {
    const notice = $("setupNotice");
    if (status && status.setup_required) {
      authState.setupRequired = true;
      notice.innerHTML =
        "This deployment has no engineer account yet. Set " +
        "<code>ENGINEER_EMAIL</code> and <code>ENGINEER_PASSWORD</code> in the server " +
        "configuration and restart, or run <code>python -m app.cli set-password</code>.";
      notice.className = "notice warn";
      notice.hidden = false;
      if (status.email_hint && !$("email").value) $("email").value = status.email_hint;
      $("authIntro").textContent = "Account setup is required before signing in.";
    } else {
      authState.setupRequired = false;
      notice.hidden = true;
      $("authIntro").textContent = "One engineer account. Enter your email and password.";
    }
  }

  function activateSession(session) {
    state.csrf = session.csrf_token || cookie("bv_csrf");
    $("accountBtn").hidden = false;
    $("accountBtn").onclick = signOut;
    showAuthModal(false);
    $("password").value = "";
  }

  async function checkSession() {
    try {
      const status = await api("/api/auth/status");
      if (status.authenticated) {
        activateSession(status);
        return true;
      }
      showSetupNotice(status);
      showAuthModal(true);
      return false;
    } catch (err) {
      // Server unreachable or still starting: say so instead of implying bad details.
      showAuthModal(true);
      setAuthError(
        "Cannot reach the BuildVisual server right now. Check your connection, then try again.",
        { help: "If this persists, the server may be restarting. Your work is stored on the server." }
      );
      return false;
    }
  }

  async function signOut() {
    try { await api("/api/auth/logout", { method: "POST" }); } catch { /* ignore */ }
    state.csrf = null;
    state.activeJobId = null;
    location.reload();
  }

  $("togglePassword").addEventListener("click", () => {
    const input = $("password");
    const revealed = input.type === "text";
    input.type = revealed ? "password" : "text";
    const button = $("togglePassword");
    button.setAttribute("aria-pressed", String(!revealed));
    button.setAttribute("aria-label", revealed ? "Show password" : "Hide password");
    $("eyeSlash").setAttribute("visibility", revealed ? "hidden" : "visible");
    input.focus();
  });

  $("forgotBtn").addEventListener("click", async () => {
    setAuthError(null);
    try {
      const result = await api("/api/auth/password-reset", { method: "POST" });
      setAuthError(null, {
        help:
          escapeHtml(result.message).replace(
            /(python -m app\.cli set-password)/,
            "<code>$1</code>"
          ),
      });
    } catch {
      setAuthError(null, {
        help:
          "Password recovery is handled by whoever operates this deployment: run " +
          "<code>python -m app.cli set-password</code> on the server, then sign in again.",
      });
    }
  });

  $("authForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (authState.busy) return; // guards against double submission

    // Trim the address, never the password: leading/trailing spaces can be part
    // of a real password, but are always an accident in an email field.
    const email = $("email").value.trim();
    const password = $("password").value;

    if (!email || !password) {
      setAuthError(
        !email && !password
          ? "Enter your email address and password."
          : !email
            ? "Enter your email address."
            : "Enter your password."
      );
      (!email ? $("email") : $("password")).focus();
      return;
    }

    setAuthError(null);
    setAuthBusy(true);
    try {
      const session = await api("/api/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      activateSession(session);
      await Promise.all([loadCapabilities(), loadProjects(), recoverActiveJob()]);
      toast(`Signed in as ${session.display_name || session.email}.`);
    } catch (err) {
      // Distinguish transport failures from credential failures explicitly.
      const isNetwork = err instanceof TypeError || err.status === undefined;
      if (isNetwork) {
        setAuthError(
          "Could not reach the server. Check your connection and try again.",
          { help: "Your email has been kept so you only need to re-enter the password." }
        );
      } else if (err.code === "rate_limited") {
        setAuthError(err.message, {
          help: "For security, sign-in attempts are limited. Wait for the time shown above.",
        });
      } else if (err.status === 401) {
        setAuthError(err.message || "Those sign-in details were not recognised.");
        $("password").value = "";
        $("password").focus();
      } else if (err.status >= 500) {
        setAuthError(
          "The server could not complete the sign-in. It has logged the problem — please try again.",
        );
      } else {
        setAuthError(err.message || "Sign-in failed. Please try again.");
      }
      // The email stays in the field on every path above.
    } finally {
      setAuthBusy(false);
    }
  });

  // ------------------------------------------------------------------ tabs
  const tabs = ["drawing", "voice", "text"];
  tabs.forEach((name) => {
    $(`tab-${name}`).addEventListener("click", () => selectTab(name));
  });

  function selectTab(name) {
    tabs.forEach((other) => {
      const selected = other === name;
      $(`tab-${other}`).setAttribute("aria-selected", String(selected));
      show($(`panel-${other}`), selected);
    });
  }

  // -------------------------------------------------------------- drawing
  const dropZone = $("dropZone");
  const drawingInput = $("drawingInput");
  let drawingFile = null;

  function setDrawing(file) {
    drawingFile = file || null;
    const chosen = $("drawingChosen");
    if (file) {
      chosen.textContent = `${file.name} · ${formatBytes(file.size)}`;
      chosen.hidden = false;
    } else {
      chosen.hidden = true;
    }
  }

  drawingInput.addEventListener("change", () => setDrawing(drawingInput.files[0]));
  ["dragenter", "dragover"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("dragging"); })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropZone.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove("dragging"); })
  );
  dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setDrawing(file);
  });

  // ------------------------------------------------------------------ text
  const textInput = $("textInput");
  textInput.addEventListener("input", () => {
    $("textCount").textContent = String(textInput.value.length);
  });

  // ----------------------------------------------------------------- voice
  const recordBtn = $("recordBtn");
  recordBtn.addEventListener("click", async () => {
    const active = recordBtn.getAttribute("aria-pressed") === "true";
    if (active) {
      state.recording.stop();
      return;
    }
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      $("recordState").textContent = "This browser cannot record audio. Please type the requirements.";
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks = [];
      const recorder = new MediaRecorder(stream);
      state.recording = recorder;
      recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(chunks, { type: chunks[0] ? chunks[0].type : "audio/webm" });
        state.audioBlob = blob;
        const playback = $("recordPlayback");
        playback.src = URL.createObjectURL(blob);
        playback.hidden = false;
        recordBtn.setAttribute("aria-pressed", "false");
        $("recordLabel").textContent = "Record again";
        $("recordState").textContent = `Recorded ${humanDuration(blob.size / 8000)} of audio.`;
      };
      recorder.start();
      recordBtn.setAttribute("aria-pressed", "true");
      $("recordLabel").textContent = "Stop recording";
      $("recordState").textContent = "Recording… speak naturally, then press stop.";
    } catch (err) {
      $("recordState").textContent =
        "Microphone access was refused. You can still type the requirements.";
    }
  });

  // -------------------------------------------------------------- generate
  $("generateBtn").addEventListener("click", generate);
  $("composerError").hidden = true;

  async function generate() {
    const errEl = $("composerError");
    errEl.hidden = true;
    const text = textInput.value.trim();
    const chosenProject = $("projectSelect").value;
    const projectName = $("projectName").value.trim();

    if (!drawingFile && !state.audioBlob && !text) {
      errEl.textContent = "Add a drawing, record a brief, or type the requirements first.";
      errEl.hidden = false;
      return;
    }
    const form = new FormData();
    if (drawingFile) form.append("drawing", drawingFile);
    if (state.audioBlob) form.append("audio", state.audioBlob, "brief.webm");
    if (text) form.append("text", text);
    if (chosenProject) form.append("project_id", chosenProject);
    else if (projectName) form.append("project_name", projectName);

    const button = $("generateBtn");
    button.disabled = true;
    button.textContent = "Starting…";
    try {
      const idempotency = (crypto.randomUUID && crypto.randomUUID()) ||
        `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const result = await api("/api/jobs", {
        method: "POST",
        body: form,
        headers: { "Idempotency-Key": idempotency },
      });
      state.activeJobId = result.job.id;
      state.video = null;
      show($("videoCard"), false);
      show($("rectifyCard"), false);
      renderJob(result.job);
      startPolling();
    } catch (err) {
      errEl.textContent = err.message;
      errEl.hidden = false;
      if (err.code === "capability_unavailable") {
        toast(err.message, 7000);
      }
    } finally {
      button.disabled = false;
      button.textContent = "Generate Video";
    }
  }

  // ------------------------------------------------------------- progress
  function renderJob(job) {
    show($("jobCard"), true);
    $("jobTitle").textContent =
      job.kind === "rectify" ? "Applying your correction" : "Generating your walkthrough";
    $("jobMessage").textContent = job.current_message;
    $("jobMessage").className = "jobmessage" + (job.message_level === "error" ? " error" : "");
    state.startTime = state.startTime || Date.now() - (job.elapsed_seconds || 0) * 1000;
    $("jobElapsed").textContent = `${humanDuration(job.elapsed_seconds || 0)} elapsed`;

    const rail = $("progressRail");
    rail.innerHTML = "";
    (job.progress || []).forEach((stage) => {
      const li = document.createElement("li");
      li.className = stage.state;
      const marker = document.createElement("span");
      marker.className = "marker";
      marker.textContent = stage.state === "done" ? "✓" : stage.state === "failed" ? "!" : "";
      const label = document.createElement("span");
      label.textContent = stage.label;
      li.append(marker, label);
      rail.append(li);
    });

    const list = $("eventList");
    list.innerHTML = "";
    (job.events || []).slice().reverse().forEach((event) => {
      const li = document.createElement("li");
      li.className = event.level;
      const when = document.createElement("span");
      when.className = "when";
      when.textContent = new Date(event.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
      const msg = document.createElement("span");
      msg.textContent = event.message;
      li.append(when, msg);
      list.append(li);
    });

    const running = !["ready", "failed", "canceled"].includes(job.status);
    show($("cancelBtn"), running);
    const failed = job.status === "failed";
    show($("retryBtn"), failed && job.retryable === true);
    if (failed) {
      $("jobMessage").textContent = job.error ? job.error.message : job.current_message;
      $("jobMessage").className = "jobmessage error";
    }
    if (job.status === "needs_input" && job.clarification) {
      $("jobMessage").textContent = job.clarification.question || job.current_message;
    }
    if (job.video) showVideo(job.video);
    if (job.cost && job.cost.spent_cents) {
      $("jobElapsed").textContent =
        `${humanDuration(job.elapsed_seconds || 0)} · ${(job.cost.spent_cents / 100).toFixed(2)} USD estimated`;
    }
  }

  function startPolling() {
    clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollJob, 1800);
    pollJob();
  }

  async function pollJob() {
    if (!state.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${state.activeJobId}`);
      renderJob(job);
      if (["ready", "failed", "canceled", "needs_input"].includes(job.status)) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        if (job.project_id) loadVersions(job.project_id);
        if (job.status === "needs_input") {
          toast(job.clarification && job.clarification.question
            ? job.clarification.question
            : "This job needs one more detail.", 8000);
        }
        if (job.status === "failed") {
          toast(job.error ? job.error.message : "The job failed.", 8000);
        }
      }
    } catch (err) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function recoverActiveJob() {
    try {
      const [active, projects] = await Promise.all([api("/api/jobs/active"), api("/api/projects")]);
      if (active) {
        state.activeJobId = active.id;
        renderJob(active);
        startPolling();
      } else if (projects.length) {
        await loadVersions(projects[0].id);
      }
    } catch { /* signed out or no jobs yet */ }
  }

  // ----------------------------------------------------------------- video
  function showVideo(video) {
    if (!video) return;
    state.video = video;
    show($("videoCard"), true);
    const player = $("player");
    const badge = $("videoBadge");
    badge.textContent =
      `${video.generation_mode === "external_ai" ? "AI generated" : "Model render"} · v${video.version}`;
    badge.className = "badge" + (video.qa_passed ? "" : " warn");
    $("videoTitle").textContent = video.title || "Your walkthrough";
    $("videoMeta").textContent =
      `Version ${video.version} · ${humanDuration(video.duration_s)} · ` +
      `${video.width}×${video.height} · ${formatBytes(video.size_bytes)}` +
      (video.qa_passed ? " · passed quality checks" : " · quality review required");
    player.src = `/api/videos/${video.id}/stream.mp4`;
    player.poster = video.has_poster ? `/api/videos/${video.id}/poster` : "";
    $("downloadBtn").onclick = (event) => { event.preventDefault(); downloadVideo(video.id); };
    $("shareVersion").textContent = String(video.version);
    show($("rectifyCard"), true);
    $("rectifyCard").dataset.videoId = video.id;
  }

  async function downloadVideo(videoId) {
    try {
      const info = await api(`/api/videos/${videoId}/download`);
      window.location.href = info.url;
    } catch (err) { toast(err.message); }
  }

  $("rectifyBtn").addEventListener("click", () => {
    show($("rectifyCard"), true);
    $("rectifyCard").scrollIntoView({ behavior: "smooth", block: "start" });
    $("rectifyText").focus();
  });

  $("saveBtn").addEventListener("click", async () => {
    if (!state.video) return;
    try {
      await api(`/api/videos/${state.video.id}/save`, { method: "POST" });
      toast("Saved to the project. Every version stays available in History.");
    } catch (err) {
      toast(err.message);
    }
  });

  $("shareBtn").addEventListener("click", async () => {
    if (!state.video) return;
    const box = $("shareBox");
    if (!box.hidden) { show(box, false); return; }
    try {
      const share = await api(`/api/videos/${state.video.id}/share`, {
        method: "POST",
        body: JSON.stringify({ expires_days: 7 }),
      });
      $("shareUrl").value = share.url;
      box.dataset.shareId = share.id;
      show(box, true);
    } catch (err) { toast(err.message); }
  });

  $("copyShareBtn").addEventListener("click", async () => {
    const input = $("shareUrl");
    try {
      await navigator.clipboard.writeText(input.value);
      toast("Link copied.");
    } catch {
      input.select();
      toast("Press Ctrl/Cmd+C to copy the link.");
    }
  });

  $("revokeShareBtn").addEventListener("click", async () => {
    const box = $("shareBox");
    const id = box.dataset.shareId;
    if (!id) return;
    try {
      await api(`/api/shares/${id}`, { method: "DELETE" });
      show(box, false);
      toast("Share link revoked.");
    } catch (err) { toast(err.message); }
  });

  // --------------------------------------------------------------- rectify
  $("rectifyChips").addEventListener("click", (event) => {
    const chip = event.target.closest(".chip");
    if (!chip) return;
    $("rectifyText").value = chip.dataset.text;
    $("rectifyText").focus();
  });

  $("rectifySubmit").addEventListener("click", async () => {
    const videoId = $("rectifyCard").dataset.videoId;
    const text = $("rectifyText").value.trim();
    const errEl = $("rectifyError");
    errEl.hidden = true;
    if (!videoId) return;
    if (text.length < 3) {
      errEl.textContent = "Describe the change you want.";
      errEl.hidden = false;
      return;
    }
    const button = $("rectifySubmit");
    button.disabled = true;
    button.textContent = "Starting…";
    try {
      const result = await api(`/api/videos/${videoId}/rectify`, {
        method: "POST",
        body: JSON.stringify({ text }),
      });
      const understanding = result.understanding || {};
      $("rectifyUnderstanding").innerHTML =
        `Understood as <strong>${escapeHtml(understanding.category || "change")}</strong>` +
        (understanding.requires_new_model ? " · rebuilds the 3D model" : " · keeps the approved model");
      state.activeJobId = result.job.id;
      state.startTime = null;
      renderJob(result.job);
      startPolling();
      $("rectifyText").value = "";
    } catch (err) {
      errEl.textContent = err.message;
      errEl.hidden = false;
    } finally {
      button.disabled = false;
      button.textContent = "Apply correction";
    }
  });

  // --------------------------------------------------------------- history
  async function loadProjects() {
    try {
      const projects = await api("/api/projects");
      const select = $("projectSelect");
      select.innerHTML = '<option value="">New project</option>';
      projects.forEach((project) => {
        const option = document.createElement("option");
        option.value = project.id;
        option.textContent = `${project.name} (${project.video_count} video${project.video_count === 1 ? "" : "s"})`;
        select.append(option);
      });
      const list = $("projectList");
      list.innerHTML = "";
      projects.forEach((project) => {
        const li = document.createElement("li");
        const left = document.createElement("div");
        left.innerHTML = `<strong>${escapeHtml(project.name)}</strong>
          <div class="pmeta">${project.video_count} version${project.video_count === 1 ? "" : "s"} · ${project.status}</div>`;
        const open = document.createElement("button");
        open.type = "button";
        open.className = "ghost";
        open.textContent = "Open";
        open.onclick = () => openProject(project);
        li.append(left, open);
        list.append(li);
      });
      show($("projectCard"), projects.length > 0);
    } catch { /* not signed in yet */ }
  }

  async function openProject(project) {
    $("projectSelect").value = project.id;
    state.projectId = project.id;
    await loadVersions(project.id);
    show($("projectCard"), false);
    show($("historyCard"), true);
  }

  async function loadVersions(projectId) {
    try {
      const videos = await api(`/api/projects/${projectId}/videos`);
      const list = $("versionList");
      list.innerHTML = "";
      videos.forEach((video, index) => {
        const li = document.createElement("li");
        if (index === 0) li.classList.add("current");
        const left = document.createElement("div");
        left.innerHTML = `<strong>Version ${video.version}</strong>
          <div class="vmeta">${humanDuration(video.duration_s)} ·
          ${video.generation_mode === "external_ai" ? "AI generated" : "model render"} ·
          ${new Date(video.created_at).toLocaleString()} ·
          ${video.qa_passed ? "passed checks" : "needs review"}</div>`;
        const actions = document.createElement("div");
        const watch = document.createElement("button");
        watch.type = "button";
        watch.className = "ghost";
        watch.textContent = index === 0 ? "View" : "Open";
        watch.onclick = () => showVideo(video);
        actions.append(watch);
        li.append(left, actions);
        list.append(li);
      });
      show($("historyCard"), videos.length > 0);
    } catch { /* ignore */ }
  }

  $("historyBtn").addEventListener("click", async () => {
    await loadProjects();
    const card = $("projectCard");
    show(card, true);
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  $("cancelBtn").addEventListener("click", async () => {
    if (!state.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${state.activeJobId}/cancel`, { method: "POST" });
      renderJob(job);
      toast("Job canceled.");
    } catch (err) { toast(err.message); }
  });

  $("retryBtn").addEventListener("click", async () => {
    if (!state.activeJobId) return;
    try {
      const job = await api(`/api/jobs/${state.activeJobId}/retry`, { method: "POST" });
      renderJob(job);
      startPolling();
    } catch (err) { toast(err.message); }
  });

  // ------------------------------------------------------------------ boot
  (async function boot() {
    const signedIn = await checkSession();
    if (!signedIn) return;
    await loadCapabilities();
    await loadProjects();
    await recoverActiveJob();
  })();
})();
