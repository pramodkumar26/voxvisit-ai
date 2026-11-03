(function () {
  "use strict";

  // The ticket arrives in the URL fragment, which is never sent to the server. It is read once,
  // removed from the address bar and history entry, and kept only in this variable.
  var hash = new URLSearchParams(window.location.hash.slice(1));
  var ticket = hash.get("h");
  var codeHint = hash.get("p");
  window.history.replaceState(null, "", window.location.pathname);

  var csrf = null;
  var review = null;
  var confirmKey = null;
  var busy = false;
  var timer = null;

  function $(id) { return document.getElementById(id); }

  function say(text, kind) {
    var status = $("status");
    status.textContent = text;
    status.className = kind || "";
  }

  function show(id, visible) { $(id).hidden = !visible; }

  async function api(path, options) {
    var init = { method: (options && options.method) || "GET", credentials: "same-origin",
                 headers: {} };
    if (options && options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    if (options && options.headers) { Object.assign(init.headers, options.headers); }
    if (init.method !== "GET" && csrf) { init.headers["X-CSRF-Token"] = csrf; }
    var response;
    try {
      response = await fetch(path, init);
    } catch (e) {
      return { status: 0, ok: false, error: { code: "NETWORK" } };
    }
    var data = null;
    try { data = await response.json(); } catch (e) { data = null; }
    return { status: response.status, ok: response.ok, data: data,
             error: data && data.error ? data.error : null };
  }

  // --- what each failure means to the patient -------------------------------------------------

  function explain(result) {
    var error = result.error || {};
    var reason = error.reason || "";
    if (result.status === 0) { return "Could not reach the server. Check that it is running."; }
    if (result.status === 401) { return "Your session has ended. Please sign in again."; }
    if (error.code === "HANDOFF_INVALID") { return error.message; }
    if (error.code === "SLOT_UNAVAILABLE") {
      return "That time is no longer available. Nothing was changed. Ask the assistant to search again.";
    }
    if (reason === "CONFIRMATION_EXPIRED") {
      return "This review expired, so nothing was changed. Ask the assistant to prepare it again.";
    }
    if (reason === "CONFIRMATION_STALE") {
      return "The details changed after this review was prepared. Nothing was changed. " +
             "Ask the assistant to prepare it again.";
    }
    if (reason === "CONFIRMATION_INVALIDATED") {
      return "This review is no longer valid, so nothing was changed. Ask the assistant again.";
    }
    if (reason === "CONFIRMATION_CONSUMED") {
      return "This review was already used. Check your appointments.";
    }
    if (result.status === 403) { return "That request was refused. Please reload and sign in again."; }
    if (result.status === 429) { return "Too many attempts. Please wait a moment and try again."; }
    return error.message || "Something went wrong. Nothing was changed.";
  }

  // --- showing a review -----------------------------------------------------------------------

  var ACTIONS = { BOOK: "Book an appointment", RESCHEDULE: "Move an appointment",
                  CANCEL: "Cancel an appointment" };

  function row(list, label, value) {
    var dt = document.createElement("dt");
    dt.textContent = label;
    var dd = document.createElement("dd");
    dd.textContent = value;
    list.appendChild(dt);
    list.appendChild(dd);
  }

  function group(list, label) {
    var heading = document.createElement("div");
    heading.className = "group";
    heading.textContent = label;
    list.appendChild(heading);
  }

  function dayLabel(iso) {
    var parts = iso.split("-").map(Number);
    var day = new Date(parts[0], parts[1] - 1, parts[2], 12);
    return day.toLocaleDateString(undefined,
      { weekday: "long", year: "numeric", month: "long", day: "numeric" });
  }

  function appointment(list, when) {
    row(list, "Clinic", when.clinic);
    row(list, "Doctor", when.doctor);
    row(list, "Specialty", when.specialty);
    row(list, "Date", dayLabel(when.date));
    row(list, "Time", when.time + " (" + when.timezone + ")");
  }

  function render(view) {
    review = view;
    confirmKey = confirmKey || "review-" + window.crypto.randomUUID();
    var list = $("details");
    list.textContent = "";
    row(list, "Action", ACTIONS[view.action] || view.action);
    var summary = view.summary;
    if (view.action === "RESCHEDULE") {
      group(list, "Current appointment");
      appointment(list, summary.current);
      group(list, "New appointment");
      appointment(list, summary.proposed);
    } else if (view.action === "CANCEL") {
      appointment(list, summary.current);
    } else {
      appointment(list, summary.proposed);
    }
    show("signin", false);
    show("review", true);
    show("done", false);
    say("Please check the details, then confirm or decline.");
    countdown();
  }

  function countdown() {
    window.clearInterval(timer);
    var expires = new Date(review.expires_at).getTime();
    function tick() {
      var left = Math.max(0, Math.round((expires - Date.now()) / 1000));
      if (left === 0) {
        window.clearInterval(timer);
        $("expiry").textContent = "This review has expired.";
        $("confirm").disabled = true;
        $("decline").disabled = true;
        say("This review expired, so nothing was changed. Ask the assistant to prepare it again.",
            "error");
        return;
      }
      $("expiry").textContent = "Expires in " + Math.floor(left / 60) + ":" +
        String(left % 60).padStart(2, "0");
    }
    tick();
    timer = window.setInterval(tick, 1000);
  }

  function finish(title, kind, detail) {
    window.clearInterval(timer);
    show("review", false);
    show("done", true);
    $("done-title").textContent = title;
    var list = $("done-details");
    list.textContent = "";
    if (detail) { appointment(list, detail); }
    say(title, kind);
    review = null;
  }

  // --- the actions ----------------------------------------------------------------------------

  async function confirm() {
    if (busy || !review) { return; }
    busy = true;
    $("confirm").disabled = true;
    $("decline").disabled = true;
    say("Confirming…");
    var result = await api("/me/appointment-actions/" + review.pending_action_id + "/confirm", {
      method: "POST", body: { confirmed: true }, headers: { "Idempotency-Key": confirmKey } });
    busy = false;
    if (result.ok) {
      var verb = { BOOKED: "Your appointment is booked.", RESCHEDULED: "Your appointment was moved.",
                   CANCELLED: "Your appointment was cancelled." }[result.data.result] ||
                 "Done.";
      finish(verb, "ok", result.data.appointment);
      return;
    }
    if (result.status === 401) { review = null; await signInNeeded(); return; }
    // The same key makes a retry after a lost reply safe; any definite refusal ends the review.
    if (result.status === 0 || result.status >= 500) {
      $("confirm").disabled = false;
      $("decline").disabled = false;
      say(explain(result) + " You can try Confirm again.", "error");
      return;
    }
    finish(explain(result), "error");
  }

  async function decline() {
    if (busy || !review) { return; }
    busy = true;
    $("confirm").disabled = true;
    $("decline").disabled = true;
    var result = await api("/me/voice-handoffs/decline", { method: "POST" });
    busy = false;
    if (result.ok) {
      finish("Declined. Nothing was changed.", "ok");
    } else if (result.status === 401) {
      review = null;
      await signInNeeded();
    } else {
      $("confirm").disabled = false;
      $("decline").disabled = false;
      say(explain(result), "error");
    }
  }

  // --- signing in, then claiming the ticket ---------------------------------------------------

  async function signInNeeded() {
    csrf = null;
    show("review", false);
    show("done", false);
    show("signin", true);
    if (codeHint && !$("patient-code").value) { $("patient-code").value = codeHint; }
    say(ticket || review ? "Sign in to continue." : "Sign in to see your review.");
    $(codeHint ? "pin" : "patient-code").focus();
  }

  async function signIn(event) {
    event.preventDefault();
    var pin = $("pin");
    var body = { patient_code: $("patient-code").value.trim(), pin: pin.value };
    pin.value = "";
    say("Signing in…");
    var result = await api("/auth/patient/login", { method: "POST", body: body });
    body = null;
    if (!result.ok) {
      say(result.status === 429 ? explain(result) : "The patient code or PIN was not accepted.",
          "error");
      return;
    }
    csrf = result.data.csrf_token;
    await load();
  }

  async function load() {
    if (ticket) {
      say("Opening your review…");
      var claimed = await api("/me/voice-handoffs/claim", { method: "POST", body: { code: ticket } });
      ticket = null; // single use, whatever the outcome
      if (claimed.ok) { render(claimed.data); return; }
      if (claimed.status === 401) { await signInNeeded(); return; }
      finish(explain(claimed), "error");
      return;
    }
    var current = await api("/me/voice-handoffs/review");
    if (current.ok) { render(current.data); return; }
    if (current.status === 401) { await signInNeeded(); return; }
    finish("There is no review waiting. Ask the assistant to prepare a change.", "");
  }

  async function start() {
    $("signin").addEventListener("submit", signIn);
    $("confirm").addEventListener("click", confirm);
    $("decline").addEventListener("click", decline);
    var session = await api("/auth/session");
    if (session.ok) {
      csrf = session.data.csrf_token;
      await load();
    } else {
      await signInNeeded();
    }
  }

  start();
})();
