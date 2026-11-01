"""A deterministic, model-independent evaluation harness for the agent tool layer.

A scenario scripts what a patient asks, what a *model* does (tool calls and what it tells the
patient), what the *trusted UI* does (confirming through the real HTTP endpoint), and what the
*world* does meanwhile (another patient booking, time passing). The runner executes it against a
disposable PostgreSQL database, measures it, and enforces safety invariants that no scenario can
switch off. The model is a protocol: today a scripted stub, later a real LLM, judged by the same
runner and the same expectations (docs/evaluation-strategy.md).

Boundaries, enforced by tests:

- The model reaches the backend only through the tool runtime and its fixed allowlist.
- Confirmation happens only in `ui_driver`, over HTTP with a session cookie and CSRF token, exactly
  as the browser would. Nothing in this package imports the commit service.
- Reports and traces carry names, counts, codes, labels, and timings, never credentials, tokens,
  patient names or codes, arguments' private values, or tool result bodies.
"""
