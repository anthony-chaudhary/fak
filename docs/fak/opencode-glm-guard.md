---
title: "fak: guard the opencode/GLM dispatch lane"
description: "How to front the opencode/GLM dispatch worker with fak guard by setting the GLM base URL; live-node activation is deferred."
---

# Guarding the opencode/GLM dispatch lane (issue #730)

Use this when you run the two-lane dispatch fleet and want the **opencode/GLM**
worker to pass through the same `fak guard` decision journal as the Claude worker.
Start with the always-on fleet shape in
[`always-on-dogfood-server.md`](always-on-dogfood-server.md), then use this page
to set the GLM base URL and verify the guarded opencode command.

The fleet-through-kernel change fronts every **claude** dispatch worker with
`fak guard` by default, so the kernel adjudicates every tool call the worker proposes
and records each verdict in a durable decision journal. The **opencode/GLM** lane was
left out on purpose — and this is the doc that closes it.

## Why the GLM lane was unguarded

`tools/dispatch_worker.py::guard_wrap` refuses to guard a non-claude worker until it is
told the worker's upstream:

- The **claude** backend proxies the public Anthropic API (passthrough/subscription)
  with no base-URL override, so `fak guard --provider anthropic -- claude …` is safe by
  default.
- The **opencode** backend fronts a *local* GLM server. `fak guard --provider openai`
  with no base URL would route it to `api.openai.com` — a **misroute**. So `guard_wrap`
  returns the worker command **unchanged** (`guarded=False`) unless
  `FLEET_DOGFOOD_GUARD_BASEURL` names the GLM `/v1` endpoint.

The hook already exists in the code; until #730 **no node set the env**, so the GLM lane
dogfooded **0%**. The fix is config + wiring, not code.

## Step 1 — discover the GLM base URL the opencode worker dials

The opencode worker talks to a local GLM serving endpoint (an OpenAI-compatible
`/v1`). Find it on the node:

```bash
# Probe the usual local ports for an OpenAI-compatible /v1 (returns the first that
# answers /models). The two-lane fleet launcher (tools/issue_resolve_dispatch.py)
# dials this same endpoint.
for p in 8001 8000 8080 11434; do
  curl -sf "http://127.0.0.1:$p/v1/models" >/dev/null && echo "GLM /v1 at http://127.0.0.1:$p/v1"
done
```

`scripts/dogfood-opencode-glm.sh` runs exactly this probe when `FAK_GLM_BASE_URL` is
unset, and prints the URL it found. On a real node, **pin** the discovered value rather
than relying on the probe.

## Step 2 — set `FLEET_DOGFOOD_GUARD_BASEURL` per node

Set it in the node's dispatch config / scheduled-task env (the same place
`FLEET_WORKER_BACKEND=opencode` is set), so every opencode tick is guarded:

```bash
export FLEET_DOGFOOD_GUARD_BASEURL="http://127.0.0.1:8001/v1"   # the discovered GLM /v1
```

Or let the launcher do both — discover, export, and front the worker:

```bash
# dry-run: print the guarded argv and exit (no spawn)
FAK_GLM_BASE_URL=http://127.0.0.1:8001/v1 ./scripts/dogfood-opencode-glm.sh --lane glm

# live: spawn ONE guarded opencode worker on the GLM lane
FAK_GLM_BASE_URL=http://127.0.0.1:8001/v1 ./scripts/dogfood-opencode-glm.sh --lane glm --live
```

## Step 3 — verify the guard engages (acceptance witness)

The dry-run shows the kernel-fronted argv with `guarded:true`:

```bash
FLEET_DOGFOOD_GUARD_BASEURL=http://127.0.0.1:8001/v1 FAK_BIN=$(pwd)/tools/.bin/fak \
  python tools/dispatch_worker.py --lane glm --backend opencode --dry-run --json
```

```json
{
  "guarded": true,
  "command": ["…/fak", "guard", "--provider", "openai",
              "--base-url", "http://127.0.0.1:8001/v1",
              "--audit", ".dispatch-runs/guard-audit/glm-opencode-….jsonl",
              "--", "opencode", "run", "--dangerously-skip-permissions",
              "--agent", "dos-dispatch", "dispatch lane glm"]
}
```

Without the base URL the same command stays unguarded (`guarded:false`) — the no-misroute
default. Both behaviors are pinned by `tools/dispatch_worker_glm_guard_test.py`.

A guarded opencode worker that completes a tick writes a journal under
`.dispatch-runs/guard-audit/<lane>-opencode-*.jsonl`, which `tools/dogfood_coverage.py`
counts toward `audit_journal_evidence` — the same witness that lifts the dogfood
coverage score (issue #731).

## Deferred: the live-node flip

The **mechanism** (launcher, env wiring, doc, tests) lands here. Pointing it at a *real*
GLM server is **hardware-gated** — it needs a GLM serving node up with a reachable `/v1`,
which can't be stood up from the implementing host. On a node that has a live GLM
endpoint: discover the URL (Step 1), pin `FLEET_DOGFOOD_GUARD_BASEURL` (Step 2), run the
launcher `--live`, and confirm a journal row appears (Step 3). That activation is the
remaining step.

## Refs

- `tools/dispatch_worker.py` — `guard_wrap`, `FLEET_DOGFOOD_GUARD_BASEURL`, `guard_provider`
- `scripts/dogfood-opencode-glm.sh` — the per-node discover+export+launch wrapper
- `tools/dispatch_worker_glm_guard_test.py` — the guarded/unguarded contract test
- `tools/dogfood_coverage.py` — counts the journal a guarded worker writes (#731)
