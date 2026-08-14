---
title: "Lab GLM-5.2 guarded route witness — 2026-07-05"
description: "A privacy-scrubbed witness for #2953 that the lab's guarded loopback route to GLM-5.2 works: the local proxy returned HTTP 200 and listed the glm-5.2 model."
---

# Lab GLM-5.2 guarded route witness — 2026-07-05

Status: witnessed for #2953; public push is still pending behind shared `main`.

## Scrubbed setup

- Public alias: `@lab/glm-5.2`
- Local private route: `<local-proxy>` (loopback only; private bridge details omitted)
- Model label observed: `glm-5.2`
- Public readiness source: `scrubbed-fleet-report`

No hostnames, node IDs, channel IDs, tokens, raw transcripts, account IDs, or private
filesystem paths are recorded here.

## Checks captured

### 1. Readiness and alias

`fak lab target @lab/glm-5.2 --json` returned:

```json
{
  "schema": "fak.lab_target.resolve/v1",
  "alias": "@lab/glm-5.2",
  "machine_class": "gpu-server",
  "status": "READY_FOR_DEV_WORK",
  "model": "glm-5.2",
  "evidence": "scrubbed-fleet-report",
  "next_action": "use-guard-remote-serve-alias",
  "remote_serve_arg": "@lab/glm-5.2",
  "guard_command": "fak manage --remote-serve @lab/glm-5.2 --probe -- codex"
}
```

### 2. Local/private route

`GET <local-proxy>/healthz` returned HTTP 200.

`GET <local-proxy>/v1/models` returned HTTP 200 and included `glm-5.2`.

### 3. Direct model smoke through the local/private route

`POST <local-proxy>/v1/chat/completions` with a `glm-5.2` request returned:

```json
{"status":200,"has_ok":true,"bytes":323}
```

The response body contained `GLM52_OK`.

### 4. Guarded OpenAI client with a hash-chained guard decision

Command shape:

```bash
fak manage --remote-serve @lab/glm-5.2 --api-key-env <dummy-local-key> --provider openai --probe -- python <smoke-client>
```

The child client used `OPENAI_BASE_URL` injected by `fak manage` and did both in the
same guarded session:

1. posted one OpenAI-compatible chat request to `glm-5.2`;
2. posted one safe `/v1/fak/adjudicate` request for an intentionally unknown
   placeholder tool, proving the local guard journal path without executing a tool.

The child returned:

```json
{
  "adjudicate_elapsed_sec": 0.023,
  "adjudicate_reason": "DEFAULT_DENY",
  "adjudicate_status": 200,
  "adjudicate_verdict": "DENY",
  "chat_bytes": 323,
  "chat_elapsed_sec": 230.4,
  "chat_has_ok": true,
  "chat_status": 200,
  "ok": true
}
```

Guard/audit summary for that run:

- upstream: `<local-proxy>/v1` on the OpenAI wire
- model response observed: `GLM52_OK`
- gateway log recorded `POST /v1/chat/completions` as HTTP 200
- gateway log recorded `POST /v1/fak/adjudicate` as HTTP 200
- guard summary: `1 kernel decision(s) — 0 allowed, 1 denied, 0 repaired, 0 quarantined`
- audit journal: `.dispatch-runs/guard-audit/lab-glm52-combo-20260705-135108.jsonl`
- verification:

```text
fak audit verify .dispatch-runs/guard-audit/lab-glm52-combo-20260705-135108.jsonl
OK: 1 hash-chained row(s), chain intact
```

Scrubbed audit fold:

```json
{
  "audit_rows": 1,
  "kinds": {"DENY": 1},
  "reasons": {"DEFAULT_DENY": 1},
  "tools": {"fak_lab_glm52_smoke_unknown_tool": 1},
  "verdicts": {"DENY": 1}
}
```

The denied placeholder tool was never executed; it exists only to force one
deterministic guard decision row in the same live session that reached lab GLM-5.2.

## Not yet

- The exact unshaped `fak manage --remote-serve @lab/glm-5.2 --probe -- codex` path
  timed out earlier. A shaped Codex smoke subsequently completed under 10 minutes and
  is tracked in #2974.
- The public commits carrying the alias/private-roster support are local only until the
  shared `main` non-fast-forward is reconciled.
