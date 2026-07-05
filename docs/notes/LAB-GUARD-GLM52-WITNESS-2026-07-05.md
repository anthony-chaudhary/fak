# Lab GLM-5.2 guarded route witness — 2026-07-05

Status: partial witness for #2953.

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
  "guard_command": "fak guard --remote-serve @lab/glm-5.2 --probe -- codex"
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

### 4. Guarded minimal OpenAI client

Command shape:

```bash
fak guard --remote-serve @lab/glm-5.2 --api-key-env <dummy-local-key> --provider openai --probe -- python <smoke-client>
```

The child client used `OPENAI_BASE_URL` injected by `fak guard` and posted one
OpenAI-compatible chat request. It returned:

```json
{"status":200,"has_ok":true,"bytes":323}
```

Guard summary for that run:

- upstream: `<local-proxy>/v1` on the OpenAI wire
- model response observed: `GLM52_OK`
- guard audit journal: created, but contained `0` decision rows because the minimal
  smoke client made no tool calls.

## Not yet

- `fak guard --remote-serve @lab/glm-5.2 --probe -- codex` did not complete within a
  20-minute harness timeout. The Codex prompt path is too heavy for the current
  bridge-proxy route and remains open.
- A nonzero guard audit decision row was not captured in this minimal smoke because the
  child made no tool calls.
- The public commits carrying the alias/private-roster support are local only until the
  shared `main` non-fast-forward is reconciled.

