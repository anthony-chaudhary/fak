# Harness web UI operating-envelope witness — 2026-08-15

Issue: #6790 follow-on to the shipped #6882 spine.

## Captured live browser renders

Playwright Desktop Chrome captured the running loopback product at `http://127.0.0.1:8791`:

| Surface | Query | Artifact |
|---|---|---|
| Normal message/tool/artifact run | `?scenario=normal` | [`normal.png`](../_witnesses/harness-web-demo/normal.png) |
| Scoped approval pending | `?scenario=approval` | [`approval.png`](../_witnesses/harness-web-demo/approval.png) |
| Typed failure, minimal skin | `?scenario=failure&skin=minimal` | [`failure-minimal.png`](../_witnesses/harness-web-demo/failure-minimal.png) |

Capture command shape:

```text
go run ./cmd/harnesswebdemo -addr 127.0.0.1:8791
playwright screenshot --device="Desktop Chrome" --full-page URL OUTPUT.png
```

The render tests assert semantic message/tool payload projection, approval/failure controls, `aria-live`, responsive layout metadata, Content-Security-Policy, and the second skin. `go run ./cmd/harnesswebdemo -selfcheck` additionally drives message start/delta/completion, reconnect, approval, and failure protocol paths and emits an HTML SHA-256 receipt.

## Boundary

This proves the operating envelope of the offline reference product, not a complete coding harness. It still lacks a live fak agent adapter, durable process-restart sessions, authenticated remote deployment, and independent screen-reader automation. Those remain explicit gaps under #6790 rather than being hidden by the captured renders.
