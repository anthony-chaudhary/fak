---
title: "Harness web UI operating-envelope witness — 2026-08-15"
description: "Documentation for Harness web UI operating-envelope witness — 2026-08-15, including the captured behavior, operating context, and reproducible fak evidence."
---

# Harness web UI operating-envelope witness — 2026-08-15

Issue: #6790 follow-on to the shipped #6882 spine.

## Captured live browser renders

Playwright Desktop Chrome captured the running loopback product at `http://127.0.0.1:8791`:

| Surface | Query | Artifact |
|---|---|---|
| Normal message/tool/artifact run | `?scenario=normal` | [`normal.png`](normal.png) |
| Scoped approval pending | `?scenario=approval` | [`approval.png`](approval.png) |
| Typed failure, minimal skin | `?scenario=failure&skin=minimal` | [`failure-minimal.png`](failure-minimal.png) |
| Live native coding run after restart | `?run=live-1` | [`live-coding-gpt-5.6-sol-2026-08-15.png`](live-coding-gpt-5.6-sol-2026-08-15.png) · [receipt](LIVE-CODING-GPT-5.6-SOL-2026-08-15.md) |
| Operational homepage: agents, goals, and gateway dashboards | `/` | [`operator-home-2026-08-20.png`](operator-home-2026-08-20.png) · [receipt](OPERATOR-HOME-2026-08-20.md) |
| Gateway typed startup state, desktop + narrow before/after | `/` | [`receipt`](GATEWAY-STARTUP-2026-08-29.md) · [`desktop`](gateway-startup-desktop-1440-2026-08-29.png) · [`narrow`](gateway-startup-narrow-390-2026-08-29.png) |
| Operational homepage edge/adversarial sweep | `/` and `/api/status` | [receipt](EDGE-ADVERSARIAL-2026-08-20.md) |
| Real-work dogfood, configured gateway unavailable | `/` and `/api/status` | [`dogfood-live-work-offline-1440-2026-08-22.png`](dogfood-live-work-offline-1440-2026-08-22.png) · [`status`](dogfood-live-work-offline-status-2026-08-22.json) |
| Real-work dogfood, connected workspace control plane | `/` and `/api/status` | [`dogfood-live-work-connected-1440-2026-08-22.png`](dogfood-live-work-connected-1440-2026-08-22.png) · [`status`](dogfood-live-work-connected-status-2026-08-22.json) · [receipt](DOGFOOD-LIVE-WORK-2026-08-22.md) |

Capture command shape:

```text
go run ./cmd/harnesswebdemo -addr 127.0.0.1:8791
playwright screenshot --device="Desktop Chrome" --full-page URL OUTPUT.png
```

The render tests assert the operational navigation and totals, semantic message/tool payload projection, approval/failure controls, `aria-live`, responsive layout metadata, Content-Security-Policy, and the second skin. `go run ./cmd/harnesswebdemo -selfcheck` additionally drives message start/delta/completion, reconnect, approval, and failure protocol paths and emits an HTML SHA-256 receipt.

## Boundary

This proves the captured offline scenarios, persisted local runs, canonical goal projection, and the live fak adapter's bounded agent/fleet reads. It does not prove authenticated remote deployment or independent screen-reader automation. Those remain explicit gaps under #6790 rather than being hidden by the captured renders.
