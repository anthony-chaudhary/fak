# Codex-session native UI dogfood packet (#8742)

## Verdict and evidence boundary

This packet is the smallest reproducible **fak-controlled deterministic journey** for the completed Codex-session UI adapter. `python validate.py` proves two independent replays have the same semantic-event hash, final diff, focused test result, and 12 required captures each. It also proves reconnect, cancellation, and denial produce zero duplicate workspace effects.

It is deliberately **not a live Codex-provider benchmark**. Events whose `owner` is `codex` are typed fixture inputs derived from the adapter contract; the receipts say `provider_claim: not observed`. Codex executes the real model/session in production. fak supplies the native UI, semantic projection, policy/approval bridge, reconnect mapping, and operational receipts.

## Reproduce from a clean checkout

Prerequisites: Python 3, Go 1.26+, no API key, model, browser, or GPU.

```powershell
cd docs/_witnesses/codex-session-ui
python validate.py
python validate.py # second clean deterministic check
```

The validator copies `seed/` to a fresh temporary directory and runs `go test ./...`; it does not mutate the checkout. Expected output includes `"runs_match": true`, `"capture_count": 12`, `"focused_test": "go test ./...: PASS"`, and `"scrub": "PASS"`.

## Journey

1. Open the bounded seeded task and inspect plan/progress (`start-*`, `streaming-*`).
2. Surface a destructive shell request, deny it, and retain an unchanged workspace (`approval-*`, `denial-*`). Broad auto-approval is never enabled.
3. Apply the one expected comma edit, disconnect, reconnect the same logical session, and prove `replayed_effects: 0` (`reconnect-*`).
4. Cancel a turn (`workspace_effects: 0`), retry once, run the focused test, and review `final.diff` (`completion-*`).

Both desktop (900 px) and narrow (360 px) SVG render witnesses are captured for start, streaming work, pending approval, reconnect, denial/failure, and completion. `semantic.jsonl` exposes progress and authority without terminal logs. `packet-receipt.json` binds the matching semantic hash, diff hash, test result, capture count, and scrub result.

## Matched direct-Codex comparison

| Dimension | Direct Codex TUI (next-best option) | fak native UI journey |
|---|---|---|
| Model/session execution | Codex-controlled; **not observed in this deterministic packet** | Same Codex responsibility in production; replay invokes no model |
| Typed semantic projection | Not measured here | fak-controlled fixture contract and hash |
| Approval authority | Codex behavior not measured here | fak-controlled pending/deny projection; no broad auto-approval |
| Reconnect/cancel effects | Not measured here | deterministic receipt proves zero replayed/duplicate/cancel effects |
| Operator interventions | Live baseline not measured | fixture has deny, reconnect, cancel, retry |

Therefore this packet proves reproducibility and the fak-owned UX/operations contract, not latency, token usage, or superiority over the direct TUI. A live matched run can replace `provider_claim: not observed` only when its external receipts are captured.

## Data retention, troubleshooting, rollback, limitations

All artifacts are committed, synthetic, and scrubbed; no prompt transcript, credential, private path, or provider payload is retained. The scrub check rejects common credential/private-path patterns. If hashes diverge, inspect `semantic.jsonl` and `final.diff`; if the test fails, run `go test ./...` in a copy of `seed/`; if a capture is absent, regenerate it from the same typed state before accepting the packet.

Rollback is deletion or git-revert of this directory; it changes no runtime path. Known limitations: SVGs are deterministic render witnesses rather than browser screenshots, usage/elapsed-time values are omitted because no provider ran, and direct-Codex behavior remains explicitly unobserved. Live browser kill/reconnect and provider comparison remain a separate sanctioned observation, not silently simulated here.