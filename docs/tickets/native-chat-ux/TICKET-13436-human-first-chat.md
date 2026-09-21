<!-- fak-chat-key: human-first-repl-20260920 -->
<!-- fak-public-issue: anthony-chaudhary/fak#13436 -->

```routing
lane: cmd
paths:
  - cmd/fak/chat.go
  - cmd/fak/chat_test.go
  - docs/tickets/native-chat-ux/TICKET-13436-human-first-chat.md
expected_steps: 5
github_issue: 13436
```

# TICKET-13436: human-first interactive `fak chat`

## Working spine

Keep the existing owned-loop chat path and replace its internal-first terminal presentation with a compact, recoverable end-user conversation.

## Current state

`fak chat` is production-reachable through `cmdChat -> runChat -> RunGovernedArmStream`, but interactive output exposes an absolute model path, raw tool arguments, and engine counters. A turn-cap exit can look like a blank answer, and a provider outage is described by an internal termination token before a recovery action.

## Why this is next

The execution path, streamed deltas, multi-turn history, and typed termination classifier already exist. Updating the renderer at that seam delivers the missing user experience without creating a parallel harness.

## Core through-line

Short startup and local slash commands -> existing `runChat` seam -> compact default activity and actionable failures -> deterministic `cmd/fak` chat tests.

## Gold-plating boundary

No full-screen TUI, scroll positioning, provider retry/routing change, policy change, new dependency, or headless text/JSON output change.

## Prior-art ledger (2026-09-20)

| Source | Pinned evidence | License / use | Local mapping |
| --- | --- | --- | --- |
| `google-gemini/gemini-cli@cfbcaa8df13ea4610bb379b377b56d62980c0032` | `docs/cli/cli-reference.md:41-48,70` documents `/help`, `/quit`, explicit debug logging, and separate text/JSON/stream-JSON output | Apache-2.0; ADAPT interface convention only, no source copied | Keep interactive output quiet, expose diagnostics explicitly, preserve automation output |
| Anthropic Claude Code CLI reference, retrieved 2026-09-20 | Documents explicit `--verbose` turn-by-turn output | Proprietary documentation; INSPIRE-ONLY | Use a local `/verbose` toggle rather than default transcript noise |

Default frontier: human-readable interactive output with an obvious recovery action. Coverage frontier: raw headless output remains stable for automation. The local tracer bullet is a renderer-only change exercised by scripted planners.

## Done condition / witness

Interactive startup is concise; default tool activity is compact; verbose mode reveals diagnostics; streamed answers appear once; turn limits and provider failures are actionable; the next prompt remains usable.

Witness: `go test ./cmd/fak -run '^(TestChat|TestRunChat|TestRenderChat)' -count=1`

## Definition of done

- [x] Every startup line, including auto-connect notices, hides the absolute model path; the banner uses a short model label and points to `/help`.
- [x] Default output hides raw tool arguments and engine/model counters.
- [x] `/verbose` opt-in exposes the bounded existing diagnostics.
- [x] Streamed final answers are not duplicated.
- [x] Turn-cap and provider failures explain a next action and keep the REPL open.
- [x] Headless behavior remains compatible.

## Evidence (2026-09-20)

- The full deterministic chat witness passed after the final code edit.
- The real `fak chat` entrypoint rendered the short model label and handled `/help`, `/status`, and `/verbose` without a provider call.
- A post-push entrypoint smoke check exposed the remaining auto-connect path leak; `TestChatAutoConnectDiagnosticUsesShortModelLabel` reproduced it through `cmdChat`, failed on the parent, and passed after both automatic connection notices stopped printing raw model identifiers.
- `TestFeatureIslandsWiredAudit`, staged leak audit, and all five placement gates passed.
- Independent review found and drove a turn-cap history regression from RED to GREEN, then reported no remaining findings.

## Value

- Centrality: Core
- P1 Context: advanced - completes the normal native-chat entrypoint rather than adding a second UI.
- P2 Net value: advanced - users can distinguish progress, incompletion, and provider failure without reading engine telemetry.
- P3 Adaptation: preserved - owned loop, history, policy, and headless contracts remain unchanged.
- P4 Operations: advanced - diagnostics remain available through explicit verbose mode.
