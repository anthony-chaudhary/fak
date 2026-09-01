---
title: "Provider clear and new commands: fak session boundaries"
description: "How fak maps a provider-side clear/new command onto a fresh fak trace without resetting cumulative limits."
---

# Provider clear and new commands: fak session boundaries

Yes: when a wrapped provider reports `SessionStart(source=clear)`—or an adapter
normalizes another typed reset event to that contract—fak starts a new fak
session too. It does **not** restart the guard or gateway process, and it does
not treat the event as context compaction. Command names alone are not the
contract: a command that only repaints the terminal causes no fak boundary.

The boundary is a small atomic control-plane transition:

1. the provider creates a new conversation/thread id;
2. its `SessionStart` hook sends `provider`, `source`, and `session_id` over the
   authenticated, process-local guard lifecycle socket;
3. fak closes the old trace as `STOPPED` with
   `reason=PROVIDER_SESSION_CLEAR`;
4. fak creates a deterministic child trace and switches the gateway default to
   it, so requests that omit `X-Trace-Id` immediately use the new session; and
5. a repeated hook delivery for the same provider id is a no-op.

The old record is retained for audit. The child carries a typed
`fak.session.provider_boundary.v1` row with the provider, source, previous fak
trace, and new provider session id.

## What resets and what carries

| State | At provider `clear` / `new` |
|---|---|
| Provider transcript and context window | Fresh |
| Context-token remaining | Re-armed to the configured context cap |
| Goal, objective pin, assumptions, operator span pins | Cleared |
| Pending turn, cost ring, cache affinity, reset transaction | Cleared |
| Turns/output/query/spend/tool-call remaining | Carried unchanged |
| Wall-clock elapsed and throughput observation | Carried |
| Priority, pace, and QA envelope | Carried |

This split prevents quota laundering: typing `/clear` cannot buy a new spend,
turn, or tool-call allowance. Context is the one budget axis that becomes fresh
because the provider actually created a fresh context window.

## Boundary vocabulary

These events are intentionally different:

| Provider event | fak interpretation |
|---|---|
| `startup` | Initial binding; keep the launch trace |
| `resume` | Same logical session, replacement execution binding |
| `compact` | Same logical session, smaller context representation |
| Typed `clear` / adapter-normalized reset | New logical fak session |
| UI-only clear with no new provider conversation | No fak session change |
| `fork` | Branch semantics; not treated as clear |

That distinction preserves the broader session-client contract: a provider
thread may change during resume, migration, or compaction without changing the
logical session. An explicit user clear is the declaration that the old logical
conversation is finished.

## Provider coverage

| Provider harness | Detection | Current wiring |
|---|---|---|
| Claude Code under `fak guard -- claude` | `SessionStart` input with `source=clear` and `session_id` | Automatic; the existing guard SessionStart hook carries the provider tag and lifecycle socket credentials |
| OpenAI Codex under `fak guard -- codex` | `/clear` starts a replacement thread with `SessionStart(source=clear)`; `/new` starts a thread but reports `source=startup` | Automatic; a trusted per-launch hook maps `clear` directly and treats a later `startup` with a new `session_id` as the same boundary |
| Gemini CLI | Its `clear` command (`new` alias) emits `SessionEnd(clear)`, mints a session id, then emits `SessionStart(clear)` | Core-ready; first-class `fak guard -- gemini` hook installation is tracked in [#8219](https://github.com/anthony-chaudhary/fak/issues/8219) |
| Other harnesses | Send JSON containing `source:"clear"` and one of `session_id`, `thread_id`, or `conversation_id` to `fak guard-sessionstart --provider NAME` inside a guarded child; adapters own normalization when their reset event uses another source token | Adapter-owned |

The hook is fail-open so a broken local lifecycle socket cannot wedge the
provider UI. On failure the provider still clears, fak prints a bounded stderr
warning, and the old fak trace remains in force; cumulative limits therefore
fail toward the stricter state rather than being reset.

For Codex, the launch adapter stores only the current provider session id and
fak trace in its private run directory. The first `startup` binds the launch
trace, repeated delivery of that id is a no-op, and a later `startup` carrying a
different id is `/new`. The explicit `clear` source does not need inference.
The hook is added in Codex's `sessionFlags` layer, so user, project, managed, and
plugin hook layers stay active. Fak trusts the exact injected handler hash; it
does not pass Codex's all-hooks trust bypass.

## Captured witness

The end-to-end test drives the real hook actuator and authenticated lifecycle
socket, then reads both fak session records and the gateway's new default trace:

```bash
go test ./internal/session -run 'TestBeginProviderSession|TestDescriptorRoundTripPreservesProviderBoundary' -count=1
go test ./cmd/fak -run 'TestGuardSessionStart(ClearCreatesFakSessionBoundary|CodexClearAndNewCreateOneBoundaryEach)' -count=1
```

The captured Codex 0.148 TUI run, exact hook-trust proof, and fail-before result
are in [`issue-8218-codex-session-boundary.json`](../_witnesses/issue-8218-codex-session-boundary.json).

Live cross-provider dogfood and outcome counts are tracked in [#8220](https://github.com/anthony-chaudhary/fak/issues/8220).

## Exhaustive `openai/codex` inventory refresh (issue #8998)

This refresh applies the `study-repo` exhaustive-denominator pass and the `field-borrow` decision pass to upstream revision [`2764e83626efe55f64e04d153fc99a157327f3c2`](https://github.com/openai/codex/commit/2764e83626efe55f64e04d153fc99a157327f3c2), observed 2026-08-26. The generated map is [`docs/research/inventory/openai-codex.json`](../research/inventory/openai-codex.json): 6,613 files, 848 directories, 67,249,248 bytes, and 14 immediate subsystems. It walks every regular file outside `.git`; its classifier reports 4,278 runtime files, 751 test files, and 615 documentation files.

### Source-class audit

| Required class | Pinned evidence and result |
|---|---|
| README and docs | The map covers [`README.md`](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/README.md), top-level [`docs/`](https://github.com/openai/codex/tree/2764e83626efe55f64e04d153fc99a157327f3c2/docs), and package READMEs. Hosted product docs are outside the reproducible repository boundary. |
| Architecture and design | The classifier found only one filename-level design hit, so architecture is reconstructed from the Rust workspace and subsystem READMEs rather than claimed from a canonical architecture document. [`codex-rs/`](https://github.com/openai/codex/tree/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs) dominates (6,177 files); the map also records SDK, CLI, automation, Bazel, and tooling. |
| Runtime source | All 4,278 runtime-classified files are covered. [`slash_dispatch.rs`](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs/tui/src/chatwidget/slash_dispatch.rs) and [`event_dispatch.rs`](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs/tui/src/app/event_dispatch.rs) show `/new` emits `NewSession`, `/clear` emits `ClearUi`, and each starts a replacement thread while only clear supplies `ThreadStartSource::Clear`. |
| Tests and fixtures | All 751 test-classified files are covered, including [`chatwidget/tests/`](https://github.com/openai/codex/tree/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs/tui/src/chatwidget/tests), app-server session tests, snapshots, and fixtures. The runtime conclusion has behavioral witnesses. |
| History, changelog, and releases | History is pinned at the checked commit. On 2026-08-26 GitHub exposed stable [`rust-v0.149.1`](https://github.com/openai/codex/releases/tag/rust-v0.149.1) and prereleases through [`rust-v0.150.0-alpha.11`](https://github.com/openai/codex/releases/tag/rust-v0.150.0-alpha.11). The map enumerates release automation; no root changelog is treated as authoritative. |
| Open/closed issues, PRs, and discussions | GitHub GraphQL observed 13,653 open and 10,628 closed issues, 170 open, 5,181 closed-unmerged, and 9,964 merged PRs, plus 759 discussions on 2026-08-26. The checked revision is merged PR [`#40751`](https://github.com/openai/codex/pull/40751); reset-history search retains merged clear-context PR [`#17499`](https://github.com/openai/codex/pull/17499). Counts are observations, not repository-controlled facts. |
| Roadmap and TODOs | The inventory found no roadmap or marker-comment source class and GitHub milestones were empty. FAK does not infer a roadmap from incidental source comments or release cadence. |
| License and provenance | [`LICENSE`](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/LICENSE) is Apache-2.0; [`NOTICE`](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/NOTICE) records OpenAI copyright. This pass borrows behavior and documentation, not source. |
| FAK self-query witness | FAK-tree query finds this guide, `cmd/fak/provider_session_reset_test.go`, `cmd/fak/provider_session_reset_more_test.go`, and the session-boundary witness. They already prove provider-neutral cut/reset behavior and adapter distinction. |
| Candidate matrix | The matrix below records every material candidate; none is silently promoted from observation to recommendation. |
| Completeness critic | The map gives a complete tree denominator but cannot contain mutable GitHub objects or hosted docs, queried separately above. Architecture classification is conservative, counts are point-in-time, and no live-provider run was needed for this source-level thread-boundary question. |
| Issue tracking | FAK [`#8998`](https://github.com/anthony-chaudhary/fak/issues/8998) owns this refresh under [`#8936`](https://github.com/anthony-chaudhary/fak/issues/8936). No implementation follow-on was filed because no unmet FAK behavior remained; future drift stays with the monitor cycle. |

### Field-borrow candidate matrix and FAK decisions

| Candidate | Decision | Reason / follow-on |
|---|---|---|
| `/clear` as a typed thread-start source plus replacement thread. | **Keep (already borrowed).** | It is the strongest signal for an explicit cut; current adapter behavior and tests cover it. |
| `/new` as the same physical replacement boundary despite different provenance. | **Keep the distinction.** | FAK may observe a new physical session but must not infer clear provenance merely from replacement. |
| Codex TUI events, managed defaults, or transcript architecture in the kernel. | **Reject here.** | These harness concerns are unnecessary for FAK's contract and would add provider coupling without witnessed gain. |
| Release cadence, issue volume, TODOs, or absent milestones as roadmap. | **Reject; monitor only.** | They are mutable observations, not commitments. The pinned registry and recurring monitor are the follow-on. |

**FAK outcome:** retain the typed adapter contract and provider-neutral tests; add no code or speculative implementation issue. The durable change is the exhaustive pinned denominator plus this adopt/reject/watch record.

## Design decision and upstream evidence

Observed at `2026-08-20T15:53:41Z`. This was a deliberate quick study of one
capability—the semantics and hook shape of clear/new commands—not a broad review
of either upstream repository.

Problem framing:

- **For:** operators who use provider-native `/clear` or `/new` while fak is in front.
- **Problem:** the provider can replace its conversation while fak continues routing
  omitted-trace requests under stale session state.
- **Today:** Claude, Codex, and Gemini expose lifecycle events at the reset seam,
  but command vocabulary alone does not identify equivalent semantics.
- **Better because:** one authenticated boundary write keeps identity, budgets, and
  audit state aligned without restarting the gateway.
- **Witness:** the hook-to-IPC-to-session-table test above.

Centrality: **Enabling**. P1 managed context is preserved by separating clear from
compaction; P2 net-true efficiency avoids a guard restart and claims no cache gain;
P3 adaptation stays bounded by a closed reset event; P4 operations get a typed
old/new trace record and idempotent transaction.

| Source | Pin / observation | License | Relevant evidence | Disposition |
|---|---|---|---|---|
| [Claude Code hooks reference](https://code.claude.com/docs/en/hooks) | Live docs observed 2026-08-20 | Documentation terms | `SessionStart` distinguishes `startup`, `resume`, `clear`, `compact`, and `fork`; command-hook JSON arrives on stdin | INSPIRE: consume the native lifecycle event instead of terminal text |
| [openai/codex](https://github.com/openai/codex/tree/2764e83626efe55f64e04d153fc99a157327f3c2) | `2764e83626efe55f64e04d153fc99a157327f3c2`, commit time `2026-08-26T01:20:40Z` | Apache-2.0 | [`/new` and `/clear` dispatch](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs/tui/src/chatwidget/slash_dispatch.rs) both start fresh threads, but [`event_dispatch.rs`](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs/tui/src/app/event_dispatch.rs) supplies `ThreadStartSource::Clear` only for clear; the [`thread/start` contract](https://github.com/openai/codex/blob/2764e83626efe55f64e04d153fc99a157327f3c2/codex-rs/app-server/README.md) exposes that source | ADAPT behavior only; `/new` needs stateful adapter normalization; no source copied |
| [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli/tree/e90c63fa158b8facd1872d32b34b07e516308f2b) | `e90c63fa158b8facd1872d32b34b07e516308f2b`, commit time `2026-08-19T21:20:10Z` | Apache-2.0 | [`clearCommand.ts`](https://github.com/google-gemini/gemini-cli/blob/e90c63fa158b8facd1872d32b34b07e516308f2b/packages/cli/src/ui/commands/clearCommand.ts) defines `new` as an alias, ends the old session, mints a UUID, resets chat state, and emits `SessionStart(Clear)` | ADAPT behavior only; no source copied |

The alternatives lose for concrete reasons: restarting the guard discards warm
process state; deleting the old record destroys audit history; treating clear as
Recontinue leaks the old objective and pins; resetting every budget enables quota
laundering; and parsing terminal escape sequences ignores typed events all three
studied harnesses already provide.

No upstream code was copied. The recurring source-registry update remains tracked
in [#8221](https://github.com/anthony-chaudhary/fak/issues/8221) because that registry
was under a peer-owned edit during this pass.

## Exhaustive Gemini CLI inventory refresh (2026-08-25)

Issue [#8999](https://github.com/anthony-chaudhary/fak/issues/8999) refreshes the
Gemini evidence against verified commit
[`64b5b79a6dd89ea96e65cb761c23aae6c0b33ce4`](https://github.com/google-gemini/gemini-cli/commit/64b5b79a6dd89ea96e65cb761c23aae6c0b33ce4)
(commit time `2026-08-25T18:52:30Z`, tree
`95e04bcee2859ff1c997f3fc333609045011197b`). The generated exhaustive map is
[`docs/research/inventory/google-gemini-gemini-cli.json`](../research/inventory/google-gemini-gemini-cli.json):
2,971 files and 294 directories were walked, including README/docs,
architecture, runtime, 1,110 test/fixture files, history/changelogs/releases,
`ROADMAP.md`, and Apache-2.0 provenance. Only `.git` and the vendored
`packages/core/vendor` dependency tree were skipped.

Non-tree read-back used GitHub GraphQL aggregate counts and REST metadata at the
same observation cutoff: 14,309 issues (599 open, 13,710 closed), 12,659 pull
requests (264 open, 5,727 closed-unmerged, 6,668 merged), 662 discussions, and
601 releases. The inventory records the exact methods, counts, FAK self-queries,
and candidate dispositions rather than treating a local tree walk as the whole
denominator.

The refreshed source still fires `SessionEnd(Clear)`, mints a new UUID, resets
chat state, and fires `SessionStart(Clear)`; `new` remains an alias. FAK's
provider-boundary adaptation is therefore retained as **already owned**, with no
source copied. Gemini checkpoint/session management is rejected as a separate
borrow because FAK already owns portable session images, recall, context reuse,
and live-session control. Gemini's hook/extension/MCP stack is also rejected as
a direct port because it duplicates FAK's guarded adapters, gateway, and
capability floor without a matched performance or correctness gain.

No new follow-on survived this audit. First-class Gemini guard integration
shipped under [#8219](https://github.com/anthony-chaudhary/fak/issues/8219), and
the remaining live cross-provider dogfood witness is already tracked by
[#8220](https://github.com/anthony-chaudhary/fak/issues/8220).
