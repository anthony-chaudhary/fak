# fix(cmd): a corrupt session-state snapshot refuses before the model load, not after it

## Parent context

Child of TICKET-03 (Vulkan memory accounting and residency) in this directory,
parent #13761 (https://github.com/anthony-chaudhary/fak/issues/13761), serving #13668.
Evidence: operator-held silicon receipt (schema `fak.v41_native_probe.v1`, probed 2026-10-09T23:37Z, binary sha256 `2f3f2298bbae73a0`).

<!-- fak-model-key: waived-vulkan-session-state-restore-after-load -->
<!-- fak-public-issue: anthony-chaudhary/fak#13773 -->

```routing
lane: cmd/serve
paths: ["cmd/fak/serve_stages.go", "cmd/fak/serve_durability.go", "docs/tickets/waived-verification-2026-10-09/TICKET-03a-session-state-restore-after-load.md"]
expected_steps: 2
public_issue: anthony-chaudhary/fak#13773
```

Process cause: verification-gap

## Scope class

single-package

## Current state

[HW-WITNESSED] On a Strix Halo appliance at fak 42dc8adb624, a V4.1 Flash arm-B
`fak serve --backend vulkan` loaded for 1270 s, then exited 1 with
`fak serve: --session-state ~/.config/fak/session-state.snap: snapshot: decode envelope:
invalid character 'f' after top-level value`. No flag set the path: it is the
default from `resolveServeSessionState` (`cmd/fak/serve_durability.go`). The
restore runs at `cmd/fak/serve_stages.go:915` (`restoreServeSessions` then
`os.Exit(1)`), after the model load.

## Problem

A present-but-corrupt snapshot fails loudly by design, but it fails 21 minutes too late.
The whole load is wasted, and `/healthz` never comes up.

## Why this is next

It is the step that blocks the next #13668 first-token attempt on the bounded dense route.

## Working spine

Decode or validate the snapshot before the model load. Then either refuse typed before any
tensor is read, or quarantine the corrupt file (rename it aside with a warning) and boot fresh.

## Core through-line

The snapshot verdict is known before the load begins.

## Gold-plating boundary

No snapshot format change and no new persistence fields.

## Verifiable Witness

```bash
go test ./cmd/fak/ -run 'SessionState|RestoreServeSessions' -count=1
```

## Done condition / witness

Witness: a unit test that fails before the fix and passes after it, run with the command above.

- [ ] [SW-VERIFIED] A corrupt default snapshot is rejected or quarantined before `loadServeInKernelModel` runs. A test asserts the ordering, or asserts a typed sentinel.
- [ ] [HW-WITNESSED] Arm-B re-probe on a Halo reaches `/healthz` without `FAK_SESSION_STATE=off`.

## Witness

The SW box needs a fail-before/pass-after test. The HW box needs a silicon serve log.

## Definition of done

Both boxes are ticked, and the resolving commit cites this issue.

## Acceptance gate

The `cmd/fak` tests stay green, and boundary and scrub checks are green.

## Closure binding

The resolving commit cites this public issue and carries the `cmd` leaf in its subject.

## Likely files

- `cmd/fak/serve_stages.go`: restore call site at :915.
- `cmd/fak/serve_durability.go`: default path resolution.

## Lane

cmd/serve

## Blast radius and affected lanes

One package, `cmd/fak`. This ticket changes only when the boot verdict is reached; the
corrupt-file policy stays the same.

## Quarantined fallback mechanism

`FAK_SESSION_STATE=off` or `--session-state off` stays the operator workaround.

## Problem frame

- Centrality: Enabling (unblocks the #13668 first-token probe)
- P1 Context: preserved - no change to context, KV or prompt handling; only the touched load or test path moves.
- P2 Net value: advanced - a corrupt snapshot costs a sub-second refusal instead of a 21-minute wasted load.
- P3 Adaptation: preserved - no new external mechanism is adopted; existing fak paths are corrected.
- P4 Operations: advanced - fail-fast boot verdict.

## Expected steps

2

## Work estimate

Estimate: 1 points

## Overall completion contribution

Contribution: 1/9 points toward parent #13761.

## Completion standard

development

github_issue: anthony-chaudhary/fak#13773
