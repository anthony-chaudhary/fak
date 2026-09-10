# feat(agent): bind current-run Read versions into native code mutations

```routing
lane: agent
paths:
  - internal/agent/codetools.go
  - internal/agent/loop_turn.go
  - internal/codetools/args.go
  - internal/agent/codetools_loop_test.go
expected_steps: 7
repo: fak
```

<!-- fak-agent-key: native-current-run-code-cas-binding -->
<!-- fak-public-issue: https://github.com/anthony-chaudhary/fak/issues/12748 -->

## Current state

A native Strix Halo coding witness successfully read its target but spent every mutation turn on missing, mistyped, or invented CAS fields. The code-tool engine correctly refused those calls, so no filesystem mutation ran. The opaque full version already exists in the successful Read result, but the owned agent loop did not retain it.

## Why this is next

This is the smallest public runtime seam between a correct Read and a useful local-model edit. It removes fragile copying of an opaque hash while preserving the engine's byte-checked optimistic concurrency boundary.

## Parent context

Public issues #12748 and #11414.

## Core through-line

Successful native Read -> per-RunArm canonical-path observation -> omitted mutation CAS binding -> unchanged engine stale comparison -> witnessed mutation or refusal.

## Working spine

1. Capture the currently armed code toolset for one RunArm.
2. Record only successful Read results carrying a full `fv1:` version.
3. Resolve observations and mutation calls through the toolset's existing mutation resolver.
4. Fill only omitted Write/Edit fields; never replace explicit values.
5. Clear an observation after a successful mutation or a stale refusal.
6. Keep the direct engine validators and unknown-field refusal unchanged.
7. Prove success and isolation with independently authored literal-planner race tests.

## Gold-plating boundary

No global version cache, cross-run or parent/child sharing, fuzzy path matching, mutation-engine relaxation, new tool protocol, or acceptance of unknown fields. ApplyPatch and non-filesystem tools remain outside this leaf.

- Centrality: Core
- P1 Context: advanced - one run reuses its own exact Read result.
- P2 Net value: advanced - local models avoid copying opaque hashes across long turns.
- P3 Adaptation: N/A - no adaptive policy surface.
- P4 Operations: advanced - the real RunArm and kernel-mediated code-tool path is exercised.

## Done condition

- [x] Successful Read records a canonical target and full version inside one RunArm.
- [x] Same-path Write may omit mode/version and becomes an overwrite using that exact version.
- [x] Same-path Edit may omit version and uses that exact version.
- [x] Explicit versions remain unchanged and stale values remain refused.
- [x] A peer write after Read remains `FS_STALE_VERSION`.
- [x] Different paths and later RunArm instances cannot use the observation.
- [x] Unknown fields remain `MALFORMED`; direct engine validation remains strict.

## Definition of done

The owned native loop performs the four omitted-field mutations covered by the independent fixture and preserves every closed refusal case.

## Verifiable Witness

```powershell
$env:FAK_FAST='0'; .\test.ps1 -race -run 'TestOwnedLoopCASObservation' -count=1 ./internal/agent ./internal/codetools
```

Independent test context: `/root`; implementation context: `/root/agent_path_audit`. Test file SHA-256 before implementation was `D63335FA9F669C4C8CB2D5E9792EB677B8EAAD0E1E67E4F22762981B81DA1589`. Baseline RED is retained at `C:\work\fak-private\docs\benchmarks\receipts\strix1-daily-20260910\coding-parent\cas-observation-red.txt`. With production changes, the focused race run passed `internal/agent` in 1.125 seconds and `internal/codetools` in 1.014 seconds with no matching codetools test.

## Witness

The literal planner does not invoke the older test-only version binder. It emits exactly the fields a local model emitted, checks bytes after every refusal, and exercises separate RunArm instances.

## Acceptance gate

The focused independent race tests and existing agent/codetools package tests pass; `git diff --check` and the public boundary/leak hooks remain green.

## Closure binding

The resolving public commit cites #12748 and #11414 and carries `(fak agent)`.

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: closes the missing Read-to-mutation bridge in the daily native coding path.

## Completion standard

production

## Target operating envelope

- selected deterministic test pass rate: 100 percent
- explicit stale mutation byte changes: zero

## Witnessed operating envelope

- selected deterministic test pass rate: 100 percent
- explicit stale mutation byte changes: zero

## Likely files

- `internal/agent/codetools.go`
- `internal/agent/loop_turn.go`
- `internal/codetools/args.go`
- `internal/agent/codetools_loop_test.go`

## Lane

agent

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12748