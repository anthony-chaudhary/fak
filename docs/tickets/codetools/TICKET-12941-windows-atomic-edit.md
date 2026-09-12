# TICKET-12941: Windows atomic edit sharing contention

<!-- fak-codetools-key: windows-atomic-edit-sharing-refusal -->
GitHub: https://github.com/anthony-chaudhary/fak/issues/12941
Parent: #12773 native prefill validation; native coding acceptance remains open.

```routing
lane: codetools
paths: ["internal/codetools/mutation.go", "internal/codetools/rename_windows.go", "internal/codetools/rename_other.go", "internal/codetools/rename_windows_test.go"]
expected_steps: 5
```

## Current state and working spine

The native agent's Read/Grep/Glob/Edit path publishes synchronized edits through
`codetools.atomicReplace`. At base `66d9036aeb7cd2f5b425dc7368fa2c510d779828`,
`TestAgentZeroSubprocessSpawnsAcross1000Operations` failed with Windows rename
`Access is denied` at iteration 42 in a full package run and iteration 16 in an
isolated run. Both runs used different temporary directories. Read/search handles
and the synchronized temporary file are closed before publication. Foreign
no-share-delete contention is a supported hypothesis, not process attribution.

## Core through-line

Retry only wrapped Windows access-denied/sharing-violation errors while retaining
the synchronized temporary file. Attempts use backoffs 0/1/2/4/8/16 ms, totaling
31 ms of scheduled sleep. This is not a hard wall-clock deadline on OS calls.
Return other errors immediately and the final transient error on exhaustion.
Non-Windows publication remains one rename. Creation retains its atomic
create-if-absent hard-link path. No destination deletion or subprocess is added.

## Problem frame

Centrality: Core
P1: advanced - Reliable local editing is necessary for native coding.
P2: advanced - Repair the actual integration failure blocking #12773 validation.
P3: preserved - Keep atomic publication, permissions and create-if-absent semantics.
P4: preserved - Retain error reporting and independent implementation/test evidence.

## Gold-plating boundary

No scanner exclusions, test skips, permanent-error suppression, inference changes,
filesystem redesign or physical-performance claim. The proof grader summarizes
recorded test outcomes; it is not a cryptographically bound execution receipt.

## Witness and done condition

- [x] Deterministic tests cover wrapped transient success, bounded exhaustion,
      permanent-error refusal, and the retry schedule.
- [x] Existing mutation/create-if-absent tests remain green.
- [ ] `go test ./internal/codetools ./internal/agent -count=1` passes on Windows.
- [x] `go vet ./internal/codetools ./internal/agent` passes.
- [x] Non-Windows compilation succeeds, reported as compilation only.
- [ ] The signed resolving commit links #12941 and preserves separated provenance.

The original failed Windows logs are retained by the #12773 validation run; this
leaf does not mark the broader Halo goal or the physical prefill gate complete.
## Recorded validation (2026-09-12)

The focused retry matrix passed. The complete Windows codetools package passed
in 66.621 seconds; the original repeated-edit agent integration passed in
83.319 seconds. Both packages passed vet. Linux/amd64 codetools test compilation
with CGO disabled succeeded; it was not executed. Generated-document freshness
also passed after a standard source rebuild recovered the earlier cache miss.

The full agent package remains red on the separate
`TestBlackboard_ZeroCopyVsJSONLatency` assertion (1.82 microseconds against a
1-microsecond threshold). Its failure log is preserved. No timing-test retry or
threshold relaxation was used to claim this rename correction's scoped checks.
The broader full-agent and physical-native gates under #12773 remain open.