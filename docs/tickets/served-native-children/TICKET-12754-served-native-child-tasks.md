<!-- fak-public-issue: anthony-chaudhary/fak#12754 -->
<!-- fak-agent-key: served-native-child-request-scope -->
# Serve request-scoped native child tasks

Tracking: [#12754](https://github.com/anthony-chaudhary/fak/issues/12754).
Related: #12740, #12750, #11414, and #2387.

```routing
lane: gateway
paths: ["internal/gateway", "docs/tickets/served-native-children"]
expected_steps: 6
repo: fak
```

## Current state

The owned native `/v1/messages` loop has server-owned coding tools but does not
arm the request-scoped child runner from #12750. Its model catalog therefore
omits all four task lifecycle tools.

## Core through-line

Native HTTP parent -> request-scoped task state -> reusable native child runner
-> server-owned code catalog -> terminal child result -> joined parent cleanup.

## Gold-plating boundary

No server-wide semaphore, tenant scheduling, client-declared child capability,
deeper nesting, new configuration, or hardware-specific behavior.

## Done condition

- [x] Buffered and streamed native parents share one child-runner option builder.
- [x] The feature activates only with an armed native code workspace.
- [x] Same child identifiers remain isolated across concurrent HTTP parents.
- [x] Cancellation and return join only the owning request's child effects.
- [x] Child catalogs use only server-owned code tools and remain depth one.
- [x] Existing native gateway behavior remains green.

## Witness

```bash
go test -race ./internal/gateway -run TestNativeHTTPChildTasksAreRequestScopedAndJoined -count=1
```

At baseline `bcdd9923f0eee055324a2e983952036016d18595`, both cases fail with
`fak arm turn 1: server did not advertise task_spawn` before any child planner
entry. The independent test source SHA-256 is
`f8c5024c9a19cf61a54cd196928b15592b683553b14e872df3b5577e9a5d70f5`.

## Closure binding

The resolving commit cites #12754 with the `(fak serve)` leaf suffix.

## Recovery verification (2026-09-12)

The recovered request-scoped builder deliberately excludes the draft's
server-wide semaphore, matching the explicit issue boundary above. The original
draft remains preserved. The independent HTTP fixture now covers buffered and
streamed parents, task cancellation, and request-context cancellation. The
request-cancellation cases dispatch through the HTTP handler directly to observe
server-side cleanup after the client's context is cancelled. A separate witness
checks both loop entry points without an owned workspace.

At baseline `2e99556e8019703603afb674398c923533a6067d`, a Go overlay of the
unchanged native serving implementation fails the HTTP witness with
`server did not advertise task_spawn`. The candidate passes:

```text
go test ./internal/gateway -run 'TestNative(HTTPChildTasks|ChildTasksRequire)' -count=1
go test -race ./internal/gateway -run 'TestNative(HTTPChildTasks|ChildTasksRequire)' -count=1
go vet ./internal/gateway/...
```

The Windows race witness uses `CGO_ENABLED=1` and GCC from an installed MSYS2
UCRT64 toolchain, with its bin directory on PATH. Go temporary build files stay
outside the repository. This is software verification using deterministic
planners, with no model-quality or hardware-performance claim.

The recovery implementation and fixture extensions were edited in one context:
`COLLOCATED` fallback, because the original dirty draft has no independently
attested authorship and this bounded recovery forbids recursive delegation.

Final affected-package verification: `go test ./internal/gateway/... -count=1`
passed (64.807 seconds), and `go vet ./internal/gateway/...` passed. The first
full run exposed a pre-existing observer-equivalence assertion comparing
independently timed `ToolElapsedMs` values. The failure reproduced 30-run testing
with the unchanged baseline implementation overlaid. The repaired assertion
compares semantic copies with only tool duration excluded, preserving raw
measurements and every lifecycle/outcome assertion. Its 30-run focused witness
also passed.

Landed and pushed: 6d960c89c875224c9fcfe0d77f4b7999cd4e937f. Parent guarded landing passed prospective and post-merge verification; public leak audit passed. Full gateway suite passed (64.807 s); focused Windows race passed (1.927 s), vet passed, and observer timing regression passed 30 repetitions. Ref #12754.
