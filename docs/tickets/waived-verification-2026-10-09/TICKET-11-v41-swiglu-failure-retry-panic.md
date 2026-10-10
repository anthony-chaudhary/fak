# TICKET-11: V4.1 clamped device SwiGLU failure-retry test panics and aborts internal/model

## Current state

At fak `4cbddeafffb`, `TestV41ClampedDeviceSwiGLUFailureFreesAndRetries/read` panics: "backend cpu-ref forward deepseek41 via v41-routed-expert failed closed at layer 0 (gate/up): compute: test-device [execution_failed] at Read: vulkan execution failed; session closed, no CPU retry [recovered, repanicked]".

The panic ends the package run. Every later `internal/model` test goes unreported, so other reds (#13770–#13772) may be hidden behind it. Fix this first.

Related: #13769 (TICKET-02b, SwiGLU dispatch count) touches the same clamped-SwiGLU device path. Likely source: a waived SwiGLU or routed-expert commit after `24033b4080f` (for example d6e79ea3d3d); bisect to confirm.

Parent: #13764. Evidence: fak-private `_scratch/goals/v41-attn-verify-1009/gate-4cbd.log` (local).

## Working spine

1. Reproduce `go test -count=1 -run 'V41ClampedDeviceSwiGLUFailureFreesAndRetries' ./internal/model/`.
2. Decide the contract for an injected device read failure: free the buffers and retry, or close the session with a typed refusal. Whichever applies, the test must observe it as an error, never a panic.
3. Rerun the whole package so that any red hidden behind the panic surfaces, and record which reds it reveals.

## Witness

`go test -count=1 -timeout 60m ./internal/model/` runs to completion with no panic.

## Done condition

- [ ] No panic; the failure-retry contract is asserted with a sentinel.
- [ ] Any reds the panic was hiding are listed on #13764.
