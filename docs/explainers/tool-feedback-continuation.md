---
title: "Tool refusal is feedback, not task completion"
description: "General continuation contract learned from Pi and fak guard: preserve the denied effect, make refusal legible, and resume the model at the harness turn boundary with a bound."
---

# Tool refusal is feedback, not task completion

An agent can ask to use a tool and have every proposed call refused by the fak
capability floor. The API response then has no tool call for the client to
execute. The wire must finish that response normally; leaving a tool-call stop
reason would make a client wait for a call that does not exist. A finished **wire
turn** does not mean the user's **task** is finished. The harness owns the next
decision: continue with the refusal feedback, or settle under a declared stop
policy.

Pi made this distinction concrete. `fak guard -- pi` already repoints Pi's
provider through the gateway, and the gateway emits an in-band refusal note
(`denySummary` on empty-prose OpenAI turns, `adjudicationNote` otherwise).
Pi's default loop treats the response without a surviving tool call
as a final assistant turn. The guard now installs a session-scoped Pi extension
that handles Pi's `turn_end` boundary: if the response contains fak's refusal
note, no tool call or result survived, and the run completed, it appends a
non-displayed feedback message and requests one more model turn. Ordinary final
answers settle. Six identical hard refusals or 25 consecutive feedback turns
cause the adapter to stand down rather than spin forever; retryable tool
feedback uses the longer bound. This was checked
against Pi 0.87.1's installed extension API; future Pi releases need the same
boundary check.

## The portable harness rule

1. **Decide effects in the kernel.** Adjudicate each proposed call before
   execution. A denied call never executes and continuation never upgrades its
   authority.
2. **Report the result on the wire.** If no call survived, close the provider
   response using the protocol's normal no-tool finish reason. Carry the exact
   refusal and allowed recovery path to the model. Do not silently drop the
   call.
3. **Continue at a harness-owned boundary.** A harness with a native turn hook
   should append feedback to context and request the next model turn there.
   `fak guard` uses Claude's `Stop` hook and Pi's `turn_end` hook for this.
   This must be based on the current turn's outcome, not a process-wide metric
   that another session can change.
4. **Bound and reset.** Count repeated equivalent refusals and consecutive
   feedback turns; settle visibly at the bound. Clear counters after a clean
   turn. Never launch a new tool call merely because the adapter continued.

Fak's own native harness can use the structured adjudication outcome directly,
without recognizing a text prefix. Its existing stopgate test
`TestStopgateDenyAllNudgeContinuationInAgentLoop` is the reference for the
kernel-to-harness handoff. The Pi adapter is the external-harness translation
of that contract, not a second authority for tool decisions.

## Evidence and limits

`go test ./cmd/fak -run '^TestGuardPiContinuation' -count=1` executes the
generated Pi extension in Node and verifies refusal continuation, ordinary
completion, and the repeated-refusal bounds on both Anthropic and OpenAI wire
routes. `TestPiRefusalContinuation` verifies buffered and streamed OpenAI
gateway feedback with model prose. In a local loopback canary, installed Pi
0.87.1 received a `denySummary` first and a final answer second; its request
ledger contained two `/v1/chat/completions` calls and the output was
`LOCAL_SECOND_TURN_OK`. `TestGuardPiWire` separately proves the fake Pi child
reaches the guard gateway. These are software witnesses. A live `fak guard`
plus Pi session is a separate integration check; raw Pi-to-router 401/502
incidents do not establish guard continuation behavior.

Code: `cmd/fak/guard_pi.go`, `internal/gateway/refusal_notes.go`,
`internal/agent/loop_stopgate_test.go`. Pi contract: installed
`@earendil-works/pi-coding-agent` 0.87.1 `docs/extensions.md` and
`dist/core/extensions/types.d.ts` (`turn_end` and `BoundaryResult`).
