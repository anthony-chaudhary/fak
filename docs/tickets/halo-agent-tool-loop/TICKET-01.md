# fix(up): preserve structured tool turns across buffered and streaming chat completions

```routing
lane: allinone
paths:
  - cmd/fak/up.go
  - cmd/fak/up_tool_loop_test.go
expected_steps: 7
```

<!-- fak-up-key: turnkey-structured-tool-loop-v1 -->
<!-- fak-public-issue: https://github.com/anthony-chaudhary/fak/issues/12680 -->

## Current state

At public base `644bd060041a89caa7e1024b2cf1540b058cce16`, `cmd/fak/up.go:537-559` defines the turnkey `/v1/chat/completions` message as role plus string content and does not decode `tools`, `tool_choice`, assistant `tool_calls`, or tool-message `tool_call_id`. At `cmd/fak/up.go:610`, the handler calls `planner.Complete` with a nil tool catalog. The response serializes assistant content only, so a client cannot complete a function-call turn or continue it with correlated tool output. The streaming path must obey the same structured contract rather than reducing tool calls to text.

This is a public all-in-one runtime contract. It makes no hardware performance or model-quality claim.

## Why this is next

The default one-touch endpoint is the last protocol seam between a tool-capable client and the native planner. Closing it enables real agent tool turns through `fak up` with a deterministic software witness.

## Parent context

GitHub issue #12680.

## Core through-line

Preserve the OpenAI-compatible structured tool loop through the turnkey endpoint:

1. Decode request tools and tool choice.
2. Preserve prior assistant tool calls and tool-message call IDs in planner history.
3. Pass the decoded tool catalog and sampling choice to `planner.Complete`.
4. Serialize planner tool calls, including stable call ID, function name, JSON arguments, and `finish_reason: "tool_calls"`.
5. Emit equivalent structured information for buffered JSON and SSE responses so the next client request can replay the complete turn unchanged.

## Working spine

1. Extend the turnkey wire request and message representation with the canonical structured tool fields.
2. Forward tools, tool choice, and complete message history through the planner boundary.
3. Map the planner result to one shared assistant response shape.
4. Serialize that shape through buffered JSON and ordered SSE tool-call deltas.
5. Prove request, response, correlation, and plain-text compatibility with scripted-planner HTTP tests.

## Gold-plating boundary

Do not redesign the planner, add a tool executor to `fak up`, introduce a second agent loop, change model behavior, or claim native Qwen tool-selection quality. The endpoint translates protocol state; the client remains responsible for executing tools and submitting results. Live-model quality is a later independent witness.

- Centrality: Core
- P1 Context: advanced - preserves the complete multi-message tool turn.
- P2 Net value: advanced - restores useful tool work on the default all-in-one path without a model dependency.
- P3 Adaptation: preserved - forwards explicit client tool choice without inventing policy.
- P4 Operations: advanced - buffered and streaming protocol witnesses cover the live HTTP seam.

## Dedupe

Public issue search on 2026-09-10 found exact existing issue #12680, which owns the dropped tool catalog and response contract in `fak up`. This ticket extends its witness detail to prior-turn correlation and buffered/SSE parity without creating a duplicate. #12594 concerns a separate Anthropic adapter, and #12651 concerns a separate all-in-one supervisor endpoint seam.

## Done condition

- [x] A request containing a function tool and explicit `tool_choice` reaches `planner.Complete` with both values intact.
- [x] Prior assistant `tool_calls` and a following role=`tool` message retain their call IDs, names, and JSON arguments in planner history.
- [x] A planner completion containing a tool call produces buffered JSON with the same call ID, function name, JSON arguments, and `finish_reason: "tool_calls"`.
- [x] SSE emits a valid ordered tool-call delta sequence, a terminal `finish_reason: "tool_calls"`, and `[DONE]`.
- [x] A completion reporting dropped tool calls with no parsed calls fails closed as JSON HTTP 502 before any buffered or SSE success is written.
- [x] Explicit zero temperature reaches the planner and missing usage remains zero rather than being fabricated.
- [x] Plain text completions remain compatible in buffered and streaming modes.
- [x] A deterministic loopback test fails at witnessed baseline `c8c7150621b3eeaeadc976ee7d4e68fdf5a8cc27` and passes with the fix; its `cmd/fak/up.go` blob is identical to the gap source at `644bd060041a89caa7e1024b2cf1540b058cce16`.

## Definition of done

The turnkey endpoint preserves a complete structured tool turn through planner ingress and buffered/SSE egress, with deterministic coverage for correlation fields and plain-text compatibility.

## Verifiable Witness

```powershell
go test ./cmd/fak -short -v -run '^(TestTurnkey|TestUp|TestRunTurnkey|TestStartTurnkey)' -count=1 -timeout=180s
```

The tests use a scripted planner and loopback HTTP; they require no external service or downloaded model. Native local-model tool-choice quality remains outside this software-contract witness.

RED at witnessed baseline `c8c7150621b3eeaeadc976ee7d4e68fdf5a8cc27`: after adding only the seam needed to inject and observe the planner (interface/accessor/context-optional wiring), `TestTurnkeyServerToolCallRoundTrip` reached its first protocol assertion and observed that the planner received nil tools. The baseline `cmd/fak/up.go` blob is identical to `644bd060041a89caa7e1024b2cf1540b058cce16`; this isolates the defect from model behavior. The RED receipt is `.gotmp/halo-up-baseline-red.txt` in the flow worktree until landing.

GREEN with the fix:

```text
ok github.com/anthony-chaudhary/fak/cmd/fak 9.444s
```

All three new tool-loop tests and the selected existing turnkey context, native, and mock tests passed on the published seam parent. The process witness was explicitly skipped by `-short`; this receipt makes no hardware claim. The earlier pre-split run also passed in 31.302 seconds and is retained at `.gotmp/halo-up-final-tests.log` in the flow worktree until landing.

## Witness

Run the deterministic `go test` command above and retain its package result with the resolving commit. The scripted planner must observe request-side tool state, and the buffered and SSE assertions must observe the same response-side tool call.

## Acceptance gate

All three new tool-loop tests and the selected existing turnkey context, native, and mock tests pass at the resolving revision.

## Closure binding

The resolving public commit cites #12680 and uses `(fak up)`. Attach the exact deterministic test command and result; do not attach a hardware or model-quality claim.

## Landing evidence

The planner-injection seam landed first as behavior-preserving public commit `e34774426adb4c9f11ae4ee475475dccf75f7c60`, after scoped existing turnkey tests and the normal companion leak/5-Gate hook passed. The managed single-commit fix landing could not establish its aggregate symptom witness: prospective verification consumed 884.862 seconds and returned `SYMPTOM_UNWITNESSED`; a separate seam-only full-package diagnostic reached the global 180-second timeout while executing unrelated agent-profile tests, with no preceding assertion failure. Under the repository's Decoupled Lane Forward Progress rule, the functional leaf therefore uses the normal signed, explicit-path commit workflow with all standard hooks intact and the focused RED/GREEN protocol witness above. The managed symptom gate remains unconfirmed; the standard commit hooks passed unchanged.

## Work estimate

Estimate: 3 points.

## Overall completion contribution

Contribution: closes the single protocol leaf tracked by #12680.

## Completion standard

production

## Target operating envelope

- selected command test pass rate: >= 100 percent

## Witnessed operating envelope

- selected command test pass rate: 100 percent

## Likely files

- `cmd/fak/up.go`
- `cmd/fak/up_tool_loop_test.go`

## Lane

allinone

## Tracking

GitHub: https://github.com/anthony-chaudhary/fak/issues/12680
