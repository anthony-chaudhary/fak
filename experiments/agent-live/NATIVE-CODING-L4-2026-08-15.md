# fak-owned native coding run — L4 witness, 2026-08-15

Issues: #1380, #6910, runtime repair #6917.

## Provenance

- Committed source: `internal/agent@r298+g111a069af8` (derived by `fak version modules --json`).
- Commit: `111a069af88e` (`fix(agent): prevent unsupported CUDA prefix admission panic`).
- CUDA image: `us-central1-docker.pkg.dev/dos-rlvr-admit-20260608/fak/fak-gpu@sha256:8c61355aff8e32b5545d0d5a098aade703369da461aaab6d1792aa099e15260d`.
- Builder: Cloud Build, `Dockerfile.cuda`, `CUDA_ARCH=sm_89`.
- Compute: sanctioned GCP `fak-realmodel`, NVIDIA L4, CUDA backend.
- Serve flags included `--backend cuda --native --native-code-workspace /workspace`; `/workspace/hello.go` was a bounded fixture.

The live non-test owner is committed at `internal/gateway/native_serve.go:286`:

```text
return agent.RunArm(ctx, s.planner, seed.Task, true, s.nativeMaxTurns, nil, opts...)
```

## Repair witness

Before `111a069af88e`, a current CUDA image panicked while admitting a device checkpoint for a model whose prefix-reuse contract had left `p.tree` nil. The captured stack ended at:

```text
radixkv.(*Tree).LookupNS(0x0, ...)
agent.(*InKernelPlanner).admitPrefixSnapshot (.../inkernel_decode.go:434)
agent.(*InKernelPlanner).generateReusedContextWithBias (.../inkernel_decode.go:183)
```

The fix gates checkpoint admission on the existing `reuse` decision. Its regression is `TestInKernelBackendWithoutPrefixReuseSkipsCheckpointAdmission`.

## Live results

### Basic owned turn

Trace `gw-1` returned HTTP 200 and `Ready.`:

```json
{"arm":"fak","turns":1,"tool_calls":0,"vdso_hits":0,"engine_calls":0,"prompt_tokens":1148,"completion_tokens":2,"final_answer":"Ready."}
```

This is the same configuration that returned `handler_panic` before the repair.

### Real coding tool

Trace `gw-2` required the model to call `Read` on `hello.go`. HTTP 200 returned the actual file contents and:

```json
{"arm":"fak","turns":2,"tool_calls":1,"tool_errors":0,"repairs":0,"vdso_hits":0,"denies":0,"quarantines":0,"engine_calls":1,"prompt_tokens":2440,"completion_tokens":63}
```

`engine_calls=1` proves the bounded coding engine executed inside fak's owned loop; no client or external harness supplied the tool result.

### vDSO materialization before consumption

Trace `gw-4` requested repeated identical `Read` calls. HTTP 200 reported:

```json
{"arm":"fak","turns":2,"tool_calls":3,"tool_errors":0,"repairs":0,"vdso_hits":3,"denies":0,"quarantines":0,"engine_calls":0,"prompt_tokens":2640,"completion_tokens":139}
```

The result was materialized locally: `VDSOHits` advanced while `EngineCalls` stayed flat at zero. This satisfies the live materialization half of #1380.

## Honest boundary

This artifact proves the fak-owned CUDA loop, a real bounded coding tool, per-turn native-arm metrics, and vDSO serving on sanctioned hardware. It does **not** yet prove #1380's required live speculative suspend/promote-or-squash plus rollback witness, so #1380 remains open. It also does not prove browser session reopen; #6910 remains open until the web adapter consumes the public session/progress contract and survives process restart.


## Live speculative suspend, serve, squash, and rollback

The final missing #1380 envelope was run from the public `/v1/messages` seam with
`--native-speculate` enabled. The immutable deployment was built from committed trunk
`1a03173523da` (which contains rollback-receipt commit `aedb79407dc1`):

```text
Cloud Build: b656afd1-83ec-4846-a475-3f2c792175e5
Image: us-central1-docker.pkg.dev/dos-rlvr-admit-20260608/fak/fak-gpu@sha256:54203be53b6b3058f996a56611790945b03cf8122b89518d06553af7313c8ff3
Modules: internal/abi@r33+gaedb79407d; internal/agent@r301+gaedb79407d
Trace: spec-live-rollback-1
```

The request asked the model to separate the predicted read from the authoritative next
call:

```text
Call Read on hello.go, and only after that tool result call Glob using {"pattern":"**/*.go"}.
```

The privacy-safe SSE progress sequence was:

```text
message_start
turn_started
  tool_started Read
  call_adjudicated Read ALLOW
  result_admitted Read clean
turn_done
turn_started
  content_block_start
  content_block_delta
turn_done
content_block_stop
message_delta
message_stop
```

The terminal response carried the same-run native-arm receipt:

```json
{
  "turns": 2,
  "tool_calls": 1,
  "vdso_hits": 0,
  "engine_calls": 1,
  "spec_issued": 1,
  "spec_squashed": 1,
  "spec_rollbacks": 1,
  "spec_served": 1
}
```

`spec_issued=1` and `spec_served=1` show that the effect-free prediction suspended and
served before the authoritative continuation. The model did not issue the predicted
`Glob` call authoritatively on turn 2, so the pending transaction resolved as a
mispredict: `spec_squashed=1`. The matching `spec_rollbacks=1` is not inferred from the
squash counter; `internal/agent@r301+gaedb79407d` computes it directly from the delta in
`BufferSink.Rollbacks()` around `pending.Resume`, whose rollback count is maintained by
`internal/abi@r33+gaedb79407d`.

Remote logs bind the public trace to two real CUDA inference passes and the terminal HTTP
request. They report `backend=cuda`, `forward_path=device/generic`, HTTP 200, and
`trace_id=spec-live-rollback-1`. The first pass consumed 1,168 prompt tokens and the
second 1,258; the request completed in 55,638 ms. No external harness supplied or
resolved the tool result.

## #1380 acceptance closure

1. **Owned non-test loop:** `internal/gateway/native_serve.go:286` calls `agent.RunArm`
   from the live `fak serve --native` path (and line 324 calls its streaming sibling).
2. **Per-turn receipts:** the served terminal extension carries `fak.native_arm`, including
   turns, tool calls, repairs, quarantines, vDSO hits, engine calls, and speculation
   outcomes.
3. **vDSO before consumption:** trace `gw-4` above has `vdso_hits=3` while
   `engine_calls=0` for three repeated `Read` calls.
4. **Live speculation resolution:** trace `spec-live-rollback-1` has
   `spec_issued=1`, `spec_served=1`, `spec_squashed=1`, and the directly matched
   `spec_rollbacks=1` in one response.
5. **AgentDojo:** not applicable; this witness intentionally used the issue's allowed
   small-coding-task route, so no ASR claim is made.

## Boundary

This is a witnessed single-model/single-prompt L4 run, not a speculation hit-rate or
latency-gain benchmark. The prediction was served and safely rolled back, but it did not
match the authoritative continuation, so no speculative commit or performance gain is
claimed. The session-descriptor persistence warnings in the container log concern an
unwritable default home and did not alter the HTTP 200 owned-loop result; this artifact
makes no session-persistence claim.
