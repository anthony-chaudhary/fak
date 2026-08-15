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

