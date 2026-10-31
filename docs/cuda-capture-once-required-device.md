# Required-device capture-once witness

Scope: the landed capture-once dense-F32 single-session path plus this witness.
No model downloads, fake backends, quantized/hybrid support, concurrent unrelated
submissions, performance or full-serving-parity claim. Run only on an authorized
CUDA node after claiming the hardware request, in isolated processes.

## Exact native and cgo profile

Use the existing native builder, recorded flags/cache identity and local resource
supervisor. Serialize heavy compilers; check disk, actual memory pressure and VRAM.
Choose the node's supported architecture, not a guessed GPU target. Do NOT add
pending-compressor or Q4-async profile tags: GraphOnce deliberately refuses those
unqualified combinations. The selected archive must contain the graph native ABI.

The following recipe is for a prepared Linux/WSL shell, not Windows cmd.exe.
The native builder synthesizes its child cgo environment with
`computebuild.SynthesizeCUDACgoEnv`; those values DO NOT propagate back to this
shell. Before running the recipe, set `CC`, `CXX`, `CGO_CFLAGS`, `CGO_LDFLAGS`,
`LD_LIBRARY_PATH` and `LIBRARY_PATH` to the verified selected toolchain's values
used by that builder. Record literal values in the hardware receipt. Include the
actual CUDA and driver link/loader paths (including the real WSL driver directory
when applicable), not guessed system paths or stub libraries. The last two values
may be empty only when the verified environment requires no extra search path.
Do not substitute `build-cuda test`: it runs a different selector and forces a
GRAPH=0 pass, which is incompatible with this required GRAPH=1 witness.

```bash
set -euo pipefail
: "${CC:?set the verified builder C compiler}"
: "${CXX:?set the verified builder C++ compiler}"
: "${CGO_CFLAGS:?set the verified builder cgo include flags}"
: "${CGO_LDFLAGS:?set the verified builder cgo link flags}"
: "${LD_LIBRARY_PATH?set the verified loader search path, possibly empty}"
: "${LIBRARY_PATH?set the verified link search path, possibly empty}"
export CC CXX CGO_CFLAGS CGO_LDFLAGS LD_LIBRARY_PATH LIBRARY_PATH
export CGO_ENABLED=1 GOFLAGS=
# Replace sm_XX with the authorized node's supported target.
go run ./cmd/fak-dev build-cuda build --arch sm_XX --capture-once-qualification --v41-defaults=false
export FAK_CUDA_REQUIRE_DEVICE=1 FAK_CUDA_GRAPH=1 FAK_CUDA_CAPTURE_ONCE=1
TAGS=cuda,fak_cuda_capture_once_qualification,fak_cuda_capture_once_device_witness
go test -json -p 1 -parallel 1 -count=1 -tags "$TAGS" ./internal/compute -run '^TestCUDAGraphOnceRequiredBorrowedLifetime$' | tee graph-compute.jsonl
go test -json -p 1 -parallel 1 -count=1 -tags "$TAGS" ./internal/model -run '^TestCUDAGraphOnceRequiredDenseModel$' | tee graph-model.jsonl
python3 - <<'CHECK'
import json
for path, name in [("graph-compute.jsonl", "TestCUDAGraphOnceRequiredBorrowedLifetime"),
                   ("graph-model.jsonl", "TestCUDAGraphOnceRequiredDenseModel")]:
    with open(path, encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    selected = [event for event in events if event.get("Test") == name]
    assert sum(event.get("Action") == "run" for event in selected) == 1, (path, "required test did not run exactly once")
    assert sum(event.get("Action") == "pass" for event in selected) == 1, (path, "required test did not pass")
    assert not any(event.get("Action") in ("skip", "fail") for event in events), (path, "skip/failure is not qualification")
print("Both named required-device witnesses ran and passed; retain their original evidence.")
CHECK
```

The complementary required-profile tests fail if the witness tag is selected
without cuda/cgo/capture-once qualification; they are negative admission controls,
not device execution. The JSON check also rejects missing named tests rather than
accepting Go's no-matching-tests success. Never treat an expected negative-control
failure as a hardware PASS.

Preserve exact commit, native build receipt/archive SHA, GPU/driver/toolkit details,
commands and literal cgo environment, JSON/stdout/stderr, exit codes and peak resource
telemetry. Success requires real model 1 warmup + 1 capture + 6 replays, logits at
the existing Approx gate (argmax exact/cosine >=0.999), conservative exact F32 KV
payload/positions, stable input/output/arena/KV addresses, position refusal and
checked teardown. The separate lifetime case uses real capture and MatMul;
foreign-receiver Free must retain borrowed storage, invalidate capture and refuse
initial launch until checked abort completes.

The qualification snapshot takes cudaMu and copies Go metadata only; no native
call, synchronization, mutation or capture enablement. Counters advance only after
successful native end/replay completion. Device tests and recipe are unrun at
preparation time. Source review, host controls and compilation are not runtime or
throughput qualification.
