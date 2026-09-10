# Native ROCm/HIP backend

The `rocm` build tag enables a Linux AMD device implementation of `compute.Backend`. HIP kernels execute inside fak's native model path. The default build needs no ROCm libraries. Select the backend explicitly with `--backend rocm`; registration requires successful device initialization.

## Build on Linux

Install a matching HIP compiler/runtime for the target GPU. `hipcc` and the runtime libraries must come from the same ROCm installation. Use the GPU's reported architecture; the example below targets `gfx1151`.

From the repository root:

```sh
export ROCM_PATH=/opt/rocm
"$ROCM_PATH/bin/hipcc" --version
"$ROCM_PATH/bin/rocm_agent_enumerator"
"$ROCM_PATH/bin/hipcc" -O3 -std=c++17 -fPIC --offload-arch=gfx1151 \
  -c internal/compute/rocm_kernels.hip -o internal/compute/rocm_kernels.o
ar rcs internal/compute/libfakrocm.a internal/compute/rocm_kernels.o
export CGO_ENABLED=1
export CGO_LDFLAGS="-L$ROCM_PATH/lib -Wl,-rpath,$ROCM_PATH/lib"
export LIBRARY_PATH="$ROCM_PATH/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
go build -tags rocm -o /tmp/fak-rocm ./cmd/fak
```

Build a fresh static library whenever its HIP source or header changes. This first backend selects visible device zero; `HIP_VISIBLE_DEVICES` can constrain which physical device that represents. It uses one serialized HIP stream.

## Correctness witness

```sh
FAK_ROCM_REQUIRE_DEVICE=1 go test -tags rocm ./internal/compute ./internal/model \
  -run 'Test(ROCmBackend|HALROCm)' -count=1 -v
```

The required-device setting makes missing registration a test failure. A default-build host test, successful compilation, or skipped device test does not establish hardware correctness. The backend is `Approx`: floating reductions may differ from the CPU reference, while the complete-forward fixture requires exact greedy token selection and prefill logit cosine of at least 0.999.

## Execution scope

The core implementation provides resident F32 tensors, packed Q8_0/Q4_K/Q5_K/Q6_K projection weights, matmul and batched matmul, RMSNorm, RoPE, SwiGLU, residual/bias addition, grouped attention, device argmax and resident KV append/clone/eviction. Unsupported storage formats or geometries fail explicitly.

This core slice targets standard transformer forwards. Hybrid Qwen GDN, automatic backend priority, multi-GPU collectives and comparative performance qualification require their own implementations and witnesses. The existing model admission gate rejects a hybrid model whose selected backend lacks the whole-operation GDN capability.

## References

- [HIP compilation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html)
- [ROCm Linux hardware compatibility](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
- [Implementation contract](../../docs/tickets/native-rocm-backend/TICKET-01.md)
