# Experimental packed main-KV attention source

**UNCOMPILED, UNREGISTERED, UNEXECUTED.** Shared for independent compiler and
module review, not enabled GPU support. Compiler, SPIR-V validation, structural
inspection, device execution and numerical parity are all **UNRUN**. No ABI
admission, performance or device-support claim is made. Do not dispatch this
source through the existing F32 attention ABI or copy its output into a runtime
module directory.

## Source and provenance

`v41_packed_main_attention.comp` is the exact source-reviewed float-control
candidate (10,859 bytes), SHA-256:
`8f2cc0694be7f53d60b5be60e51f761320d7c2052ac2bd154ff833773cefd59c`.
Packaging makes no shader edits. The unchanged [CONTRACT.md](CONTRACT.md)
preserves the original qualification obligations, format provenance and proposed
interface, including mandatory host validation. Its historical source-only
statements describe preparation, not subsequent repository publication.

The candidate derives from Fak's `internal/compute/shaders/v41_shared_attention.comp`
at baseline `6ba2d64f4019ae37dd9750f1234c4236a34b7778` (baseline source SHA-256
`b2259bc3fca8511feb7d080853b857cae66fab70f7b392ebc336b2464bade063`). The initial
packed candidate hash was `9644b4eab6bf21bf100b1173b38c0ae09b52b52d01d15d8ff0c98cc351655f74`.
The final candidate only adds the `GL_EXT_spirv_intrinsics` directive, three global
float-control declarations and comments to that packed candidate. Existing
comments are preserved; repository [LICENSE](../../LICENSE) and
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) remain applicable.

Planned compiler: [Shaderc v2025.2](https://github.com/google/shaderc/blob/v2025.2/DEPS),
pinning glslang `697683e6b8871420d7d942b1a2fe233242ab5608` and SPIRV-Headers
`aa6cef192b8e693916eb713e7a9ccadf06062ceb`. The declarations use the documented
[Khronos extension](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_spirv_intrinsics.txt):
DenormPreserve (capability 4464/mode 4459), SignedZeroInfNanPreserve (4466/4461),
and RoundingModeRTE (4467/4462), each width 32. These are core since SPIR-V 1.4;
Vulkan 1.2's default SPIR-V 1.5 target is unchanged. Source declarations are not
proof of emitted instructions, arithmetic behavior or device support.

## Why normal builds cannot discover this candidate

Inspected at `e84aadb46907dec51dcfaa5d4a288ee7401d6ea5`:

- `internal/computebuild/vulkan_shaders.go` resolves registry stems exclusively
  beneath `internal/compute/shaders`; `vulkan_registry*.go` has no packed candidate.
- `internal/compute/build_vulkan.ps1` uses an explicit list under the same source
  directory. `internal/compute/vulkan_shim.cpp` has no packed candidate loader.
- `internal/amdgpu/strix_validation.go` and `strix_ssd_restart.go` glob only
  `src/internal/compute/shaders/*.comp`. `strixinstaller.go` probes explicit
  deployment/source shader directories and globs only their immediate children.

This experimental directory is outside those roots. No registry, build recipe,
loader, capability, generated module or default changes accompany it. Keep the
source here until a separately reviewed integration; being shared is not admission.

## Next verifier: explicit source-only compiler witness

Use an owned compiler lane and verified pinned tools. From the repository root,
set GLSLC, SPIRV_VAL and SPIRV_DIS to verified absolute tool paths; OUT must be a
fresh, caller-owned evidence directory outside all runtime module directories.
These are real tool commands, **not commands run for this publication**:

```sh
src=experiments/v41-packed-attention/v41_packed_main_attention.comp
sha256sum "$src"
"$GLSLC" --version
"$SPIRV_VAL" --version
"$SPIRV_DIS" --version
"$GLSLC" -O --target-env=vulkan1.2 -fshader-stage=comp "$src" -o "$OUT/candidate.spv"
"$SPIRV_VAL" --target-env vulkan1.2 "$OUT/candidate.spv"
"$SPIRV_DIS" "$OUT/candidate.spv" -o "$OUT/candidate.spvasm"
sha256sum "$OUT/candidate.spv"
```

Require the exact source hash before compiling. Record tool provenance, versions,
commands, exit statuses, logs and module hash; stop at each failed command. Do not
add an incompatible target-SPIR-V override. Successful compilation/validation
still does not satisfy the contract. Review actual module descriptor types,
push offsets, scalar NoContraction dependency chains, all three capabilities and
width-32 execution modes. There is no claimed packed-ABI inspector in this package.
Then separately qualify device float-control properties, immutable host admission,
resource lifetime, upload/dispatch/status/readback, exponential accuracy and the
full numerical cases in CONTRACT.md. Keep each unperformed stage UNRUN.
