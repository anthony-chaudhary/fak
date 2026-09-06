# Build the native Vulkan backend on Linux

The `vulkan` build includes the existing fak compute kernels on Linux and Windows.
Linux requires cgo, a C++17 compiler, `ar`, Vulkan headers and loader, and `glslc`.
At runtime, install the Vulkan ICD for the physical GPU (for example Mesa RADV on
AMD). Installing HIP alone does not register a fak compute backend.

From an isolated source checkout, allocate a build directory with `fak tree-doctor
--scratch-dir vulkan-linux`, then compile the shaders and static shim:

```sh
build="$PWD/_scratch/vulkan-linux"
mkdir -p "$build/spirv"
for shader in internal/compute/shaders/*.comp; do
  name=$(basename "$shader" .comp)
  glslc -O --target-env=vulkan1.2 -fshader-stage=comp "$shader" -o "$build/spirv/$name.spv" || exit
done
c++ -O3 -std=c++17 -fPIC -c internal/compute/vulkan_shim.cpp -o "$build/vulkan_shim.o" || exit
ar rcs "$build/libfakvulkan.a" "$build/vulkan_shim.o" || exit
CGO_ENABLED=1 CGO_LDFLAGS="-L$build" go test -c -tags vulkan -o "$build/compute.test" ./internal/compute
```

The Linux cgo file links `libvulkan` and `libstdc++`; Windows retains its existing
`build_vulkan.ps1` toolchain and linker configuration. Generated archives, objects,
SPIR-V, and binaries stay in the allocated build directory.

Run a bounded physical-device witness before loading a model. Select the correct
ICD explicitly when multiple drivers are installed. Substitute the expected
physical-device name or architecture below; accepting a software renderer does
not witness GPU execution.

```sh
FAK_VULKAN_SPIRV="$build/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 \
  FAK_VULKAN_EXPECT_DEVICE=8060S \
  timeout 30s "$build/compute.test" -test.run '^TestVulkanArgmaxExact$' -test.v -test.timeout 25s
```

This test reports the registered device and checks exact numerical results for
three small argmax inputs. The required-device mode fails when registration is
absent, instead of accepting a skipped device test. Keep its binary hash, source
revision, driver version, and output together. A passing primitive witness proves
only this dispatch; full-model quality, GDN execution, and throughput require
separate model-bound witnesses.

Build the serving binary with the same link configuration and `-tags vulkan`.
Explicitly select `--engine inkernel --backend vulkan --gguf <model>` and set
`FAK_VULKAN_SPIRV` when launching it. Use a separate loopback listener and verify
available memory and session ownership before a model run; never replace an active
endpoint to collect a benchmark.

The [2026-09-06 physical-device receipt](../benchmarks/vulkan-linux-11902.json)
records a successful AMD Radeon 8060S argmax witness. It does not establish
full-model execution or inference throughput.
