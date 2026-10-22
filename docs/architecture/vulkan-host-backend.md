# Vulkan host backend boundaries

The Vulkan backend currently needs its native bridge. The `vulkan`-tagged,
cgo-enabled Go adapter links `libfakvulkan` and calls the flat `fvk_*` ABI in
`internal/compute/vulkan_backend.h`. The implementation creates the device,
owns buffers and memory, builds pipelines, records commands, and submits work.
Deleting it would remove those operations from the supported Vulkan backend.
Default builds exclude this backend.

Vulkan exposes a C API; it does not require C++. The current host implementation
uses C++ containers, atomics, and templates. Replacing that implementation with
direct Go bindings is a separate migration tracked by #12210. The Windows
pure-Go loader probe establishes an API version, not a replacement compute
backend.

`internal/compute/vulkan_spirv_admission.h` provides a separate, standard-library
only API in `fak::vulkan::spirv`. Callers pass immutable module bytes and receive
ABI admission results. It owns no Vulkan handles, global device state, file
access, or initialization. The shim composes these checks with its selected
device capabilities before creating pipelines. Header-only definitions preserve
the existing single-object native build and support independent translation
units. The standalone test compiles that API without Vulkan headers or a loader.

Admission proves only the narrow structural properties checked by each parser.
It does not replace full SPIR-V validation, source and archive provenance,
floating-point device qualification, or physical numerical witnesses.

The remaining host runtime is still a singleton serialized by the Go execution
owner and `vulkanMu`. Allocation pools, descriptor lifetimes, batch recording,
submission failures, and quarantine share that ownership domain. They remain in
the shim; this extraction does not establish independent device contexts or
change their lifetime rules. Further decomposition needs explicit owners and
dependencies for those resources, with the C ABI kept stable during migration.
