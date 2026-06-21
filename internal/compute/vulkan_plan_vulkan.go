//go:build vulkan && windows

package compute

// vulkan_plan_vulkan.go — the Vulkan half of the issue #10 command-buffer wiring. It is
// compiled ONLY under -tags vulkan (and Windows, where the AMD GPU is reachable through the
// Vulkan loader) and does nothing but pin the contract: the Vulkan backend, which already
// exposes BeginBatch/FlushBatch (vulkan.go), MUST satisfy the BatchRecorder seam declared in
// the always-compiled vulkan_plan.go. If a future edit changes either side's signature, this
// assertion fails the Vulkan build instead of silently degrading the forward pass back to a
// per-op submit. On a non-Vulkan build this file is absent, so the default artifact stays
// pure-Go and compiles without it (the guarded-stub rule prefill_cuda.go uses for the
// CUDA-graph capturer).
var _ BatchRecorder = (*vulkanBackend)(nil)
