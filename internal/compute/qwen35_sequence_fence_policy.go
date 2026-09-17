// qwen35_sequence_fence_policy.go - tag-free fence policy for the Qwen3.8 hybrid
// sequence prefill (#13043). The prefill layer loop used to issue one
// fvk_batch_flush_status() submit+fence per layer (64 per dense prefill) even though
// the whole sequence is bracketed by fvk_batch_begin() and a single final-fence.
//
// The default policy records every layer into that single batch and flushes once at
// the boundary; the legacy per-layer round-trip is restored only under
// FAK_QWEN35_SEQUENCE_PER_LAYER_FENCE=1 for a matched hardware A/B. This policy read
// lives outside the cgo-gated Vulkan source so it can be witnessed without a device.
package compute

import "os"

const qwen35VulkanSequenceLayerFenceEnv = "FAK_QWEN35_SEQUENCE_PER_LAYER_FENCE"

// qwen35VulkanSequencePerLayerFence reports whether the legacy per-layer fence policy
// is active. It is a pure policy read: any value other than exactly "1" keeps the
// collapsed default, so the flag is an explicit ablation, never an ambient toggle.
func qwen35VulkanSequencePerLayerFence() bool {
	return os.Getenv(qwen35VulkanSequenceLayerFenceEnv) == "1"
}
