package compute

import "testing"

// TestQwen35VulkanSequencePerLayerFencePolicy witnesses the SPOB fence-collapse from
// #13043 without a physical device. The prefill layer loop used to issue one
// fvk_batch_flush_status() submit+fence per layer (64 per dense prefill); the policy
// now defaults to zero per-layer flushes, with the legacy round-trip restored only
// under FAK_QWEN35_SEQUENCE_PER_LAYER_FENCE=1 for a matched A/B.
//
// This is a [SW-VERIFIED] policy witness: it asserts the flush decision function, not
// device execution or logits parity. Physical parity on gfx1151 remains
// [HW-WITNESSED] and is not claimed here.
func TestQwen35VulkanSequencePerLayerFencePolicy(t *testing.T) {
	t.Setenv(qwen35VulkanSequenceLayerFenceEnv, "")
	if qwen35VulkanSequencePerLayerFence() {
		t.Fatalf("default prefill policy must not fence per layer (%s unset)", qwen35VulkanSequenceLayerFenceEnv)
	}

	t.Setenv(qwen35VulkanSequenceLayerFenceEnv, "1")
	if !qwen35VulkanSequencePerLayerFence() {
		t.Fatalf("%s=1 must restore the legacy per-layer fence", qwen35VulkanSequenceLayerFenceEnv)
	}

	t.Setenv(qwen35VulkanSequenceLayerFenceEnv, "true")
	if qwen35VulkanSequencePerLayerFence() {
		t.Fatalf("%s must require the exact value \"1\"", qwen35VulkanSequenceLayerFenceEnv)
	}

	t.Setenv(qwen35VulkanSequenceLayerFenceEnv, "0")
	if qwen35VulkanSequencePerLayerFence() {
		t.Fatalf("%s=0 must remain collapsed", qwen35VulkanSequenceLayerFenceEnv)
	}
}
