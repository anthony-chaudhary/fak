package modelengine

import (
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestKVUtilizationEmission(t *testing.T) {
	prom := `# TYPE nv_trt_llm_kv_cache_block_metrics gauge
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="used"} 32
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="max"} 128
nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="tokens_per_block"} 64
`
	snap, ok := ParseTritonKVBlockMetrics(prom)
	if !ok {
		t.Fatal("ParseTritonKVBlockMetrics returned ok=false")
	}
	if snap.UsedBlocks != 32 || snap.MaxBlocks != 128 || snap.TokensPerBlock != 64 {
		t.Fatalf("parsed=%+v", snap)
	}
	if math.Abs(snap.KVUtilization-0.25) > 1e-9 {
		t.Fatalf("KVUtilization=%v, want 0.25", snap.KVUtilization)
	}
	if snap.MaxTokenCapacity != 8192 {
		t.Fatalf("MaxTokenCapacity=%d, want 8192", snap.MaxTokenCapacity)
	}

	var b strings.Builder
	writeNativePreemptionMetrics(&b, NativePreemptionStats{UsedBlocks: 32, MaxBlocks: 128})
	if !strings.Contains(b.String(), "fak_sched_preempt_used_blocks 32") {
		t.Fatalf("missing used_blocks in %s", b.String())
	}
}
