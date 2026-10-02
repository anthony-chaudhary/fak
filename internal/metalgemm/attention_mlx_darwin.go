//go:build darwin && arm64 && cgo && metal

// Ordered Qwen attention uses the adapted MLX single-pass kernel for fp32 head
// dimensions 64, 128 and 256 with at most 4096 KV rows, unless the existing
// split-KV decode policy applies. FAK_QWEN35_ATTN_MLX=0 selects the historical
// kernel for A/B checks; =1 also admits other supported multiples of 32 up to
// 256. Devices without a 1024-thread pipeline retain the historical kernel.
// Qualification covers attention parity and GPU timing, not full-model speed.
package metalgemm

/*
unsigned long long mg_qwen35_attention_mlx_dispatch_count(void);
*/
import "C"

// AttentionMLXDispatchCount returns the number of copied MLX attention encoders
// dispatched by this process. It distinguishes the candidate from a fallback
// during bounded parity and timing checks; it does not count completed GPU work.
func AttentionMLXDispatchCount() uint64 {
	return uint64(C.mg_qwen35_attention_mlx_dispatch_count())
}
