package model

// metal_cb_accounting.go — session-local Metal command-buffer accounting for the
// resident-Q4_K Qwen3.8 decode lanes. It is the observation hook the W1
// command-buffer regression test reads: one increment per observed native
// dispatch (GEMV/GEMM/group/fused-MLP) plus one per fused Qwen35 decode graph.
// The counters are written only on the single-owner generation goroutine and are
// zero on the pure-Go path; the helpers are allocation-free when unused.

// ResetMetalCommandBuffers zeroes the session-local count of Metal command
// buffers committed by decode dispatch seams. Call once before a measured step.
func (s *Session) ResetMetalCommandBuffers() {
	if s == nil {
		return
	}
	s.metalCommandBuffers = 0
	s.metalGraphCommandBuffers = 0
}

// MetalCommandBuffers returns the number of Metal command buffers committed by
// this session since the last ResetMetalCommandBuffers: one buffer per observed
// native dispatch plus one per fused Qwen35 decode graph. This is the unit the
// W1 decode lever is defined on.
func (s *Session) MetalCommandBuffers() int {
	if s == nil {
		return 0
	}
	return s.metalCommandBuffers + s.metalGraphCommandBuffers
}

// MetalGraphCommandBuffers returns the subset committed by fused Qwen35 decode
// graphs (each graph is exactly one command buffer).
func (s *Session) MetalGraphCommandBuffers() int {
	if s == nil {
		return 0
	}
	return s.metalGraphCommandBuffers
}

// MetalDispatchCommandBuffers returns the subset committed by per-dispatch Metal
// calls (GEMV/GEMM/group/fused-MLP), i.e. the buffers a whole-token graph avoids.
func (s *Session) MetalDispatchCommandBuffers() int {
	if s == nil {
		return 0
	}
	return s.metalCommandBuffers
}

// countMetalCommandBuffer is called by the darwin dispatch seams; it is a no-op
// on the portable path because the counter simply accumulates zero.
func (s *Session) countMetalCommandBuffer(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.metalCommandBuffers += n
}

// countMetalGraphCommandBuffer records one or more fused Qwen35 decode-graph
// command buffers.
func (s *Session) countMetalGraphCommandBuffer(n int) {
	if s == nil || n <= 0 {
		return
	}
	s.metalGraphCommandBuffers += n
}
