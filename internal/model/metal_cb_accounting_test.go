package model

import "testing"

// TestMetalCommandBufferAccountingRoundTrip pins the session-local Metal
// command-buffer accounting hook the W1 decode tests read: reset zeroes both
// counters, dispatch and graph increments are independent and additive, and a
// nil/large/zero increment is inert.
func TestMetalCommandBufferAccountingRoundTrip(t *testing.T) {
	s := &Session{}
	s.ResetMetalCommandBuffers()
	if got := s.MetalCommandBuffers(); got != 0 {
		t.Fatalf("fresh MetalCommandBuffers=%d, want 0", got)
	}
	s.countMetalCommandBuffer(3)
	s.countMetalGraphCommandBuffer(1)
	if got, want := s.MetalCommandBuffers(), 4; got != want {
		t.Fatalf("MetalCommandBuffers=%d, want %d", got, want)
	}
	if got, want := s.MetalDispatchCommandBuffers(), 3; got != want {
		t.Fatalf("MetalDispatchCommandBuffers=%d, want %d", got, want)
	}
	if got, want := s.MetalGraphCommandBuffers(), 1; got != want {
		t.Fatalf("MetalGraphCommandBuffers=%d, want %d", got, want)
	}
	s.countMetalCommandBuffer(0)
	s.countMetalCommandBuffer(-2)
	s.countMetalGraphCommandBuffer(0)
	if got, want := s.MetalCommandBuffers(), 4; got != want {
		t.Fatalf("inert increments changed total to %d, want %d", got, want)
	}
	s.ResetMetalCommandBuffers()
	if got := s.MetalCommandBuffers(); got != 0 {
		t.Fatalf("reset MetalCommandBuffers=%d, want 0", got)
	}
	var nilSession *Session
	nilSession.ResetMetalCommandBuffers()
	if got := nilSession.MetalCommandBuffers(); got != 0 {
		t.Fatalf("nil session MetalCommandBuffers=%d, want 0", got)
	}
}
