//go:build !(darwin && arm64 && cgo)

package metalgemm

import "errors"

// PromptPanelMaxTokens is the widest prompt panel the Qwen3.8 whole-forward graph
// admits in one command buffer. It is ALSO needed by portable callers (e.g.
// internal/model's Qwen3.6 prefill panel walk) that size their panels from it
// without linking the Metal graph, so the constant is part of the portable
// contract42 and must not be darwin-only. Kept byte-identical to graph.go's value;
// prompt_panel_lockstep_test.go pins the two to each other.
const PromptPanelMaxTokens = 128

type GraphReceipt struct {
	Committed, CompletedWait, TimingAvailable bool
	Encoders, IntermediateWaits               int
	IntermediateReadbacks, HostReadbacks      int
	HostUploadBytes, HostReadbackBytes        uint64
	GPUMilliseconds, WaitMilliseconds         float64
}
type GraphResult struct{}
type ProjectionGraph struct{}

func BeginProjectionGraph([]float32, []int8, []float32, int, int) (*ProjectionGraph, error) {
	return nil, errors.New("metalgemm: projection graph unavailable")
}
