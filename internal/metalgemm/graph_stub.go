//go:build !(darwin && arm64 && cgo)

package metalgemm

import "errors"

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
