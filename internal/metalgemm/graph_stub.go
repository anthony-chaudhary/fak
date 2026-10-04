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
	// Graph-tracked allocation evidence; unavailable on the portable stub.
	AllocatedBuffers    int
	RetainedBufferBytes uint64
	// Executed P=1 GEMV kernels; always empty on the portable stub.
	Q4KGEMVKernels, Q6KGEMVKernels GEMVKernelSet
}
type GraphResult struct{}
type ProjectionGraph struct{}

func BeginProjectionGraph([]float32, []int8, []float32, int, int) (*ProjectionGraph, error) {
	return nil, errors.New("metalgemm: projection graph unavailable")
}

// AddBiasInPlace is the portable stub of the graph bias op. A ProjectionGraph can never
// be constructed without the Metal graph, so this is only reachable through a nil or
// zero receiver and always declines.
func (g *ProjectionGraph) AddBiasInPlace(*GraphResult, []float32) error {
	return errors.New("metalgemm: projection graph unavailable")
}

// DeviceKV is the portable stub for the device-resident KV triple (#13087). The
// real type lives in the darwin/arm64/cgo graph.go; portable callers (internal/model)
// reference the type and its method surface to size and thread a pair behind the
// admission seam, so the surface must exist on every build. NewDeviceKV always
// returns nil here, which callers treat as a decline (fail-open to the host walk);
// construction never succeeds without the Metal graph, so the mutators below are
// only reachable through a nil receiver and return errDeviceKVUnavailable.
type DeviceKV struct{}

// errDeviceKVUnavailable is returned by every stub mutator: a DeviceKV can never be
// constructed without the Metal graph, so the portable lane declines rather than
// pretending to have device storage.
var errDeviceKVUnavailable = errors.New("metalgemm: device KV unavailable without the Metal graph")

// NewDeviceKV always declines on the portable lane: there is no device allocator.
func NewDeviceKV(layers, tokens, kvWidth int) *DeviceKV { return nil }

// NewDeviceKVAttendOnly always declines on the portable lane.
func NewDeviceKVAttendOnly(layers, tokens, kvWidth int) *DeviceKV { return nil }

// DeviceKVResidentBytes is 0 on the portable lane: no pair can be allocated.
func DeviceKVResidentBytes() int64 { return 0 }

// ResidentBytes reports 0 for the stub (no allocation).
func (d *DeviceKV) ResidentBytes() int64 { return 0 }

// LayerStride reports 0 for the stub (no allocation).
func (d *DeviceKV) LayerStride() int { return 0 }

// Upload is unreachable on a nil stub pair; it declines.
func (d *DeviceKV) Upload(side int, src []float32) error { return errDeviceKVUnavailable }

// UploadRegion is unreachable on a nil stub pair; it declines.
func (d *DeviceKV) UploadRegion(side, off int, src []float32) error {
	return errDeviceKVUnavailable
}

// Download is unreachable on a nil stub pair; it declines.
func (d *DeviceKV) Download(side int, dst []float32) error { return errDeviceKVUnavailable }

// DownloadRegion is unreachable on a nil stub pair; it declines.
func (d *DeviceKV) DownloadRegion(side, off int, dst []float32) error {
	return errDeviceKVUnavailable
}

// Close is a no-op on the stub (nothing to free), safe on a nil receiver.
func (d *DeviceKV) Close() {}
