package ggufload

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// estimate.go — the load-time device-fit pre-check for the GGUF loader (issue #709; the
// capacity-bridge Plank 5, docs/explainers/hardware-limits-and-capacity.md). The Go load
// paths (WeightSource.QuantModelProfile / QuantModelQ4K) allocate optimistically (the
// make([]byte, ...) in TensorBytes, the dequant buffers); a model too big for the box
// OOM-panics mid-load, losing the sizing context of what was needed. EstimateLoadBytes lifts
// the bytes the loader will demand OFF THE HEADER ALONE (no tensor read, no full load), and
// FitOnDevice turns compute.FitsOnDevice's verdict into a typed refusal BEFORE the allocation
// — fail-open on a backend that cannot probe (cpu-ref), so the portable floor loads exactly
// as before.
//
// EstimateLoadBytes sums each tensor's on-disk block payload (tensorPayloadBytes: the bytes
// the loader must read). For the memory-lean resident paths this is the resident footprint to
// within a small constant factor: the direct-Q4_K path holds those bytes RAW (byte-for-byte
// the ggml layout the forward reads), and the Q8_0 re-quant path's resident bytes are the same
// order (out*in int8 codes + per-block f32 scales). It is therefore a faithful order-of-
// magnitude fit proxy for the lean paths the GGUF loader exists to serve; the f32 dequant path
// (WeightSource.Model / LoadModel) resident is larger (elems*4), so treat it as a lower bound
// there. Either way an oversize model exceeds a small KNOWN ceiling and is refused; an unknown
// ceiling (cpu-ref) never is.

// EstimateLoadBytes reports the GGUF weight payload the loader will read, summed from the
// parsed header's tensor directory WITHOUT reading a single tensor — so a caller can ask
// "will this fit?" before the make/append in QuantModelProfile/QuantModelQ4K. It walks
// s.File.Tensors (the header) and sums each tensor's block payload (tensorPayloadBytes), the
// same sizing TensorBytes uses to allocate its read buffer, so the estimate tracks the real
// allocation rather than guessing. Safe to call right after OpenWeights (which parses only the
// header); it never touches a tensor byte.
func (s *WeightSource) EstimateLoadBytes() (int64, error) {
	var total uint64
	for _, info := range s.File.Tensors {
		n, err := tensorPayloadBytes(info)
		if err != nil {
			return 0, fmt.Errorf("gguf: estimate tensor %s: %w", info.Name, err)
		}
		total += n
	}
	if total > math.MaxInt64 {
		return 0, fmt.Errorf("gguf: estimated load bytes %d overflow int64", total)
	}
	return int64(total), nil
}

// FitOnDevice is the load-time device-fit refusal for a GGUF WeightSource: it estimates the
// load bytes off the header and returns a *compute.FitError ("needs ~W GiB, device has ~A
// GiB") ONLY when be is a capacity-reporting backend that KNOWS the model exceeds its ceiling.
// A backend that cannot probe (the cpu-ref floor, a device without a memory query) reports
// unknown capacity, so this returns nil — the load proceeds unchanged and the portable floor
// is never blocked (the fail-open contract). Call it BEFORE QuantModel / QuantModelQ4K to turn
// an oversize model into a typed refusal instead of an OOM panic; headroom in [0,1) reserves
// that fraction of the budget for the KV cache / activations / per-op scratch that do not
// pass through this single check (see compute.FitsOnDevice).
func (s *WeightSource) FitOnDevice(be compute.Backend, headroom float64) error {
	want, err := s.EstimateLoadBytes()
	if err != nil {
		return err
	}
	return compute.RefuseIfTooBig(be, want, headroom)
}
