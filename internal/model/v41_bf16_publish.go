package model

import (
	"errors"
	"fmt"
	"math"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Test-only witnesses. Production pays one atomic load per host round: the
// counters move only while a test has switched the witness on.
var (
	v41HostBF16Witness   atomic.Bool
	v41HostBF16Rounds    atomic.Int64
	v41BF16PublishCalls  atomic.Int64
	v41BF16ResidentCalls atomic.Int64
)

// enableV41BF16Witness zeroes the host-round and device-publication counters and
// turns them on until the returned stop runs.
func enableV41BF16Witness() (stop func()) {
	v41HostBF16Rounds.Store(0)
	v41BF16PublishCalls.Store(0)
	v41BF16ResidentCalls.Store(0)
	v41IndexWeightScaleCalls.Store(0)
	v41HostBF16Witness.Store(true)
	return func() { v41HostBF16Witness.Store(false) }
}

// v41BF16Widened reports whether every value is finite widened BF16. RNE is the
// identity on such a row, so publishing it performs no arithmetic.
func v41BF16Widened(values []float32) bool {
	for _, v := range values {
		if math.Float32bits(v)&0xffff != 0 || !finite32(v) {
			return false
		}
	}
	return true
}

// v41BF16Overflows is the bit test for a finite F32 whose BF16 RNE is infinite:
// every magnitude from the tie above the largest finite BF16 up to infinity.
func v41BF16Overflows(value float32) bool {
	magnitude := math.Float32bits(value) & 0x7fffffff
	return magnitude >= 0x7f7f8000 && magnitude < 0x7f800000
}

// v41BF16PublishFault returns the lowest index a BF16 publication refuses
// (nonfinite, or finite with an infinite RNE) without rounding anything.
func v41BF16PublishFault(values []float32) (int, bool) {
	for i, v := range values {
		if !finite32(v) || v41BF16Overflows(v) {
			return i, true
		}
	}
	return 0, false
}

// v41PublishBF16 returns an owned widened-BF16 copy of values. An already
// widened row is copied unchanged; otherwise a bound publication rounds on the
// device and a nil publication keeps the caller's host rounding.
func v41PublishBF16(layer int, site string, values []float32, publish v41BF16BoundaryFunc, host func([]float32) ([]float32, error)) ([]float32, error) {
	if v41BF16Widened(values) {
		out := make([]float32, len(values))
		copy(out, values)
		return out, nil
	}
	if publish == nil {
		return host(values)
	}
	return v41BF16BoundaryCopy(layer, site, values, publish)
}

// v41LatentNormBF16Publish is v41LatentNormBF16Copy routed through the session's
// ordinary device publication when one is bound.
func v41LatentNormBF16Publish(layer int, leaf, phase string, values []float32, publish v41BF16BoundaryFunc) ([]float32, error) {
	return v41PublishBF16(layer, leaf+" "+phase, values, publish, func(values []float32) ([]float32, error) {
		return v41LatentNormBF16Copy(layer, leaf, phase, values)
	})
}

const (
	v41TailRoPEInputSite       = "tail-rope input"
	v41TailRoPEOutputSite      = "tail-rope output"
	v41InverseOutputInputSite  = "inverse-output input"
	v41InverseOutputOutputSite = "inverse-output output"
	v41RoutedGateUpSite        = "routed gate/up"
	v41RoutedDownSite          = "routed down"
)

var v41BF16PublishSites = map[string]uint32{
	"attn.wq_a_norm.weight projection":          compute.V41BF16PublishQueryNormInput,
	"attn.wq_a_norm.weight normalization":       compute.V41BF16PublishQueryNormOutput,
	"attn.kv_norm.weight projection":            compute.V41BF16PublishKVNormInput,
	"attn.kv_norm.weight normalization":         compute.V41BF16PublishKVNormOutput,
	"ffn_norm.weight collapse":                  compute.V41BF16PublishFFNNormInput,
	"ffn_norm.weight normalization":             compute.V41BF16PublishFFNNormOutput,
	v41TailRoPEInputSite:                        compute.V41BF16PublishTailRoPEInput,
	v41TailRoPEOutputSite:                       compute.V41BF16PublishTailRoPEOutput,
	v41InverseOutputInputSite:                   compute.V41BF16PublishInverseOutputInput,
	v41InverseOutputOutputSite:                  compute.V41BF16PublishInverseOutputOutput,
	"attn.wo_a.weight input":                    compute.V41BF16PublishGroupedOutputInput,
	"attn.wo_a.weight projection":               compute.V41BF16PublishGroupedOutputProjection,
	"attn.wo_b.weight projection":               compute.V41BF16PublishGroupedOutputProjection,
	"ffn.shared_experts.w1.weight input":        compute.V41BF16PublishSharedExpertInput,
	"ffn.shared_experts.w1.weight projection":   compute.V41BF16PublishSharedExpertProjection,
	"ffn.shared_experts.w3.weight projection":   compute.V41BF16PublishSharedExpertProjection,
	"ffn.shared_experts.w2.weight projection":   compute.V41BF16PublishSharedExpertProjection,
	"ffn.shared_experts.activation down input":  compute.V41BF16PublishSharedExpertDownInput,
	v41RoutedGateUpSite:                         compute.V41BF16PublishRoutedGateUp,
	v41RoutedDownSite:                           compute.V41BF16PublishRoutedDown,
	"indexer.wk.weight projection":              compute.V41BF16PublishIndexerProjection,
	"indexer.wq_b.weight projection":            compute.V41BF16PublishIndexerProjection,
	"indexer.weights_proj.weight projection":    compute.V41BF16PublishIndexerProjection,
	"indexer.k_norm.weight normalization":       compute.V41BF16PublishIndexerKeyNorm,
	"indexer.wq_b.weight rotary":                compute.V41BF16PublishIndexerRotary,
	"indexer.k_norm.weight rotary":              compute.V41BF16PublishIndexerRotary,
	"indexer.weights_proj.weight scaled weight": compute.V41BF16PublishIndexerScaledWeight,
	"attn.compressor.norm.weight input":         compute.V41BF16PublishCompressorNormInput,
	"attn.compressor.norm.weight output":        compute.V41BF16PublishCompressorNormOutput,
}

// v41BF16PublishSiteID is the closed model-side site table: final-norm sites
// belong to layer -1, every other site to a configured decoder layer.
func v41BF16PublishSiteID(layer, layers int, site string) uint32 {
	if layer == -1 {
		switch site {
		case "model.norm.weight input":
			return compute.V41BF16PublishFinalNormInput
		case "model.norm.weight output":
			return compute.V41BF16PublishFinalNormOutput
		}
		return 0
	}
	if layer < 0 || layer >= layers {
		return 0
	}
	return v41BF16PublishSites[site]
}

// v41BF16PublishFunc binds the ordinary checked BF16 publication for every
// full-profile boundary except the attention-input norm, which keeps its own
// entry. Missing capability retains the host rounding before selection; a
// selected failure retires this Session without host replay. Backend identity is
// immutable for the Session lifetime; restored state binds a new closure.
func (s *Session) v41BF16PublishFunc() v41BF16BoundaryFunc {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!backendHasComparableIdentity(s.Backend) ||
		!compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.F32) {
		return nil
	}
	backend := s.Backend
	layers := s.M.Cfg.NumLayers
	operation, ok := backend.(compute.V41BF16PublishBackend)
	if !ok || !operation.SupportsV41BF16Publish(max(1, s.M.Cfg.HiddenSize)) {
		return nil
	}
	return func(layer int, site string, values []float32) (result []float32, cause error) {
		s.ensureOpenBackendSession()
		stage := "payload"
		closeSelected := func(primary error) {
			if s.halFailure == nil {
				s.halFailure = primary
			}
			if s.halClosed {
				return
			}
			// A secondary Close panic must not replace the selected failure
			// or leave this session eligible for another operation.
			defer func() { s.halClosed = true; _ = recover() }()
			s.Close()
		}
		closeFailure := func(err error) error {
			if s.halFailure != nil {
				closeSelected(s.halFailure)
				return s.halFailure
			}
			selected := &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: err}
			closed := &BackendForwardOperationError{Backend: backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-bf16-publish", Layer: layer, Stage: stage, Cause: selected}
			closeSelected(closed)
			return closed
		}
		defer func() {
			if r := recover(); r != nil {
				if err, ok := r.(error); ok {
					var closed *BackendForwardOperationError
					if errors.As(err, &closed) {
						closeSelected(err)
						panic(r)
					}
					var backendErr *compute.BackendError
					if errors.As(err, &backendErr) {
						result, cause = nil, closeFailure(err)
						return
					}
				}
				if err, ok := compute.ConvertCUDAPanic(r, "", ""); ok {
					if original, ok := r.(error); ok {
						err = original
					}
					result, cause = nil, closeFailure(err)
					return
				}
				err, ok := r.(error)
				if !ok {
					err = fmt.Errorf("unclassified backend panic: %v", r)
				}
				closeFailure(err)
				panic(r)
			}
		}()
		siteID := v41BF16PublishSiteID(layer, layers, site)
		if siteID == 0 || len(values) == 0 || len(values) > math.MaxInt32 || len(values) > math.MaxInt/4 ||
			!operation.SupportsV41BF16Publish(len(values)) {
			return nil, closeFailure(errV41BF16BoundaryResult)
		}
		run := func() (result []float32, cause error) {
			var owned []compute.Tensor
			defer func() {
				primaryPanic := recover()
				var cleanupPanic any
				for i := len(owned) - 1; i >= 0; i-- {
					func() {
						defer func() {
							if r := recover(); r != nil && cleanupPanic == nil {
								cleanupPanic = r
							}
						}()
						backend.Free(owned[i])
					}()
				}
				if primaryPanic != nil {
					panic(primaryPanic)
				}
				if cause == nil && cleanupPanic != nil {
					panic(cleanupPanic)
				}
			}()
			stage = "input upload"
			input := uploadHostF32Class(backend, []int{len(values)}, values, compute.MemoryActivation, "V4.1 BF16 publication input")
			if input.Buf() == nil || input.Backend() != backend {
				return nil, errV41BF16BoundaryResult
			}
			owned = append(owned, input)
			if input.Dtype != compute.F32 || input.Layout != compute.RowMajor || input.Quant != nil ||
				len(input.Shape) != 1 || input.Shape[0] != len(values) {
				return nil, errV41BF16BoundaryResult
			}
			// The device validates every word; a nonfinite input or rounded overflow
			// is attributed to its lowest index and the output is withheld.
			stage = "checked publication"
			out, err := operation.V41BF16Publish(input, siteID)
			fresh := out.Buf() != nil && out.Backend() == backend && out.Buf() != input.Buf()
			if fresh {
				owned = append(owned, out)
			}
			if err != nil {
				return nil, err
			}
			if !fresh || out.Dtype != compute.F32 || out.Layout != compute.RowMajor || out.Quant != nil ||
				len(out.Shape) != 1 || out.Shape[0] != len(values) {
				return nil, errV41BF16BoundaryResult
			}
			if v41HostBF16Witness.Load() {
				v41BF16PublishCalls.Add(1)
			}
			stage = "checked publication readback"
			read := backend.Read(out)
			if len(read) != len(values) || !v41BF16Widened(read) {
				return nil, errV41BF16BoundaryResult
			}
			return append([]float32(nil), read...), nil
		}
		result, cause = run()
		if cause != nil {
			return nil, closeFailure(cause)
		}
		return result, nil
	}
}

// v41PublishResidentBF16 publishes a device-resident F32 [N] result before its
// one readback, so a selected seam never rounds on the host. ok=false means the
// backend lacks the ordinary operation and the caller keeps its host rounding.
// The returned tensor is fresh and owned by the caller.
func (s *Session) v41PublishResidentBF16(t compute.Tensor, site uint32) (compute.Tensor, bool, error) {
	operation, ok := s.Backend.(compute.V41BF16PublishBackend)
	count, valid := 1, len(t.Shape) > 0
	for _, d := range t.Shape {
		if d <= 0 || count > math.MaxInt32/d {
			valid = false
			break
		}
		count *= d
	}
	if !ok || !valid || !operation.SupportsV41BF16Publish(count) {
		return compute.Tensor{}, false, nil
	}
	// The publication is elementwise, so a multi-dimensional row-major result is
	// published through a flat [N] view of the same storage.
	flat := t
	flat.Shape = []int{count}
	out, err := operation.V41BF16Publish(flat, site)
	if err != nil {
		if out.Buf() != nil && out.Buf() != t.Buf() {
			s.Backend.Free(out)
		}
		return compute.Tensor{}, true, err
	}
	if out.Buf() == nil || out.Buf() == t.Buf() || out.Dtype != compute.F32 || len(out.Shape) != 1 || out.Shape[0] != count {
		if out.Buf() != nil && out.Buf() != t.Buf() {
			s.Backend.Free(out)
		}
		return compute.Tensor{}, true, errV41BF16BoundaryResult
	}
	if v41HostBF16Witness.Load() {
		v41BF16ResidentCalls.Add(1)
	}
	return out, true, nil
}

// v41DeviceOnlyBF16Bound reports whether every reachable BF16 boundary
// publishes on the device: the ordinary publication plus, on the full profile,
// the attention-input pair, which keeps its own checked entry.
func v41DeviceOnlyBF16Bound(s *Session) bool {
	if s == nil || s.M == nil {
		return false
	}
	full, err := v41ForwardGeometry(s.M.Cfg)
	if err != nil || s.v41BF16PublishFunc() == nil {
		return false
	}
	return !full || s.v41AttentionBF16Func() != nil
}
