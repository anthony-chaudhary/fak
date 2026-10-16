package model

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"sync/atomic"
)

// Copies share one logical normalization-loop claim. A native MISS does not
// mutate host data or native owners, but the scalar continuation still executes
// this loop once; copied carriers cannot replay that completed scalar work.
// Copies share one consumption state. Only the mHC retained readback constructor
// can preserve its flat allocation; arbitrary row slices cannot mint a carrier.
type v41OwnedFFNCollapse struct {
	state *v41OwnedFFNCollapseState
	rows  [][]float32
}
type v41OwnedFFNCollapseState struct {
	owner       *v41MHCResident
	flat        []float32
	width       int
	loopClaimed atomic.Bool
}

func newV41OwnedFFNCollapse(owner *v41MHCResident, flat []float32, rows, width int) *v41OwnedFFNCollapse {
	out := &v41OwnedFFNCollapse{state: &v41OwnedFFNCollapseState{owner: owner, flat: flat, width: width}, rows: make([][]float32, rows)}
	for row := range out.rows {
		out.rows[row] = flat[row*width : (row+1)*width : (row+1)*width]
	}
	return out
}

// Replacing ordinary behavior constructs a product without native authority.
// No function-pointer identity or inherited wrapper method grants a panel loan.
const v41FFNPoolPanelRows = 64
const v41FFNPoolPanelFallbackRows = 32

type v41FFNNormBinding struct {
	ordinary    v41FFNNormFunc
	owner       *Session
	model       *Model
	backend     compute.Backend
	publication *v41BF16PublicationBinding
}

func (b *v41FFNNormBinding) normFunc() v41FFNNormFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}
func (s *Session) v41BindFFNNorm() *v41FFNNormBinding {
	ordinary := s.v41FFNNormFunc()
	if ordinary == nil {
		return nil
	}
	b := &v41FFNNormBinding{ordinary: ordinary}
	if s.v41Forward == nil || s.v41Forward.gatherOps == nil {
		return b
	}
	publication := s.v41Forward.gatherOps.bf16Publish
	if publication != nil && publication.core != nil && publication.owner == s && publication.model == s.M &&
		backendHasComparableIdentity(s.Backend) && publication.backend == s.Backend {
		b.owner, b.model, b.backend, b.publication = s, s.M, s.Backend, publication
	}
	return b
}

func (s *Session) v41FFNPanelFailure(layer int, stage string, cause error) error {
	if s.halFailure == nil {
		s.halFailure = &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-ffn-norm", Layer: layer, Stage: stage, Cause: cause}
	}
	if !s.halClosed {
		func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
	}
	return s.halFailure
}

// panel takes one exclusive carrier, falling back within untouched chunks. Once
// native transport selects, a failure is fatal and the original input is consumed.
func (b *v41FFNNormBinding) panel(m *Model, st *v41ForwardState, scratch *v41ProjScratch, resident *v41MHCResident, layer int, input *v41OwnedFFNCollapse, eps float32) (out [][]float32, handled bool, cause error) {
	carrierFailure := func() ([][]float32, bool, error) {
		if input != nil && input.state != nil {
			owner := input.state.owner
			if owner != nil && owner.d != nil && owner.d.s != nil {
				return nil, true, owner.d.s.v41FFNPanelFailure(layer, "panel ownership", errV41ProjectionResult)
			}
		}
		if b != nil && b.owner != nil {
			return nil, true, b.owner.v41FFNPanelFailure(layer, "panel ownership", errV41ProjectionResult)
		}
		return nil, true, errV41ProjectionResult
	}
	// A claimed carrier may contain raw F32 output. Reject it before any
	// binding or payload decline, including callback replacement after a loan.
	if input != nil {
		if input.state == nil || input.state.loopClaimed.Load() {
			return carrierFailure()
		}
		n, ok := checkedMulInt(len(input.rows), input.state.width)
		if !ok || input.state.width <= 0 || n != len(input.state.flat) {
			return carrierFailure()
		}
		for i, row := range input.rows {
			if len(row) != input.state.width || cap(row) != input.state.width || &row[0] != &input.state.flat[i*input.state.width] {
				return carrierFailure()
			}
		}
	}
	if b == nil || b.owner == nil || input == nil || input.state == nil || len(input.rows) < 2 {
		return nil, false, nil
	}
	s := b.owner
	valid := func() bool {
		return b.model == m && s.M == m && backendHasComparableIdentity(s.Backend) && s.Backend == b.backend &&
			!s.halClosed && s.halFailure == nil && s.v41Forward != nil && s.v41Forward.ffnNorm == b &&
			st != nil && st.ffnNorm == b && st.callbackOwner == s && scratch != nil && scratch.ffnNorm == b &&
			st.gatherOps != nil && scratch.gatherOps == st.gatherOps && st.gatherOps.owner == s && st.gatherOps.bf16Publish == b.publication &&
			resident != nil && resident.d != nil && resident.d.s == s && input.state.owner == resident && resident.rows == len(input.rows)
	}
	if !valid() || layer < 0 || layer >= m.Cfg.NumLayers || eps != float32(m.Cfg.RMSNormEps) {
		return nil, false, nil
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full || m.Cfg.LayerNorm || m.Cfg.NormGain1p {
		return nil, false, nil
	}
	width := m.Cfg.HiddenSize
	count, ok := checkedMulInt(len(input.rows), width)
	if !ok || width <= 0 || input.state.width != width || len(input.state.flat) != count || !finite32(eps) || eps <= 0 {
		return nil, false, nil
	}
	for _, row := range input.rows {
		if len(row) != width || !v41BF16Widened(row) {
			return nil, false, nil
		}
	}
	if !input.state.loopClaimed.CompareAndSwap(false, true) {
		return nil, true, s.v41FFNPanelFailure(layer, "panel ownership", errV41ProjectionResult)
	}
	handled = true
	complete := false
	var primary any
	defer func() {
		if !complete {
			err, ok := primary.(error)
			if !ok {
				err = fmt.Errorf("incomplete FFN panel: %v", primary)
			}
			cause = s.v41FFNPanelFailure(layer, "panel completion", err)
			out = nil
		}
	}()
	body := func() {
		out = make([][]float32, len(input.rows))
		var weight compute.Tensor
		publish := b.publication.publicationFunc()
		const leaf = "ffn_norm.weight"
		for lo := 0; lo < len(input.rows); {
			if !valid() {
				cause = s.v41FFNPanelFailure(layer, "panel binding", errV41ProjectionResult)
				break
			}
			hi := min(lo+v41FFNPoolPanelRows, len(input.rows))
			// Exactly the scalar-required first-row publication/copy. A native miss
			// continues with this prepared row, rather than publishing it a second time.
			first, err := v41LatentNormBF16Publish(layer, leaf, "collapse", input.rows[lo], publish)
			if err != nil {
				cause = err
				break
			}
			selected := false
			var published *compute.V41FFNNormPublication
			flat := input.state.flat[lo*width : hi*width : hi*width]
			attempt := func(gain compute.Tensor) ([]float32, bool, error) {
				var err error
				try := func() {
					published, selected, err = compute.TryV41FFNNormPublication(b.backend, gain, flat, hi-lo, width, eps)
					if published != nil && err == nil && v41HostBF16Witness.Load() {
						v41BF16PublishCalls.Add(1)
						v41BF16PanelCalls.Add(1)
						v41BF16PanelRows.Add(int64(hi - lo))
					}
					if selected || err != nil {
						return
					}
					selected, err = compute.TryV41PoolNormPanel(b.backend, gain, flat, hi-lo, width, eps)
				}
				try()
				// Only a pre-effect miss of both routes can choose 32 rows.
				if !selected && err == nil && hi-lo > v41FFNPoolPanelFallbackRows {
					hi = lo + v41FFNPoolPanelFallbackRows
					flat = input.state.flat[lo*width : hi*width : hi*width]
					try()
				}
				if selected || err != nil {
					return flat, true, err
				}
				return nil, false, nil
			}
			raw, err := s.v41RMSNormPrepared(layerName(layer, leaf), first, width, 1, "v41-ffn-norm", layer, nil, &weight, attempt)
			first = nil
			if err != nil {
				cause = v41ProjectionOperationErr(layer, leaf, err)
				break
			}
			for row := lo; row < hi; row++ {
				if !valid() {
					cause = s.v41FFNPanelFailure(layer, "panel binding", errV41ProjectionResult)
					break
				}
				values := raw
				if selected {
					values = flat[(row-lo)*width : (row-lo+1)*width : (row-lo+1)*width]
				} else if row > lo {
					staged, err := v41LatentNormBF16Publish(layer, leaf, "collapse", input.rows[row], publish)
					if err != nil {
						cause = err
						break
					}
					values, err = s.v41RMSNormPrepared(layerName(layer, leaf), staged, width, 1, "v41-ffn-norm", layer, nil, &weight, nil)
					if err != nil {
						cause = v41ProjectionOperationErr(layer, leaf, err)
						break
					}
				}
				if len(values) != width {
					cause = v41ProjectionOperationErr(layer, leaf, s.v41FFNPanelFailure(layer, "readback", errV41ProjectionResult))
					break
				}
				for _, v := range values {
					if !finite32(v) {
						cause = v41ProjectionOperationErr(layer, leaf, s.v41FFNPanelFailure(layer, "readback", errV41ProjectionResult))
						break
					}
				}
				if cause != nil {
					break
				}
				if published != nil {
					// Raw finiteness above precedes every rounding fault in this
					// row. Global device minimum is translated to a row-local index.
					out[row], err = published.Row(row - lo)
					if err != nil {
						err = s.v41CloseBF16PublicationFailure(b.backend, layer, leaf+" normalization", "checked publication", err)
					} else if !v41BF16Widened(out[row]) {
						err = s.v41CloseBF16PublicationFailure(b.backend, layer, leaf+" normalization", "checked publication readback", errV41BF16BoundaryResult)
					}
				} else {
					out[row], err = v41LatentNormBF16Publish(layer, leaf, "normalization", values, publish)
				}
				if err != nil {
					cause = err
					break
				}
				if len(out[row]) != width {
					cause = v41ProjectionOperationErr(layer, leaf, s.v41FFNPanelFailure(layer, "publication result", errV41ProjectionResult))
					break
				}
			}
			if cause != nil {
				break
			}
			lo = hi
		}
		if cause != nil {
			out = nil
		}
	}
	func() { defer func() { primary = recover() }(); body(); complete = true }()
	if !complete {
		panic(primary)
	}
	return out, handled, cause
}
