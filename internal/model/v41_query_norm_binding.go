package model

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Callback replacement creates an ordinary product without pool authority.
// The panel is a private synchronous continuation over the fresh wq_a result.
type v41QueryNormBinding struct {
	ordinary    v41QueryNormFunc
	owner       *Session
	model       *Model
	backend     compute.Backend
	publication *v41BF16PublicationBinding
}

func (b *v41QueryNormBinding) normFunc() v41QueryNormFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}
func (s *Session) v41BindQueryNorm() *v41QueryNormBinding {
	ordinary := s.v41QueryNormFunc()
	if ordinary == nil {
		return nil
	}
	b := &v41QueryNormBinding{ordinary: ordinary}
	if s.v41Forward == nil || s.v41Forward.gatherOps == nil {
		return b
	}
	publication := s.v41Forward.gatherOps.bf16Publish
	if publication != nil && publication.core != nil && publication.owner == s && publication.model == s.M && backendHasComparableIdentity(s.Backend) && publication.backend == s.Backend {
		b.owner, b.model, b.backend, b.publication = s, s.M, s.Backend, publication
	}
	return b
}
func (s *Session) v41QueryPanelFailure(layer int, stage string, cause error) error {
	if s.halFailure == nil {
		s.halFailure = &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-query-norm", Layer: layer, Stage: stage, Cause: cause}
	}
	if !s.halClosed {
		func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
	}
	return s.halFailure
}

// panel runs scalar row zero first. Every miss runs one ordinary separator row;
// it never flushes merely to manufacture pool eligibility. Successful chunks
// overwrite only this private projection panel and publish traces row by row.
func (b *v41QueryNormBinding) panel(m *Model, scratch *v41ProjScratch, layer int, flat []float32, rows int, eps float32, trace func(int, []float32)) (handled bool, cause error) {
	if b == nil || b.owner == nil || m == nil || scratch == nil || rows < 3 {
		return false, nil
	}
	s := b.owner
	valid := func() bool {
		return b.model == m && s.M == m && backendHasComparableIdentity(s.Backend) && s.Backend == b.backend && !s.halClosed && s.halFailure == nil && s.v41Forward != nil && s.v41Forward.queryNorm == b && scratch.queryNorm == b && s.v41Forward.gatherOps != nil && scratch.gatherOps == s.v41Forward.gatherOps && scratch.gatherOps.owner == s && scratch.gatherOps.bf16Publish == b.publication
	}
	if !valid() || layer < 0 || layer >= m.Cfg.NumLayers || eps != float32(m.Cfg.RMSNormEps) || !finite32(eps) || eps <= 0 {
		return false, nil
	}
	full, err := v41ForwardGeometry(m.Cfg)
	width := m.Cfg.QLoraRank
	count, ok := checkedMulInt(rows, width)
	if err != nil || !full || m.Cfg.LayerNorm || m.Cfg.NormGain1p || width != 1280 || m.Cfg.HiddenSize != 5120 || !ok || len(flat) != count {
		return false, nil
	}
	handled = true
	complete := false
	var primary any
	defer func() {
		if !complete {
			err, ok := primary.(error)
			if !ok {
				err = fmt.Errorf("incomplete query panel: %v", primary)
			}
			cause = s.v41QueryPanelFailure(layer, "panel completion", err)
		}
	}()
	const leaf = "attn.wq_a_norm.weight"
	publish := b.publication.publicationFunc()
	scalar := func(row int) error {
		values := flat[row*width : (row+1)*width : (row+1)*width]
		if err := m.v41QueryNormInPlaceWithPublish(layer, values, eps, b.ordinary, publish); err != nil {
			return err
		}
		trace(row, values)
		return nil
	}
	body := func() {
		// This exact original prefix stages the gain and drains prior projection
		// loans as a normal consequence of scalar execution, not a forced warmup.
		if cause = scalar(0); cause != nil {
			return
		}
		capacityRows := min(rows, 64)
		for lo := 1; lo < rows; {
			if !valid() {
				cause = s.v41QueryPanelFailure(layer, "panel binding", errV41ProjectionResult)
				return
			}
			hi := min(lo+64, rows)
			chunk := flat[lo*width : hi*width : hi*width]
			gain, present := s.halW[layerName(layer, leaf)]
			// Pure numeric precheck can decline a chunk, but never reports a later
			// input error ahead of the earlier row's normalization/publication.
			_, inputFault := v41BF16PublishFault(chunk)
			var result *compute.V41QueryNormPublication
			selected := false
			if hi-lo >= 2 && present && gain.Buf() != nil && !inputFault {
				result, selected, err = compute.TryV41QueryNormPublication(b.backend, gain, chunk, hi-lo, width, capacityRows, eps)
			}
			if err != nil {
				cause = v41ProjectionOperationErr(layer, leaf, s.v41QueryPanelFailure(layer, "panel transport", err))
				return
			}
			if !selected {
				if cause = scalar(lo); cause != nil {
					return
				}
				lo++
				continue
			}
			if result == nil {
				cause = v41ProjectionOperationErr(layer, leaf, s.v41QueryPanelFailure(layer, "panel transport", errV41ProjectionResult))
				return
			}
			if v41HostBF16Witness.Load() {
				v41BF16PublishCalls.Add(2)
				v41BF16PanelCalls.Add(2)
				v41BF16PanelRows.Add(int64(2 * (hi - lo)))
			}
			for row := lo; row < hi; row++ {
				if !valid() {
					cause = s.v41QueryPanelFailure(layer, "panel binding", errV41ProjectionResult)
					return
				}
				if err = result.InputRowError(row - lo); err != nil {
					cause = s.v41CloseBF16PublicationFailure(b.backend, layer, leaf+" projection", "checked publication", err)
					return
				}
				raw := flat[row*width : (row+1)*width : (row+1)*width]
				for _, v := range raw {
					if !finite32(v) {
						cause = v41ProjectionOperationErr(layer, leaf, s.v41QueryPanelFailure(layer, "readback", errV41ProjectionResult))
						return
					}
				}
				values, rowErr := result.Row(row - lo)
				if rowErr != nil {
					cause = s.v41CloseBF16PublicationFailure(b.backend, layer, leaf+" normalization", "checked publication", rowErr)
					return
				}
				if len(values) != width || !v41BF16Widened(values) {
					cause = s.v41CloseBF16PublicationFailure(b.backend, layer, leaf+" normalization", "checked publication readback", errV41BF16BoundaryResult)
					return
				}
				copy(raw, values)
				trace(row, raw)
			}
			lo = hi
		}
	}
	func() { defer func() { primary = recover() }(); body(); complete = true }()
	if !complete {
		panic(primary)
	}
	return handled, cause
}
