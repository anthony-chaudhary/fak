package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Ordinary callback replacement removes authority to combine device operations.
type v41KVNormBinding struct {
	ordinary    v41KVNormFunc
	owner       *Session
	model       *Model
	backend     compute.Backend
	publication *v41BF16PublicationBinding
}

func (b *v41KVNormBinding) normFunc() v41KVNormFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}
func (s *Session) v41BindKVNorm() *v41KVNormBinding {
	ordinary := s.v41KVNormFunc()
	if ordinary == nil {
		return nil
	}
	b := &v41KVNormBinding{ordinary: ordinary}
	if s.v41Forward != nil && s.v41Forward.gatherOps != nil {
		pub := s.v41Forward.gatherOps.bf16Publish
		if pub != nil && pub.core != nil && pub.owner == s && pub.model == s.M && backendHasComparableIdentity(s.Backend) && pub.backend == s.Backend {
			b.owner, b.model, b.backend, b.publication = s, s.M, s.Backend, pub
		}
	}
	return b
}

func (m *Model) v41KVNormWithBinding(layer int, input []float32, eps float32, b *v41KVNormBinding, ops *v41DeviceGatherOps) error {
	if selected, err := b.try(m, layer, input, eps, ops); selected || err != nil {
		return err
	}
	return m.v41KVNormInPlaceWithPublish(layer, input, eps, b.normFunc(), ops.bf16PublishFunc())
}

// try never mutates input before successful transport and input/raw/output
// validation. Misses retain the original scalar call, including gain staging.
func (b *v41KVNormBinding) try(m *Model, layer int, input []float32, eps float32, ops *v41DeviceGatherOps) (selected bool, cause error) {
	if b == nil || b.owner == nil || m == nil || ops == nil || len(input) != v41KVLoraRank || layer < 0 || layer >= m.Cfg.NumLayers || eps != float32(m.Cfg.RMSNormEps) || !finite32(eps) || eps <= 0 || m.Cfg.LayerNorm || m.Cfg.NormGain1p {
		return false, nil
	}
	full, err := v41ForwardGeometry(m.Cfg)
	if err != nil || !full || v41BF16Widened(input) {
		return false, nil
	}
	if _, bad := v41BF16PublishFault(input); bad {
		return false, nil
	}
	s := b.owner
	state, scope, epoch := s.v41Forward, s.backendExecutionScope, s.cacheGeometryEpoch
	valid := func() bool {
		if b.model != m || s.M != m || !backendHasComparableIdentity(s.Backend) || s.Backend != b.backend || s.halClosed || s.halFailure != nil || !s.modelWeightsHeld || state == nil || s.v41Forward != state || state.callbackOwner != s || state.kvNorm != b || state.gatherOps != ops || ops.owner != s || ops.bf16Publish != b.publication || scope == nil || !scope.live || scope.owner == nil || s.backendExecutionScope != scope || s.cacheGeometryEpoch != epoch {
			return false
		}
		for _, member := range scope.sessions {
			if member == s {
				return true
			}
		}
		return false
	}
	if !valid() {
		return false, nil
	}
	const leaf = "attn.kv_norm.weight"
	name := layerName(layer, leaf)
	meta, present := m.manifest[name]
	if !present || len(meta.Shape) != 1 || meta.Shape[0] != v41KVLoraRank {
		return false, nil
	}
	gain, present := s.halW[name]
	if !present || gain.Buf() == nil {
		return false, nil
	}
	hostGain := m.tensor(name)
	if len(hostGain) != v41KVLoraRank {
		return false, nil
	}
	for _, v := range hostGain {
		if !finite32(v) {
			return false, nil
		}
	}
	closeNorm := func(stage string, err error) error {
		if s.halFailure == nil {
			s.halFailure = &BackendForwardOperationError{Backend: b.backend.Name(), Forward: ForwardPathKind("deepseek41"), Path: "v41-kv-norm", Layer: layer, Stage: stage, Cause: err}
		}
		if !s.halClosed {
			func() { defer func() { s.halClosed = true; _ = recover() }(); s.closeBackendOwned() }()
		}
		return v41ProjectionOperationErr(layer, leaf, s.halFailure)
	}
	closePub := func(phase, stage string, err error) error {
		site := leaf + " " + phase
		closed := s.v41CloseBF16PublicationFailure(b.backend, layer, site, stage, err)
		return &V41BF16BoundaryOperationError{Layer: layer, Site: site, Cause: closed}
	}
	complete := false
	var primary any
	defer func() {
		if !complete {
			selected = true
			if cause == nil {
				cause = s.halFailure
			}
			if cause == nil {
				cause, _ = primary.(error)
			}
			if cause == nil {
				cause = fmt.Errorf("incomplete KV norm chain")
			}
			cause = closeNorm("chain completion", cause)
		}
	}()
	run := func() (selected bool, cause error) {
		image, chosen, err := compute.TryV41KVNormPublication(b.backend, gain, input, eps)
		if err != nil {
			selected = true
			cause = closeNorm("chain transport", err)
			complete = true
			return
		}
		if !chosen {
			complete = true
			return false, nil
		}
		selected = true
		// Both kernels have completed transport, even if an envelope or raw
		// numerical check now fails. These diagnostic counts describe work,
		// not two accepted logical publications.
		if v41HostBF16Witness.Load() {
			v41BF16PublishCalls.Add(2)
			v41BF16PanelCalls.Add(2)
			v41BF16PanelRows.Add(2)
		}
		if !valid() || image == nil {
			cause = closeNorm("chain binding", errV41ProjectionResult)
			complete = true
			return
		}
		if err = image.InputError(); err != nil {
			cause = closePub("projection", "checked publication", err)
			complete = true
			return
		}
		raw := image.Raw()
		if len(raw) != v41KVLoraRank {
			cause = closeNorm("readback", errV41ProjectionResult)
			complete = true
			return
		}
		for _, v := range raw {
			if !finite32(v) {
				cause = closeNorm("readback", errV41ProjectionResult)
				complete = true
				return
			}
		}
		published, err := image.Published()
		if err != nil {
			cause = closePub("normalization", "checked publication", err)
			complete = true
			return
		}
		if len(published) != v41KVLoraRank || !v41BF16Widened(published) {
			cause = closePub("normalization", "checked publication readback", errV41BF16BoundaryResult)
			complete = true
			return
		}
		if !valid() {
			cause = closeNorm("chain binding", errV41ProjectionResult)
			complete = true
			return
		}
		copy(input, published)
		complete = true
		return true, nil
	}
	interrupted := false
	func() {
		returned := false
		defer func() {
			if !returned {
				interrupted = true
				primary = recover()
			}
		}()
		selected, cause = run()
		returned = true
	}()
	if interrupted {
		selected = true
		if err, ok := primary.(error); ok {
			var prior *BackendForwardOperationError
			if errors.As(err, &prior) {
				// Keep an already selected operation's exact sticky identity.
				if s.halFailure == nil {
					s.halFailure = err
				}
				cause = err
				cause = closeNorm("chain transport", err)
				complete = true
				panic(primary)
			}
			var backendErr *compute.BackendError
			if errors.As(err, &backendErr) {
				cause = err
				cause = closeNorm("chain transport", err)
				complete = true
				return selected, cause
			}
		}
		if err, ok := compute.ConvertCUDAPanic(primary, "", ""); ok {
			if original, ok := primary.(error); ok {
				err = original
			}
			cause = err
			cause = closeNorm("chain transport", err)
			complete = true
			return selected, cause
		}
		panic(primary)
	} // raw panic(nil) resumes; Goexit never reaches this line
	return selected, cause
}
