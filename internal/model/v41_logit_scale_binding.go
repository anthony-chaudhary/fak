package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// v41LogitScaleBinding is immutable after construction. State copies share the
// callback/provenance value, never mutable callback slots or device tensors.
// A replacement creates an ordinary-only binding even when it wraps the same
// function; function-pointer identity cannot grant native composition authority.
type v41LogitScaleBinding struct {
	ordinary  v41LogitScaleFunc
	owner     *Session
	model     *Model
	backend   compute.Backend
	operation compute.V41LogitScaleBackend
}

func (st *v41ForwardState) logitScale() v41LogitScaleFunc {
	if st == nil || st.logitScaleBinding == nil {
		return nil
	}
	return st.logitScaleBinding.ordinary
}

func (st *v41ForwardState) setLogitScale(scale v41LogitScaleFunc) {
	st.logitScaleBinding = nil
	if scale != nil {
		st.logitScaleBinding = &v41LogitScaleBinding{ordinary: scale}
	}
}

func (st *v41ForwardState) bindCanonicalLogitScale(s *Session) {
	scale := s.v41LogitScaleFunc()
	st.setLogitScale(scale)
	if scale != nil {
		operation, _ := s.Backend.(compute.V41LogitScaleBackend)
		st.logitScaleBinding = &v41LogitScaleBinding{ordinary: scale, owner: s, model: s.M, backend: s.Backend, operation: operation}
	}
}
