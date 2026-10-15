package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// Ordinary callbacks and native authority travel together as one immutable
// binding. Assigning or wrapping an ordinary function never transfers authority.
type v41RoutedSumBinding struct {
	ordinary v41RoutedSumFunc
	native   *v41RoutedSumAuthority
}

type v41RoutedSumAuthority struct {
	owner   *Session
	model   *Model
	backend compute.Backend
	width   int
	inner   int
	limit   float32
}

func newV41RoutedSumBinding(sum v41RoutedSumFunc) *v41RoutedSumBinding {
	if sum == nil {
		return nil
	}
	return &v41RoutedSumBinding{ordinary: sum}
}

func (b *v41RoutedSumBinding) sumFunc() v41RoutedSumFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}

func (st *v41ForwardState) routedSumFunc() v41RoutedSumFunc {
	if st == nil {
		return nil
	}
	return st.routedSumBinding.sumFunc()
}

func (scratch *v41ProjScratch) routedSumFunc() v41RoutedSumFunc {
	if scratch == nil {
		return nil
	}
	return scratch.routedSumBinding.sumFunc()
}

func (st *v41ForwardState) setRoutedSum(sum v41RoutedSumFunc) {
	st.routedSumBinding = newV41RoutedSumBinding(sum)
}

func (scratch *v41ProjScratch) setRoutedSum(sum v41RoutedSumFunc) {
	scratch.routedSumBinding = newV41RoutedSumBinding(sum)
}

func (scratch *v41ProjScratch) copyRoutedSumFrom(st *v41ForwardState) {
	if st == nil {
		scratch.routedSumBinding = nil
		return
	}
	scratch.routedSumBinding = st.routedSumBinding
}

// Mint provenance at the factory, never by attaching it to a supplied closure.
// The ordinary callback keeps its historical dynamic-session behavior. Native
// use additionally requires the exact backend/model/geometry seen by this
// factory to remain live. The cold cohort binds its projection/deferred-status
// interfaces afresh from that same backend; no cached projection closure is
// inferred to be canonical from callbackOwner or function identity.
func (s *Session) v41CanonicalRoutedSumBinding() *v41RoutedSumBinding {
	if s == nil || s.M == nil || !backendHasComparableIdentity(s.Backend) {
		return newV41RoutedSumBinding(s.v41RoutedSumFunc())
	}
	a := &v41RoutedSumAuthority{owner: s, model: s.M, backend: s.Backend, width: s.M.Cfg.HiddenSize, inner: s.M.Cfg.MoEIntermediateSize, limit: float32(s.M.Cfg.SwigluLimit)}
	b := newV41RoutedSumBinding(s.v41RoutedSumFunc())
	if b != nil && a.current(s) {
		b.native = a
	}
	return b
}

func (a *v41RoutedSumAuthority) current(s *Session) bool {
	return a != nil && s != nil && s.M != nil && a.owner == s && s.M == a.model && s.Backend == a.backend &&
		s.M.Cfg.HiddenSize == a.width && s.M.Cfg.MoEIntermediateSize == a.inner &&
		float32(s.M.Cfg.SwigluLimit) == a.limit && !s.halClosed && s.halFailure == nil
}

func (scratch *v41ProjScratch) routedOutputOwner(s *Session, layer, width int, picks [][]routePick) *v41RoutedOutputOwner {
	if scratch == nil || scratch.routedSumBinding == nil || !scratch.routedSumBinding.native.current(s) {
		return nil
	}
	authority := scratch.routedSumBinding.native
	owner := newV41RoutedOutputOwner(s, layer, width, picks)
	// Capability probes may run backend code. They cannot transfer an old
	// factory's authority to a replacement backend, geometry or live scope.
	if owner != nil && (!authority.current(s) || !owner.live()) {
		owner.release(0, true) // metadata only: no publication has transferred
		return nil
	}
	return owner
}
