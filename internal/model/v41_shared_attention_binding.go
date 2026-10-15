package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// An immutable binding keeps an ordinary callback and its canonical provenance together.
// Reconstructing or wrapping the callback creates an ordinary-only binding;
// function identity never transfers canonical authority. No validation bypass
// is available through this binding or the ordinary callback.
type v41SharedAttentionBinding struct {
	ordinary v41SharedAttentionFunc
	native   v41BoundSharedAttention
}

type v41BoundSharedAttention struct {
	core    *v41SharedAttentionCore
	owner   *Session
	model   *Model
	backend compute.Backend
}

func newV41SharedAttentionBinding(attend v41SharedAttentionFunc) *v41SharedAttentionBinding {
	if attend == nil {
		return nil
	}
	return &v41SharedAttentionBinding{ordinary: attend}
}

func (b *v41SharedAttentionBinding) attentionFunc() v41SharedAttentionFunc {
	if b == nil {
		return nil
	}
	return b.ordinary
}

func (st *v41ForwardState) sharedAttentionFunc() v41SharedAttentionFunc {
	if st == nil {
		return nil
	}
	return st.sharedAttentionBinding.attentionFunc()
}

func (scratch *v41ProjScratch) sharedAttentionFunc() v41SharedAttentionFunc {
	if scratch == nil {
		return nil
	}
	return scratch.sharedAttentionBinding.attentionFunc()
}

// Replacing a callback always revokes canonical provenance, even when the
// replacement is the same ordinary function returned by sharedAttentionFunc.
func (st *v41ForwardState) setSharedAttention(attend v41SharedAttentionFunc) {
	st.sharedAttentionBinding = newV41SharedAttentionBinding(attend)
}

// Canonical binding and trusted copies preserve the immutable binding identity.
// Invocation still uses the ordinary callback and its full validation path.
func (st *v41ForwardState) bindCanonicalSharedAttention(binding *v41SharedAttentionBinding) {
	st.sharedAttentionBinding = binding
}

func (st *v41ForwardState) copySharedAttentionFrom(from *v41ForwardState) {
	if from == nil {
		st.sharedAttentionBinding = nil
		return
	}
	st.sharedAttentionBinding = from.sharedAttentionBinding
}

func (scratch *v41ProjScratch) copySharedAttentionFrom(from *v41ForwardState) {
	if from == nil {
		scratch.sharedAttentionBinding = nil
		return
	}
	scratch.sharedAttentionBinding = from.sharedAttentionBinding
}

func (scratch *v41ProjScratch) clearSharedAttention() {
	scratch.sharedAttentionBinding = nil
}
