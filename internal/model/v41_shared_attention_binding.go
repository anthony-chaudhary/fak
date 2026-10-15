package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// A binding keeps an ordinary callback and its canonical provenance together.
// Reconstructing or wrapping the callback creates an ordinary-only binding;
// function identity never transfers canonical authority. No validation bypass
// is available through this binding or the ordinary callback.
type v41SharedAttentionBinding struct {
	ordinary v41SharedAttentionFunc
	native   v41BoundSharedAttention
}

type v41BoundSharedAttention struct {
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
	return st.sharedAttention
}

func (scratch *v41ProjScratch) sharedAttentionFunc() v41SharedAttentionFunc {
	if scratch == nil {
		return nil
	}
	return scratch.sharedAttention
}

func (st *v41ForwardState) setSharedAttention(attend v41SharedAttentionFunc) {
	st.sharedAttention = attend
}

// The raw-field transition deliberately discards canonical provenance. It
// cannot authorize a fast path while direct callback replacements still exist.
func (st *v41ForwardState) bindCanonicalSharedAttention(binding *v41SharedAttentionBinding) {
	st.sharedAttention = binding.attentionFunc()
}

func (st *v41ForwardState) copySharedAttentionFrom(from *v41ForwardState) {
	st.sharedAttention = from.sharedAttentionFunc()
}

func (scratch *v41ProjScratch) copySharedAttentionFrom(from *v41ForwardState) {
	scratch.sharedAttention = from.sharedAttentionFunc()
}

func (scratch *v41ProjScratch) clearSharedAttention() {
	scratch.sharedAttention = nil
}
