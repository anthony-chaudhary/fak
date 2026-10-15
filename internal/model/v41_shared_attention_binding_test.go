package model

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This is a CPU routing and strict-validation witness, not device qualification.
// fak-test:runtime fast est=100ms lane=default
func TestV41SharedAttentionBinding(t *testing.T) {
	fixture := func(t *testing.T) (*Session, *v41SharedAttentionTestBackend, *v41SharedAttentionBinding) {
		t.Helper()
		b := newV41SharedAttentionTestBackend()
		s := &Session{M: &Model{}, Backend: b}
		t.Cleanup(s.Close)
		binding := s.v41BindSharedAttention()
		if binding == nil || binding.attentionFunc() == nil || binding.native.core == nil || binding.native.core.run == nil ||
			binding.native.owner != s || binding.native.model != s.M || binding.native.backend != b {
			t.Fatal("canonical factory lost its callback or provenance")
		}
		return s, b, binding
	}

	t.Run("ordinary-bindings-and-transition-adapters", func(t *testing.T) {
		_, b, binding := fixture(t)
		if newV41SharedAttentionBinding(nil) != nil || (*v41SharedAttentionBinding)(nil).attentionFunc() != nil ||
			(*v41ForwardState)(nil).sharedAttentionFunc() != nil || (*v41ProjScratch)(nil).sharedAttentionFunc() != nil ||
			(*Session)(nil).v41BindSharedAttention() != nil {
			t.Fatal("nil binding changed host selection")
		}
		ordinary := binding.attentionFunc()
		wrappedCalls := 0
		wrapped := func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
			wrappedCalls++
			return ordinary(layer, request)
		}
		for _, callback := range []v41SharedAttentionFunc{ordinary, wrapped} {
			rebound := newV41SharedAttentionBinding(callback)
			if rebound == nil || rebound.attentionFunc() == nil || rebound.native != (v41BoundSharedAttention{}) {
				t.Fatal("ordinary reconstruction inherited canonical provenance")
			}
		}
		st, copied, scratch := &v41ForwardState{}, &v41ForwardState{}, &v41ProjScratch{}
		st.bindCanonicalSharedAttention(binding)
		copied.copySharedAttentionFrom(st)
		scratch.copySharedAttentionFrom(st)
		if st.sharedAttentionBinding != binding || copied.sharedAttentionBinding != binding || scratch.sharedAttentionBinding != binding {
			t.Fatal("trusted bind or copy changed canonical binding identity")
		}
		st.setSharedAttention(wrapped)
		if st.sharedAttentionBinding == nil || st.sharedAttentionBinding == binding || st.sharedAttentionBinding.native != (v41BoundSharedAttention{}) ||
			copied.sharedAttentionBinding != binding || scratch.sharedAttentionBinding != binding {
			t.Fatal("wrapper replacement retained authority or changed a trusted copy")
		}
		request := v41SharedAttentionRequest{mode: compute.V41SharedAttentionPlain,
			q: []float32{0, 0}, kv: []float32{2, 4}, idx: []int32{0},
			plain: V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 1, TopK: 1, Softmax: 1}}
		for _, callback := range []v41SharedAttentionFunc{copied.sharedAttentionFunc(), scratch.sharedAttentionFunc(), st.sharedAttentionFunc()} {
			out, err := callback(0, request)
			if err != nil || !reflect.DeepEqual(out, []float32{2, 4}) {
				t.Fatalf("adapter changed strict callback output: %v, %v", out, err)
			}
		}
		if wrappedCalls != 1 || len(b.calls) != 3 || len(b.allocations) != 0 {
			t.Fatal("copy/replacement changed dispatch or retained transient storage")
		}
		st.bindCanonicalSharedAttention(binding)
		st.setSharedAttention(st.sharedAttentionFunc())
		if st.sharedAttentionBinding == nil || st.sharedAttentionBinding == binding || st.sharedAttentionBinding.native != (v41BoundSharedAttention{}) {
			t.Fatal("same-callback replacement retained canonical provenance")
		}
		copied.copySharedAttentionFrom(st)
		scratch.copySharedAttentionFrom(st)
		if copied.sharedAttentionBinding != st.sharedAttentionBinding || scratch.sharedAttentionBinding != st.sharedAttentionBinding {
			t.Fatal("trusted copy changed ordinary-only binding identity")
		}
		scratch.copySharedAttentionFrom(nil)
		if scratch.sharedAttentionBinding != nil {
			t.Fatal("nil-source scratch copy retained a binding")
		}
		scratch.copySharedAttentionFrom(st)
		st.setSharedAttention(nil)
		copied.copySharedAttentionFrom(nil)
		scratch.clearSharedAttention()
		if st.sharedAttentionBinding != nil || copied.sharedAttentionBinding != nil || scratch.sharedAttentionBinding != nil ||
			st.sharedAttentionFunc() != nil || copied.sharedAttentionFunc() != nil || scratch.sharedAttentionFunc() != nil {
			t.Fatal("reset retained a callback")
		}
		st.bindCanonicalSharedAttention(binding)
		st.bindCanonicalSharedAttention(nil)
		if st.sharedAttentionBinding != nil {
			t.Fatal("nil canonical bind retained a binding")
		}
	})

	t.Run("dormant-core-remains-strict", func(t *testing.T) {
		for _, entry := range []string{"ordinary", "private-core"} {
			t.Run(entry, func(t *testing.T) {
				s, b, binding := fixture(t)
				attend := binding.attentionFunc()
				if entry == "private-core" {
					attend = binding.native.core.run
				}
				request := v41SharedAttentionRequest{mode: compute.V41SharedAttentionPlain,
					q: []float32{0, 0}, idx: []int32{0, -1},
					rowKV: &v41SharedAttentionRows{window: [][]float32{{2, 4}}, compressed: [][]float32{{6, 8}}},
					plain: V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 1, HeadDim: 2, N: 2, TopK: 2, Softmax: 1}}
				out, err := v41SelectedSharedAttention(0, request, attend)
				if err != nil || !reflect.DeepEqual(out, []float32{2, 4}) || len(b.calls) != 1 || b.uploads != 3 || len(b.allocations) != 0 {
					t.Fatalf("strict entry changed output or lifecycle: out=%v err=%v calls=%d uploads=%d", out, err, len(b.calls), b.uploads)
				}
				request.rowKV.compressed[0][1] = float32(math.NaN())
				out, err = v41SelectedSharedAttention(0, request, attend)
				var selected *V41SharedAttentionOperationError
				var closed *BackendForwardOperationError
				if out != nil || !errors.As(err, &selected) || !errors.As(err, &closed) || !errors.Is(err, ErrV41ForwardStage) ||
					!s.halClosed || s.halFailure == nil || len(b.calls) != 1 || b.uploads != 3 || len(b.allocations) != 0 {
					t.Fatalf("strict entry hid a masked nonfinite row: out=%v err=%v calls=%d uploads=%d closed=%v", out, err, len(b.calls), b.uploads, s.halClosed)
				}
			})
		}
	})

	t.Run("wrapper-nonfinite-future-row-is-revalidated", func(t *testing.T) {
		s, b, binding := fixture(t)
		prior := v41ResidentKVDisabled
		v41ResidentKVDisabled = false
		defer func() { v41ResidentKVDisabled = prior }()
		st := &v41ForwardState{}
		st.bindCanonicalSharedAttention(binding)
		ordinary := st.sharedAttentionFunc()
		wrappedCalls := 0
		st.setSharedAttention(func(layer int, request v41SharedAttentionRequest) ([]float32, error) {
			wrappedCalls++
			if request.rowKV == nil || !reflect.DeepEqual(request.idx, []int32{0, -1}) {
				t.Fatal("expected an original row stream with a causally masked group")
			}
			request.rowKV.compressed[0][1] = float32(math.NaN())
			return ordinary(layer, request)
		})
		out, err := v41CombinedAttentionResident(0, 0, 2, 1, 2, []float32{0, 0}, nil,
			[][]float32{{2, 4}}, [][]float32{{6, 8}}, []int32{0}, 1, &v41ResidentKVView{}, st.sharedAttentionFunc())
		var selected *V41SharedAttentionOperationError
		var closed *BackendForwardOperationError
		if out != nil || !errors.As(err, &selected) || !errors.As(err, &closed) || !errors.Is(err, ErrV41ForwardStage) ||
			wrappedCalls != 1 || !s.halClosed || s.halFailure == nil || len(b.calls) != 0 || b.uploads != 0 || len(b.allocations) != 0 {
			t.Fatalf("wrapper bypassed original-row validation or retirement: out=%v err=%v calls=%d uploads=%d closed=%v", out, err, len(b.calls), b.uploads, s.halClosed)
		}
	})
}
