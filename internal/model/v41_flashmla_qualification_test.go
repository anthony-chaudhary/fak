package model

import (
	"bytes"
	"errors"
	"math"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/v41kv"
)

func v41FlashQualificationFixture() v41SharedAttentionRequest {
	rows := [][]float32{make([]float32, 512), make([]float32, 512)}
	for i := range rows[0] {
		rows[0][i], rows[1][i] = 1, 2
	}
	q := make([]float32, 32768)
	q[0] = 1.00390625 // BF16 midpoint, even result 1.
	return v41SharedAttentionRequest{mode: compute.V41SharedAttentionPlain, q: q, sink: make([]float32, 64),
		rowKV: &v41SharedAttentionRows{window: rows}, idx: []int32{1, -1, 1, 0},
		plain: V41SparseAttentionSinkOptions{B: 1, M: 1, Heads: 64, HeadDim: 512, N: 2, TopK: 4, Softmax: 0.125}}
}

// These are packing/ownership witnesses, not model numerical or GPU witnesses.
// fak-test:runtime fast est=20ms lane=default
func TestV41FlashQualificationCompactSnapshot(t *testing.T) {
	r := v41FlashQualificationFixture()
	x, err := v41PackFlashMLAQualification(r)
	if err != nil {
		t.Fatal(err)
	}
	if x.Blocks != 2 || x.PageSize != 1 || x.Length != 4 || len(x.Indices) != 64 || x.Query[0] != 0x3f80 {
		t.Fatal("geometry/query")
	}
	wantIDs := []int32{0, -1, 0, 1}
	for i, id := range x.Indices {
		want := int32(-1)
		if i < len(wantIDs) {
			want = wantIDs[i]
		}
		if id != want {
			t.Fatalf("slot %d=%d want %d", i, id, want)
		}
	}
	layout, _ := v41kv.V41KVCacheLayoutFor(v41kv.V41KVFP8)
	a, _ := layout.V41KVQuantizeRow(r.rowKV.row(1)[:448], r.rowKV.row(1)[448:])
	b, _ := layout.V41KVQuantizeRow(r.rowKV.row(0)[:448], r.rowKV.row(0)[448:])
	if !bytes.Equal(x.KV, append(a, b...)) {
		t.Fatal("codec bytes/order changed")
	}
	before := append([]byte(nil), x.KV...)
	r.rowKV.row(0)[0], r.q[0], r.sink[0], r.idx[0] = 99, 99, 99, 0
	if !bytes.Equal(before, x.KV) || x.Query[0] != 0x3f80 || x.Sink[0] != 0 || x.Indices[0] != 0 {
		t.Fatal("borrowed input escaped snapshot")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41FlashQualificationAllInvalidAndPadding(t *testing.T) {
	for _, slots := range []int{0, 1, 63, 64, 65} {
		r := v41FlashQualificationFixture()
		r.idx = make([]int32, slots)
		r.plain.TopK = slots
		for i := range r.idx {
			r.idx[i] = -1
		}
		x, err := v41PackFlashMLAQualification(r)
		if err != nil {
			t.Fatal(err)
		}
		if x.Blocks != 1 || len(x.KV) != 528 || len(x.Query) != 32768 || len(x.Sink) != 64 || x.Length != int32(slots) || len(x.Indices) != max(64, (slots+63)/64*64) {
			t.Fatal("empty request lost real operands")
		}
		for _, id := range x.Indices {
			if id != -1 {
				t.Fatal("padding selected dummy row")
			}
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41FlashQualificationRejectsMaskedMalformedAndBF16Overflow(t *testing.T) {
	for _, kind := range []string{"masked-row", "masked-id", "query-overflow", "budget"} {
		r := v41FlashQualificationFixture()
		r.plain.TopKLength = []int32{0}
		switch kind {
		case "masked-row":
			r.rowKV.row(0)[0] = float32(math.NaN())
		case "masked-id":
			r.idx[3] = 2
		case "query-overflow":
			r.q[0] = math.MaxFloat32
		case "budget":
			r.plain.N = 1 << 20
			r.plain.TopK = 1 << 20
		}
		if _, err := v41PackFlashMLAQualification(r); err == nil {
			t.Fatalf("accepted %s", kind)
		}
	}
}

// Only qualification refusal is exercised; embedded backend operations must
// never be used. Name can hold failure attribution to witness overlap safely.
type v41FlashQualificationIdentityBackend struct {
	compute.Backend
	entered, release chan struct{}
	once             sync.Once
}

func (b *v41FlashQualificationIdentityBackend) Name() string {
	if b.entered != nil {
		b.once.Do(func() { close(b.entered); <-b.release })
	}
	return "qualification-identity-control"
}

// fak-test:runtime fast est=1ms lane=default
func TestV41FlashQualificationStaleBindingDoesNotRetireReplacement(t *testing.T) {
	s := &Session{M: &Model{}, Backend: &v41FlashQualificationIdentityBackend{}}
	s.v41Forward = &v41ForwardState{callbackOwner: s}
	s.backendExecutionScope = &backendExecutionScope{owner: &sync.Mutex{}, sessions: []*Session{s}, live: true}
	old := s.v41FlashMLAQualificationBinding()
	s.v41Forward.sharedAttentionBinding = old
	replacement := s.v41FlashMLAQualificationBinding()
	s.v41Forward.sharedAttentionBinding = replacement
	if out, err := old.attentionFunc()(0, v41SharedAttentionRequest{}); err == nil || out != nil {
		t.Fatal("stale selected binding accepted")
	}
	if s.halClosed || s.halFailure != nil || s.v41Forward.sharedAttentionBinding != replacement {
		t.Fatal("stale callback retired replacement")
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestV41FlashQualificationUnavailableAndOverlapAreTerminal(t *testing.T) {
	be := &v41FlashQualificationIdentityBackend{entered: make(chan struct{}), release: make(chan struct{})}
	s := &Session{M: &Model{}, Backend: be}
	s.v41Forward = &v41ForwardState{callbackOwner: s}
	s.backendExecutionScope = &backendExecutionScope{owner: &sync.Mutex{}, sessions: []*Session{s}, live: true}
	binding := s.v41FlashMLAQualificationBinding()
	s.v41Forward.sharedAttentionBinding = binding
	done := make(chan error, 1)
	released := false
	defer func() {
		if !released {
			close(be.release)
		}
	}()
	go func() { _, err := binding.attentionFunc()(0, v41FlashQualificationFixture()); done <- err }()
	select {
	case <-be.entered:
	case err := <-done:
		t.Fatalf("did not reach selected failure attribution: %v", err)
	}
	if out, err := binding.attentionFunc()(0, v41SharedAttentionRequest{}); err == nil || out != nil {
		t.Fatal("overlap accepted")
	}
	if s.halClosed || s.halFailure != nil {
		t.Fatal("overlap retired active owner before completion")
	}
	close(be.release)
	released = true
	if err := <-done; !errors.Is(err, compute.ErrFlashMLADecodeQualificationUnavailable) {
		t.Fatalf("lost selected refusal: %v", err)
	}
	if !s.halClosed || !errors.Is(s.halFailure, compute.ErrFlashMLADecodeQualificationUnavailable) {
		t.Fatal("selected owner not retired")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41FlashQualificationTypedNilBackendRefused(t *testing.T) {
	var be *v41FlashQualificationIdentityBackend
	s := &Session{M: &Model{}, Backend: be}
	if out, err := s.v41FlashMLAQualificationBinding().attentionFunc()(0, v41SharedAttentionRequest{}); !errors.Is(err, compute.ErrFlashMLADecodeQualificationUnavailable) || out != nil {
		t.Fatal("typed nil backend accepted")
	}
}
