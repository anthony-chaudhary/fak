package model

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41RingTestBackend adds a host transcription of the device compressor ring
// to the dense-projection recorder. The emitted row comes from a fresh host
// V41CompressorPool over the ring rows, so the ring and host-staged sessions
// share one arithmetic; numerical device parity is
// TestVulkanV41CompressorRingMatchesPool in internal/compute.
type v41RingTestBackend struct {
	*v41DenseTestBackend
	ring        bool
	rows        map[compute.Buffer][]float32
	outputs     map[compute.Buffer][]float32
	reads       int
	stores      int
	emits       int
	storeFault  error
	forceStatus float32
}

func newV41RingTestBackend(ring bool) *v41RingTestBackend {
	return &v41RingTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), ring: ring, rows: map[compute.Buffer][]float32{}, outputs: map[compute.Buffer][]float32{}}
}

func (b *v41RingTestBackend) SupportsV41CompressorRing() bool { return b.ring }

func (b *v41RingTestBackend) V41CompressorRingAlloc(ratio, width int) (compute.Tensor, compute.Tensor, error) {
	if err := compute.ValidateV41CompressorRingAlloc(ratio, width); err != nil {
		return compute.Tensor{}, compute.Tensor{}, err
	}
	kvRows, scoreRows := make([]float32, ratio*width), make([]float32, ratio*width)
	kv, scores := compute.NewF32(b, []int{ratio, width}, kvRows), compute.NewF32(b, []int{ratio, width}, scoreRows)
	b.rows[kv.Buf()], b.rows[scores.Buf()] = kvRows, scoreRows
	return kv, scores, nil
}

func (b *v41RingTestBackend) V41CompressorRingStore(kv, scores, kvRow, scoreRow compute.Tensor, slot, ratio, width int, emit bool) (compute.Tensor, error) {
	if err := compute.ValidateV41CompressorRingStore(kv, scores, kvRow, scoreRow, slot, ratio, width); err != nil {
		return compute.Tensor{}, err
	}
	kvRing, scoreRing := b.rows[kv.Buf()], b.rows[scores.Buf()]
	if kvRing == nil || scoreRing == nil || kv.Backend() != b || scores.Backend() != b {
		return compute.Tensor{}, compute.ErrV41CompressorRingGeometry
	}
	if b.storeFault != nil {
		return compute.Tensor{}, b.storeFault
	}
	copy(kvRing[slot*width:(slot+1)*width], b.v41DenseTestBackend.Read(kvRow))
	copy(scoreRing[slot*width:(slot+1)*width], b.v41DenseTestBackend.Read(scoreRow))
	b.stores++
	if !emit {
		return compute.Tensor{}, nil
	}
	values := make([]float32, 2*width)
	for d := 0; d < width; d++ {
		for t := 0; t < ratio; t++ {
			if !finite32(kvRing[t*width+d]) {
				values[width+d] = float32(int(values[width+d]) | compute.V41CompressorRingKVNonFinite)
			}
			if !finite32(scoreRing[t*width+d]) {
				values[width+d] = float32(int(values[width+d]) | compute.V41CompressorRingScoreNonFinite)
			}
		}
	}
	values[2*width-1] += b.forceStatus
	pool, err := NewV41CompressorPool(ratio, width)
	if err != nil {
		return compute.Tensor{}, err
	}
	for t := 0; t < ratio; t++ {
		out, emitted, err := pool.Push(t, kvRing[t*width:(t+1)*width], scoreRing[t*width:(t+1)*width])
		if err != nil {
			break
		}
		if emitted {
			copy(values, out)
		}
	}
	out := compute.NewF32(b, []int{2, width}, values)
	b.outputs[out.Buf()] = values
	b.emits++
	return out, nil
}

func (b *v41RingTestBackend) Read(x compute.Tensor) []float32 {
	b.reads++
	if values, ok := b.outputs[x.Buf()]; ok {
		return append([]float32(nil), values...)
	}
	return b.v41DenseTestBackend.Read(x)
}

func (b *v41RingTestBackend) Free(x compute.Tensor) {
	if b.outputs[x.Buf()] != nil || b.rows[x.Buf()] != nil {
		delete(b.outputs, x.Buf())
		delete(b.rows, x.Buf())
		return
	}
	b.v41DenseTestBackend.Free(x)
}

func v41RingBitsEqualRows(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: logit %d bits %#08x, want %#08x", what, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func v41RingSessions(t *testing.T) (*Model, *Session, *v41RingTestBackend, *Session, *v41RingTestBackend) {
	t.Helper()
	m, _ := v41IncrementalEngramFixture(t, true)
	control, _ := v41IncrementalEngramFixture(t, true)
	ringBackend, hostBackend := newV41RingTestBackend(true), newV41RingTestBackend(false)
	s, h := v41DenseTestSession(t, m, ringBackend), v41DenseTestSession(t, control, hostBackend)
	if s.v41State().gatherOps.compressorRing == nil || h.v41State().gatherOps.compressorRing != nil {
		t.Fatal("ring binding does not follow the backend capability")
	}
	return m, s, ringBackend, h, hostBackend
}

func v41RingStats(s *Session) v41CompressorRingStats {
	return s.v41State().gatherOps.compressorRing.stats
}

// fak-test:runtime fast est=1ms lane=default
func TestV41CompressorRingSlotHoldsExactBits(t *testing.T) {
	slot := v41CompressorRingSlot{valid: true, pos: 6, input: []float32{0, 1.5}}
	if !slot.holds(6, []float32{0, 1.5}) {
		t.Fatal("identical row rejected")
	}
	for name, ok := range map[string]bool{
		"negative zero": slot.holds(6, []float32{float32(math.Copysign(0, -1)), 1.5}),
		"position":      slot.holds(7, []float32{0, 1.5}),
		"width":         slot.holds(6, []float32{0}),
		"invalid":       v41CompressorRingSlot{pos: 6, input: []float32{0, 1.5}}.holds(6, []float32{0, 1.5}),
	} {
		if ok {
			t.Errorf("%s mismatch accepted", name)
		}
	}
}

// fak-test:runtime medium est=10s lane=default
// Every decode token stores one ring row per compressed source layer with no
// readback, and each closed group costs one readback instead of 2*ratio
// projection readbacks. Logits stay bit-identical to the host-staged session,
// both from an even prefix (empty partial group) and an odd one, where the
// prefill-seeded row is reseeded once per layer.
func TestV41CompressorRingDecodeMatchesHostStaged(t *testing.T) {
	for _, prefixLen := range []int{4, 3} {
		m, s, rb, h, hb := v41RingSessions(t)
		layers, ratio := m.Cfg.NumLayers, 2
		prefix, next := v41IncrementalPrefix(m.Cfg, prefixLen)
		v41RingBitsEqualRows(t, "prefill", s.Prefill(prefix), h.Prefill(prefix))
		if rb.stores != 0 || rb.emits != 0 {
			t.Fatalf("prefill used the decode ring: stores=%d emits=%d", rb.stores, rb.emits)
		}
		ringReads, hostReads := rb.reads, hb.reads
		const steps = 5
		for i := 0; i < steps; i++ {
			v41RingBitsEqualRows(t, "step", s.Step(next), h.Step(next))
			next = (next*7 + 3) % m.Cfg.VocabSize
		}
		groups := (prefixLen+steps)/ratio - prefixLen/ratio
		reseeds := prefixLen % ratio
		got := v41RingStats(s)
		want := v41CompressorRingStats{Stores: (steps + reseeds) * layers, Emits: groups * layers, Reseeds: reseeds * layers}
		if got != want || rb.stores != want.Stores || rb.emits != want.Emits {
			t.Fatalf("prefix %d ring stats %+v (backend stores=%d emits=%d), want %+v", prefixLen, got, rb.stores, rb.emits, want)
		}
		// The host-staged control reads 2*ratio projection rows per closed group
		// (its pool is on the host here); the ring reads one emitted row.
		saved := (hb.reads - hostReads) - (rb.reads - ringReads)
		if wantSaved := (2*ratio - 1) * groups * layers; saved != wantSaved {
			t.Fatalf("prefix %d readbacks saved %d, want %d", prefixLen, saved, wantSaved)
		}
	}
}

// fak-test:runtime medium est=10s lane=default
// A restore into a fresh session (empty ring) of a snapshot taken mid-group
// must reseed the retained row from the host input before the group closes,
// after the source session's ring has already moved past that group.
func TestV41CompressorRingReseedsAfterRestore(t *testing.T) {
	m, s, rb, h, _ := v41RingSessions(t)
	layers := m.Cfg.NumLayers
	prefix, a := v41IncrementalPrefix(m.Cfg, 4)
	s.Prefill(prefix)
	s.Step(a)
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	defer snap.Close()
	b, c, d := (a+1)%m.Cfg.VocabSize, (a+2)%m.Cfg.VocabSize, (a+3)%m.Cfg.VocabSize
	s.Step(b)
	s.Step(c)
	h.Prefill(prefix)
	h.Step(a)
	want := h.Step(d)

	target := v41DenseTestSession(t, m, rb)
	if err := snap.Restore(target); err != nil {
		t.Fatal(err)
	}
	v41RingBitsEqualRows(t, "restored session", target.Step(d), want)
	if got := v41RingStats(target); got != (v41CompressorRingStats{Stores: 2 * layers, Emits: layers, Reseeds: layers}) {
		t.Fatalf("restore ring stats %+v", got)
	}
}

// fak-test:runtime medium est=10s lane=default
// A step that stores ring rows and then fails in a later layer rolls back; the
// production fallback commits the token and decode continues on the ring with
// logits bit-identical to a host-staged session given the same failure.
func TestV41CompressorRingSurvivesFailedStep(t *testing.T) {
	m, s, rb, h, _ := v41RingSessions(t)
	last := m.Cfg.NumLayers - 1
	prefix, next := v41IncrementalPrefix(m.Cfg, 4)
	v41RingBitsEqualRows(t, "prefill", s.Prefill(prefix), h.Prefill(prefix))
	for _, session := range []*Session{s, h} {
		session.v41Forward.layerState(last).nextWindowPos = len(session.v41Forward.history) + 5
	}
	stored := rb.stores
	v41RingBitsEqualRows(t, "failed step fallback", s.Step(next), h.Step(next))
	if rb.stores == stored {
		t.Fatal("the failed step never reached the ring")
	}
	for i := 0; i < 4; i++ {
		next = (next*7 + 3) % m.Cfg.VocabSize
		v41RingBitsEqualRows(t, "step after fallback", s.Step(next), h.Step(next))
	}
	if got := v41RingStats(s); got.Emits < 2*m.Cfg.NumLayers {
		t.Fatalf("decode left the ring after the fallback: %+v", got)
	}
}

// fak-test:runtime medium est=10s lane=default
// The ring is selected only for the session's canonical native projection and
// stays one switch away; both declines keep the host-staged arithmetic.
func TestV41CompressorRingDeclines(t *testing.T) {
	t.Run("env off", func(t *testing.T) {
		t.Setenv(v41CompressorRingEnv, "0")
		m, _ := v41IncrementalEngramFixture(t, true)
		s := v41DenseTestSession(t, m, newV41RingTestBackend(true))
		if s.v41State().gatherOps.compressorRing != nil {
			t.Fatal("FAK_V41_COMPRESSOR_RING=0 still bound the ring")
		}
	})
	t.Run("replaced projection", func(t *testing.T) {
		m, s, rb, h, _ := v41RingSessions(t)
		prefix, next := v41IncrementalPrefix(m.Cfg, 4)
		st := s.v41State()
		ordinary := st.denseProjection()
		calls := 0
		st.setDenseProjection(func(layer int, leaf string, panel []float32, out, in, rows int) ([]float32, v41DenseProjectionOutcome, error) {
			calls++
			return ordinary(layer, leaf, panel, out, in, rows)
		})
		v41RingBitsEqualRows(t, "prefill", s.Prefill(prefix), h.Prefill(prefix))
		v41RingBitsEqualRows(t, "step", s.Step(next), h.Step(next))
		v41RingBitsEqualRows(t, "step", s.Step(next), h.Step(next))
		if rb.stores != 0 || calls == 0 {
			t.Fatalf("replaced projection: ring stores %d, replacement calls %d", rb.stores, calls)
		}
	})
}

// fak-test:runtime medium est=10s lane=default
// A selected ring failure, or a non-finite group, closes the session with the
// ring's path and the original cause; nothing is replayed on the host.
func TestV41CompressorRingFailureClosesSession(t *testing.T) {
	fault := &compute.BackendError{Backend: "test-device", Class: compute.VulkanClassExecutionFailed, Err: compute.ErrVulkanExecutionFailed}
	for _, tc := range []struct {
		name  string
		arm   func(*v41RingTestBackend)
		cause error
	}{
		{"store fault", func(b *v41RingTestBackend) { b.storeFault = fault }, fault},
		{"non-finite status", func(b *v41RingTestBackend) { b.forceStatus = compute.V41CompressorRingScoreNonFinite }, compute.ErrV41CompressorRingNonFinite},
		{"malformed status", func(b *v41RingTestBackend) { b.forceStatus = 0.5 }, compute.ErrV41CompressorRingStatus},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m, s, rb, _, _ := v41RingSessions(t)
			prefix, next := v41IncrementalPrefix(m.Cfg, 4)
			s.Prefill(prefix)
			s.Step(next)
			tc.arm(rb)
			err := recoverError(func() { s.Step(next) })
			var closed *BackendForwardOperationError
			if !errors.As(err, &closed) || closed.Path != "v41-compressor-ring" || !errors.Is(err, tc.cause) || !s.BackendSessionClosed() {
				t.Fatalf("ring failure lost identity: %v", err)
			}
			stores := rb.stores
			if retry := recoverError(func() { s.Step(next) }); retry == nil || rb.stores != stores {
				t.Fatal("closed ring session retried")
			}
		})
	}
}
