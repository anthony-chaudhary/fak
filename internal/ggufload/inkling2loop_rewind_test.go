package ggufload

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	_ Inkling2LoopRewinder = (*inkling2HostSession)(nil)
	_ Inkling2LoopRewinder = (*inkling2ResidentSession)(nil)
)

// ink2ReuseDriver mirrors the serve engine's prefix-reuse policy: the checkpoint
// covers the previous prompt minus its last token, and decode tokens are never
// part of it.
type ink2ReuseDriver struct {
	s     Inkling2LoopSession
	ck    []int
	hasCk bool
}

func ink2LCP(a, b []int) int {
	n := 0
	for n < len(a) && n < len(b) && a[n] == b[n] {
		n++
	}
	return n
}

// turn runs one request: prefill prompt (reusing the checkpoint when it is a
// strict prefix) then greedy-decode gen tokens. It returns the tokens and how
// many prompt tokens were reused.
func (d *ink2ReuseDriver) turn(prompt []int, gen int) ([]int, int, error) {
	rw := d.s.(Inkling2LoopRewinder)
	n := len(prompt)
	reused := 0
	if d.hasCk && ink2LCP(d.ck, prompt) == len(d.ck) && len(d.ck) < n {
		switch err := rw.Restore(); {
		case err == nil:
			reused = len(d.ck)
		case !errors.Is(err, ErrInkling2NoCheckpoint):
			return nil, 0, err
		}
	}
	if reused == 0 {
		if err := rw.Reset(); err != nil {
			return nil, 0, err
		}
	}
	d.hasCk = false
	if n-1 > reused {
		if _, err := d.s.Prefill(context.Background(), prompt[reused:n-1]); err != nil {
			return nil, 0, err
		}
	}
	// the last prompt token and gen-1 decode steps run past the checkpoint
	if err := rw.Checkpoint(gen); err != nil {
		return nil, 0, err
	}
	d.ck, d.hasCk = append(d.ck[:0], prompt[:n-1]...), true
	next, err := d.s.Prefill(context.Background(), prompt[n-1:])
	if err != nil {
		return nil, 0, err
	}
	out, err := ink2DecodeFrom(d.s, next, gen)
	return out, reused, err
}

func ink2DecodeFrom(s Inkling2LoopSession, next, gen int) ([]int, error) {
	out := []int{next}
	for len(out) < gen {
		var err error
		if next, err = s.Step(next); err != nil {
			return nil, err
		}
		out = append(out, next)
	}
	return out, nil
}

func ink2FreshTurn(m *Inkling2LoopModel, prompt []int, gen int) ([]int, error) {
	s, err := NewInkling2LoopSession(m, nil)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	next, err := s.Prefill(context.Background(), prompt)
	if err != nil {
		return nil, err
	}
	return ink2DecodeFrom(s, next, gen)
}

func ink2Cat(parts ...[]int) []int {
	var out []int
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// ink2Rerender stands in for a chat template that re-tokenizes the assistant turn
// differently from the tokens that were generated.
func ink2Rerender(gen []int) []int {
	out := make([]int, len(gen))
	for i, t := range gen {
		out[i] = (t + 3) % 10
	}
	return out
}

func TestInkling2HostPrefixReuseMatchesFreshSession(t *testing.T) {
	m := ink2SessTiny()
	const gen = 6 // longer than the tiny model's 3-token sliding window
	for _, rerender := range []bool{false, true} {
		asm := func(g []int) []int {
			if rerender {
				return ink2Rerender(g)
			}
			return g
		}
		s, err := NewInkling2LoopSession(m, nil)
		if err != nil {
			t.Fatal(err)
		}
		d := &ink2ReuseDriver{s: s}
		var prev []int
		var prevGen []int
		type turn struct {
			name   string
			prompt func() []int
			reused func(p []int) int
		}
		turns := []turn{
			{"first", func() []int { return []int{1, 4, 2, 7, 3} }, func([]int) int { return 0 }},
			{"extend", func() []int { return ink2Cat(prev, asm(prevGen), []int{5, 8}) }, func([]int) int { return len(prev) - 1 }},
			{"extend-again", func() []int { return ink2Cat(prev, asm(prevGen), []int{6}) }, func([]int) int { return len(prev) - 1 }},
			{"repeat", func() []int { return append([]int(nil), prev...) }, func([]int) int { return len(prev) - 1 }},
			{"diverge", func() []int { return ink2Cat(prev[:2], []int{9, 0, 9, 2, 5}) }, func([]int) int { return 0 }},
			{"extend-after-diverge", func() []int { return ink2Cat(prev, asm(prevGen), []int{3, 3, 1}) }, func([]int) int { return len(prev) - 1 }},
			{"shrink", func() []int { return append([]int(nil), prev[:4]...) }, func([]int) int { return 0 }},
			{"single-token", func() []int { return []int{7} }, func([]int) int { return 0 }},
			{"after-single", func() []int { return []int{7, 2} }, func([]int) int { return 0 }},
		}
		for _, tc := range turns {
			p := tc.prompt()
			wantReused := tc.reused(p)
			got, reused, err := d.turn(p, gen)
			if err != nil {
				t.Fatalf("rerender=%v %s: %v", rerender, tc.name, err)
			}
			want, err := ink2FreshTurn(m, p, gen)
			if err != nil {
				t.Fatal(err)
			}
			ref, err := Inkling2LoopGreedy(m, p, gen)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(want, ref) {
				t.Fatalf("rerender=%v %s prompt %v: reuse %v, fresh %v, reference %v", rerender, tc.name, p, got, want, ref)
			}
			if reused != wantReused {
				t.Fatalf("rerender=%v %s: reused %d prompt tokens, want %d", rerender, tc.name, reused, wantReused)
			}
			if pos := s.(Inkling2LoopRewinder).Pos(); pos != len(p)+gen-1 {
				t.Fatalf("rerender=%v %s: pos %d after %d prompt + %d decode steps", rerender, tc.name, pos, len(p), gen-1)
			}
			prev, prevGen = p, got
		}
		s.Close()
	}
}

func TestInkling2HostRewinderCheckpointRestoreReset(t *testing.T) {
	m := ink2SessTiny()
	s, err := NewInkling2LoopSession(m, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	rw := s.(Inkling2LoopRewinder)
	if err := rw.Restore(); !errors.Is(err, ErrInkling2NoCheckpoint) {
		t.Fatalf("restore without checkpoint: %v", err)
	}
	prefix := []int{1, 4, 2, 7, 3, 8}
	if _, err := s.Prefill(context.Background(), prefix); err != nil {
		t.Fatal(err)
	}
	if rw.Pos() != len(prefix) {
		t.Fatalf("pos %d after prefill of %d", rw.Pos(), len(prefix))
	}
	if err := rw.Checkpoint(1 << 20); err != nil {
		t.Fatal(err)
	}
	run := func() []int {
		next, err := s.Prefill(context.Background(), []int{5})
		if err != nil {
			t.Fatal(err)
		}
		out, err := ink2DecodeFrom(s, next, 7)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	first := run()
	if rw.Pos() != len(prefix)+7 {
		t.Fatalf("pos %d after decode", rw.Pos())
	}
	for i := 0; i < 2; i++ {
		if err := rw.Restore(); err != nil {
			t.Fatal(err)
		}
		if rw.Pos() != len(prefix) {
			t.Fatalf("restore %d: pos %d, want %d", i, rw.Pos(), len(prefix))
		}
		if again := run(); !reflect.DeepEqual(again, first) {
			t.Fatalf("restore %d: stream %v, want %v", i, again, first)
		}
	}
	want, err := ink2FreshTurn(m, ink2Cat(prefix, []int{5}), 7)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, want) {
		t.Fatalf("checkpointed stream %v, fresh %v", first, want)
	}
	if err := rw.Reset(); err != nil {
		t.Fatal(err)
	}
	if rw.Pos() != 0 {
		t.Fatalf("pos %d after reset", rw.Pos())
	}
	if err := rw.Restore(); !errors.Is(err, ErrInkling2NoCheckpoint) {
		t.Fatalf("reset kept the checkpoint: %v", err)
	}
	p := []int{2, 9, 9, 1}
	next, err := s.Prefill(context.Background(), p)
	if err != nil {
		t.Fatal(err)
	}
	got, err := ink2DecodeFrom(s, next, 5)
	if err != nil {
		t.Fatal(err)
	}
	if want, _ := ink2FreshTurn(m, p, 5); !reflect.DeepEqual(got, want) {
		t.Fatalf("after reset %v, fresh %v", got, want)
	}
}

func TestInkling2RingSpansCoverPositionsInOrder(t *testing.T) {
	for _, tc := range []struct {
		lo, hi, cap int
		want        []ink2Span
	}{
		{0, 0, 4, nil},
		{0, 3, 4, []ink2Span{{0, 0, 3}}},
		{2, 6, 4, []ink2Span{{2, 0, 2}, {0, 2, 2}}},
		{5, 8, 3, []ink2Span{{2, 0, 1}, {0, 1, 2}}},
		{1027, 1030, 515, []ink2Span{{512, 0, 3}}},
		{1030, 1033, 515, []ink2Span{{0, 0, 3}}},
	} {
		got := inkling2RingSpans(tc.lo, tc.hi, tc.cap)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("spans [%d,%d) cap %d = %v, want %v", tc.lo, tc.hi, tc.cap, got, tc.want)
		}
		covered := 0
		for _, sp := range got {
			for i := 0; i < sp.n; i++ {
				if p := tc.lo + sp.off + i; p%tc.cap != sp.ring+i {
					t.Errorf("position %d maps to ring %d, want %d", p, sp.ring+i, p%tc.cap)
				}
			}
			covered += sp.n
		}
		if covered != tc.hi-tc.lo {
			t.Errorf("spans [%d,%d) cover %d rows", tc.lo, tc.hi, covered)
		}
	}
}

func TestInkling2SavedSpanCoversOnlyRowsTheHorizonOverwrites(t *testing.T) {
	const w = 1024
	for _, c := range []struct{ pos, cap, horizon, lo, hi int }{
		{pos: 3000, cap: w, horizon: 24, lo: 1976, hi: 2000},
		{pos: 3000, cap: w, horizon: 5000, lo: 1976, hi: 3000},
		{pos: 3000, cap: w + 512, horizon: 160, lo: 1976, hi: 1976},
		{pos: 3000, cap: w + 512, horizon: 600, lo: 1976, hi: 2064},
		{pos: 100, cap: w, horizon: 24, lo: 0, hi: 0},
		{pos: 1030, cap: w, horizon: 24, lo: 6, hi: 30},
	} {
		lo, hi := inkling2SavedSpan(c.pos, w, c.cap, c.horizon)
		if lo != c.lo || hi != c.hi {
			t.Errorf("inkling2SavedSpan(%d, %d, %d, %d) = [%d,%d), want [%d,%d)", c.pos, w, c.cap, c.horizon, lo, hi, c.lo, c.hi)
		}
	}
}

func TestInkling2CheckpointRowWordsKeepGlobalF16InPlace(t *testing.T) {
	hp := inkling2MiniKVShape()
	s := &inkling2Resident{hp: hp, kvHalf: true}
	s.hist = make([]ink2Ref, len(inkling2ResidentCaps(hp, 131072, false, 0)))
	kvDim := hp.KVHeads * hp.HeadDim
	for slot := range s.hist {
		got := s.checkpointRowWords(slot)
		if s.localAt(slot) {
			if got != kvDim {
				t.Fatalf("local slot %d checkpoint row words = %d, want FP32 width %d", slot, got, kvDim)
			}
		} else if got != 0 {
			t.Fatalf("global FP16 slot %d checkpoint row words = %d, want in-place zero", slot, got)
		}
	}
}

func TestInkling2CheckpointKVSideStorageIsHorizonBoundAt128K(t *testing.T) {
	hp := inkling2MiniKVShape()
	const ctx, horizon = 131072, 64
	caps := inkling2ResidentCaps(hp, ctx, false, 0)
	s := &inkling2Resident{hp: hp, kvHalf: true, caps: caps, hist: make([]ink2Ref, len(caps))}
	var bytes int64
	localSlots := 0
	maxRows, maxWords := 0, 0
	for slot, cap := range caps {
		words := s.checkpointRowWords(slot)
		if words == 0 {
			continue
		}
		localSlots++
		lo, hi := inkling2SavedSpan(ctx-1, hp.SlidingWindow, cap, horizon)
		rows := hi - lo
		maxRows, maxWords = max(maxRows, rows), max(maxWords, words)
		bytes += int64(2 * rows * words * 4)
	}
	bytes += int64(maxRows * maxWords * 4) // the one shared exact-copy fill
	wantMax := int64((2*localSlots + 1) * horizon * hp.KVHeads * hp.HeadDim * 4)
	if bytes > wantMax {
		t.Fatalf("128K checkpoint KV side storage = %d bytes, want <= horizon bound %d", bytes, wantMax)
	}
	if fullGlobalCopy := int64(8 * 2 * ctx * hp.KVHeads * hp.HeadDim * 2); bytes >= fullGlobalCopy {
		t.Fatalf("checkpoint side storage %d duplicates the %d-byte global FP16 window", bytes, fullGlobalCopy)
	}
}

func (d *fakeInk2Cohort) Ink2Add(a, b, out compute.Ink2Ref, n int) error {
	for i := range n {
		d.rows(out)[i] = d.rows(a)[i] + d.rows(b)[i]
	}
	return nil
}

func inkling2RewindFixture(t *testing.T, half bool) (*inkling2ResidentSession, *fakeInk2Cohort) {
	t.Helper()
	d := newFakeInk2Cohort()
	hp := model.Inkling2LoopHParams{
		Hidden: 1, Layers: 2, KVHeads: 1, HeadDim: 2, SlidingWindow: 4, ConvKernel: 2,
		CoreStart: 0, CoreEnd: 2, Passes: 1, LocalLayer: []bool{false, true},
	}
	alloc := func(n int) ink2Ref {
		tt, err := d.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return compute.Ink2At(tt, 0)
	}
	s := &inkling2Resident{
		be: d, dev: d, hp: hp, kvHalf: half, ctx: 16, pos: 6,
		caps: []int{16, 4}, hist: make([]ink2Ref, 2), kc: make([]ink2Ref, 2), vc: make([]ink2Ref, 2),
	}
	for slot := range s.hist {
		s.hist[slot] = alloc(s.histFloats())
		words := inkling2BatchKVFloats(s.caps[slot]*hp.KVHeads*hp.HeadDim, s.kvHalfSlot(slot))
		s.kc[slot], s.vc[slot] = alloc(words), alloc(words)
	}
	r := &inkling2ResidentSession{s: s}
	t.Cleanup(func() { r.Close() })
	return r, d
}

func TestInkling2F16CheckpointRestoresLocalRingAndHistoryOnly(t *testing.T) {
	r, d := inkling2RewindFixture(t, true)
	s := r.s
	for slot := range s.hist {
		for i := range d.rows(s.hist[slot]) {
			d.rows(s.hist[slot])[i] = float32(100*slot + i + 1)
		}
		for i := range d.rows(s.kc[slot]) {
			d.rows(s.kc[slot])[i] = float32(1000*slot + i + 1)
			d.rows(s.vc[slot])[i] = float32(2000*slot + i + 1)
		}
	}
	wantHist := append([]float32(nil), d.rows(s.hist[1])...)
	wantK := append([]float32(nil), d.rows(s.kc[1])...)
	wantV := append([]float32(nil), d.rows(s.vc[1])...)
	if err := r.Checkpoint(2); err != nil {
		t.Fatal(err)
	}
	if s.checkpointRowWords(0) != 0 || s.checkpointRowWords(1) != 2 || r.ck.k[0].T.Ready() || r.ck.v[0].T.Ready() {
		t.Fatalf("checkpoint format globalWords=%d localWords=%d globalK=%v globalV=%v", s.checkpointRowWords(0), s.checkpointRowWords(1), r.ck.k[0], r.ck.v[0])
	}
	globalK := append([]float32(nil), d.rows(s.kc[0])...)
	for i := range d.rows(s.hist[1]) {
		d.rows(s.hist[1])[i] = -1
	}
	for _, p := range []int{2, 3} {
		for j := 0; j < 2; j++ {
			d.rows(s.kc[1])[(p%4)*2+j] = -2
			d.rows(s.vc[1])[(p%4)*2+j] = -3
		}
	}
	s.pos = 8
	if err := r.Restore(); err != nil {
		t.Fatal(err)
	}
	if s.pos != 6 || !reflect.DeepEqual(d.rows(s.hist[1]), wantHist) || !reflect.DeepEqual(d.rows(s.kc[1]), wantK) || !reflect.DeepEqual(d.rows(s.vc[1]), wantV) {
		t.Fatalf("restore pos=%d hist=%v k=%v v=%v", s.pos, d.rows(s.hist[1]), d.rows(s.kc[1]), d.rows(s.vc[1]))
	}
	if !reflect.DeepEqual(d.rows(s.kc[0]), globalK) {
		t.Fatal("restore rewrote in-place global FP16 words")
	}
}

func TestInkling2F16CheckpointGrowthFailureAndHorizonExhaustionInvalidate(t *testing.T) {
	r, d := inkling2RewindFixture(t, true)
	if err := r.Checkpoint(1); err != nil {
		t.Fatal(err)
	}
	r.s.pos += 2
	if err := r.Restore(); !errors.Is(err, ErrInkling2NoCheckpoint) || r.ck.valid {
		t.Fatalf("exhausted restore err=%v valid=%v", err, r.ck.valid)
	}
	r.s.pos = 6
	if err := r.Checkpoint(1); err != nil {
		t.Fatal(err)
	}
	d.capFloats = 1
	if err := r.Checkpoint(4); !errors.Is(err, ErrInkling2CheckpointUnavailable) || r.ck.valid || r.s.err != nil {
		t.Fatalf("growth failure err=%v valid=%v sessionErr=%v", err, r.ck.valid, r.s.err)
	}
	if err := r.Restore(); !errors.Is(err, ErrInkling2NoCheckpoint) {
		t.Fatalf("growth failure left a restorable checkpoint: %v", err)
	}
}

func TestInkling2CheckpointShortPromptZeroRingCopyRoundTrip(t *testing.T) {
	for _, half := range []bool{false, true} {
		t.Run(map[bool]string{false: "f32", true: "f16"}[half], func(t *testing.T) {
			r, d := inkling2RewindFixture(t, half)
			r.s.pos = 1
			for i := range d.rows(r.s.hist[1]) {
				d.rows(r.s.hist[1])[i] = float32(i + 11)
			}
			want := append([]float32(nil), d.rows(r.s.hist[1])...)
			if err := r.Checkpoint(2); err != nil {
				t.Fatal(err)
			}
			if r.ck.rows != 0 || r.ck.k != nil || r.ck.v != nil {
				t.Fatalf("short checkpoint allocated ring copies: rows=%d k=%v v=%v", r.ck.rows, r.ck.k, r.ck.v)
			}
			for i := range d.rows(r.s.hist[1]) {
				d.rows(r.s.hist[1])[i] = -1
			}
			r.s.pos = 2
			if err := r.Restore(); err != nil {
				t.Fatal(err)
			}
			if r.s.pos != 1 || !reflect.DeepEqual(d.rows(r.s.hist[1]), want) {
				t.Fatalf("restore pos=%d hist=%v want=%v", r.s.pos, d.rows(r.s.hist[1]), want)
			}
		})
	}
}
