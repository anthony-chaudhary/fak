//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"
)

type ink2SchedArm struct {
	concurrent  bool
	commitEvery int
}

func (a ink2SchedArm) String() string {
	return fmt.Sprintf("concurrent=%v commit=%d", a.concurrent, a.commitEvery)
}

func ink2SchedDevice(t testing.TB, arm ink2SchedArm) *Ink2Device {
	t.Helper()
	if !Available() {
		t.Skip("requires a Metal device")
	}
	d, err := newInk2Device(Ink2ProfileOff)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	d.setSchedule(arm.concurrent, arm.commitEvery)
	return d
}

// fak-test:runtime integration est=1s lane=default
func TestInk2ConcurrentBarriersOnlyAtHazards(t *testing.T) {
	d := ink2SchedDevice(t, ink2SchedArm{concurrent: true})
	const n = 256
	buf := ink2TestAlloc(t, d, 16*n)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i%13) - 6
	}
	if err := d.Write(buf, x); err != nil {
		t.Fatal(err)
	}
	p, err := d.PipelinePrecise(ink2TestKernels, "test_affine")
	if err != nil {
		t.Fatal(err)
	}
	type affine struct{ scale, bias float32 }
	group := [3]int{min(n, p.MaxThreads()), 1, 1}
	launch := func(src, dst int, scale float32) {
		t.Helper()
		if err := d.DispatchThreads(p, [3]int{n, 1, 1}, group, buf.At(src*n).Span(n), buf.At(dst*n).Span(n),
			Ink2Const(affine{scale, 1}), Ink2Ref{}.Span(0)); err != nil {
			t.Fatal(err)
		}
	}
	steps := []struct {
		src, dst int
		barrier  bool
		why      string
	}{
		{0, 1, false, "first dispatch"},
		{0, 2, false, "reads what is only read, writes disjoint bytes"},
		{0, 3, false, "a third reader of slot 0"},
		{1, 4, true, "reads slot 1, written above"},
		{0, 5, false, "independent of the dispatch before it"},
		{6, 0, true, "writes slot 0, which an unbarriered dispatch read"},
		{7, 0, true, "writes slot 0, which an unbarriered dispatch wrote"},
		{8, 9, false, "untouched slots"},
	}
	for _, s := range steps {
		before := d.Stats().Barriers
		launch(s.src, s.dst, 2)
		if got := d.Stats().Barriers - before; got != uint64(btoi(s.barrier)) {
			t.Fatalf("slot %d -> %d (%s): %d barriers, want %v", s.src, s.dst, s.why, got, s.barrier)
		}
	}
	before := d.Stats().Barriers
	d.Barrier()
	launch(9, 10, 1)
	if got := d.Stats().Barriers - before; got != 1 {
		t.Fatalf("a forced barrier then a hazard-free dispatch: %d barriers, want only the forced one", got)
	}
	got, err := d.Read(buf, 11*n)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		one := x[i]*2 + 1
		want := map[int]float32{1: one, 2: one, 3: one, 4: one*2 + 1, 5: one, 0: 1, 9: 1, 10: 2}
		for slot, w := range want {
			if v := got[slot*n+i]; v != w {
				t.Fatalf("slot %d element %d = %v, want %v", slot, i, v, w)
			}
		}
	}
}

// ink2SchedChain is a decode-shaped chain of the real ops over one scratch
// arena, repeated over tokens that each read the previous token's residual:
// norm, three projections of one input, routing, routed gate/up with scale,
// SITU, routed down, combine, residual adds, a mix and the argmax.
type ink2SchedChain struct {
	gu, down, proj, wo, router Ink2Weight
	routeBias, scale, normW    []float32
}

const (
	ink2SchedN      = 256
	ink2SchedExp    = 16
	ink2SchedK      = 4
	ink2SchedWidth  = 128
	ink2SchedTokens = 4
)

func ink2SchedWeights(t *testing.T) ink2SchedChain {
	t.Helper()
	store, data := ink2TestStore(t, 256)
	rng := rand.New(rand.NewSource(7))
	at := 64
	fill := func(typ uint32, rows, cols, experts int) Ink2Weight {
		m, next := ink2TestFill(t, store, data, rng, at, typ, rows, cols, experts)
		at = next
		if experts == 1 {
			w, err := m.stack.Ink2Weight(0)
			if err != nil {
				t.Fatal(err)
			}
			return w
		}
		w, ok := m.stack.Ink2Stack()
		if !ok {
			t.Fatal("expert stack split across buffers")
		}
		return w
	}
	c := ink2SchedChain{
		gu:     fill(8, 2*ink2SchedWidth, ink2SchedN, ink2SchedExp),
		down:   fill(8, ink2SchedN, ink2SchedWidth, ink2SchedExp),
		proj:   fill(8, 3*ink2SchedN, ink2SchedN, 1),
		wo:     fill(8, ink2SchedN, ink2SchedN, 1),
		router: fill(0, ink2SchedExp, ink2SchedN, 1),
	}
	c.routeBias = ink2OpsTestRand(rng, ink2SchedExp, .1)
	c.scale = make([]float32, ink2SchedExp)
	for i := range c.scale {
		c.scale[i] = .5 + rng.Float32()
	}
	c.normW = ink2OpsTestRand(rng, ink2SchedN, 1)
	return c
}

// run returns the whole arena and the argmax ids of every token.
func (c ink2SchedChain) run(t testing.TB, d *Ink2Device) ([]float32, []int32) {
	t.Helper()
	const n, k, w = ink2SchedN, ink2SchedK, ink2SchedWidth
	arena, err := d.Alloc(1 << 16)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Free(arena) })
	a := Ink2Ref{Buf: arena}
	next := 0
	slot := func(size int) Ink2Ref {
		r := a.At(next)
		next += (size + 3) &^ 3
		return r
	}
	resid, h, qkv, att, o := slot(n), slot(n), slot(3*n), slot(n), slot(n)
	logits, ids, wts := slot(ink2SchedExp), slot(k), slot(k)
	gu, act, dn, moe, mixed := slot(k*2*w), slot(k*w), slot(k*n), slot(n), slot(n)
	bias, scale, normW, outIdx := slot(ink2SchedExp), slot(ink2SchedExp), slot(n), slot(4*ink2SchedTokens)
	rng := rand.New(rand.NewSource(11))
	for _, up := range []struct {
		r Ink2Ref
		v []float32
	}{{resid, ink2OpsTestRand(rng, n, 1)}, {bias, c.routeBias}, {scale, c.scale}, {normW, c.normW}} {
		if err := d.Write(up.r, up.v); err != nil {
			t.Fatal(err)
		}
	}
	must := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	for tok := range ink2SchedTokens {
		must(d.RMS(resid, normW, h, 1, n, 1e-6))
		must(d.MatVec(c.proj, h, qkv))
		must(d.SigmoidMul(qkv, qkv.At(n), n))
		must(d.Add(qkv, qkv.At(2*n), att, n))
		must(d.MatVec(c.wo, att, o))
		must(d.Add(resid, o, resid, n))
		must(d.MatVec(c.router, resid, logits))
		must(d.Route(logits, bias, ids, wts, ink2SchedExp, k, 1.5))
		ok, err := d.ExpertMatVec(c.gu, ids, resid, 0, gu, k, scale, nil)
		must(err)
		if !ok {
			t.Fatal("gate/up expert matvec declined")
		}
		must(d.Situ(gu, act, k, w, w, 1.25, .5))
		if ok, err = d.ExpertMatVec(c.down, ids, act, w, dn, k, scale, nil); err != nil || !ok {
			t.Fatalf("down expert matvec: %v, %v", ok, err)
		}
		must(d.Combine(dn, wts, moe, k, n))
		must(d.Mix([]Ink2Ref{resid, moe, o}, h, 1, []Ink2Ref{normW}, 1e-6, mixed, n))
		must(d.Add(resid, mixed, resid, n))
		must(d.SoftcapArgmax(logits, ink2SchedExp, 30, outIdx.At(4*tok)))
	}
	ints, err := d.ReadInts(outIdx, 4*ink2SchedTokens)
	if err != nil {
		t.Fatal(err)
	}
	var argmax []int32
	for tok := range ink2SchedTokens {
		argmax = append(argmax, ints[4*tok])
	}
	out, err := d.Read(a, next)
	if err != nil {
		t.Fatal(err)
	}
	return out, argmax
}

// fak-test:runtime integration est=2s lane=default
func TestInk2ConcurrentChainMatchesSerialBitForBit(t *testing.T) {
	c := ink2SchedWeights(t)
	ref := ink2SchedDevice(t, ink2SchedArm{})
	want, wantIDs := c.run(t, ref)
	refStats := ref.Stats()
	if refStats.Barriers != 0 || refStats.Commits != 1 {
		t.Fatalf("serial reference: %+v, want no barriers and one command buffer", refStats)
	}
	for _, arm := range []ink2SchedArm{{true, 0}, {true, 1}, {true, 3}, {true, 7}, {false, 5}} {
		d := ink2SchedDevice(t, arm)
		for rep := range 3 {
			got, ids := c.run(t, d)
			if !slices.Equal(ids, wantIDs) {
				t.Fatalf("%v rep %d: argmax %v, want %v", arm, rep, ids, wantIDs)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%v rep %d: arena element %d = %v, serial %v", arm, rep, i, got[i], want[i])
				}
			}
		}
		s := d.Stats()
		if arm.concurrent && arm.commitEvery != 1 && (s.Barriers == 0 || s.Barriers >= s.Dispatches) {
			t.Fatalf("%v: %d barriers over %d dispatches, want some dispatches left unordered", arm, s.Barriers, s.Dispatches)
		}
		if arm.commitEvery > 0 && s.Commits <= s.Submissions {
			t.Fatalf("%v: %d command buffers for %d submissions, want early commits", arm, s.Commits, s.Submissions)
		}
		t.Logf("%v: %d dispatches, %d barriers, %d command buffers, %d submissions", arm, s.Dispatches, s.Barriers, s.Commits, s.Submissions)
	}
}

// TestInk2ScheduleMicrobench times a decode-shaped token on each schedule.
// FAK_INK2_MICROBENCH=1 runs it; _REPS sets the tokens per arm and round.
//
// fak-test:runtime integration est=30s lane=default
func TestInk2ScheduleMicrobench(t *testing.T) {
	if os.Getenv("FAK_INK2_MICROBENCH") != "1" {
		t.Skip("set FAK_INK2_MICROBENCH=1")
	}
	reps := 50
	if v, err := strconv.Atoi(os.Getenv("FAK_INK2_MICROBENCH_REPS")); err == nil && v > 0 {
		reps = v
	}
	arms := ink2SchedBenchArms()
	best := make([]time.Duration, len(arms))
	stats := make([]Ink2Stats, len(arms))
	devs := make([]*Ink2Device, len(arms))
	tok := make([]func(), len(arms))
	for i, arm := range arms {
		devs[i] = ink2SchedDevice(t, arm)
		tok[i] = ink2SchedBenchToken(t, devs[i])
		tok[i]()
	}
	for round := range 15 {
		for i := range arms {
			idx := (i + round) % len(arms)
			before := devs[idx].Stats()
			start := time.Now()
			for range reps {
				tok[idx]()
			}
			el := time.Since(start) / time.Duration(reps)
			if best[idx] == 0 || el < best[idx] {
				best[idx] = el
				stats[idx] = devs[idx].Stats().Sub(before)
			}
		}
	}
	for i, arm := range arms {
		s := stats[i]
		r := float64(reps)
		t.Logf("%-28v best %8.3f ms/token  gpu %.3f ms  encode %.3f ms  disp %.0f  barriers %.0f  cbs %.1f",
			arm, best[i].Seconds()*1e3, s.GPUSeconds/r*1e3, s.EncodeSeconds/r*1e3, float64(s.Dispatches)/r, float64(s.Barriers)/r, float64(s.Commits)/r)
	}
}

// ink2SchedBenchToken builds a token of 48 local layers with the decode's
// dispatch graph at model width (q8_0 weights standing in for every type) and
// returns a function that encodes and fences one token.
func ink2SchedBenchToken(t *testing.T, d *Ink2Device) func() {
	const n, exp, k, w, kvDim, layers = 2304, 32, 8, 1152, 512, 48
	store, data := ink2TestStore(t, (3*n*n+2*kvDim*n+2*exp*2*w*n)/32*34/os.Getpagesize()+64)
	rng := rand.New(rand.NewSource(3))
	at := 64
	fill := func(typ uint32, rows, cols, experts int) Ink2Weight {
		m, next := ink2TestFill(t, store, data, rng, at, typ, rows, cols, experts)
		at = next
		if experts == 1 {
			w, _ := m.stack.Ink2Weight(0)
			return w
		}
		w, _ := m.stack.Ink2Stack()
		return w
	}
	q, kw, vw, rel, sink := fill(8, n, n, 1), fill(8, kvDim, n, 1), fill(8, kvDim, n, 1), fill(0, 128, n, 1), fill(0, 32, n, 1)
	gate, wo, router := fill(8, n, n, 1), fill(8, n, n, 1), fill(0, exp, n, 1)
	gu, down := fill(8, 2*w, n, exp), fill(8, n, w, exp)
	arena, err := d.Alloc(1 << 20)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Free(arena) })
	a := Ink2Ref{Buf: arena}
	off := 0
	slot := func(size int) Ink2Ref { r := a.At(off); off += (size + 63) &^ 63; return r }
	resid, h, qo, ko, vo, relo, sinko, gateo, o := slot(n), slot(n), slot(n), slot(kvDim), slot(kvDim), slot(128), slot(32), slot(n), slot(n)
	logits, ids, wts, guo, act, dn, moe := slot(exp), slot(k), slot(k), slot(k*2*w), slot(k*w), slot(k*n), slot(n)
	normW, scale, outIdx := slot(n), slot(exp), slot(4)
	for _, up := range []struct {
		r Ink2Ref
		v []float32
	}{{resid, ink2OpsTestRand(rng, n, 1)}, {normW, ink2OpsTestRand(rng, n, 1)}, {scale, ink2OpsTestRand(rng, exp, 1)}} {
		if err := d.Write(up.r, up.v); err != nil {
			t.Fatal(err)
		}
	}
	return func() {
		for range layers {
			_ = d.RMS(resid, normW, h, 1, n, 1e-6)
			_ = d.MatVec(q, h, qo)
			_ = d.MatVec(kw, h, ko)
			_ = d.MatVec(vw, h, vo)
			_ = d.MatVec(rel, h, relo)
			_ = d.MatVec(sink, h, sinko)
			_ = d.SigmoidMul(qo, ko, kvDim)
			_ = d.MatVec(gate, h, gateo)
			_ = d.SigmoidMul(qo, gateo, n)
			_ = d.MatVec(wo, qo, o)
			_ = d.Add(resid, o, resid, n)
			_ = d.RMS(resid, normW, h, 1, n, 1e-6)
			_ = d.MatVec(router, h, logits)
			_ = d.Route(logits, Ink2Ref{}, ids, wts, exp, k, 1)
			_, _ = d.ExpertMatVec(gu, ids, h, 0, guo, k, scale, nil)
			_ = d.Situ(guo, act, k, w, w, 1, .5)
			_, _ = d.ExpertMatVec(down, ids, act, w, dn, k, scale, nil)
			_ = d.Combine(dn, wts, moe, k, n)
			_ = d.Add(resid, moe, resid, n)
		}
		_ = d.SoftcapArgmax(o, n, 30, outIdx)
		if _, err := d.ReadInts(outIdx, 1); err != nil {
			t.Fatal(err)
		}
	}
}

// TestInk2BarrierCostMicrobench times 1000 tiny launches as a dependent chain
// and as independent launches on each encoder, which bounds what removing a
// drain can save. FAK_INK2_MICROBENCH=1 runs it.
//
// fak-test:runtime integration est=10s lane=default
func TestInk2BarrierCostMicrobench(t *testing.T) {
	if os.Getenv("FAK_INK2_MICROBENCH") != "1" {
		t.Skip("set FAK_INK2_MICROBENCH=1")
	}
	const n, launches = 2304, 1000
	type affine struct{ scale, bias float32 }
	for round := range 3 {
		for _, arm := range []ink2SchedArm{{false, 0}, {true, 0}} {
			for _, chained := range []bool{true, false} {
				d := ink2SchedDevice(t, arm)
				buf := ink2TestAlloc(t, d, (launches+1)*n)
				p, err := d.PipelinePrecise(ink2TestKernels, "test_affine")
				if err != nil {
					t.Fatal(err)
				}
				var best time.Duration
				var gpu float64
				for range 10 {
					before := d.Stats()
					start := time.Now()
					for i := range launches {
						src := 0
						if chained {
							src = i
						}
						_ = d.DispatchThreads(p, [3]int{n, 1, 1}, [3]int{256, 1, 1}, buf.At(src*n).Span(n), buf.At((i+1)*n).Span(n),
							Ink2Const(affine{1, 1}), Ink2Ref{}.Span(0))
					}
					if err := d.Flush(); err != nil {
						t.Fatal(err)
					}
					if el := time.Since(start); best == 0 || el < best {
						best, gpu = el, d.Stats().Sub(before).GPUSeconds
					}
				}
				t.Logf("round %d %-22v chained=%-5v best %.3f ms  gpu %.3f ms  (%.2f us/launch)", round, arm, chained, best.Seconds()*1e3, gpu*1e3, gpu*1e6/launches)
			}
		}
	}
}

// ink2SchedBenchArms reads FAK_INK2_MICROBENCH_ARMS ("c64,s0,..." where c is
// concurrent, s serial and the number the early-commit interval).
func ink2SchedBenchArms() []ink2SchedArm {
	spec := os.Getenv("FAK_INK2_MICROBENCH_ARMS")
	if spec == "" {
		spec = "s0,c0,s192,c192,s0,c96,s384"
	}
	var arms []ink2SchedArm
	for _, f := range strings.Split(spec, ",") {
		if len(f) < 2 {
			continue
		}
		n, _ := strconv.Atoi(f[1:])
		arms = append(arms, ink2SchedArm{concurrent: f[0] == 'c', commitEvery: n})
	}
	return arms
}

// TestInk2ProjectionOverlapMicrobench times the five projections of one input
// (q, k, v, rel, sink at model width), repeated, on each encoder: what the
// concurrent encoder can overlap in a decode layer. FAK_INK2_MICROBENCH=1.
//
// fak-test:runtime integration est=20s lane=default
func TestInk2ProjectionOverlapMicrobench(t *testing.T) {
	if os.Getenv("FAK_INK2_MICROBENCH") != "1" {
		t.Skip("set FAK_INK2_MICROBENCH=1")
	}
	const n, kvDim, groups = 2304, 512, 48
	arms := []ink2SchedArm{{false, 0}, {true, 0}}
	devs := make([]*Ink2Device, len(arms))
	runs := make([]func(), len(arms))
	for i, arm := range arms {
		d := ink2SchedDevice(t, arm)
		devs[i] = d
		store, data := ink2TestStore(t, ((n*n+2*kvDim*n)/32*34+160*n*4)/os.Getpagesize()+64)
		rng := rand.New(rand.NewSource(5))
		at := 64
		fill := func(typ uint32, rows int) Ink2Weight {
			m, next := ink2TestFill(t, store, data, rng, at, typ, rows, n, 1)
			at = next
			w, _ := m.stack.Ink2Weight(0)
			return w
		}
		ws := []Ink2Weight{fill(8, n), fill(8, kvDim), fill(8, kvDim), fill(0, 128), fill(0, 32)}
		buf := ink2TestAlloc(t, d, 1<<18)
		if err := d.Write(buf, ink2OpsTestRand(rng, n, 1)); err != nil {
			t.Fatal(err)
		}
		runs[i] = func() {
			for range groups {
				out := n
				for _, w := range ws {
					_ = d.MatVec(w, buf, buf.At(out))
					out += (w.Rows + 63) &^ 63
				}
			}
			_ = d.Flush()
		}
		runs[i]()
	}
	best := make([]float64, len(arms))
	for range 15 {
		for i := range arms {
			before := devs[i].Stats()
			runs[i]()
			if g := devs[i].Stats().Sub(before).GPUSeconds; best[i] == 0 || g < best[i] {
				best[i] = g
			}
		}
	}
	for i, arm := range arms {
		t.Logf("%-22v best gpu %.3f ms for %d projection groups (%.1f us/group), %d barriers", arm, best[i]*1e3, groups, best[i]*1e6/groups, devs[i].Stats().Barriers)
	}
}
