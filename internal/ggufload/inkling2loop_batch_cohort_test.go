package ggufload

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fakeInk2Cohort is a host stand-in for the device a batch forward drives: every
// buffer is a tracked host slice, and the zero-layer forward reduces to "the next
// greedy token is the fed token plus one", the rule fakeInk2Session also follows.
type fakeInk2Cohort struct {
	compute.Backend
	compute.Inkling2Device
	compute.Inkling2PrefillDevice
	data          map[compute.Buffer][]float32
	allocs, frees int
	live          int
	failArgmax    bool
}

func newFakeInk2Cohort() *fakeInk2Cohort {
	return &fakeInk2Cohort{data: map[compute.Buffer][]float32{}}
}

func (d *fakeInk2Cohort) Name() string { return "fake-ink2" }
func (d *fakeInk2Cohort) Tier() string { return "" }

func (d *fakeInk2Cohort) Free(t compute.Tensor) {
	if buf, ok := d.data[t.Buf()]; ok {
		d.live -= len(buf)
		d.frees++
		delete(d.data, t.Buf())
	}
}

func (d *fakeInk2Cohort) Ink2Alloc(n int) (compute.Tensor, error) {
	buf := make([]float32, n)
	t := compute.NewF32(d, []int{n}, buf)
	d.data[t.Buf()] = buf
	d.allocs++
	d.live += n
	return t, nil
}

func (d *fakeInk2Cohort) rows(r compute.Ink2Ref) []float32 { return d.data[r.T.Buf()][r.Off:] }

func (d *fakeInk2Cohort) Ink2Write(dst compute.Ink2Ref, src []float32) error {
	copy(d.rows(dst), src)
	return nil
}

func (d *fakeInk2Cohort) Ink2ReadInts(src compute.Ink2Ref, n int) ([]int32, error) {
	out := make([]int32, n)
	for i, v := range d.rows(src)[:n] {
		out[i] = int32(v)
	}
	return out, nil
}

func (d *fakeInk2Cohort) Ink2GraphReset()       {}
func (d *fakeInk2Cohort) Ink2GraphBegin() error { return errors.New("fake device has no graphs") }

func (d *fakeInk2Cohort) Ink2RMS(x, _, y compute.Ink2Ref, rows, n int, _ float32) error {
	copy(d.rows(y)[:rows*n], d.rows(x)[:rows*n])
	return nil
}

func (d *fakeInk2Cohort) Ink2PRMS(x, w, y compute.Ink2Ref, rows, n int, eps float32) error {
	return d.Ink2RMS(x, w, y, rows, n, eps)
}

func (d *fakeInk2Cohort) Ink2PMix(states []compute.Ink2Ref, _ compute.Ink2Ref, nq int, out compute.Ink2Ref, n, T int) error {
	for q := 0; q < nq; q++ {
		copy(d.rows(out)[q*T*n:(q+1)*T*n], d.rows(states[len(states)-1])[:T*n])
	}
	return nil
}

func (d *fakeInk2Cohort) Ink2PMatMul(w compute.Tensor, x, y compute.Ink2Ref, T int) error {
	rows, cols := w.Shape[0], w.Shape[1]
	for t := 0; t < T; t++ {
		d.rows(y)[t*rows] = d.rows(x)[t*cols]
	}
	return nil
}

func (d *fakeInk2Cohort) Ink2SoftcapArgmax(logits compute.Ink2Ref, _ int, _ float32, outIdx compute.Ink2Ref) error {
	if d.failArgmax {
		return errors.New("device fault")
	}
	d.rows(outIdx)[0] = d.rows(logits)[0] + 1
	return nil
}

type fakeInk2Embedding struct{}

func (fakeInk2Embedding) Row(token int, dst []float32) { dst[0] = float32(token) }

func fakeInk2CohortModel() *Inkling2LoopModel {
	const h = 4
	return &Inkling2LoopModel{
		HP: model.Inkling2LoopHParams{Hidden: h, Passes: 1, Heads: 1, KVHeads: 1, HeadDim: 1, ConvKernel: 2, VocabUnpadded: 1},
		Weights: &model.Inkling2LoopWeights{TokEmbd: fakeInk2Embedding{},
			Output: &model.Inkling2LoopDenseMatrix{R: 1, C: h, Data: make([]float32, h)}},
	}
}

type fakeInk2Session struct{}

func (fakeInk2Session) Prefill(prompt []int) (int, error) { return prompt[len(prompt)-1] + 1, nil }
func (fakeInk2Session) Step(token int) (int, error)       { return token + 1, nil }
func (fakeInk2Session) Close() error                      { return nil }

type inkling2ParityCase struct {
	name     string
	prompt   []int
	maxNew   int
	stop     int // 0: none
	refuseAt int // emit refuses its refuseAt-th token; 0: never
}

func (c inkling2ParityCase) callbacks(got *[]int) (func(int) bool, func(int) bool) {
	return func(t int) bool { return c.stop != 0 && t == c.stop }, func(t int) bool {
		*got = append(*got, t)
		return c.refuseAt == 0 || len(*got) < c.refuseAt
	}
}

var inkling2ParityCases = []inkling2ParityCase{
	{name: "max tokens", prompt: []int{5}, maxNew: 4},
	{name: "single token budget", prompt: []int{20, 21}, maxNew: 1},
	{name: "stop token mid turn", prompt: []int{5}, maxNew: 10, stop: 8},
	{name: "stop token first", prompt: []int{30, 31, 32}, maxNew: 10, stop: 33},
	{name: "sink refuses mid turn", prompt: []int{40}, maxNew: 10, refuseAt: 3},
	{name: "sink refuses first", prompt: []int{50, 51}, maxNew: 10, refuseAt: 1},
	{name: "sink refuses the last budgeted token", prompt: []int{60}, maxNew: 2, refuseAt: 2},
}

// A row of a cohort counts, emits and stops exactly as the single-sequence decode
// does for the same request: the serving path must not report a different
// completion length depending on how many requests happened to share a cohort.
func TestInkling2LoopBatchCohortMatchesScalarTurnAccounting(t *testing.T) {
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	cases := inkling2ParityCases
	got := make([][]int, len(cases))
	var done []Inkling2LoopBatchResult
	reqs := make([]Inkling2LoopBatchRequest, len(cases))
	for i, c := range cases {
		stop, emit := c.callbacks(&got[i])
		reqs[i] = Inkling2LoopBatchRequest{Prompt: c.prompt, MaxTokens: c.maxNew, Stop: stop, Emit: emit,
			Done: func(r Inkling2LoopBatchResult) { done = append(done, r) }}
	}
	results, err := bt.Generate(reqs)
	if err != nil {
		t.Fatal(err)
	}
	if len(done) != len(cases) {
		t.Fatalf("Done called %d times for %d rows", len(done), len(cases))
	}
	for i, c := range cases {
		var want []int
		stop, emit := c.callbacks(&want)
		turn := Inkling2LoopTurn{MaxTokens: c.maxNew, Stop: stop, Emit: emit}
		if _, err := Inkling2LoopSessionDecode(context.Background(), fakeInk2Session{}, c.prompt, &turn); err != nil {
			t.Fatal(err)
		}
		r := results[i]
		if !reflect.DeepEqual(got[i], want) || r.Generated != turn.Generated || r.Stopped != turn.Stopped || r.Err != nil {
			t.Errorf("%s: batch emitted %v generated=%d stopped=%v err=%v; scalar emitted %v generated=%d stopped=%v",
				c.name, got[i], r.Generated, r.Stopped, r.Err, want, turn.Generated, turn.Stopped)
		}
	}
}

// Back-to-back cohorts of mixed width and length reuse one state: the weights are
// bound once, the arenas grow only past the widest and longest cohort seen, and
// nothing is left allocated after Close.
func TestInkling2LoopBatchCohortsReuseStateAcrossCohorts(t *testing.T) {
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	run := func(width, promptLen, maxNew int) {
		t.Helper()
		reqs := make([]Inkling2LoopBatchRequest, width)
		for i := range reqs {
			prompt := make([]int, promptLen)
			for j := range prompt {
				prompt[j] = 100*i + j
			}
			reqs[i] = Inkling2LoopBatchRequest{Prompt: prompt, MaxTokens: maxNew, Emit: func(int) bool { return true }}
		}
		results, err := bt.Generate(reqs)
		if err != nil {
			t.Fatal(err)
		}
		for i, r := range results {
			if r.Generated != maxNew || r.Err != nil {
				t.Fatalf("cohort %dx%d row %d = %+v", width, promptLen, i, r)
			}
		}
	}
	run(2, 3, 4)
	run(Inkling2LoopMaxBatch, 70, 2) // wider and past one 64-position block: regrow once
	allocs, live := d.allocs, d.live
	for round := 0; round < 4; round++ {
		run(2, 3, 4)
		run(Inkling2LoopMaxBatch, 70, 2)
		run(5, 10, 8)
	}
	if d.allocs != allocs || d.live != live {
		t.Fatalf("cohorts that fit allocated again: allocs %d -> %d, live floats %d -> %d", allocs, d.allocs, live, d.live)
	}
	bt.Close()
	if d.live != 0 || d.frees != d.allocs {
		t.Fatalf("after Close: %d floats live, %d frees of %d allocs", d.live, d.frees, d.allocs)
	}
}

// A failed step drops the cached state, whose binding now carries a sticky error,
// and the next cohort builds a fresh one.
func TestInkling2LoopBatchCohortsDropStateAfterFault(t *testing.T) {
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	reqs := func() []Inkling2LoopBatchRequest {
		emit := func(int) bool { return true }
		return []Inkling2LoopBatchRequest{{Prompt: []int{1}, MaxTokens: 2, Emit: emit}, {Prompt: []int{2}, MaxTokens: 2, Emit: emit}}
	}
	d.failArgmax = true
	results, err := bt.Generate(reqs())
	if err == nil || len(results) != 2 || results[0].Err == nil || results[1].Err == nil {
		t.Fatalf("faulted cohort: results=%+v err=%v", results, err)
	}
	if d.live != 0 {
		t.Fatalf("faulted cohort left %d floats live", d.live)
	}
	d.failArgmax = false
	if results, err := bt.Generate(reqs()); err != nil || results[0].Generated != 2 {
		t.Fatalf("cohort after a fault: results=%+v err=%v", results, err)
	}
}

// A row cancelled mid-cohort is retired at the next step: its Done fires with the
// cancellation while the rest of the cohort is still decoding, and it is never
// emitted to again.
func TestInkling2LoopBatchCohortRetiresCancelledRowEarly(t *testing.T) {
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var events []string
	var cancelledEmits int
	reqs := []Inkling2LoopBatchRequest{
		{Prompt: []int{1}, MaxTokens: 20, Emit: func(int) bool { events = append(events, "long"); return true },
			Done: func(Inkling2LoopBatchResult) { events = append(events, "long done") }},
		{Context: ctx, Prompt: []int{2}, MaxTokens: 20, Emit: func(int) bool {
			cancelledEmits++
			if cancelledEmits == 2 {
				cancel()
			}
			return true
		}, Done: func(r Inkling2LoopBatchResult) {
			if !errors.Is(r.Err, context.Canceled) || r.Generated != 2 {
				t.Errorf("cancelled row result %+v", r)
			}
			events = append(events, "cancelled done")
		}},
	}
	if _, err := bt.Generate(reqs); err != nil {
		t.Fatal(err)
	}
	if cancelledEmits != 2 {
		t.Fatalf("cancelled row emitted %d times, want 2", cancelledEmits)
	}
	at := -1
	for i, e := range events {
		if e == "cancelled done" {
			at = i
		}
	}
	// the long row has emitted 2 tokens when the cancel lands; it finishes 18 later
	if at < 0 || at > 3 || events[len(events)-1] != "long done" {
		t.Fatalf("cancelled row retired at event %d of %v", at, events)
	}
}

// ReleaseArenas leaves only the weight binding allocated, is a no-op when repeated
// or after a fault already dropped the state, and the next cohort allocates again.
func TestInkling2LoopBatchCohortsReleaseArenasKeepsBinding(t *testing.T) {
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	reqs := func(width int) []Inkling2LoopBatchRequest {
		out := make([]Inkling2LoopBatchRequest, width)
		for i := range out {
			out[i] = Inkling2LoopBatchRequest{Prompt: []int{10 * i, 10*i + 1}, MaxTokens: 3, Emit: func(int) bool { return true }}
		}
		return out
	}
	if _, err := bt.Generate(reqs(Inkling2LoopMaxBatch)); err != nil {
		t.Fatal(err)
	}
	bt.ReleaseArenas()
	bt.ReleaseArenas()
	binding := 0
	for _, tensor := range bt.s.owned {
		binding += len(d.data[tensor.Buf()])
	}
	if bt.b != nil || bt.s == nil || d.live != binding || d.live == 0 {
		t.Fatalf("after ReleaseArenas: %d floats live, binding holds %d", d.live, binding)
	}
	allocs := d.allocs
	if results, err := bt.Generate(reqs(2)); err != nil || results[1].Generated != 3 {
		t.Fatalf("cohort after ReleaseArenas: results=%+v err=%v", results, err)
	}
	if d.allocs == allocs || d.live <= binding {
		t.Fatalf("next cohort did not reallocate arenas: allocs %d -> %d, live %d", allocs, d.allocs, d.live)
	}
	d.failArgmax = true
	if _, err := bt.Generate(reqs(2)); err == nil {
		t.Fatal("faulted cohort succeeded")
	}
	bt.ReleaseArenas()
	if d.live != 0 {
		t.Fatalf("ReleaseArenas after a fault: %d floats live", d.live)
	}
}
