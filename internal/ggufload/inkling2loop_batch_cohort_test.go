package ggufload

import (
	"context"
	"errors"
	"os"
	"reflect"
	"strconv"
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
	// heads counts single-row output mixes, which only a cohort prefill's head runs
	heads int
	// writes counts Ink2Write calls and onWrite, when set, sees each destination
	writes  int
	onWrite func(compute.Ink2Ref)
	// writeLog records every Ink2Write in call order
	writeLog []fakeInk2Write
	// reads counts Ink2Read calls
	reads int
	// capFloats, when set, is the device's capacity: DeviceMemory reports what is not
	// live, and a device-only allocation past it is refused
	capFloats int
}

type fakeInk2Write struct {
	buf compute.Buffer
	n   int
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
	d.writes++
	d.writeLog = append(d.writeLog, fakeInk2Write{buf: dst.T.Buf(), n: len(src)})
	if d.onWrite != nil {
		d.onWrite(dst)
	}
	copy(d.rows(dst), src)
	return nil
}

func (d *fakeInk2Cohort) Ink2Read(src compute.Ink2Ref, n int) ([]float32, error) {
	d.reads++
	return append([]float32(nil), d.rows(src)[:n]...), nil
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

func (d *fakeInk2Cohort) Ink2Mix(states []compute.Ink2Ref, _ compute.Ink2Ref, _ int, _ []compute.Ink2Ref, _ float32, out compute.Ink2Ref, n int) error {
	d.heads++
	copy(d.rows(out)[:n], d.rows(states[len(states)-1])[:n])
	return nil
}

func (d *fakeInk2Cohort) Ink2AllocDeviceOnly(n int) (compute.Tensor, error) {
	if d.capFloats > 0 && d.live+n > d.capFloats {
		return compute.Tensor{}, errors.New("fake device out of memory")
	}
	return d.Ink2Alloc(n)
}

func (d *fakeInk2Cohort) DeviceMemory() (total, free int64, known bool) {
	if d.capFloats == 0 {
		return 0, compute.FreeUnknown, false
	}
	return 4 * int64(d.capFloats), 4 * int64(d.capFloats-d.live), true
}

func (d *fakeInk2Cohort) Ink2PMatMul(w compute.Tensor, x, y compute.Ink2Ref, T int) error {
	return d.matmul(w, x, y, T)
}

func (d *fakeInk2Cohort) Ink2PMatMulRows(w compute.Tensor, x, y compute.Ink2Ref, T int) error {
	return d.matmul(w, x, y, T)
}

func (d *fakeInk2Cohort) matmul(w compute.Tensor, x, y compute.Ink2Ref, T int) error {
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

func (fakeInk2Session) Prefill(_ context.Context, prompt []int) (int, error) {
	return prompt[len(prompt)-1] + 1, nil
}
func (fakeInk2Session) Step(token int) (int, error) { return token + 1, nil }
func (fakeInk2Session) Close() error                { return nil }

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
	run(Inkling2LoopMaxBatch, 70, 2) // past one 64-position block: 16 x 128
	allocs, live := d.allocs, d.live
	for round := 0; round < 4; round++ {
		run(Inkling2LoopMaxBatch, 70, 2)
		run(Inkling2LoopMaxBatch-1, 80, 4)
		run(Inkling2LoopMaxBatch, 100, 8)
	}
	if d.allocs != allocs || d.live != live {
		t.Fatalf("cohorts near the cached shape allocated again: allocs %d -> %d, live floats %d -> %d", allocs, d.allocs, live, d.live)
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

// fakeInk2CohortPrompt is a prompt of n tokens for row; the fake's first greedy token
// after it is its last token plus one.
func fakeInk2CohortPrompt(n, row int) []int {
	p := make([]int, n)
	for j := range p {
		p[j] = 1000*row + j
	}
	return p
}

// A row whose context is already cancelled is never prefilled, and a cohort whose
// every live row is cancelled stops before its next prefill chunk; a cancelled row
// leaves the live rows' tokens as they are.
func TestInkling2CohortPrefillSkipsCancelledRows(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Run("pre-cancelled row is not prefilled", func(t *testing.T) {
		d := newFakeInk2Cohort()
		bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
		defer bt.Close()
		dead, cancel := context.WithCancel(context.Background())
		cancel()
		var live0, live2 []int
		reqs := []Inkling2LoopBatchRequest{
			{Prompt: []int{10, 11, 12}, MaxTokens: 2, Emit: func(tok int) bool { live0 = append(live0, tok); return true }},
			{Context: dead, Prompt: []int{20, 21, 22, 23}, MaxTokens: 2, Emit: func(int) bool { t.Error("cancelled row emitted"); return true }},
			{Prompt: []int{30, 31}, MaxTokens: 2, Emit: func(tok int) bool { live2 = append(live2, tok); return true }},
		}
		results, err := bt.Generate(reqs)
		if err != nil {
			t.Fatal(err)
		}
		if !errors.Is(results[1].Err, context.Canceled) || results[1].Generated != 0 {
			t.Fatalf("cancelled row result %+v", results[1])
		}
		if d.heads != 2 {
			t.Fatalf("prefill heads %d, want 2: the cancelled row was prefilled", d.heads)
		}
		if !reflect.DeepEqual(live0, []int{13, 14}) || !reflect.DeepEqual(live2, []int{32, 33}) {
			t.Fatalf("live rows emitted %v and %v, want [13 14] and [32 33]", live0, live2)
		}
	})
	t.Run("all rows cancelled mid prefill stop before the next chunk", func(t *testing.T) {
		d := newFakeInk2Cohort()
		bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
		defer bt.Close()
		cohort := func(ctx context.Context) []Inkling2LoopBatchRequest {
			emit := func(int) bool { return true }
			return []Inkling2LoopBatchRequest{
				{Context: ctx, Prompt: fakeInk2CohortPrompt(1200, 0), MaxTokens: 2, Emit: emit},
				{Context: ctx, Prompt: fakeInk2CohortPrompt(1200, 1), MaxTokens: 2, Emit: emit},
			}
		}
		if _, err := bt.Generate(cohort(context.Background())); err != nil {
			t.Fatal(err)
		}
		ctx, cancelAll := context.WithCancel(context.Background())
		defer cancelAll()
		in := bt.b.in.T.Buf()
		chunks := 0
		d.onWrite = func(dst compute.Ink2Ref) {
			if dst.T.Buf() == in {
				chunks++
				cancelAll()
			}
		}
		results, err := bt.Generate(cohort(ctx))
		d.onWrite = nil
		if err != nil {
			t.Fatal(err)
		}
		if chunks != 1 {
			t.Fatalf("%d prefill chunks ran after every row was cancelled, want 1", chunks)
		}
		for i, r := range results {
			if !errors.Is(r.Err, context.Canceled) {
				t.Fatalf("row %d result %+v, want cancelled", i, r)
			}
		}
	})
}

// A row cancelled while the cohort prefills is finished before the next chunk is
// written: its Done fires with the cancellation, it is never emitted to, and its
// remaining prompt is not packed into later chunks.
func TestInkling2CohortPrefillDropsRowCancelledMidPrefill(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	const long = 3 * inkling2PrefillChunk
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	cohort := func() []Inkling2LoopBatchRequest {
		emit := func(int) bool { return true }
		return []Inkling2LoopBatchRequest{
			{Prompt: fakeInk2CohortPrompt(20, 0), MaxTokens: 2, Emit: emit},
			{Prompt: fakeInk2CohortPrompt(long, 1), MaxTokens: 2, Emit: emit},
			{Prompt: fakeInk2CohortPrompt(long, 2), MaxTokens: 2, Emit: emit},
		}
	}
	if _, err := bt.Generate(cohort()); err != nil {
		t.Fatal(err)
	}
	in, meta := bt.b.in.T.Buf(), bt.b.meta.T.Buf()
	ctxB, cancelB := context.WithCancel(context.Background())
	defer cancelB()
	var events []string
	var emitted, live0, live2 []int
	var inWritesAtDone int
	d.writeLog = nil
	d.onWrite = func(dst compute.Ink2Ref) {
		if dst.T.Buf() == in {
			cancelB()
		}
	}
	reqs := cohort()
	reqs[0].Emit = func(tok int) bool { live0 = append(live0, tok); return true }
	reqs[0].Done = func(Inkling2LoopBatchResult) { events = append(events, "live done") }
	reqs[1].Context = ctxB
	reqs[1].Emit = func(tok int) bool { emitted = append(emitted, tok); return true }
	reqs[1].Done = func(r Inkling2LoopBatchResult) {
		events = append(events, "cancelled done")
		inWritesAtDone = 0
		for _, w := range d.writeLog {
			if w.buf == in {
				inWritesAtDone++
			}
		}
		if !errors.Is(r.Err, context.Canceled) || r.Generated != 0 {
			t.Errorf("cancelled row result %+v", r)
		}
	}
	reqs[2].Emit = func(tok int) bool { live2 = append(live2, tok); return true }
	reqs[2].Done = func(Inkling2LoopBatchResult) { events = append(events, "live done") }
	results, err := bt.Generate(reqs)
	d.onWrite = nil
	if err != nil {
		t.Fatal(err)
	}
	if inWritesAtDone != 1 {
		t.Fatalf("cancelled row finished after %d prefill chunks, want 1", inWritesAtDone)
	}
	if len(events) == 0 || events[0] != "cancelled done" {
		t.Fatalf("finish order %v: the cancelled row must finish before any live row", events)
	}
	if len(emitted) != 0 {
		t.Fatalf("cancelled row emitted %v", emitted)
	}
	// prefill chunk writes come before the first decode step's writes; a step writes its
	// token row to in just before its meta row, so that one write is dropped
	var chunkTokens []int
	for _, w := range d.writeLog {
		if w.buf == meta {
			break
		}
		if w.buf == in {
			chunkTokens = append(chunkTokens, w.n/bt.b.s.hp.Hidden)
		}
	}
	chunkTokens = chunkTokens[:len(chunkTokens)-1]
	// one chunk shared by the short row and the cancelled row, then the live long row alone
	if want := 1 + (long+inkling2PrefillChunk-1)/inkling2PrefillChunk; len(chunkTokens) != want {
		t.Fatalf("%d prefill chunks %v, want %d: the cancelled row's prompt was packed", len(chunkTokens), chunkTokens, want)
	}
	tokensAfterFirst := 0
	for _, n := range chunkTokens[1:] {
		tokensAfterFirst += n
	}
	if tokensAfterFirst != long {
		t.Fatalf("%d prompt tokens written after the cancel, want %d (the live row alone)", tokensAfterFirst, long)
	}
	if !errors.Is(results[1].Err, context.Canceled) {
		t.Fatalf("cancelled row result %+v", results[1])
	}
	if want := []int{20, 21}; !reflect.DeepEqual(live0, want) {
		t.Fatalf("short live row emitted %v, want %v", live0, want)
	}
	if last := 2000 + long - 1; !reflect.DeepEqual(live2, []int{last + 1, last + 2}) {
		t.Fatalf("long live row emitted %v, want %v", live2, []int{last + 1, last + 2})
	}
}

// A row whose prompt ends in an early prefill chunk gets its first token at that
// chunk, before a longer row's remaining chunks run.
func TestInkling2CohortFirstTokenBeforeLongRowPrefill(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	cohort := func(onShort func(int) bool) []Inkling2LoopBatchRequest {
		return []Inkling2LoopBatchRequest{
			{Prompt: fakeInk2CohortPrompt(20, 0), MaxTokens: 2, Emit: onShort},
			{Prompt: fakeInk2CohortPrompt(1200, 1), MaxTokens: 2, Emit: func(int) bool { return true }},
		}
	}
	if _, err := bt.Generate(cohort(func(int) bool { return true })); err != nil {
		t.Fatal(err)
	}
	in := bt.b.in.T.Buf()
	inWrites, atFirst := 0, -1
	d.onWrite = func(dst compute.Ink2Ref) {
		if dst.T.Buf() == in {
			inWrites++
		}
	}
	var short []int
	results, err := bt.Generate(cohort(func(tok int) bool {
		if atFirst < 0 {
			atFirst = inWrites
		}
		short = append(short, tok)
		return true
	}))
	d.onWrite = nil
	if err != nil {
		t.Fatal(err)
	}
	if atFirst != 1 {
		t.Fatalf("short row's first token arrived after %d prefill chunks, want 1", atFirst)
	}
	if !reflect.DeepEqual(short, []int{20, 21}) || results[0].Generated != 2 {
		t.Fatalf("short row emitted %v, result %+v, want [20 21]", short, results[0])
	}
}

// ink2Event is one callback of a mixed-prefill cohort: a token emitted by row, or
// tok -1 for the Done of that row.
type ink2Event struct{ row, tok int }

// fakeInk2MixedCohort is a short row whose prompt ends in the first prefill chunk and
// a long row that prefills for three chunks, both recording their callbacks in order.
func fakeInk2MixedCohort(longCtx context.Context, events *[]ink2Event, got [][]int) []Inkling2LoopBatchRequest {
	return fakeInk2MixedCohortOrder(longCtx, events, got, false)
}

// fakeInk2MixedCohortOrder is fakeInk2MixedCohort with the long row first or second in
// the cohort; events and got are keyed by row identity (0 short, 1 long), not position.
func fakeInk2MixedCohortOrder(longCtx context.Context, events *[]ink2Event, got [][]int, longFirst bool) []Inkling2LoopBatchRequest {
	const long = 3 * inkling2PrefillChunk
	row := func(i int) func(int) bool {
		return func(tok int) bool {
			got[i] = append(got[i], tok)
			*events = append(*events, ink2Event{i, tok})
			return true
		}
	}
	done := func(i int) func(Inkling2LoopBatchResult) {
		return func(Inkling2LoopBatchResult) { *events = append(*events, ink2Event{i, -1}) }
	}
	short := Inkling2LoopBatchRequest{Prompt: fakeInk2CohortPrompt(20, 0), MaxTokens: 6, Emit: row(0), Done: done(0)}
	longReq := Inkling2LoopBatchRequest{Context: longCtx, Prompt: fakeInk2CohortPrompt(long, 1), MaxTokens: 2, Emit: row(1), Done: done(1)}
	if longFirst {
		return []Inkling2LoopBatchRequest{longReq, short}
	}
	return []Inkling2LoopBatchRequest{short, longReq}
}

// fakeInk2ScalarTokens is what one request emits on the single-sequence decode.
func fakeInk2ScalarTokens(t *testing.T, prompt []int, maxNew int) []int {
	t.Helper()
	var want []int
	turn := Inkling2LoopTurn{MaxTokens: maxNew, Emit: func(tok int) bool { want = append(want, tok); return true }}
	if _, err := Inkling2LoopSessionDecode(context.Background(), fakeInk2Session{}, prompt, &turn); err != nil {
		t.Fatal(err)
	}
	return want
}

// fakeInk2ChunkTokens returns the token count of each prefill chunk a cohort wrote
// before its first lockstep step; the input write of that step is dropped.
func fakeInk2ChunkTokens(t *testing.T, d *fakeInk2Cohort, in, meta compute.Buffer, hidden int) []int {
	t.Helper()
	var chunks []int
	for _, w := range d.writeLog {
		if w.buf == meta {
			break
		}
		if w.buf == in {
			chunks = append(chunks, w.n/hidden)
		}
	}
	if len(chunks) == 0 {
		t.Fatal("no prefill chunk wrote the input arena")
	}
	return chunks[:len(chunks)-1]
}

// With the long-prefill cap set, a row whose prompt has ended decodes in the chunks
// that still prefill the long row: its whole budget is emitted before the long row
// first token, and while it decodes each chunk carries at most the cap of prompt tokens.
func TestInkling2CohortMixedPrefillDecodesShortRowDuringLongPrefill(t *testing.T) {
	const capTok = 64
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", strconv.Itoa(capTok))
	const long, shortLen, shortMax = 3 * inkling2PrefillChunk, 20, 6
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if _, err := bt.Generate(fakeInk2MixedCohort(nil, &[]ink2Event{}, make([][]int, 2))); err != nil {
		t.Fatal(err)
	}
	in, meta := bt.b.in.T.Buf(), bt.b.meta.T.Buf()
	var events []ink2Event
	got := make([][]int, 2)
	d.writeLog = nil
	results, err := bt.Generate(fakeInk2MixedCohort(nil, &events, got))
	if err != nil {
		t.Fatal(err)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(shortLen, 0), shortMax); !reflect.DeepEqual(got[0], want) {
		t.Fatalf("short row emitted %v, scalar %v", got[0], want)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(long, 1), 2); !reflect.DeepEqual(got[1], want) {
		t.Fatalf("long row emitted %v, scalar %v", got[1], want)
	}
	if results[0].Generated != shortMax || results[1].Generated != 2 {
		t.Fatalf("generated %d and %d, want %d and 2", results[0].Generated, results[1].Generated, shortMax)
	}
	firstLong, shortTokens, shortDone := -1, 0, -1
	for k, e := range events {
		if e.row == 1 && firstLong < 0 {
			firstLong = k
		}
		if e.row == 0 && firstLong < 0 {
			if e.tok < 0 {
				shortDone = k
			} else {
				shortTokens++
			}
		}
	}
	if firstLong < 0 || shortTokens != shortMax || shortDone < 0 {
		t.Fatalf("events %v: short row must finish (%d tokens) before the long row first token", events, shortTokens)
	}
	chunks := fakeInk2ChunkTokens(t, d, in, meta, bt.b.s.hp.Hidden)
	// the short row decodes in chunks 1..shortMax-1, each one decode token beside the
	// cap of prompt tokens, or fewer if the chunk has less room left than the cap
	wantChunk := min(capTok, inkling2PrefillChunk-1) + 1
	for k := 1; k < shortMax; k++ {
		if chunks[k] != wantChunk {
			t.Fatalf("chunk %d carried %d tokens %v, want %d while the short row decodes", k, chunks[k], chunks, wantChunk)
		}
	}
	total := 0
	for _, n := range chunks {
		total += n
	}
	if want := long + shortLen + shortMax - 1; total != want {
		t.Fatalf("chunks carried %d tokens, want %d (every prompt once, one decode token per short-row feed)", total, want)
	}
}

// With the long row first in the cohort, the short row still gets its prompt into the
// first chunk: its first token arrives after that chunk, and all of its tokens arrive
// before the long row's first token.
func TestInkling2CohortMixedPrefillLongRowFirstShortRowNotStarved(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	const shortMax = 6
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if _, err := bt.Generate(fakeInk2MixedCohortOrder(nil, &[]ink2Event{}, make([][]int, 2), true)); err != nil {
		t.Fatal(err)
	}
	in := bt.b.in.T.Buf()
	writes, shortFirstAt := 0, -1
	d.onWrite = func(dst compute.Ink2Ref) {
		if dst.T.Buf() == in {
			writes++
		}
	}
	var events []ink2Event
	got := make([][]int, 2)
	reqs := fakeInk2MixedCohortOrder(nil, &events, got, true)
	for i := range reqs {
		if len(reqs[i].Prompt) == 20 {
			emit := reqs[i].Emit
			reqs[i].Emit = func(tok int) bool {
				if shortFirstAt < 0 {
					shortFirstAt = writes
				}
				return emit(tok)
			}
		}
	}
	d.writeLog = nil
	results, err := bt.Generate(reqs)
	d.onWrite = nil
	if err != nil {
		t.Fatal(err)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(20, 0), shortMax); !reflect.DeepEqual(got[0], want) {
		t.Fatalf("short row emitted %v, scalar %v", got[0], want)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(3*inkling2PrefillChunk, 1), 2); !reflect.DeepEqual(got[1], want) {
		t.Fatalf("long row emitted %v, scalar %v", got[1], want)
	}
	// results follow request order: the long row is first here
	if results[1].Generated != shortMax || results[0].Generated != 2 {
		t.Fatalf("generated long %d and short %d, want 2 and %d", results[0].Generated, results[1].Generated, shortMax)
	}
	if shortFirstAt != 1 {
		t.Fatalf("short row first token after %d prefill chunks, want 1", shortFirstAt)
	}
	firstLong, shortBefore, shortDoneBefore := -1, 0, false
	for k, e := range events {
		if e.row == 1 && firstLong < 0 {
			firstLong = k
		}
		if e.row == 0 && firstLong < 0 {
			if e.tok < 0 {
				shortDoneBefore = true
			} else {
				shortBefore++
			}
		}
	}
	if firstLong < 0 || shortBefore != shortMax || !shortDoneBefore {
		t.Fatalf("events %v: short row must emit %d tokens and finish before the long row first token", events, shortMax)
	}
}

// Without the cap the cohort prefills every prompt before any lockstep step, so the
// long row first token precedes the short row second: today order.
func TestInkling2CohortLongPrefillWaitsWithoutCap(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if _, err := bt.Generate(fakeInk2MixedCohort(nil, &[]ink2Event{}, make([][]int, 2))); err != nil {
		t.Fatal(err)
	}
	var events []ink2Event
	got := make([][]int, 2)
	if _, err := bt.Generate(fakeInk2MixedCohort(nil, &events, got)); err != nil {
		t.Fatal(err)
	}
	firstLong, secondShort, shortTokens := -1, -1, 0
	for k, e := range events {
		if e.row == 1 && e.tok >= 0 && firstLong < 0 {
			firstLong = k
		}
		if e.row == 0 && e.tok >= 0 {
			shortTokens++
			if shortTokens == 2 {
				secondShort = k
			}
		}
	}
	if firstLong < 0 || secondShort < 0 || firstLong > secondShort {
		t.Fatalf("events %v: without the cap the long row first token must precede the short row second", events)
	}
}

// Every parity case with a long row beside it emits what the single-sequence decode
// emits, with the cap set and the mixed chunks running.
func TestInkling2CohortMixedPrefillMatchesScalarParity(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	const long = 3 * inkling2PrefillChunk
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	cases := inkling2ParityCases
	longRow := len(cases)
	got := make([][]int, len(cases)+1)
	reqs := make([]Inkling2LoopBatchRequest, 0, len(cases)+1)
	for i, c := range cases {
		stop, emit := c.callbacks(&got[i])
		reqs = append(reqs, Inkling2LoopBatchRequest{Prompt: c.prompt, MaxTokens: c.maxNew, Stop: stop, Emit: emit})
	}
	reqs = append(reqs, Inkling2LoopBatchRequest{Prompt: fakeInk2CohortPrompt(long, 9), MaxTokens: 3,
		Emit: func(tok int) bool { got[longRow] = append(got[longRow], tok); return true }})
	results, err := bt.Generate(reqs)
	if err != nil {
		t.Fatal(err)
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
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(long, 9), 3); !reflect.DeepEqual(got[longRow], want) {
		t.Errorf("long row emitted %v, scalar %v", got[longRow], want)
	}
}

// A long row cancelled while the mixed chunks run is dropped at the next chunk
// boundary with its context error, and the short row tokens are unchanged.
func TestInkling2CohortMixedPrefillDropsRowCancelledMidMixed(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	if _, err := bt.Generate(fakeInk2MixedCohort(nil, &[]ink2Event{}, make([][]int, 2))); err != nil {
		t.Fatal(err)
	}
	in, meta := bt.b.in.T.Buf(), bt.b.meta.T.Buf()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	writes := 0
	d.onWrite = func(dst compute.Ink2Ref) {
		if dst.T.Buf() == in {
			writes++
			if writes == 2 {
				cancel()
			}
		}
	}
	var events []ink2Event
	got := make([][]int, 2)
	d.writeLog = nil
	results, err := bt.Generate(fakeInk2MixedCohort(ctx, &events, got))
	d.onWrite = nil
	if err != nil {
		t.Fatal(err)
	}
	if !errors.Is(results[1].Err, context.Canceled) || results[1].Generated != 0 {
		t.Fatalf("cancelled row result %+v", results[1])
	}
	dones := [2]int{}
	for _, e := range events {
		if e.row == 1 && e.tok >= 0 {
			t.Fatalf("cancelled row emitted %d", e.tok)
		}
		if e.tok < 0 {
			dones[e.row]++
		}
	}
	if dones != [2]int{1, 1} {
		t.Fatalf("Done fired %v times per row (short, long), want once each", dones)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(20, 0), 6); !reflect.DeepEqual(got[0], want) {
		t.Fatalf("short row emitted %v, scalar %v", got[0], want)
	}
	// the chunk that ended the short prompt and one mixed chunk ran; the remaining
	// prompt of the cancelled row was never packed
	if chunks := fakeInk2ChunkTokens(t, d, in, meta, bt.b.s.hp.Hidden); len(chunks) != 2 {
		t.Fatalf("%d prefill chunks %v ran, want 2: the cancelled row prompt was packed", len(chunks), chunks)
	}
}

// fakeInk2AdmitRows builds the two-row cohort of the admission tests: row 0 ends after
// two tokens and row 1 runs budget tokens, both on short prompts in the mixed driver.
// events records each emit (tok >= 0) and each Done (tok -1) in callback order.
func fakeInk2AdmitRows(budget int, events *[]ink2Event, got [][]int) []Inkling2LoopBatchRequest {
	row := func(i int) func(int) bool {
		return func(tok int) bool {
			got[i] = append(got[i], tok)
			*events = append(*events, ink2Event{i, tok})
			return true
		}
	}
	done := func(i int) func(Inkling2LoopBatchResult) {
		return func(Inkling2LoopBatchResult) { *events = append(*events, ink2Event{i, -1}) }
	}
	return []Inkling2LoopBatchRequest{
		{Prompt: fakeInk2CohortPrompt(5, 0), MaxTokens: 2, Emit: row(0), Done: done(0)},
		{Prompt: fakeInk2CohortPrompt(5, 1), MaxTokens: budget, Emit: row(1), Done: done(1)},
	}
}

// A request queued while the cohort runs joins the slot of the row that finished first,
// decodes exactly what the scalar decode emits, and finishes before the cohort's long row:
// it does not wait for the cohort to drain.
func TestInkling2CohortAdmitsQueuedRowIntoFinishedSlot(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	const budgetB, budgetC = 40, 5
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	var events []ink2Event
	got := make([][]int, 3)
	reqs := fakeInk2AdmitRows(budgetB, &events, got)
	queued := []Inkling2LoopBatchRequest{{Prompt: fakeInk2CohortPrompt(7, 2), MaxTokens: budgetC,
		Emit: func(tok int) bool {
			got[2] = append(got[2], tok)
			events = append(events, ink2Event{2, tok})
			return true
		},
		Done: func(Inkling2LoopBatchResult) { events = append(events, ink2Event{2, -1}) }}}
	admit := func(int) (Inkling2LoopBatchRequest, bool) {
		if len(queued) == 0 {
			return Inkling2LoopBatchRequest{}, false
		}
		r := queued[0]
		queued = queued[1:]
		return r, true
	}
	results, err := bt.GenerateAdmitting(reqs, admit)
	if err != nil {
		t.Fatal(err)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(7, 2), budgetC); !reflect.DeepEqual(got[2], want) {
		t.Fatalf("admitted row emitted %v, scalar %v", got[2], want)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(5, 1), budgetB); !reflect.DeepEqual(got[1], want) {
		t.Fatalf("long row emitted %v, scalar %v", got[1], want)
	}
	if results[0].Generated != 2 || results[1].Generated != budgetB || results[0].Err != nil || results[1].Err != nil {
		t.Fatalf("results %+v", results)
	}
	doneAt := map[int]int{}
	for k, e := range events {
		if e.tok < 0 {
			if _, dup := doneAt[e.row]; dup {
				t.Fatalf("Done fired twice for row %d: %v", e.row, events)
			}
			doneAt[e.row] = k
		}
	}
	if len(doneAt) != 3 {
		t.Fatalf("Done fired for rows %v, want all three", doneAt)
	}
	if doneAt[2] > doneAt[1] {
		t.Fatalf("admitted row finished at event %d, after the long row at %d: it waited for the cohort", doneAt[2], doneAt[1])
	}
}

// A request whose prompt and budget exceed the cohort's KV capacity is never admitted,
// even when admit hands it back: its Done gets an error, it emits nothing, and the
// cohort's other rows are unchanged.
func TestInkling2CohortNeverAdmitsRequestPastCohortContext(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	const budgetB = 40
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	var events []ink2Event
	got := make([][]int, 3)
	reqs := fakeInk2AdmitRows(budgetB, &events, got)
	var bounds []int
	var tooLong *Inkling2LoopBatchResult
	emitted := false
	queued := []Inkling2LoopBatchRequest{{Prompt: fakeInk2CohortPrompt(60, 2), MaxTokens: 10,
		Emit: func(int) bool { emitted = true; return true },
		Done: func(r Inkling2LoopBatchResult) { tooLong = &r }}}
	admit := func(maxPositions int) (Inkling2LoopBatchRequest, bool) {
		bounds = append(bounds, maxPositions)
		if len(queued) == 0 {
			return Inkling2LoopBatchRequest{}, false
		}
		r := queued[0]
		queued = queued[1:]
		return r, true
	}
	results, err := bt.GenerateAdmitting(reqs, admit)
	if err != nil {
		t.Fatal(err)
	}
	if tooLong == nil || tooLong.Err == nil || tooLong.Generated != 0 {
		t.Fatalf("too-long request result %+v, want an error and no tokens", tooLong)
	}
	if emitted {
		t.Fatal("too-long request emitted a token")
	}
	for _, b := range bounds {
		if b != bt.b.ctx {
			t.Fatalf("admit saw %d positions, want the cohort capacity %d", b, bt.b.ctx)
		}
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(5, 1), budgetB); !reflect.DeepEqual(got[1], want) {
		t.Fatalf("long row emitted %v, scalar %v", got[1], want)
	}
	if results[1].Generated != budgetB || results[1].Err != nil {
		t.Fatalf("long row result %+v", results[1])
	}
}

// With the cap unset the cohort never calls admit, and the rows run as they did before.
func TestInkling2CohortKnobOffNeverCallsAdmit(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	var events []ink2Event
	got := make([][]int, 2)
	calls := 0
	admit := func(int) (Inkling2LoopBatchRequest, bool) {
		calls++
		return Inkling2LoopBatchRequest{Prompt: []int{1}, MaxTokens: 1, Emit: func(int) bool { return true }}, true
	}
	results, err := bt.GenerateAdmitting(fakeInk2AdmitRows(40, &events, got), admit)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("admit called %d times with the cap unset", calls)
	}
	if want := fakeInk2ScalarTokens(t, fakeInk2CohortPrompt(5, 1), 40); !reflect.DeepEqual(got[1], want) || results[1].Generated != 40 {
		t.Fatalf("long row emitted %v (generated %d), scalar %v", got[1], results[1].Generated, want)
	}
}

// A joined row starts from empty conv history: resetRow zeroes that row's history in
// every slot and leaves the other rows and the row's own position bookkeeping alone.
// The host fake model has no conv slots, so this drives the batch state directly.
func TestInkling2BatchResetRowClearsOnlyThatRowsHistory(t *testing.T) {
	d := newFakeInk2Cohort()
	const C, stride, slots = 3, 10, 2
	b := &inkling2Batch{s: &inkling2Resident{dev: d}, C: C, histStride: stride, pos: []int{4, 5, 6}}
	for slot := 0; slot < slots; slot++ {
		tensor, err := d.Ink2Alloc(C * stride)
		if err != nil {
			t.Fatal(err)
		}
		b.hist = append(b.hist, compute.Ink2At(tensor, 0))
		for j := range d.rows(b.hist[slot])[:C*stride] {
			d.rows(b.hist[slot])[j] = 7
		}
	}
	if err := b.resetRow(1); err != nil {
		t.Fatal(err)
	}
	if b.pos[1] != 0 || b.pos[0] != 4 || b.pos[2] != 6 {
		t.Fatalf("positions %v after resetting row 1, want [4 0 6]", b.pos)
	}
	for slot := range b.hist {
		row := d.rows(b.hist[slot])
		for j := 0; j < C*stride; j++ {
			want := float32(7)
			if j >= stride && j < 2*stride {
				want = 0
			}
			if row[j] != want {
				t.Fatalf("slot %d element %d = %v, want %v", slot, j, row[j], want)
			}
		}
	}
}

// fakeInk2SharedPrompt is prefix tokens every row shares followed by suffix tokens of
// the row's own.
func fakeInk2SharedPrompt(prefix, suffix, row int) []int {
	p := make([]int, 0, prefix+suffix)
	for j := 0; j < prefix; j++ {
		p = append(p, 70000+j)
	}
	for j := 0; j < suffix; j++ {
		p = append(p, 1000*row+j)
	}
	return p
}

// fakeInk2SharedRun is one cohort run under a shared-prefix knob value: every row's
// emitted tokens and result, the prompt tokens of each prefill chunk, and the calls.
type fakeInk2SharedRun struct {
	got     [][]int
	results []Inkling2LoopBatchResult
	chunks  []int
	log     []fakeInk2Write
	heads   int
	reads   int
}

func (r fakeInk2SharedRun) prefilled() int {
	n := 0
	for _, c := range r.chunks {
		n += c
	}
	return n
}

// runFakeInk2Shared runs prompts as one cohort on a warmed state with the knob set to
// knob ("" unsets it); cancel lists rows whose context is cancelled before the cohort
// starts, and cancelAtChunk, when not -1, cancels that row as the first prefill chunk
// is written.
func runFakeInk2Shared(t *testing.T, knob string, prompts [][]int, maxNew int, cancel []int, cancelAtChunk int) fakeInk2SharedRun {
	t.Helper()
	t.Setenv("FAK_INKLING2_COHORT_SHARED_PREFIX", knob)
	if knob == "" {
		os.Unsetenv("FAK_INKLING2_COHORT_SHARED_PREFIX")
	}
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	reqs := func(got [][]int, ctxs []context.Context) []Inkling2LoopBatchRequest {
		out := make([]Inkling2LoopBatchRequest, len(prompts))
		for i, p := range prompts {
			out[i] = Inkling2LoopBatchRequest{Context: ctxs[i], Prompt: p, MaxTokens: maxNew,
				Emit: func(tok int) bool { got[i] = append(got[i], tok); return true }}
		}
		return out
	}
	bg := make([]context.Context, len(prompts))
	for i := range bg {
		bg[i] = context.Background()
	}
	if _, err := bt.Generate(reqs(make([][]int, len(prompts)), bg)); err != nil {
		t.Fatal(err)
	}
	in, meta := bt.b.in.T.Buf(), bt.b.meta.T.Buf()
	ctxs := append([]context.Context(nil), bg...)
	for _, i := range cancel {
		c, stop := context.WithCancel(context.Background())
		stop()
		ctxs[i] = c
	}
	var stopAtChunk context.CancelFunc
	if cancelAtChunk >= 0 {
		c, stop := context.WithCancel(context.Background())
		defer stop()
		ctxs[cancelAtChunk], stopAtChunk = c, stop
	}
	d.writeLog, d.heads, d.reads = nil, 0, 0
	d.onWrite = func(dst compute.Ink2Ref) {
		if stopAtChunk != nil && dst.T.Buf() == in {
			stopAtChunk()
		}
	}
	run := fakeInk2SharedRun{got: make([][]int, len(prompts))}
	results, err := bt.Generate(reqs(run.got, ctxs))
	d.onWrite = nil
	if err != nil {
		t.Fatal(err)
	}
	run.results, run.log, run.heads, run.reads = results, d.writeLog, d.heads, d.reads
	run.chunks = fakeInk2ChunkTokens(t, d, in, meta, bt.b.s.hp.Hidden)
	return run
}

// With the shared-prefix knob on, a cohort whose rows share a P-token prefix prefills
// that prefix once: P + the suffixes instead of every row's whole prompt.
func TestInkling2CohortSharedPrefixPrefillsPrefixOnce(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	const P = inkling2PrefillChunk + 88
	suffixes := []int{5, 1, 40, 700}
	prompts := make([][]int, len(suffixes))
	sum := 0
	for i, s := range suffixes {
		prompts[i] = fakeInk2SharedPrompt(P, s, i)
		sum += s
	}
	off := runFakeInk2Shared(t, "", prompts, 3, nil, -1)
	on := runFakeInk2Shared(t, "64", prompts, 3, nil, -1)
	if got, want := off.prefilled(), len(prompts)*P+sum; got != want {
		t.Fatalf("knob off prefilled %d tokens, want %d", got, want)
	}
	if got, want := on.prefilled(), P+sum; got != want {
		t.Fatalf("knob on prefilled %d tokens %v, want P + suffixes = %d", got, on.chunks, want)
	}
	if !reflect.DeepEqual(on.got, off.got) {
		t.Fatalf("knob on emitted %v, off %v", on.got, off.got)
	}
	// a shared prefix shorter than the knob is not worth the copy
	short := runFakeInk2Shared(t, strconv.Itoa(P+1), prompts, 3, nil, -1)
	if got, want := short.prefilled(), len(prompts)*P+sum; got != want {
		t.Fatalf("prefix below the minimum prefilled %d tokens, want %d", got, want)
	}
}

// Every row emits the same tokens and ends the same way with the knob on as off. The
// host fake's next token depends only on the fed token, so the copied state itself is
// checked by TestInkling2BatchCopyRowStateCopiesOnlyTargetRow.
func TestInkling2CohortSharedPrefixMatchesKnobOff(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	shared := fakeInk2SharedPrompt(700, 0, 0)
	cases := []struct {
		name          string
		prompts       [][]int
		cancel        []int
		cancelAtChunk int
		// saved is the prefill tokens the knob saves when no row is cancelled
		saved int
	}{
		{name: "no shared prefix", prompts: [][]int{fakeInk2CohortPrompt(100, 0), fakeInk2CohortPrompt(100, 1), fakeInk2CohortPrompt(90, 2)}, cancelAtChunk: -1},
		{name: "identical prompts", prompts: [][]int{shared[:300], shared[:300], shared[:300]}, cancelAtChunk: -1, saved: 2 * 299},
		{name: "one row a strict prefix of another", prompts: [][]int{shared[:200], shared[:650]}, cancelAtChunk: -1, saved: 199},
		{name: "pre-cancelled row", prompts: [][]int{fakeInk2SharedPrompt(120, 3, 0), fakeInk2SharedPrompt(120, 9, 1), fakeInk2SharedPrompt(120, 4, 2)},
			cancel: []int{1}, cancelAtChunk: -1},
		{name: "row cancelled mid shared prefix", prompts: [][]int{fakeInk2SharedPrompt(600, 3, 0), fakeInk2SharedPrompt(600, 9, 1), fakeInk2SharedPrompt(600, 4, 2)},
			cancelAtChunk: 1},
		{name: "one-token suffix", prompts: [][]int{fakeInk2SharedPrompt(150, 1, 0), fakeInk2SharedPrompt(150, 30, 1)}, cancelAtChunk: -1, saved: 150},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			off := runFakeInk2Shared(t, "", c.prompts, 4, c.cancel, c.cancelAtChunk)
			on := runFakeInk2Shared(t, "16", c.prompts, 4, c.cancel, c.cancelAtChunk)
			if !reflect.DeepEqual(on.got, off.got) {
				t.Fatalf("knob on emitted %v, off %v", on.got, off.got)
			}
			for i := range c.prompts {
				a, b := on.results[i], off.results[i]
				if a.Generated != b.Generated || a.Stopped != b.Stopped || (a.Err == nil) != (b.Err == nil) || (b.Err != nil && !errors.Is(a.Err, b.Err)) {
					t.Fatalf("row %d: knob on %+v, off %+v", i, a, b)
				}
			}
			if got := off.prefilled() - on.prefilled(); c.saved > 0 && got != c.saved {
				t.Fatalf("knob on saved %d prefill tokens (on %v, off %v), want %d", got, on.chunks, off.chunks, c.saved)
			}
			if c.name == "no shared prefix" && !reflect.DeepEqual(on.log, off.log) {
				t.Fatal("no shared prefix, yet the knob changed the write sequence")
			}
		})
	}
}

// With the knob off the cohort prefill packs, writes and reads exactly as before: the
// chunks are today's planner's, no state is read back, and "0" equals unset.
func TestInkling2CohortSharedPrefixKnobOffKeepsCallSequence(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	prompts := [][]int{fakeInk2SharedPrompt(600, 5, 0), fakeInk2SharedPrompt(600, 50, 1), fakeInk2SharedPrompt(600, 1, 2)}
	unset := runFakeInk2Shared(t, "", prompts, 3, nil, -1)
	zero := runFakeInk2Shared(t, "0", prompts, 3, nil, -1)
	if !reflect.DeepEqual(unset.log, zero.log) || unset.heads != zero.heads || unset.reads != zero.reads {
		t.Fatal("knob 0 changed the call sequence against unset")
	}
	lens := make([]int, len(prompts))
	for i, p := range prompts {
		lens[i] = len(p)
	}
	var want []int
	for _, segs := range inkling2PrefillChunks(lens, inkling2PrefillChunk) {
		n := 0
		for _, sg := range segs {
			n = max(n, sg.off+sg.n)
		}
		want = append(want, n)
	}
	if !reflect.DeepEqual(unset.chunks, want) {
		t.Fatalf("knob off chunks %v, want the planner's packing %v", unset.chunks, want)
	}
	if unset.reads != 0 {
		t.Fatalf("knob off read state back %d times", unset.reads)
	}
}

// The mixed driver keeps every row's tokens with the knob on.
func TestInkling2CohortSharedPrefixMixedMatchesKnobOff(t *testing.T) {
	t.Setenv("FAK_INKLING2_COHORT_PREFILL", "1")
	t.Setenv("FAK_NATIVE_LONG_PREFILL_TOKEN_THRESHOLD", "64")
	prompts := [][]int{fakeInk2SharedPrompt(300, 2, 0), fakeInk2SharedPrompt(300, 900, 1), fakeInk2SharedPrompt(300, 20, 2)}
	off := runFakeInk2Shared(t, "", prompts, 5, nil, -1)
	on := runFakeInk2Shared(t, "16", prompts, 5, nil, -1)
	if !reflect.DeepEqual(on.got, off.got) {
		t.Fatalf("mixed knob on emitted %v, off %v", on.got, off.got)
	}
	for i, p := range prompts {
		if want := fakeInk2ScalarTokens(t, p, 5); !reflect.DeepEqual(on.got[i], want) {
			t.Fatalf("row %d emitted %v, scalar %v", i, on.got[i], want)
		}
	}
	if on.prefilled() >= off.prefilled() {
		t.Fatalf("mixed knob on wrote %d chunk tokens, off %d: the shared prefix was not deduped", on.prefilled(), off.prefilled())
	}
}

// fakeInk2Repeat adds the device row copy to the host fake.
type fakeInk2Repeat struct {
	*fakeInk2Cohort
	repeats int
}

func (d *fakeInk2Repeat) Ink2VRepeatRows(x, out compute.Ink2Ref, n, rows, k int) error {
	d.repeats++
	for t := 0; t < rows; t++ {
		for j := 0; j < k; j++ {
			copy(d.rows(out)[(t*k+j)*n:(t*k+j+1)*n], d.rows(x)[t*n:(t+1)*n])
		}
	}
	return nil
}

// copyRowState makes the destination row's caches and conv histories bitwise the
// source row's in every slot, leaves every other row alone, and copies on the device
// when the backend can.
func TestInkling2BatchCopyRowStateCopiesOnlyTargetRow(t *testing.T) {
	for _, device := range []bool{false, true} {
		d := newFakeInk2Cohort()
		rep := &fakeInk2Repeat{fakeInk2Cohort: d}
		var be compute.Backend = d
		if device {
			be = rep
		}
		const C, stride, kvDim = 3, 5, 2
		hp := model.Inkling2LoopHParams{KVHeads: 1, HeadDim: kvDim}
		b := &inkling2Batch{s: &inkling2Resident{be: be, dev: d, hp: hp}, C: C, histStride: stride, caps: []int{4, 2}}
		fill := func(n, tag int) ink2Ref {
			tensor, err := d.Ink2Alloc(n)
			if err != nil {
				t.Fatal(err)
			}
			r := compute.Ink2At(tensor, 0)
			for j := range d.rows(r)[:n] {
				d.rows(r)[j] = float32(100*tag+j) + 0.25
			}
			return r
		}
		for slot, cap := range b.caps {
			b.kc = append(b.kc, fill(C*cap*kvDim, slot))
			b.vc = append(b.vc, fill(C*cap*kvDim, slot+10))
			b.hist = append(b.hist, fill(C*stride, slot+20))
		}
		before := map[compute.Buffer][]float32{}
		for buf, data := range d.data {
			before[buf] = append([]float32(nil), data...)
		}
		if err := b.copyRowState(0, 2); err != nil {
			t.Fatal(err)
		}
		check := func(r ink2Ref, rowLen int) {
			got, was := d.data[r.T.Buf()], before[r.T.Buf()]
			for j := range got {
				want := was[j]
				if j/rowLen == 2 {
					want = was[j%rowLen]
				}
				if got[j] != want {
					t.Fatalf("device=%v element %d = %v, want %v", device, j, got[j], want)
				}
			}
		}
		for slot, cap := range b.caps {
			check(b.kc[slot], cap*kvDim)
			check(b.vc[slot], cap*kvDim)
			check(b.hist[slot], stride)
		}
		if device && (rep.repeats == 0 || d.reads != 0) {
			t.Fatalf("device copy ran %d repeats and %d host reads", rep.repeats, d.reads)
		}
		if !device && d.reads == 0 {
			t.Fatal("host copy never read the source row")
		}
	}
}
