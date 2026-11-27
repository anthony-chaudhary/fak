package ggufload

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeInk2HalfPrefill struct{ notReady error }

func (f fakeInk2HalfPrefill) Ink2PHalfKVReady() error { return f.notReady }

func (fakeInk2HalfPrefill) Ink2PQKVPostHalf(q, r, k, v, qclip, rclip, kcache, vcache compute.Ink2Ref, p compute.Ink2AttnParams, cap, pos0, T int) error {
	return nil
}
func (fakeInk2HalfPrefill) Ink2PAttendHalf(q, r, sinks, relBias, kcache, vcache, out compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap, pos0, T int) error {
	return nil
}

func TestInkling2CohortPrefillKVGate(t *testing.T) {
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	for _, c := range []struct {
		name   string
		env    map[string]string
		half   bool
		be     any
		on     bool
		halfOp bool
	}{
		{"fp32 default", nil, false, struct{}{}, true, false},
		{"fp32 ignores the f16 opt-in", map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "f16"}, false, struct{}{}, true, false},
		{"fp32 ignores any knob value", map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "bf16"}, false, struct{}{}, true, false},
		{"fp32 lockstep knob", map[string]string{"FAK_INKLING2_COHORT_PREFILL": "0"}, false, struct{}{}, false, false},
		{"f16 without opt-in stays lockstep", nil, true, fakeInk2HalfPrefill{}, false, false},
		{"f16 explicit off", map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "off"}, true, fakeInk2HalfPrefill{}, false, false},
		{"f16 opt-in", map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "f16"}, true, fakeInk2HalfPrefill{}, true, true},
		{"f16 opt-in any case", map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": " FP16 "}, true, fakeInk2HalfPrefill{}, true, true},
		{"lockstep knob wins over opt-in", map[string]string{"FAK_INKLING2_COHORT_PREFILL": "0", "FAK_INKLING2_COHORT_PREFILL_KV": "f16"}, true, fakeInk2HalfPrefill{}, false, false},
	} {
		on, half, err := inkling2CohortPrefillKV(c.be, c.half, env(c.env))
		if err != nil || on != c.on || (half != nil) != c.halfOp {
			t.Errorf("%s: on %v half ops %v err %v, want on %v half ops %v", c.name, on, half != nil, err, c.on, c.halfOp)
		}
	}
	if _, _, err := inkling2CohortPrefillKV(fakeInk2HalfPrefill{}, true, env(map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "bf16"})); err == nil ||
		!strings.Contains(err.Error(), "FAK_INKLING2_COHORT_PREFILL_KV") {
		t.Errorf("unknown cache type: err %v, want a refusal naming the knob", err)
	}
	if _, _, err := inkling2CohortPrefillKV(struct{}{}, true, env(map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "f16"})); err == nil {
		t.Error("f16 opt-in on a backend without f16 prefill ops must refuse, not feed prompts lockstep")
	}
	notReady := fakeInk2HalfPrefill{notReady: errors.New("float attend selected")}
	if _, _, err := inkling2CohortPrefillKV(notReady, true, env(map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "f16"})); err == nil ||
		!strings.Contains(err.Error(), "float attend selected") {
		t.Errorf("f16 opt-in on a backend whose f16 ops cannot run: err %v, want the refusal at cohort setup", err)
	}
}

// Row i's f16 cache starts i*cap*kvDim elements in, which is half as many floats.
func TestInkling2CohortKVRowOffset(t *testing.T) {
	for _, c := range []struct{ row, cap, kvDim, f32, f16 int }{{0, 64, 1024, 0, 0}, {1, 64, 1024, 65536, 32768}, {3, 576, 1024, 1769472, 884736}} {
		if got := inkling2CohortKVRow(c.row, c.cap, c.kvDim, false); got != c.f32 {
			t.Errorf("fp32 row %d: %d floats, want %d", c.row, got, c.f32)
		}
		if got := inkling2CohortKVRow(c.row, c.cap, c.kvDim, true); got != c.f16 || got != inkling2BatchKVFloats(c.row*c.cap*c.kvDim, true) {
			t.Errorf("f16 row %d: %d floats, want %d", c.row, got, c.f16)
		}
	}
}

// fakeInk2HalfCohort is the cohort fake with the sequence-batched ops f16 caches need
// and the f16 prefill ops; its zero-layer forward never reaches the caches.
type fakeInk2HalfCohort struct {
	*fakeInk2Cohort
	fakeInk2HalfPrefill
}

func (d fakeInk2HalfCohort) Ink2BQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta compute.Ink2Ref, p compute.Ink2AttnParams, cap int, cv *compute.Ink2KVConv, st compute.Ink2SeqStrides, kvHalf bool, C int) error {
	return nil
}
func (d fakeInk2HalfCohort) Ink2BAttend(q, r, sinks, relBias, kcache, vcache, scores, out, meta compute.Ink2Ref, p compute.Ink2AttnParams, extent, window, cap int, st compute.Ink2SeqStrides, kvHalf bool, C int) error {
	return nil
}
func (d fakeInk2HalfCohort) Ink2BConv(x, w, hist, out, resid compute.Ink2Ref, n, kn int, norm float32, st compute.Ink2SeqStrides, C int) error {
	return nil
}

// Ink2BSoftcapArgmax runs the single-row argmax per sequence, so the batched step picks
// the same greedy token the per-sequence path does.
func (d fakeInk2HalfCohort) Ink2BSoftcapArgmax(logits compute.Ink2Ref, n, ld int, softcap float32, outIdx compute.Ink2Ref, C int) error {
	for i := 0; i < C; i++ {
		if err := d.Ink2SoftcapArgmax(compute.Ink2Ref{T: logits.T, Off: logits.Off + i*ld}, n, softcap,
			compute.Ink2Ref{T: outIdx.T, Off: outIdx.Off + 4*i}); err != nil {
			return err
		}
	}
	return nil
}

// With f16 batch caches, FAK_INKLING2_COHORT_PREFILL_KV=f16 keeps the cohort prefill
// (only it runs single-row output heads) instead of lockstep prompt feeding, and the
// tokens are the lockstep ones.
func TestInkling2CohortPrefillRunsOnHalfKVWhenOptedIn(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_KV", "f16")
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "0")
	run := func(optIn string) ([][]int, int, *inkling2Batch) {
		t.Setenv("FAK_INKLING2_COHORT_PREFILL_KV", optIn)
		d := newFakeInk2Cohort()
		bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), fakeInk2HalfCohort{fakeInk2Cohort: d})
		defer bt.Close()
		prompts := [][]int{{3, 4, 5}, {7}, {9, 10}}
		got := make([][]int, len(prompts))
		var reqs []Inkling2LoopBatchRequest
		for i, pr := range prompts {
			i := i
			reqs = append(reqs, Inkling2LoopBatchRequest{Prompt: pr, MaxTokens: 3, Emit: func(tok int) bool { got[i] = append(got[i], tok); return true }})
		}
		if _, err := bt.Generate(reqs); err != nil {
			t.Fatalf("FAK_INKLING2_COHORT_PREFILL_KV=%q: %v", optIn, err)
		}
		return got, d.heads, bt.b
	}
	lock, lockHeads, lb := run("")
	pre, preHeads, pb := run("f16")
	if lockHeads != 0 || lb == nil || lb.prefillT != 0 || lb.half != nil {
		t.Fatalf("f16 caches without the opt-in: %d prefill heads, state %+v; want lockstep prompts", lockHeads, lb)
	}
	if preHeads == 0 || pb == nil || pb.prefillT == 0 || pb.half == nil || !pb.kvHalf {
		t.Fatalf("f16 caches with the opt-in: %d prefill heads; want the cohort prefill on f16 caches", preHeads)
	}
	if !reflect.DeepEqual(pre, lock) {
		t.Fatalf("f16 cohort prefill tokens %v, lockstep %v", pre, lock)
	}
}

func TestInkling2CohortLookupSelectsHalfKVForPlanAndRun(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_KV", "f32")
	t.Setenv("FAK_INKLING2_COHORT_PREFILL_KV", "off")
	t.Setenv("FAK_INKLING2_BATCH_SEQOPS", "0")
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "1")
	settings := map[string]string{
		"FAK_INKLING2_BATCH_KV":          "f16",
		"FAK_INKLING2_COHORT_PREFILL_KV": "f16",
		"FAK_INKLING2_BATCH_SEQOPS":      "1",
	}
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohortsWithLookup(fakeInk2CohortModel(), fakeInk2HalfCohort{fakeInk2Cohort: d}, func(key string) (string, bool) {
		v, ok := settings[key]
		return v, ok
	})
	defer bt.Close()

	prompts := [][]int{{3, 4, 5}, {7}, {9, 10}}
	const maxNew = 3
	ink2SplitBudget(t, bt, d, len(prompts), inkling2BatchCtxAlign)
	groups, err := bt.Split(ink2SplitPositions(prompts, maxNew))
	if err != nil || len(groups) != 1 || len(groups[0]) != len(prompts) {
		t.Fatalf("split = %v, err = %v, want one full-width cohort", groups, err)
	}
	shape, err := bt.s.batchShape(len(prompts), inkling2BatchCtxAlign)
	if err != nil || !shape.kvHalf || shape.half == nil || shape.seq == nil || shape.prefillT == 0 {
		t.Fatalf("planned shape = %+v, err = %v, want f16 cohort prefill with sequence ops", shape, err)
	}
	ink2SplitRun(t, bt, prompts, groups[0], maxNew, make([][]int, len(prompts)))
	if b := bt.b; !b.kvHalf || b.half == nil || b.seq == nil || b.prefillT != shape.prefillT || b.need != shape.need || d.heads == 0 {
		t.Fatalf("allocated cohort does not match the f16 plan or ran without cohort prefill")
	}
}

// On f16 caches the shared-prefix row copy moves one row of cap*kvDim halves, which is
// cap*kvDim/2 floats, and leaves every other row alone.
func TestInkling2BatchCopyRowStateHalfKV(t *testing.T) {
	d := newFakeInk2Cohort()
	const C, stride, kvDim = 3, 5, 4
	hp := model.Inkling2LoopHParams{KVHeads: 1, HeadDim: kvDim}
	b := &inkling2Batch{s: &inkling2Resident{be: d, dev: d, hp: hp}, C: C, histStride: stride, caps: []int{4, 2}, kvHalf: true}
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
		b.kc = append(b.kc, fill(inkling2BatchKVFloats(C*cap*kvDim, true), slot))
		b.vc = append(b.vc, fill(inkling2BatchKVFloats(C*cap*kvDim, true), slot+10))
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
				t.Fatalf("element %d of a %d-float row = %v, want %v", j, rowLen, got[j], want)
			}
		}
	}
	for slot, cap := range b.caps {
		check(b.kc[slot], cap*kvDim/2)
		check(b.vc[slot], cap*kvDim/2)
		check(b.hist[slot], stride)
	}
}
