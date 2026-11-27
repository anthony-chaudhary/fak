package ggufload

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2LoopBatchStreamCohortCtx(t *testing.T) {
	req := func(prompt, maxTok int) Inkling2LoopBatchRequest {
		return Inkling2LoopBatchRequest{Prompt: make([]int, prompt), MaxTokens: maxTok}
	}
	for _, tc := range []struct {
		name     string
		reqs     []Inkling2LoopBatchRequest
		resident int
		want     int
	}{
		{"rounds the longest row up to 64", []Inkling2LoopBatchRequest{req(10, 20), req(100, 1)}, 4096, 128},
		{"exact multiple stays", []Inkling2LoopBatchRequest{req(60, 4), req(1, 1)}, 4096, 64},
		{"short cohort is small", []Inkling2LoopBatchRequest{req(1, 1), req(2, 1)}, 4096, 64},
		{"capped at resident", []Inkling2LoopBatchRequest{req(4000, 97), req(1, 1)}, 4096, 4096},
	} {
		got, err := inkling2BatchCohortCtx(tc.reqs, tc.resident)
		if err != nil || got != tc.want {
			t.Fatalf("%s: ctx=%d err=%v, want %d", tc.name, got, err, tc.want)
		}
	}
	if _, err := inkling2BatchCohortCtx([]Inkling2LoopBatchRequest{req(4000, 98), req(1, 1)}, 4096); err == nil {
		t.Fatal("a row past the resident context must be refused")
	}
}

// The cohort context must satisfy the batch's own position bound and cap rule.
func TestInkling2LoopBatchStreamCohortCtxFitsBatchCaps(t *testing.T) {
	reqs := []Inkling2LoopBatchRequest{{Prompt: make([]int, 37), MaxTokens: 50}, {Prompt: make([]int, 3), MaxTokens: 9}}
	ctx, err := inkling2BatchCohortCtx(reqs, Inkling2LoopResidentContext)
	if err != nil {
		t.Fatal(err)
	}
	if ctx < 1 || ctx > inkling2ResidentCtx {
		t.Fatalf("ctx %d outside newInkling2Batch's 1..%d", ctx, inkling2ResidentCtx)
	}
	// lockstep steps = longest prompt+budget-1; every position must stay below ctx
	if steps := 37 + 50 - 1; steps > ctx {
		t.Fatalf("ctx %d below the %d positions the cohort steps", ctx, steps)
	}
}

func TestInkling2LoopBatchStreamLaneAcceptanceIsRowIsolated(t *testing.T) {
	var first, second []int
	now := time.Now()
	lanes := []inkling2BatchLane{
		newInkling2BatchLane(Inkling2LoopBatchRequest{Context: context.Background(), Prompt: []int{1}, MaxTokens: 4, Stop: func(token int) bool { return token == 9 }, Emit: func(token int) bool { first = append(first, token); return true }}, now),
		newInkling2BatchLane(Inkling2LoopBatchRequest{Context: context.Background(), Prompt: []int{2}, MaxTokens: 1, Emit: func(token int) bool { second = append(second, token); return true }}, now),
	}
	lanes[0].accept(8, now)
	lanes[1].accept(7, now)
	lanes[0].accept(9, now)
	if !reflect.DeepEqual(first, []int{8}) || !reflect.DeepEqual(second, []int{7}) {
		t.Fatalf("row emissions crossed: first=%v second=%v", first, second)
	}
	if !lanes[0].result.Stopped || lanes[0].result.Generated != 1 || !lanes[1].done || lanes[1].result.Generated != 1 {
		t.Fatalf("row states crossed: first=%+v second=%+v", lanes[0].result, lanes[1].result)
	}
}

func TestInkling2LoopBatchStreamCancelledLaneFinishes(t *testing.T) {
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	lane := newInkling2BatchLane(Inkling2LoopBatchRequest{Context: cancelled, Prompt: []int{1}, MaxTokens: 3, Emit: func(int) bool { t.Fatal("emitted on a cancelled row"); return false }}, time.Now())
	lane.accept(5, time.Now())
	if !lane.done || !errors.Is(lane.result.Err, context.Canceled) || lane.result.Generated != 0 {
		t.Fatalf("cancelled lane = %+v", lane.result)
	}
}

func TestInkling2LoopBatchStreamRefusesBadCohort(t *testing.T) {
	if _, err := Inkling2LoopBatchGenerate(nil, nil, make([]Inkling2LoopBatchRequest, 1)); err == nil {
		t.Fatal("a one-row cohort must be refused")
	}
	if _, err := Inkling2LoopBatchGenerate(nil, nil, make([]Inkling2LoopBatchRequest, Inkling2LoopMaxBatch+1)); err == nil {
		t.Fatal("a cohort past the max batch must be refused")
	}
	if Inkling2LoopBatchEligible(nil, nil) {
		t.Fatal("nil model is not batch eligible")
	}
}

func TestInkling2AdmitCheckBoundary(t *testing.T) {
	emit := func(int) bool { return true }
	// len(Prompt) + MaxTokens - 1 == maxPositions must be accepted (matches Generate).
	if err := inkling2AdmitCheck(Inkling2LoopBatchRequest{Prompt: make([]int, 100), MaxTokens: 29, Emit: emit}, 128); err != nil {
		t.Fatalf("exact boundary 100+29-1 == 128 refused: %v", err)
	}
	// One token past the boundary must be refused.
	if err := inkling2AdmitCheck(Inkling2LoopBatchRequest{Prompt: make([]int, 100), MaxTokens: 30, Emit: emit}, 128); err == nil {
		t.Fatal("100+30-1 == 129 > 128 must be refused")
	}
	// Invalid request fields still refused.
	if err := inkling2AdmitCheck(Inkling2LoopBatchRequest{Prompt: nil, MaxTokens: 1, Emit: emit}, 128); err == nil {
		t.Fatal("empty prompt must be refused")
	}
	if err := inkling2AdmitCheck(Inkling2LoopBatchRequest{Prompt: []int{1}, MaxTokens: 0, Emit: emit}, 128); err == nil {
		t.Fatal("zero MaxTokens must be refused")
	}
	if err := inkling2AdmitCheck(Inkling2LoopBatchRequest{Prompt: []int{1}, MaxTokens: 1, Emit: nil}, 128); err == nil {
		t.Fatal("nil Emit must be refused")
	}
}

func TestInkling2CohortsSetMaxContext(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "0")
	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()

	if got := bt.maxContext(); got != Inkling2LoopResidentContext {
		t.Fatalf("default maxContext = %d, want %d", got, Inkling2LoopResidentContext)
	}
	bt.SetMaxContext(16384)
	if got := bt.maxContext(); got != 16384 {
		t.Fatalf("after SetMaxContext(16384): %d", got)
	}
	reqs := []Inkling2LoopBatchRequest{
		{Prompt: make([]int, 8192), MaxTokens: 2, Emit: func(int) bool { return true }},
		{Prompt: make([]int, 16), MaxTokens: 2, Emit: func(int) bool { return true }},
	}
	if _, err := bt.Generate(reqs); err != nil {
		t.Fatalf("Generate at 8192 with SetMaxContext(16384): %v", err)
	}
	if bt.b == nil || bt.b.ctx < 8192 {
		t.Fatalf("cohort ctx = %v, want >= 8192", bt.b)
	}
	bt.SetMaxContext(32768)
	if got := bt.maxContext(); got != 32768 {
		t.Fatalf("after SetMaxContext(32768): %d", got)
	}
	bt.SetMaxContext(0)
	if got := bt.maxContext(); got != Inkling2LoopResidentContext {
		t.Fatalf("after SetMaxContext(0): %d, want %d", got, Inkling2LoopResidentContext)
	}
}

type fakeInk2MemBackend struct {
	compute.Backend
	name  string
	total int64
	free  int64
	known bool
}

func (b fakeInk2MemBackend) Name() string       { return b.name }
func (b fakeInk2MemBackend) Caps() compute.Caps { return compute.Caps{CapacityProbe: b.known} }
func (b fakeInk2MemBackend) DeviceMemory() (int64, int64, bool) {
	return b.total, b.free, b.known
}

type fakeInk2HalfMemBackend struct {
	fakeInk2MemBackend
	fakeInk2SeqOps
	fakeInk2HalfPrefill
}

func TestInkling2ResolveServeContext(t *testing.T) {
	local := make([]bool, 24)
	for i := range local {
		local[i] = i%4 != 3
	}
	m := &Inkling2LoopModel{
		HP: model.Inkling2LoopHParams{
			Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3,
			Hidden: 2304, Heads: 16, KVHeads: 8, HeadDim: 128,
			LocalLayer: local,
		},
	}
	unset := func(string) (string, bool) { return "", false }
	explicitEnv := func(k string) (string, bool) {
		if k == Inkling2ResidentCtxEnv {
			return "8192", true
		}
		return "", false
	}

	// Explicit requested > 0 returns requested (clamped to Metal hard max on metal).
	cuda := fakeInk2MemBackend{name: "cuda", total: 16 << 30, free: 4 << 30, known: true}
	metal := fakeInk2MemBackend{name: "metal", total: 64 << 30, free: 32 << 30, known: true}
	if got := Inkling2ResolveServeContext(m, cuda, 16384, unset); got != 16384 {
		t.Fatalf("requested 16384 on cuda = %d, want 16384", got)
	}
	if got := Inkling2ResolveServeContext(m, metal, 65536, unset); got != Inkling2ResidentCtxMaxMetal {
		t.Fatalf("requested 65536 on metal = %d, want %d", got, Inkling2ResidentCtxMaxMetal)
	}
	// Explicit FAK_INKLING2_CTX returns the parsed value from lookup.
	if got := Inkling2ResolveServeContext(m, cuda, 0, explicitEnv); got != 8192 {
		t.Fatalf("explicit env FAK_INKLING2_CTX=8192 = %d, want 8192", got)
	}
	// Unknown free memory returns Inkling2LoopResidentContext.
	unknown := fakeInk2MemBackend{name: "cuda", free: compute.FreeUnknown, known: false}
	if got := Inkling2ResolveServeContext(m, unknown, 0, unset); got != Inkling2LoopResidentContext {
		t.Fatalf("unknown free = %d, want %d", got, Inkling2LoopResidentContext)
	}
	// Auto on 16 GiB-class GPU (~2.5 GiB free) with f16-capable backend -> 16384.
	half16G := fakeInk2HalfMemBackend{fakeInk2MemBackend: fakeInk2MemBackend{name: "cuda", total: 16 << 30, free: 2534 << 20, known: true}}
	if got := Inkling2ResolveServeContext(m, half16G, 0, unset); got != 16384 {
		t.Fatalf("auto 16GiB f16 backend = %d, want 16384", got)
	}
	// Auto on 24-32 GiB-class GPU (>= 8 GiB free) with f16-capable backend scales via fit planner up to 262144.
	half32G := fakeInk2HalfMemBackend{fakeInk2MemBackend: fakeInk2MemBackend{name: "cuda", total: 32 << 30, free: 12 << 30, known: true}}
	if got := Inkling2ResolveServeContext(m, half32G, 0, unset); got < 65536 || got > Inkling2ResidentCtxFitMax {
		t.Fatalf("auto 32GiB f16 backend = %d, want in [65536, %d]", got, Inkling2ResidentCtxFitMax)
	}
	// Auto on 80 GiB-class GPU with f16-capable backend advertises up to 262144.
	half80G := fakeInk2HalfMemBackend{fakeInk2MemBackend: fakeInk2MemBackend{name: "cuda", total: 80 << 30, free: 64 << 30, known: true}}
	if got := Inkling2ResolveServeContext(m, half80G, 0, unset); got != Inkling2ResidentCtxFitMax {
		t.Fatalf("auto 80GiB f16 backend = %d, want %d", got, Inkling2ResidentCtxFitMax)
	}
	// Explicit f32, seqops=0, or batch=0 disables auto-f16 sizing and keeps 16384 ceiling.
	for _, tc := range []struct {
		name string
		k, v string
	}{
		{"BATCH_KV=f32", "FAK_INKLING2_BATCH_KV", "f32"},
		{"BATCH_SEQOPS=0", "FAK_INKLING2_BATCH_SEQOPS", "0"},
		{"SERVE_BATCH=0", "FAK_INKLING2_SERVE_BATCH", "0"},
		{"SERVE_BATCH=off", "FAK_INKLING2_SERVE_BATCH", "off"},
	} {
		env := func(k string) (string, bool) {
			if k == tc.k {
				return tc.v, true
			}
			return "", false
		}
		if got := Inkling2ResolveServeContext(m, half32G, 0, env); got != Inkling2AutoCtxCeiling {
			t.Fatalf("%s on 32GiB = %d, want %d", tc.name, got, Inkling2AutoCtxCeiling)
		}
	}
	// Auto on non-f16 backend with >= 8 GiB free stays at Inkling2AutoCtxCeiling (16384).
	f32Big := fakeInk2MemBackend{name: "cuda", total: 32 << 30, free: 12 << 30, known: true}
	if got := Inkling2ResolveServeContext(m, f32Big, 0, unset); got != Inkling2AutoCtxCeiling {
		t.Fatalf("auto 32GiB f32 backend = %d, want %d", got, Inkling2AutoCtxCeiling)
	}
}

func TestInkling2PrefixStoreAutoHalfKVWithPrefillOff(t *testing.T) {
	be := fakeInk2HalfCohort{fakeInk2Cohort: newFakeInk2Cohort()}
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	// Auto-f16 + explicit COHORT_PREFILL_KV=off disables cohort prefill, so prefix store must be refused.
	if err := inkling2PrefixStoreUsable(be, env(map[string]string{"FAK_INKLING2_COHORT_PREFILL_KV": "off"})); err == nil {
		t.Fatal("prefix store must be refused under auto-f16 with FAK_INKLING2_COHORT_PREFILL_KV=off")
	}
	// Auto-f16 with default (unset) COHORT_PREFILL_KV auto-enables f16 cohort prefill, so prefix store is usable.
	if err := inkling2PrefixStoreUsable(be, env(nil)); err != nil {
		t.Fatalf("prefix store must be usable under auto-f16 + auto f16 cohort prefill: %v", err)
	}
}

// A row whose hard MaxTokens exceeds initialDecodeHeadroom (e.g. 8192) and that
// genuinely decodes past the initial cohort KV capacity (b.ctx) is cleanly
// continued after the cohort drains and produces bit-identical greedy output.
// fak-test:runtime fast est=20ms
func TestInkling2CohortsGrowOverflowCompletesBitIdentical(t *testing.T) {
	t.Setenv("FAK_INKLING2_BATCH_ADMIT", "0")
	t.Setenv(Inkling2InitialDecodeHeadroomEnv, "64")

	d := newFakeInk2Cohort()
	bt := NewInkling2LoopCohorts(fakeInk2CohortModel(), d)
	defer bt.Close()
	bt.SetMaxContext(2048)

	// 4 rows: rows 0, 2, 3 stop early (10 tokens); row 1 genuinely generates 180 tokens
	// (past initial headroom 64 and initial cohort ctx 128).
	const (
		promptLen = 32
		maxBudget = 1024
		shortWant = 10
		longWant  = 180
	)
	prompts := make([][]int, 4)
	for i := range prompts {
		prompts[i] = make([]int, promptLen)
		for j := range prompts[i] {
			prompts[i][j] = (i+1)*10000 + j
		}
	}
	got := make([][]int, 4)
	doneCalls := make([]int, 4)
	doneRes := make([]Inkling2LoopBatchResult, 4)
	reqs := make([]Inkling2LoopBatchRequest, 4)
	for i := range reqs {
		row := i
		wantN := shortWant
		if row == 1 {
			wantN = longWant
		}
		stopTok := prompts[row][promptLen-1] + wantN + 1
		reqs[row] = Inkling2LoopBatchRequest{
			Prompt:    prompts[row],
			MaxTokens: maxBudget,
			Stop:      func(tok int) bool { return tok == stopTok },
			Emit: func(tok int) bool {
				got[row] = append(got[row], tok)
				return true
			},
			Done: func(r Inkling2LoopBatchResult) {
				doneCalls[row]++
				doneRes[row] = r
			},
		}
	}

	// Verify initial cohort ctx is 128 (align64(32 + 64) = 128), not 1088.
	initCtx, err := inkling2BatchCohortCtxFor(reqs, bt.maxContext(), 64)
	if err != nil || initCtx != 128 {
		t.Fatalf("initial cohort ctx=%d err=%v, want 128", initCtx, err)
	}

	results, err := bt.Generate(reqs)
	if err != nil {
		t.Fatalf("Generate with overflow row: %v", err)
	}

	// Compare every row against the scalar reference session for bit-identical greedy output.
	for i := range reqs {
		wantN := shortWant
		if i == 1 {
			wantN = longWant
		}
		stopTok := prompts[i][promptLen-1] + wantN + 1
		var want []int
		turn := Inkling2LoopTurn{
			MaxTokens: maxBudget,
			Stop:      func(tok int) bool { return tok == stopTok },
			Emit:      func(tok int) bool { want = append(want, tok); return true },
		}
		if _, err := Inkling2LoopSessionDecode(context.Background(), fakeInk2Session{}, prompts[i], &turn); err != nil {
			t.Fatalf("scalar ref row %d: %v", i, err)
		}
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("row %d: emitted len=%d, want len=%d (first mismatch)", i, len(got[i]), len(want))
		}
		if doneCalls[i] != 1 {
			t.Fatalf("row %d: Done called %d times, want 1", i, doneCalls[i])
		}
		if results[i].Err != nil || results[i].Generated != wantN || !results[i].Stopped {
			t.Fatalf("row %d: result=%+v, want Generated=%d Stopped=true Err=nil", i, results[i], wantN)
		}
		if doneRes[i].Generated != results[i].Generated || doneRes[i].Stopped != results[i].Stopped || doneRes[i].Err != nil {
			t.Fatalf("row %d: Done result=%+v != returned result=%+v", i, doneRes[i], results[i])
		}
	}
}
