package main

// fak-test:runtime fast est=0s

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

const fitCommitGiB = int64(1) << 30

// fakeOverReportingDevice models a backend whose capacity probe over-reports
// free memory: the planner believes it can fill the whole shrunken budget, but
// the real aperture only holds actualFree bytes, so every attempt that plans
// more than actualFree fails its allocation exactly like a live split-aperture
// or stale-probe device would.
type fakeOverReportingDevice struct {
	actualFree int64
	calls      int
	lastLayout serveFitCommitLayout
}

func (d *fakeOverReportingDevice) step(l serveFitCommitLayout) (serveFitCommitResult, error) {
	d.calls++
	d.lastLayout = l
	planned := l.SizingFit.avail()
	// A live load computes its memory plan BEFORE the allocation fails, so the
	// failed attempt carries the same plan it tried to commit.
	plan := compute.MemoryPlan{{Scope: compute.MemoryScopeDevice, Class: compute.MemoryWeights, Detail: "synth-weights", Bytes: planned}}
	if planned > d.actualFree {
		return serveFitCommitResult{Plan: plan}, &compute.DeviceAllocError{Bytes: int(planned), Site: "dalloc", Class: compute.MemoryKVCache}
	}
	return serveFitCommitResult{Plan: plan}, nil
}

// TestServeFitCommitConvergesWithinCap is the ticket's first required case: a
// backend that over-reports free fails its first allocation, and the back-off
// converges to a committed plan that fits the REAL free within the retry cap.
// The cut schedule (1%, 2%, 4%, 8%, 16% of the original reading) reaches a
// fitting plan on the fifth retry for a 32 GiB reported / 27 GiB real aperture.
func TestServeFitCommitConvergesWithinCap(t *testing.T) {
	dev := &fakeOverReportingDevice{actualFree: 27 * fitCommitGiB}
	cfg := serveFitCommitConfig{
		PinnedContext:  false,
		OriginalTokens: 8192,
		SizingFit:      serveFitBudget{Base: 32 * fitCommitGiB},
		Arm:            "device-resident-q4k",
	}
	res, err := runServeFitCommit(cfg, dev.step)
	if err != nil {
		t.Fatalf("back-off should converge within the cap, got %v", err)
	}
	if dev.calls != 6 {
		t.Fatalf("attempts = %d, want 6 (original + 5 retries)", dev.calls)
	}
	var committed int64
	for _, row := range res.Plan {
		committed += row.Bytes
	}
	if committed > dev.actualFree {
		t.Fatalf("committed plan %d exceeds real free %d", committed, dev.actualFree)
	}
	if committed == 0 {
		t.Fatal("committed plan is empty")
	}
}

// TestServeFitCommitRepeatedLayoutTypedRefusal is the ticket's second required
// case: an oscillating backend whose planner granularity makes every shrunken
// budget plan IDENTICALLY. The driver must stop on the repeated layout with the
// typed refusal reachable via errors.Is, naming the last layout.
func TestServeFitCommitRepeatedLayoutTypedRefusal(t *testing.T) {
	const samePlanBytes = 8 * fitCommitGiB
	steps := 0
	step := func(l serveFitCommitLayout) (serveFitCommitResult, error) {
		steps++
		// Every believed-free reading plans the same bytes: the repeated layout.
		return serveFitCommitResult{
			Plan: compute.MemoryPlan{{Scope: compute.MemoryScopeDevice, Class: compute.MemoryWeights, Detail: "synth-weights", Bytes: samePlanBytes}},
		}, &agent.InKernelOOMError{Bytes: int(samePlanBytes), Site: "test"}
	}
	cfg := serveFitCommitConfig{
		PinnedContext:  true,
		OriginalTokens: 4096,
		SizingFit:      serveFitBudget{Base: 32 * fitCommitGiB},
		Arm:            "device-resident-q4k",
	}
	_, err := runServeFitCommit(cfg, step)
	if steps != 2 {
		t.Fatalf("steps = %d, want 2 (stop on the repeated layout)", steps)
	}
	if !errors.Is(err, ErrServeFitCommit) {
		t.Fatalf("errors.Is(err, ErrServeFitCommit) = false, got %T (%v)", err, err)
	}
	var fce *serveFitCommitError
	if !errors.As(err, &fce) {
		t.Fatalf("errors.As did not recover *serveFitCommitError: %v", err)
	}
	if fce.Attempts != 2 {
		t.Fatalf("Attempts = %d, want 2", fce.Attempts)
	}
	if !strings.Contains(fce.LastLayout, "ctx=4096") || !strings.Contains(fce.LastLayout, "arm=device-resident-q4k") {
		t.Fatalf("last layout %q does not name the pinned context or arm", fce.LastLayout)
	}
	if !strings.Contains(err.Error(), "last layout:") {
		t.Fatalf("refusal %q does not name the last layout", err.Error())
	}
}

// TestServeFitCommitPinnedContextNeverMoves pins the context axis explicitly,
// forces allocation failures on every attempt, and asserts the pinned token
// count is byte-identical across all layouts while the refusal still names the
// last layout.
func TestServeFitCommitPinnedContextNeverMoves(t *testing.T) {
	const pinned = 4096
	var seen []int
	step := func(l serveFitCommitLayout) (serveFitCommitResult, error) {
		seen = append(seen, l.ContextTokens)
		return serveFitCommitResult{}, &compute.FitError{Verdict: compute.FitTooBig}
	}
	cfg := serveFitCommitConfig{
		PinnedContext:  true,
		OriginalTokens: pinned,
		SizingFit:      serveFitBudget{Base: 32 * fitCommitGiB},
		Arm:            "device-resident-q4k",
	}
	_, err := runServeFitCommit(cfg, step)
	if !errors.Is(err, ErrServeFitCommit) {
		t.Fatalf("want the typed refusal, got %v", err)
	}
	// Pinned + identical failure shape: the layout repeats after the first two
	// attempts (the budgets differ but the plan they produced here is fixed), so
	// the driver stops early; whatever it ran, EVERY layout kept the pin.
	for i, tokens := range seen {
		if tokens != pinned {
			t.Fatalf("attempt %d moved the pinned context to %d, want %d", i, tokens, pinned)
		}
	}
	var fce *serveFitCommitError
	if !errors.As(err, &fce) || !strings.Contains(fce.LastLayout, fmt.Sprintf("ctx=%d", pinned)) {
		t.Fatalf("refusal %v does not name the pinned last layout", err)
	}
}

// TestServeFitCommitNonAllocationSurfacesUnchanged asserts a non-allocation
// failure (corrupt weights, tokenizer miss, ...) is returned unchanged after a
// single attempt: shrinking the believed free cannot cure it.
func TestServeFitCommitNonAllocationSurfacesUnchanged(t *testing.T) {
	cause := errors.New("gguf: corrupt tensor header")
	steps := 0
	_, err := runServeFitCommit(serveFitCommitConfig{SizingFit: serveFitBudget{Base: 16 * fitCommitGiB}}, func(l serveFitCommitLayout) (serveFitCommitResult, error) {
		steps++
		return serveFitCommitResult{}, cause
	})
	if steps != 1 {
		t.Fatalf("steps = %d, want 1 (no retry on non-allocation failure)", steps)
	}
	if err != cause {
		t.Fatalf("error = %v, want the unchanged cause %v", err, cause)
	}
	if errors.Is(err, ErrServeFitCommit) {
		t.Fatal("non-allocation failure must not be classified as a fit-commit refusal")
	}
}

// TestServeFitCommitCutSchedule pins the doubling cut: 1%, 2%, 4%, 8%, 16% of
// the ORIGINAL reading per retry, clamped to the whole budget.
func TestServeFitCommitCutSchedule(t *testing.T) {
	cfg := serveFitCommitConfig{
		OriginalTokens: 1024,
		SizingFit:      serveFitBudget{Base: 10000},
		Arm:            "test",
	}
	want := []int64{10000, 9900, 9800, 9600, 9200, 8400}
	for attempt, base := range want {
		got := cfg.layoutFor(attempt).SizingFit.Base
		if got != base {
			t.Fatalf("layoutFor(%d).SizingFit.Base = %d, want %d", attempt, got, base)
		}
	}
	// A budget smaller than one percent of itself still cuts by at least a
	// byte, never to zero-and-below.
	tiny := serveFitCommitConfig{SizingFit: serveFitBudget{Base: 1}}
	if got := tiny.layoutFor(1).SizingFit.Base; got != 0 {
		t.Fatalf("tiny budget layoutFor(1).Base = %d, want 0 (clamped)", got)
	}
}

// TestServeFitCommitFingerprintStableAcrossReadings asserts the stop condition
// itself: two attempts with different believed-free readings that plan
// identically map to the SAME fingerprint, while a different plan does not.
func TestServeFitCommitFingerprintStableAcrossReadings(t *testing.T) {
	plan := compute.MemoryPlan{
		{Scope: compute.MemoryScopeHost, Class: compute.MemoryWeights, Detail: "b", Bytes: 2},
		{Scope: compute.MemoryScopeDevice, Class: compute.MemoryKVCache, Detail: "a", Bytes: 1},
	}
	l1 := serveFitCommitLayout{ContextTokens: 512, SizingFit: serveFitBudget{Base: 100}}
	l2 := serveFitCommitLayout{ContextTokens: 512, SizingFit: serveFitBudget{Base: 99}}
	fp1 := serveFitCommitFingerprint(plan, l1, "arm")
	fp2 := serveFitCommitFingerprint(plan, l2, "arm")
	if fp1 != fp2 {
		t.Fatalf("identical plans under different readings must share a fingerprint:\n%s\n%s", fp1, fp2)
	}
	other := compute.MemoryPlan{
		{Scope: compute.MemoryScopeHost, Class: compute.MemoryWeights, Detail: "b", Bytes: 2},
		{Scope: compute.MemoryScopeDevice, Class: compute.MemoryKVCache, Detail: "a", Bytes: 3},
	}
	if fp3 := serveFitCommitFingerprint(other, l1, "arm"); fp3 == fp1 {
		t.Fatal("different plans must not share a fingerprint")
	}
}
