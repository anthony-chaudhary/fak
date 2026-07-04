package bench

// fanoverlap_test.go — issue #651: the comm/compute-overlap arm.
//
// MPI analogue (SHAPE ONLY): non-blocking communication / computation overlap
// (MPI_Isend/MPI_Irecv + a progress engine). Borrowed for shape, NOT for any
// number.
//
// What this measures — fak's ACTUAL overlap seam: kernel.Reap releases k.mu
// BEFORE it calls engine.Complete, so W goroutines each reaping a DISTINCT
// pending handle run their engine dwell concurrently, while a serialized driver
// pays the dwell W times back-to-back. overlap-efficiency =
// (serial-reap wall) / (concurrent-reap wall): ~1 at W=1 (nothing to overlap),
// climbing toward W as the dwells stack.
//
// HONESTY CAVEAT (do not misread this number): this is host-driven concurrent
// Reap over an engine whose Complete runs inline — it is NOT an MPI non-blocking
// progress thread, NOT MPI/HPC latency-hiding, and it borrows NO MPI/InfiniBand
// overlap or message-rate figure. The engine dwell is a FIXED, deterministic
// stand-in for engine latency, not a measured model number.
//
// Determinism / -race: the offline engine keeps only an atomic counter (unlike
// engine.Mock, whose m.calls++ races under concurrent Reap), and the kernel is
// built in isolation (an injected per-kernel allow-chain via WithAdjudicators +
// an additively-registered offline engine under a unique id), so the arm neither
// reads nor mutates the process-global adjudicator world the sibling bench tests
// rely on. It is reproducible and clean under `go test -race`.

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// overlapAllow is the isolated per-kernel policy: it allows every call, so the
// arm exercises the Submit->Reap->engine path without depending on (or mutating)
// the process-global adjudicator registry. Injected via kernel.WithAdjudicators.
type overlapAllow struct{}

func (overlapAllow) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "fanbench-overlap"}
}
func (overlapAllow) Caps() []abi.Capability { return nil }

// offlineSleepEngine is a DETERMINISTIC offline engine: every Complete blocks for
// a fixed dwell (a stand-in for engine latency) and touches only an atomic
// counter. Because Reap unlocks before Complete, W concurrent reaps overlap their
// dwells while a serialized driver pays dwell*W. NOT an inference engine.
type offlineSleepEngine struct {
	dwell time.Duration
	calls int64
}

func (e *offlineSleepEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	time.Sleep(e.dwell)
	atomic.AddInt64(&e.calls, 1)
	return &abi.Result{Call: c, Status: abi.StatusOK, Payload: c.Args}, nil
}
func (e *offlineSleepEngine) Caps() []abi.Capability { return nil }

const overlapEngineID = "fanbench-overlap"

func overlapCall(i int) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: "fanbench_overlap_probe",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(fmt.Sprintf(`{"i":%d}`, i))},
	}
}

// newOverlapKernel builds an ISOLATED kernel: an injected allow-chain (no global
// adjudicator mutation), vDSO off (every call reaches the engine — no fast-path
// interception), routed to a freshly registered offline engine. RegisterEngine is
// additive under a unique id, so it neither resets nor collides with the world the
// sibling bench tests rely on.
func newOverlapKernel(dwell time.Duration) (*kernel.Kernel, *offlineSleepEngine) {
	eng := &offlineSleepEngine{dwell: dwell}
	abi.RegisterEngine(overlapEngineID, eng)
	k := kernel.New(overlapEngineID, kernel.WithAdjudicators([]abi.Adjudicator{overlapAllow{}}))
	k.SetVDSO(false)
	return k, eng
}

// reapSerial submits then reaps width calls one at a time; the engine dwell is
// paid width times back-to-back (the serialized-adjudication baseline).
func reapSerial(t *testing.T, k *kernel.Kernel, width int) time.Duration {
	t.Helper()
	ctx := context.Background()
	start := time.Now()
	for i := 0; i < width; i++ {
		h, v := k.Submit(ctx, overlapCall(i))
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("serial submit %d: verdict=%v, want Allow", i, v.Kind)
		}
		if _, err := k.Reap(ctx, h); err != nil {
			t.Fatalf("serial reap %d: %v", i, err)
		}
	}
	return time.Since(start)
}

// reapConcurrent submits width calls, then reaps them from width goroutines; the
// dwells overlap because Reap releases k.mu before engine.Complete. Each goroutine
// writes a distinct errs index (race-free); errors are surfaced after the join.
func reapConcurrent(t *testing.T, k *kernel.Kernel, width int) time.Duration {
	t.Helper()
	ctx := context.Background()
	handles := make([]abi.SubmissionHandle, width)
	for i := 0; i < width; i++ {
		h, v := k.Submit(ctx, overlapCall(i))
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("concurrent submit %d: verdict=%v, want Allow", i, v.Kind)
		}
		handles[i] = h
	}
	errs := make([]error, width)
	var wg sync.WaitGroup
	start := time.Now()
	for i := 0; i < width; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, errs[idx] = k.Reap(ctx, handles[idx])
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start)
	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent reap %d: %v", i, err)
		}
	}
	return elapsed
}

// TestFanbenchOverlapArm is the comm/compute-overlap arm (#651): across a width
// sweep it reports overlap-efficiency = serial-reap-wall / concurrent-reap-wall
// and witnesses that concurrent Reap over distinct handles overlaps the engine
// dwell that a serialized driver pays in full. Deterministic and -race clean
// (atomic-only engine state); the exact wall varies with the host, so the
// assertions are magnitude-robust (eff > 1 once there is more than one dwell to
// overlap), never brittle equalities.
func TestFanbenchOverlapArm(t *testing.T) {
	const dwell = 10 * time.Millisecond
	for _, width := range []int{1, 2, 4, 8} {
		k, eng := newOverlapKernel(dwell)
		serial := reapSerial(t, k, width)
		concurrent := reapConcurrent(t, k, width)

		if got := atomic.LoadInt64(&eng.calls); got != int64(2*width) {
			t.Fatalf("width=%d: engine dispatched %d calls, want %d (vDSO off — every reap must hit the engine)", width, got, 2*width)
		}

		eff := serial.Seconds() / concurrent.Seconds()
		t.Logf("width=%d overlap-efficiency=%.2f (serial=%s concurrent=%s dwell=%s)", width, eff, serial, concurrent, dwell)

		if width == 1 {
			continue // one dwell: nothing to overlap, eff ~ 1 by construction
		}
		if concurrent >= serial {
			t.Errorf("width=%d: concurrent reap (%s) did not overlap serial (%s) — the Reap-unlock-before-Complete seam is not overlapping", width, concurrent, serial)
		}
		if eff <= 1.0 {
			t.Errorf("width=%d: overlap-efficiency=%.2f, want > 1.0 (concurrent Reap must beat serialized adjudication+inference)", width, eff)
		}
	}
}
