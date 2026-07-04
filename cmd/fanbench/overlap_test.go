package main

// The comm/compute-overlap arm (issue #651, part of the MPI-shaped message-passing
// epic #639). It isolates the one overlap factor no other fanbench arm measures:
// fak's Reap releases k.mu BEFORE calling eng.Complete (internal/kernel/kernel.go),
// so W host goroutines reaping DISTINCT handles run their inference CONCURRENTLY.
// This fixture drives the REAL kernel via Submit/Reap over a deterministic offline
// engine and reports
//
//	overlap-efficiency = (serial adjudicate + serial infer) / measured wall
//
// as a reproducible go-test number — no CLI flag, no artifact on disk.
//
// HONESTY CAVEAT (the same line the bench doc carries, docs/proofs/async-addressing.md):
// this measures fak's ACTUAL overlap SHAPE — host-driven concurrent Reap over an engine
// whose Complete runs INLINE on the reaping goroutine — NOT an MPI non-blocking progress
// thread and NOT MPI/HPC latency-hiding. It borrows the Isend/Irecv + progress STORY for
// shape only; it never carries over an MPI/InfiniBand overlap, message-rate, or throughput
// number. The engine is atomic/offline (its call tally is a sync/atomic counter, not
// engine.Mock's racy `m.calls++`) so the fixture ships a race-clean, reproducible value.

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

// overlapEngineID is the id this arm registers its deterministic offline engine under
// and binds its kernels to. It is unique to this fixture so registering it never
// clobbers a real engine driver.
const overlapEngineID = "fanbench-overlap-atomic"

// atomicOverlapEngine is the deterministic, offline inference engine this arm runs on.
// Every Complete costs a FIXED `infer` duration (the reproducible per-call inference
// span the overlap factor is measured against) and returns a payload derived purely
// from (tool, args) — so a result is bit-identical regardless of which goroutine reaped
// it or in what order. Its call tally is a sync/atomic counter: the whole reason the arm
// does NOT use engine.Mock, whose plain `m.calls++` races under the concurrent Reap this
// fixture drives (and would make it flake under -race).
type atomicOverlapEngine struct {
	infer time.Duration
	calls atomic.Int64
}

func (e *atomicOverlapEngine) Caps() []abi.Capability { return nil }

func (e *atomicOverlapEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	time.Sleep(e.infer) // the deterministic per-call inference cost — runs OUTSIDE k.mu
	body := []byte(fmt.Sprintf(`{"tool":%q,"in":%q,"ok":true}`, c.Tool, string(c.Args.Inline)))
	return &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body)), Taint: abi.TaintTrusted},
		Status:  abi.StatusOK,
	}, nil
}

// allowOverlapArm admits every call so the arm isolates the Reap-side overlap rather
// than an adjudication policy. It is injected as an EXPLICIT chain (kernel.WithAdjudicators)
// so the fixture never depends on whatever the process-global adjudicator registry holds.
type allowOverlapArm struct{}

func (allowOverlapArm) Caps() []abi.Capability { return nil }
func (allowOverlapArm) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "overlap-arm"}
}

// overlapRun is one Submit-all-then-Reap-all pass over the kernel.
type overlapRun struct {
	adjudicateWall time.Duration // wall to Submit (adjudicate) all n calls, serial in both worlds
	inferWall      time.Duration // wall to Reap (infer) all n handles (serial if workers<=1, else overlapped)
	results        []string      // per-call payload, indexed by submission order
	engineCalls    int64         // this kernel's engine-call tally (must be exactly n)
}

// runOverlapArm submits n calls to a fresh kernel bound to the atomic engine, then reaps
// all n handles. With workers<=1 the reaps are fully serialized; with workers>1 they run
// over that many goroutines, exercising the real overlap path (Reap unlocks k.mu before
// eng.Complete, so distinct handles' inference overlaps). The vDSO fast path is ABLATED so
// every call provably reaches the engine — the engine-call count is exactly n and no cached
// fast-path result can short-circuit a Reap and hide the overlap.
func runOverlapArm(t *testing.T, ctx context.Context, chain []abi.Adjudicator, n, workers int) overlapRun {
	t.Helper()
	k := kernel.New(overlapEngineID, kernel.WithAdjudicators(chain))
	k.SetVDSO(false)

	handles := make([]abi.SubmissionHandle, n)
	t0 := time.Now()
	for i := 0; i < n; i++ {
		args := []byte(fmt.Sprintf(`{"i":%d}`, i))
		c := &abi.ToolCall{
			Tool: "bench.overlap",
			Args: abi.Ref{Kind: abi.RefInline, Inline: args, Len: int64(len(args)), Taint: abi.TaintTrusted},
		}
		h, v := k.Submit(ctx, c)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("submit %d: verdict = %v, want Allow", i, v.Kind)
		}
		handles[i] = h
	}
	adjWall := time.Since(t0)

	results := make([]string, n)
	t1 := time.Now()
	if workers <= 1 {
		for i, h := range handles {
			r, err := k.Reap(ctx, h)
			if err != nil {
				t.Fatalf("serial reap %d: %v", i, err)
			}
			results[i] = string(r.Payload.Inline)
		}
	} else {
		// A work queue of handle indices drained by `workers` goroutines: each writes a
		// DISTINCT results[i], so the only shared mutation is inside the kernel (mutex- and
		// atomic-guarded). t.Fatalf is illegal off the test goroutine, so a worker reports
		// via t.Errorf and bails.
		var wg sync.WaitGroup
		idx := make(chan int, n)
		for i := range handles {
			idx <- i
		}
		close(idx)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range idx {
					r, err := k.Reap(ctx, handles[i])
					if err != nil {
						t.Errorf("concurrent reap %d: %v", i, err)
						return
					}
					results[i] = string(r.Payload.Inline)
				}
			}()
		}
		wg.Wait()
	}
	inferWall := time.Since(t1)

	return overlapRun{
		adjudicateWall: adjWall,
		inferWall:      inferWall,
		results:        results,
		engineCalls:    k.Counters().EngineCalls,
	}
}

// TestOverlapArm_ConcurrentReapOverlapsInference is the fanbench comm/compute-overlap arm.
// It runs the same n Submit/Reap calls two ways over a deterministic offline engine — once
// with a single reaper (inference fully serialized) and once with W concurrent reapers
// (inference overlapped, the real Reap-unlocks-before-Complete path) — and reports
// overlap-efficiency = (serial adjudicate + serial infer) / measured wall. The number is
// MEASURED, so the test asserts only the structural, non-flaky facts (deterministic engine
// tally, reproducible results, overlapped infer strictly faster than serial) and LOGS the
// efficiency rather than pinning a wall-clock value.
func TestOverlapArm_ConcurrentReapOverlapsInference(t *testing.T) {
	const (
		n       = 64                   // submissions
		workers = 8                    // concurrent Reap goroutines
		infer   = 2 * time.Millisecond // deterministic per-call inference cost
	)

	eng := &atomicOverlapEngine{infer: infer}
	abi.RegisterEngine(overlapEngineID, eng)

	ctx := context.Background()
	chain := []abi.Adjudicator{allowOverlapArm{}}

	serial := runOverlapArm(t, ctx, chain, n, 1)     // one reaper: inference serialized
	over := runOverlapArm(t, ctx, chain, n, workers) // W reapers: inference overlapped
	if t.Failed() {
		return
	}

	// (a) Determinism: every call reached the engine exactly once per run (vDSO ablated,
	// nothing denied), and the atomic tally is race-clean across BOTH runs — the concurrent
	// run alone fired n overlapping Add(1)s with no lost update.
	if serial.engineCalls != n || over.engineCalls != n {
		t.Fatalf("engine calls: serial=%d overlapped=%d, want %d each", serial.engineCalls, over.engineCalls, n)
	}
	if got := eng.calls.Load(); got != int64(2*n) {
		t.Fatalf("atomic engine tally = %d, want %d (race-free across both runs)", got, 2*n)
	}

	// (b) Reproducible results: the payload for each call is identical whether it was reaped
	// serially or concurrently — Reap order never changes a result.
	for i := 0; i < n; i++ {
		if serial.results[i] != over.results[i] {
			t.Fatalf("call %d payload differs across runs: serial=%q overlapped=%q", i, serial.results[i], over.results[i])
		}
		if serial.results[i] == "" {
			t.Fatalf("call %d produced an empty payload", i)
		}
	}

	// (c) overlap-efficiency = (serial adjudicate + serial infer) / measured wall.
	serialWall := serial.adjudicateWall + serial.inferWall
	measuredWall := over.adjudicateWall + over.inferWall
	efficiency := float64(serialWall) / float64(measuredWall)
	t.Logf("fanbench overlap arm: n=%d workers=%d infer=%s | serial: adjudicate=%s infer=%s | overlapped: adjudicate=%s infer=%s | overlap-efficiency=%.2fx (host-driven concurrent Reap over inline Complete — NOT an MPI progress thread)",
		n, workers, infer, serial.adjudicateWall, serial.inferWall, over.adjudicateWall, over.inferWall, efficiency)

	// The structural witness (non-flaky by a wide margin: serial infer ~= n*infer,
	// overlapped ~= ceil(n/workers)*infer, so the ratio is ~workers): concurrent Reap
	// overlaps inference, so the overlapped infer wall is strictly below the serialized
	// one and the efficiency exceeds 1. No hard wall-clock number is asserted.
	if over.inferWall >= serial.inferWall {
		t.Fatalf("no overlap witnessed: overlapped infer wall %s >= serial %s", over.inferWall, serial.inferWall)
	}
	if efficiency <= 1.0 {
		t.Fatalf("overlap-efficiency %.2fx <= 1.0 — concurrent Reap did not overlap inference", efficiency)
	}
}
