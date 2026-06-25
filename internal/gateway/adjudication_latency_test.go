package gateway

import (
	"context"
	"sort"
	"testing"
	"time"
)

// TestAdjudicationLatencyUnder100us is the host-tractable acceptance witness for
// issue #282 (B-004, Continuous Batching Integration). When fak fronts a serving
// engine's continuous batching (vLLM/SGLang), every request crosses the gateway's
// adjudication hop BEFORE it is forwarded to the batched engine — so the load-bearing
// per-request cost fak adds to the batch is exactly s.adjudicate:
//
//	buildCall  ->  k.Decide (the adjudicator chain)  ->  renderVerdict
//
// no engine dispatch, no vDSO fold (that path is s.syscall). The #282 acceptance bar
// is "Adjudication latency ≤ 100µs per request". This test MEASURES that hop on the
// representative admit-and-forward case (an ALLOW request the front door passes
// through to the engine) and gates the typical per-request cost at ≤ 100µs.
//
// Why this is the honest form of the criterion (vs. the two GPU-gated siblings):
//   - #282's "compatible with vLLM continuous batching" and "throughput within 10% of
//     direct vLLM" both require a live vLLM peer + GPU not attached to this host, so
//     they stay deferred (see docs/notes/track-b-performance-parity-tracking-306.md).
//   - The adjudication-latency bar is in-process and device-free; it is the one #282
//     acceptance box that can be turned from a PROJECTED number in a companion doc
//     (docs/benchmarks/GUARD-HOP-OVERHEAD-PENDING.md: ~2.4µs in-process ceil) into a
//     MEASURED, CI-enforced regression gate. That ~2.4µs ceil leaves ~40x headroom
//     under the 100µs bar, which is what keeps this gate non-flaky on a loaded box.
//
// The gate is stated on the distribution, not one sample, so a stray GC/scheduler
// pause cannot flip a green tree red: it requires the MEDIAN ≤ 100µs (the typical
// per-request cost the acceptance is really about) AND ≥ 99% of requests ≤ 100µs (the
// honest "per request" reading, tolerant of rare outliers on a shared host).
func TestAdjudicationLatencyUnder100us(t *testing.T) {
	const (
		budget    = 100 * time.Microsecond // #282 acceptance bar
		iters     = 5000                   // enough samples for a stable p50/p99
		warmup    = 200                    // amortize first-call allocation/path warmup
		minUnder  = 0.99                   // ≥99% of requests must clear the bar
		args      = `{"x":1}`              // a small, representative ALLOW payload
		allowTool = "allow_read"           // toolAdj: "allow*" -> ALLOW (admit + forward)
	)

	srv := newTestServer(t)
	ctx := context.Background()

	// Quiet the per-operation structured log: it is observability emitted AFTER the
	// verdict, not part of adjudication, and a synchronous stderr write per call would
	// both dominate the measured tail and spam 5k lines into the suite. The metrics
	// fold (observeOperation) stays, so the measured window is the adjudication work.
	srv.logf = func(string, ...any) {}

	// Warm the path so the measured window reflects steady-state per-request cost, not
	// one-time setup. Also asserts the tool actually ADMITs — a silent DENY would make
	// us measure the wrong (cheaper) branch and quietly under-report the real overhead.
	for i := 0; i < warmup; i++ {
		wv, _, err := srv.adjudicate(ctx, allowTool, args, false, "", "warmup")
		if err != nil {
			t.Fatalf("adjudicate warmup: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Fatalf("warmup verdict = %q, want ALLOW (must measure the admit-and-forward path)", wv.Kind)
		}
	}

	durs := make([]time.Duration, iters)
	for i := 0; i < iters; i++ {
		start := time.Now()
		wv, _, err := srv.adjudicate(ctx, allowTool, args, false, "", "lat-bench")
		d := time.Since(start)
		if err != nil {
			t.Fatalf("adjudicate iter %d: %v", i, err)
		}
		if wv.Kind != "ALLOW" {
			t.Fatalf("iter %d verdict = %q, want ALLOW", i, wv.Kind)
		}
		durs[i] = d
	}

	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	p50 := durs[len(durs)*50/100]
	p99 := durs[len(durs)*99/100]
	maxd := durs[len(durs)-1]

	under := 0
	for _, d := range durs {
		if d <= budget {
			under++
		}
	}
	frac := float64(under) / float64(len(durs))

	t.Logf("#282 gateway adjudication latency over %d requests: p50=%v p99=%v max=%v; %.2f%% ≤ %v",
		iters, p50, p99, maxd, frac*100, budget)

	if p50 > budget {
		t.Errorf("#282 acceptance: median adjudication latency %v exceeds the ≤ %v per-request bar", p50, budget)
	}
	if frac < minUnder {
		t.Errorf("#282 acceptance: only %.2f%% of requests cleared the ≤ %v bar (want ≥ %.0f%%)",
			frac*100, budget, minUnder*100)
	}
}
