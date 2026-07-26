package gateway

import (
	"context"
	"os"
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
//     under the 100µs bar, which is what keeps this gate non-flaky on a loaded box —
//     but only on a clock that can resolve a single hop; see the CLOCK-QUANTIZATION
//     note below for where that 40x headroom silently collapses to ~1.4x and what
//     carries the gate there instead.
//
// The gate is stated on the distribution, not one sample, so a stray GC/scheduler
// pause cannot flip a green tree red: it requires the MEDIAN ≤ 100µs (the typical
// per-request cost the acceptance is really about) AND ≥ 99% of requests ≤ 100µs (the
// honest "per request" reading, tolerant of rare outliers on a shared host) — the
// latter wherever the host clock can actually resolve a sub-100µs sample, plus an
// always-on batch-mean gate that does not depend on the clock at all.
//
// #282 CLOCK-QUANTIZATION NOTE (why the per-request bar is conditional and why the
// batch mean exists — this is an issue-derived acceptance contract, so the re-gating is
// written down rather than done silently). Go's Windows nanotime reads the system
// interrupt time, whose period is the machine-wide timer resolution; the package helper
// monotonicGranularity measures ~505µs on THIS host (501-519µs typical, 316µs-997µs across
// runs as the machine-wide resolution drifts with whatever process last raised it — every
// one of those readings 3x or more ABOVE the 100µs bar this test asserts). A ~3µs
// adjudication therefore never measures as ~3µs: it measures 0 (the hop fit inside the
// current tick) or one whole ~505µs tick (a tick boundary fell inside it). Nothing
// between 0 and one tick is representable — which is exactly why the log below reads
// p50=0s and p99=0s while max sits at one tick or a real ms-scale pause.
//
// The arithmetic that this forces, and its two consequences:
//
//   - Every over-bar sample needs a tick boundary inside it, and boundaries are disjoint
//     across iterations, so the over-bar COUNT is just the number of ticks the measured
//     loop spans: overBarFrac ≈ loopWall/granularity/iters = meanAdjudicate/granularity.
//     So "≥99% ≤ 100µs" is algebraically "mean ≤ granularity/100" = mean ≤ 5.05µs on this
//     host — a bar 20x TIGHTER than the stated 100µs, with essentially no headroom over
//     the real cost. Measured here before this correction, 10 consecutive runs: 99.20%,
//     99.24%, 99.26%, 99.28% (x3), 99.32%, 99.34%, 99.38%, 99.40% — i.e. mean 3.0-4.0µs
//     against an implied 5.05µs bar, ~70% of the budget consumed by a hop the issue
//     budgets 100µs for, and no margin at all left for the ordinary drift in the batch
//     figures below. The witness was reading the timer interrupt period, not the hop.
//     (The sibling #2219 gate hit the same wall and straddled it; #4969's identical bar
//     has been observed at 98.99/98.79/98.67% across its own retry attempts, which is why
//     retrying never helps — the artifact is systematic, not random.)
//   - No rephrasing of a PER-SAMPLE ≤100µs bar can recover information a ~505µs-quantized
//     clock never captured. A genuine 20x regression to 100µs/request would still measure
//     0 on every sample that happened to miss a tick boundary; only the over-bar fraction
//     would move, and it moves for slow-and-cheap alike. The regression signal has to be
//     read off an interval long enough for the clock to resolve it.
//
// There is no performance regression behind any of this: the granularity-immune reading
// added below puts the hop at 2.3-2.9µs (cheapest batch mean over 11 runs; median batch
// mean 2.9-3.4µs), matching the ~2.4µs projected anchor plus the two time.Now() reads the
// per-sample form adds — i.e. 36x UNDER the 100µs the issue asks for, on the same host
// whose per-sample reading straddles the bar. Hence the shape below, identical to the one
// already landed on the sibling TestSyscallServeLatencyDistribution: the strict bar runs
// UNCHANGED wherever the clock can resolve it (granularity ≤ 100µs — Linux/macOS CI,
// where it has always been the real gate), or on demand via FAK_STRICT_SERVE_LATENCY=1;
// and an always-on batch-mean gate carries the regression witness everywhere else, where
// p50=0s previously asserted nothing at all.
func TestAdjudicationLatencyUnder100us(t *testing.T) {
	if raceDetectorEnabled {
		t.Skip("latency distribution gate is not meaningful under go test -race instrumentation")
	}

	const (
		budget    = 100 * time.Microsecond // #282 acceptance bar
		iters     = 20_000                 // 10 batches of `batch`; also a stable p50/p99
		warmup    = 200                    // amortize first-call allocation/path warmup
		minUnder  = 0.99                   // ≥99% of requests must clear the bar
		args      = `{"x":1}`              // a small, representative ALLOW payload
		allowTool = "allow_read"           // toolAdj: "allow*" -> ALLOW (admit + forward)
		// The granularity-immune gate: batch the measured loop and bar the FLOOR (cheapest)
		// batch mean. Batch size is set by the clock, not by taste: one batch of 2000 hops is
		// ~6ms of wall, ~12 of the coarsest ~505µs ticks, so the batch window RESOLVES where a
		// single sample cannot and quantization costs one tick spread over 2000 iterations
		// (~0.25µs/iter, <10% of the hop). At batch=500 the window was only ~3 ticks wide and
		// the floor read 1.0-2.1µs against a 3.0µs median — a third of the reading was the
		// tick, so the batch was widened until it was not.
		// Why the floor and not the median: host contention is ONE-SIDED — it can only make
		// a batch slower — so the cheapest of 10 batches is the low-variance estimator of the
		// path's true cost, while a genuine regression is systemic and lifts EVERY batch,
		// the floor included. That buys a tighter bar and less flake at the same time.
		batch      = 2000                  // 10 batches of 2000; batch wall (~6ms) >> one ~505µs tick
		meanBudget = 20 * time.Microsecond // ~7x over the 2.3-2.9µs floor mean measured on this host
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

	// Measure twice over the same hops: per-sample durations for the distribution, and
	// per-BATCH wall for the granularity-immune mean. The batch window brackets `batch`
	// adjudications in ONE pair of clock reads, so it resolves on a coarse clock where an
	// individual sample cannot (see the CLOCK-QUANTIZATION note above).
	durs := make([]time.Duration, 0, iters)
	batchMeans := make([]time.Duration, 0, iters/batch)
	for b := 0; b < iters/batch; b++ {
		batchStart := time.Now()
		for i := 0; i < batch; i++ {
			start := time.Now()
			wv, _, err := srv.adjudicate(ctx, allowTool, args, false, "", "lat-bench")
			d := time.Since(start)
			if err != nil {
				t.Fatalf("adjudicate iter %d: %v", b*batch+i, err)
			}
			if wv.Kind != "ALLOW" {
				t.Fatalf("iter %d verdict = %q, want ALLOW", b*batch+i, wv.Kind)
			}
			durs = append(durs, d)
		}
		batchMeans = append(batchMeans, time.Since(batchStart)/batch)
	}

	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	p50 := durs[len(durs)*50/100]
	p99 := durs[len(durs)*99/100]
	maxd := durs[len(durs)-1]

	sort.Slice(batchMeans, func(i, j int) bool { return batchMeans[i] < batchMeans[j] })
	floorMean, medMean := batchMeans[0], batchMeans[len(batchMeans)/2]

	under := 0
	for _, d := range durs {
		if d <= budget {
			under++
		}
	}
	frac := float64(under) / float64(len(durs))

	// The clock's own quantum, reported alongside the numbers it explains: a reader seeing
	// p50=0s with max in the ms has to be able to tell "the hops are sub-tick" from "the
	// hops are slow", and granularity is the fact that settles it.
	gran := monotonicGranularity()

	t.Logf("#282 gateway adjudication latency over %d requests (%d batches of %d): p50=%v p99=%v max=%v; %.2f%% ≤ %v; per-request batch mean floor=%v median=%v; clock granularity=%v",
		iters, iters/batch, batch, p50, p99, maxd, frac*100, budget, floorMean, medMean, gran)

	if p50 > budget {
		t.Errorf("#282 acceptance: median adjudication latency %v exceeds the ≤ %v per-request bar", p50, budget)
	}
	// Always on, on every host: the regression witness that does not depend on the clock
	// resolving a single adjudication.
	if floorMean > meanBudget {
		t.Errorf("#282 acceptance: cheapest per-request batch mean %v exceeds the ≤ %v bar — every one of %d batches was slower than the bar, so this is a systemic adjudication regression, not host contention (batch means=%v)",
			floorMean, meanBudget, len(batchMeans), batchMeans)
	}
	// The per-request tail bar, asserted only where a sample of a sub-100µs hop is
	// REPRESENTABLE. On a coarser clock this statistic counts timer ticks spanned by the
	// loop, not requests over budget (CLOCK-QUANTIZATION note above), so gating on it tests
	// the host's timer resolution rather than fak; it is still reported, and
	// FAK_STRICT_SERVE_LATENCY=1 forces the bar for a deliberate run on a quiet host.
	switch strict := gran <= budget || os.Getenv("FAK_STRICT_SERVE_LATENCY") == "1"; {
	case strict && frac < minUnder:
		t.Errorf("#282 acceptance: only %.2f%% of requests cleared the ≤ %v bar (want ≥ %.0f%%; clock granularity %v)",
			frac*100, budget, minUnder*100, gran)
	case !strict:
		t.Logf("#282: per-request ≤ %v bar NOT gated — clock granularity %v cannot resolve it, so the %.2f%% reading counts timer ticks spanned by the loop, not slow requests (batch mean floor %v carries the gate; set FAK_STRICT_SERVE_LATENCY=1 to force)",
			budget, gran, frac*100, floorMean)
	}
}
