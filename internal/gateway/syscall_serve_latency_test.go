package gateway

import (
	"context"
	"os"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// monotonicGranularity probes the SMALLEST non-zero interval this host's monotonic clock
// can report — the quantum time.Since actually resolves, not the ns its type implies.
// It exists because the #2219 per-request bar below is only a statement about SERVE cost
// on a clock fine enough to resolve it (see that test's doc comment): Go's Windows
// nanotime reads the system interrupt time, whose period is the machine-wide timer
// resolution (15.6ms idle, 0.5-1ms once any process raises it), so on Windows every
// sample of a few-µs serve is quantized to either 0 or one whole tick and NOTHING in
// between. The min over several probes is the right estimator: preemption can only make
// an observed delta LARGER than the quantum, never smaller.
func monotonicGranularity() time.Duration {
	const probes = 16
	quantum := time.Duration(0)
	for i := 0; i < probes; i++ {
		start := time.Now()
		var d time.Duration
		for d == 0 { // spin until the clock actually advances
			d = time.Since(start)
		}
		if quantum == 0 || d < quantum {
			quantum = d
		}
	}
	return quantum
}

// TestSyscallServeLatencyDistribution is the host-tractable acceptance witness for issue
// #2219 (epic #2218, gap G1): the vDSO SERVE path — the fast path that answers a read
// LOCALLY on s.syscall without an engine round-trip — had no distribution gate. The
// sibling TestAdjudicationLatencyUnder100us gates the adjudicate hop (s.adjudicate) only;
// the vDSO fold rides a DIFFERENT path (s.syscall), so a pure-Go regression on the one
// path that crosses every served tool call of every turn (an added allocation, lock
// contention on the hit path) would trip nothing. This is that missing gate, in the same
// shape as the adjudication one: N in-process folds, a p50 bar plus a per-request p99
// bar, both with headroom stated.
//
// It measures the SERVE, not a miss: the warmup fills the vDSO tier for (tool, args) — one
// cold engine call — and every subsequent read is served locally from it. The measured
// loop is asserted to be pure vDSO hits — VDSOHits advances by one per iteration and
// EngineCalls stays flat — so a silent fallthrough to the (slower) engine path can never
// quietly pass the gate by measuring the wrong branch.
//
// Bars (measured anchor: 1-shot serve p50 ~3.4µs wall-clock,
// examples/turntax/EXAMPLE-OUTPUT.md):
//   - p50 ≤ 50µs — ~15x headroom over the ~3.4µs anchor, so an order-of-magnitude median
//     serve regression trips it while normal serves stay green.
//   - cheapest BATCH mean ≤ 25µs — the always-on, clock-granularity-immune gate (below).
//   - ≥99% of serves ≤ 100µs — the honest per-request reading, tolerant of the rare
//     GC/scheduler outlier on a loaded shared host (same tolerance the #282 gate uses) —
//     but gated on a clock that can actually RESOLVE 100µs (below).
//
// The gate is stated on the distribution, not one sample, so a stray pause cannot flip a
// green tree red.
//
// #2219 CLOCK-QUANTIZATION CORRECTION (why the per-request bar is conditional and why the
// batch mean exists). The original form of this test was a coin flip on Windows, and NOT
// because of ordinary scheduler noise: Go's Windows nanotime resolves in whole system
// timer ticks (monotonicGranularity measures ~500µs on this class of host), so a ~5µs
// serve NEVER measures as ~5µs — it measures 0 (the serve fit inside the current tick) or
// ~500µs (a tick boundary fell inside it). A sample strictly between 0 and one tick is
// unrepresentable. Two consequences, both load-bearing:
//
//   - Each over-bar sample needs a tick boundary inside it, and boundaries are disjoint
//     across iterations, so the over-bar COUNT is just the number of ticks the measured
//     loop spans: overBarFrac ≈ loopWall/granularity/iters = meanServe/granularity. On
//     this host that made "≥99% ≤ 100µs" algebraically equivalent to "mean ≤ 5µs" — a bar
//     20x tighter than the stated 100µs and with ZERO headroom over the ~3.4µs anchor.
//     Observed: 98.94%-99.20% across 8 runs, straddling the 99% bar. The witness was
//     measuring the timer interrupt period, not the serve. (The sibling #282 gate survives
//     only because its hop is ~2.4µs: 2.4/500 = 0.5% over-bar, half the 1% allowance.)
//   - No rephrasing of a PER-SAMPLE ≤100µs bar can recover information a 500µs-quantized
//     clock never captured; a genuine 20x regression to 100µs/serve would still measure 0
//     on every sample that missed a tick. So the regression signal has to be read off an
//     interval long enough for the clock to resolve it.
//
// Hence: the strict per-request bar still runs UNCHANGED wherever the clock can resolve it
// (granularity ≤ 100µs — Linux/macOS CI, where it has always been the real gate), or on
// demand via FAK_STRICT_SERVE_LATENCY=1; and an always-on batch-mean gate carries the
// regression witness everywhere. Batch wall (~500 serves, ~2.5ms) is 5x the coarsest tick,
// so quantization costs ~1µs per serve against 5x headroom, and taking the cheapest of 10
// batches rejects the preemption spikes (max samples of 1-14ms show up run to run) that
// corrupt a whole-loop mean. That is strictly MORE witness than before on a coarse-clock
// host, where p50 reads 0s and asserted nothing at all.
func TestSyscallServeLatencyDistribution(t *testing.T) {
	if raceDetectorEnabled {
		t.Skip("latency distribution gate is not meaningful under go test -race instrumentation")
	}

	const (
		p50Budget = 50 * time.Microsecond  // ~15x over the ~3.4µs serve anchor
		p99Budget = 100 * time.Microsecond // tolerant of rare host outliers
		iters     = 5000                   // enough samples for a stable p50/p99
		warmup    = 200                    // fill the vDSO tier + amortize first-call warmup
		minUnder  = 0.99                   // ≥99% of serves must clear the p99 bar
		// The granularity-immune gate: batch the measured loop and bar the FLOOR (min)
		// batch mean. ~5x headroom over the ~5µs steady-state mean — itself ~3.4µs of serve
		// plus the two time.Now() reads per iteration that ride inside the batch window —
		// so an order-of-magnitude regression on the hottest path trips it on ANY clock.
		// Why the floor and not the median: host contention is ONE-SIDED (it can only make
		// a batch slower — the median batch mean was observed at 22.9µs on a loaded box vs
		// ~5µs quiet), so the cheapest of N batches is the low-variance estimator of the
		// path's true cost, while a real regression is systemic and lifts EVERY batch,
		// floor included. That buys a tighter bar and less flake at the same time.
		batch      = 500                   // 10 batches of 500; batch wall (~2.5ms) >> one 500µs tick
		meanBudget = 25 * time.Microsecond // ~5x over the ~5µs measured floor mean
		// A read-shaped, idempotent tool; readOnly so buildCall stamps the readOnly+idempotent
		// hints the vDSO fill gate needs. The sharing harness admits every call and wires a
		// FRESH vDSO, so the cache/counters are hermetic (no sibling-test contamination).
		tool = "get_doc"
		args = `{"id":"turntax-serve-2219"}`
	)

	// A fresh real vDSO registered as the kernel FastPath + Emitter (fills tier-2 from
	// EvComplete) at Global granularity — the same wiring `fak serve` uses, but hermetic.
	srv, _ := newSharingServer(t, vdso.Global)
	ctx := context.Background()

	// Quiet the per-operation structured log: it is observability emitted AFTER the verdict,
	// and a synchronous stderr write per call would dominate the measured tail and spam the
	// suite. The metrics fold stays, so the measured window is the serve work.
	srv.logf = func(string, ...any) {}

	// Warm the serve path: the first read misses -> engine -> fills the vDSO tier; every
	// subsequent read is served locally from it. Assert the tool ADMITs — a silent DENY
	// would make us measure the wrong (cheaper) branch and under-report the real cost.
	for i := 0; i < warmup; i++ {
		wv, _, err := srv.syscall(ctx, tool, args, true, "", "warmup")
		if err != nil {
			t.Fatalf("serve warmup: %v", err)
		}
		if wv.Kind != "ALLOW" {
			t.Fatalf("warmup verdict = %q, want ALLOW (must measure the served read path)", wv.Kind)
		}
	}

	hitsBefore := srv.k.Counters().VDSOHits
	engineBefore := srv.k.Counters().EngineCalls

	// Measure twice over the same folds: per-sample durations for the distribution, and
	// per-BATCH wall for the granularity-immune mean. The batch window brackets `batch`
	// serves in ONE pair of clock reads, so it resolves on a coarse clock where an
	// individual sample cannot (see the #2219 correction above).
	durs := make([]time.Duration, 0, iters)
	batchMeans := make([]time.Duration, 0, iters/batch)
	for b := 0; b < iters/batch; b++ {
		batchStart := time.Now()
		for i := 0; i < batch; i++ {
			start := time.Now()
			wv, _, err := srv.syscall(ctx, tool, args, true, "", "serve-bench")
			d := time.Since(start)
			if err != nil {
				t.Fatalf("serve iter %d: %v", b*batch+i, err)
			}
			if wv.Kind != "ALLOW" {
				t.Fatalf("iter %d verdict = %q, want ALLOW", b*batch+i, wv.Kind)
			}
			durs = append(durs, d)
		}
		batchMeans = append(batchMeans, time.Since(batchStart)/batch)
	}

	// Prove the measured window was the SERVE path, not the engine path: every iteration
	// was a vDSO hit (VDSOHits advanced by iters) and NO engine call ran (EngineCalls flat).
	// Without this the gate could quietly measure a slower fallthrough and still pass.
	if got := srv.k.Counters().VDSOHits - hitsBefore; got < int64(iters) {
		t.Fatalf("measured window served only %d of %d reads from the vDSO — not the serve path", got, iters)
	}
	if got := srv.k.Counters().EngineCalls; got != engineBefore {
		t.Fatalf("EngineCalls %d -> %d during the measured window — a served read fell through to the engine", engineBefore, got)
	}

	sort.Slice(durs, func(i, j int) bool { return durs[i] < durs[j] })
	p50 := durs[len(durs)*50/100]
	p99 := durs[len(durs)*99/100]
	maxd := durs[len(durs)-1]

	sort.Slice(batchMeans, func(i, j int) bool { return batchMeans[i] < batchMeans[j] })
	floorMean, medMean := batchMeans[0], batchMeans[len(batchMeans)/2]

	under := 0
	for _, d := range durs {
		if d <= p99Budget {
			under++
		}
	}
	frac := float64(under) / float64(len(durs))

	// The clock's own quantum, reported alongside the numbers it explains: a reader seeing
	// p50=0s with max in the ms has to be able to tell "the serves are sub-tick" from "the
	// serves are slow", and granularity is the fact that settles it.
	gran := monotonicGranularity()

	t.Logf("#2219 vDSO serve latency over %d in-process folds (%d batches of %d): p50=%v p99=%v max=%v; %.2f%% ≤ %v; per-serve batch mean floor=%v median=%v; clock granularity=%v",
		iters, iters/batch, batch, p50, p99, maxd, frac*100, p99Budget, floorMean, medMean, gran)

	if p50 > p50Budget {
		t.Errorf("#2219: median vDSO serve latency %v exceeds the ≤ %v bar (order-of-magnitude regression on the hottest path)", p50, p50Budget)
	}
	// Always on, on every host: the regression witness that does not depend on the clock
	// resolving a single serve.
	if floorMean > meanBudget {
		t.Errorf("#2219: cheapest per-serve batch mean %v exceeds the ≤ %v bar — every one of %d batches was slower than the bar, so this is a systemic serve regression, not host contention (batch means=%v)",
			floorMean, meanBudget, len(batchMeans), batchMeans)
	}
	// The per-request tail bar, asserted only where a sample of a sub-100µs serve is
	// REPRESENTABLE. On a coarser clock this statistic counts timer ticks spanned, not
	// serves over budget (#2219 correction above), so gating on it tests the host's timer
	// resolution; it is still reported, and FAK_STRICT_SERVE_LATENCY=1 forces the bar for a
	// deliberate run on a quiet host.
	switch strict := gran <= p99Budget || os.Getenv("FAK_STRICT_SERVE_LATENCY") == "1"; {
	case strict && frac < minUnder:
		t.Errorf("#2219: only %.2f%% of serves cleared the ≤ %v bar (want ≥ %.0f%%; clock granularity %v)", frac*100, p99Budget, minUnder*100, gran)
	case !strict:
		t.Logf("#2219: per-request ≤ %v bar NOT gated — clock granularity %v cannot resolve it, so the %.2f%% reading counts timer ticks spanned by the loop, not slow serves (batch mean floor %v carries the gate; set FAK_STRICT_SERVE_LATENCY=1 to force)",
			p99Budget, gran, frac*100, floorMean)
	}
}
