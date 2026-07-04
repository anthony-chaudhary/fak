package gateway

import (
	"context"
	"sort"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

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
//   - ≥99% of serves ≤ 100µs — the honest per-request reading, tolerant of the rare
//     GC/scheduler outlier on a loaded shared host (same tolerance the #282 gate uses).
//
// The gate is stated on the distribution, not one sample, so a stray pause cannot flip a
// green tree red.
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

	durs := make([]time.Duration, iters)
	for i := 0; i < iters; i++ {
		start := time.Now()
		wv, _, err := srv.syscall(ctx, tool, args, true, "", "serve-bench")
		d := time.Since(start)
		if err != nil {
			t.Fatalf("serve iter %d: %v", i, err)
		}
		if wv.Kind != "ALLOW" {
			t.Fatalf("iter %d verdict = %q, want ALLOW", i, wv.Kind)
		}
		durs[i] = d
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

	under := 0
	for _, d := range durs {
		if d <= p99Budget {
			under++
		}
	}
	frac := float64(under) / float64(len(durs))

	t.Logf("#2219 vDSO serve latency over %d in-process folds: p50=%v p99=%v max=%v; %.2f%% ≤ %v",
		iters, p50, p99, maxd, frac*100, p99Budget)

	if p50 > p50Budget {
		t.Errorf("#2219: median vDSO serve latency %v exceeds the ≤ %v bar (order-of-magnitude regression on the hottest path)", p50, p50Budget)
	}
	if frac < minUnder {
		t.Errorf("#2219: only %.2f%% of serves cleared the ≤ %v bar (want ≥ %.0f%%)", frac*100, p99Budget, minUnder*100)
	}
}
