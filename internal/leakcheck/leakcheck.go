// Package leakcheck provides the three reusable PROOF primitives a memory/goroutine-leak
// sweep needs, so a regression guard is a few lines instead of a hand-rolled harness each
// time. They are the generalized forms of the proofs written by hand during the 2026-06-21
// hot-path sweep:
//
//	Stable        goroutine-count sentinel — run a body N times, assert the live goroutine
//	              count returns to baseline (catches a per-call goroutine/ticker leak).
//	              (generalizes the gateway streaming-handler sentinel)
//
//	AllocScaling  per-op allocation must not scale with a size knob — measure bytes/op at a
//	              small vs large setup, fail if the ratio blows past a bound. (catches an
//	              O(n) per-op buffer that makes a loop O(n²); generalizes the decode-scores guard)
//
//	BoundedSize   a size metric must stay ≤ cap across N ops, no matter how many pass
//	              through. (catches an unbounded map/ledger/cache; generalizes the
//	              ctxmmu / normgate / radixkv bound proofs)
//
// All three take a TB (which *testing.T / *testing.B satisfy) and FAIL the test on a
// violation; none of them PROVE
// leak-freedom in general — they pin one specific property the audit identified. They are
// machine-independent (ratios / counts / bounds, not absolute timings or byte counts), and
// allocation/goroutine measurement uses runtime.MemStats.TotalAlloc (cumulative, GC-safe)
// and runtime.NumGoroutine with a GC+poll settle window, so they are as stable as those
// runtime facilities allow. -race is the orthogonal tool for data races; this is for growth.
package leakcheck

import (
	"runtime"
	"time"
)

// TB is the slice of testing.TB the helpers use. *testing.T and *testing.B satisfy it
// structurally, so callers pass `t` unchanged; a test can also pass a recording fake to
// prove a guard FIRES (see leakcheck_test.go) without failing the parent test. Every
// Fatalf call site in this package is followed by an explicit return, so a fake Fatalf
// need not abort the goroutine for the helpers to behave correctly.
type TB interface {
	Helper()
	Errorf(format string, args ...any)
	Fatalf(format string, args ...any)
}

// StableOpts configures Stable. Zero values fall back to the documented defaults.
type StableOpts struct {
	Iters  int           // body invocations after baseline (default 40)
	Warmup int           // body invocations before the baseline sample (default 2)
	Slack  int           // goroutines allowed over baseline at the end (default 4)
	Settle time.Duration // max wait for goroutines to wind down before sampling (default 2s)
}

func (o StableOpts) norm() StableOpts {
	if o.Iters <= 0 {
		o.Iters = 40
	}
	if o.Warmup < 0 {
		o.Warmup = 0
	} else if o.Warmup == 0 {
		o.Warmup = 2
	}
	if o.Slack <= 0 {
		o.Slack = 4
	}
	if o.Settle <= 0 {
		o.Settle = 2 * time.Second
	}
	return o
}

// settle polls (GC + NumGoroutine) until the count is ≤ target or the window elapses,
// returning the final count. It lets transient goroutines (e.g. just-closed HTTP conns)
// wind down so the sample reflects steady state rather than a momentary spike.
func settle(target int, within time.Duration) int {
	deadline := time.Now().Add(within)
	for {
		runtime.GC()
		n := runtime.NumGoroutine()
		if n <= target || !time.Now().Before(deadline) {
			return n
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// Stable asserts that invoking body(i) opts.Iters times does not grow the live goroutine
// count beyond baseline+Slack. body(i) should perform one full unit of work AND any cleanup
// the caller is responsible for (e.g. cancel a context, close a response body). It is the
// per-request-goroutine-leak proof: a body that leaks one goroutine per call ends ~Iters
// goroutines over baseline, far past Slack. Returns (baseline, final) for logging.
func Stable(t TB, opts StableOpts, body func(i int)) (baseline, final int) {
	t.Helper()
	o := opts.norm()
	for i := 0; i < o.Warmup; i++ {
		body(i)
	}
	baseline = settle(0, o.Settle) // 0 target → poll the full window, return the stable count
	for i := 0; i < o.Iters; i++ {
		body(o.Warmup + i)
	}
	final = settle(baseline+o.Slack, o.Settle)
	if final > baseline+o.Slack {
		t.Errorf("goroutine leak: baseline=%d final=%d (grew by %d over %d iters; slack %d)",
			baseline, final, final-baseline, o.Iters, o.Slack)
	}
	return baseline, final
}

// AllocBytesPerOp returns the average heap bytes one op() call allocates, measured over
// `iters` calls after `warmup` calls (warmup lets any one-time growth — a scratch buffer
// reaching its high-water size — happen outside the measured window). TotalAlloc is
// cumulative and unaffected by GC, so this measures allocation TRAFFIC, not live heap.
func AllocBytesPerOp(iters, warmup int, op func()) float64 {
	if iters <= 0 {
		iters = 64
	}
	for i := 0; i < warmup; i++ {
		op()
	}
	var a, b runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&a)
	for i := 0; i < iters; i++ {
		op()
	}
	runtime.ReadMemStats(&b)
	return float64(b.TotalAlloc-a.TotalAlloc) / float64(iters)
}

// ScalingOpts configures AllocScaling.
type ScalingOpts struct {
	Small, Large int     // the size knob's two settings (Large should dwarf Small)
	Iters        int     // ops measured per setting (default 64)
	Warmup       int     // ops run before measuring per setting (default = Iters)
	MaxRatio     float64 // fail if bytes/op(Large)/bytes/op(Small) exceeds this (default 2.0)
}

// AllocScaling asserts per-op allocation does NOT scale with a size knob. setup(size)
// returns an op closure already warmed to `size` (e.g. a decode session prefilled to
// `size` tokens). AllocScaling measures bytes/op at Small and Large and fails if the
// ratio exceeds MaxRatio — the O(n²)-churn guard: an op that allocates O(size) per call
// (a per-iteration buffer sized to the context) makes the ratio track Large/Small, while a
// reused-scratch op keeps it ≈ 1. Returns the measured ratio.
func AllocScaling(t TB, opts ScalingOpts, setup func(size int) (op func())) float64 {
	t.Helper()
	iters := opts.Iters
	if iters <= 0 {
		iters = 64
	}
	warmup := opts.Warmup
	if warmup <= 0 {
		warmup = iters
	}
	maxRatio := opts.MaxRatio
	if maxRatio <= 0 {
		maxRatio = 2.0
	}
	small := AllocBytesPerOp(iters, warmup, setup(opts.Small))
	large := AllocBytesPerOp(iters, warmup, setup(opts.Large))
	if small <= 0 {
		// The small op allocates nothing per call. If the large op also allocates nothing,
		// there is no per-op churn at all — the ideal — so pass. If only the large op
		// allocates, that IS scaling from zero, so fire.
		if large <= 0 {
			return 0
		}
		t.Errorf("per-op allocation appears only at the large size "+
			"(small=0 B/op, large=%.0f B/op) — allocation scales with size from zero", large)
		return large
	}
	ratio := large / small
	if ratio > maxRatio {
		t.Errorf("per-op allocation scales with size (ratio %.2f > %.2f): "+
			"small(size=%d)=%.0f B/op, large(size=%d)=%.0f B/op — an O(n) per-op buffer regressed",
			ratio, maxRatio, opts.Small, small, opts.Large, large)
	}
	return ratio
}

// BoundedSize runs step(i) for `steps` iterations and asserts size() never exceeds cap at
// any point — the unbounded-growth guard for a map/ledger/cache. A version that only ever
// grows reaches size()==steps; a properly-bounded one plateaus at cap. Fails on the first
// breach (with the iteration index) so the regression is localized.
func BoundedSize(t TB, steps, cap int, step func(i int), size func() int) {
	t.Helper()
	for i := 0; i < steps; i++ {
		step(i)
		if s := size(); s > cap {
			t.Fatalf("size %d exceeds cap %d after %d ops (unbounded growth)", s, cap, i+1)
			return
		}
	}
}
