// Package launchlatency is a pure fold over worker-LAUNCH records: the time
// between a dispatch decision ("spawn worker W now") and that worker's first
// heartbeat ("W is alive and running"). It exists so an operator can see how
// long dispatch-to-alive takes across the fleet — the p50/p95 launch latency
// and where the mass sits across caller-chosen buckets — before deciding
// whether a slow launch tail (cold caches, container pulls, quota back-pressure)
// is eating into the 400-issues/hour throughput budget.
//
// Launch latency is distinct from worker-SESSION duration (internal/fleetmetrics):
// session duration is how long a worker RAN once alive; launch latency is the
// dead time BEFORE it was alive. Both share the same nearest-rank percentile
// fold, so this package reuses fleetmetrics.Percentiles rather than
// re-implementing it — it hands the launch latencies to that fold and keeps only
// the launch-specific record shape and histogram here.
//
// # Determinism
//
// Every time is a fixed input (dispatch and heartbeat unix seconds passed in by
// the caller); nothing here calls time.Now(). A fixture with known pairs
// therefore folds to hand-verifiable bucket counts and percentiles.
//
// # Edge cases
//
//   - empty input          -> every percentile is 0; every bucket count is 0.
//   - negative latency      -> a heartbeat stamped BEFORE its dispatch (clock
//     skew or a mis-ordered ledger) is clamped to 0, and the count of such
//     records is surfaced separately as Negatives so it is visible rather than
//     silently folded into the first bucket as a legitimate fast launch.
//
// Everything here is stdlib + internal/fleetmetrics only, imports nothing else,
// and is off the hot path.
package launchlatency

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
)

// Launch is one worker-launch record. DispatchSec is the unix time (seconds) at
// which the dispatcher DECIDED to spawn WorkerID; HeartbeatSec is the unix time
// at which that worker reported its first alive heartbeat. Both are caller-fixed
// inputs so the fold is deterministic.
type Launch struct {
	WorkerID     string
	DispatchSec  float64
	HeartbeatSec float64
}

// LatencySec is the launch latency of one record: heartbeat minus dispatch, in
// seconds. A heartbeat stamped before its dispatch (negative raw latency) is
// clamped to 0 so a skewed clock cannot pull a percentile below zero; use
// IsNegative to detect the skew.
func (l Launch) LatencySec() float64 {
	d := l.HeartbeatSec - l.DispatchSec
	if d < 0 {
		return 0
	}
	return d
}

// IsNegative reports whether this record's heartbeat is stamped strictly before
// its dispatch — a clock-skew / mis-ordered-ledger signal that LatencySec()
// clamps to 0.
func (l Launch) IsNegative() bool {
	return l.HeartbeatSec < l.DispatchSec
}

// BucketCount is one row of the latency histogram: every latency in
// [Lo, Hi) seconds fell into this bucket. The final bucket is open-topped, so
// its Hi is +Inf and Unbounded is true.
type BucketCount struct {
	Lo        float64
	Hi        float64
	Unbounded bool // true for the final open-topped [lastEdge, +Inf) bucket
	Count     int
}

// Histogram folds the launch latencies into half-open buckets defined by the
// caller-provided ascending bucket edges (e.g. []float64{1, 2, 5, 10, 30}).
// Given edges e0<e1<...<e_{k-1} the buckets are:
//
//	[0,   e0)
//	[e0,  e1)
//	...
//	[e_{k-2}, e_{k-1})
//	[e_{k-1}, +Inf)        // open-topped final bucket
//
// so k edges yield k+1 buckets. A latency exactly on an edge falls into the
// bucket that STARTS at that edge (half-open, lower-closed). Negative-latency
// records are clamped to 0 (via LatencySec) and therefore land in the first
// bucket; their raw count is also returned separately by Negatives so the caller
// can distinguish "genuinely sub-e0 fast launch" from "clock skew".
//
// Edges are used as given; if the caller passes an unsorted slice it is sorted
// on a copy first (the caller's slice is left untouched). Empty edges yield a
// single [0, +Inf) bucket holding every record. Empty launches yield all-zero
// counts.
func Histogram(launches []Launch, buckets []float64) []BucketCount {
	edges := make([]float64, len(buckets))
	copy(edges, buckets)
	sort.Float64s(edges)

	out := make([]BucketCount, 0, len(edges)+1)
	lo := 0.0
	for _, e := range edges {
		out = append(out, BucketCount{Lo: lo, Hi: e})
		lo = e
	}
	out = append(out, BucketCount{Lo: lo, Hi: 0, Unbounded: true})

	for _, l := range launches {
		lat := l.LatencySec()
		// Find the last bucket whose Lo <= lat. Because buckets are contiguous
		// and the final one is open-topped, exactly one always matches.
		idx := len(out) - 1
		for i := range out {
			if out[i].Unbounded {
				continue
			}
			if lat < out[i].Hi {
				idx = i
				break
			}
		}
		out[idx].Count++
	}
	return out
}

// Negatives counts the records whose heartbeat is stamped strictly before their
// dispatch (clock skew / mis-ordered ledger). These are clamped to 0 by
// LatencySec and thus fold into the first histogram bucket, so this separate
// count is how a caller sees the skew rather than mistaking it for fast launches.
func Negatives(launches []Launch) int {
	n := 0
	for _, l := range launches {
		if l.IsNegative() {
			n++
		}
	}
	return n
}

// latencies projects the clamped LatencySec out of a slice of launches.
func latencies(launches []Launch) []float64 {
	ls := make([]float64, len(launches))
	for i, l := range launches {
		ls[i] = l.LatencySec()
	}
	return ls
}

// P50P95 returns the median and 95th-percentile launch latency, in seconds, over
// the records. It reuses the nearest-rank fold in internal/fleetmetrics rather
// than re-deriving percentiles. Empty input yields 0/0; a single record yields
// that record's (clamped) latency for both.
func P50P95(launches []Launch) (p50, p95 float64) {
	m := fleetmetrics.Percentiles(latencies(launches), 50, 95)
	return m[50], m[95]
}

// Render formats the launch-latency histogram plus p50/p95 for operator output.
// Empty input is reported explicitly. Example over a fixture:
//
//	worker launch latency (n=8): p50=3.0s p95=25.0s
//	  [0.0,1.0)s   1
//	  [1.0,2.0)s   1
//	  [2.0,5.0)s   3
//	  [5.0,10.0)s  1
//	  [10.0,30.0)s 1
//	  [30.0,+Inf)s 1
//
// A negative-latency count, if any, is appended to the header so clock skew is
// visible on the same line as the summary.
func Render(launches []Launch, buckets []float64) string {
	n := len(launches)
	if n == 0 {
		return "worker launch latency (n=0): no launches"
	}
	p50, p95 := P50P95(launches)

	var b strings.Builder
	fmt.Fprintf(&b, "worker launch latency (n=%d): p50=%.1fs p95=%.1fs", n, p50, p95)
	if neg := Negatives(launches); neg > 0 {
		fmt.Fprintf(&b, " (negative-latency records clamped to 0: %d)", neg)
	}
	for _, bc := range Histogram(launches, buckets) {
		if bc.Unbounded {
			fmt.Fprintf(&b, "\n  [%.1f,+Inf)s %d", bc.Lo, bc.Count)
		} else {
			fmt.Fprintf(&b, "\n  [%.1f,%.1f)s %d", bc.Lo, bc.Hi, bc.Count)
		}
	}
	return b.String()
}
