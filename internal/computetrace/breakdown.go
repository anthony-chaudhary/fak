// breakdown.go — the same-run end-to-end time breakdown of a compute trace.
//
// A trace artifact records the engine's timed events (kernels, copies, weight
// loads, syncs, host CPU work) with their wall-clock intervals. BreakdownOf
// folds those events into the six objective classes and reports how much of the
// traced span each class accounts for, plus the unexplained remainder (the span
// time no event covers). It is the reader that turns a captured trace into the
// "where did the time go" answer an agent needs without hand-digging the raw
// events: `fak compute-trace breakdown --in trace.json`.
package computetrace

import (
	"sort"
	"strings"
)

// The closed time-breakdown class vocabulary. Every event is attributed to
// exactly one class; the span time no event covers is reported as Unexplained.
const (
	ClassKernel = "kernel" // device compute (GEMM, attention, ...)
	ClassCopy   = "copy"   // host<->device and device<->device transfers
	ClassWeight = "weight" // weight loading / materialization
	ClassSync   = "sync"   // device synchronization / barriers
	ClassCPU    = "cpu"    // host-side compute
	ClassOther  = "other"  // timed work not in the classes above
)

// BreakdownClasses is the closed class vocabulary in report order.
var BreakdownClasses = []string{ClassKernel, ClassCopy, ClassWeight, ClassSync, ClassCPU, ClassOther}

// Breakdown is the same-run end-to-end time breakdown of one trace artifact.
type Breakdown struct {
	// SpanNS is the wall interval from the first event start to the last event end.
	SpanNS int64 `json:"span_ns"`
	// CoveredNS is the time covered by at least one event (the union of intervals).
	CoveredNS int64 `json:"covered_ns"`
	// UnexplainedNS is SpanNS - CoveredNS: the traced time no event accounts for.
	UnexplainedNS int64 `json:"unexplained_ns"`
	// ClassNS[class] is the summed DurationNS of the events attributed to class.
	// When events of different classes overlap, the class sums can exceed CoveredNS.
	ClassNS map[string]int64 `json:"class_ns"`
	// Events is the number of events folded; Dropped the artifact's dropped counter.
	Events  int    `json:"events"`
	Dropped uint64 `json:"dropped_events"`
}

// BreakdownOf folds a trace artifact into its six-class time breakdown. An
// artifact with no events yields a zero breakdown.
func BreakdownOf(a Artifact) Breakdown {
	b := Breakdown{ClassNS: map[string]int64{}, Events: len(a.Events), Dropped: a.Dropped}
	if len(a.Events) == 0 {
		return b
	}
	ivs := make([][2]int64, 0, len(a.Events))
	for _, e := range a.Events {
		start := e.StartedAt.UnixNano()
		ivs = append(ivs, [2]int64{start, start + e.DurationNS})
		b.ClassNS[classifyEvent(e)] += e.DurationNS
	}
	var minStart, maxEnd int64
	for i, iv := range ivs {
		if i == 0 || iv[0] < minStart {
			minStart = iv[0]
		}
		if i == 0 || iv[1] > maxEnd {
			maxEnd = iv[1]
		}
	}
	b.SpanNS = maxEnd - minStart
	b.CoveredNS = unionNS(ivs)
	b.UnexplainedNS = b.SpanNS - b.CoveredNS
	if b.UnexplainedNS < 0 {
		b.UnexplainedNS = 0
	}
	return b
}

// unionNS is the total length of the union of the half-open intervals [s,e).
// Intervals with no positive duration cover no time and are ignored.
func unionNS(ivs [][2]int64) int64 {
	sorted := make([][2]int64, 0, len(ivs))
	for _, iv := range ivs {
		if iv[1] > iv[0] {
			sorted = append(sorted, iv)
		}
	}
	if len(sorted) == 0 {
		return 0
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i][0] != sorted[j][0] {
			return sorted[i][0] < sorted[j][0]
		}
		return sorted[i][1] < sorted[j][1]
	})
	var total int64
	curS, curE := sorted[0][0], sorted[0][1]
	for _, iv := range sorted[1:] {
		if iv[0] > curE {
			total += curE - curS
			curS, curE = iv[0], iv[1]
		} else if iv[1] > curE {
			curE = iv[1]
		}
	}
	total += curE - curS
	return total
}

// classifyEvent attributes one event to exactly one breakdown class. The
// precedence is sync > weight > copy > cpu > kernel > other: a host weight load
// is weight (not cpu), a device copy is copy (not kernel), and only an event
// that is neither a sync, a load, nor a transfer falls through to the compute
// classes by device.
func classifyEvent(e Event) string {
	op := strings.ToLower(e.Operation)
	phase := strings.ToLower(e.Phase)
	kernel := strings.ToLower(e.Kernel)
	switch {
	case op == "sync" || phase == "sync" || strings.Contains(op, "sync") || strings.Contains(phase, "barrier"):
		return ClassSync
	case op == "weight_load" || phase == "weight" || strings.Contains(op, "weight"):
		return ClassWeight
	case op == "copy" || phase == "copy" || isCopyKernel(kernel):
		return ClassCopy
	case e.Device == "host" || e.Backend == "cpu-ref" || e.Backend == "cpu":
		return ClassCPU
	case phase == "kernel" || op == "matmul" || op == "matvec" || op == "attention":
		return ClassKernel
	default:
		return ClassOther
	}
}

// isCopyKernel reports whether a kernel name is a transfer (host<->device or
// device<->device) rather than compute. It mirrors the compute backend's
// transfer pseudo-kernel set without importing it: computetrace is the lower
// package the backends already call, so it cannot depend on them.
func isCopyKernel(kernel string) bool {
	switch kernel {
	case "d2d_copy", "h2d_upload", "d2d_gather", "fill", "readback", "fak_paged_kv_copy_blocks":
		return true
	}
	return strings.Contains(kernel, "copy") || strings.Contains(kernel, "upload") ||
		strings.Contains(kernel, "h2d") || strings.Contains(kernel, "d2h")
}
