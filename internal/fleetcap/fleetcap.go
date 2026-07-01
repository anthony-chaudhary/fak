// Package fleetcap is a Little's-law capacity calculator: it translates a target
// issue-resolution rate and a median agent-session duration into the number of
// concurrent workers that must be in flight to sustain that rate. It is one leaf
// of the "safe 400 GitHub issues/hour parallel-agent throughput" program (issue
// #1820, fleet-400iph).
//
// Little's law for a stable queueing system relates the long-run average number
// of items in the system (L), the average arrival rate (λ), and the average time
// an item spends in the system (W):
//
//	L = λ · W
//
// Mapped onto an issue-resolving agent fleet: an "item in the system" is an
// in-flight worker session, λ is the target issue-completion rate, and W is the
// mean time a session occupies a worker (its duration). So the number of workers
// that must run concurrently to keep the pipeline at the target rate is just
// L = λ · W. With λ = 400 issues/hour and a 10-minute median session
// (W = 10 min = 1/6 hour), L = 400 · 1/6 ≈ 66.7, i.e. 67 concurrent workers once
// rounded up — you cannot run a fractional worker.
//
// The calculator is a pure lens, not a scheduler: it does NOT launch, count, or
// observe any real worker. It answers the planning question "how many live
// workers does this target imply?" so an operator can size the fleet before
// dispatching. It is stdlib-only and imports nothing internal — off the hot path.
package fleetcap

import (
	"fmt"
	"math"
	"strings"
)

// minutesPerHour converts a session duration expressed in minutes into the hours
// unit the arrival rate uses, so that λ (per hour) · W (hours) is dimensionless.
const minutesPerHour = 60.0

// RequiredWorkers applies Little's law (L = λ·W) to return the number of
// concurrent workers required to sustain targetRatePerHour issue completions when
// each session occupies a worker for medianSessionMinutes, rounded UP to a whole
// worker (you cannot run a fractional worker).
//
// Non-positive or non-finite inputs yield 0: a rate or duration that is zero,
// negative, or NaN/Inf describes no sustained work and so requires no standing
// worker.
func RequiredWorkers(targetRatePerHour, medianSessionMinutes float64) int {
	if !valid(targetRatePerHour) || !valid(medianSessionMinutes) {
		return 0
	}
	w := medianSessionMinutes / minutesPerHour // W in hours
	l := targetRatePerHour * w                 // L = λ · W
	return int(math.Ceil(l))
}

// valid reports whether x is a usable positive, finite magnitude.
func valid(x float64) bool {
	return x > 0 && !math.IsInf(x, 0) && !math.IsNaN(x)
}

// Capacity is the resolved capacity answer for one (rate, session) pair: the
// inputs plus the exact (un-rounded) Little's-law load and the ceil'd worker
// count an operator would provision.
type Capacity struct {
	TargetRatePerHour    float64 // λ, issue completions per hour
	MedianSessionMinutes float64 // W expressed in minutes
	ExactLoad            float64 // L = λ·W, the un-rounded in-flight count
	RequiredWorkers      int     // ceil(L), the workers to provision
}

// Compute resolves a Capacity for one (rate, session) pair.
func Compute(targetRatePerHour, medianSessionMinutes float64) Capacity {
	var load float64
	if valid(targetRatePerHour) && valid(medianSessionMinutes) {
		load = targetRatePerHour * (medianSessionMinutes / minutesPerHour)
	}
	return Capacity{
		TargetRatePerHour:    targetRatePerHour,
		MedianSessionMinutes: medianSessionMinutes,
		ExactLoad:            load,
		RequiredWorkers:      RequiredWorkers(targetRatePerHour, medianSessionMinutes),
	}
}

// Row is one line of a capacity table: a median session duration and the workers
// it requires at the table's target rate.
type Row struct {
	MedianSessionMinutes float64
	ExactLoad            float64
	RequiredWorkers      int
}

// DefaultSessionMinutes is the median-session sweep the ticket's done-condition
// fixes: operators must be able to read required live workers for 5, 10, 15, and
// 30 minute median sessions.
var DefaultSessionMinutes = []float64{5, 10, 15, 30}

// Table returns the required-worker count for each session duration in
// DefaultSessionMinutes at the given target rate, in ascending duration order.
// Because workers scale linearly with W at fixed λ, the rows are monotonically
// non-decreasing in MedianSessionMinutes.
func Table(targetRatePerHour float64) []Row {
	return TableFor(targetRatePerHour, DefaultSessionMinutes)
}

// TableFor is Table over a caller-supplied set of session durations. The rows are
// returned in the order given (callers pass DefaultSessionMinutes, already
// ascending).
func TableFor(targetRatePerHour float64, sessionMinutes []float64) []Row {
	rows := make([]Row, 0, len(sessionMinutes))
	for _, m := range sessionMinutes {
		c := Compute(targetRatePerHour, m)
		rows = append(rows, Row{
			MedianSessionMinutes: c.MedianSessionMinutes,
			ExactLoad:            c.ExactLoad,
			RequiredWorkers:      c.RequiredWorkers,
		})
	}
	return rows
}

// Render renders the capacity table for targetRatePerHour as an aligned,
// human-readable block: a header naming the target rate and Little's law, then one
// line per session duration giving the required concurrent workers.
func Render(targetRatePerHour float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "Little's-law fleet capacity  (L = lambda * W)\n")
	fmt.Fprintf(&b, "target rate: %s issues/hour\n", trim(targetRatePerHour))
	fmt.Fprintf(&b, "%13s   %16s   %s\n", "session (min)", "exact load (L)", "required workers")
	for _, r := range Table(targetRatePerHour) {
		fmt.Fprintf(&b, "%13s   %16s   %d\n",
			trim(r.MedianSessionMinutes), trimFixed(r.ExactLoad), r.RequiredWorkers)
	}
	return b.String()
}

// trim formats a float without a trailing ".0" when it is integral, so "5" reads
// cleaner than "5.0" in the table.
func trim(x float64) string {
	if x == math.Trunc(x) && !math.IsInf(x, 0) {
		return fmt.Sprintf("%d", int64(x))
	}
	return fmt.Sprintf("%g", x)
}

// trimFixed formats the exact load to two decimals (e.g. "66.67"), trimming the
// fraction when the value is integral.
func trimFixed(x float64) string {
	if x == math.Trunc(x) && !math.IsInf(x, 0) {
		return fmt.Sprintf("%d", int64(x))
	}
	return fmt.Sprintf("%.2f", x)
}

// Verdict is the capacity judgment of an Estimate: whether the workers a fleet has
// available can sustain the target rate the demand side implies.
type Verdict string

const (
	// Sufficient means the available concurrency meets or exceeds the required
	// worker count (available >= required): the fleet can sustain the target rate.
	Sufficient Verdict = "SUFFICIENT"
	// UnderCapacity means the available concurrency is below the required worker
	// count (available < required): the fleet cannot sustain the target rate and
	// must add workers.
	UnderCapacity Verdict = "UNDER_CAPACITY"
)

// Estimate is a dry-run capacity assessment: the Little's-law worker demand for a
// (rate, session) pair, the available worker concurrency it is measured against,
// and the verdict + shortfall that comparison yields. It launches, counts, and
// observes NO real worker — it is the planning answer to "can this fleet sustain
// the target rate?", not a reading of one that is running.
type Estimate struct {
	Capacity                 // the demand side: target, session, exact load, required workers
	AvailableWorkers int     // the supply side: the concurrent-worker ceiling on offer
	ShortfallWorkers int     // max(0, RequiredWorkers - AvailableWorkers); 0 when sufficient
	Verdict          Verdict // SUFFICIENT when available >= required, else UNDER_CAPACITY
}

// Assess resolves an Estimate for a (rate, session) pair against availableWorkers
// concurrent workers. The verdict is UNDER_CAPACITY when the Little's-law required
// worker count exceeds what is available, otherwise SUFFICIENT — meeting demand
// exactly is sufficient (the boundary is >=). A negative availableWorkers is
// clamped to zero (no negative concurrency); zero demand (a non-positive rate or
// session) is always SUFFICIENT because it needs no standing worker.
func Assess(targetRatePerHour, medianSessionMinutes float64, availableWorkers int) Estimate {
	c := Compute(targetRatePerHour, medianSessionMinutes)
	if availableWorkers < 0 {
		availableWorkers = 0
	}
	shortfall := c.RequiredWorkers - availableWorkers
	verdict := Sufficient
	if shortfall > 0 {
		verdict = UnderCapacity
	} else {
		shortfall = 0
	}
	return Estimate{
		Capacity:         c,
		AvailableWorkers: availableWorkers,
		ShortfallWorkers: shortfall,
		Verdict:          verdict,
	}
}

// AvailableFrom folds a set of concurrency limits — a host cap, a seat inventory, a
// cadence budget, any other ceiling on how many workers may run at once — into the
// single available-worker count Assess measures demand against: the MINIMUM of the
// positive limits, because a fleet can run no more workers than its tightest
// constraint allows. Non-positive limits carry no ceiling and are ignored; if no
// limit is positive the result is 0 (no known available capacity).
func AvailableFrom(limits ...int) int {
	avail := 0
	seen := false
	for _, l := range limits {
		if l <= 0 {
			continue
		}
		if !seen || l < avail {
			avail, seen = l, true
		}
	}
	return avail
}

// Line renders the estimate as one operator-facing line: the verdict, the required
// vs available worker counts, the target rate and session it assumed, and — only
// when under capacity — the worker shortfall to close.
func (e Estimate) Line() string {
	line := fmt.Sprintf("%s: need %d workers, have %d (target %s issues/hour, %s-min sessions)",
		e.Verdict, e.RequiredWorkers, e.AvailableWorkers,
		trim(e.TargetRatePerHour), trim(e.MedianSessionMinutes))
	if e.Verdict == UnderCapacity {
		line += fmt.Sprintf("; short %d", e.ShortfallWorkers)
	}
	return line
}
