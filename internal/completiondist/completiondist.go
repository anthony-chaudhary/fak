// Package completiondist folds historical issue-closure durations into a
// duration DISTRIBUTION the capacity model can consume, instead of the single
// assumed median that internal/fleetcap otherwise takes on faith. It is one leaf
// of the "safe 400 GitHub issues/hour parallel-agent throughput" program (issue
// #1820, fleet-400iph; this leaf is #1821).
//
// # Why a distribution, not a median
//
// internal/fleetcap.RequiredWorkers turns a target issue-rate and a median
// session duration into a required concurrent-worker count via Little's law. If
// you feed it a median you GUESSED, the whole capacity plan inherits that guess.
// This package closes the loop: it takes the durations we ACTUALLY observed
// closing past issues, folds them into a Distribution (count, min/max, median,
// p95, mean, and difficulty-mirroring histogram buckets), and hands
// fleetcap.RequiredWorkers a MedianSec() drawn from real data. The p95 sits
// beside the median so an operator can see the heavy tail a median alone hides —
// the same reason internal/fleetmetrics reports p95 next to p50.
//
// # Percentile method
//
// Median and p95 are computed by internal/fleetmetrics.Percentiles, which uses
// the NEAREST-RANK method (no interpolation): the returned value is always an
// element that is actually present in the input, so a fixture's expected median
// and p95 are trivially hand-verifiable. This package does NOT re-implement the
// percentile fold; it reuses fleetmetrics so there is one canonical method.
//
// Everything here is deterministic (no time.Now), stdlib + internal/fleetmetrics
// + internal/fleetcap only, and off the hot path. An empty sample set folds to a
// zero-valued Distribution.
package completiondist

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleetcap"
	"github.com/anthony-chaudhary/fak/internal/fleetmetrics"
)

// ClosureSample is one historical issue closure: which issue it was and how long
// it took to close, in seconds. DurationSec is wall-clock from open (or from the
// moment a worker picked it up) to close — the fold does not care which epoch, it
// only folds the durations it is handed.
type ClosureSample struct {
	Issue       int     `json:"issue"`
	DurationSec float64 `json:"duration_sec"`
}

// Bucket is one histogram bucket of the closure-duration distribution: a
// human label, its half-open second bounds [MinSec, MaxSec), and how many
// samples fell into it. The top bucket carries MaxSec == 0 to mean "no upper
// bound" (everything at or above MinSec).
type Bucket struct {
	Label  string  // human label, e.g. "<15m" or "1-4h"
	MinSec float64 // inclusive lower bound in seconds
	MaxSec float64 // exclusive upper bound in seconds; 0 == unbounded (top bucket)
	Count  int     // samples that fell in [MinSec, MaxSec)
}

// DefaultBuckets mirrors the difficulty buckets used elsewhere in the fleet: a
// sub-15-minute quick close, a 15-60 minute normal, a 1-4 hour hard, and an
// unbounded >4h tail. Callers that want their own edges pass them to
// BuildWith; Build uses these.
func DefaultBuckets() []Bucket {
	const (
		min15 = 15 * 60.0     // 900s
		min60 = 60 * 60.0     // 3600s
		hr4   = 4 * 60 * 60.0 // 14400s
	)
	return []Bucket{
		{Label: "<15m", MinSec: 0, MaxSec: min15},
		{Label: "15-60m", MinSec: min15, MaxSec: min60},
		{Label: "1-4h", MinSec: min60, MaxSec: hr4},
		{Label: ">4h", MinSec: hr4, MaxSec: 0},
	}
}

// Distribution is the folded closure-duration distribution: the summary
// statistics plus the populated histogram buckets. All durations are in seconds.
// A Distribution built from no samples is the zero value with empty Buckets.
type Distribution struct {
	Count      int      // number of samples folded
	MinSec     float64  // smallest duration; 0 when Count == 0
	MaxSec     float64  // largest duration; 0 when Count == 0
	MeanSec    float64  // arithmetic mean; 0 when Count == 0
	MedianSecV float64  // p50 (nearest-rank); 0 when Count == 0
	P95Sec     float64  // p95 (nearest-rank); 0 when Count == 0
	Buckets    []Bucket // histogram buckets with populated Counts
}

// Build folds samples into a Distribution using DefaultBuckets. It is the common
// entry point; BuildWith takes caller-supplied buckets.
func Build(samples []ClosureSample) Distribution {
	return BuildWith(samples, DefaultBuckets())
}

// BuildWith folds samples into a Distribution over the given histogram buckets.
// Median and p95 are computed via fleetmetrics.Percentiles (nearest-rank), so
// they are always values present in the input. Buckets are copied so the
// caller's slice is left untouched; each returned bucket carries the count of
// samples whose DurationSec fell in its half-open [MinSec, MaxSec) range (the
// top bucket, MaxSec == 0, is unbounded above). An empty sample set yields the
// zero-valued Distribution with the buckets present but all counts 0.
func BuildWith(samples []ClosureSample, buckets []Bucket) Distribution {
	// Copy the caller's bucket edges so we own the Count fields we write.
	bs := make([]Bucket, len(buckets))
	copy(bs, buckets)

	if len(samples) == 0 {
		return Distribution{Buckets: bs}
	}

	durations := make([]float64, len(samples))
	var sum, min, max float64
	for i, s := range samples {
		d := s.DurationSec
		durations[i] = d
		sum += d
		if i == 0 || d < min {
			min = d
		}
		if i == 0 || d > max {
			max = d
		}
		bucketFor(bs, d)
	}

	pct := fleetmetrics.Percentiles(durations, 50, 95)

	return Distribution{
		Count:      len(samples),
		MinSec:     min,
		MaxSec:     max,
		MeanSec:    sum / float64(len(samples)),
		MedianSecV: pct[50],
		P95Sec:     pct[95],
		Buckets:    bs,
	}
}

// bucketFor increments the count of the first bucket whose [MinSec, MaxSec)
// range contains d. A bucket with MaxSec == 0 is unbounded above, so it catches
// everything at or above its MinSec. Samples below the lowest MinSec fall
// through uncounted (DefaultBuckets start at 0, so nothing is lost there).
func bucketFor(buckets []Bucket, d float64) {
	for i := range buckets {
		b := buckets[i]
		if d < b.MinSec {
			continue
		}
		if b.MaxSec == 0 || d < b.MaxSec {
			buckets[i].Count++
			return
		}
	}
}

// MedianSec is the distribution's median closure duration in seconds — the value
// the capacity model should feed to fleetcap.RequiredWorkers instead of a
// hard-coded median. fleetcap takes minutes, so a caller composes:
//
//	dist := completiondist.Build(samples)
//	workers := fleetcap.RequiredWorkers(400, dist.MedianSec()/60)
//
// On an empty distribution this is 0, which fleetcap treats as "no standing
// worker required" (a zero/negative duration describes no sustained work).
func (d Distribution) MedianSec() float64 { return d.MedianSecV }

// P95 is the 95th-percentile closure duration in seconds. It is exposed so a
// tail-aware sizing can provision against the p95 rather than the median when a
// heavy tail would otherwise starve the fleet.
func (d Distribution) P95() float64 { return d.P95Sec }

// RequiredWorkersAtMedian is the direct composition this leaf exists to enable:
// it sizes the fleet at targetRatePerHour issue completions using THIS
// distribution's real median (converted seconds→minutes) as Little's-law W,
// instead of a guessed median. It is a thin convenience over
// fleetcap.RequiredWorkers so the composition has a single, testable name.
func (d Distribution) RequiredWorkersAtMedian(targetRatePerHour float64) int {
	return fleetcap.RequiredWorkers(targetRatePerHour, d.MedianSec()/60.0)
}

// Render formats the distribution as a compact operator block: the count and
// summary percentiles on one line, then one line per populated bucket. An empty
// distribution reports "no samples" rather than a misleading row of zeros.
func (d Distribution) Render() string {
	if d.Count == 0 {
		return "issue-closure duration: no samples"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "issue-closure duration (n=%d): min=%.0fs p50=%.0fs mean=%.0fs p95=%.0fs max=%.0fs\n",
		d.Count, d.MinSec, d.MedianSecV, d.MeanSec, d.P95Sec, d.MaxSec)
	for _, bk := range d.Buckets {
		fmt.Fprintf(&b, "  %-7s %d\n", bk.Label, bk.Count)
	}
	return b.String()
}

// ParseSamples reads a JSONL stream of ClosureSample records (one JSON object
// per line) and returns them in file order. Blank lines and lines that are only
// whitespace are skipped; any line that is not valid JSON, or whose
// duration_sec is negative, is an error naming the 1-based line number so a bad
// fixture is easy to locate. It is deterministic and does not touch the clock.
func ParseSamples(data []byte) ([]ClosureSample, error) {
	var out []ClosureSample
	sc := bufio.NewScanner(bytes.NewReader(data))
	// Allow long lines (default token cap is 64KiB; closure rows are tiny, but
	// be generous so a wide line does not silently truncate).
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := bytes.TrimSpace(sc.Bytes())
		if len(raw) == 0 {
			continue
		}
		var s ClosureSample
		if err := json.Unmarshal(raw, &s); err != nil {
			return nil, fmt.Errorf("line %d: %w", line, err)
		}
		if s.DurationSec < 0 {
			return nil, fmt.Errorf("line %d: negative duration_sec %g", line, s.DurationSec)
		}
		out = append(out, s)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("scan: %w", err)
	}
	return out, nil
}
