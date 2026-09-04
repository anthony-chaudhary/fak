// Package completiondist folds historical issue-closure durations into an
// empirical duration distribution (count, min, max, mean, nearest-rank median,
// p95, and histogram buckets) used by the capacity model to size agent fleets.
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

// ClosureSample records one historical issue closure and its elapsed duration in seconds.
type ClosureSample struct {
	Issue       int     `json:"issue"`
	DurationSec float64 `json:"duration_sec"`
}

// Bucket represents a duration histogram bin bounded by [MinSec, MaxSec).
// MaxSec == 0 denotes an unbounded upper bucket.
type Bucket struct {
	Label  string  // Human-readable interval label.
	MinSec float64 // Inclusive lower bound in seconds.
	MaxSec float64 // Exclusive upper bound in seconds; 0 means unbounded.
	Count  int     // Number of samples falling within [MinSec, MaxSec).
}

// DefaultBuckets returns standard fleet difficulty duration intervals:
// sub-15m quick closes, 15-60m normal, 1-4h hard, and unbounded >4h tails.
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

// Distribution holds folded closure-duration summary statistics and histogram buckets.
type Distribution struct {
	Count      int      // Total samples folded into the distribution.
	MinSec     float64  // Minimum duration observed in seconds.
	MaxSec     float64  // Maximum duration observed in seconds.
	MeanSec    float64  // Arithmetic mean duration in seconds.
	MedianSecV float64  // Nearest-rank 50th percentile duration in seconds.
	P95Sec     float64  // Nearest-rank 95th percentile duration in seconds.
	Buckets    []Bucket // Histogram buckets with populated sample counts.
}

// Build folds duration samples into a Distribution using DefaultBuckets.
func Build(samples []ClosureSample) Distribution {
	return BuildWith(samples, DefaultBuckets())
}

// BuildWith folds samples into a Distribution over caller-provided histogram buckets.
// Median and p95 percentiles are computed via fleetmetrics nearest-rank selection.
func BuildWith(samples []ClosureSample, buckets []Bucket) Distribution {
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

// bucketFor increments the matching bucket count for duration d.
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

// MedianSec returns the distribution's median closure duration in seconds.
func (d Distribution) MedianSec() float64 { return d.MedianSecV }

// P95 returns the 95th-percentile closure duration in seconds for tail-risk analysis.
func (d Distribution) P95() float64 { return d.P95Sec }

// RequiredWorkersAtMedian calculates required concurrent workers at targetRatePerHour
// using the empirical median closure duration via Little's law.
func (d Distribution) RequiredWorkersAtMedian(targetRatePerHour float64) int {
	return fleetcap.RequiredWorkers(targetRatePerHour, d.MedianSec()/60.0)
}

// Render formats the distribution as an operator summary with bucket counts.
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

// ParseSamples decodes a JSONL byte stream of ClosureSample records into a slice.
func ParseSamples(data []byte) ([]ClosureSample, error) {
	var out []ClosureSample
	sc := bufio.NewScanner(bytes.NewReader(data))
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
