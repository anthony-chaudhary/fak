// Adapted from ai-dynamo/aiperf
// from the original): the good-request counter behind goodput. User SLO thresholds
// arrive in each metric's display unit, are normalized once into the unit the
// records carry, and a request is good only when every SLO holds in that
// metric's own direction. The source's class-level registry and mutable
// thresholds are re-expressed as an explicit metric table and an immutable
// value built once per run.
package perfledger

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// GoodputMetric describes one metric an SLO may name.
type GoodputMetric struct {
	// LargerIsBetter selects the passing side of the threshold: at or above it
	// when true (throughput-like), at or below it when false (latency-like).
	LargerIsBetter bool
	// DisplayToInternal multiplies a display-unit threshold into the unit the
	// per-record values use. Zero means the two units are the same.
	DisplayToInternal float64
}

type goodputThreshold struct {
	tag            string
	value          float64
	largerIsBetter bool
}

// GoodputSLOs is a normalized, immutable SLO set. The zero value and nil both
// behave as "no SLOs configured".
type GoodputSLOs struct {
	thresholds []goodputThreshold
}

// NewGoodputSLOs normalizes display-unit thresholds against the metric table.
// It refuses tags the table does not know, non-finite thresholds and unusable
// unit factors: each would silently decide every request the same way, all
// good or all bad. Every unknown tag is reported, not just the first.
func NewGoodputSLOs(slos map[string]float64, metrics map[string]GoodputMetric) (*GoodputSLOs, error) {
	tags := make([]string, 0, len(slos))
	var unknown []string
	for tag := range slos {
		if _, ok := metrics[tag]; !ok {
			unknown = append(unknown, tag)
			continue
		}
		tags = append(tags, tag)
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		return nil, fmt.Errorf("perfledger: unknown metric tag(s) in goodput SLOs: %s", strings.Join(unknown, ", "))
	}
	sort.Strings(tags)

	out := &GoodputSLOs{thresholds: make([]goodputThreshold, 0, len(tags))}
	for _, tag := range tags {
		m := metrics[tag]
		value := slos[tag]
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("perfledger: goodput SLO for %q is not finite", tag)
		}
		if scale := m.DisplayToInternal; scale != 0 {
			if math.IsNaN(scale) || math.IsInf(scale, 0) || scale < 0 {
				return nil, fmt.Errorf("perfledger: cannot convert goodput SLO for %q: unit factor %v", tag, scale)
			}
			value *= scale
			if math.IsInf(value, 0) {
				return nil, fmt.Errorf("perfledger: goodput SLO for %q overflows after unit conversion", tag)
			}
		}
		out.thresholds = append(out.thresholds, goodputThreshold{tag: tag, value: value, largerIsBetter: m.LargerIsBetter})
	}
	return out, nil
}

// RequiredTags lists, sorted, the metrics every record must carry to be
// judged. It is nil when no SLOs are configured.
func (g *GoodputSLOs) RequiredTags() []string {
	if g == nil || len(g.thresholds) == 0 {
		return nil
	}
	tags := make([]string, len(g.thresholds))
	for i, th := range g.thresholds {
		tags[i] = th.tag
	}
	return tags
}

// Threshold returns the normalized (internal-unit) threshold for tag.
func (g *GoodputSLOs) Threshold(tag string) (float64, bool) {
	if g == nil {
		return 0, false
	}
	for _, th := range g.thresholds {
		if th.tag == tag {
			return th.value, true
		}
	}
	return 0, false
}

// IsGood reports whether record meets every SLO. A missing or NaN value fails
// its SLO, and an empty SLO set makes no request good: goodput counts requests
// that satisfied a stated objective, so "nothing stated" is zero, not all.
// This is the same all-of, fail-closed rule webbench applies to its latency
// SLOs; unlike there, a zero or negative threshold is applied, not ignored.
func (g *GoodputSLOs) IsGood(record map[string]float64) bool {
	if g == nil || len(g.thresholds) == 0 {
		return false
	}
	for _, th := range g.thresholds {
		value, ok := record[th.tag]
		if !ok || math.IsNaN(value) {
			return false
		}
		if th.largerIsBetter {
			if value < th.value {
				return false
			}
		} else if value > th.value {
			return false
		}
	}
	return true
}

// SLO tags for the measurements a Record carries.
const (
	GoodputTagE2E        = "e2e"
	GoodputTagTTFT       = "ttft"
	GoodputTagPrefillTPS = "prefill_tps"
	GoodputTagDecodeTPS  = "decode_tps"
)

// RecordGoodputMetrics is the metric table for Record: latency SLOs are stated
// in seconds against millisecond fields, and token rates are larger-is-better
// in the fields' own tok/s.
func RecordGoodputMetrics() map[string]GoodputMetric {
	return map[string]GoodputMetric{
		GoodputTagE2E:        {DisplayToInternal: 1000},
		GoodputTagTTFT:       {DisplayToInternal: 1000},
		GoodputTagPrefillTPS: {LargerIsBetter: true},
		GoodputTagDecodeTPS:  {LargerIsBetter: true},
	}
}

// GoodputRecordValues projects a Record onto its SLO tags. A failed turn has
// no values, and a field that is zero, negative or non-finite is left out: the
// ledger writes zero for "not observed", and reading that as a measured 0 ms
// would pass every latency SLO.
func GoodputRecordValues(r Record) map[string]float64 {
	if r.Error != "" {
		return nil
	}
	values := make(map[string]float64, 4)
	for tag, v := range map[string]float64{
		GoodputTagE2E:        r.E2EMS,
		GoodputTagTTFT:       r.TTFTMS,
		GoodputTagPrefillTPS: r.PrefillTPS,
		GoodputTagDecodeTPS:  r.DecodeTPS,
	} {
		if v > 0 && !math.IsInf(v, 0) {
			values[tag] = v
		}
	}
	return values
}

// GoodRecordCount counts the ledger records that meet every SLO.
func (g *GoodputSLOs) GoodRecordCount(records []Record) int {
	good := 0
	for i := range records {
		if g.IsGood(GoodputRecordValues(records[i])) {
			good++
		}
	}
	return good
}

// GoodRequestCount counts the records that meet every SLO.
func (g *GoodputSLOs) GoodRequestCount(records []map[string]float64) int {
	good := 0
	for _, record := range records {
		if g.IsGood(record) {
			good++
		}
	}
	return good
}
