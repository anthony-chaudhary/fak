// Adapted from ai-dynamo/aiperf src/aiperf/server_metrics/units.py:25
// Reimplemented cleanly in Go as a Prometheus/OpenMetrics unit inference
// priority chain with bounded LRU caching.

package gateway

import (
	"container/list"
	"sort"
	"strings"
	"sync"
)

// PromUnit is the canonical normalized unit string inferred for a Prometheus or
// OpenMetrics server metric.
type PromUnit string

const (
	PromUnitUnknown      PromUnit = ""
	PromUnitSeconds      PromUnit = "seconds"
	PromUnitMilliseconds PromUnit = "milliseconds"
	PromUnitMicroseconds PromUnit = "microseconds"
	PromUnitNanoseconds  PromUnit = "nanoseconds"
	PromUnitMinutes      PromUnit = "minutes"
	PromUnitHours        PromUnit = "hours"
	PromUnitDays         PromUnit = "days"
	PromUnitBytes        PromUnit = "bytes"
	PromUnitKilobytes    PromUnit = "kilobytes"
	PromUnitMegabytes    PromUnit = "megabytes"
	PromUnitGigabytes    PromUnit = "gigabytes"
	PromUnitTerabytes    PromUnit = "terabytes"
	PromUnitBits         PromUnit = "bits"
	PromUnitRatio        PromUnit = "ratio"
	PromUnitPercent      PromUnit = "percent"
	PromUnitCount        PromUnit = "count"
	PromUnitInfo         PromUnit = "info"
	PromUnitTokens       PromUnit = "tokens"
	PromUnitRequests     PromUnit = "requests"
	PromUnitWatts        PromUnit = "watts"
	PromUnitJoules       PromUnit = "joules"
	PromUnitCelsius      PromUnit = "celsius"
	PromUnitHertz        PromUnit = "hertz"
	PromUnitPerSecond    PromUnit = "per_second"
)

const defaultPromUnitCacheSize = 4096

// PromMetricMeta describes one Prometheus/OpenMetrics metric family or series
// for unit inference.
type PromMetricMeta struct {
	Name         string
	Type         string
	ExplicitUnit string
	Help         string
}

var explicitUnitAliases = map[string]PromUnit{
	"s":            PromUnitSeconds,
	"sec":          PromUnitSeconds,
	"secs":         PromUnitSeconds,
	"second":       PromUnitSeconds,
	"seconds":      PromUnitSeconds,
	"ms":           PromUnitMilliseconds,
	"msec":         PromUnitMilliseconds,
	"millisecond":  PromUnitMilliseconds,
	"milliseconds": PromUnitMilliseconds,
	"us":           PromUnitMicroseconds,
	"µs":           PromUnitMicroseconds,
	"usec":         PromUnitMicroseconds,
	"microsecond":  PromUnitMicroseconds,
	"microseconds": PromUnitMicroseconds,
	"ns":           PromUnitNanoseconds,
	"nsec":         PromUnitNanoseconds,
	"nanosecond":   PromUnitNanoseconds,
	"nanoseconds":  PromUnitNanoseconds,
	"m":            PromUnitMinutes,
	"min":          PromUnitMinutes,
	"mins":         PromUnitMinutes,
	"minute":       PromUnitMinutes,
	"minutes":      PromUnitMinutes,
	"h":            PromUnitHours,
	"hr":           PromUnitHours,
	"hrs":          PromUnitHours,
	"hour":         PromUnitHours,
	"hours":        PromUnitHours,
	"d":            PromUnitDays,
	"day":          PromUnitDays,
	"days":         PromUnitDays,
	"b":            PromUnitBytes,
	"by":           PromUnitBytes,
	"byte":         PromUnitBytes,
	"bytes":        PromUnitBytes,
	"kb":           PromUnitKilobytes,
	"kib":          PromUnitKilobytes,
	"kilobyte":     PromUnitKilobytes,
	"kilobytes":    PromUnitKilobytes,
	"kibibytes":    PromUnitKilobytes,
	"mb":           PromUnitMegabytes,
	"mib":          PromUnitMegabytes,
	"megabyte":     PromUnitMegabytes,
	"megabytes":    PromUnitMegabytes,
	"mebibytes":    PromUnitMegabytes,
	"gb":           PromUnitGigabytes,
	"gib":          PromUnitGigabytes,
	"gigabyte":     PromUnitGigabytes,
	"gigabytes":    PromUnitGigabytes,
	"gibibytes":    PromUnitGigabytes,
	"tb":           PromUnitTerabytes,
	"tib":          PromUnitTerabytes,
	"terabyte":     PromUnitTerabytes,
	"terabytes":    PromUnitTerabytes,
	"tebibytes":    PromUnitTerabytes,
	"bit":          PromUnitBits,
	"bits":         PromUnitBits,
	"ratio":        PromUnitRatio,
	"fraction":     PromUnitRatio,
	"0-1":          PromUnitRatio,
	"percent":      PromUnitPercent,
	"pct":          PromUnitPercent,
	"perc":         PromUnitPercent,
	"percentage":   PromUnitPercent,
	"%":            PromUnitPercent,
	"0-100":        PromUnitPercent,
	"count":        PromUnitCount,
	"counts":       PromUnitCount,
	"total":        PromUnitCount,
	"1":            PromUnitCount,
	"info":         PromUnitInfo,
	"token":        PromUnitTokens,
	"tokens":       PromUnitTokens,
	"request":      PromUnitRequests,
	"requests":     PromUnitRequests,
	"w":            PromUnitWatts,
	"watt":         PromUnitWatts,
	"watts":        PromUnitWatts,
	"j":            PromUnitJoules,
	"joule":        PromUnitJoules,
	"joules":       PromUnitJoules,
	"c":            PromUnitCelsius,
	"celsius":      PromUnitCelsius,
	"degc":         PromUnitCelsius,
	"hz":           PromUnitHertz,
	"hertz":        PromUnitHertz,
	"per_second":   PromUnitPerSecond,
	"/s":           PromUnitPerSecond,
	"1/s":          PromUnitPerSecond,
}

type unitPattern struct {
	pattern string
	unit    PromUnit
}

var helpScalePatterns = []unitPattern{
	{"(0-100)", PromUnitPercent},
	{"[0, 100]", PromUnitPercent},
	{"[0,100]", PromUnitPercent},
	{"(0.0-100.0)", PromUnitPercent},
	{"0 to 100", PromUnitPercent},
	{"between 0 and 100", PromUnitPercent},
	{"0% to 100%", PromUnitPercent},
	{"0-100%", PromUnitPercent},
	{"(0-1)", PromUnitRatio},
	{"[0, 1]", PromUnitRatio},
	{"[0,1]", PromUnitRatio},
	{"(0.0-1.0)", PromUnitRatio},
	{"0.0 to 1.0", PromUnitRatio},
	{"0 to 1", PromUnitRatio},
	{"between 0 and 1", PromUnitRatio},
	{"1 means 100 percent", PromUnitRatio},
	{"1 means 100%", PromUnitRatio},
}

var helpTextPatterns = sortPatternsLongestFirst([]unitPattern{
	{"in nanoseconds", PromUnitNanoseconds},
	{"unit: nanoseconds", PromUnitNanoseconds},
	{"(nanoseconds)", PromUnitNanoseconds},
	{"in microseconds", PromUnitMicroseconds},
	{"unit: microseconds", PromUnitMicroseconds},
	{"(microseconds)", PromUnitMicroseconds},
	{"in milliseconds", PromUnitMilliseconds},
	{"unit: milliseconds", PromUnitMilliseconds},
	{"(milliseconds)", PromUnitMilliseconds},
	{"in seconds", PromUnitSeconds},
	{"unit: seconds", PromUnitSeconds},
	{"(seconds)", PromUnitSeconds},
	{"in minutes", PromUnitMinutes},
	{"(minutes)", PromUnitMinutes},
	{"in hours", PromUnitHours},
	{"(hours)", PromUnitHours},
	{"in days", PromUnitDays},
	{"(days)", PromUnitDays},
	{"in terabytes", PromUnitTerabytes},
	{"in gigabytes", PromUnitGigabytes},
	{"in megabytes", PromUnitMegabytes},
	{"in kilobytes", PromUnitKilobytes},
	{"in bytes", PromUnitBytes},
	{"unit: bytes", PromUnitBytes},
	{"(bytes)", PromUnitBytes},
	{"as a percentage", PromUnitPercent},
	{"in percentage", PromUnitPercent},
	{"in percent", PromUnitPercent},
	{"in watts", PromUnitWatts},
	{"(watts)", PromUnitWatts},
	{"in joules", PromUnitJoules},
	{"(joules)", PromUnitJoules},
	{"in celsius", PromUnitCelsius},
	{"(celsius)", PromUnitCelsius},
	{"in hertz", PromUnitHertz},
	{"(hertz)", PromUnitHertz},
	{"per second", PromUnitPerSecond},
	{"(ns)", PromUnitNanoseconds},
	{"(us)", PromUnitMicroseconds},
	{"(µs)", PromUnitMicroseconds},
	{"(ms)", PromUnitMilliseconds},
	{"(s)", PromUnitSeconds},
	{"(gb)", PromUnitGigabytes},
	{"(gib)", PromUnitGigabytes},
	{"(mb)", PromUnitMegabytes},
	{"(mib)", PromUnitMegabytes},
	{"(kb)", PromUnitKilobytes},
	{"(kib)", PromUnitKilobytes},
})

var nameSuffixTable = sortPatternsLongestFirst([]unitPattern{
	{"_nanoseconds", PromUnitNanoseconds},
	{"_microseconds", PromUnitMicroseconds},
	{"_milliseconds", PromUnitMilliseconds},
	{"_seconds", PromUnitSeconds},
	{"_minutes", PromUnitMinutes},
	{"_hours", PromUnitHours},
	{"_days", PromUnitDays},
	{"_terabytes", PromUnitTerabytes},
	{"_gigabytes", PromUnitGigabytes},
	{"_megabytes", PromUnitMegabytes},
	{"_kilobytes", PromUnitKilobytes},
	{"_kibibytes", PromUnitKilobytes},
	{"_mebibytes", PromUnitMegabytes},
	{"_gibibytes", PromUnitGigabytes},
	{"_tebibytes", PromUnitTerabytes},
	{"_bytes", PromUnitBytes},
	{"_bits", PromUnitBits},
	{"_per_second", PromUnitPerSecond},
	{"_percentage", PromUnitPercent},
	{"_percent", PromUnitPercent},
	{"_perc", PromUnitPercent},
	{"_pct", PromUnitPercent},
	{"_ratio", PromUnitRatio},
	{"_tokens", PromUnitTokens},
	{"_requests", PromUnitRequests},
	{"_reqs", PromUnitRequests},
	{"_watts", PromUnitWatts},
	{"_joules", PromUnitJoules},
	{"_celsius", PromUnitCelsius},
	{"_hertz", PromUnitHertz},
	{"_hz", PromUnitHertz},
	{"_msec", PromUnitMilliseconds},
	{"_usec", PromUnitMicroseconds},
	{"_nsec", PromUnitNanoseconds},
	{"_sec", PromUnitSeconds},
	{"_ms", PromUnitMilliseconds},
	{"_us", PromUnitMicroseconds},
	{"_ns", PromUnitNanoseconds},
	{"_tb", PromUnitTerabytes},
	{"_gb", PromUnitGigabytes},
	{"_mb", PromUnitMegabytes},
	{"_kb", PromUnitKilobytes},
})

func sortPatternsLongestFirst(in []unitPattern) []unitPattern {
	out := append([]unitPattern(nil), in...)
	sort.Slice(out, func(i, j int) bool {
		if len(out[i].pattern) != len(out[j].pattern) {
			return len(out[i].pattern) > len(out[j].pattern)
		}
		return out[i].pattern < out[j].pattern
	})
	return out
}

// ParseOpenMetricsUnitLine parses an OpenMetrics `# UNIT <metric> <unit>`
// header line and returns the metric name and normalized PromUnit.
func ParseOpenMetricsUnitLine(line string) (metric string, unit PromUnit, ok bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "#") {
		return "", PromUnitUnknown, false
	}
	rest := strings.TrimSpace(trimmed[1:])
	if !strings.HasPrefix(rest, "UNIT") {
		return "", PromUnitUnknown, false
	}
	fields := strings.Fields(rest)
	if len(fields) < 3 || fields[0] != "UNIT" {
		return "", PromUnitUnknown, false
	}
	u := NormalizePrometheusUnit(strings.Join(fields[2:], " "))
	if u == PromUnitUnknown {
		return "", PromUnitUnknown, false
	}
	return fields[1], u, true
}

// NormalizePrometheusUnit maps unit synonyms (for example "s", "ms", "By", "%")
// onto canonical PromUnit values, returning lowercased unknown tokens unchanged.
func NormalizePrometheusUnit(raw string) PromUnit {
	s := strings.ToLower(strings.TrimSpace(raw))
	if s == "" {
		return PromUnitUnknown
	}
	if u, ok := explicitUnitAliases[s]; ok {
		return u
	}
	return PromUnit(s)
}

type promUnitCacheEntry struct {
	key  PromMetricMeta
	unit PromUnit
}

// PromUnitInferrer resolves normalized units for Prometheus metrics using the
// priority chain: explicit # UNIT > Help scale cue > Help text > metric type
// and longest-first name suffix rules, backed by a bounded LRU cache.
type PromUnitInferrer struct {
	mu       sync.Mutex
	capacity int
	ll       *list.List
	cache    map[PromMetricMeta]*list.Element
}

// NewPromUnitInferrer constructs a bounded LRU unit inferrer. Values <= 0
// select defaultPromUnitCacheSize (4096).
func NewPromUnitInferrer(capacity int) *PromUnitInferrer {
	if capacity <= 0 {
		capacity = defaultPromUnitCacheSize
	}
	return &PromUnitInferrer{
		capacity: capacity,
		ll:       list.New(),
		cache:    make(map[PromMetricMeta]*list.Element),
	}
}

var defaultPromUnitInferrer = NewPromUnitInferrer(defaultPromUnitCacheSize)

// InferPrometheusUnit infers the canonical unit for meta via the shared bounded
// LRU inferrer.
func InferPrometheusUnit(meta PromMetricMeta) PromUnit {
	return defaultPromUnitInferrer.Infer(meta)
}

// Infer resolves and caches the inferred unit for meta.
func (inf *PromUnitInferrer) Infer(meta PromMetricMeta) PromUnit {
	if inf == nil {
		return inferPrometheusUnitUncached(meta)
	}
	inf.mu.Lock()
	if elem, ok := inf.cache[meta]; ok {
		inf.ll.MoveToFront(elem)
		u := elem.Value.(*promUnitCacheEntry).unit
		inf.mu.Unlock()
		return u
	}
	inf.mu.Unlock()

	u := inferPrometheusUnitUncached(meta)

	inf.mu.Lock()
	defer inf.mu.Unlock()
	if elem, ok := inf.cache[meta]; ok {
		inf.ll.MoveToFront(elem)
		return elem.Value.(*promUnitCacheEntry).unit
	}
	if inf.ll.Len() >= inf.capacity {
		if tail := inf.ll.Back(); tail != nil {
			inf.ll.Remove(tail)
			delete(inf.cache, tail.Value.(*promUnitCacheEntry).key)
		}
	}
	elem := inf.ll.PushFront(&promUnitCacheEntry{key: meta, unit: u})
	inf.cache[meta] = elem
	return u
}

func inferPrometheusUnitUncached(meta PromMetricMeta) PromUnit {
	// 1. Explicit # UNIT metadata wins unconditionally.
	if u := NormalizePrometheusUnit(meta.ExplicitUnit); u != PromUnitUnknown {
		return u
	}

	helpLower := strings.ToLower(meta.Help)

	// 2. Description scale cues override description prose and misleading suffixes
	// (e.g. *_perc documented as "(0-1)" or "1 means 100 percent").
	if helpLower != "" {
		for _, p := range helpScalePatterns {
			if strings.Contains(helpLower, p.pattern) {
				return p.unit
			}
		}
		// 3. Description text unit phrasing.
		for _, p := range helpTextPatterns {
			if strings.Contains(helpLower, p.pattern) {
				return p.unit
			}
		}
	}

	// 4. Metric type & longest-first Prometheus naming suffix conventions.
	nameLower := strings.ToLower(strings.TrimSpace(meta.Name))
	typeLower := strings.ToLower(strings.TrimSpace(meta.Type))

	if typeLower == "info" || strings.HasSuffix(nameLower, "_info") {
		return PromUnitInfo
	}
	if strings.HasSuffix(nameLower, "_count") {
		return PromUnitCount
	}

	// Histogram/summary _bucket and _sum inherit their base metric's unit.
	base := nameLower
	switch {
	case strings.HasSuffix(base, "_bucket"):
		base = strings.TrimSuffix(base, "_bucket")
	case strings.HasSuffix(base, "_sum"):
		base = strings.TrimSuffix(base, "_sum")
	}

	isCounter := typeLower == "counter"
	if strings.HasSuffix(base, "_total") {
		isCounter = true
		base = strings.TrimSuffix(base, "_total")
	}

	for _, p := range nameSuffixTable {
		if strings.HasSuffix(base, p.pattern) {
			return p.unit
		}
	}

	if isCounter {
		return PromUnitCount
	}
	return PromUnitUnknown
}
