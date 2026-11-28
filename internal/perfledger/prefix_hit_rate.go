// prefix_hit_rate.go
//
// NOTICE / attribution:
//
// file src/aiperf/dataset/synthesis/prefix_analyzer.py
// (Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES).
// Licensed under the Apache License, Version 2.0 (Apache-2.0).
// SPDX-License-Identifier: Apache-2.0
//
// The matching/overlap logic is mirrored faithfully:
//   - the first pass counts every prefix tuple and every (position, hash_id)
//     pair, exactly as _process_trace_first_pass does;
//   - the per-request cache hit rate uses the dynamo "first unseen block"
//     algorithm from _compute_per_request_hit_rates;
//   - the context / unique-prompt split mirrors _compute_context_lengths,
//     including its all-repeated shortcut (context == full ISL) and its
//     repeated_count * block_size rule;
//   - the prefix-reuse ratio and shared-first-block group count mirror
//     _compute_prefix_reuse and _compute_num_prefix_groups.
//
// The source also builds a RadixTree in the first pass, but no field of the
// returned AnalysisStats reads it, so this port omits that dead structure.
package perfledger

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

// DefaultBlockSize mirrors aiperf's InputTokensDefaults.BLOCK_SIZE.
const DefaultBlockSize = 512

// Trace is one request record from a trace: the input/output token lengths and
// the ordered KV-cache block hash IDs for the prompt. HashIDs may be empty, in
// which case the request carries no prefix information.
type Trace struct {
	InputLength  int
	OutputLength int
	HashIDs      []int
}

// MetricStats is the summary statistics for one numeric series, matching the
// aiperf MetricStats fields (mean, population std-dev, min, p25, median, p75,
// max).
type MetricStats struct {
	Mean   float64
	StdDev float64
	Min    float64
	P25    float64
	Median float64
	P75    float64
	Max    float64
}

// AnalysisStats mirrors aiperf's AnalysisStats: the ISL/OSL distributions, the
// theoretical cache hit rate, the prefix-reuse ratio, and the extended
// per-metric statistics.
type AnalysisStats struct {
	TotalRequests           int
	UniquePrefixes          int
	NumPrefixGroups         int
	CacheHitRate            float64
	MinISL                  int
	MaxISL                  int
	AvgISL                  float64
	MinOSL                  int
	MaxOSL                  int
	AvgOSL                  float64
	PrefixReuseRatio        float64
	ISLStats                *MetricStats
	OSLStats                *MetricStats
	ContextLengthStats      *MetricStats
	UniquePromptLengthStats *MetricStats
	HitRateStats            *MetricStats
}

// PrefixAnalyzer extracts ISL/OSL statistics and prefix patterns from traces,
// computing the theoretical cache hit rate assuming an infinite cache.
type PrefixAnalyzer struct {
	blockSize int
}

// NewPrefixAnalyzer returns a PrefixAnalyzer using the given tokens-per-block.
// A non-positive blockSize falls back to DefaultBlockSize.
func NewPrefixAnalyzer(blockSize int) *PrefixAnalyzer {
	if blockSize <= 0 {
		blockSize = DefaultBlockSize
	}
	return &PrefixAnalyzer{blockSize: blockSize}
}

// posHash is a (position, hash_id) pair, the unit of the context-length
// overlap counter.
type posHash struct {
	pos  int
	hash int
}

// firstPassState holds the accumulated counters from the first pass.
type firstPassState struct {
	isls                []int
	osls                []int
	hashIDsPerTrace     [][]int
	prefixCounter       map[string]int
	hashPositionCounter map[posHash]int
}

// firstPass collects the basic data and builds the prefix and
// (position, hash_id) counters, mirroring _process_trace_first_pass.
func (a *PrefixAnalyzer) firstPass(traces []Trace) *firstPassState {
	st := &firstPassState{
		isls:                make([]int, 0, len(traces)),
		osls:                make([]int, 0, len(traces)),
		hashIDsPerTrace:     make([][]int, 0, len(traces)),
		prefixCounter:       make(map[string]int),
		hashPositionCounter: make(map[posHash]int),
	}

	for _, t := range traces {
		st.isls = append(st.isls, t.InputLength)
		st.osls = append(st.osls, t.OutputLength)

		if len(t.HashIDs) == 0 {
			st.hashIDsPerTrace = append(st.hashIDsPerTrace, nil)
			continue
		}

		st.hashIDsPerTrace = append(st.hashIDsPerTrace, t.HashIDs)

		// Track prefix patterns: every prefix of the hash-id sequence.
		var key strings.Builder
		for i := 0; i < len(t.HashIDs); i++ {
			if i > 0 {
				key.WriteByte(',')
			}
			key.WriteString(strconv.Itoa(t.HashIDs[i]))
			st.prefixCounter[key.String()]++
		}

		// Track (position, hash_id) pairs for context-length calculation.
		for pos, hashID := range t.HashIDs {
			st.hashPositionCounter[posHash{pos: pos, hash: hashID}]++
		}
	}

	return st
}

// perRequestHitRates computes the per-request cache hit rates assuming an
// infinite cache, mirroring _compute_per_request_hit_rates. For each request it
// finds the first hash_id position not yet seen and reports that fraction as
// the hit rate, then marks all of the request's hash_ids as seen.
func (st *firstPassState) perRequestHitRates() []float64 {
	if len(st.hashIDsPerTrace) == 0 {
		return nil
	}

	seen := make(map[int]struct{})
	hitRates := make([]float64, 0, len(st.hashIDsPerTrace))

	for _, hashIDs := range st.hashIDsPerTrace {
		if len(hashIDs) == 0 {
			continue
		}

		firstUnseen := len(hashIDs)
		for idx, hashID := range hashIDs {
			if _, ok := seen[hashID]; !ok {
				firstUnseen = idx
				break
			}
		}

		hitRates = append(hitRates, float64(firstUnseen)/float64(len(hashIDs)))

		for _, h := range hashIDs {
			seen[h] = struct{}{}
		}
	}

	return hitRates
}

// contextLengths computes the context (shared-prefix) and unique-prompt lengths
// for each request, mirroring _compute_context_lengths.
func (a *PrefixAnalyzer) contextLengths(st *firstPassState) (contextLengths, uniquePromptLengths []int) {
	repeated := make(map[posHash]struct{}, len(st.hashPositionCounter))
	for ph, count := range st.hashPositionCounter {
		if count > 1 {
			repeated[ph] = struct{}{}
		}
	}

	for i, isl := range st.isls {
		hashIDs := st.hashIDsPerTrace[i]
		if len(hashIDs) == 0 {
			contextLengths = append(contextLengths, 0)
			uniquePromptLengths = append(uniquePromptLengths, isl)
			continue
		}

		allRepeated := true
		repeatedCount := 0
		for pos, hashID := range hashIDs {
			if _, ok := repeated[posHash{pos: pos, hash: hashID}]; ok {
				repeatedCount++
			} else {
				allRepeated = false
			}
		}

		var contextLen, uniqueLen int
		if allRepeated {
			contextLen = isl
			uniqueLen = 0
		} else {
			contextLen = repeatedCount * a.blockSize
			uniqueLen = isl - contextLen
		}

		contextLengths = append(contextLengths, contextLen)
		uniquePromptLengths = append(uniquePromptLengths, uniqueLen)
	}

	return contextLengths, uniquePromptLengths
}

// PerRequestHitRates returns the per-request cache hit rates (0.0..1.0) for the
// given traces, assuming an infinite cache.
func (a *PrefixAnalyzer) PerRequestHitRates(traces []Trace) []float64 {
	return a.firstPass(traces).perRequestHitRates()
}

// AnalyzeTraces computes the full prefix analysis statistics for the given
// traces, mirroring analyze_traces.
func (a *PrefixAnalyzer) AnalyzeTraces(traces []Trace) AnalysisStats {
	st := a.firstPass(traces)

	contextLengths, uniquePromptLengths := a.contextLengths(st)
	hitRates := st.perRequestHitRates()

	cacheHitRate := 0.0
	if len(hitRates) > 0 {
		sum := 0.0
		for _, hr := range hitRates {
			sum += hr
		}
		cacheHitRate = sum / float64(len(hitRates))
	}

	// Number of prefix groups: distinct first blocks shared by 2+ sequences.
	firstBlockCounts := make(map[int]int)
	for _, h := range st.hashIDsPerTrace {
		if len(h) > 0 {
			firstBlockCounts[h[0]]++
		}
	}
	numPrefixGroups := 0
	for _, count := range firstBlockCounts {
		if count > 1 {
			numPrefixGroups++
		}
	}

	// Prefix reuse ratio: reused prefixes over total prefixes.
	reused, total := 0, 0
	for _, count := range st.prefixCounter {
		total += count
		if count > 1 {
			reused += count
		}
	}
	prefixReuse := 0.0
	if total > 0 {
		prefixReuse = float64(reused) / float64(total)
	}

	totalRequests := len(st.isls)
	minISL, maxISL, avgISL := 0, 0, 0.0
	if totalRequests > 0 {
		minISL, maxISL = st.isls[0], st.isls[0]
		sum := 0
		for _, v := range st.isls {
			if v < minISL {
				minISL = v
			}
			if v > maxISL {
				maxISL = v
			}
			sum += v
		}
		avgISL = float64(sum) / float64(totalRequests)
	}

	minOSL, maxOSL, avgOSL := 0, 0, 0.0
	if totalRequests > 0 {
		minOSL, maxOSL = st.osls[0], st.osls[0]
		sum := 0
		for _, v := range st.osls {
			if v < minOSL {
				minOSL = v
			}
			if v > maxOSL {
				maxOSL = v
			}
			sum += v
		}
		avgOSL = float64(sum) / float64(totalRequests)
	}

	return AnalysisStats{
		TotalRequests:           totalRequests,
		UniquePrefixes:          len(st.prefixCounter),
		NumPrefixGroups:         numPrefixGroups,
		CacheHitRate:            cacheHitRate,
		MinISL:                  minISL,
		MaxISL:                  maxISL,
		AvgISL:                  avgISL,
		MinOSL:                  minOSL,
		MaxOSL:                  maxOSL,
		AvgOSL:                  avgOSL,
		PrefixReuseRatio:        prefixReuse,
		ISLStats:                metricStats(intsToFloats(st.isls)),
		OSLStats:                metricStats(intsToFloats(st.osls)),
		ContextLengthStats:      metricStats(intsToFloats(contextLengths)),
		UniquePromptLengthStats: metricStats(intsToFloats(uniquePromptLengths)),
		HitRateStats:            metricStats(hitRates),
	}
}

// intsToFloats converts an int slice to a float64 slice.
func intsToFloats(values []int) []float64 {
	out := make([]float64, len(values))
	for i, v := range values {
		out[i] = float64(v)
	}
	return out
}

// metricStats computes the full MetricStats for a list of values, or nil when
// the list is empty, mirroring _compute_metric_stats. The standard deviation is
// the population std-dev (numpy's default) and the percentiles use numpy's
// default linear interpolation (shared with percentileLinear).
func metricStats(values []float64) *MetricStats {
	if len(values) == 0 {
		return nil
	}

	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)

	n := float64(len(sorted))
	mean := 0.0
	for _, v := range sorted {
		mean += v
	}
	mean /= n

	variance := 0.0
	for _, v := range sorted {
		variance += (v - mean) * (v - mean)
	}
	stdDev := math.Sqrt(variance / n)

	return &MetricStats{
		Mean:   mean,
		StdDev: stdDev,
		Min:    sorted[0],
		P25:    percentileLinear(sorted, 0.25),
		Median: percentileLinear(sorted, 0.50),
		P75:    percentileLinear(sorted, 0.75),
		Max:    sorted[len(sorted)-1],
	}
}
