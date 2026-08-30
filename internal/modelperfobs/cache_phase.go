package modelperfobs

import (
	"sync"
	"time"
)

// CachePipelinePhase is the bounded model-pipeline dimension used for cache latency.
// NormalizeCachePipelinePhase must be applied before storage or export so caller input
// cannot create an unbounded telemetry label set.
type CachePipelinePhase string

const (
	CachePipelinePhasePrefill CachePipelinePhase = "prefill"
	CachePipelinePhaseDecode  CachePipelinePhase = "decode"
	CachePipelinePhaseOther   CachePipelinePhase = "other"
)

var cachePipelinePhases = [...]CachePipelinePhase{
	CachePipelinePhasePrefill,
	CachePipelinePhaseDecode,
	CachePipelinePhaseOther,
}

// NormalizeCachePipelinePhase maps every value outside the closed vocabulary to other.
func NormalizeCachePipelinePhase(phase CachePipelinePhase) CachePipelinePhase {
	switch phase {
	case CachePipelinePhasePrefill, CachePipelinePhaseDecode:
		return phase
	default:
		return CachePipelinePhaseOther
	}
}

// CachePhaseLatency is one deterministic phase bucket in a latency receipt.
type CachePhaseLatency struct {
	Phase        CachePipelinePhase `json:"phase"`
	Observations uint64             `json:"observations"`
	Total        time.Duration      `json:"total"`
}

// CachePhaseLatencyReceipt reports the existing unlabeled total beside its bounded
// phase attribution. Phases always use the fixed vocabulary and fixed ordering.
type CachePhaseLatencyReceipt struct {
	Observations uint64              `json:"observations"`
	Total        time.Duration       `json:"total"`
	Phases       []CachePhaseLatency `json:"phases"`
}

// CachePhaseLatencyRecorder accumulates cache-coupled pipeline latency. Callers supply
// measured durations, which keeps aggregation deterministic and independently testable.
type CachePhaseLatencyRecorder struct {
	mu     sync.Mutex
	counts [len(cachePipelinePhases)]uint64
	totals [len(cachePipelinePhases)]time.Duration
}

// Observe adds one duration after collapsing the phase into the closed vocabulary.
func (r *CachePhaseLatencyRecorder) Observe(phase CachePipelinePhase, duration time.Duration) {
	if duration < 0 {
		duration = 0
	}
	phase = NormalizeCachePipelinePhase(phase)
	idx := len(cachePipelinePhases) - 1
	for i, candidate := range cachePipelinePhases {
		if candidate == phase {
			idx = i
			break
		}
	}
	r.mu.Lock()
	r.counts[idx]++
	r.totals[idx] += duration
	r.mu.Unlock()
}

// Receipt returns a stable snapshot whose unlabeled totals are derived from the
// phase buckets, preventing the two views from drifting into separate counter stacks.
func (r *CachePhaseLatencyRecorder) Receipt() CachePhaseLatencyReceipt {
	r.mu.Lock()
	defer r.mu.Unlock()

	receipt := CachePhaseLatencyReceipt{Phases: make([]CachePhaseLatency, 0, len(cachePipelinePhases))}
	for i, phase := range cachePipelinePhases {
		bucket := CachePhaseLatency{Phase: phase, Observations: r.counts[i], Total: r.totals[i]}
		receipt.Phases = append(receipt.Phases, bucket)
		receipt.Observations += bucket.Observations
		receipt.Total += bucket.Total
	}
	return receipt
}
