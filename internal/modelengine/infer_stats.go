// Ported: the per-request phase-split stats of InferenceStatsAggregator
// (request/queue/compute_input/compute_infer/compute_output durations, cache
// hit/miss counts and durations, success/failure split with FailureReason).
// fak adds: a per-model registry, nil-safe recorders so wiring stays default-off,
// per-reason failure counts, copy-out snapshots, and a typed refusal for
// out-of-order timestamps where upstream's unsigned subtraction would wrap.
//
// Copyright 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//   - Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//   - Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//   - Neither the name of NVIDIA CORPORATION nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package modelengine

import (
	"fmt"
	"sort"
	"sync"
)

// InferStatsFailureReason mirrors upstream's FailureReason taxonomy.
type InferStatsFailureReason int

const (
	InferStatsFailureRejected InferStatsFailureReason = iota
	InferStatsFailureCanceled
	InferStatsFailureBackend
	InferStatsFailureOther

	inferStatsFailureReasonCount
)

// String returns the upstream metric suffix (inf_failure_<REASON>); unknown
// values fold into OTHER exactly as failureReasonToString's default does.
func (r InferStatsFailureReason) String() string {
	switch r {
	case InferStatsFailureRejected:
		return "REJECTED"
	case InferStatsFailureCanceled:
		return "CANCELED"
	case InferStatsFailureBackend:
		return "BACKEND"
	default:
		return "OTHER"
	}
}

func (r InferStatsFailureReason) index() int {
	if r < 0 || r >= inferStatsFailureReasonCount {
		return int(InferStatsFailureOther)
	}
	return int(r)
}

// InferStats is the cumulative per-model phase split, all durations in ns.
type InferStats struct {
	FailureCount      uint64
	FailureDurationNS uint64
	// FailureByReason is indexed by InferStatsFailureReason.
	FailureByReason [inferStatsFailureReasonCount]uint64

	SuccessCount            uint64
	RequestDurationNS       uint64
	QueueDurationNS         uint64
	ComputeInputDurationNS  uint64
	ComputeInferDurationNS  uint64
	ComputeOutputDurationNS uint64

	CacheHitCount       uint64
	CacheHitDurationNS  uint64
	CacheMissCount      uint64
	CacheMissDurationNS uint64
}

// ComputeDurationNS is the compute share of request time, so callers can split
// latency into queue versus compute without re-deriving it per phase.
func (s InferStats) ComputeDurationNS() uint64 {
	return s.ComputeInputDurationNS + s.ComputeInferDurationNS + s.ComputeOutputDurationNS
}

// InferStatsTimestamps carries the monotonic stamps of one successful request.
type InferStatsTimestamps struct {
	RequestStartNS       uint64
	QueueStartNS         uint64
	ComputeStartNS       uint64
	ComputeInputEndNS    uint64
	ComputeOutputStartNS uint64
	ComputeEndNS         uint64
	RequestEndNS         uint64
}

// InferStatsOrderError refuses a sample whose stamps run backwards; upstream
// would record a wrapped uint64 and poison the cumulative total forever.
type InferStatsOrderError struct {
	Earlier, Later     string
	EarlierNS, LaterNS uint64
}

func (e *InferStatsOrderError) Error() string {
	return fmt.Sprintf("infer stats: %s (%d ns) is after %s (%d ns)", e.Earlier, e.EarlierNS, e.Later, e.LaterNS)
}

func inferStatsSpan(earlier string, earlierNS uint64, later string, laterNS uint64) (uint64, error) {
	if earlierNS > laterNS {
		return 0, &InferStatsOrderError{Earlier: earlier, Later: later, EarlierNS: earlierNS, LaterNS: laterNS}
	}
	return laterNS - earlierNS, nil
}

// ModelInferStatsSnapshot is a copy of one model's aggregate.
type ModelInferStatsSnapshot struct {
	Model          string
	InferenceCount uint64
	Stats          InferStats
}

// ModelInferStats aggregates request stats for one model. A nil receiver is a
// no-op recorder so an unwired scheduler pays nothing.
type ModelInferStats struct {
	mu             sync.Mutex
	model          string
	inferenceCount uint64
	stats          InferStats
}

// NewModelInferStats returns an empty aggregate for model.
func NewModelInferStats(model string) *ModelInferStats {
	return &ModelInferStats{model: model}
}

// UpdateFailure records a failed request and its end-to-end duration.
func (m *ModelInferStats) UpdateFailure(requestStartNS, requestEndNS uint64, reason InferStatsFailureReason) error {
	if m == nil {
		return nil
	}
	d, err := inferStatsSpan("request_start", requestStartNS, "request_end", requestEndNS)
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.stats.FailureCount++
	m.stats.FailureDurationNS += d
	m.stats.FailureByReason[reason.index()]++
	m.mu.Unlock()
	return nil
}

// UpdateSuccess derives the three compute phases from stamps and records them.
func (m *ModelInferStats) UpdateSuccess(batchSize uint64, ts InferStatsTimestamps) error {
	if m == nil {
		return nil
	}
	in, err := inferStatsSpan("compute_start", ts.ComputeStartNS, "compute_input_end", ts.ComputeInputEndNS)
	if err != nil {
		return err
	}
	infer, err := inferStatsSpan("compute_input_end", ts.ComputeInputEndNS, "compute_output_start", ts.ComputeOutputStartNS)
	if err != nil {
		return err
	}
	out, err := inferStatsSpan("compute_output_start", ts.ComputeOutputStartNS, "compute_end", ts.ComputeEndNS)
	if err != nil {
		return err
	}
	return m.UpdateSuccessWithDuration(batchSize, ts.RequestStartNS, ts.QueueStartNS, ts.ComputeStartNS, ts.RequestEndNS, in, infer, out)
}

// UpdateSuccessWithDuration records a success whose compute phases were
// measured directly, as backends that batch across requests report them.
func (m *ModelInferStats) UpdateSuccessWithDuration(batchSize, requestStartNS, queueStartNS, computeStartNS, requestEndNS, computeInputNS, computeInferNS, computeOutputNS uint64) error {
	if m == nil {
		return nil
	}
	req, err := inferStatsSpan("request_start", requestStartNS, "request_end", requestEndNS)
	if err != nil {
		return err
	}
	queue, err := inferStatsSpan("queue_start", queueStartNS, "compute_start", computeStartNS)
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.inferenceCount += batchSize
	m.stats.SuccessCount++
	m.stats.RequestDurationNS += req
	m.stats.QueueDurationNS += queue
	m.stats.ComputeInputDurationNS += computeInputNS
	m.stats.ComputeInferDurationNS += computeInferNS
	m.stats.ComputeOutputDurationNS += computeOutputNS
	m.mu.Unlock()
	return nil
}

// UpdateSuccessCacheHit records a request served from cache. It never reaches
// compute, so queue ends at cache lookup and inference count is untouched.
func (m *ModelInferStats) UpdateSuccessCacheHit(requestStartNS, queueStartNS, cacheLookupStartNS, requestEndNS, cacheHitNS uint64) error {
	if m == nil {
		return nil
	}
	req, err := inferStatsSpan("request_start", requestStartNS, "request_end", requestEndNS)
	if err != nil {
		return err
	}
	queue, err := inferStatsSpan("queue_start", queueStartNS, "cache_lookup_start", cacheLookupStartNS)
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.stats.SuccessCount++
	m.stats.RequestDurationNS += req
	m.stats.QueueDurationNS += queue
	m.stats.CacheHitCount++
	m.stats.CacheHitDurationNS += cacheHitNS
	m.mu.Unlock()
	return nil
}

// UpdateSuccessCacheMiss adds cache-insertion time to request duration, since
// insertion runs after the success sample already closed the request span.
func (m *ModelInferStats) UpdateSuccessCacheMiss(cacheMissNS uint64) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.stats.RequestDurationNS += cacheMissNS
	m.stats.CacheMissCount++
	m.stats.CacheMissDurationNS += cacheMissNS
	m.mu.Unlock()
}

// Snapshot returns a copy that does not alias the live aggregate.
func (m *ModelInferStats) Snapshot() ModelInferStatsSnapshot {
	if m == nil {
		return ModelInferStatsSnapshot{}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return ModelInferStatsSnapshot{Model: m.model, InferenceCount: m.inferenceCount, Stats: m.stats}
}

// InferStatsRegistry keys ModelInferStats by model id. A nil registry hands out
// nil recorders, which keeps unwired callers on the no-op path.
type InferStatsRegistry struct {
	mu      sync.Mutex
	byModel map[string]*ModelInferStats
}

// NewInferStatsRegistry returns an empty registry.
func NewInferStatsRegistry() *InferStatsRegistry {
	return &InferStatsRegistry{byModel: map[string]*ModelInferStats{}}
}

// Model returns the aggregate for model, creating it on first use.
func (r *InferStatsRegistry) Model(model string) *ModelInferStats {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	m, ok := r.byModel[model]
	if !ok {
		m = NewModelInferStats(model)
		r.byModel[model] = m
	}
	return m
}

// Snapshot returns every model's aggregate sorted by model id.
func (r *InferStatsRegistry) Snapshot() []ModelInferStatsSnapshot {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	models := make([]*ModelInferStats, 0, len(r.byModel))
	for _, m := range r.byModel {
		models = append(models, m)
	}
	r.mu.Unlock()
	out := make([]ModelInferStatsSnapshot, 0, len(models))
	for _, m := range models {
		out = append(out, m.Snapshot())
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Model < out[j].Model })
	return out
}
