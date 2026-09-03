package servingsim

import (
	"math"
	"sort"
)

// EventType identifies the nature of a simulation event in the discrete-event loop.
type EventType string

const (
	// EventRequestArrival signals the arrival of a new inference request.
	EventRequestArrival EventType = "request_arrival"
	// EventStepComplete signals that a batch execution step has finished on the hardware.
	EventStepComplete EventType = "step_complete"
)

// SimEvent is a discrete-event structure managed by a min-heap priority queue.
type SimEvent struct {
	TimeMS  float64       `json:"time_ms"`
	Type    EventType     `json:"type"`
	Seq     int64         `json:"seq"`
	Request *RequestState `json:"request,omitempty"`
	Payload any           `json:"payload,omitempty"`

	// index tracks position in container/heap.
	index int
}

// RequestState represents the lifecycle and metrics for an individual LLM inference request.
type RequestState struct {
	ID               string  `json:"id"`
	ArrivalTimeMS    float64 `json:"arrival_time_ms"`
	PromptTokens     int     `json:"prompt_tokens"`
	OutputTarget     int     `json:"output_target"`
	PrefillComputed  int     `json:"prefill_computed"`
	TokensGenerated  int     `json:"tokens_generated"`
	FirstTokenTimeMS float64 `json:"first_token_time_ms"`
	CompletionTimeMS float64 `json:"completion_time_ms"`
	SpecAccepted     int     `json:"spec_accepted"`
	SpecProposed     int     `json:"spec_proposed"`
	QueueTimeMS      float64 `json:"queue_time_ms"`

	// Internal state
	AllocatedKVBlocks int  `json:"allocated_kv_blocks,omitempty"`
	Admitted          bool `json:"admitted,omitempty"`
}

// HardwareLatencyTable models execution latency for prefill chunk and decode step
// as functions of batch size, token count, and speculative draft length K.
type HardwareLatencyTable struct {
	// Linear execution model parameters (fallback when functions are nil)
	BasePrefillMS     float64 `json:"base_prefill_ms"`
	PerTokenPrefillMS float64 `json:"per_token_prefill_ms"`
	PerBatchPrefillMS float64 `json:"per_batch_prefill_ms"`

	BaseDecodeMS     float64 `json:"base_decode_ms"`
	PerBatchDecodeMS float64 `json:"per_batch_decode_ms"`
	PerDraftDecodeMS float64 `json:"per_draft_decode_ms"`

	// Optional custom function overrides
	PrefillChunkLatencyFunc func(tokens int, batchSize int) float64                                        `json:"-"`
	DecodeStepLatencyFunc   func(batchSize int, draftK int) float64                                        `json:"-"`
	StepLatencyFunc         func(prefillTokens int, prefillBatch int, decodeBatch int, draftK int) float64 `json:"-"`
}

// DefaultHardwareLatencyTable provides typical baseline latencies for modern data-center GPUs (e.g. H100/A100).
func DefaultHardwareLatencyTable() HardwareLatencyTable {
	return HardwareLatencyTable{
		BasePrefillMS:     1.5,
		PerTokenPrefillMS: 0.015,
		PerBatchPrefillMS: 0.1,

		BaseDecodeMS:     7.5,
		PerBatchDecodeMS: 0.15,
		PerDraftDecodeMS: 1.2,
	}
}

// PrefillChunkLatency calculates the execution latency of a prefill chunk.
func (h HardwareLatencyTable) PrefillChunkLatency(tokens int, batchSize int) float64 {
	if tokens <= 0 || batchSize <= 0 {
		return 0
	}
	if h.PrefillChunkLatencyFunc != nil {
		return h.PrefillChunkLatencyFunc(tokens, batchSize)
	}
	base := h.BasePrefillMS
	perToken := h.PerTokenPrefillMS
	perBatch := h.PerBatchPrefillMS
	if base == 0 && perToken == 0 && perBatch == 0 {
		def := DefaultHardwareLatencyTable()
		base, perToken, perBatch = def.BasePrefillMS, def.PerTokenPrefillMS, def.PerBatchPrefillMS
	}
	return base + perToken*float64(tokens) + perBatch*float64(batchSize)
}

// DecodeStepLatency calculates the execution latency of a decode step.
func (h HardwareLatencyTable) DecodeStepLatency(batchSize int, draftK int) float64 {
	if batchSize <= 0 {
		return 0
	}
	if h.DecodeStepLatencyFunc != nil {
		return h.DecodeStepLatencyFunc(batchSize, draftK)
	}
	base := h.BaseDecodeMS
	perBatch := h.PerBatchDecodeMS
	perDraft := h.PerDraftDecodeMS
	if base == 0 && perBatch == 0 && perDraft == 0 {
		def := DefaultHardwareLatencyTable()
		base, perBatch, perDraft = def.BaseDecodeMS, def.PerBatchDecodeMS, def.PerDraftDecodeMS
	}
	return base + perBatch*float64(batchSize) + perDraft*float64(draftK)
}

// StepLatency calculates execution latency for a combined or distinct serving step.
func (h HardwareLatencyTable) StepLatency(prefillTokens int, prefillBatch int, decodeBatch int, draftK int) float64 {
	if prefillBatch <= 0 && decodeBatch <= 0 {
		return 0
	}
	if h.StepLatencyFunc != nil {
		return h.StepLatencyFunc(prefillTokens, prefillBatch, decodeBatch, draftK)
	}
	latPrefill := 0.0
	if prefillBatch > 0 && prefillTokens > 0 {
		latPrefill = h.PrefillChunkLatency(prefillTokens, prefillBatch)
	}
	latDecode := 0.0
	if decodeBatch > 0 {
		latDecode = h.DecodeStepLatency(decodeBatch, draftK)
	}
	if latPrefill > 0 && latDecode > 0 {
		// Piggybacked / fused step: prefill chunk FLOPs execute alongside decode memory reads.
		return latPrefill + latDecode*0.6
	}
	return latPrefill + latDecode
}

// SpeculativeMode specifies the stochastic or deterministic model for draft token acceptance.
type SpeculativeMode string

const (
	// SpeculativeModePrefix tests tokens sequentially from position 0 to K-1, stopping at the first rejection.
	SpeculativeModePrefix SpeculativeMode = "prefix"
	// SpeculativeModeBinomial samples acceptance count from Binomial(K, alpha).
	SpeculativeModeBinomial SpeculativeMode = "binomial"
	// SpeculativeModePoisson samples acceptance count from Poisson(K * alpha).
	SpeculativeModePoisson SpeculativeMode = "poisson"
	// SpeculativeModeDeterministic accepts round(K * alpha) tokens deterministically.
	SpeculativeModeDeterministic SpeculativeMode = "deterministic"
)

// SimulatorConfig configures the continuous-batching discrete-event scheduler.
type SimulatorConfig struct {
	MaxBatchSize        int                   `json:"max_batch_size"`
	MaxTokensPerStep    int                   `json:"max_tokens_per_step"` // chunked prefill budget
	KVBlockTokens       int                   `json:"kv_block_tokens"`     // tokens per KV block
	TotalKVBlocks       int                   `json:"total_kv_blocks"`     // available KV blocks
	SpeculativeHorizon  int                   `json:"speculative_horizon"` // draft length K (0 = disabled)
	AcceptanceRate      float64               `json:"acceptance_rate"`     // average speculative acceptance probability (0.0 - 1.0)
	PositionalAlphaFunc func(pos int) float64 `json:"-"`                   // optional position-dependent alpha(pos)

	SpeculativeMode SpeculativeMode `json:"speculative_mode,omitempty"`
	Deterministic   bool            `json:"deterministic,omitempty"`
	Seed            int64           `json:"seed,omitempty"`
	EnableTrace     bool            `json:"enable_trace,omitempty"`
}

// PercentileLatency holds P50, P90, P95, and P99 summary latency figures.
type PercentileLatency struct {
	P50  float64 `json:"p50"`
	P90  float64 `json:"p90"`
	P95  float64 `json:"p95"`
	P99  float64 `json:"p99"`
	Min  float64 `json:"min,omitempty"`
	Max  float64 `json:"max,omitempty"`
	Mean float64 `json:"mean,omitempty"`
}

// SimulationMetrics summarizes the performance and resource efficiency of a simulation run.
type SimulationMetrics struct {
	TotalRequests       int               `json:"total_requests"`
	SimulatedDurationMS float64           `json:"simulated_duration_ms"`
	RequestThroughput   float64           `json:"request_throughput"` // requests / second
	TokenThroughput     float64           `json:"token_throughput"`   // output tokens / second
	TTFT                PercentileLatency `json:"ttft"`               // Time To First Token (ms)
	TPOT                PercentileLatency `json:"tpot"`               // Time Per Output Token (ms)
	PeakKVBlocksUsed    int               `json:"peak_kv_blocks_used"`
	KVBlockUtilization  float64           `json:"kv_block_utilization"` // time-weighted average utilization [0, 1]
	SpeculativeYield    float64           `json:"speculative_yield"`    // accepted draft tokens / proposed draft tokens
	SpeculativeWaste    float64           `json:"speculative_waste"`    // rejected draft tokens / proposed draft tokens

	CompletedRequests []RequestState `json:"completed_requests,omitempty"`
	TraceEvents       []TraceEvent   `json:"trace_events,omitempty"`
}

// ComputePercentiles calculates P50, P90, P95, P99, Min, Max, and Mean from a slice of values.
func ComputePercentiles(values []float64) PercentileLatency {
	n := len(values)
	if n == 0 {
		return PercentileLatency{}
	}
	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)

	total := 0.0
	for _, v := range sorted {
		total += v
	}
	mean := total / float64(n)

	p50 := quantile(sorted, 0.50)
	p90 := quantile(sorted, 0.90)
	p95 := quantile(sorted, 0.95)
	p99 := quantile(sorted, 0.99)

	return PercentileLatency{
		P50:  p50,
		P90:  p90,
		P95:  p95,
		P99:  p99,
		Min:  sorted[0],
		Max:  sorted[n-1],
		Mean: mean,
	}
}

func quantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}
	if q <= 0 {
		return sorted[0]
	}
	if q >= 1.0 {
		return sorted[n-1]
	}

	idx := q * float64(n-1)
	low := int(math.Floor(idx))
	high := int(math.Ceil(idx))
	frac := idx - float64(low)
	return sorted[low]*(1.0-frac) + sorted[high]*frac
}
