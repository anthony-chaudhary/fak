package agentxbench

// SchemaIdentifier is the canonical schema for AgentX benchmark receipts.
const SchemaIdentifier = "fak.agentx-benchmark-receipt.v1"

// ClientPhases decomposes the request lifecycle across client-visible boundaries.
// Invariant: all phase durations must be non-negative.
type ClientPhases struct {
	QueueWaitMS      float64 `json:"queue_wait_ms"`
	AgentExecutionMS float64 `json:"agent_execution_ms"`
	EvaluationMS     float64 `json:"evaluation_ms"`
	TotalLifecycleMS float64 `json:"total_lifecycle_ms"`
}

// InteractivityMetrics records latency and interactivity percentiles for a request.
// Invariant: TTFT and ITL metrics must be non-negative.
type InteractivityMetrics struct {
	TTFTMS                 float64 `json:"ttft_ms"`
	ITLMedianMS            float64 `json:"itl_median_ms"`
	ITLP90MS               float64 `json:"itl_p90_ms"`
	ITLP95MS               float64 `json:"itl_p95_ms"`
	ITLP99MS               float64 `json:"itl_p99_ms"`
	ITLMaxMS               float64 `json:"itl_max_ms"`
	NormalizedInteractivity float64 `json:"normalized_interactivity_tok_per_sec"`
}

// RequestRecord captures one agent request turn with client/server lifecycle join.
type RequestRecord struct {
	RequestID               string               `json:"request_id"`
	AgentID                 string               `json:"agent_id"`
	TurnIndex               int                  `json:"turn_index"`
	PromptTokens            int                  `json:"prompt_tokens"`
	CompletionTokens        int                  `json:"completion_tokens"`
	CachedTokens            int                  `json:"cached_tokens"`
	CacheHitRatio           float64              `json:"cache_hit_ratio"`
	ClientPhases            ClientPhases         `json:"client_phases"`
	Interactivity           InteractivityMetrics `json:"interactivity"`
	TokenTimestampsUnixNano []int64              `json:"token_timestamps_unix_nano,omitempty"`
	Success                 bool                 `json:"success"`
	Error                   string               `json:"error,omitempty"`
	ResponseText            string               `json:"response_text,omitempty"`
}

// AggregatedMetrics holds summarized statistics across all executed agent requests.
type AggregatedMetrics struct {
	TotalRequests               int     `json:"total_requests"`
	SuccessfulRequests          int     `json:"successful_requests"`
	FailedRequests              int     `json:"failed_requests"`
	SuccessRate                 float64 `json:"success_rate"`
	TotalPromptTokens           int     `json:"total_prompt_tokens"`
	TotalCompletionTokens       int     `json:"total_completion_tokens"`
	TotalCachedTokens           int     `json:"total_cached_tokens"`
	AggregateCacheHitRatio      float64 `json:"aggregate_cache_hit_ratio"`
	ColdTTFTMeanMS              float64 `json:"cold_ttft_mean_ms"`
	WarmTTFTMeanMS              float64 `json:"warm_ttft_mean_ms"`
	PrefixSpeedupRatio          float64 `json:"prefix_speedup_ratio"`
	TTFTP50MS                   float64 `json:"ttft_p50_ms"`
	TTFTP95MS                   float64 `json:"ttft_p95_ms"`
	ITLP50MS                    float64 `json:"itl_p50_ms"`
	ITLP95MS                    float64 `json:"itl_p95_ms"`
	NormalizedInteractivity     float64 `json:"normalized_interactivity_tok_per_sec"`
	RequestThroughputPerSec     float64 `json:"request_throughput_per_sec"`
	OutputTokenThroughputPerSec float64 `json:"output_token_throughput_per_sec"`
	ClusterTokenThroughputPerSec float64 `json:"cluster_token_throughput_per_sec"`
	TotalWallTimeMS             float64 `json:"total_wall_time_ms"`
}

// AgentXReceipt represents the full benchmark artifact witnessing an AgentX evaluation.
type AgentXReceipt struct {
	Schema           string            `json:"schema"`
	BenchmarkID      string            `json:"benchmark_id"`
	TimestampISO     string            `json:"timestamp_iso"`
	Hardware         string            `json:"hardware"`
	Endpoint         string            `json:"endpoint"`
	Model            string            `json:"model"`
	Engine           string            `json:"engine"`
	AgentCount       int               `json:"agent_count"`
	TurnsPerAgent    int               `json:"turns_per_agent"`
	Aggregated       AggregatedMetrics `json:"aggregated"`
	Requests         []RequestRecord   `json:"requests"`
	ValidationStatus string            `json:"validation_status"`
	ValidationErrors []string          `json:"validation_errors,omitempty"`
}

// Config specifies parameters for running an AgentX benchmark.
type Config struct {
	EndpointURL       string  `json:"endpoint_url"`
	Model             string  `json:"model"`
	Engine            string  `json:"engine"`
	Hardware          string  `json:"hardware"`
	AgentCount        int     `json:"agent_count"`
	TurnsPerAgent     int     `json:"turns_per_agent"`
	MaxTokens         int     `json:"max_tokens"`
	Temperature       float64 `json:"temperature"`
	SharedPrefix      string  `json:"shared_prefix"`
	DeterministicSeed int64   `json:"deterministic_seed"`
	MockExecution     bool    `json:"mock_execution"`
}
