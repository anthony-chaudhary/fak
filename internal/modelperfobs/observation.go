// Package modelperfobs measures OpenAI-compatible inference requests at the
// harness/backend seam and writes query-friendly JSONL observations.
package modelperfobs

import "time"

const Schema = "fak-model-perf/1"

// Observation is one inference request. Duration fields use milliseconds so
// JSONL remains directly sortable and queryable without duration parsing.
type Observation struct {
	Schema             string    `json:"schema"`
	Timestamp          time.Time `json:"timestamp"`
	RequestID          string    `json:"request_id"`
	Model              string    `json:"model,omitempty"`
	Backend            string    `json:"backend"`
	Status             int       `json:"status"`
	Streaming          bool      `json:"streaming"`
	PromptTokens       int64     `json:"prompt_tokens,omitempty"`
	CompletionTokens   int64     `json:"completion_tokens,omitempty"`
	TotalTokens        int64     `json:"total_tokens,omitempty"`
	DurationMS         float64   `json:"duration_ms"`
	TTFTMS             float64   `json:"ttft_ms,omitempty"`
	TPOTMS             float64   `json:"tpot_ms,omitempty"`
	OutputTokensPerSec float64   `json:"output_tokens_per_second,omitempty"`
	InterChunkCount    int64     `json:"inter_chunk_count,omitempty"`
	InterChunkP50MS    float64   `json:"inter_chunk_p50_ms,omitempty"`
	InterChunkP95MS    float64   `json:"inter_chunk_p95_ms,omitempty"`
	Error              string    `json:"error,omitempty"`
}
