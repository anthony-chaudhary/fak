package agentxbench

import (
	"fmt"
	"math"
	"strings"
)

// ValidateReceipt inspects an AgentXReceipt and enforces fail-closed validation rules.
// It returns a slice of refusal reasons; if empty, the receipt is considered verified.
func ValidateReceipt(receipt *AgentXReceipt) []string {
	var errs []string

	if receipt == nil {
		return []string{"RECEIPT_NIL: receipt object cannot be nil"}
	}

	if receipt.Schema != SchemaIdentifier {
		errs = append(errs, fmt.Sprintf("SCHEMA_MISMATCH: expected %q, got %q", SchemaIdentifier, receipt.Schema))
	}

	if strings.TrimSpace(receipt.BenchmarkID) == "" {
		errs = append(errs, "BENCHMARK_ID_MISSING: benchmark_id is required")
	}
	if strings.TrimSpace(receipt.Model) == "" {
		errs = append(errs, "MODEL_MISSING: model identifier is required")
	}
	if strings.TrimSpace(receipt.Engine) == "" {
		errs = append(errs, "ENGINE_MISSING: engine identifier is required")
	}
	if strings.TrimSpace(receipt.Hardware) == "" {
		errs = append(errs, "HARDWARE_MISSING: hardware description is required")
	}

	if receipt.AgentCount <= 0 {
		errs = append(errs, fmt.Sprintf("AGENT_COUNT_INVALID: agent_count must be > 0, got %d", receipt.AgentCount))
	}
	if receipt.TurnsPerAgent <= 0 {
		errs = append(errs, fmt.Sprintf("TURNS_PER_AGENT_INVALID: turns_per_agent must be > 0, got %d", receipt.TurnsPerAgent))
	}

	expectedRequests := receipt.AgentCount * receipt.TurnsPerAgent
	if len(receipt.Requests) != expectedRequests {
		errs = append(errs, fmt.Sprintf("REQUEST_COUNT_MISMATCH: expected %d requests (%d agents * %d turns), got %d",
			expectedRequests, receipt.AgentCount, receipt.TurnsPerAgent, len(receipt.Requests)))
	}

	if receipt.Aggregated.TotalRequests != len(receipt.Requests) {
		errs = append(errs, fmt.Sprintf("AGGREGATE_TOTAL_MISMATCH: aggregated.total_requests (%d) != len(requests) (%d)",
			receipt.Aggregated.TotalRequests, len(receipt.Requests)))
	}

	for i, req := range receipt.Requests {
		prefix := fmt.Sprintf("request[%d](id=%s,agent=%s,turn=%d)", i, req.RequestID, req.AgentID, req.TurnIndex)

		if strings.TrimSpace(req.RequestID) == "" {
			errs = append(errs, fmt.Sprintf("%s: REQUEST_ID_EMPTY", prefix))
		}
		if strings.TrimSpace(req.AgentID) == "" {
			errs = append(errs, fmt.Sprintf("%s: AGENT_ID_EMPTY", prefix))
		}
		if req.TurnIndex < 0 {
			errs = append(errs, fmt.Sprintf("%s: TURN_INDEX_NEGATIVE", prefix))
		}

		if req.PromptTokens < 0 {
			errs = append(errs, fmt.Sprintf("%s: PROMPT_TOKENS_NEGATIVE (%d)", prefix, req.PromptTokens))
		}
		if req.CompletionTokens < 0 {
			errs = append(errs, fmt.Sprintf("%s: COMPLETION_TOKENS_NEGATIVE (%d)", prefix, req.CompletionTokens))
		}
		if req.CachedTokens < 0 {
			errs = append(errs, fmt.Sprintf("%s: CACHED_TOKENS_NEGATIVE (%d)", prefix, req.CachedTokens))
		}
		if req.CachedTokens > req.PromptTokens {
			errs = append(errs, fmt.Sprintf("%s: CACHED_TOKENS_EXCEEDS_PROMPT (%d > %d)", prefix, req.CachedTokens, req.PromptTokens))
		}

		// Phase intervals validation
		if req.ClientPhases.QueueWaitMS < 0 || math.IsNaN(req.ClientPhases.QueueWaitMS) {
			errs = append(errs, fmt.Sprintf("%s: QUEUE_WAIT_INVALID", prefix))
		}
		if req.ClientPhases.AgentExecutionMS < 0 || math.IsNaN(req.ClientPhases.AgentExecutionMS) {
			errs = append(errs, fmt.Sprintf("%s: AGENT_EXECUTION_INVALID", prefix))
		}
		if req.ClientPhases.EvaluationMS < 0 || math.IsNaN(req.ClientPhases.EvaluationMS) {
			errs = append(errs, fmt.Sprintf("%s: EVALUATION_INVALID", prefix))
		}
		if req.ClientPhases.TotalLifecycleMS < req.ClientPhases.AgentExecutionMS {
			errs = append(errs, fmt.Sprintf("%s: LIFECYCLE_LESS_THAN_EXECUTION (%f < %f)",
				prefix, req.ClientPhases.TotalLifecycleMS, req.ClientPhases.AgentExecutionMS))
		}

		// Success invariants
		if req.Success {
			if req.CompletionTokens == 0 {
				errs = append(errs, fmt.Sprintf("%s: SUCCESS_WITHOUT_COMPLETION_TOKENS", prefix))
			}
			if req.Interactivity.TTFTMS < 0 || math.IsNaN(req.Interactivity.TTFTMS) {
				errs = append(errs, fmt.Sprintf("%s: TTFT_INVALID (%f)", prefix, req.Interactivity.TTFTMS))
			}
			if req.Interactivity.NormalizedInteractivity < 0 || math.IsNaN(req.Interactivity.NormalizedInteractivity) {
				errs = append(errs, fmt.Sprintf("%s: NORMALIZED_INTERACTIVITY_INVALID (%f)", prefix, req.Interactivity.NormalizedInteractivity))
			}

			// Timestamps monotonicity
			if len(req.TokenTimestampsUnixNano) > 1 {
				for tIdx := 1; tIdx < len(req.TokenTimestampsUnixNano); tIdx++ {
					if req.TokenTimestampsUnixNano[tIdx] < req.TokenTimestampsUnixNano[tIdx-1] {
						errs = append(errs, fmt.Sprintf("%s: TOKEN_TIMESTAMPS_NOT_MONOTONIC at index %d (%d < %d)",
							prefix, tIdx, req.TokenTimestampsUnixNano[tIdx], req.TokenTimestampsUnixNano[tIdx-1]))
						break
					}
				}
			}
		} else {
			if strings.TrimSpace(req.Error) == "" {
				errs = append(errs, fmt.Sprintf("%s: FAILED_WITHOUT_ERROR_MESSAGE", prefix))
			}
		}
	}

	return errs
}
