package atif

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// HarborVersion is the required schema_version for Harbor ATIF v1.7.
const HarborVersion = "ATIF-v1.7"

// AgentV17 identifies the agent that produced a Harbor trajectory.
type AgentV17 struct {
	Name            string           `json:"name"`
	Version         string           `json:"version"`
	ModelName       *string          `json:"model_name,omitempty"`
	ToolDefinitions []map[string]any `json:"tool_definitions,omitempty"`
	Extra           map[string]any   `json:"extra,omitempty"`
}

// HarborTrajectory is the standards-shaped, inert Harbor ATIF v1.7 boundary.
type HarborTrajectory struct {
	SchemaVersion          string             `json:"schema_version,omitempty"`
	SessionID              *string            `json:"session_id,omitempty"`
	TrajectoryID           *string            `json:"trajectory_id,omitempty"`
	Agent                  AgentV17           `json:"agent"`
	Steps                  []StepV17          `json:"steps"`
	Notes                  *string            `json:"notes,omitempty"`
	FinalMetrics           *FinalMetricsV17   `json:"final_metrics,omitempty"`
	ContinuedTrajectoryRef *string            `json:"continued_trajectory_ref,omitempty"`
	Extra                  map[string]any     `json:"extra,omitempty"`
	SubagentTrajectories   []HarborTrajectory `json:"subagent_trajectories,omitempty"`
}

type StepV17 struct {
	StepID           int             `json:"step_id"`
	Timestamp        *string         `json:"timestamp,omitempty"`
	Source           string          `json:"source"`
	ModelName        *string         `json:"model_name,omitempty"`
	ReasoningEffort  any             `json:"reasoning_effort,omitempty"`
	Message          any             `json:"message"`
	ReasoningContent *string         `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCallV17   `json:"tool_calls,omitempty"`
	Observation      *ObservationV17 `json:"observation,omitempty"`
	Metrics          *MetricsV17     `json:"metrics,omitempty"`
	IsCopiedContext  *bool           `json:"is_copied_context,omitempty"`
	LLMCallCount     *int            `json:"llm_call_count,omitempty"`
	Extra            map[string]any  `json:"extra,omitempty"`
}

type ToolCallV17 struct {
	ToolCallID   string         `json:"tool_call_id"`
	FunctionName string         `json:"function_name"`
	Arguments    map[string]any `json:"arguments"`
	Extra        map[string]any `json:"extra,omitempty"`
}

type ObservationV17 struct {
	Results []ObservationResultV17 `json:"results"`
}
type ObservationResultV17 struct {
	SourceCallID          *string          `json:"source_call_id,omitempty"`
	Content               any              `json:"content,omitempty"`
	SubagentTrajectoryRef []SubagentRefV17 `json:"subagent_trajectory_ref,omitempty"`
	Extra                 map[string]any   `json:"extra,omitempty"`
}
type SubagentRefV17 struct {
	TrajectoryID   *string        `json:"trajectory_id,omitempty"`
	SessionID      *string        `json:"session_id,omitempty"`
	TrajectoryPath *string        `json:"trajectory_path,omitempty"`
	Extra          map[string]any `json:"extra,omitempty"`
}
type MetricsV17 struct {
	PromptTokens       *int           `json:"prompt_tokens,omitempty"`
	CompletionTokens   *int           `json:"completion_tokens,omitempty"`
	CachedTokens       *int           `json:"cached_tokens,omitempty"`
	CostUSD            *float64       `json:"cost_usd,omitempty"`
	PromptTokenIDs     []int          `json:"prompt_token_ids,omitempty"`
	CompletionTokenIDs []int          `json:"completion_token_ids,omitempty"`
	Logprobs           []any          `json:"logprobs,omitempty"`
	Extra              map[string]any `json:"extra,omitempty"`
}
type FinalMetricsV17 struct {
	TotalPromptTokens     *int           `json:"total_prompt_tokens,omitempty"`
	TotalCompletionTokens *int           `json:"total_completion_tokens,omitempty"`
	TotalCachedTokens     *int           `json:"total_cached_tokens,omitempty"`
	TotalCostUSD          *float64       `json:"total_cost_usd,omitempty"`
	TotalSteps            *int           `json:"total_steps,omitempty"`
	Extra                 map[string]any `json:"extra,omitempty"`
}

// WriteHarbor writes one Harbor ATIF v1.7 trajectory. Unlike .faksession it is
// inert interchange data: decoding never restores state or invokes tools.
func WriteHarbor(w io.Writer, t HarborTrajectory) error {
	if t.SchemaVersion == "" {
		t.SchemaVersion = HarborVersion
	}
	if err := ValidateHarbor(t); err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	return enc.Encode(t)
}

// ReadHarbor negotiates only Harbor ATIF v1.7 and rejects unknown fields so
// required data from a newer schema cannot be silently discarded.
func ReadHarbor(r io.Reader) (HarborTrajectory, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return HarborTrajectory{}, err
	}
	var envelope struct {
		SchemaVersion string `json:"schema_version"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return HarborTrajectory{}, err
	}
	if envelope.SchemaVersion != HarborVersion {
		return HarborTrajectory{}, fmt.Errorf("atif: unsupported schema_version %q (legacy %q remains available through WriteBundle)", envelope.SchemaVersion, SchemaVersion)
	}
	var t HarborTrajectory
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&t); err != nil {
		return HarborTrajectory{}, fmt.Errorf("atif: decode %s: %w", HarborVersion, err)
	}
	if err := ValidateHarbor(t); err != nil {
		return HarborTrajectory{}, err
	}
	return t, nil
}

func ValidateHarbor(t HarborTrajectory) error {
	if t.SchemaVersion != HarborVersion {
		return fmt.Errorf("atif: schema_version must be %q", HarborVersion)
	}
	if t.Agent.Name == "" || t.Agent.Version == "" {
		return errors.New("atif: agent.name and agent.version are required")
	}
	for i, step := range t.Steps {
		if step.StepID < 1 {
			return fmt.Errorf("atif: steps[%d].step_id must be positive", i)
		}
		if step.Source != "system" && step.Source != "user" && step.Source != "agent" {
			return fmt.Errorf("atif: steps[%d].source %q is invalid", i, step.Source)
		}
		if step.Message == nil {
			return fmt.Errorf("atif: steps[%d].message is required", i)
		}
		for j, call := range step.ToolCalls {
			if call.ToolCallID == "" || call.FunctionName == "" || call.Arguments == nil {
				return fmt.Errorf("atif: steps[%d].tool_calls[%d] is incomplete", i, j)
			}
		}
		if step.Observation != nil {
			for j, result := range step.Observation.Results {
				for k, ref := range result.SubagentTrajectoryRef {
					if ref.TrajectoryID == nil && ref.TrajectoryPath == nil {
						return fmt.Errorf("atif: steps[%d].observation.results[%d].subagent_trajectory_ref[%d] is not resolvable", i, j, k)
					}
				}
			}
		}
	}
	for i := range t.SubagentTrajectories {
		if err := ValidateHarbor(t.SubagentTrajectories[i]); err != nil {
			return fmt.Errorf("atif: subagent_trajectories[%d]: %w", i, err)
		}
	}
	return nil
}
