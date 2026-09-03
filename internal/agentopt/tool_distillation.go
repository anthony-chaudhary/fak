package agentopt

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// Family 15: Training & adaptation for agent optimization.
// Tool-use distillation dataset pipeline from frontier agent traces.

// TrajectoryStep represents one step or action within a frontier agent trajectory.
type TrajectoryStep struct {
	StepIndex  int        `json:"step_index"`
	Thought    string     `json:"thought,omitempty"`
	ToolCall   ToolCall   `json:"tool_call"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolResult string     `json:"tool_result,omitempty"`
	Error      string     `json:"error,omitempty"`
	DurationMs int64      `json:"duration_ms,omitempty"`
}

// AllCalls returns all tool calls invoked in this step.
func (s TrajectoryStep) AllCalls() []ToolCall {
	if len(s.ToolCalls) > 0 {
		return s.ToolCalls
	}
	if s.ToolCall.Name != "" {
		return []ToolCall{s.ToolCall}
	}
	return nil
}

// FrontierTrajectory represents an uncurated agent execution trace collected from a frontier model.
type FrontierTrajectory struct {
	ID            string           `json:"id"`
	SystemPrompt  string           `json:"system_prompt"`
	UserQuery     string           `json:"user_query"`
	Steps         []TrajectoryStep `json:"steps"`
	FinalResponse string           `json:"final_response"`
	Success       bool             `json:"success"`
	Metadata      map[string]any   `json:"metadata,omitempty"`
}

// DistillationExample represents a curated, high-fidelity demonstration formatted
// for instruction-tuning smaller local models on tool-use capabilities.
type DistillationExample struct {
	ID            string         `json:"id,omitempty"`
	SystemPrompt  string         `json:"system_prompt"`
	UserQuery     string         `json:"user_query"`
	Thought       string         `json:"thought,omitempty"`
	ToolCalls     []ToolCall     `json:"tool_calls"`
	FinalResponse string         `json:"final_response"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// DistillationFilterConfig defines the rules for admitting frontier traces into training datasets.
type DistillationFilterConfig struct {
	// DiscardFailedCalls rejects trajectories containing any step with an error.
	DiscardFailedCalls bool `json:"discard_failed_calls"`

	// DiscardCycles rejects trajectories containing repeated identical tool invocations.
	DiscardCycles bool `json:"discard_cycles"`

	// DiscardRepeatedReads rejects trajectories with repeated identical read-only tool calls.
	DiscardRepeatedReads bool `json:"discard_repeated_reads"`

	// DiscardBacktracking rejects trajectories with explicit trial-and-error reasoning or recovery.
	DiscardBacktracking bool `json:"discard_backtracking"`

	// RequireSuccess requires the trajectory to have succeeded.
	RequireSuccess bool `json:"require_success"`

	// RequireFinalResponse requires a non-empty final response.
	RequireFinalResponse bool `json:"require_final_response"`

	// RequireToolCalls requires at least MinToolCalls.
	RequireToolCalls bool `json:"require_tool_calls"`

	// MinToolCalls is the minimum number of tool calls required.
	MinToolCalls int `json:"min_tool_calls"`

	// MaxToolCalls is the maximum number of tool calls permitted (0 for unlimited).
	MaxToolCalls int `json:"max_tool_calls"`

	// ValidateSchema enforces that tool calls conform strictly to registered ToolSchema.
	ValidateSchema bool `json:"validate_schema"`
}

// DefaultDistillationFilterConfig returns production-grade filtering defaults for Family 15 distillation.
func DefaultDistillationFilterConfig() DistillationFilterConfig {
	return DistillationFilterConfig{
		DiscardFailedCalls:   true,
		DiscardCycles:        true,
		DiscardRepeatedReads: true,
		DiscardBacktracking:  true,
		RequireSuccess:       true,
		RequireFinalResponse: true,
		RequireToolCalls:     true,
		MinToolCalls:         1,
		MaxToolCalls:         20,
		ValidateSchema:       true,
	}
}

// FilterResult details whether a trajectory was accepted or rejected by the distillation filters.
type FilterResult struct {
	Admitted bool     `json:"admitted"`
	Reason   string   `json:"reason,omitempty"`
	Issues   []string `json:"issues,omitempty"`
}

// DistillationFilterStats collects rejection and admission metrics across a distillation run.
type DistillationFilterStats struct {
	TotalProcessed  int `json:"total_processed"`
	Admitted        int `json:"admitted"`
	Discarded       int `json:"discarded"`
	DiscardedErrors int `json:"discarded_errors"`
	DiscardedCycles int `json:"discarded_cycles"`
	DiscardedSchema int `json:"discarded_schema"`
	DiscardedStatus int `json:"discarded_status"`
	DiscardedLength int `json:"discarded_length"`
}

// ToolDistillationPipeline ingests frontier agent traces, filters out backtracking,
// failed calls, loops, and schema violations, and produces instruction-tuning datasets.
type ToolDistillationPipeline struct {
	config           DistillationFilterConfig
	schemaValidator  *SchemaValidator
	defaultSysPrompt string
}

// NewToolDistillationPipeline creates a pipeline with filter rules and an optional schema validator.
func NewToolDistillationPipeline(cfg DistillationFilterConfig, validator *SchemaValidator) *ToolDistillationPipeline {
	return &ToolDistillationPipeline{
		config:          cfg,
		schemaValidator: validator,
	}
}

// RegisterSchema adds a tool schema to the pipeline's validator.
func (p *ToolDistillationPipeline) RegisterSchema(s ToolSchema) {
	if p.schemaValidator == nil {
		p.schemaValidator = NewSchemaValidator(s)
		return
	}
	p.schemaValidator.RegisterSchema(s)
}

// SetDefaultSystemPrompt sets a fallback system prompt when a trajectory does not supply one.
func (p *ToolDistillationPipeline) SetDefaultSystemPrompt(prompt string) {
	p.defaultSysPrompt = prompt
}

var backtrackingKeywords = []string{
	"that failed",
	"failed to find",
	"didn't work",
	"did not work",
	"let me backtrack",
	"let's backtrack",
	"backtracking",
	"try another approach",
	"try a different approach",
	"reverting back",
	"undo previous",
}

func isKnownReadOnly(name string) bool {
	switch strings.ToLower(name) {
	case "read", "glob", "grep", "read_file", "list_dir", "cat", "search", "view", "get":
		return true
	default:
		return false
	}
}

func containsBacktrackingKeywords(thought string) bool {
	lower := strings.ToLower(thought)
	for _, phrase := range backtrackingKeywords {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

func isErrorOutput(out string) bool {
	trimmed := strings.TrimSpace(out)
	return strings.HasPrefix(trimmed, "Error: ") || strings.HasPrefix(trimmed, "FATAL: ") || strings.HasPrefix(trimmed, "panic: ")
}

// FilterTrajectory evaluates a single frontier trajectory against all active filter rules.
func (p *ToolDistillationPipeline) FilterTrajectory(traj FrontierTrajectory) FilterResult {
	if p.config.RequireSuccess && !traj.Success {
		return FilterResult{
			Admitted: false,
			Reason:   "trajectory execution did not succeed",
		}
	}

	if p.config.RequireFinalResponse && strings.TrimSpace(traj.FinalResponse) == "" {
		return FilterResult{
			Admitted: false,
			Reason:   "trajectory has empty final response",
		}
	}

	var allCalls []ToolCall
	callCounts := make(map[string]int)
	readCounts := make(map[string]int)

	for stepIdx, step := range traj.Steps {
		if p.config.DiscardFailedCalls && (step.Error != "" || isErrorOutput(step.ToolResult)) {
			errMsg := step.Error
			if errMsg == "" {
				errMsg = step.ToolResult
			}
			return FilterResult{
				Admitted: false,
				Reason:   fmt.Sprintf("failed tool call in step %d: %s", stepIdx, errMsg),
			}
		}

		if p.config.DiscardBacktracking && containsBacktrackingKeywords(step.Thought) {
			return FilterResult{
				Admitted: false,
				Reason:   fmt.Sprintf("backtracking detected in step %d thought: %q", stepIdx, step.Thought),
			}
		}

		calls := step.AllCalls()
		for _, call := range calls {
			if call.Name == "" {
				continue
			}
			allCalls = append(allCalls, call)
			key := DigestCall(call.Name, call.Args)
			callCounts[key]++

			isRead := call.ReadOnly || isKnownReadOnly(call.Name)
			if isRead {
				readCounts[key]++
			}
		}
	}

	if p.config.RequireToolCalls {
		if len(allCalls) < p.config.MinToolCalls {
			return FilterResult{
				Admitted: false,
				Reason:   fmt.Sprintf("trajectory contains %d tool calls, minimum required is %d", len(allCalls), p.config.MinToolCalls),
			}
		}
	}

	if p.config.MaxToolCalls > 0 && len(allCalls) > p.config.MaxToolCalls {
		return FilterResult{
			Admitted: false,
			Reason:   fmt.Sprintf("trajectory contains %d tool calls, maximum allowed is %d", len(allCalls), p.config.MaxToolCalls),
		}
	}

	if p.config.DiscardRepeatedReads {
		for key, count := range readCounts {
			if count > 1 {
				return FilterResult{
					Admitted: false,
					Reason:   fmt.Sprintf("repeated identical read-only tool call detected (hash: %s, count: %d)", key, count),
				}
			}
		}
	}

	if p.config.DiscardCycles {
		for key, count := range callCounts {
			if count > 1 {
				return FilterResult{
					Admitted: false,
					Reason:   fmt.Sprintf("cycle detected: repeated identical tool call (hash: %s, count: %d)", key, count),
				}
			}
		}
	}

	if p.config.ValidateSchema && p.schemaValidator != nil {
		for _, call := range allCalls {
			res := p.schemaValidator.ValidateToolCallMap(call.Name, call.Args)
			if !res.Valid {
				return FilterResult{
					Admitted: false,
					Reason:   fmt.Sprintf("tool call %q violated schema: %s", call.Name, strings.Join(res.Violations, "; ")),
					Issues:   res.Violations,
				}
			}
		}
	}

	return FilterResult{
		Admitted: true,
	}
}

// ProcessTrajectory admits a clean trajectory and formats it into a DistillationExample.
func (p *ToolDistillationPipeline) ProcessTrajectory(traj FrontierTrajectory) (*DistillationExample, error) {
	res := p.FilterTrajectory(traj)
	if !res.Admitted {
		return nil, fmt.Errorf("trajectory rejected: %s", res.Reason)
	}

	var thoughts []string
	var toolCalls []ToolCall

	for _, step := range traj.Steps {
		t := strings.TrimSpace(step.Thought)
		if t != "" {
			thoughts = append(thoughts, t)
		}
		toolCalls = append(toolCalls, step.AllCalls()...)
	}

	sysPrompt := traj.SystemPrompt
	if sysPrompt == "" {
		sysPrompt = p.defaultSysPrompt
	}

	example := &DistillationExample{
		ID:            traj.ID,
		SystemPrompt:  sysPrompt,
		UserQuery:     traj.UserQuery,
		Thought:       strings.Join(thoughts, "\n"),
		ToolCalls:     toolCalls,
		FinalResponse: traj.FinalResponse,
		Metadata:      traj.Metadata,
	}

	return example, nil
}

// Process filters a collection of trajectories, tracking rejection statistics, and returns a DistillationDataset.
func (p *ToolDistillationPipeline) Process(trajectories []FrontierTrajectory) (*DistillationDataset, DistillationFilterStats) {
	stats := DistillationFilterStats{
		TotalProcessed: len(trajectories),
	}

	dataset := &DistillationDataset{
		CreatedAt: time.Now().UTC(),
	}

	for _, traj := range trajectories {
		res := p.FilterTrajectory(traj)
		if !res.Admitted {
			stats.Discarded++
			switch {
			case strings.Contains(res.Reason, "failed tool call") || strings.Contains(res.Reason, "backtracking"):
				stats.DiscardedErrors++
			case strings.Contains(res.Reason, "cycle") || strings.Contains(res.Reason, "repeated"):
				stats.DiscardedCycles++
			case strings.Contains(res.Reason, "schema"):
				stats.DiscardedSchema++
			case strings.Contains(res.Reason, "succeed") || strings.Contains(res.Reason, "final response"):
				stats.DiscardedStatus++
			case strings.Contains(res.Reason, "tool calls"):
				stats.DiscardedLength++
			default:
				stats.DiscardedErrors++
			}
			continue
		}

		ex, err := p.ProcessTrajectory(traj)
		if err == nil && ex != nil {
			dataset.Examples = append(dataset.Examples, *ex)
			stats.Admitted++
		}
	}

	return dataset, stats
}

// DistillationDataset represents a curated instruction-tuning dataset ready for model training.
type DistillationDataset struct {
	Name        string                `json:"name,omitempty"`
	Description string                `json:"description,omitempty"`
	Examples    []DistillationExample `json:"examples"`
	CreatedAt   time.Time             `json:"created_at,omitempty"`
}

// Len returns the count of examples in the dataset.
func (d *DistillationDataset) Len() int {
	if d == nil {
		return 0
	}
	return len(d.Examples)
}

// ToJSON exports the full dataset as indented JSON bytes.
func (d *DistillationDataset) ToJSON() ([]byte, error) {
	return json.MarshalIndent(d, "", "  ")
}

// ToJSONL writes each DistillationExample as a single JSON line to w.
func (d *DistillationDataset) ToJSONL(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, ex := range d.Examples {
		if err := enc.Encode(ex); err != nil {
			return err
		}
	}
	return nil
}

// ChatMessage represents a single message in an instruction-tuning chat sequence.
type ChatMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	Thought   string     `json:"thought,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ChatDemonstration formats an example into multi-turn chat format (OpenAI / Qwen chat template compatible).
type ChatDemonstration struct {
	ID       string        `json:"id,omitempty"`
	Messages []ChatMessage `json:"messages"`
}

// ToChatDemonstration converts a DistillationExample into a ChatDemonstration.
func (e DistillationExample) ToChatDemonstration() ChatDemonstration {
	var msgs []ChatMessage
	if e.SystemPrompt != "" {
		msgs = append(msgs, ChatMessage{
			Role:    "system",
			Content: e.SystemPrompt,
		})
	}
	msgs = append(msgs, ChatMessage{
		Role:    "user",
		Content: e.UserQuery,
	})
	msgs = append(msgs, ChatMessage{
		Role:      "assistant",
		Thought:   e.Thought,
		ToolCalls: e.ToolCalls,
		Content:   e.FinalResponse,
	})
	return ChatDemonstration{
		ID:       e.ID,
		Messages: msgs,
	}
}

// ToChatJSONL writes each example converted to ChatDemonstration format as a JSON line.
func (d *DistillationDataset) ToChatJSONL(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, ex := range d.Examples {
		if err := enc.Encode(ex.ToChatDemonstration()); err != nil {
			return err
		}
	}
	return nil
}

// AlpacaDemonstration formats an example into classic Alpaca instruction-input-output structure.
type AlpacaDemonstration struct {
	Instruction string `json:"instruction"`
	Input       string `json:"input,omitempty"`
	Output      string `json:"output"`
}

// ToAlpacaDemonstration converts a DistillationExample into an AlpacaDemonstration.
func (e DistillationExample) ToAlpacaDemonstration() AlpacaDemonstration {
	var outputParts []string
	if e.Thought != "" {
		outputParts = append(outputParts, "<thought>\n"+e.Thought+"\n</thought>")
	}
	if len(e.ToolCalls) > 0 {
		b, _ := json.MarshalIndent(e.ToolCalls, "", "  ")
		outputParts = append(outputParts, "<tool_calls>\n"+string(b)+"\n</tool_calls>")
	}
	if e.FinalResponse != "" {
		outputParts = append(outputParts, e.FinalResponse)
	}

	instruction := e.SystemPrompt
	input := e.UserQuery
	if instruction == "" {
		instruction = e.UserQuery
		input = ""
	}

	return AlpacaDemonstration{
		Instruction: instruction,
		Input:       input,
		Output:      strings.Join(outputParts, "\n\n"),
	}
}

// ToAlpacaJSONL writes each example converted to Alpaca format as a JSON line.
func (d *DistillationDataset) ToAlpacaJSONL(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, ex := range d.Examples {
		if err := enc.Encode(ex.ToAlpacaDemonstration()); err != nil {
			return err
		}
	}
	return nil
}

// ShareGPTTurn represents one turn in a ShareGPT conversation.
type ShareGPTTurn struct {
	From  string `json:"from"`
	Value string `json:"value"`
}

// ShareGPTDemonstration formats an example into ShareGPT conversations schema.
type ShareGPTDemonstration struct {
	Conversations []ShareGPTTurn `json:"conversations"`
}

// ToShareGPTDemonstration converts a DistillationExample into a ShareGPTDemonstration.
func (e DistillationExample) ToShareGPTDemonstration() ShareGPTDemonstration {
	var turns []ShareGPTTurn
	if e.SystemPrompt != "" {
		turns = append(turns, ShareGPTTurn{From: "system", Value: e.SystemPrompt})
	}
	turns = append(turns, ShareGPTTurn{From: "human", Value: e.UserQuery})

	var assistantContent []string
	if e.Thought != "" {
		assistantContent = append(assistantContent, "<thought>\n"+e.Thought+"\n</thought>")
	}
	if len(e.ToolCalls) > 0 {
		b, _ := json.Marshal(e.ToolCalls)
		assistantContent = append(assistantContent, string(b))
	}
	if e.FinalResponse != "" {
		assistantContent = append(assistantContent, e.FinalResponse)
	}

	turns = append(turns, ShareGPTTurn{
		From:  "gpt",
		Value: strings.Join(assistantContent, "\n\n"),
	})
	return ShareGPTDemonstration{Conversations: turns}
}

// ToShareGPTJSONL writes each example converted to ShareGPT format as a JSON line.
func (d *DistillationDataset) ToShareGPTJSONL(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, ex := range d.Examples {
		if err := enc.Encode(ex.ToShareGPTDemonstration()); err != nil {
			return err
		}
	}
	return nil
}
