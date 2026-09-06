package sessionview

import (
	"time"
)

// RowKind discriminates the type of materialized session event row.
type RowKind string

const (
	RowKindSession    RowKind = "session"
	RowKindLlmCall    RowKind = "llm_call"
	RowKindTokenUsage RowKind = "token_usage"
	RowKindToolCall   RowKind = "tool_call"
	RowKindAuditEvent RowKind = "audit_event"
)

// Row is the universal read-model interface implemented by all materialized view rows.
type Row interface {
	RowKind() RowKind
	RowID() string
	Session() string
	EventTime() time.Time
	OccurredAt() time.Time
}

// SessionRow captures session-level status, metadata, and identity.
type SessionRow struct {
	SessionID string            `json:"session_id"`
	TraceID   string            `json:"trace_id,omitempty"`
	ParentID  string            `json:"parent_id,omitempty"`
	State     string            `json:"state,omitempty"`
	Model     string            `json:"model,omitempty"`
	CreatedAt time.Time         `json:"created_at"`
	UpdatedAt time.Time         `json:"updated_at"`
	Labels    map[string]string `json:"labels,omitempty"`
	Metadata  map[string]any    `json:"metadata,omitempty"`
}

func (r SessionRow) RowKind() RowKind      { return RowKindSession }
func (r SessionRow) RowID() string         { return r.SessionID }
func (r SessionRow) Session() string       { return r.SessionID }
func (r SessionRow) EventTime() time.Time  { return r.UpdatedAt }
func (r SessionRow) OccurredAt() time.Time { return r.UpdatedAt }

func (r SessionRow) clone() SessionRow {
	c := r
	c.Labels = copyStringMap(r.Labels)
	c.Metadata = copyAnyMap(r.Metadata)
	return c
}

// LlmCallRow represents an individual LLM invocation / completion turn.
type LlmCallRow struct {
	CallID       string         `json:"call_id"`
	SessionID    string         `json:"session_id"`
	TurnID       string         `json:"turn_id,omitempty"`
	Model        string         `json:"model"`
	PromptTokens int64          `json:"prompt_tokens"`
	OutputTokens int64          `json:"output_tokens"`
	TotalTokens  int64          `json:"total_tokens"`
	CachedTokens int64          `json:"cached_tokens,omitempty"`
	Duration     time.Duration  `json:"duration"`
	FinishReason string         `json:"finish_reason,omitempty"`
	Error        string         `json:"error,omitempty"`
	Timestamp    time.Time      `json:"timestamp"`
	Attributes   map[string]any `json:"attributes,omitempty"`
}

func (r LlmCallRow) RowKind() RowKind      { return RowKindLlmCall }
func (r LlmCallRow) RowID() string         { return r.CallID }
func (r LlmCallRow) Session() string       { return r.SessionID }
func (r LlmCallRow) EventTime() time.Time  { return r.Timestamp }
func (r LlmCallRow) OccurredAt() time.Time { return r.Timestamp }

func (r LlmCallRow) clone() LlmCallRow {
	c := r
	c.Attributes = copyAnyMap(r.Attributes)
	return c
}

// TokenUsageRow represents a granular token debit / usage event.
type TokenUsageRow struct {
	UsageID      string    `json:"usage_id"`
	SessionID    string    `json:"session_id"`
	CallID       string    `json:"call_id,omitempty"`
	Model        string    `json:"model,omitempty"`
	PromptTokens int64     `json:"prompt_tokens"`
	OutputTokens int64     `json:"output_tokens"`
	CachedTokens int64     `json:"cached_tokens,omitempty"`
	TotalTokens  int64     `json:"total_tokens"`
	CostUSD      float64   `json:"cost_usd,omitempty"`
	Timestamp    time.Time `json:"timestamp"`
}

func (r TokenUsageRow) RowKind() RowKind      { return RowKindTokenUsage }
func (r TokenUsageRow) RowID() string         { return r.UsageID }
func (r TokenUsageRow) Session() string       { return r.SessionID }
func (r TokenUsageRow) EventTime() time.Time  { return r.Timestamp }
func (r TokenUsageRow) OccurredAt() time.Time { return r.Timestamp }

func (r TokenUsageRow) clone() TokenUsageRow {
	return r
}

// ToolCallRow represents an invoked tool execution within a session.
type ToolCallRow struct {
	ToolCallID string        `json:"tool_call_id"`
	SessionID  string        `json:"session_id"`
	TurnID     string        `json:"turn_id,omitempty"`
	Tool       string        `json:"tool,omitempty"`
	ToolName   string        `json:"tool_name,omitempty"`
	Arguments  string        `json:"arguments,omitempty"`
	Result     string        `json:"result,omitempty"`
	Duration   time.Duration `json:"duration"`
	Admitted   bool          `json:"admitted"`
	Error      string        `json:"error,omitempty"`
	Timestamp  time.Time     `json:"timestamp"`
}

func (r ToolCallRow) RowKind() RowKind      { return RowKindToolCall }
func (r ToolCallRow) RowID() string         { return r.ToolCallID }
func (r ToolCallRow) Session() string       { return r.SessionID }
func (r ToolCallRow) EventTime() time.Time  { return r.Timestamp }
func (r ToolCallRow) OccurredAt() time.Time { return r.Timestamp }

// Name returns the effective tool name, prioritizing ToolName if set, otherwise Tool.
func (r ToolCallRow) Name() string {
	if r.ToolName != "" {
		return r.ToolName
	}
	return r.Tool
}

func (r ToolCallRow) clone() ToolCallRow {
	return r
}

// AuditEventRow represents an audit, policy, or governance record in a session.
type AuditEventRow struct {
	EventID   string         `json:"event_id"`
	SessionID string         `json:"session_id"`
	TraceID   string         `json:"trace_id,omitempty"`
	Component string         `json:"component"`
	Action    string         `json:"action"`
	Severity  string         `json:"severity,omitempty"`
	Message   string         `json:"message"`
	Payload   map[string]any `json:"payload,omitempty"`
	Timestamp time.Time      `json:"timestamp"`
}

func (r AuditEventRow) RowKind() RowKind      { return RowKindAuditEvent }
func (r AuditEventRow) RowID() string         { return r.EventID }
func (r AuditEventRow) Session() string       { return r.SessionID }
func (r AuditEventRow) EventTime() time.Time  { return r.Timestamp }
func (r AuditEventRow) OccurredAt() time.Time { return r.Timestamp }

func (r AuditEventRow) clone() AuditEventRow {
	c := r
	c.Payload = copyAnyMap(r.Payload)
	return c
}

// SnapshotSummary aggregates cumulative point-in-time metrics across all
// processed events in the session. All counter metrics are monotonic and never
// decrease upon FIFO eviction of detail rows.
type SnapshotSummary struct {
	SessionID        string    `json:"session_id"`
	TotalEvents      int64     `json:"total_events"`
	TotalCalls       int64     `json:"total_calls"`
	TotalToolCalls   int64     `json:"total_tool_calls"`
	TotalAuditEvents int64     `json:"total_audit_events"`
	TotalTokens      int64     `json:"total_tokens"`
	PromptTokens     int64     `json:"prompt_tokens"`
	OutputTokens     int64     `json:"output_tokens"`
	CachedTokens     int64     `json:"cached_tokens"`
	TotalCostUSD     float64   `json:"total_cost_usd"`
	TotalErrors      int64     `json:"total_errors"`
	EvictedEvents    int64     `json:"evicted_events"`
	RetainedEvents   int64     `json:"retained_events"`
	Capacity         int       `json:"capacity"`
	FirstEventAt     time.Time `json:"first_event_at,omitempty"`
	LastEventAt      time.Time `json:"last_event_at,omitempty"`
	SnapshotAt       time.Time `json:"snapshot_at"`
}

// Drift returns the counter discrepancy between TotalEvents and (EvictedEvents + RetainedEvents).
// A correctly behaving MaterializedView always yields 0.
func (s SnapshotSummary) Drift() int64 {
	return s.TotalEvents - (s.EvictedEvents + s.RetainedEvents)
}

func copyStringMap(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func copyAnyMap(m map[string]any) map[string]any {
	if m == nil {
		return nil
	}
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
