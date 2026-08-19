package agentquery

import (
	"sort"
	"strings"
	"time"
)

const Schema = "fak-agents/1"

type Row struct {
	AgentID          string   `json:"agent_id"`
	LogicalSessionID string   `json:"logical_session_id"`
	ExecutionEpoch   *string  `json:"execution_epoch"`
	RootID           *string  `json:"root_id"`
	ParentID         *string  `json:"parent_id"`
	Group            *string  `json:"group"`
	Lane             *string  `json:"lane"`
	Host             *string  `json:"host"`
	PID              *int     `json:"pid"`
	State            string   `json:"state"`
	Liveness         string   `json:"liveness"`
	StartedAt        *string  `json:"started_at"`
	LastProgressAt   *string  `json:"last_progress_at"`
	EndedAt          *string  `json:"ended_at"`
	ObservedAt       string   `json:"observed_at"`
	ElapsedMS        *int64   `json:"elapsed_ms"`
	Model            *string  `json:"model"`
	Provider         *string  `json:"provider"`
	Turns            *int64   `json:"turns"`
	ToolCalls        *int64   `json:"tool_calls"`
	Cost             *float64 `json:"cost"`
	StopReason       *string  `json:"stop_reason"`
	Source           string   `json:"source"`
	SourceVersion    string   `json:"source_version"`
	Stale            bool     `json:"stale"`
}
type SourceHealth struct {
	Status          string `json:"status"`
	TotalRows       int    `json:"total_rows"`
	BlankRows       int    `json:"blank_rows"`
	AcceptedRows    int    `json:"accepted_rows"`
	MalformedRows   int    `json:"malformed_rows"`
	WrongSchemaRows int    `json:"wrong_schema_rows"`
	MissingIDRows   int    `json:"missing_id_rows"`
	ScanError       string `json:"scan_error,omitempty"`
	ReadError       string `json:"read_error,omitempty"`
}

type Metadata struct {
	Schema       string        `json:"schema"`
	Source       string        `json:"source"`
	ObservedAt   string        `json:"observed_at"`
	Freshness    string        `json:"freshness"`
	Truncated    bool          `json:"truncated"`
	Limit        int           `json:"limit"`
	LiveRows     int           `json:"live_rows"`
	HistoryRows  int           `json:"history_rows"`
	Deduplicated int           `json:"deduplicated"`
	History      *SourceHealth `json:"history"`
	AsOf         *string       `json:"as_of"`
}
type Result struct {
	Metadata Metadata `json:"metadata"`
	Rows     []Row    `json:"rows"`
}

// Union joins only exact execution identities. Live observations take precedence over matching journal facts.
func Union(live, history []Row, source string, activeOnly bool, limit int, observed time.Time) Result {
	selected := make([]Row, 0, len(live)+len(history))
	dedup := 0
	switch source {
	case "live":
		selected = append(selected, live...)
	case "history":
		selected = append(selected, history...)
	default:
		seen := make(map[string]struct{}, len(live))
		for _, r := range live {
			selected = append(selected, r)
			seen[r.AgentID] = struct{}{}
		}
		for _, r := range history {
			if _, ok := seen[r.AgentID]; ok {
				dedup++
				continue
			}
			selected = append(selected, r)
		}
	}
	if activeOnly {
		dst := selected[:0]
		for _, r := range selected {
			if strings.EqualFold(r.Liveness, "LIVE") {
				dst = append(dst, r)
			}
		}
		selected = dst
	}
	sort.SliceStable(selected, func(i, j int) bool {
		a, b := int64(-1), int64(-1)
		if selected[i].ElapsedMS != nil {
			a = *selected[i].ElapsedMS
		}
		if selected[j].ElapsedMS != nil {
			b = *selected[j].ElapsedMS
		}
		if a != b {
			return a > b
		}
		return selected[i].AgentID < selected[j].AgentID
	})
	truncated := false
	if limit > 0 && len(selected) > limit {
		selected = selected[:limit]
		truncated = true
	}
	freshness := "current"
	if source == "history" {
		freshness = "recorded"
	}
	return Result{Metadata: Metadata{Schema: Schema, Source: source, ObservedAt: observed.UTC().Format(time.RFC3339), Freshness: freshness, Truncated: truncated, Limit: limit, LiveRows: len(live), HistoryRows: len(history), Deduplicated: dedup}, Rows: selected}
}
