package sessionmine

import (
	"errors"
	"sort"
	"strings"
)

var ErrSessionNotFound = errors.New("historical session not found")

type HistoryOptions struct {
	Provider  string
	MinErrors int
	Limit     int
	SessionID string
	Tool      string
}

type HistoryReport struct {
	Schema    string    `json:"schema"`
	IndexedAt string    `json:"indexed_at"`
	Metrics   Metrics   `json:"metrics"`
	Sessions  []Session `json:"sessions,omitempty"`
	Session   *Session  `json:"session,omitempty"`
}

// ExploreIndex renders aggregate, ranked-list, or one-session views without
// reopening the raw provider transcripts represented by the index.
func ExploreIndex(path string, opts HistoryOptions) (HistoryReport, error) {
	state, err := LoadIndex(path)
	if err != nil {
		return HistoryReport{}, err
	}
	sessions := make([]Session, 0, len(state.Files))
	for _, file := range state.Files {
		s := file.Session
		if opts.Provider != "" && !strings.EqualFold(s.Provider, opts.Provider) {
			continue
		}
		if s.ToolErrors < opts.MinErrors {
			continue
		}
		if opts.Tool != "" && !trajectoryContains(s.Trajectory, opts.Tool) {
			continue
		}
		sessions = append(sessions, s)
	}
	sort.Slice(sessions, func(i, j int) bool {
		if sessions[i].ToolErrors != sessions[j].ToolErrors {
			return sessions[i].ToolErrors > sessions[j].ToolErrors
		}
		if sessions[i].EndedAt != sessions[j].EndedAt {
			return sessions[i].EndedAt > sessions[j].EndedAt
		}
		return sessions[i].ID < sessions[j].ID
	})
	report := HistoryReport{Schema: "fak-session-history/1", IndexedAt: state.UpdatedAt, Metrics: historyMetrics(sessions)}
	if opts.SessionID != "" {
		for i := range sessions {
			if sessions[i].ID == opts.SessionID {
				report.Session = &sessions[i]
				return report, nil
			}
		}
		return HistoryReport{}, ErrSessionNotFound
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = 25
	}
	if limit > len(sessions) {
		limit = len(sessions)
	}
	report.Sessions = sessions[:limit]
	return report, nil
}

func trajectoryContains(trajectory []string, step string) bool {
	for _, candidate := range trajectory {
		if candidate == step {
			return true
		}
	}
	return false
}

func historyMetrics(sessions []Session) Metrics {
	m := Metrics{ByProvider: map[string]int{}, Sessions: len(sessions)}
	durations := make([]int64, 0, len(sessions))
	for _, s := range sessions {
		m.ByProvider[s.Provider]++
		m.ToolCalls += s.ToolCalls
		m.ToolResults += s.ToolResults
		m.ToolErrors += s.ToolErrors
		m.UserTurns += s.UserTurns
		m.AssistantTurns += s.AssistantTurns
		if s.DurationMS > 0 {
			m.DurationsMS += s.DurationMS
			durations = append(durations, s.DurationMS)
		}
	}
	if len(durations) > 0 {
		sort.Slice(durations, func(i, j int) bool { return durations[i] < durations[j] })
		m.P50DurationMS = percentile(durations, 50)
		m.P95DurationMS = percentile(durations, 95)
	}
	return m
}
