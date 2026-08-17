package main

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

const (
	loopStateLive     = "live"
	loopStateTerminal = "terminal"
	loopStateUnknown  = "ambiguous"
)

type loopStateSummary struct {
	Live      []string
	Terminal  []string
	Ambiguous []string
}

var readSessionRows = func() ([]sessionregistry.Record, error) {
	return (sessionregistry.Store{Path: sessionregistry.DefaultPath()}).ReadAll()
}

func classifyLoopStates(rep codexLoopRecentReport) loopStateSummary {
	var out loopStateSummary
	rows, err := readSessionRows()
	if err != nil && !os.IsNotExist(err) {
		for _, d := range rep.Diagnoses {
			if d.Verdict == "LOOP" {
				out.Ambiguous = append(out.Ambiguous, d.SessionID)
			}
		}
		return out
	}
	for _, d := range rep.Diagnoses {
		if d.Verdict != "LOOP" {
			continue
		}
		state := loopStateForSession(rows, d.SessionID)
		switch state {
		case loopStateLive:
			out.Live = append(out.Live, d.SessionID)
		case loopStateTerminal:
			out.Terminal = append(out.Terminal, d.SessionID)
		default:
			out.Ambiguous = append(out.Ambiguous, d.SessionID)
		}
	}
	return out
}

func loopStateForSession(rows []sessionregistry.Record, sessionID string) string {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return loopStateUnknown
	}
	var matched []sessionregistry.Record
	for _, row := range rows {
		if strings.EqualFold(strings.TrimSpace(row.Identity.Runtime), "codex") && (row.Identity.SessionID == sessionID || row.Identity.ThreadID == sessionID) {
			matched = append(matched, row)
		}
	}
	if len(matched) == 0 {
		return loopStateUnknown
	}
	latest := matched[len(matched)-1]
	switch latest.State {
	case sessionregistry.StateActive:
		return loopStateLive
	case sessionregistry.StateCompleted, sessionregistry.StateFailed, sessionregistry.StateCancelled, sessionregistry.StateLost, sessionregistry.StateReaped:
		return loopStateTerminal
	default:
		return loopStateUnknown
	}
}

func loopStatePayload(s loopStateSummary) map[string]any {
	return map[string]any{
		"live_count":         len(s.Live),
		"live_sessions":      s.Live,
		"terminal_count":     len(s.Terminal),
		"terminal_sessions":  s.Terminal,
		"ambiguous_count":    len(s.Ambiguous),
		"ambiguous_sessions": s.Ambiguous,
	}
}
