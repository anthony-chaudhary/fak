package main

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

const event1000WindowMS int64 = 2 * 60 * 1000

var readEvent1000Records = gatherWinConsoleFaultRecords

type event1000Cause struct {
	AtMS      int64  `json:"at_ms"`
	Tool      string `json:"tool"`
	Cause     string `json:"cause"`
	Detail    string `json:"detail"`
	Source    string `json:"source"`
	SessionID string `json:"session_id,omitempty"`
}

func matchEvent1000Crashes(rows []sessionjournal.Classified, events []toolprocgate.ConsoleFaultEvent, windowMS int64) []event1000Cause {
	if windowMS <= 0 {
		windowMS = event1000WindowMS
	}
	crashes := make([]sessionjournal.Classified, 0, len(rows))
	for _, row := range rows {
		if row.Status == sessionjournal.StatusCrashed {
			crashes = append(crashes, row)
		}
	}
	var out []event1000Cause
	for _, event := range events {
		if event.Class != toolprocgate.ConsoleRendererExit {
			continue
		}
		bestID := ""
		bestDelta := windowMS + 1
		for _, crash := range crashes {
			at := crash.Session.LastSeen.UnixMilli()
			delta := event.AtMS - at
			if delta < 0 {
				delta = -delta
			}
			if delta <= windowMS && delta < bestDelta {
				bestDelta, bestID = delta, crash.Session.ID
			}
		}
		if bestID == "" {
			continue
		}
		out = append(out, event1000Cause{
			AtMS: event.AtMS, Tool: event.Tool, Cause: event1000CauseName(event),
			Detail: event.Detail, Source: "windows_eventlog_1000", SessionID: bestID,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].AtMS < out[j].AtMS })
	return out
}

func event1000CauseName(event toolprocgate.ConsoleFaultEvent) string {
	detail := strings.ToLower(event.Detail)
	switch {
	case strings.Contains(detail, "microsoftterminal") || strings.Contains(detail, "windowsterminal") || strings.Contains(detail, "terminal.control"):
		return "WINDOWS_TERMINAL_CRASH"
	case strings.Contains(detail, "winappruntime"):
		return "WINAPPRUNTIME_CRASH"
	default:
		return fmt.Sprintf("APPLICATION_ERROR_%s", strings.ToUpper(event.Tool))
	}
}
