package repoguard

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path"
	"strings"
)

// ReasonLiveMonitorOutputRead is the structured refusal token for Read-polling
// a live Monitor's tasks/<id>.output file before the harness has materialized it.
const ReasonLiveMonitorOutputRead = "LIVE_MONITOR_OUTPUT_READ"

const liveMonitorOutputFix = "do not Read-poll this file; live Monitor events are pushed, and tasks/<id>.output appears only after the stream ends"

// LiveMonitorOutputTaskID extracts <id> from a tasks/<id>.output path.
func LiveMonitorOutputTaskID(filePath string) (string, bool) {
	p := normalize(filePath)
	if p == "" {
		return "", false
	}
	p = strings.ReplaceAll(p, "\\", "/")
	clean := path.Clean(p)
	if clean == "." || clean == "/" {
		return "", false
	}
	if path.Base(path.Dir(clean)) != "tasks" {
		return "", false
	}
	base := path.Base(clean)
	id, ok := strings.CutSuffix(base, ".output")
	if !ok || strings.TrimSpace(id) == "" {
		return "", false
	}
	return id, true
}

// LiveMonitorTaskIDsFromJournal reads the tool-process journal shape and returns
// Monitor task ids that are still live. sessionID narrows the set when the hook
// payload carries a session; empty means all sessions.
func LiveMonitorTaskIDsFromJournal(r io.Reader, sessionID string) (map[string]bool, error) {
	type event struct {
		Kind    string `json:"kind"`
		CallID  string `json:"call_id"`
		Session string `json:"session"`
		Tool    string `json:"tool"`
	}
	type liveProc struct {
		tool    string
		session string
	}
	live := map[string]liveProc{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		raw := strings.TrimSpace(sc.Text())
		if raw == "" || strings.HasPrefix(raw, "#") {
			continue
		}
		var ev event
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			return nil, fmt.Errorf("toolproc journal line %d: %w", line, err)
		}
		switch ev.Kind {
		case "spawn":
			if isMonitorTool(ev.Tool) {
				live[ev.CallID] = liveProc{tool: ev.Tool, session: ev.Session}
			}
		case "exit", "kill":
			delete(live, ev.CallID)
		case "session_end":
			for id, proc := range live {
				if proc.session == ev.Session {
					delete(live, id)
				}
			}
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("toolproc journal: %w", err)
	}
	ids := map[string]bool{}
	for callID, proc := range live {
		if sessionID != "" && proc.session != "" && proc.session != sessionID {
			continue
		}
		id := monitorTaskIDForCall(callID)
		if id != "" {
			ids[id] = true
		}
	}
	return ids, nil
}

// LiveMonitorTaskIDsFromJournalFile opens path and folds live Monitor ids.
func LiveMonitorTaskIDsFromJournalFile(filePath, sessionID string) (map[string]bool, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return LiveMonitorTaskIDsFromJournal(f, sessionID)
}

func classifyLiveMonitorOutputRead(filePath string, liveMonitorIDs map[string]bool) []Violation {
	if len(liveMonitorIDs) == 0 {
		return nil
	}
	id, ok := LiveMonitorOutputTaskID(filePath)
	if !ok || !liveMonitorIDs[id] {
		return nil
	}
	return []Violation{{
		Reason:   ReasonLiveMonitorOutputRead,
		Op:       "read",
		Target:   filePath,
		Resolved: "tasks/" + id + ".output",
		Why:      "live Monitor output file is not materialized until the stream ends",
		Fix:      liveMonitorOutputFix,
	}}
}

func isMonitorTool(tool string) bool {
	return tool == "Monitor" || tool == "Monitor[bg]"
}

func monitorTaskIDForCall(callID string) string {
	id := strings.TrimSpace(callID)
	id = strings.TrimPrefix(id, "bg:")
	return id
}

func renderLiveMonitorOutputReason(violations []Violation) string {
	parts := make([]string, len(violations))
	for i, v := range violations {
		parts[i] = v.Target + " (" + v.Why + " - fix: " + v.Fix + ")"
	}
	return ReasonLiveMonitorOutputRead + ": this is a live Monitor output path, not a ready file. " +
		strings.Join(parts, "; ") + "."
}
