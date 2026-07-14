package hostresurrect

import (
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/hostfault"
)

const Schema = "fak.host-resurrection.v1"

const (
	MaxLaunchesPerWindow = 10
	LaunchWindow         = 300 * time.Second
)

type Request struct {
	Schema       string   `json:"schema"`
	EventID      string   `json:"event_id"`
	CrashClass   string   `json:"crash_class"`
	Session      string   `json:"session"`
	CWD          string   `json:"cwd"`
	Command      []string `json:"command"`
	ResumeHandle string   `json:"resume_handle"`
	WindowID     string   `json:"window_id,omitempty"`
}

// Plan joins one independently-observed host crash to the durable interactive
// inventory. already is keyed by event-id + session handle, making repeated Event
// Log scans harmless. limit is the existing per-wave launch budget.
func Plan(signal hostfault.HostCrashSignal, rows []guardsessions.Row, already map[string]bool, limit int) []Request {
	if signal.Schema != hostfault.HostCrashSignalSchema || strings.TrimSpace(signal.EventID) == "" || limit <= 0 {
		return nil
	}
	live := guardsessions.LiveInteractive(rows)
	sort.SliceStable(live, func(i, j int) bool { return live[i].StartedAt < live[j].StartedAt })
	out := make([]Request, 0, min(limit, len(live)))
	for _, row := range live {
		key := Key(signal.EventID, row.Handle)
		if already[key] {
			continue
		}
		command := resumeCommand(row.Command, row.ResumeHandle)
		if len(command) == 0 {
			continue
		}
		out = append(out, Request{Schema: Schema, EventID: signal.EventID, CrashClass: string(signal.Class), Session: row.Handle, CWD: row.CWD, Command: command, ResumeHandle: row.ResumeHandle, WindowID: row.WindowID})
		if len(out) == limit {
			break
		}
	}
	return out
}

// resumeCommand makes continuation explicit. A registry row may contain the original
// cold launch (`claude`) or an already-resumable argv; the actuator must never cold-start.
func resumeCommand(original []string, handle string) []string {
	handle = strings.TrimSpace(handle)
	if len(original) == 0 || handle == "" {
		return nil
	}
	command := append([]string(nil), original...)
	for i, arg := range command {
		if arg == "--resume" {
			if i+1 < len(command) {
				command[i+1] = handle
				return command
			}
			return append(command, handle)
		}
	}
	return append(command, "--resume", handle)
}

func Key(eventID, handle string) string {
	return strings.TrimSpace(eventID) + "|" + strings.TrimSpace(handle)
}

// RecentCount returns launches in the rolling window for the global death-spiral cap.
func RecentCount(times []time.Time, now time.Time, window time.Duration) int {
	n := 0
	for _, at := range times {
		if !at.After(now) && now.Sub(at) <= window {
			n++
		}
	}
	return n
}
