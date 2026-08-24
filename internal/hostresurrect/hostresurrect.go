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

// Cohort is a liveness snapshot captured by the control loop before a host
// crash. PID is part of the identity so a stale registry handle cannot borrow
// a later process that reused the same numeric PID.
type Cohort struct {
	CapturedAt string        `json:"captured_at"`
	Sessions   []CohortEntry `json:"sessions"`
}

type CohortEntry struct {
	Handle string `json:"handle"`
	PID    int    `json:"pid"`
}

// Selection makes a relaunch wave auditable without exposing commands or
// resume handles. Every untombstoned row is either a candidate or has a typed
// exclusion below.
type Selection struct {
	Inventory              int `json:"inventory"`
	Untombstoned           int `json:"untombstoned"`
	SnapshotSize           int `json:"snapshot_size"`
	Candidates             int `json:"candidates"`
	ExcludedNotInCohort    int `json:"excluded_not_in_cohort"`
	ExcludedPIDMismatch    int `json:"excluded_pid_mismatch"`
	ExcludedInvalidResume  int `json:"excluded_invalid_resume"`
	ExcludedAlreadyHandled int `json:"excluded_already_handled"`
	Selected               int `json:"selected"`
}

// Plan joins one independently-observed host crash to the durable interactive
// inventory. already is keyed by event-id + session handle, making repeated Event
// Log scans harmless. limit is the existing per-wave launch budget.
func Plan(signal hostfault.HostCrashSignal, rows []guardsessions.Row, cohort Cohort, already map[string]bool, limit int) ([]Request, Selection) {
	counts := Selection{Inventory: len(rows), SnapshotSize: len(cohort.Sessions)}
	if signal.Schema != hostfault.HostCrashSignalSchema || strings.TrimSpace(signal.EventID) == "" || limit <= 0 {
		return nil, counts
	}
	live := guardsessions.LiveInteractive(rows)
	counts.Untombstoned = len(live)
	members := make(map[string]int, len(cohort.Sessions))
	for _, entry := range cohort.Sessions {
		if strings.TrimSpace(entry.Handle) != "" && entry.PID > 0 {
			members[entry.Handle] = entry.PID
		}
	}
	// Prefer the newest witnessed sessions if a cohort itself exceeds the cap.
	sort.SliceStable(live, func(i, j int) bool { return live[i].StartedAt > live[j].StartedAt })
	out := make([]Request, 0, min(limit, len(live)))
	for _, row := range live {
		pid, ok := members[row.Handle]
		if !ok {
			counts.ExcludedNotInCohort++
			continue
		}
		if pid != row.PID {
			counts.ExcludedPIDMismatch++
			continue
		}
		counts.Candidates++
		key := Key(signal.EventID, row.Handle)
		if already[key] {
			counts.ExcludedAlreadyHandled++
			continue
		}
		command := resumeCommand(row.Command, row.ResumeHandle)
		if len(command) == 0 {
			counts.ExcludedInvalidResume++
			continue
		}
		out = append(out, Request{Schema: Schema, EventID: signal.EventID, CrashClass: string(signal.Class), Session: row.Handle, CWD: row.CWD, Command: command, ResumeHandle: row.ResumeHandle, WindowID: row.WindowID})
		if len(out) == limit {
			break
		}
	}
	counts.Selected = len(out)
	return out, counts
}

// resumeCommand makes continuation explicit. A registry row may contain the original
// cold launch (`claude`) or an already-resumable argv; the actuator must never cold-start.
func resumeCommand(original []string, handle string) []string {
	handle = strings.TrimSpace(handle)
	if len(original) == 0 || handle == "" {
		return nil
	}
	command := make([]string, 0, len(original)+2)
	for i := 0; i < len(original); i++ {
		switch original[i] {
		case "--resume":
			if i+1 < len(original) {
				i++
			}
			continue
		case "--continue", "-c":
			continue
		default:
			command = append(command, original[i])
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
