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
	Handle    string `json:"handle"`
	PID       int    `json:"pid"`
	StartedAt string `json:"started_at"`
	HostPID   int    `json:"host_pid,omitempty"`
}

// Selection makes a relaunch wave auditable without exposing commands or
// resume handles. Every untombstoned row is either a candidate or has a typed
// exclusion below.
type Selection struct {
	Inventory              int `json:"inventory"`
	Untombstoned           int `json:"untombstoned"`
	SnapshotSize           int `json:"snapshot_size"`
	Candidates             int `json:"candidates"`
	ExcludedNotOptedIn     int `json:"excluded_not_opted_in"`
	ExcludedNotInCohort    int `json:"excluded_not_in_cohort"`
	ExcludedPIDMismatch    int `json:"excluded_pid_mismatch"`
	ExcludedHostMismatch   int `json:"excluded_host_mismatch"`
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
	members := make(map[string]CohortEntry, len(cohort.Sessions))
	for _, entry := range cohort.Sessions {
		if strings.TrimSpace(entry.Handle) != "" && entry.PID > 0 {
			members[entry.Handle] = entry
		}
	}
	// Prefer the newest witnessed sessions if a cohort itself exceeds the cap.
	sort.SliceStable(live, func(i, j int) bool { return live[i].StartedAt > live[j].StartedAt })
	out := make([]Request, 0, min(limit, len(live)))
	for _, row := range live {
		if !guardsessions.HostRecoveryEnabled(row) {
			counts.ExcludedNotOptedIn++
			continue
		}
		entry, ok := members[row.Handle]
		if !ok {
			counts.ExcludedNotInCohort++
			continue
		}
		if entry.PID != row.PID || entry.StartedAt != row.StartedAt {
			counts.ExcludedPIDMismatch++
			continue
		}
		if terminalHostSignal(signal) && (entry.HostPID <= 0 || int64(entry.HostPID) != signal.HostPID) {
			counts.ExcludedHostMismatch++
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
		if len(out) < limit {
			out = append(out, Request{Schema: Schema, EventID: signal.EventID, CrashClass: string(signal.Class), Session: row.Handle, CWD: row.CWD, Command: command, ResumeHandle: row.ResumeHandle, WindowID: row.WindowID})
		}
	}
	counts.Selected = len(out)
	return out, counts
}

// resumeCommand makes continuation explicit. A registry row may contain the original
// cold launch (`claude`) or an already-resumable argv; the actuator must never cold-start.
func terminalHostSignal(signal hostfault.HostCrashSignal) bool {
	app := strings.ToLower(strings.TrimSpace(signal.FaultingApp))
	return signal.HostPID > 0 && (app == "windowsterminal.exe" || app == "openconsole.exe")
}

func resumeCommand(original []string, handle string) []string {
	handle = strings.TrimSpace(handle)
	if len(original) == 0 || handle == "" {
		return nil
	}
	if commandBase(original[0]) == "codex" {
		return exactCodexResumeCommand(original)
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

// Codex continuation uses a subcommand and Codex's own UUID. A fak guard handle
// is not a Codex thread identity, and --last/picker fallback could attach the
// wrong transcript, so only a pre-bound exact invocation is actuator-safe.
func exactCodexResumeCommand(command []string) []string {
	resumeAt := -1
	for i := 1; i+1 < len(command); i++ {
		if command[i] == "resume" && isUUID(command[i+1]) {
			resumeAt = i
			break
		}
	}
	if resumeAt < 0 {
		return nil
	}

	end := len(command)
	if end >= resumeAt+4 && command[end-2] == "--resume" && strings.TrimSpace(command[end-1]) != "" {
		// Older generic resurrection appended fak's guard handle to Codex's
		// already-exact invocation. Strip that poisoned suffix while retaining
		// Codex's UUID and any legitimate resume-subcommand arguments.
		end -= 2
	}
	return append([]string(nil), command[:end]...)
}

func commandBase(command string) string {
	command = strings.TrimSpace(strings.ReplaceAll(command, `\`, "/"))
	if cut := strings.LastIndexByte(command, '/'); cut >= 0 {
		command = command[cut+1:]
	}
	command = strings.ToLower(command)
	for _, suffix := range []string{".exe", ".cmd", ".bat", ".ps1"} {
		command = strings.TrimSuffix(command, suffix)
	}
	return command
}

func isUUID(value string) bool {
	value = strings.TrimSpace(value)
	if len(value) != 36 {
		return false
	}
	for i, c := range value {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if c != '-' {
				return false
			}
			continue
		}
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
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
