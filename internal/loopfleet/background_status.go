package loopfleet

import (
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

const BackgroundStatusSchema = "fak.background-fleet-status.v1"

// Process is the small, platform-neutral process view consumed by BackgroundStatus.
type Process struct {
	PID       int       `json:"pid"`
	ParentPID int       `json:"parent_pid,omitempty"`
	StartedAt time.Time `json:"started_at,omitempty"`
	Command   string    `json:"command"`
}

type BackgroundLoop struct {
	ID              string `json:"id"`
	Kind            string `json:"kind"`
	Source          string `json:"source"`
	State           string `json:"state"`
	PID             int    `json:"pid,omitempty"`
	AgeSeconds      int64  `json:"age_seconds,omitempty"`
	LastSeenSeconds int64  `json:"last_seen_seconds,omitempty"`
	Detail          string `json:"detail,omitempty"`
}

type BackgroundStatus struct {
	Schema      string           `json:"schema"`
	GeneratedAt time.Time        `json:"generated_at"`
	Total       int              `json:"total"`
	Live        int              `json:"live"`
	Managed     int              `json:"managed"`
	ProcessOnly int              `json:"process_only"`
	Stale       int              `json:"stale"`
	Loops       []BackgroundLoop `json:"loops"`
}

// BuildBackgroundStatus joins the durable loop-manager ledger with live OS process
// evidence. The two sources are intentionally retained rather than forcing a lossy
// identity merge: the ledger explains managed work, while process discovery catches
// super-loops and legacy background drivers that predate loopmgr registration.
func BuildBackgroundStatus(status loopmgr.Status, processes []Process, now time.Time) BackgroundStatus {
	if now.IsZero() {
		now = time.Now().UTC()
	}
	out := BackgroundStatus{Schema: BackgroundStatusSchema, GeneratedAt: now.UTC(), Loops: make([]BackgroundLoop, 0)}
	for _, snap := range status.Loops {
		// A terminal ledger row is history, not a background loop. Reporting it here
		// made the Slack card look busy long after the process had exited.
		if snap.State != string(loopmgr.StateRunning) && snap.State != string(loopmgr.StateArmed) {
			continue
		}
		age := int64(0)
		if snap.LastEventUnixNano > 0 {
			age = max64(0, now.Unix()-time.Unix(0, snap.LastEventUnixNano).Unix())
		}
		state := string(snap.State)
		live := snap.LastKind == loopmgr.EventHeartbeat && age <= 180
		if live {
			state = "live"
			out.Live++
		} else if age > 900 {
			state = "stale"
			out.Stale++
		}
		out.Managed++
		out.Loops = append(out.Loops, BackgroundLoop{
			ID: snap.LoopID, Kind: classifyName(snap.LoopID), Source: "loopmgr", State: state,
			LastSeenSeconds: age, Detail: string(snap.LastKind),
		})
	}
	selected := make(map[int]BackgroundLoop)
	parents := make(map[int]int, len(processes))
	for _, proc := range processes {
		parents[proc.PID] = proc.ParentPID
		kind, name, ok := ClassifyBackgroundCommand(proc.Command)
		if !ok {
			continue
		}
		age := int64(0)
		if !proc.StartedAt.IsZero() {
			age = max64(0, now.Unix()-proc.StartedAt.Unix())
		}
		selected[proc.PID] = BackgroundLoop{ID: name, Kind: kind, Source: "process", State: "live", PID: proc.PID, AgeSeconds: age, Detail: compactCommand(proc.Command)}
	}
	for pid, row := range selected {
		// One logical loop commonly has a fak guard parent and a claude/codex child
		// carrying the same prompt. Keep the outer process as the durable owner.
		ancestor := parents[pid]
		duplicate := false
		for ancestor != 0 {
			if parent, ok := selected[ancestor]; ok && parent.Kind == row.Kind && parent.ID == row.ID {
				duplicate = true
				break
			}
			ancestor = parents[ancestor]
		}
		if duplicate {
			continue
		}
		out.ProcessOnly++
		out.Live++
		out.Loops = append(out.Loops, row)
	}
	/* process discovery above intentionally follows parent chains before rendering. */
	sort.SliceStable(out.Loops, func(i, j int) bool {
		if out.Loops[i].Source != out.Loops[j].Source {
			return out.Loops[i].Source < out.Loops[j].Source
		}
		if out.Loops[i].Kind != out.Loops[j].Kind {
			return out.Loops[i].Kind < out.Loops[j].Kind
		}
		return out.Loops[i].ID < out.Loops[j].ID
	})
	out.Total = len(out.Loops)
	return out
}

// ClassifyBackgroundCommand recognizes durable background drivers by behavior,
// not by Slack naming. Keep this additive: unknown processes are ignored rather than
// guessed, while newly introduced loop families can add one explicit signature.
func ClassifyBackgroundCommand(command string) (kind, name string, ok bool) {
	c := strings.ToLower(strings.TrimSpace(command))
	if c == "" {
		return "", "", false
	}
	signatures := []struct {
		needles    []string
		kind, name string
	}{
		{[]string{"meta_superloop"}, "super-loop", "meta-superloop"},
		{[]string{"super-loop"}, "super-loop", "super-loop"},
		{[]string{"superloop"}, "super-loop", "superloop"},
		{[]string{"dispatch-loop"}, "dispatch-loop", "dispatch-loop"},
		{[]string{"dispatch_loop"}, "dispatch-loop", "dispatch-loop"},
		{[]string{"fleet_resume_watchdog"}, "watchdog", "fleet-resume-watchdog"},
		{[]string{"fleet_supervisor_watchdog"}, "watchdog", "fleet-supervisor-watchdog"},
		{[]string{"fak", " loop run"}, "kernel-loop", "fak-loop-run"},
		{[]string{"fak", " loop drive"}, "kernel-loop", "fak-loop-drive"},
		{[]string{"nightrun"}, "nightrun", "nightrun"},
	}
	for _, sig := range signatures {
		match := true
		for _, needle := range sig.needles {
			if !strings.Contains(c, needle) {
				match = false
				break
			}
		}
		if match {
			return sig.kind, commandIdentity(c, sig.name), true
		}
	}
	return "", "", false
}

func commandIdentity(command, fallback string) string {
	fields := strings.Fields(strings.NewReplacer("\"", " ", "'", " ").Replace(command))
	for i := 0; i+1 < len(fields); i++ {
		if fields[i] == "--lane" {
			lane := strings.Trim(fields[i+1], "\"' ,;")
			if lane != "" {
				return fallback + "/" + lane
			}
		}
	}
	return fallback
}

func classifyName(name string) string {
	c := strings.ToLower(name)
	switch {
	case strings.Contains(c, "super"):
		return "super-loop"
	case strings.Contains(c, "dispatch"):
		return "dispatch-loop"
	case strings.Contains(c, "watchdog"):
		return "watchdog"
	case strings.Contains(c, "night"):
		return "nightrun"
	default:
		return "managed-loop"
	}
}
func compactCommand(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 180 {
		return s[:177] + "..."
	}
	return s
}
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
