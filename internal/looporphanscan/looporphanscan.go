package looporphanscan

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/looporphan"
	"github.com/anthony-chaudhary/fak/internal/looprecover"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// Config holds the command-line marker vocabularies. The zero value is unusable;
// callers take DefaultConfig and adjust.
type Config struct {
	// SupervisorMarkers: a process whose command line contains any of these is a
	// candidate loop/drainer engine (the thing we may reap).
	SupervisorMarkers []string
	// WorkerMarkers: a descendant whose command line contains any of these is a
	// live leaf worker - its presence in a supervisor's subtree makes that
	// supervisor a keeper.
	WorkerMarkers []string
	// DriverMarkers: a live parent whose command line contains any of these is a
	// recognizable owning session/driver, so the supervisor's parent counts as
	// alive rather than an unrecognized (possibly PID-reused) process.
	DriverMarkers []string
	// LaneFlags: command-line flags whose following token is the loop's lane
	// identity, tried in order (e.g. --lane, --region, --goal).
	LaneFlags []string
}

// DefaultConfig returns the marker vocabulary matched to fak/dos conventions
// (see the dispatch worker and janitor protected-command lists).
func DefaultConfig() Config {
	return Config{
		SupervisorMarkers: []string{"dos loop", "fak loop drive", "fak loop region"},
		WorkerMarkers:     []string{"claude -p", "fak c "},
		DriverMarkers:     []string{"dos loop", "fak loop", "fak guard", "fak serve", "claude", "goal"},
		LaneFlags:         []string{"--lane", "--region", "--goal"},
	}
}

func containsAny(s string, subs []string) bool {
	for _, sub := range subs {
		if sub != "" && strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func derefPID(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// parseLane extracts the loop's lane identity from a command line by scanning for
// any LaneFlag and taking the following whitespace-delimited token. It also
// accepts the --flag=value form. Returns "" when no lane flag is present.
func parseLane(cmdline string, laneFlags []string) string {
	fields := strings.Fields(cmdline)
	for i, f := range fields {
		for _, flag := range laneFlags {
			if f == flag && i+1 < len(fields) {
				return fields[i+1]
			}
			if strings.HasPrefix(f, flag+"=") {
				return strings.TrimPrefix(f, flag+"=")
			}
		}
	}
	return ""
}

// ExtractSupervisors turns a raw census into the pure core's input: one
// Supervisor per process matching a supervisor marker, with its lane parsed,
// parent liveness decided, and live workers in its subtree counted.
func ExtractSupervisors(procs []procguard.Proc, cfg Config) []looporphan.Supervisor {
	byPID := make(map[int]procguard.Proc, len(procs))
	children := make(map[int][]int, len(procs))
	for _, p := range procs {
		byPID[p.PID] = p
		if ppid := derefPID(p.PPID); ppid > 0 {
			children[ppid] = append(children[ppid], p.PID)
		}
	}

	var out []looporphan.Supervisor
	for _, p := range procs {
		if !containsAny(p.Cmdline, cfg.SupervisorMarkers) {
			continue
		}
		out = append(out, looporphan.Supervisor{
			PID:         p.PID,
			PPID:        derefPID(p.PPID),
			Start:       p.Start,
			Cmdline:     p.Cmdline,
			Lane:        parseLane(p.Cmdline, cfg.LaneFlags),
			Parent:      parentState(p, byPID, cfg),
			LiveWorkers: countSubtreeWorkers(p.PID, byPID, children, cfg),
		})
	}
	return out
}

// parentState decides a supervisor's parent liveness from census membership plus
// a plausibility check. A missing parent is a confirmed orphan; a present but
// unrecognizable parent is Unknown (it may be a reused PID) so the core fails
// closed rather than treat it as either alive or orphaned.
func parentState(sup procguard.Proc, byPID map[int]procguard.Proc, cfg Config) looporphan.ParentState {
	ppid := derefPID(sup.PPID)
	if ppid <= 0 {
		return looporphan.ParentUnknown
	}
	parent, ok := byPID[ppid]
	if !ok {
		return looporphan.ParentDead
	}
	if containsAny(parent.Cmdline, cfg.DriverMarkers) {
		return looporphan.ParentAlive
	}
	return looporphan.ParentUnknown
}

// countSubtreeWorkers counts live leaf workers in a supervisor's process subtree.
// Every process in the census is by definition live, so a matching descendant is
// a live worker. A visited set guards against a malformed PPID cycle.
func countSubtreeWorkers(root int, byPID map[int]procguard.Proc, children map[int][]int, cfg Config) int {
	visited := map[int]bool{root: true}
	stack := append([]int(nil), children[root]...)
	n := 0
	for len(stack) > 0 {
		pid := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if visited[pid] {
			continue
		}
		visited[pid] = true
		if p, ok := byPID[pid]; ok && containsAny(p.Cmdline, cfg.WorkerMarkers) {
			n++
		}
		stack = append(stack, children[pid]...)
	}
	return n
}

// Scan is the whole gather+fold: extract supervisors from the census and run the
// pure core over them.
func Scan(procs []procguard.Proc, cfg Config) looporphan.Report {
	return looporphan.Plan(ExtractSupervisors(procs, cfg), looporphan.DefaultConfig())
}

// ConfirmReap is the kill-time fence: it re-reads a PID's current start via the
// injected Liveness and reports whether it is still the same process the plan
// targeted (plannedStart). A recycled or vanished PID returns false, so the
// caller's kill never lands on a different process than was planned.
func ConfirmReap(pid int, plannedStart string, live looprecover.Liveness) bool {
	return looprecover.Probe(pid, plannedStart, live) == looprecover.ProbeAlive
}
