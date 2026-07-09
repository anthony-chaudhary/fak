// loop_reap.go wires `fak loop reap` - the impure shell over the pure
// internal/looporphan core. It scans the live process relation table, matches
// detached loop/drainer SUPERVISOR engines by command-line marker, folds each
// into a looporphan.Supervisor (lane identity + start-time fence + parent
// liveness + live-worker count), and lets looporphan.Plan decide keep/reap. The
// core never kills; this shell tree-kills only the supervisors Plan recommends
// REAP, and only behind an explicit --reap opt-in.
//
// Safety starts in the pure core: it refuses to reap a supervisor that parents
// live work, fails closed (UNKNOWN) on a missing start-time fence or thin identity,
// and flags same-lane collisions for an operator instead of guessing. This shell
// then adds two belt-and-suspenders guards at the kill site: (1) it never passes
// its own PID (or its parent's) to the killer, and (2) it re-collects a fresh
// process snapshot and re-verifies each REAP pid's start-time identity immediately
// before killing, refusing any pid that is now fence-less, reused, or vanished - so
// a PID recycled between the plan and the kill is never the process that dies.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"text/tabwriter"

	"github.com/anthony-chaudhary/fak/internal/looporphan"
	"github.com/anthony-chaudhary/fak/internal/looprecover"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// --- injectable seams (overridden in tests so no live fleet is touched) --- //

// loopReapCollectRelations returns the process relations snapshot (PPID/cmdline/
// start). Production uses procguard.CollectRelations; tests inject a fixture.
var loopReapCollectRelations = procguard.CollectRelations

// loopReapKill is the destructive tree reaper (native tree kill on Windows,
// process-group/descendant SIGKILL on POSIX). Tests inject a recorder so nothing
// is killed. Mirrors fleetKillPID (fleet.go) and loopTreeKill (loop.go).
var loopReapKill = procguard.KillPID

// --- markers ------------------------------------------------------------- //

// defaultLoopSupervisorMarkers are the cmdline substrings that identify a
// detached loop/drainer supervisor - the long-running engines that re-spawn agent
// workers. Extend at the call site with --supervisor-marker.
var defaultLoopSupervisorMarkers = []string{"loop drive", "dos loop", "superloop drive"}

// defaultLoopWorkerMarkers are the cmdline substrings that identify a live agent
// worker inside a supervisor's subtree. Their presence in the subtree is what
// makes the core KEEP a supervisor. Extend with --worker-marker.
var defaultLoopWorkerMarkers = []string{"claude -p", "fak c "}

// loopLaneFlags are the flags a supervisor's cmdline carries its loop identity in,
// tried in order. The first that resolves becomes the lane the core groups on.
var loopLaneFlags = []string{"--lane", "--loop", "--goal", "--region"}

// --- JSON contract ------------------------------------------------------- //

type loopReapRow struct {
	PID       int    `json:"pid"`
	Lane      string `json:"lane"`
	Group     string `json:"group"`
	GroupSize int    `json:"group_size"`
	Action    string `json:"action"`
	Reason    string `json:"reason"`
	Detail    string `json:"detail"`
}

type loopReapKilled struct {
	PID    int    `json:"pid"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail"`
}

type loopReapReport struct {
	Schema      string           `json:"schema"`
	Supervisors int              `json:"supervisors"`
	Keep        int              `json:"keep"`
	Reap        int              `json:"reap"`
	Collision   int              `json:"collision"`
	Unknown     int              `json:"unknown"`
	Enacted     bool             `json:"enacted"`
	Verdicts    []loopReapRow    `json:"verdicts"`
	Killed      []loopReapKilled `json:"killed,omitempty"`
}

// --- census collector ---------------------------------------------------- //

// loopReapCensus folds a process relation snapshot into the supervisor census the
// pure core consumes. It matches supervisors by cmdline marker and, for each,
// resolves its lane identity, carries its start-time fence, probes its parent's
// liveness against the same snapshot, and counts the live workers in its subtree.
// All I/O (the scan) happens in the caller; this transform is pure and testable.
func loopReapCensus(procs []procguard.Proc, supMarkers, workerMarkers []string) []looporphan.Supervisor {
	byPID := make(map[int]procguard.Proc, len(procs))
	children := map[int][]int{}
	for _, p := range procs {
		if p.PID <= 0 {
			continue
		}
		byPID[p.PID] = p
		if p.PPID != nil {
			children[*p.PPID] = append(children[*p.PPID], p.PID)
		}
	}

	// Liveness closure over the snapshot: a pid is "running" iff a row holds it,
	// and carries that row's start-time identity so Probe can defeat pid reuse.
	live := looprecover.Liveness(func(pid int) (string, bool) {
		if p, ok := byPID[pid]; ok {
			return p.Start, true
		}
		return "", false
	})

	var census []looporphan.Supervisor
	for _, p := range procs {
		if p.PID <= 0 || !procMatchesLoopMarker(p, supMarkers) {
			continue
		}
		ppid := 0
		if p.PPID != nil {
			ppid = *p.PPID
		}
		census = append(census, looporphan.Supervisor{
			PID:         p.PID,
			PPID:        ppid,
			Start:       p.Start,
			Cmdline:     strings.TrimSpace(p.Cmdline),
			Lane:        parseLoopLane(p.Cmdline),
			Parent:      loopParentState(ppid, byPID, supMarkers, live),
			LiveWorkers: countLoopWorkers(p.PID, children, byPID, workerMarkers),
		})
	}
	return census
}

// procMatchesLoopMarker reports whether a process's name+cmdline contains any
// marker (case-insensitive substring).
func procMatchesLoopMarker(p procguard.Proc, markers []string) bool {
	hay := strings.ToLower(p.Name + " " + p.Cmdline)
	for _, m := range markers {
		if m = strings.ToLower(strings.TrimSpace(m)); m != "" && strings.Contains(hay, m) {
			return true
		}
	}
	return false
}

// parseLoopLane extracts the loop identity from a supervisor's cmdline: the first
// of --lane/--loop/--goal/--region that resolves (space- or =-separated). "" when
// none is present - the core then groups on the raw cmdline instead.
func parseLoopLane(cmdline string) string {
	fields := strings.Fields(cmdline)
	for i, f := range fields {
		for _, name := range loopLaneFlags {
			if f == name && i+1 < len(fields) {
				// Guard the space form against a missing value: "--lane --json" must
				// not yield the lane "--json". A value that is itself a flag means the
				// lane was omitted here (a later flag may still supply it).
				if v := fields[i+1]; !strings.HasPrefix(v, "-") {
					return strings.TrimSpace(v)
				}
			}
			if strings.HasPrefix(f, name+"=") {
				return strings.TrimSpace(strings.TrimPrefix(f, name+"="))
			}
		}
	}
	return ""
}

// loopParentState maps a parent-liveness probe onto the core's tri-state. It
// probes against the live snapshot with an empty recorded start, so it errs
// toward ParentAlive (KEEP) when a pid is merely reused - never toward a wrong
// reap. On POSIX an orphan reparents to init (pid 1); a supervisor whose parent
// is now init has lost its owning session and is treated as orphaned.
func loopParentState(ppid int, byPID map[int]procguard.Proc, supMarkers []string, live looprecover.Liveness) looporphan.ParentState {
	if ppid <= 0 {
		return looporphan.ParentUnknown
	}
	// A present parent that is itself a recognized loop supervisor is a live owner -
	// checked BEFORE the pid-1 rule so a container entrypoint that drives loops AS
	// pid 1 is not mistaken for orphaned-to-init. This exception can only KEEP a loop
	// the blanket rule would have reaped; it never makes anything more reapable.
	if p, ok := byPID[ppid]; ok && procMatchesLoopMarker(p, supMarkers) {
		return looporphan.ParentAlive
	}
	if runtime.GOOS != "windows" && ppid == 1 {
		// Reparented to real init (pid 1, and NOT one of our supervisors): the owning
		// session/driver exited and the OS re-homed this loop under init - the
		// canonical POSIX orphan signal.
		return looporphan.ParentDead
	}
	switch looprecover.Probe(ppid, "", live) {
	case looprecover.ProbeAlive:
		return looporphan.ParentAlive
	case looprecover.ProbeDead:
		return looporphan.ParentDead
	default:
		return looporphan.ParentUnknown
	}
}

// countLoopWorkers counts the live worker processes in a supervisor's subtree
// (transitive descendants matching a worker marker). A breadth-first walk over
// the child index; guards against cycles via a visited set.
func countLoopWorkers(root int, children map[int][]int, byPID map[int]procguard.Proc, markers []string) int {
	seen := map[int]bool{root: true}
	queue := append([]int{}, children[root]...)
	n := 0
	for len(queue) > 0 {
		pid := queue[0]
		queue = queue[1:]
		if seen[pid] {
			continue
		}
		seen[pid] = true
		if p, ok := byPID[pid]; ok && procMatchesLoopMarker(p, markers) {
			n++
		}
		queue = append(queue, children[pid]...)
	}
	return n
}

// --- shell --------------------------------------------------------------- //

func runLoopReap(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("loop reap", flag.ContinueOnError)
	fs.SetOutput(stderr)
	doReap := fs.Bool("reap", false, "DESTRUCTIVE: tree-kill the supervisors the report recommends REAP (default: report only)")
	asJSON := fs.Bool("json", false, "emit the machine-readable JSON report")
	allowUnfenced := fs.Bool("allow-unfenced", false, "surface a fence-less supervisor as REAP in the report (off: fail closed to UNKNOWN). The kill-time fence still refuses a fence-less kill, so this affects the report, not what dies.")
	var supMarkers multiFlag
	var workerMarkers multiFlag
	fs.Var(&supMarkers, "supervisor-marker", "extra cmdline substring identifying a loop supervisor (repeatable)")
	fs.Var(&workerMarkers, "worker-marker", "extra cmdline substring identifying a live worker (repeatable)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak loop reap: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	procs, collectErr := loopReapCollectRelations()
	if collectErr != "" {
		// A failed scan is not a clean host; refuse to conclude "nothing to reap".
		fmt.Fprintf(stderr, "fak loop reap: process scan failed: %s\n", collectErr)
		return 1
	}

	sup := append(append([]string{}, defaultLoopSupervisorMarkers...), supMarkers...)
	wrk := append(append([]string{}, defaultLoopWorkerMarkers...), workerMarkers...)
	census := loopReapCensus(procs, sup, wrk)

	cfg := looporphan.DefaultConfig()
	cfg.AllowUnfencedReap = *allowUnfenced
	rep := looporphan.Plan(census, cfg)

	// Belt-and-suspenders: never let the reaper kill itself or its own parent,
	// regardless of what a marker matched.
	protected := map[int]bool{os.Getpid(): true, os.Getppid(): true}

	var killed []loopReapKilled
	if *doReap {
		// Kill-time fence: re-collect a FRESH snapshot and re-verify each REAP pid's
		// start-time identity immediately before killing it. A pid that exited (and
		// was possibly reused by an unrelated process) between the plan and now fails
		// the re-probe and is refused; a pid carrying no start-time fence at all
		// cannot be identity-checked and is refused outright - so even --allow-unfenced
		// (which lets the core surface a REAP for a fence-less supervisor in the
		// report) can never land an unverifiable kill. This is the belt-and-suspenders
		// backstop over the core's own fence: the plan proposes, the fence disposes.
		plannedStart := make(map[int]string, len(census))
		for _, s := range census {
			plannedStart[s.PID] = s.Start
		}
		fresh, rescanErr := loopReapCollectRelations()
		freshByPID := make(map[int]procguard.Proc, len(fresh))
		for _, p := range fresh {
			freshByPID[p.PID] = p
		}
		freshLive := looprecover.Liveness(func(pid int) (string, bool) {
			if p, ok := freshByPID[pid]; ok {
				return p.Start, true
			}
			return "", false
		})
		for _, pid := range rep.ReapPIDs() {
			if protected[pid] {
				continue // never kill the reaper itself or its parent
			}
			if rescanErr != "" {
				// A failed kill-time rescan means we cannot re-verify identity; a stale
				// plan must not authorize a blind kill.
				killed = append(killed, loopReapKilled{PID: pid, OK: false,
					Detail: "kill-time rescan failed; refusing kill: " + rescanErr})
				continue
			}
			if strings.TrimSpace(plannedStart[pid]) == "" {
				killed = append(killed, loopReapKilled{PID: pid, OK: false,
					Detail: "no start-time fence; refusing kill (cannot prove PID identity)"})
				continue
			}
			if looprecover.Probe(pid, plannedStart[pid], freshLive) != looprecover.ProbeAlive {
				killed = append(killed, loopReapKilled{PID: pid, OK: false,
					Detail: "start-fence changed (PID reused or vanished); refusing kill"})
				continue
			}
			ok, detail := loopReapKill(pid)
			killed = append(killed, loopReapKilled{PID: pid, OK: ok, Detail: detail})
		}
	}

	if *asJSON {
		report := loopReapReport{
			Schema:      "fak-loop-reap/1",
			Supervisors: len(census),
			Keep:        rep.Keep,
			Reap:        rep.Reap,
			Collision:   rep.Collision,
			Unknown:     rep.Unknown,
			Enacted:     *doReap,
			Verdicts:    loopReapRows(rep),
			Killed:      killed,
		}
		return encodeJSONOrFail(stdout, stderr, report, "fak loop reap")
	}

	renderLoopReap(stdout, rep, len(census), *doReap, killed)

	// Report mode with pending reap work exits 3 (actionable, not enacted) so a
	// scheduler can gate on "is the loop fleet clean?" without parsing output.
	if !*doReap && rep.Reap > 0 {
		return 3
	}
	return 0
}

func loopReapRows(rep looporphan.Report) []loopReapRow {
	rows := make([]loopReapRow, 0, len(rep.Verdicts))
	for _, v := range rep.Verdicts {
		rows = append(rows, loopReapRow{
			PID: v.PID, Lane: v.Lane, Group: v.Group, GroupSize: v.GroupSize,
			Action: string(v.Action), Reason: v.Reason, Detail: v.Detail,
		})
	}
	return rows
}

func renderLoopReap(w io.Writer, rep looporphan.Report, supervisors int, enacted bool, killed []loopReapKilled) {
	if supervisors == 0 {
		fmt.Fprintln(w, "fak loop reap: no loop/drainer supervisors found - nothing to reap")
		return
	}
	banner := ""
	if enacted {
		banner = " (ENACTED)"
	}
	fmt.Fprintf(w, "fak loop reap: supervisors=%d keep=%d reap=%d collision=%d unknown=%d%s\n\n",
		supervisors, rep.Keep, rep.Reap, rep.Collision, rep.Unknown, banner)

	tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "PID\tLANE\tACTION\tREASON\tDETAIL")
	for _, v := range rep.Verdicts {
		lane := v.Lane
		if strings.TrimSpace(lane) == "" {
			lane = "-"
		}
		fmt.Fprintf(tw, "%d\t%s\t%s\t%s\t%s\n", v.PID, lane, v.Action, v.Reason, v.Detail)
	}
	_ = tw.Flush()

	if enacted && len(killed) > 0 {
		fmt.Fprintln(w)
		for _, k := range killed {
			status := "killed"
			if !k.OK {
				status = "kill-failed"
			}
			fmt.Fprintf(w, "  %s pid=%d %s\n", status, k.PID, k.Detail)
		}
	}
	if !enacted && rep.Reap > 0 {
		fmt.Fprintf(w, "\nnext: re-run with --reap to tree-kill the %d REAP supervisor(s)\n", rep.Reap)
	}
}
