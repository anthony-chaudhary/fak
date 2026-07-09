package main

// fak looporphan - the impure shell over internal/looporphan + internal/looporphanscan.
//
// It is the duplicate-loop-supervisor reaper the resume-consolidate-duplicate-loops
// pain asks for: detached `dos loop` / `fak loop` engines that outlived their
// /goal session keep looping and burn account seats. This shell gathers a live
// process census (procguard.CollectRelations), hands it to the pure keep/reap core
// (which keeps the supervisor still parenting live work and flags idle
// orphans/duplicates), and prints the plan. It kills NOTHING by default; --reap
// opts into the destructive rung, and even then re-fences every PID's start time
// immediately before the kill (looporphanscan.ConfirmReap) so a PID recycled
// between the plan and the kill is never the process that dies.
//
// The testable gather heuristics live in internal/looporphanscan (a tier-2 package
// buildable in isolation); this file is thin CLI glue.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/looporphan"
	"github.com/anthony-chaudhary/fak/internal/looporphanscan"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const looporphanUsage = `fak looporphan - reap duplicate/orphaned loop-supervisor processes

  Detects detached loop/drainer engines (dos loop / fak loop) that outlived
  their owning session and burn account seats. Keeps the one still parenting a
  live worker; flags idle orphans and duplicates. Detect-only by default.

Flags:
  --json            emit the plan as JSON instead of a table
  --reap            actually tree-kill the REAP set (default: detect only)
  --allow-unfenced  reap even supervisors lacking a start-time fence (unsafe;
                    off by default so a PID that cannot be reuse-checked is spared)`

func cmdLooporphan(argv []string) { os.Exit(runLooporphan(os.Stdout, os.Stderr, argv)) }

// protectedReapCmd are command-line substrings that must never be killed even if
// they somehow reach the REAP set - a belt-and-suspenders mirror of the janitor's
// protected set.
var protectedReapCmd = []string{"dos_mcp", "dos.mcp", "mcp.server", "mcp serve", "fak guard", "fak serve"}

func runLooporphan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("looporphan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	var (
		asJSON       = fs.Bool("json", false, "")
		doReap       = fs.Bool("reap", false, "")
		allowUnfence = fs.Bool("allow-unfenced", false, "")
	)
	if err := fs.Parse(argv); err != nil {
		fmt.Fprintln(stderr, looporphanUsage)
		return 2
	}

	procs, errDetail := procguard.CollectRelations()
	if len(procs) == 0 {
		fmt.Fprintf(stderr, "fak looporphan: no process census available (%s)\n", errDetail)
		return 1
	}

	cfg := looporphanscan.DefaultConfig()
	sv := looporphanscan.ExtractSupervisors(procs, cfg)

	coreCfg := looporphan.DefaultConfig()
	coreCfg.AllowUnfencedReap = *allowUnfence
	rep := looporphan.Plan(sv, coreCfg)

	// plannedStart maps a supervisor PID to the start-time fence the plan saw, so
	// the reap step can re-probe it.
	plannedStart := make(map[int]string, len(sv))
	for _, s := range sv {
		plannedStart[s.PID] = s.Start
	}

	var reaped []reapOutcome
	if *doReap {
		reaped = reapPlan(rep, plannedStart)
	}

	if *asJSON {
		return emitLooporphanJSON(stdout, rep, reaped, *doReap)
	}
	return emitLooporphanText(stdout, rep, reaped, *doReap)
}

type reapOutcome struct {
	PID     int    `json:"pid"`
	Lane    string `json:"lane"`
	Killed  bool   `json:"killed"`
	Skipped string `json:"skipped,omitempty"` // reason a REAP pid was spared
}

// reapPlan performs the destructive rung: for each REAP verdict, re-collect a
// fresh census, confirm the PID's start still matches the plan, refuse anything on
// the protected list, then tree-kill it.
func reapPlan(rep looporphan.Report, plannedStart map[int]string) []reapOutcome {
	fresh, _ := procguard.CollectRelations()
	freshByPID := make(map[int]procguard.Proc, len(fresh))
	for _, p := range fresh {
		freshByPID[p.PID] = p
	}
	live := func(pid int) (string, bool) {
		p, ok := freshByPID[pid]
		return p.Start, ok
	}

	var out []reapOutcome
	for _, v := range rep.Verdicts {
		if v.Action != looporphan.REAP {
			continue
		}
		oc := reapOutcome{PID: v.PID, Lane: v.Lane}
		if p, ok := freshByPID[v.PID]; ok && containsAnyCmd(p.Cmdline, protectedReapCmd) {
			oc.Skipped = "protected command"
			out = append(out, oc)
			continue
		}
		if !looporphanscan.ConfirmReap(v.PID, plannedStart[v.PID], live) {
			oc.Skipped = "start-fence changed (PID reused or vanished)"
			out = append(out, oc)
			continue
		}
		killed, _ := procguard.KillPID(v.PID)
		oc.Killed = killed
		if !killed {
			oc.Skipped = "kill failed"
		}
		out = append(out, oc)
	}
	return out
}

func containsAnyCmd(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func emitLooporphanText(stdout io.Writer, rep looporphan.Report, reaped []reapOutcome, didReap bool) int {
	fmt.Fprintf(stdout, "loop supervisors: keep=%d reap=%d collision=%d unknown=%d\n",
		rep.Keep, rep.Reap, rep.Collision, rep.Unknown)
	for _, v := range rep.Verdicts {
		lane := v.Lane
		if lane == "" {
			lane = "-"
		}
		fmt.Fprintf(stdout, "  %-9s pid=%-7d lane=%-16s %s  (%s)\n",
			v.Action, v.PID, lane, v.Reason, v.Detail)
	}
	if didReap {
		fmt.Fprintln(stdout, "reap:")
		if len(reaped) == 0 {
			fmt.Fprintln(stdout, "  (nothing to reap)")
		}
		for _, oc := range reaped {
			status := "killed"
			if !oc.Killed {
				status = "spared: " + oc.Skipped
			}
			fmt.Fprintf(stdout, "  pid=%-7d lane=%-16s %s\n", oc.PID, oc.Lane, status)
		}
	} else if rep.Reap > 0 {
		fmt.Fprintln(stdout, "(detect only; pass --reap to tree-kill the REAP set)")
	}
	return 0
}

func emitLooporphanJSON(stdout io.Writer, rep looporphan.Report, reaped []reapOutcome, didReap bool) int {
	payload := struct {
		Keep      int                  `json:"keep"`
		Reap      int                  `json:"reap"`
		Collision int                  `json:"collision"`
		Unknown   int                  `json:"unknown"`
		Verdicts  []looporphan.Verdict `json:"verdicts"`
		Reaped    []reapOutcome        `json:"reaped,omitempty"`
		DidReap   bool                 `json:"did_reap"`
	}{
		Keep: rep.Keep, Reap: rep.Reap, Collision: rep.Collision, Unknown: rep.Unknown,
		Verdicts: rep.Verdicts, Reaped: reaped, DidReap: didReap,
	}
	enc := json.NewEncoder(stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(payload); err != nil {
		return 1
	}
	return 0
}
