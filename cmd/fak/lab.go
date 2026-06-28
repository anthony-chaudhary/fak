package main

// `fak lab` — the fast local front door to lab-fleet status. It answers the one
// question that used to require hand-running a Python Slack bridge per control
// channel: "which lab nodes are alive right now?"
//
// It is a thin CLI over the public, transport-agnostic internal/fleet core (the same
// roster + fold + render + readiness score `fleetctl` uses), wired with defaults that
// remove the friction: a GENERIC default roster is embedded so the fleet is known
// with zero setup, and the reports dir defaults to the documented drop path the
// private bridge writes into. When no live reports exist yet, `status` degrades
// HONESTLY — every box reads `unknown` with a one-line hint to start the bridge or
// self-report — never a misleading score-0 "outage".
//
//	fak lab status [--roster F] [--reports DIR] [--group G] [--class C] [--all] [--json]
//	fak lab report --id ID --state live|idle|draining|down [--version V] [--note N] [--reports DIR]
//	fak lab ls     [--roster F] [--group G] [--class C] [--json]
//
// THE PUBLIC/PRIVATE BOUNDARY IS A DATA CONTRACT. The embedded roster is generic (an
// id, a class, a group — never a real lab host, channel, or token); the private Slack
// bridge owns the id->channel map and the live `!sessions` liveness probe on its side
// and writes one fak.fleet.report/v1 file per box. `fak lab report` is the PUBLIC
// producer half: a box self-reports its state with no private bridge, closing the
// loop for that box today. See docs/dgx-slack-boundary.md and docs/fak/lab-dev-loop.md.

import (
	"bytes"
	_ "embed"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/fleet"
)

// labDefaultRosterJSON is the generic lab roster embedded at compile time. go:embed
// cannot traverse parent dirs, so the file is a sibling here (the same pattern
// guard.go uses for its default policy). Keep it GENERIC — see lab-roster.json.
//
//go:embed lab-roster.json
var labDefaultRosterJSON []byte

func cmdLab(argv []string) { os.Exit(runLab(os.Stdout, os.Stderr, argv)) }

func runLab(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak lab <status|report|ls> [flags]")
		fmt.Fprintln(stderr, "       fak lab status            # which lab nodes are alive right now?")
		fmt.Fprintln(stderr, "       fak lab report --id ID --state live   # self-report this box")
		return 2
	}
	switch argv[0] {
	case "status":
		return labStatus(stdout, stderr, argv[1:])
	case "report":
		return labReport(stdout, stderr, argv[1:])
	case "ls", "list":
		return labLs(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, "usage: fak lab <status|report|ls> [flags]")
		fmt.Fprintln(stdout, "  status   fold the lab roster against the reports dir and render the fleet view")
		fmt.Fprintln(stdout, "  report   write one fak.fleet.report/v1 line for a box (self-report; no bridge)")
		fmt.Fprintln(stdout, "  ls       list the boxes in the (default or --roster) lab roster")
		return 0
	default:
		fmt.Fprintf(stderr, "fak lab: unknown subcommand %q (want status|report|ls)\n", argv[0])
		return 2
	}
}

// labReportsDir resolves the reports directory in the order that makes the common
// case zero-flag: an explicit --reports wins, then $FAK_FLEET_REPORTS, then the
// documented default the private bridge writes into (~/.config/fak/fleet/reports,
// %APPDATA%\fak\fleet\reports on Windows). It reuses nodeConfigDir() so the lab and
// node tooling agree on the config root.
func labReportsDir(flagVal string) (string, error) {
	if flagVal != "" {
		return flagVal, nil
	}
	if env := os.Getenv("FAK_FLEET_REPORTS"); env != "" {
		return env, nil
	}
	cfgDir, err := nodeConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(cfgDir, "fleet", "reports"), nil
}

// labLoadRoster loads the roster from --roster, or the embedded generic default when
// no flag is given, and validates it (fail-loud on a bad roster).
func labLoadRoster(stderr io.Writer, rosterPath string) (fleet.Roster, bool) {
	var (
		ro  fleet.Roster
		err error
	)
	if rosterPath != "" {
		ro, err = fleet.LoadRosterFile(rosterPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak lab: %v\n", err)
			return fleet.Roster{}, false
		}
	} else {
		ro, err = fleet.LoadRoster(bytes.NewReader(labDefaultRosterJSON))
		if err != nil {
			fmt.Fprintf(stderr, "fak lab: embedded roster is corrupt: %v\n", err)
			return fleet.Roster{}, false
		}
	}
	if probs := ro.Validate(); len(probs) > 0 {
		fmt.Fprintf(stderr, "fak lab: invalid roster (%d problem(s)):\n", len(probs))
		for _, p := range probs {
			fmt.Fprintf(stderr, "  - %s\n", p)
		}
		return fleet.Roster{}, false
	}
	return ro, true
}

func labStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("lab status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rosterPath := fs.String("roster", "", "roster file (default: the embedded generic lab roster)")
	reports := fs.String("reports", "", "reports dir (default: $FAK_FLEET_REPORTS or ~/.config/fak/fleet/reports)")
	group := fs.String("group", "", "only boxes in this group")
	class := fs.String("class", "", "only boxes of this class")
	all := fs.Bool("all", false, "also print a per-box table")
	asJSON := fs.Bool("json", false, "emit the folded snapshot as JSON")
	staleMin := fs.Float64("stale-min", fleet.DefaultStaleSec/60.0, "minutes of silence before a box is flagged stale")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	ro, ok := labLoadRoster(stderr, *rosterPath)
	if !ok {
		return 2
	}
	ro = labSelect(ro, *group, *class)
	if (*group != "" || *class != "") && len(ro.Boxes) == 0 {
		fmt.Fprintln(stderr, "fak lab: no boxes match the --group/--class filter")
	}

	dir, err := labReportsDir(*reports)
	if err != nil {
		fmt.Fprintf(stderr, "fak lab: %v\n", err)
		return 2
	}

	// Honest degrade: a missing or empty reports dir is NOT an outage — it is the
	// common "no live reports yet" state. Fold against whatever is there (every box
	// without a file reads `unknown`), and tell the operator how to populate it
	// rather than letting a silent score 0 read like a fleet-wide emergency.
	live := labReportsPopulated(dir)
	var reps []fleet.Report
	if live {
		reps = fleet.ReadReports(dir, ro)
	}
	snap := fleet.Fold(ro, reps, fleet.FoldOpts{StaleSec: *staleMin * 60})

	if *asJSON {
		if err := writeIndentedJSON(stdout, snap); err != nil {
			fmt.Fprintf(stderr, "fak lab: encode: %v\n", err)
			return 1
		}
		return 0
	}

	fmt.Fprintln(stdout, fleet.Render(snap, *all, 72))
	if !live {
		fmt.Fprintf(stdout, "\nno live reports in %s\n", dir)
		fmt.Fprintln(stdout, "  every box reads `unknown` (not down). Populate liveness with either:")
		fmt.Fprintln(stdout, "    - the private Slack bridge (writes one report per lab box), or")
		fmt.Fprintln(stdout, "    - `fak lab report --id <box> --state live` on a box that can self-report.")
	}
	return 0
}

func labReport(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("lab report", flag.ContinueOnError)
	fs.SetOutput(stderr)
	id := fs.String("id", "", "box id to report for (must match a roster id)")
	state := fs.String("state", "", "operational state: live|idle|draining|down")
	version := fs.String("version", "", "version string the box is running (optional)")
	note := fs.String("note", "", "short generic note (optional; NO host/IP/channel/token)")
	reports := fs.String("reports", "", "reports dir (default: $FAK_FLEET_REPORTS or ~/.config/fak/fleet/reports)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *id == "" || *state == "" {
		fmt.Fprintln(stderr, "usage: fak lab report --id ID --state live|idle|draining|down [--version V] [--note N]")
		return 2
	}
	st := fleet.State(*state)
	if !st.Known() || st == fleet.StateUnknown {
		fmt.Fprintf(stderr, "fak lab report: --state %q must be one of live|idle|draining|down\n", *state)
		return 2
	}

	dir, err := labReportsDir(*reports)
	if err != nil {
		fmt.Fprintf(stderr, "fak lab report: %v\n", err)
		return 1
	}
	if err := fleet.WriteReport(dir, *id, fleet.Report{State: st, Version: *version, Note: *note}); err != nil {
		fmt.Fprintf(stderr, "fak lab report: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "[fak lab] wrote %s state=%s -> %s\n", *id, st, filepath.Join(dir, *id+".json"))
	return 0
}

func labLs(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("lab ls", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rosterPath := fs.String("roster", "", "roster file (default: the embedded generic lab roster)")
	group := fs.String("group", "", "only boxes in this group")
	class := fs.String("class", "", "only boxes of this class")
	asJSON := fs.Bool("json", false, "emit the roster as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	ro, ok := labLoadRoster(stderr, *rosterPath)
	if !ok {
		return 2
	}
	ro = labSelect(ro, *group, *class)
	if *asJSON {
		if err := writeIndentedJSON(stdout, ro); err != nil {
			fmt.Fprintf(stderr, "fak lab: encode: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "%-10s %-10s %-10s %s\n", "ID", "CLASS", "GROUP", "ENDPOINT")
	for _, b := range ro.Boxes {
		fmt.Fprintf(stdout, "%-10s %-10s %-10s %s\n", b.ID, fleet.Dash(b.Class), fleet.Dash(b.Group), fleet.Dash(b.Ref()))
	}
	fmt.Fprintf(stdout, "\n%d box(es)\n", len(ro.Boxes))
	return 0
}

// labSelect filters a roster by group and/or class, preserving roster order. An empty
// filter matches all. Mirrors fleetctl's selectBoxes so the two front doors agree.
func labSelect(ro fleet.Roster, group, class string) fleet.Roster {
	if group == "" && class == "" {
		return ro
	}
	out := fleet.Roster{Schema: ro.Schema}
	for _, b := range ro.Boxes {
		if group != "" && b.Group != group {
			continue
		}
		if class != "" && b.Class != class {
			continue
		}
		out.Boxes = append(out.Boxes, b)
	}
	return out
}

// labReportsPopulated reports whether the reports dir exists and holds at least one
// *.json file — the signal that distinguishes "no live reports yet" (honest unknown)
// from a dir the bridge is actively writing. A missing dir is simply not populated.
func labReportsPopulated(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
			return true
		}
	}
	return false
}
