// Command fleetctl is the public, transport-agnostic control surface for a fleet of
// boxes — GPU servers, worker nodes — an operator drives over the private Slack
// control-bridge.
//
// WHY IT EXISTS. The dev-ex of operating the box fleet was ~two dozen scattered
// tools/fleet_*.py scripts: no single entry point, no typed roster, no view that
// stays readable past a handful of boxes. fleetctl is the Go home those helpers port
// into — one binary, a typed roster, a deterministic fold + readiness score, and a
// render that summarizes cleanly at 100+ boxes. It is Go-only and depends on nothing
// outside the standard library.
//
// THE PUBLIC/PRIVATE BOUNDARY IS A DATA CONTRACT, NOT A CODE IMPORT. The live control
// plane — the Slack control-bridge to the lab boxes — is private (it speaks a lab
// protocol and carries lab identifiers, so it lives in fak-private; see
// docs/dgx-slack-boundary.md and docs/private-comms-channel.md). The seam between it
// and this public tool is the per-box REPORT JSON (fak.fleet.report/v1, report.go):
// the private bridge emits one report file per box from live state; fleetctl reads,
// folds, renders, and scores them. Neither side imports the other, and everything
// here is generic — a box id, a class, a state word, a version, an age — never a
// host, a channel, or a token.
//
// COMMANDS.
//
//	fleetctl template --count 100 --class a100x8 > roster.json   # scaffold a 100-box roster
//	fleetctl validate --roster roster.json                       # lint it (fail-loud)
//	fleetctl ls       --roster roster.json [--group G] [--class C] [--json]
//	fleetctl status   --roster roster.json --reports DIR [--group G] [--class C] [--json] [--all]
//	fleetctl score    --roster roster.json --reports DIR [--min N] [--group G] [--class C]
//
// HONESTY. This is the public CORE: roster + fold + render + score + the file
// transport that reads reports off disk. It does NOT reach a live box — producing the
// reports is the private bridge's job. Pointed at a reports directory the bridge wrote
// (or a fixture) it is fully exercised; pointed at no reports it honestly shows every
// box as unreachable.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}
	switch os.Args[1] {
	case "template":
		cmdTemplate(os.Args[2:])
	case "validate":
		cmdValidate(os.Args[2:])
	case "ls", "list":
		cmdLs(os.Args[2:])
	case "status":
		cmdStatus(os.Args[2:])
	case "score":
		cmdScore(os.Args[2:])
	case "-h", "--help", "help":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "fleetctl: unknown subcommand %q\n", os.Args[1])
		usage(os.Stderr)
		os.Exit(2)
	}
}

func usage(w io.Writer) {
	fmt.Fprint(w, `usage: fleetctl <subcommand> [flags]

The public, transport-agnostic control surface for a fleet of boxes. The live Slack
control-bridge (private) writes one report JSON per box; fleetctl folds, renders, and
scores them. See docs/fleet.md.

subcommands:
  template   scaffold a roster of N boxes to stdout (the "add up to 100 boxes" verb)
  validate   load a roster file and fail loud on any problem
  ls         list the boxes in a roster
  status     fold the roster against a reports directory and render the fleet view
  score      print just the 0-100 readiness score (optionally gate on --min)

common flags:
  --roster FILE    the roster file (default roster.json)
  --reports DIR    directory of per-box report JSON files (the private bridge's output)
  --group G        operate only on boxes in group G (ls/status/score)
  --class C        operate only on boxes of class C (ls/status/score)
  --stale-min N    minutes of silence before a box is flagged stale (status/score; default 15)

exit codes: 0 ok · 1 the score --min gate fired · 2 a usage/roster/reports error.
run "fleetctl <subcommand> -h" for that subcommand's flags.
`)
}

func cmdTemplate(args []string) {
	fs := flag.NewFlagSet("template", flag.ExitOnError)
	count := fs.Int("count", 10, "number of boxes to scaffold")
	class := fs.String("class", "a100x8", "hardware/role class for every box")
	group := fs.String("group", "", "logical group for every box")
	prefix := fs.String("prefix", "box", "id prefix (ids are <prefix>-NNN)")
	_ = fs.Parse(args)

	if *count < 1 || *count > MaxBoxes {
		fmt.Fprintf(os.Stderr, "fleetctl template: --count must be in [1,%d]\n", MaxBoxes)
		os.Exit(2)
	}
	emitJSON(Template(*count, *class, *group, *prefix))
}

func cmdValidate(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	roster := fs.String("roster", "roster.json", "roster file to validate")
	_ = fs.Parse(args)

	ro, err := LoadRosterFile(*roster)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fleetctl validate: %v\n", err)
		os.Exit(2)
	}
	if probs := ro.Validate(); len(probs) > 0 {
		fmt.Fprintf(os.Stderr, "fleetctl validate: %d problem(s) in %s:\n", len(probs), *roster)
		for _, p := range probs {
			fmt.Fprintf(os.Stderr, "  - %s\n", p)
		}
		os.Exit(2)
	}
	fmt.Printf("ok: %d box(es), schema %s\n", len(ro.Boxes), firstNonEmpty(ro.Schema, RosterSchema))
}

func cmdLs(args []string) {
	fs := flag.NewFlagSet("ls", flag.ExitOnError)
	roster := fs.String("roster", "roster.json", "roster file")
	group := fs.String("group", "", "only boxes in this group")
	class := fs.String("class", "", "only boxes of this class")
	asJSON := fs.Bool("json", false, "emit the roster as JSON")
	_ = fs.Parse(args)

	ro := selectBoxes(mustLoad(*roster), *group, *class)
	if *asJSON {
		emitJSON(ro)
		return
	}
	fmt.Printf("%-18s %-12s %-10s %s\n", "ID", "CLASS", "GROUP", "ENDPOINT")
	for _, b := range ro.Boxes {
		fmt.Printf("%-18s %-12s %-10s %s\n", b.ID, dash(b.Class), dash(b.Group), dash(b.ref()))
	}
	fmt.Printf("\n%d box(es)\n", len(ro.Boxes))
}

func cmdStatus(args []string) {
	fs := flag.NewFlagSet("status", flag.ExitOnError)
	roster := fs.String("roster", "roster.json", "roster file")
	reports := fs.String("reports", "", "directory of per-box report JSON files")
	group := fs.String("group", "", "only boxes in this group")
	class := fs.String("class", "", "only boxes of this class")
	asJSON := fs.Bool("json", false, "emit the folded snapshot as JSON")
	all := fs.Bool("all", false, "also print a per-box table (default: summary only)")
	staleMin := fs.Float64("stale-min", DefaultStaleSec/60.0, "minutes of silence before a box is flagged stale")
	_ = fs.Parse(args)

	snap := foldFor(*roster, *reports, *group, *class, *staleMin)
	if *asJSON {
		emitJSON(snap)
		return
	}
	fmt.Println(Render(snap, *all, 72))
}

func cmdScore(args []string) {
	fs := flag.NewFlagSet("score", flag.ExitOnError)
	roster := fs.String("roster", "roster.json", "roster file")
	reports := fs.String("reports", "", "directory of per-box report JSON files")
	group := fs.String("group", "", "only boxes in this group")
	class := fs.String("class", "", "only boxes of this class")
	staleMin := fs.Float64("stale-min", DefaultStaleSec/60.0, "minutes of silence before a box is flagged stale")
	min := fs.Int("min", -1, "if set, exit non-zero when the score is below this floor")
	_ = fs.Parse(args)

	snap := foldFor(*roster, *reports, *group, *class, *staleMin)
	fmt.Println(snap.Score)
	if *min >= 0 && snap.Score < *min {
		fmt.Fprintf(os.Stderr, "fleetctl score: %d is below the --min floor %d\n", snap.Score, *min)
		os.Exit(1)
	}
}

// foldFor loads + validates a roster, applies the group/class selector, gathers the
// reports via the file transport, and folds them. It fails loud on the two config
// errors a watchdog must NOT mistake for a real outage: a bad roster (exit 2, via
// mustLoad) and a --reports path that is missing or not a directory (exit 2). When no
// --reports is given every box reads as unreachable BY DESIGN, so it says so on stderr
// rather than letting a silent score 0 look like a fleet-wide emergency.
func foldFor(rosterPath, reportsDir, group, class string, staleMin float64) Snapshot {
	ro := selectBoxes(mustLoad(rosterPath), group, class)
	if (group != "" || class != "") && len(ro.Boxes) == 0 {
		fmt.Fprintln(os.Stderr, "fleetctl: no boxes match the --group/--class filter")
	}
	var reps []Report
	if reportsDir != "" {
		mustReportsDir(reportsDir)
		reps = ReadReports(reportsDir, ro)
	} else {
		fmt.Fprintln(os.Stderr, "fleetctl: no --reports given; every box reads as unreachable (score 0)")
	}
	return Fold(ro, reps, FoldOpts{StaleSec: staleMin * 60})
}

// selectBoxes returns the boxes matching the group and/or class filter (each empty
// filter matches all), preserving roster order. The roster schema rides along so a
// filtered roster still round-trips through `ls --json`.
func selectBoxes(ro Roster, group, class string) Roster {
	if group == "" && class == "" {
		return ro
	}
	out := Roster{Schema: ro.Schema}
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

// mustReportsDir fails loud (exit 2) when --reports names a path that does not exist
// or is not a directory, so a typo'd flag can never silently degrade every box to
// "no report" and page like a real outage.
func mustReportsDir(dir string) {
	fi, err := os.Stat(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fleetctl: --reports %s: %v\n", dir, rootErr(err))
		os.Exit(2)
	}
	if !fi.IsDir() {
		fmt.Fprintf(os.Stderr, "fleetctl: --reports %s is not a directory\n", dir)
		os.Exit(2)
	}
}

// mustLoad loads and validates a roster, exiting 2 (a config error, distinct from the
// score --min gate's exit 1) on any problem.
func mustLoad(path string) Roster {
	ro, err := LoadRosterFile(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fleetctl: %v\n", err)
		os.Exit(2)
	}
	if probs := ro.Validate(); len(probs) > 0 {
		fmt.Fprintf(os.Stderr, "fleetctl: invalid roster %s (%d problem(s)); run `fleetctl validate` for the list\n", path, len(probs))
		os.Exit(2)
	}
	return ro
}

func emitJSON(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		fmt.Fprintf(os.Stderr, "fleetctl: encode: %v\n", err)
		os.Exit(1)
	}
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
