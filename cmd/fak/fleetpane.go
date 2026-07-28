package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
)

var fleetPaneRunner fleetpane.Runner = fleetpane.OSRunner{}

func runFleetPane(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak fleetpane", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: discovered from VERSION)")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	args := fs.Args()
	if len(args) == 0 {
		fmt.Fprintln(stderr, "fak fleetpane: missing subcommand (status | fleet | loop-list | loop-check | loop-audit | selfcheck)")
		return 2
	}
	// selfcheck needs no config/runner — it is the no-I/O proof of the worker-health
	// decenter fold, so dispatch it before the config load can fail.
	if args[0] == "selfcheck" {
		return runReportSelfcheck(stdout, stderr, args[1:], "fleetpane", fleetpane.TriageSelfcheck,
			"SELFCHECK OK -- decenter-the-human at the fleet pane: an account auth wall waits on a "+
				"person; dead/stale workers are the launcher's relaunch and the monitor's attention flag "+
				"is a fresh-context look, not a page.")
	}
	cfg, err := loadFleetPaneConfig(*root)
	if err != nil {
		fmt.Fprintf(stderr, "fak fleetpane: %v\n", err)
		return 1
	}
	opts := fleetpane.Options{Runner: fleetPaneRunner}
	switch args[0] {
	case "status":
		return runFleetPaneStatus(stdout, stderr, cfg, opts, args[1:])
	case "fleet":
		return runFleetPaneFleet(stdout, stderr, cfg, opts, args[1:])
	case "loop-list":
		return runFleetPaneLoopList(stdout, stderr, cfg, opts, args[1:])
	case "loop-check":
		return runFleetPaneLoopCheck(stdout, stderr, cfg, opts, args[1:])
	case "loop-audit":
		return runFleetPaneLoopAudit(stdout, stderr, cfg, opts, args[1:])
	case "init", "tick", "recover", "supervisor", "sync", "publish", "commit", "bootstrap", "doctor", "setup-plan", "loop-scaffold", "loop-set", "loop-add":
		fmt.Fprintf(stderr, "fak fleetpane %s: unsupported in native read-only subset; use tools/fleet_control_pane.py for this mutating/recovery operation\n", args[0])
		return 2
	case "help", "-h", "--help":
		fleetPaneUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak fleetpane: unknown subcommand %q\n", args[0])
		return 2
	}
}

func loadFleetPaneConfig(root string) (fleetpane.Config, error) {
	if root == "" {
		discovered, err := fleetpane.FindRepoRoot(".")
		if err != nil {
			return fleetpane.Config{}, err
		}
		root = discovered
	}
	return fleetpane.LoadConfig(root)
}

// emitFleetPaneDoc is the one emit path every fleetpane verb shares: with --json the
// document goes out as JSON, otherwise the verb's rendered pane text does. `text` is a
// thunk so the renderer never runs in JSON mode. It returns 0, or 1 after naming the
// verb whose JSON write failed — so a broken readout is attributable to its subcommand.
func emitFleetPaneDoc(stdout, stderr io.Writer, verb string, asJSON bool, doc any, text func() string) int {
	if !asJSON {
		fmt.Fprintln(stdout, text())
		return 0
	}
	if err := fleetpane.WriteJSON(stdout, doc); err != nil {
		fmt.Fprintf(stderr, "fak fleetpane %s: write JSON: %v\n", verb, err)
		return 1
	}
	return 0
}

func runFleetPaneStatus(stdout, stderr io.Writer, cfg fleetpane.Config, opts fleetpane.Options, argv []string) int {
	fs := flag.NewFlagSet("fak fleetpane status", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	refresh := fs.Bool("refresh", false, "unsupported in native read-only subset")
	write := fs.Bool("write", false, "unsupported in native read-only subset")
	failOnAction := fs.Bool("fail-on-action", false, "exit 1 when verdict is ACTION")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *refresh || *write {
		fmt.Fprintln(stderr, "fak fleetpane status: --refresh/--write are unsupported in the native read-only subset")
		return 2
	}
	doc := fleetpane.CollectStatus(context.Background(), cfg, opts)
	if rc := emitFleetPaneDoc(stdout, stderr, "status", *asJSON, doc, func() string {
		pane := fleetpane.PaneText(doc)
		// Decenter the human at the source: under FAK_FLEETPANE_TRIAGE_GATE=enforce,
		// append the worker-health split — the workers that genuinely wait on a person
		// (an account auth wall) vs the ones the fleet's launcher clears itself. Default
		// ("", "warn") leaves the pane readout byte-for-byte unchanged while it soaks.
		if fleetpane.FleetTriageEnforced(os.Getenv("FAK_FLEETPANE_TRIAGE_GATE")) {
			if line := fleetpane.WorkerHealthTriageLine(doc.WorkerHealth); line != "" {
				pane += "\n" + line
			}
		}
		return pane
	}); rc != 0 {
		return rc
	}
	if *failOnAction && doc.Verdict != "OK" {
		return 1
	}
	return 0
}

func runFleetPaneFleet(stdout, stderr io.Writer, cfg fleetpane.Config, opts fleetpane.Options, argv []string) int {
	fs := flag.NewFlagSet("fak fleetpane fleet", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	failOnAction := fs.Bool("fail-on-action", false, "exit 1 when verdict is ACTION")
	snapshotOnly := fs.Bool("snapshot-only", false, "read published snapshots only")
	refreshLocal := fs.Bool("refresh-local", false, "unsupported in native read-only subset")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *refreshLocal {
		fmt.Fprintln(stderr, "fak fleetpane fleet: --refresh-local is unsupported in the native read-only subset")
		return 2
	}
	doc := fleetpane.FleetView(context.Background(), cfg, !*snapshotOnly, false, opts)
	if rc := emitFleetPaneDoc(stdout, stderr, "fleet", *asJSON, doc, func() string {
		return fleetpane.FleetText(doc)
	}); rc != 0 {
		return rc
	}
	if *failOnAction && doc.Verdict != "OK" {
		return 1
	}
	return 0
}

func runFleetPaneLoopList(stdout, stderr io.Writer, cfg fleetpane.Config, opts fleetpane.Options, argv []string) int {
	fs := flag.NewFlagSet("fak fleetpane loop-list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	doc := fleetpane.LoopList(cfg, opts)
	if rc := emitFleetPaneDoc(stdout, stderr, "loop-list", *asJSON, doc, func() string {
		return fleetpane.LoopListText(doc)
	}); rc != 0 {
		return rc
	}
	if !doc.OK {
		return 1
	}
	return 0
}

func runFleetPaneLoopCheck(stdout, stderr io.Writer, cfg fleetpane.Config, opts fleetpane.Options, argv []string) int {
	argv = normalizeLoopCheckArgv(argv)
	fs := flag.NewFlagSet("fak fleetpane loop-check", flag.ContinueOnError)
	fs.SetOutput(stderr)
	recover := fs.Bool("recover", false, "show recovery plan when action is needed")
	apply := fs.Bool("apply", false, "unsupported in native read-only subset")
	failOnAction := fs.Bool("fail-on-action", false, "exit 1 when loop reports ACTION")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	if *apply {
		fmt.Fprintln(stderr, "fak fleetpane loop-check: --apply is unsupported in the native read-only subset")
		return 2
	}
	args := fs.Args()
	if len(args) != 1 {
		fmt.Fprintln(stderr, "fak fleetpane loop-check: expected exactly one loop name")
		return 2
	}
	doc := fleetpane.LoopCheckPlan(context.Background(), cfg, args[0], *recover, false, opts)
	if rc := emitFleetPaneDoc(stdout, stderr, "loop-check", *asJSON, doc, func() string {
		return fleetpane.LoopCheckText(doc)
	}); rc != 0 {
		return rc
	}
	if !doc.OK {
		return 1
	}
	if *failOnAction && doc.NeedsAction {
		return 1
	}
	return 0
}

func normalizeLoopCheckArgv(argv []string) []string {
	boolFlags := map[string]bool{
		"--recover":        true,
		"--apply":          true,
		"--fail-on-action": true,
		"--json":           true,
		"-h":               true,
		"--help":           true,
	}
	var flags []string
	var positional []string
	for _, arg := range argv {
		if boolFlags[arg] {
			flags = append(flags, arg)
			continue
		}
		positional = append(positional, arg)
	}
	return append(flags, positional...)
}

func runFleetPaneLoopAudit(stdout, stderr io.Writer, cfg fleetpane.Config, opts fleetpane.Options, argv []string) int {
	fs := flag.NewFlagSet("fak fleetpane loop-audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	namesCSV := fs.String("names", "", "comma-separated subset of loops to audit")
	failOnAction := fs.Bool("fail-on-action", false, "also exit 1 when any loop is in the action bucket")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}
	names := fleetPaneSplitCSV(*namesCSV)
	doc := fleetpane.LoopAudit(context.Background(), cfg, names, opts)
	if rc := emitFleetPaneDoc(stdout, stderr, "loop-audit", *asJSON, doc, func() string {
		return fleetpane.LoopAuditText(doc)
	}); rc != 0 {
		return rc
	}
	if !doc.OK {
		return 1
	}
	if *failOnAction && doc.Counts["action"] > 0 {
		return 1
	}
	return 0
}

func fleetPaneUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak fleetpane [--root DIR] <status|fleet|loop-list|loop-check|loop-audit> [options]")
	fmt.Fprintln(w, "native read-only subset of tools/fleet_control_pane.py")
}

func fleetPaneSplitCSV(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(raw, ",") {
		if name := strings.TrimSpace(part); name != "" {
			out = append(out, name)
		}
	}
	return out
}
