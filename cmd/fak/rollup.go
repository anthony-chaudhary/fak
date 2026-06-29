package main

// fak rollup -- the executive activity roll-up: one read-only fold over the
// agentic-fleet planes (dispatch throughput + closure honesty, dark loops,
// quality cadence + ship-stamp rate, box liveness) into a single signal-dense
// page a human can read in a glance. It is the answer to "how does one person
// keep up with a city of agents": the firehose of volume is folded down to the
// marquee signal-to-noise ratio (witnessed-real vs claimed) and the short list
// of things that actually need a human now.
//
// The pure fold + envelope live in internal/execrollup; this file is the impure
// shell that measures each plane live (mirroring internal/cadencereport's
// collect.go split):
//
//   - dispatch: shells tools/dispatch_status.py --json (closure honesty + throughput)
//   - loops:    folds the cross-ledger loop-health view in-process (loopfleet.Fold)
//   - cadence:  WORK-DONE from git (always, cheap) + the SCORES pane (optional)
//   - fleet:    folds the lab roster against its reports dir in-process (fleet.Fold)
//
// Every plane degrades honestly: a collector that fails (or is skipped with
// --fast) reads UNMEASURED, which forces the fleet verdict to WATCH — never a
// silent GREEN. Each plane also accepts a pre-captured --*-from payload so the
// cron / Slack wrapper can run the slow folds once and the report stays
// deterministic and testable.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/execrollup"
	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func cmdRollup(argv []string) { os.Exit(runRollup(os.Stdout, os.Stderr, argv)) }

func runRollup(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("rollup", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit the machine-readable control-pane envelope")
	mdPath := fs.String("md", "", "write the human roll-up doc (with front-matter) to this path instead of stdout")
	fast := fs.Bool("fast", false, "skip the slow closure audit + scorecard pane; those planes read UNMEASURED (→ WATCH)")
	check := fs.Bool("check", false, "advisory gate: exit non-zero when the fleet verdict is RED")
	window := fs.Int("window", cadencereport.DefaultWindowDays, "trailing window (days) for the ship-stamp rate")
	timeout := fs.Int("timeout", 300, "per-collector timeout seconds (the slow Python folds)")
	python := fs.String("python", "", "python interpreter for the dispatch/scores folds (default: auto)")
	dispatchFrom := fs.String("dispatch-from", "", "read a dispatch_status.py --json payload (file or '-') instead of running it")
	scoresFrom := fs.String("scores-from", "", "read a scorecard_control_pane.py --json payload (file or '-') for the SCORES overlay")
	loopsFrom := fs.String("loops-from", "", "read a loop-fleet-health --json payload instead of folding in-process")
	fleetFrom := fs.String("fleet-from", "", "read a fak lab status --json payload instead of folding in-process")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak rollup: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	in := collectRollup(stderr, root, *python, *fast, *window, *timeout, fromPaths{
		dispatch: *dispatchFrom, scores: *scoresFrom, loops: *loopsFrom, fleet: *fleetFrom,
	})
	in.Workspace = root
	in.Commit = cadencereport.HeadCommit(root)
	in.GeneratedAt = time.Now().UTC().Format(time.RFC3339)

	r := execrollup.Fold(in)

	switch {
	case *asJSON:
		if err := writeIndentedJSON(stdout, r); err != nil {
			fmt.Fprintf(stderr, "fak rollup: encode: %v\n", err)
			return 1
		}
	case *mdPath != "":
		if err := writeRollupDoc(*mdPath, r); err != nil {
			fmt.Fprintf(stderr, "fak rollup: write %s: %v\n", *mdPath, err)
			return 1
		}
		fmt.Fprintf(stderr, "fak rollup: wrote %s (%s)\n", *mdPath, r.Verdict)
	default:
		fmt.Fprint(stdout, execrollup.Render(r))
	}

	if *check && r.Verdict == execrollup.VerdictRed {
		return 1
	}
	return 0
}

type fromPaths struct{ dispatch, scores, loops, fleet string }

// collectRollup measures each plane live, honoring the --*-from overrides and the
// --fast skip. Any collector failure becomes an UNMEASURED plane in the fold,
// never a panic and never a silent pass.
func collectRollup(stderr io.Writer, root, python string, fast bool, window, timeoutSec int, from fromPaths) execrollup.Inputs {
	to := time.Duration(timeoutSec) * time.Second
	var in execrollup.Inputs

	// dispatch — closure honesty (the marquee) + throughput. Slow (gh); skipped by --fast.
	switch {
	case from.dispatch != "":
		in.Dispatch = payloadFromFile(from.dispatch)
	case fast:
		in.Dispatch = execrollup.PlaneInput{Err: "skipped (--fast): closure honesty + throughput not measured"}
	default:
		p, e := cadencereport.RunPyEnvelope(root, []string{"tools/dispatch_status.py", "--json"}, python, to)
		in.Dispatch = execrollup.PlaneInput{Payload: p, Err: e}
	}

	// loops — cross-ledger dark-loop fold, in-process and instant.
	if from.loops != "" {
		in.Loops = payloadFromFile(from.loops)
	} else {
		rep := loopfleet.Fold(root, time.Now(), loopmgr.HealthThresholds{})
		in.Loops = execrollup.PlaneInput{Payload: structToMap(rep)}
	}

	// cadence — WORK-DONE (git, always) feeds the ship-stamp marquee; SCORES is an
	// optional overlay (the ~4-minute pane), skipped by --fast.
	cad := map[string]any{"work": structToMap(cadencereport.WorkFromGit(root, window))}
	switch {
	case from.scores != "":
		cad["scores"] = structToMap(cadencereport.InterpretScoresFromFile(from.scores, os.Stdin))
	case !fast:
		sp, se := cadencereport.RunPyEnvelope(root, cadencereport.ScoresArgv, python, to)
		cad["scores"] = structToMap(cadencereport.InterpretScores(sp, se))
	}
	in.Cadence = execrollup.PlaneInput{Payload: cad}

	// fleet — lab roster folded against its reports dir, in-process.
	if from.fleet != "" {
		in.Fleet = payloadFromFile(from.fleet)
	} else if snap, ok := foldFleetSnapshot(stderr); ok {
		in.Fleet = execrollup.PlaneInput{Payload: structToMap(snap)}
	} else {
		in.Fleet = execrollup.PlaneInput{Err: "lab roster/reports unavailable"}
	}

	return in
}

// foldFleetSnapshot replicates `fak lab status` in-process: load the embedded
// generic roster and fold it against the default reports dir.
//
// Signal-to-noise call: when the reports dir is EMPTY (the common case off the
// lab control-plane host — e.g. this agent-host), every box would fold to
// `unknown` and surface as "down or unreachable" crits. That conflates "we have
// no telemetry" with "confirmed down" — a phantom emergency, the exact false
// alarm an executive roll-up must not raise. So no-reports reads UNMEASURED (a
// WATCH gap), not a fleet of crits. On a host where reports ARE populated it
// folds normally and the real box verdicts flow through.
func foldFleetSnapshot(stderr io.Writer) (fleet.Snapshot, bool) {
	ro, ok := labLoadRoster(io.Discard, "")
	if !ok {
		return fleet.Snapshot{}, false
	}
	ro = labSelect(ro, "", "")
	dir, err := labReportsDir("")
	if err != nil || !labReportsPopulated(dir) {
		return fleet.Snapshot{}, false
	}
	return fleet.Fold(ro, fleet.ReadReports(dir, ro), fleet.FoldOpts{}), true
}

// structToMap round-trips a typed fold payload through JSON into the
// map[string]any shape execrollup's interpreters consume — so the command speaks
// the same on-the-wire contract a shelled `--json` collector would.
func structToMap(v any) map[string]any {
	b, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return nil
	}
	return m
}

// payloadFromFile reads a pre-captured --*-from JSON payload (a file path, or "-"
// for stdin). A read/parse failure becomes an errored PlaneInput, never a silent
// nil, so an injected-but-broken payload reads UNMEASURED.
func payloadFromFile(path string) execrollup.PlaneInput {
	var (
		data []byte
		err  error
	)
	if path == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return execrollup.PlaneInput{Err: path + ": " + err.Error()}
	}
	var m map[string]any
	if json.Unmarshal(data, &m) != nil || m == nil {
		return execrollup.PlaneInput{Err: path + ": not a JSON object payload"}
	}
	return execrollup.PlaneInput{Payload: m}
}

// writeRollupDoc writes the human roll-up with jekyll-seo-tag front-matter so the
// committed doc renders on Pages and never 404s a description (the same discipline
// tools/dispatch_status.py --md follows).
func writeRollupDoc(path string, r execrollup.Rollup) error {
	var b []byte
	b = append(b, []byte("---\n")...)
	b = append(b, []byte("title: \"fak executive activity roll-up: the few signals a human owes attention\"\n")...)
	b = append(b, []byte("description: \"Auto-generated signal-to-noise roll-up across the agentic-fleet planes — closure honesty, dark loops, ship-stamp rate, box liveness — folded into one GREEN/WATCH/RED verdict and a ranked what-needs-you list.\"\n")...)
	b = append(b, []byte("---\n\n")...)
	b = append(b, []byte("# Fleet activity roll-up — "+r.GeneratedAt+"\n\n")...)
	b = append(b, []byte("_Auto-generated by `fak rollup --md`. Do not hand-edit; re-run the tool. "+
		"Commit `"+r.Commit+"`. Provenance: WITNESSED (proven from git/tests) · OBSERVED (a live reading) · CLAIMED (self-reported, no witness)._\n\n")...)
	b = append(b, []byte(execrollup.Render(r))...)
	return os.WriteFile(path, b, 0o644)
}
