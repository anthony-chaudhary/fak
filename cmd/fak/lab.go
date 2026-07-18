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
//	fak lab report --id ID --state live|idle|draining|down [--version V] [--note N] [--inference ready|degraded|warming|blocked|unknown] [--reports DIR]
//	fak lab readiness [--file F] [--phase CLEAR|WAITING|WORKING [--detail D]|--from-reports] [--write F|--write-default] [--json]
//	fak lab target ALIAS [--targets F] [--readiness F] [--roster F] [--reports DIR] [--json]
//	fak lab ls     [--roster F] [--group G] [--class C] [--json]
//
// THE PUBLIC/PRIVATE BOUNDARY IS A DATA CONTRACT. The embedded roster is generic (an
// id, a class, a group — never a real lab host, channel, or token); the private Slack
// bridge owns the id->channel map and the live `!sessions` liveness probe on its side
// and writes one fak.fleet.report/v1 file per box. `fak lab report` is the PUBLIC
// producer half: a box self-reports its state with no private bridge, closing the
// loop for that box today. See docs/gpu-server-private-boundary.md and docs/fak/lab-dev-loop.md.

import (
	"bytes"
	_ "embed"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleet"
	"github.com/anthony-chaudhary/fak/internal/linkstate"
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
		fmt.Fprintln(stderr, "usage: fak lab <status|report|readiness|target|ls|private-path> [flags]")
		fmt.Fprintln(stderr, "       fak lab status            # which lab nodes are alive right now?")
		fmt.Fprintln(stderr, "       fak lab readiness --json  # public-safe lab dispatch gate")
		fmt.Fprintln(stderr, "       fak lab target @lab/glm-5.2 --json  # validate a local lab inference alias")
		fmt.Fprintln(stderr, "       fak lab report --id ID --state live   # self-report this box")
		return 2
	}
	switch argv[0] {
	case "status":
		return labStatus(stdout, stderr, argv[1:])
	case "report":
		return labReport(stdout, stderr, argv[1:])
	case "readiness":
		return labReadiness(stdout, stderr, argv[1:])
	case "target":
		return labTarget(stdout, stderr, argv[1:])
	case "ls", "list":
		return labLs(stdout, stderr, argv[1:])
	case "private-path":
		return runPrivatePath(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, "usage: fak lab <status|report|readiness|target|ls|private-path> [flags]")
		fmt.Fprintln(stdout, "  status   fold the lab roster against the reports dir and render the fleet view")
		fmt.Fprintln(stdout, "  report   write one fak.fleet.report/v1 line for a box (self-report; no bridge)")
		fmt.Fprintln(stdout, "  readiness read or write the public fak.lab_readiness/v1 dispatch gate")
		fmt.Fprintln(stdout, "  target   validate a public-safe @lab/<model> alias for fak guard --remote-serve")
		fmt.Fprintln(stdout, "  ls       list the boxes in the (default or --roster) lab roster")
		fmt.Fprintln(stdout, "  private-path resolve an opaque run directory in the paired private repo")
		return 0
	default:
		fmt.Fprintf(stderr, "fak lab: unknown subcommand %q (want status|report|readiness|target|ls|private-path)\n", argv[0])
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

func labReadinessPath(flagVal string) (string, error) {
	if flagVal != "" {
		return flagVal, nil
	}
	if env := os.Getenv("FAK_LAB_READINESS"); env != "" {
		return env, nil
	}
	cfgDir, err := nodeConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(cfgDir, "fleet", "lab-readiness.json"), nil
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
	if !parseFlags(fs, argv) {
		return 2
	}

	ro, ok := labLoadRoster(stderr, *rosterPath)
	if !ok {
		return 2
	}
	fullRoster := ro // keep the unfiltered roster for orphan detection (below)
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
	// Surface report files that bind to no roster box (a producer writing under
	// non-roster keys) so a real "0/N reachable" is not mistaken for a dead fleet when
	// reports are in fact present, just misfiled. Detect against the FULL roster so a
	// --group/--class filter does not flag a filtered-out box's report as an orphan.
	snap.Orphans = fleet.OrphanReports(dir, fullRoster)

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
	inference := fs.String("inference", "", "serving usefulness: ready|degraded|warming|blocked|unknown")
	engine := fs.String("engine", "", "generic inference engine label for --inference (optional)")
	model := fs.String("model", "", "generic model label for --inference (optional; never a private path)")
	outputTPS := fs.Float64("output-tps", 0, "observed output tokens/sec for --inference (optional)")
	probeLatencyMS := fs.Float64("probe-latency-ms", 0, "scrubbed end-to-end probe latency in milliseconds for --inference (optional)")
	reason := fs.String("reason", "", "short generic inference reason for --inference (optional)")
	reports := fs.String("reports", "", "reports dir (default: $FAK_FLEET_REPORTS or ~/.config/fak/fleet/reports)")
	if !parseFlags(fs, argv) {
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
	if *outputTPS < 0 {
		fmt.Fprintln(stderr, "fak lab report: --output-tps must be non-negative")
		return 2
	}
	if *probeLatencyMS < 0 {
		fmt.Fprintln(stderr, "fak lab report: --probe-latency-ms must be non-negative")
		return 2
	}

	dir, err := labReportsDir(*reports)
	if err != nil {
		fmt.Fprintf(stderr, "fak lab report: %v\n", err)
		return 1
	}
	rep := fleet.Report{State: st, Version: *version, Note: *note}
	if *inference != "" || *engine != "" || *model != "" || *outputTPS > 0 || *probeLatencyMS > 0 || *reason != "" {
		status := fleet.InferenceStatus(*inference)
		if status == "" {
			status = fleet.InferenceUnknown
		}
		rep.Inference = &fleet.InferenceStats{
			Status:         status,
			Engine:         *engine,
			Model:          *model,
			OutputTPS:      *outputTPS,
			ProbeLatencyMS: *probeLatencyMS,
			Reason:         *reason,
		}
	}
	if err := fleet.WriteReport(dir, *id, rep); err != nil {
		fmt.Fprintf(stderr, "fak lab report: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "[fak lab] wrote %s state=%s -> %s\n", *id, st, filepath.Join(dir, *id+".json"))
	return 0
}

func labReadiness(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("lab readiness", flag.ContinueOnError)
	fs.SetOutput(stderr)
	file := fs.String("file", "", "readiness record path (default: $FAK_LAB_READINESS or ~/.config/fak/fleet/lab-readiness.json)")
	rosterPath := fs.String("roster", "", "with --from-reports: roster file (default: the embedded generic lab roster)")
	reports := fs.String("reports", "", "with --from-reports: reports dir (default: $FAK_FLEET_REPORTS or ~/.config/fak/fleet/reports)")
	machineClass := fs.String("class", "gpu-server", "generic machine class")
	phaseFlag := fs.String("phase", "", "emit this link-state phase (CLEAR|WAITING|WORKING) instead of reading a file")
	detailFlag := fs.String("detail", "", "link-state detail sub-status for --phase (default: the phase's canonical detail)")
	statusFlag := fs.String("status", "", "DEPRECATED: legacy readiness status, coarsened onto a --phase during rollover")
	fromReports := fs.Bool("from-reports", false, "derive the phase from scrubbed lab status/inference reports instead of reading a record")
	fromStatus := fs.Bool("from-status", false, "DEPRECATED alias for --from-reports")
	nextAction := fs.String("next-action", "", "generic next action for --phase")
	evidence := fs.String("evidence", "", "generic evidence label for --phase")
	writePath := fs.String("write", "", "write the emitted record to this path")
	writeDefault := fs.Bool("write-default", false, "write the emitted record to the default readiness path")
	asJSON := fs.Bool("json", false, "emit the readiness record as JSON")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *writeDefault && *writePath != "" {
		fmt.Fprintln(stderr, "fak lab readiness: use only one of --write or --write-default")
		return 2
	}
	deriving := *fromReports || *fromStatus

	// Resolve the requested phase from --phase (primary) or the deprecated --status
	// alias (coarsened for safe rollover while callers still speak the old vocabulary).
	reqPhase := linkstate.Phase(*phaseFlag)
	reqDetail := *detailFlag
	if *statusFlag != "" {
		if *phaseFlag != "" {
			fmt.Fprintln(stderr, "fak lab readiness: use only one of --phase or --status")
			return 2
		}
		reqPhase, reqDetail = linkstate.Coarsen(*statusFlag)
	}
	if deriving && reqPhase != "" {
		fmt.Fprintln(stderr, "fak lab readiness: use only one of --phase or --from-reports")
		return 2
	}

	var rec fleet.LabReadiness
	emitRecord := reqPhase != "" || deriving
	if deriving {
		ro, ok := labLoadRoster(stderr, *rosterPath)
		if !ok {
			return 2
		}
		dir, err := labReportsDir(*reports)
		if err != nil {
			fmt.Fprintf(stderr, "fak lab readiness: %v\n", err)
			return 2
		}
		var reps []fleet.Report
		if labReportsPopulated(dir) {
			reps = fleet.ReadReports(dir, ro)
		}
		// Reconcile the report-key/roster join (#5065): the private bridge keys its
		// report files by its own labels while the generic roster keys by box id, so a
		// live lab could fold as "0 reachable + orphans" and hold dispatch closed on an
		// INDETERMINATE verdict. Adopt every orphan file that parses as a valid report
		// as its own synthetic box so a correctly-produced report reaches a determinate
		// phase under whatever key it was filed; only files that CANNOT be read as
		// reports remain orphans, keeping "reports-under-non-roster-keys" for genuinely
		// unreconcilable keys and every fail-closed hold (silent, stale, warming,
		// blocked lab) exactly as strict as before.
		adopted, orphans := fleet.AdoptOrphanReports(dir, ro)
		foldRo := ro
		if len(adopted) > 0 {
			boxes := make([]fleet.Box, 0, len(ro.Boxes)+len(adopted))
			boxes = append(boxes, ro.Boxes...)
			if reps == nil {
				reps = fleet.ReadReports(dir, ro) // align reports[i] with boxes[i] before appending
			}
			for _, r := range adopted {
				boxes = append(boxes, fleet.Box{ID: r.ID, Class: *machineClass})
				reps = append(reps, r)
			}
			foldRo = fleet.Roster{Schema: ro.Schema, Boxes: boxes}
		}
		snap := fleet.Fold(foldRo, reps, fleet.FoldOpts{})
		snap.Orphans = orphans // name a genuine key-mismatch instead of "no reports"
		rec = labReadinessFromSnapshot(*machineClass, snap, time.Now())
		if probs := rec.Validate(); len(probs) > 0 {
			fmt.Fprintf(stderr, "fak lab readiness: invalid record: %s\n", strings.Join(probs, "; "))
			return 2
		}
	} else if reqPhase != "" {
		rec = fleet.NewLabReadiness(*machineClass, reqPhase, reqDetail, *nextAction, *evidence, time.Now())
		if probs := rec.Validate(); len(probs) > 0 {
			fmt.Fprintf(stderr, "fak lab readiness: invalid record: %s\n", strings.Join(probs, "; "))
			return 2
		}
	} else {
		if *writePath != "" || *writeDefault {
			fmt.Fprintln(stderr, "fak lab readiness: --write/--write-default requires --phase or --from-reports")
			return 2
		}
		path, err := labReadinessPath(*file)
		if err != nil {
			fmt.Fprintf(stderr, "fak lab readiness: %v\n", err)
			return 1
		}
		f, err := os.Open(path)
		if err != nil {
			rec = fleet.IndeterminateLabReadiness(*machineClass, "publish-lab-readiness", "no-readiness-record", time.Now())
		} else {
			defer f.Close()
			rec, err = fleet.LoadLabReadiness(f)
			if err != nil {
				fmt.Fprintf(stderr, "fak lab readiness: invalid %s: %v\n", path, err)
				rec = fleet.IndeterminateLabReadiness(*machineClass, "fix-lab-readiness-record", "invalid-readiness-record", time.Now())
			}
		}
	}

	if emitRecord {
		if *writeDefault {
			path, err := labReadinessPath(*file)
			if err != nil {
				fmt.Fprintf(stderr, "fak lab readiness: %v\n", err)
				return 1
			}
			*writePath = path
		}
		if *writePath != "" {
			if err := writeIndentedJSONFile(*writePath, rec); err != nil {
				fmt.Fprintf(stderr, "fak lab readiness: write %s: %v\n", *writePath, err)
				return 1
			}
		}
	}

	cmds := fleet.DefaultLabReadinessCommands()
	rec.Commands = &cmds
	if *asJSON {
		if err := writeIndentedJSON(stdout, rec); err != nil {
			fmt.Fprintf(stderr, "fak lab readiness: encode: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintf(stdout, "lab readiness %s: %s/%s (%s)\n", rec.Subject, rec.Phase, rec.Detail, rec.NextAction)
		if !rec.AdmitDispatch {
			fmt.Fprintf(stdout, "mark clear: %s\n", cmds.MarkClear)
			fmt.Fprintf(stdout, "mark waiting: %s\n", cmds.MarkWaiting)
		}
	}
	if rec.AdmitDispatch {
		return 0
	}
	return 1
}

func labReadinessFromSnapshot(machineClass string, snap fleet.Snapshot, now time.Time) fleet.LabReadiness {
	var reported, useful, warming, blocked, latencyDegraded int
	for _, row := range snap.Rows {
		if row.Err != "" || row.Inference == nil || row.AgeSec > fleet.DefaultStaleSec {
			continue
		}
		reported++
		switch row.Inference.Status {
		case fleet.InferenceReady, fleet.InferenceDegraded:
			if row.State.Healthy() {
				if labTargetProbeLatencyDegraded(row.Inference) {
					latencyDegraded++
					continue
				}
				useful++
			}
		case fleet.InferenceWarming:
			warming++
		case fleet.InferenceBlocked:
			blocked++
		}
	}
	if useful > 0 {
		return fleet.NewLabReadiness(machineClass, linkstate.Clear, linkstate.DetailReady, "admit-lab-backed-dispatch", "scrubbed-fleet-report", now)
	}
	switch {
	case latencyDegraded > 0:
		return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailPrivateRecovery, "route-latency-exceeds-dev-budget-refresh-report-or-use-fallback", "scrubbed-fleet-report", now)
	case warming > 0:
		return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailPrivateRecovery, "wait-lab-inference-ready", "scrubbed-fleet-report", now)
	case blocked > 0:
		return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailPrivateRecovery, "recover-lab-inference", "scrubbed-fleet-report", now)
	case reported > 0:
		return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailIndeterminate, "publish-inference-report", "no-useful-lab-report", now)
	}
	if snap.Reachable == 0 {
		// Reports may be PRESENT yet bind to no box (a producer writing under non-roster
		// keys): that reads as reachable==0 too, but the honest remedy is to reconcile the
		// report keys with the roster, NOT "publish a report" (there already is one).
		if len(snap.Orphans) > 0 {
			return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailIndeterminate, "reconcile-report-keys-with-roster", "reports-under-non-roster-keys", now)
		}
		return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailIndeterminate, "publish-lab-readiness", "no-fresh-lab-report", now)
	}
	return fleet.NewLabReadiness(machineClass, linkstate.Waiting, linkstate.DetailIndeterminate, "publish-inference-report", "no-useful-lab-report", now)
}

func labLs(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("lab ls", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rosterPath := fs.String("roster", "", "roster file (default: the embedded generic lab roster)")
	group := fs.String("group", "", "only boxes in this group")
	class := fs.String("class", "", "only boxes of this class")
	asJSON := fs.Bool("json", false, "emit the roster as JSON")
	if !parseFlags(fs, argv) {
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
