package main

// fak operator -- human pacing surfaces. The brief subcommand folds control-pane
// JSON into one "what needs a person vs what agents can handle" envelope. It is
// artifact-first by default, with an explicit --collect mode for live reports.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cadencereport"
	"github.com/anthony-chaudhary/fak/internal/heavinessscore"
	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/operatorbrief"
	"github.com/anthony-chaudhary/fak/internal/programreport"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

var (
	operatorCollectCadence   = collectCadenceBriefReport
	operatorCollectProgram   = collectProgramBriefReport
	operatorCollectMilestone = collectMilestoneBriefReport
	operatorCollectHeaviness = collectHeavinessBriefReport
	operatorCollectOSP       = collectOSPBriefInput
)

func runOperatorBrief(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("operator brief", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root stamp (default: repo root)")
	cadencePath := fs.String("cadence", "", "path to `fak cadence --json` output")
	programPath := fs.String("program", "", "path to `fak program report --json` output")
	milestonePath := fs.String("milestone", "", "path to `fak milestone report --json` output")
	heavinessPath := fs.String("heaviness", "", "path to `fak operator heaviness --json` output")
	ospPath := fs.String("osp", "", "path to `fak steer prs --json` (operator-steerability overlay) output; a missing/unreadable payload marks the source unmeasured rather than failing the brief")
	debtPath := fs.String("debt-witnesses", "", "path to task/session debt witness JSON (array or {records:[...]})")
	previousPath := fs.String("previous", "", "path to previous `fak operator brief --json` output for change compression")
	collect := fs.Bool("collect", false, "collect any missing report input live before folding (slower; artifact inputs still win)")
	collectTimeout := fs.Int("collect-timeout", 300, "per-source timeout seconds for --collect cadence/report calls")
	scoresFrom := fs.String("scores-from", "", "with --collect: pass a scorecard_control_pane.py JSON payload to cadence instead of rerunning the slow score pane")
	epicsFrom := fs.String("epics-from", "", "with --collect: pass a tracked-epic JSON file to milestone report")
	cacheLedger := fs.String("cache-ledger", "", "with --collect: pass a cache-value ledger path to program report")
	repo := fs.String("repo", "", "with --collect: owner/name for milestone gh roadmap queries")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	full := fs.Bool("full", false, "expand every item with its detail and action (default is a compact one-line-per-item view)")
	check := fs.Bool("check", false, "paging gate: exit non-zero only when a human operator item exists")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak operator brief: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if stdinUses(*cadencePath, *programPath, *milestonePath, *heavinessPath, *ospPath, *debtPath, *previousPath) > 1 {
		fmt.Fprintln(stderr, "fak operator brief: only one report input may use '-' for stdin")
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	snapDate := *date
	if snapDate == "" {
		snapDate = time.Now().UTC().Format("2006-01-02")
	}

	in := operatorbrief.Inputs{
		Workspace: root,
		Date:      snapDate,
		// Decenter-the-human paging gate, soakable via env: "enforce" pages only
		// on genuine authority decisions; "warn"/unset (default) leaves paging
		// unchanged. `fak operator triage` is the always-on lens for the same fold.
		TriageGate: os.Getenv("FAK_OPERATOR_TRIAGE_GATE"),
	}

	if *cadencePath != "" {
		r, err := loadCadenceBriefReport(*cadencePath, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: --cadence: %v\n", err)
			return 2
		}
		in.Cadence = &r
	}
	if *cadencePath == "" && *collect {
		r, err := operatorCollectCadence(root, snapDate, *collectTimeout, *scoresFrom)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: collect cadence: %v\n", err)
			return 1
		}
		in.Cadence = &r
	}
	if *programPath != "" {
		r, err := loadProgramBriefReport(*programPath, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: --program: %v\n", err)
			return 2
		}
		in.Program = &r
	}
	if *programPath == "" && *collect {
		r, err := operatorCollectProgram(root, snapDate, *cacheLedger)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: collect program: %v\n", err)
			return 1
		}
		in.Program = &r
	}
	if *milestonePath != "" {
		r, err := loadMilestoneBriefReport(*milestonePath, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: --milestone: %v\n", err)
			return 2
		}
		in.Milestone = &r
	}
	if *milestonePath == "" && *collect {
		r, err := operatorCollectMilestone(root, snapDate, *repo, *epicsFrom)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: collect milestone: %v\n", err)
			return 1
		}
		in.Milestone = &r
	}
	if *heavinessPath != "" {
		r, err := loadHeavinessBriefReport(*heavinessPath, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: --heaviness: %v\n", err)
			return 2
		}
		in.Heaviness = &r
	}
	if *heavinessPath == "" && *collect {
		r, err := operatorCollectHeaviness(root)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: collect heaviness: %v\n", err)
			return 1
		}
		in.Heaviness = &r
	}
	if *ospPath != "" {
		o := loadOSPBriefInput(*ospPath, os.Stdin)
		in.OSP = &o
	}
	if *debtPath != "" {
		records, err := loadDebtWitnesses(*debtPath, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: --debt-witnesses: %v\n", err)
			return 2
		}
		in.DebtWitnesses = records
	}
	if *ospPath == "" && *collect {
		// Unlike the program/milestone/heaviness collect arms, the OSP overlay
		// is an optional source that must NEVER red the brief (#5126): a producer
		// that exits non-zero or emits no parseable envelope folds as unmeasured,
		// exactly as a missing --osp payload would, so --collect still exits as it
		// would have without the overlay.
		o := operatorCollectOSP(root)
		in.OSP = &o
	}
	if *previousPath != "" {
		r, err := loadPreviousOperatorBrief(*previousPath, os.Stdin)
		if err != nil {
			fmt.Fprintf(stderr, "fak operator brief: --previous: %v\n", err)
			return 2
		}
		in.Previous = &r
	}

	report := operatorbrief.Fold(in)
	if *check {
		code, message := operatorbrief.CheckGate(report)
		if *asJSON {
			_ = writeIndentedJSONNoEscape(stdout, report.WithGate(code, message))
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
	}
	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, report)
	} else if *full {
		fmt.Fprintln(stdout, operatorbrief.Render(report))
	} else {
		fmt.Fprintln(stdout, operatorbrief.RenderCompact(report))
	}
	if report.OK {
		return 0
	}
	return 1
}

// operatorTriageView is the JSON envelope for `fak operator triage --json`: the
// triaged brief with the list of items choicetriage routed off the human bucket.
type operatorTriageView struct {
	operatorbrief.Report
	Reassignments []operatorbrief.Reassignment `json:"reassignments,omitempty"`
}

// runOperatorTriage is the decenter-the-human lens: it loads a `fak operator brief
// --json` artifact and re-partitions its human bucket through choicetriage so the
// gate pages only on genuine authority decisions. Everything else is routed back
// to the fleet (act directly, spawn a fresh-context judge, or file a ticket). It
// always enforces — the env soak switch governs only the brief's own gate.
func runOperatorTriage(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "selfcheck" {
		return runReportSelfcheck(stdout, stderr, argv[1:], "operator triage", operatorbrief.TriageSelfcheck,
			"SELFCHECK OK -- decenter-the-human gate: a genuine authority decision keeps "+
				"paging; a knowable evaluation routes to the fleet and stops paging.")
	}
	fs := flag.NewFlagSet("operator triage", flag.ContinueOnError)
	fs.SetOutput(stderr)
	briefPath := fs.String("brief", "", "path to `fak operator brief --json` output to triage ('-' for stdin)")
	asJSON := fs.Bool("json", false, "emit the triaged brief plus reassignments as JSON")
	check := fs.Bool("check", false, "paging gate: exit non-zero only when a genuine human-residual item remains after triage")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak operator triage: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *briefPath == "" {
		fmt.Fprintln(stderr, "fak operator triage: --brief PATH (or '-') is required; produce it with `fak operator brief --json`")
		return 2
	}

	loaded, err := loadPreviousOperatorBrief(*briefPath, os.Stdin)
	if err != nil {
		fmt.Fprintf(stderr, "fak operator triage: --brief: %v\n", err)
		return 2
	}

	triaged, moved := operatorbrief.TriageHumanBucket(loaded)
	triaged = triaged.Reconcile()
	code, message := operatorbrief.CheckGate(triaged)

	if *asJSON {
		_ = writeIndentedJSONNoEscape(stdout, operatorTriageView{
			Report:        triaged.WithGate(code, message),
			Reassignments: moved,
		})
	} else {
		fmt.Fprintln(stdout, message)
		if len(moved) == 0 {
			fmt.Fprintln(stdout, "routed to fleet: none (every human item is a genuine authority decision)")
		}
		for _, m := range moved {
			resolve := m.Resolve
			if strings.TrimSpace(resolve) == "" {
				resolve = "(no action)"
			}
			why := ""
			if strings.TrimSpace(m.Reason) != "" {
				why = " (" + m.Reason + ")"
			}
			fmt.Fprintf(stdout, "routed to fleet: %s/%s -> %s%s: %s\n", m.Source, m.Title, m.Disposition, why, resolve)
		}
	}

	if *check {
		return code
	}
	return 0
}

func stdinUses(paths ...string) int {
	var n int
	for _, p := range paths {
		if p == "-" {
			n++
		}
	}
	return n
}

func loadOperatorBriefReport[T any](path string, stdin io.Reader, wantSchema string, schemaOf func(T) string) (T, error) {
	var report T
	if err := loadBriefJSON(path, stdin, &report); err != nil {
		return report, err
	}
	if schema := schemaOf(report); schema != "" && schema != wantSchema {
		return report, fmt.Errorf("schema %q, want %q", schema, wantSchema)
	}
	return report, nil
}

func loadCadenceBriefReport(path string, stdin io.Reader) (cadencereport.Report, error) {
	return loadOperatorBriefReport(path, stdin, cadencereport.Schema, func(r cadencereport.Report) string { return r.Schema })
}

func loadProgramBriefReport(path string, stdin io.Reader) (programreport.Report, error) {
	return loadOperatorBriefReport(path, stdin, programreport.Schema, func(r programreport.Report) string { return r.Schema })
}

func loadMilestoneBriefReport(path string, stdin io.Reader) (milestonereport.Report, error) {
	return loadOperatorBriefReport(path, stdin, milestonereport.Schema, func(r milestonereport.Report) string { return r.Schema })
}

func loadHeavinessBriefReport(path string, stdin io.Reader) (scorecard.Payload, error) {
	return loadOperatorBriefReport(path, stdin, heavinessscore.Schema, func(r scorecard.Payload) string { return r.Schema })
}

func loadDebtWitnesses(path string, stdin io.Reader) ([]operatorbrief.DebtWitnessRecord, error) {
	var data []byte
	var err error
	if path == "-" {
		data, err = io.ReadAll(stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, err
	}
	var records []operatorbrief.DebtWitnessRecord
	if err := json.Unmarshal(data, &records); err == nil {
		return records, nil
	}
	var envelope struct {
		Records []operatorbrief.DebtWitnessRecord `json:"records"`
	}
	if err := json.Unmarshal(data, &envelope); err != nil {
		return nil, fmt.Errorf("parse JSON: %w", err)
	}
	return envelope.Records, nil
}

func loadPreviousOperatorBrief(path string, stdin io.Reader) (operatorbrief.Report, error) {
	var r operatorbrief.Report
	if err := loadBriefJSON(path, stdin, &r); err != nil {
		return r, err
	}
	if r.Schema != "" && r.Schema != operatorbrief.Schema {
		return r, fmt.Errorf("schema %q, want %q", r.Schema, operatorbrief.Schema)
	}
	return r, nil
}

// loadOSPBriefInput reads a `fak steer prs --json` (or any steerpr overlay, e.g.
// `fak release prplan --json`) payload into an operatorbrief.OSP. Per the
// acceptance gate, a missing or unreadable payload does NOT fail the brief: it
// returns an Unreadable OSP so the source folds as unmeasured, never as a clean
// zero. Only the {schema, units} envelope is consumed; the units carry their own
// bands.
func loadOSPBriefInput(path string, stdin io.Reader) operatorbrief.OSP {
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return operatorbrief.OSP{Unreadable: true, Note: fmt.Sprintf("read %s: %v", path, err)}
	}
	var env struct {
		Schema string         `json:"schema"`
		Units  []steerpr.Unit `json:"units"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		return operatorbrief.OSP{Unreadable: true, Note: fmt.Sprintf("parse %s: %v", path, err)}
	}
	return operatorbrief.OSP{Schema: env.Schema, Units: env.Units}
}

// collectOSPBriefInput folds the steer-overlay's {schema, units} view into an
// operatorbrief.OSP for the --collect path. It consumes the overlay's PURE fold
// (buildSteerPRsView) directly rather than the exit-code-bearing CLI entry point
// runSteerPRs: the --check exit code must stay reachable only from the operator
// CLI dispatch (the internal/architest steer-overlay floor,
// TestSteerOverlayCheckStaysOffCommitAndPromotionPaths), never consumed off a
// commit/hook/ship/promotion path — a blocking wire here would be
// OVERLAY_WOULD_GATE. Per #5126 the overlay is an optional source that must never
// red the brief: a fold that errors returns an Unreadable OSP so the source folds
// as unmeasured (never a clean zero), exactly as a missing --osp payload does
// through loadOSPBriefInput.
func collectOSPBriefInput(root string) operatorbrief.OSP {
	view, err := buildSteerPRsView(root, "", "")
	if err != nil {
		return operatorbrief.OSP{Unreadable: true, Note: fmt.Sprintf("collect steer prs: %v", err)}
	}
	schema, _ := view["schema"].(string)
	units, _ := view["units"].([]steerpr.Unit)
	return operatorbrief.OSP{Schema: schema, Units: units}
}

func loadBriefJSON(path string, stdin io.Reader, dst any) error {
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("parse JSON: %w", err)
	}
	return nil
}

func collectCadenceBriefReport(root, date string, timeout int, scoresFrom string) (cadencereport.Report, error) {
	args := []string{"--workspace", root, "--date", date, "--json", "--timeout", strconv.Itoa(timeout)}
	if scoresFrom != "" {
		args = append(args, "--scores-from", scoresFrom)
	}
	var r cadencereport.Report
	err := collectBriefJSON("cadence", func(stdout, stderr io.Writer) int {
		return runCadence(stdout, stderr, args)
	}, &r)
	return r, err
}

func collectProgramBriefReport(root, date, cacheLedger string) (programreport.Report, error) {
	args := []string{"--workspace", root, "--date", date, "--json"}
	if cacheLedger != "" {
		args = append(args, "--cache-ledger", cacheLedger)
	}
	var r programreport.Report
	err := collectBriefJSON("program report", func(stdout, stderr io.Writer) int {
		return runProgramReport(stdout, stderr, args)
	}, &r)
	return r, err
}

func collectMilestoneBriefReport(root, date, repo, epicsFrom string) (milestonereport.Report, error) {
	args := []string{"--workspace", root, "--date", date, "--json"}
	if repo != "" {
		args = append(args, "--repo", repo)
	}
	if epicsFrom != "" {
		args = append(args, "--epics-from", epicsFrom)
	}
	var r milestonereport.Report
	err := collectBriefJSON("milestone report", func(stdout, stderr io.Writer) int {
		return runMilestoneReport(stdout, stderr, args)
	}, &r)
	return r, err
}

func collectHeavinessBriefReport(root string) (scorecard.Payload, error) {
	args := []string{"--workspace", root, "--json"}
	var r scorecard.Payload
	err := collectBriefJSON("operator heaviness", func(stdout, stderr io.Writer) int {
		return runOperatorHeaviness(stdout, stderr, args)
	}, &r)
	return r, err
}

func collectBriefJSON(label string, run func(stdout, stderr io.Writer) int, dst any) error {
	var out, errb bytes.Buffer
	code := run(&out, &errb)
	if err := json.Unmarshal(out.Bytes(), dst); err != nil {
		detail := strings.TrimSpace(errb.String())
		if detail == "" {
			detail = strings.TrimSpace(out.String())
		}
		if detail == "" {
			detail = "no JSON output"
		}
		return fmt.Errorf("%s exited %d without parseable JSON: %s", label, code, detail)
	}
	return nil
}
