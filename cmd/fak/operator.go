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
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

var (
	operatorCollectCadence   = collectCadenceBriefReport
	operatorCollectProgram   = collectProgramBriefReport
	operatorCollectMilestone = collectMilestoneBriefReport
	operatorCollectHeaviness = collectHeavinessBriefReport
)

func cmdOperator(argv []string) {
	dispatchSubcommands("operator", "brief | heaviness", argv,
		subcommand{"brief", runOperatorBrief},
		subcommand{"heaviness", runOperatorHeaviness},
	)
}

func runOperatorBrief(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("operator brief", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root stamp (default: repo root)")
	cadencePath := fs.String("cadence", "", "path to `fak cadence --json` output")
	programPath := fs.String("program", "", "path to `fak program report --json` output")
	milestonePath := fs.String("milestone", "", "path to `fak milestone report --json` output")
	heavinessPath := fs.String("heaviness", "", "path to `fak operator heaviness --json` output")
	previousPath := fs.String("previous", "", "path to previous `fak operator brief --json` output for change compression")
	collect := fs.Bool("collect", false, "collect any missing report input live before folding (slower; artifact inputs still win)")
	collectTimeout := fs.Int("collect-timeout", 300, "per-source timeout seconds for --collect cadence/report calls")
	scoresFrom := fs.String("scores-from", "", "with --collect: pass a scorecard_control_pane.py JSON payload to cadence instead of rerunning the slow score pane")
	epicsFrom := fs.String("epics-from", "", "with --collect: pass a tracked-epic JSON file to milestone report")
	cacheLedger := fs.String("cache-ledger", "", "with --collect: pass a cache-value ledger path to program report")
	repo := fs.String("repo", "", "with --collect: owner/name for milestone gh roadmap queries")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "paging gate: exit non-zero only when a human operator item exists")
	date := fs.String("date", "", "snapshot date YYYY-MM-DD (default: today UTC)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak operator brief: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if stdinUses(*cadencePath, *programPath, *milestonePath, *heavinessPath, *previousPath) > 1 {
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
	} else {
		fmt.Fprintln(stdout, operatorbrief.Render(report))
	}
	if report.OK {
		return 0
	}
	return 1
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

func loadCadenceBriefReport(path string, stdin io.Reader) (cadencereport.Report, error) {
	var r cadencereport.Report
	if err := loadBriefJSON(path, stdin, &r); err != nil {
		return r, err
	}
	if r.Schema != "" && r.Schema != cadencereport.Schema {
		return r, fmt.Errorf("schema %q, want %q", r.Schema, cadencereport.Schema)
	}
	return r, nil
}

func loadProgramBriefReport(path string, stdin io.Reader) (programreport.Report, error) {
	var r programreport.Report
	if err := loadBriefJSON(path, stdin, &r); err != nil {
		return r, err
	}
	if r.Schema != "" && r.Schema != programreport.Schema {
		return r, fmt.Errorf("schema %q, want %q", r.Schema, programreport.Schema)
	}
	return r, nil
}

func loadMilestoneBriefReport(path string, stdin io.Reader) (milestonereport.Report, error) {
	var r milestonereport.Report
	if err := loadBriefJSON(path, stdin, &r); err != nil {
		return r, err
	}
	if r.Schema != "" && r.Schema != milestonereport.Schema {
		return r, fmt.Errorf("schema %q, want %q", r.Schema, milestonereport.Schema)
	}
	return r, nil
}

func loadHeavinessBriefReport(path string, stdin io.Reader) (scorecard.Payload, error) {
	var r scorecard.Payload
	if err := loadBriefJSON(path, stdin, &r); err != nil {
		return r, err
	}
	if r.Schema != "" && r.Schema != heavinessscore.Schema {
		return r, fmt.Errorf("schema %q, want %q", r.Schema, heavinessscore.Schema)
	}
	return r, nil
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
