package main

// balance.go — `fak balance`, the NIGHT-BALANCE readout (#3128): recovery-vs-stranding
// and gardening-vs-throughput, side by side, folded from the two subsystems that own each.
// It is the WIRE half of internal/balance's pure renderer — it does the I/O the leaf must
// not: walk a resume transcript store into per-session resume states (the #3124 half) and
// read a `fak superloop walk --json` report for its work mix (the #3126 half), then hands
// both to balance.Render.
//
// Either half is optional, which is the whole point of the surface: with no --store the
// resume half degrades to "not measured", with no --walk the mix half degrades, and with
// neither there is nothing to report. The exit code carries the ONE hard alarm — a resume
// budget whose re-stranding outpaces completion exits non-zero, so a night gate reading $?
// sees an underwater recovery budget without scraping the panel.
//
//	fak balance --store ~/.claude/projects/<project>            # resume half only
//	fak balance --walk walk.json                                # mix half only
//	fak balance --store DIR --walk walk.json [--json]           # both, folded
//	fak superloop walk run-the-night --json | fak balance --store DIR --walk -

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/balance"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func cmdBalance(argv []string) { os.Exit(runBalance(os.Stdout, os.Stderr, argv)) }

// runBalance folds whichever halves the flags supply and renders the balance panel.
// Returns the process exit code: 0 ok, 1 either a runtime error OR a red recovery budget
// (the hard alarm), 2 a usage error.
func runBalance(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("balance", flag.ContinueOnError)
	fs.SetOutput(stderr)
	store := fs.String("store", "", "resume transcript store (directory of .jsonl) — folds the resume-budget half")
	ledger := fs.String("ledger", defaultResumeLedger(), "durable resume ledger JSONL (the launch record the resume fold joins against)")
	maxAttempts := fs.Int("max-attempts", resume.DefaultMaxResumeAttempts, "give-up cap on automatic resumes of one session")
	walk := fs.String("walk", "", "a `fak superloop walk --json` report (file path, or - for stdin) — folds the work-mix half")
	asJSON := fs.Bool("json", false, "emit the folded evidence + status as JSON instead of the human panel")
	if !parseFlags(fs, argv) {
		return 2
	}
	if *store == "" && *walk == "" {
		fmt.Fprintln(stderr, "fak balance: need --store DIR (resume half) and/or --walk FILE (mix half) — nothing to fold")
		return 2
	}

	var ev balance.Evidence
	if *store != "" {
		states, err := collectResumeStates(pathutil.ExpandTilde(*store), pathutil.ExpandTilde(*ledger), *maxAttempts)
		if err != nil {
			fmt.Fprintf(stderr, "fak balance: %v\n", err)
			return 1
		}
		ev.Resume = balance.FoldResumeBudget(states)
	}
	if *walk != "" {
		mix, err := readWalkMix(*walk)
		if err != nil {
			fmt.Fprintf(stderr, "fak balance: read walk %q: %v\n", *walk, err)
			return 1
		}
		ev.Mix = mix
	}

	if *asJSON {
		payload := map[string]any{
			"schema": "fak.balance.v1",
			"status": ev.Status(),
			"resume": ev.Resume,
			"mix":    ev.Mix, // nil when the mix half was not measured — an honest null
		}
		if rc := encodeJSONOrFail(stdout, stderr, payload, "fak balance"); rc != 0 {
			return rc
		}
	} else {
		for _, row := range balance.Render(ev) {
			fmt.Fprintln(stdout, row)
		}
	}
	// The one hard alarm rides the exit code: a recovery budget underwater is a night-gate
	// failure, a mix lean is not. Everything else (ok / leaning / no-data) exits clean.
	if ev.Status() == "red" {
		return 1
	}
	return 0
}

// collectResumeStates walks the transcript store, folds each session's resume journey to
// its one state (the SAME foldStatusRow chain `fak resume status` uses, so the two can
// never disagree on where a session stands), and returns the states of the sessions that
// are budget subjects — those that crashed or carry any resume history. Clean, never-
// resumed sessions are not part of the recovery budget and are skipped, mirroring resume
// status's default filter.
func collectResumeStates(storeDir, ledgerPath string, maxAttempts int) ([]resume.ResumeState, error) {
	entries, err := os.ReadDir(storeDir)
	if err != nil {
		return nil, fmt.Errorf("read store %q: %w", storeDir, err)
	}
	history := loadResumeHistory(ledgerPath)
	admit := foldHostAdmission(ledgerPath)
	now := time.Now().Unix()
	var states []resume.ResumeState
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".jsonl") {
			continue
		}
		sid := strings.TrimSuffix(e.Name(), ".jsonl")
		f, err := os.Open(filepath.Join(storeDir, e.Name()))
		if err != nil {
			continue // a transcript we cannot read simply does not enter the fold
		}
		tr := scanTranscriptForStatus(f)
		f.Close()
		hist := history[sid]
		row := foldStatusRow(sid, tr, hist, admit, maxAttempts, now)
		if row.Crash == resume.CrashNone && len(hist) == 0 {
			continue // neither crashed nor part of any resume journey — not a budget subject
		}
		states = append(states, row.State)
	}
	return states, nil
}

// readWalkMix reads a `fak superloop walk --json` report (a WalkReport payload) from a
// file path or stdin ("-") and returns its work-mix split as the measured mix half. The
// mix is always present in a walk report, so a successfully-read report is always measured;
// only the ABSENCE of --walk leaves the mix half nil (unmeasured) upstream.
func readWalkMix(path string) (*superloop.WorkMix, error) {
	var data []byte
	var err error
	if path == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(pathutil.ExpandTilde(path))
	}
	if err != nil {
		return nil, err
	}
	var rep superloop.WalkReport
	if err := json.Unmarshal(data, &rep); err != nil {
		return nil, fmt.Errorf("not a superloop walk --json report: %w", err)
	}
	mix := rep.Mix
	return &mix, nil
}
