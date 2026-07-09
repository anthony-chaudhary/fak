package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// runFrontiersweCompare is the C12 compare verb (#1706, #1718): the surface that
// folds a raw arm and a fak arm into the single governed raw-vs-fak table a
// FrontierSWE completion-time row is recorded from. It enforces the runbook ordering
// in code — C11 score-parity first, then the C14 time-to-solution ratio, and only
// from solved trials under a passing gate. The verdict carries its own provenance
// (MEASURED_WIN vs PROJECTED_WIN) so a mocked run's projected floor can never be
// quoted as a measured win. RUNNABLE NOW over committed arm fixtures; the same verb
// consumes the real `run`+`eval` arm artifacts the moment a witnessed run lands.
func runFrontiersweCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rawPath := fs.String("raw", "", "the raw arm's graded-trials JSON (the harness run without fak) — required")
	fakPath := fs.String("fak", "", "the fak arm's graded-trials JSON (the same harness routed through fak) — required")
	taskName := fs.String("task", "", "task name for the report header (default: inferred from the first trial)")
	tolerance := fs.Float64("tolerance", 0, "score-parity tolerance (0 => the library default)")
	out := fs.String("out", "", "directory to capture compare.json + compare.md into (optional)")
	asJSON := fs.Bool("json", false, "emit only the compare JSON on stdout (no human summary on stderr)")
	if !parseFlags(fs, argv) {
		return 2
	}

	if *rawPath == "" || *fakPath == "" {
		fmt.Fprintln(stderr, "fak frontierswe compare: --raw and --fak graded-trials JSON are both required")
		return 2
	}

	raw, err := loadFrontiersweArm(*rawPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe compare: --raw: %v\n", err)
		return 1
	}
	fak, err := loadFrontiersweArm(*fakPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe compare: --fak: %v\n", err)
		return 1
	}

	task := *taskName
	if task == "" {
		task = inferFrontiersweTask(raw, fak)
	}

	rep := frontierswe.CompareArms(task, raw, fak, *tolerance)

	jb, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe compare: marshal: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(jb))
	if !*asJSON {
		printFrontiersweCompareSummary(stderr, rep)
	}

	if *out != "" {
		if err := writeFrontiersweCompareOut(*out, jb, rep); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe compare: --out: %v\n", err)
			return 1
		}
	}

	// A regressed parity gate is the one verdict that is a hard failure to act on;
	// every other verdict (WIN, NO_WIN, GATED) is an honest state, not an error.
	if rep.Verdict == frontierswe.VerdictParityFailed {
		return 1
	}
	return 0
}

// frontiersweArmFile is the on-disk shape of one arm: a graded-trials list. A bare
// JSON array of trials is also accepted (loadFrontiersweArm tries both), so an arm
// file can be either {"trials":[...]} or just [...].
type frontiersweArmFile struct {
	Trials []frontierswe.GradedTrial `json:"trials"`
}

// loadFrontiersweArm reads an arm's graded trials, accepting either the object form
// {"trials":[...]} or a bare [...] array.
func loadFrontiersweArm(path string) ([]frontierswe.GradedTrial, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var obj frontiersweArmFile
	if err := json.Unmarshal(data, &obj); err == nil && obj.Trials != nil {
		return obj.Trials, nil
	}
	var arr []frontierswe.GradedTrial
	if err := json.Unmarshal(data, &arr); err != nil {
		return nil, fmt.Errorf("not a graded-trials JSON (object with \"trials\" or a bare array): %w", err)
	}
	return arr, nil
}

// inferFrontiersweTask picks a report task name from the first trial carrying one.
func inferFrontiersweTask(raw, fak []frontierswe.GradedTrial) string {
	for _, arm := range [][]frontierswe.GradedTrial{raw, fak} {
		for _, t := range arm {
			if t.Score.Task != "" {
				return t.Score.Task
			}
		}
	}
	return ""
}

// writeFrontiersweCompareOut captures the compare JSON + the markdown table under
// dir for traceability (the artifact a FRONTIERSWE-RESULTS.md row is recorded from).
func writeFrontiersweCompareOut(dir string, jb []byte, rep frontierswe.CompareReport) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "compare.json"), jb, 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "compare.md"), []byte(frontierswe.RenderCompareMarkdown(rep)), 0o644)
}

// printFrontiersweCompareSummary writes the human-readable verdict on stderr,
// keeping stdout clean JSON. Parity and provenance are stated before the ratio so
// the summary reads in the honest order the gate enforces.
func printFrontiersweCompareSummary(w io.Writer, r frontierswe.CompareReport) {
	fmt.Fprintf(w, "\n== fak frontierswe compare (%s) ==\n", r.Schema)
	fmt.Fprintf(w, "task        : %s\n", r.Task)
	fmt.Fprintf(w, "parity      : passed=%t\n", r.Parity.Passed)
	if !r.Parity.Passed {
		for _, f := range r.Parity.Failures {
			fmt.Fprintf(w, "  - %s\n", f)
		}
	}
	fmt.Fprintf(w, "raw arm     : solved %d/%d, mean wall %.1fs (%s)\n", r.Raw.ReachedTrials, r.Raw.Trials, r.Raw.MeanWallSec, r.Raw.Provenance)
	fmt.Fprintf(w, "fak arm     : solved %d/%d, mean wall %.1fs (%s)\n", r.Fak.ReachedTrials, r.Fak.Trials, r.Fak.MeanWallSec, r.Fak.Provenance)
	if r.TTSRatio != nil {
		fmt.Fprintf(w, "TTS ratio   : %.4f  (%s)\n", *r.TTSRatio, r.Provenance)
	} else {
		fmt.Fprintf(w, "TTS ratio   : — (not claimed)\n")
	}
	fmt.Fprintf(w, "VERDICT     : %s\n", r.Verdict)
	fmt.Fprintf(w, "%s\n", r.Headline)
}
