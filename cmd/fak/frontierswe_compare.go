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
// quoted as a measured win.
//
// Each arm is given in one of two forms: --raw/--fak FILE, a graded-trials JSON
// (committed fixtures, RUNNABLE NOW), or --raw-run/--fak-run DIR, a run DIRECTORY
// carrying the pinned run+eval artifact contract (frontierswe.LoadArmRun): meta.json
// (the fak.frontierswe.run.v1 identity + mocked provenance), tts-trace.json (the C14
// per-turn trace + C8 reuse series), and eval.json — the fak.frontierswe.eval.v1
// grade whose leaderboard_score field carries the C3 score (`fak frontierswe eval
// --out DIR` persists it, #1719).
func runFrontiersweCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("frontierswe compare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	rawPath := fs.String("raw", "", "the raw arm's graded-trials JSON (the harness run without fak)")
	fakPath := fs.String("fak", "", "the fak arm's graded-trials JSON (the same harness routed through fak)")
	rawRun := fs.String("raw-run", "", "the raw arm's run DIRECTORY (the pinned run+eval artifact contract: "+frontierswe.ArmMetaFile+" + "+frontierswe.ArmTraceFile+" + "+frontierswe.ArmEvalFile+")")
	fakRun := fs.String("fak-run", "", "the fak arm's run DIRECTORY (same contract as --raw-run)")
	taskName := fs.String("task", "", "task name for the report header (default: inferred from the first trial)")
	tolerance := fs.Float64("tolerance", 0, "score-parity tolerance (0 => the library default)")
	out := fs.String("out", "", "directory to capture compare.json + compare.md into (optional)")
	mdPath := fs.String("md", "", "write the raw-vs-fak markdown table to this FILE (optional; independent of --out)")
	asJSON := fs.Bool("json", false, "emit only the compare JSON on stdout (no human summary on stderr)")
	if !parseFlags(fs, argv) {
		return 2
	}

	if code := checkFrontiersweArmFlags(stderr, "raw", *rawPath, *rawRun); code != 0 {
		return code
	}
	if code := checkFrontiersweArmFlags(stderr, "fak", *fakPath, *fakRun); code != 0 {
		return code
	}

	raw, err := loadFrontiersweCompareArm(*rawPath, *rawRun)
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe compare: raw arm: %v\n", err)
		return 1
	}
	fak, err := loadFrontiersweCompareArm(*fakPath, *fakRun)
	if err != nil {
		fmt.Fprintf(stderr, "fak frontierswe compare: fak arm: %v\n", err)
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
	if *mdPath != "" {
		if err := os.WriteFile(*mdPath, []byte(frontierswe.RenderCompareMarkdown(rep)), 0o644); err != nil {
			fmt.Fprintf(stderr, "fak frontierswe compare: --md: %v\n", err)
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

// checkFrontiersweArmFlags enforces exactly-one-of --<arm> FILE / --<arm>-run DIR:
// an arm with neither source is a usage error, and both at once is ambiguous.
func checkFrontiersweArmFlags(stderr io.Writer, arm, file, runDir string) int {
	switch {
	case file == "" && runDir == "":
		fmt.Fprintf(stderr, "fak frontierswe compare: the %s arm is required: --%s FILE (graded-trials JSON) or --%s-run DIR (the run+eval artifact directory)\n", arm, arm, arm)
		return 2
	case file != "" && runDir != "":
		fmt.Fprintf(stderr, "fak frontierswe compare: give --%s or --%s-run, not both\n", arm, arm)
		return 2
	}
	return 0
}

// loadFrontiersweCompareArm resolves one arm from whichever source was given: a
// graded-trials JSON file, or a run directory carrying the pinned artifact contract
// (frontierswe.LoadArmRun). Exactly one is set by the time this is called.
func loadFrontiersweCompareArm(file, runDir string) ([]frontierswe.GradedTrial, error) {
	if runDir != "" {
		return frontierswe.LoadArmRun(runDir)
	}
	return loadFrontiersweArm(file)
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
	fmt.Fprintf(w, "raw arm     : score avg %.3f/best %.3f, correct %d/%d, solved %d/%d, mean wall %.1fs, reuse r=%.4f (%s)\n",
		r.Parity.Raw.AvgScore, r.Parity.Raw.BestScore, r.Parity.Raw.CorrectCount, r.Parity.Raw.Trials,
		r.Raw.ReachedTrials, r.Raw.Trials, r.Raw.MeanWallSec, r.Raw.MeanReuseRate, r.Raw.Provenance)
	fmt.Fprintf(w, "fak arm     : score avg %.3f/best %.3f, correct %d/%d, solved %d/%d, mean wall %.1fs, reuse r=%.4f (%s)\n",
		r.Parity.Fak.AvgScore, r.Parity.Fak.BestScore, r.Parity.Fak.CorrectCount, r.Parity.Fak.Trials,
		r.Fak.ReachedTrials, r.Fak.Trials, r.Fak.MeanWallSec, r.Fak.MeanReuseRate, r.Fak.Provenance)
	if r.TTSRatio != nil {
		fmt.Fprintf(w, "TTS ratio   : %.4f  (%s)\n", *r.TTSRatio, r.Provenance)
	} else {
		fmt.Fprintf(w, "TTS ratio   : — (not claimed)\n")
	}
	if r.FloorRatio != nil {
		flag := ""
		if r.OverClaim {
			flag = "  ** OVER-CLAIM vs floor **"
		}
		fmt.Fprintf(w, "C4 floor    : %.4f (the projection the measurement is checked against)%s\n", *r.FloorRatio, flag)
	}
	fmt.Fprintf(w, "VERDICT     : %s\n", r.Verdict)
	fmt.Fprintf(w, "%s\n", r.Headline)
}
