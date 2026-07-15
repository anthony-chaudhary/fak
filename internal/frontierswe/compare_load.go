package frontierswe

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// This file pins the C12 ARM-RUN ARTIFACT CONTRACT (epic #1706, issue #1718): the
// named files an arm's run directory must carry for `fak frontierswe compare
// --raw-run/--fak-run DIR` to fold it into the governed raw-vs-fak table, and the
// loader that reads them. The contract is the join between the C9 run driver, the
// C13 grader, and this compare:
//
//	ArmMetaFile  (meta.json)      — the fak.frontierswe.run.v1 RunMeta `fak
//	                                frontierswe run --output DIR` writes: the run's
//	                                task identity and its Mocked flag, the C14
//	                                provenance bit (projected vs measured wall-clock)
//	                                the verdict vocabulary hangs on.
//	ArmTraceFile (tts-trace.json) — the per-turn TTSTrace the same run writes: the
//	                                C14 wall-clock/turn series plus the folded C8
//	                                reuse series (realized reuse rate).
//	ArmEvalFile  (eval.json)      — THE NAMED RESULTS FILE THAT CARRIES THE C3
//	                                SCORE: the fak.frontierswe.eval.v1 EvalResult,
//	                                whose `leaderboard_score` field is the gated C3
//	                                leaderboard number (with `correctness` and
//	                                `speedup` beside it). `fak frontierswe eval
//	                                --task <name> --out DIR` persists it into the
//	                                run directory (the C13 seam, #1719).
//
// The loader fabricates nothing: a directory missing an artifact is an error naming
// the missing file and the command that produces it; a GATED grade (available=false)
// is refused rather than folded in as a fabricated 0-score trial; and the Mocked
// provenance comes from the run meta the driver stamped, never from a default that
// could upgrade a projected floor into a measured win.

// The pinned artifact filenames of one arm run directory. These are the single
// source of truth for the run/eval/compare join — producers and this loader must
// agree on them by constant, not by convention.
const (
	ArmMetaFile  = "meta.json"      // fak.frontierswe.run.v1 RunMeta (task identity + Mocked provenance)
	ArmTraceFile = "tts-trace.json" // per-turn TTSTrace (C14 series + C8 reuse fold)
	ArmEvalFile  = "eval.json"      // fak.frontierswe.eval.v1 EvalResult — leaderboard_score IS the C3 score
)

// LoadArmRun loads one arm's run directory — the pinned meta.json + tts-trace.json
// + eval.json contract above — into the graded trials CompareArms consumes. One run
// directory carries one graded trial (the C9 driver is single-trajectory today); the
// slice shape leaves room for a multi-trial run layout without changing the caller.
// The trial ID is the directory's base name, so a compare report row is traceable
// back to the directory it was loaded from.
func LoadArmRun(dir string) ([]GradedTrial, error) {
	var ev EvalResult
	if err := readArmArtifact(dir, ArmEvalFile, &ev); err != nil {
		return nil, fmt.Errorf("%w (the %s results file whose leaderboard_score field carries the C3 score; produce it with `fak frontierswe eval --task <name> --out %s`)", err, EvalSchema, dir)
	}
	if ev.Schema != EvalSchema {
		return nil, fmt.Errorf("%s: schema %q, want %q — not a graded eval result", filepath.Join(dir, ArmEvalFile), ev.Schema, EvalSchema)
	}
	// An honest gate is a valid EVAL result but not a comparable trial: folding an
	// available=false grade in would fabricate a 0-score trial the arm never earned.
	if !ev.Available {
		return nil, fmt.Errorf("%s: the grade is GATED (available=false: %s) — no C3 score to compare; grade the arm first", filepath.Join(dir, ArmEvalFile), ev.Reason)
	}

	var trace TTSTrace
	if err := readArmArtifact(dir, ArmTraceFile, &trace); err != nil {
		return nil, fmt.Errorf("%w (the per-turn C14 TTS trace `fak frontierswe run --output %s` writes)", err, dir)
	}

	var meta RunMeta
	if err := readArmArtifact(dir, ArmMetaFile, &meta); err != nil {
		return nil, fmt.Errorf("%w (the %s meta carrying the run's mocked/measured provenance)", err, RunSchema)
	}

	// The join guard: an eval graded against one task cannot be folded under a run
	// meta from another — that would compare artifacts of different trials.
	if meta.Task != "" && ev.Task != "" && meta.Task != ev.Task {
		return nil, fmt.Errorf("%s: task mismatch: run meta is %q but eval graded %q — not the same trial's artifacts", dir, meta.Task, ev.Task)
	}

	return []GradedTrial{{
		Score: TrialScore{
			ID:            filepath.Base(dir),
			Task:          ev.Task,
			Correctness:   ev.Correctness,
			Speedup:       ev.Speedup,
			Score:         ev.Score,
			AntiCheatFlag: ev.AntiCheatFlag,
		},
		Trace:  trace,
		Mocked: meta.Mocked,
	}}, nil
}

// readArmArtifact reads and decodes one named artifact of the arm-run contract.
// A missing file surfaces the os error (which names the path); a malformed file
// surfaces a parse error naming the path — never a silently zeroed struct.
func readArmArtifact(dir, name string, v any) error {
	path := filepath.Join(dir, name)
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(b, v); err != nil {
		return fmt.Errorf("parse %s: %w", path, err)
	}
	return nil
}
