package frontierswe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The committed arm-run fixtures pin the ON-DISK artifact contract (raw JSON bytes,
// not Go round-trips): meta.json + tts-trace.json + eval.json per arm directory.
const (
	armRunRawFixture = "testdata/frontierswe/armrun/raw"
	armRunFakFixture = "testdata/frontierswe/armrun/fak"
)

// writeArmRun writes an arm-run directory from the contract structs, for the
// error-path tests that need a deliberately broken directory.
func writeArmRun(t *testing.T, dir string, meta *RunMeta, trace *TTSTrace, ev *EvalResult) {
	t.Helper()
	write := func(name string, v any) {
		t.Helper()
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal %s: %v", name, err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	if meta != nil {
		write(ArmMetaFile, meta)
	}
	if trace != nil {
		write(ArmTraceFile, trace)
	}
	if ev != nil {
		write(ArmEvalFile, ev)
	}
}

func armRunMeta(task string, mocked bool) *RunMeta {
	return &RunMeta{Schema: RunSchema, Task: task, BudgetSec: 3600, Mocked: mocked}
}

func armRunTrace(wallSec float64, turns int) *TTSTrace {
	tr := traceOf(wallSec, turns)
	return &tr
}

func armRunEval(task string, correctness, score float64) *EvalResult {
	return &EvalResult{Schema: EvalSchema, Task: task, Available: true, Correctness: correctness, Score: score}
}

// #1718: the loader reads the pinned artifact contract — eval.json's
// leaderboard_score is the C3 score, tts-trace.json is the C14 trace, meta.json's
// mocked flag is the provenance — into one graded trial whose ID names the directory.
func TestLoadArmRunReadsPinnedContract(t *testing.T) {
	trials, err := LoadArmRun(armRunRawFixture)
	if err != nil {
		t.Fatalf("LoadArmRun(%s): %v", armRunRawFixture, err)
	}
	if len(trials) != 1 {
		t.Fatalf("one run dir must load one graded trial, got %d", len(trials))
	}
	tr := trials[0]
	if tr.Score.ID != "raw" || tr.Score.Task != "git-to-zig" {
		t.Fatalf("trial identity wrong: id=%q task=%q", tr.Score.ID, tr.Score.Task)
	}
	if !floatsEqual(tr.Score.Score, 1.0) || !floatsEqual(tr.Score.Correctness, 1.0) {
		t.Fatalf("C3 score not carried from eval.json leaderboard_score: %+v", tr.Score)
	}
	if !floatsEqual(tr.Trace.TotalWallSec, 1000) || tr.Trace.Turns != 40 {
		t.Fatalf("C14 trace not carried from tts-trace.json: wall=%v turns=%d", tr.Trace.TotalWallSec, tr.Trace.Turns)
	}
	if !floatsEqual(tr.Trace.CacheSeries.RealizedReuseRate, 0.80) {
		t.Fatalf("C8 realized reuse not carried: %v", tr.Trace.CacheSeries.RealizedReuseRate)
	}
	if tr.Mocked {
		t.Fatalf("meta.json says mocked=false; provenance must not be upgraded to projected")
	}
}

// #1718 end-to-end over the committed fixtures: two loaded arm-run directories fold
// through CompareArms into a MEASURED_WIN with the ratio, reuse, and C4 floor —
// numbers tracing to eval.json (C3), tts-trace.json (C14/C8), and the budget (C4).
func TestLoadArmRunCompareEndToEnd(t *testing.T) {
	raw, err := LoadArmRun(armRunRawFixture)
	if err != nil {
		t.Fatalf("raw arm: %v", err)
	}
	fak, err := LoadArmRun(armRunFakFixture)
	if err != nil {
		t.Fatalf("fak arm: %v", err)
	}
	rep := CompareArms("git-to-zig", raw, fak, 0)
	if rep.Verdict != VerdictMeasuredWin {
		t.Fatalf("expected MEASURED_WIN, got %s (headline=%q)", rep.Verdict, rep.Headline)
	}
	if rep.TTSRatio == nil || !floatsEqual(*rep.TTSRatio, 0.25) {
		t.Fatalf("expected ratio 0.25 (250s/1000s), got %v", rep.TTSRatio)
	}
	if !floatsEqual(rep.Raw.MeanReuseRate, 0.80) || !floatsEqual(rep.Fak.MeanReuseRate, 0.90) {
		t.Fatalf("realized reuse must flow from the traces: raw=%v fak=%v", rep.Raw.MeanReuseRate, rep.Fak.MeanReuseRate)
	}
	if rep.FloorRatio == nil || rep.OverClaim {
		t.Fatalf("budgeted arms must carry a C4 floor without over-claim: floor=%v overclaim=%t", rep.FloorRatio, rep.OverClaim)
	}
}

// The contract fails closed: a run directory without the named C3 results file is an
// error that names eval.json and the command that produces it — never a 0-score trial.
func TestLoadArmRunMissingEvalNamesContract(t *testing.T) {
	dir := t.TempDir()
	writeArmRun(t, dir, armRunMeta("git-to-zig", false), armRunTrace(100, 4), nil)
	_, err := LoadArmRun(dir)
	if err == nil {
		t.Fatalf("missing %s must be an error", ArmEvalFile)
	}
	for _, want := range []string{ArmEvalFile, "C3", "fak frontierswe eval"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error must name %q; got: %v", want, err)
		}
	}
}

// A GATED grade (available=false) is a valid eval artifact but not a comparable
// trial: folding it in would fabricate a 0-score the arm never earned.
func TestLoadArmRunRefusesGatedEval(t *testing.T) {
	dir := t.TempDir()
	ev := &EvalResult{Schema: EvalSchema, Task: "git-to-zig", Available: false, Reason: EvalGatedReason}
	writeArmRun(t, dir, armRunMeta("git-to-zig", false), armRunTrace(100, 4), ev)
	_, err := LoadArmRun(dir)
	if err == nil || !strings.Contains(err.Error(), "GATED") {
		t.Fatalf("a gated eval must be refused, got: %v", err)
	}
}

// The eval artifact must be the versioned results file, not any JSON that happens
// to sit at the pinned name.
func TestLoadArmRunRejectsWrongEvalSchema(t *testing.T) {
	dir := t.TempDir()
	ev := armRunEval("git-to-zig", 1.0, 1.0)
	ev.Schema = "not-an-eval"
	writeArmRun(t, dir, armRunMeta("git-to-zig", false), armRunTrace(100, 4), ev)
	_, err := LoadArmRun(dir)
	if err == nil || !strings.Contains(err.Error(), EvalSchema) {
		t.Fatalf("wrong schema must be refused naming %s, got: %v", EvalSchema, err)
	}
}

// The join guard: a run meta from one task cannot be folded under an eval graded
// against another — those are different trials' artifacts.
func TestLoadArmRunRejectsTaskMismatch(t *testing.T) {
	dir := t.TempDir()
	writeArmRun(t, dir, armRunMeta("cranelift", false), armRunTrace(100, 4), armRunEval("git-to-zig", 1.0, 1.0))
	_, err := LoadArmRun(dir)
	if err == nil || !strings.Contains(err.Error(), "task mismatch") {
		t.Fatalf("task mismatch must be refused, got: %v", err)
	}
}

// Provenance flows from meta.json: a mocked run loads as a projected trial, so the
// downstream verdict can never dress the floor up as a measured win.
func TestLoadArmRunCarriesMockedProvenance(t *testing.T) {
	dir := t.TempDir()
	writeArmRun(t, dir, armRunMeta("git-to-zig", true), armRunTrace(100, 4), armRunEval("git-to-zig", 1.0, 1.0))
	trials, err := LoadArmRun(dir)
	if err != nil {
		t.Fatalf("LoadArmRun: %v", err)
	}
	if !trials[0].Mocked {
		t.Fatalf("meta.json mocked=true must flow into the graded trial")
	}
}
