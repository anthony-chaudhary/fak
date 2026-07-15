package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/frontierswe"
)

// writeCompareArmRunDir writes one arm's run directory in the pinned artifact
// contract (meta.json + tts-trace.json + eval.json) via the real contract structs,
// so the CLI test witnesses the same round-trip `run`+`eval` produce.
func writeCompareArmRunDir(t *testing.T, dir string, wallSec float64, turns int, reuse float64, mocked bool) {
	t.Helper()
	meta := frontierswe.RunMeta{
		Schema: frontierswe.RunSchema, Task: "git-to-zig", Agent: "claude-code",
		Model: "test-model", BudgetSec: 3600, Turns: turns, ElapsedSec: wallSec, Mocked: mocked,
	}
	trace := frontierswe.TTSTrace{
		Schema: "fak.frontierswe.tts-trace.v1", Turns: turns, TotalWallSec: wallSec, BudgetSec: 3600,
		CacheSeries: frontierswe.CacheWitnessSeries{Schema: frontierswe.CacheWitnessSchema, RealizedReuseRate: reuse},
	}
	ev := frontierswe.EvalResult{
		Schema: frontierswe.EvalSchema, Task: "git-to-zig", GateClass: "implementation",
		Available: true, Source: "existing-reward", Correctness: 1.0, Score: 1.0,
	}
	for name, v := range map[string]any{
		frontierswe.ArmMetaFile:  meta,
		frontierswe.ArmTraceFile: trace,
		frontierswe.ArmEvalFile:  ev,
	} {
		if err := writeJSONFile(filepath.Join(dir, name), v); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
}

// TestFrontiersweCompareFromRunDirs is the #1718 CLI acceptance: `fak frontierswe
// compare --task NAME --raw-run DIR --fak-run DIR --out DIR --md FILE` folds two
// run directories' pinned artifacts into the compare JSON + the two markdown/JSON
// capture paths, with the score/TTS/reuse/floor numbers tracing to eval.json (C3),
// tts-trace.json (C14/C8), and the budget (C4).
func TestFrontiersweCompareFromRunDirs(t *testing.T) {
	rawDir, fakDir, outDir := t.TempDir(), t.TempDir(), t.TempDir()
	writeCompareArmRunDir(t, rawDir, 1000, 40, 0.80, false)
	writeCompareArmRunDir(t, fakDir, 250, 10, 0.90, false)
	mdPath := filepath.Join(outDir, "table.md")

	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"compare", "--task", "git-to-zig",
		"--raw-run", rawDir, "--fak-run", fakDir,
		"--out", outDir, "--md", mdPath, "--json",
	})
	if code != 0 {
		t.Fatalf("compare exit = %d, want 0\nstderr:\n%s", code, stderr.String())
	}

	var rep frontierswe.CompareReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("stdout is not compare JSON: %v\n%s", err, stdout.String())
	}
	if rep.Task != "git-to-zig" || rep.Verdict != frontierswe.VerdictMeasuredWin {
		t.Fatalf("unexpected report: task=%q verdict=%q", rep.Task, rep.Verdict)
	}
	if rep.TTSRatio == nil || *rep.TTSRatio != 0.25 {
		t.Fatalf("expected TTS ratio 0.25 (250s/1000s), got %v", rep.TTSRatio)
	}
	if rep.FloorRatio == nil || rep.OverClaim {
		t.Fatalf("expected a C4 floor without over-claim: floor=%v overclaim=%t", rep.FloorRatio, rep.OverClaim)
	}

	// The capture artifacts: --out's compare.json + compare.md, and --md's named file.
	for _, p := range []string{filepath.Join(outDir, "compare.json"), filepath.Join(outDir, "compare.md"), mdPath} {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("missing capture artifact %s: %v", p, err)
		}
	}
	md, err := os.ReadFile(mdPath)
	if err != nil {
		t.Fatalf("read --md file: %v", err)
	}
	for _, want := range []string{"avg score", "realized reuse", "C4 floor", "TTS ratio"} {
		if !strings.Contains(string(md), want) {
			t.Errorf("--md table must surface %q; got:\n%s", want, md)
		}
	}
}

// A run directory that was never graded (no eval.json) fails honestly, naming the
// pinned results file and the eval command — never folding in a fabricated score.
func TestFrontiersweCompareRunDirMissingEval(t *testing.T) {
	rawDir, fakDir := t.TempDir(), t.TempDir()
	writeCompareArmRunDir(t, rawDir, 1000, 40, 0.80, false)
	writeCompareArmRunDir(t, fakDir, 250, 10, 0.90, false)
	if err := os.Remove(filepath.Join(fakDir, frontierswe.ArmEvalFile)); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runFrontierswe(&stdout, &stderr, []string{
		"compare", "--raw-run", rawDir, "--fak-run", fakDir, "--json",
	})
	if code != 1 {
		t.Fatalf("compare exit = %d, want 1\nstderr:\n%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), frontierswe.ArmEvalFile) || !strings.Contains(stderr.String(), "fak frontierswe eval") {
		t.Fatalf("error must name the missing %s and the eval command; got:\n%s", frontierswe.ArmEvalFile, stderr.String())
	}
}

// Arm flags are exactly-one-of: an arm with both a FILE and a DIR is ambiguous, and
// an arm with neither is a usage error (exit 2, not a load failure).
func TestFrontiersweCompareArmFlagExclusivity(t *testing.T) {
	dir := t.TempDir()
	writeCompareArmRunDir(t, dir, 100, 4, 0.5, true)

	var so, se bytes.Buffer
	if code := runFrontierswe(&so, &se, []string{"compare", "--raw", "x.json", "--raw-run", dir, "--fak-run", dir}); code != 2 {
		t.Fatalf("both --raw and --raw-run: exit = %d, want 2\nstderr:\n%s", code, se.String())
	}
	so.Reset()
	se.Reset()
	if code := runFrontierswe(&so, &se, []string{"compare", "--raw-run", dir}); code != 2 {
		t.Fatalf("missing fak arm: exit = %d, want 2\nstderr:\n%s", code, se.String())
	}
	if !strings.Contains(se.String(), "--fak-run") {
		t.Fatalf("usage error must name both arm flags; got:\n%s", se.String())
	}
}
