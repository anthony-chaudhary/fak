package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// runOptimizeTest drives the testable runOptimize core with captured streams.
func runOptimizeTest(args ...string) (int, string, string) {
	var out, errb bytes.Buffer
	code := runOptimize(&out, &errb, args)
	return code, out.String(), errb.String()
}

// OptimizePlan --json emits the OptimizePlan for the committed graph's next ready
// lever, with the exact follow-on commands and the committed planning band.
func TestOptimizePlanJSON(t *testing.T) {
	code, out, errb := runOptimizeTest("plan", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var OptimizePlan struct {
		Schema          string   `json:"schema"`
		LeverID         string   `json:"lever_id"`
		Envelope        string   `json:"envelope"`
		ExpectedFloor   *float64 `json:"expected_floor"`
		DependencyReady bool     `json:"dependency_ready"`
		MeasureCmd      string   `json:"measure_cmd"`
		VerifyCmd       string   `json:"verify_cmd"`
	}
	if err := json.Unmarshal([]byte(out), &OptimizePlan); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if OptimizePlan.Schema != "fak-optimize-plan/v1" || OptimizePlan.LeverID != "metal.command-buffer-amortization" {
		t.Fatalf("OptimizePlan identity drifted: %+v", OptimizePlan)
	}
	if !OptimizePlan.DependencyReady {
		t.Fatalf("selected lever must be dependency-ready: %+v", OptimizePlan)
	}
	if OptimizePlan.ExpectedFloor == nil || *OptimizePlan.ExpectedFloor != 5 {
		t.Fatalf("expected floor = %v, want 5", OptimizePlan.ExpectedFloor)
	}
	if !strings.Contains(OptimizePlan.MeasureCmd, "fak optimize measure --lever metal.command-buffer-amortization") {
		t.Fatalf("measure_cmd = %q", OptimizePlan.MeasureCmd)
	}
	if !strings.Contains(OptimizePlan.VerifyCmd, "fak optimize verify --baseline") {
		t.Fatalf("verify_cmd = %q", OptimizePlan.VerifyCmd)
	}
}

// loop --dry-run stops after the first plan: one JSONL ledger line, no measurement.
func TestOptimizeLoopDryRun(t *testing.T) {
	code, out, errb := runOptimizeTest("loop", "--max-iters", "1", "--dry-run", "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 1 {
		t.Fatalf("want one ledger line, got %d: %s", len(lines), out)
	}
	var line struct {
		Iter      int      `json:"iter"`
		LeverID   string   `json:"lever_id"`
		Metric    string   `json:"metric"`
		Verdict   string   `json:"verdict"`
		Baseline  *float64 `json:"baseline"`
		Candidate *float64 `json:"candidate"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &line); err != nil {
		t.Fatalf("ledger line json: %v\n%s", err, lines[0])
	}
	if line.Iter != 1 || line.Verdict != "planned" || line.Metric != "decode_tok_s" || line.LeverID != "metal.command-buffer-amortization" {
		t.Fatalf("ledger line = %+v", line)
	}
	if line.Baseline != nil || line.Candidate != nil {
		t.Fatalf("dry-run line must not carry throughput fields: %+v", line)
	}
}

// measure refuses with the closed mapping class (exit 2) for a committed
// hardware lever: the OptimizeRefusal is structured, not a crash or a fabricated number.
func TestOptimizeMeasureRefusal(t *testing.T) {
	code, out, _ := runOptimizeTest("measure", "--lever", "metal.command-buffer-amortization", "--trace", filepath.Join(t.TempDir(), "absent.json"), "--json")
	if code != 2 {
		t.Fatalf("exit=%d, want 2\nstdout=%s", code, out)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("OptimizeRefusal json: %v\n%s", err, out)
	}
	if body.Error != "no_arm_mapping" {
		t.Fatalf("OptimizeRefusal class = %q, want no_arm_mapping", body.Error)
	}
}

// loop refuses non-positive iteration counts with the closed class (exit 2).
func TestOptimizeLoopRefusesNonPositive(t *testing.T) {
	code, out, _ := runOptimizeTest("loop", "--max-iters", "0")
	if code != 2 {
		t.Fatalf("exit=%d, want 2\nstdout=%s", code, out)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("OptimizeRefusal json: %v\n%s", err, out)
	}
	if body.Error != "non_positive_iterations" {
		t.Fatalf("OptimizeRefusal class = %q", body.Error)
	}
}

// verify exits 0 on accept and 1 on reject under the committed planning band.
func TestOptimizeVerifyExitCodes(t *testing.T) {
	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.json")
	candPath := filepath.Join(dir, "cand.json")
	base := `{"schema":"fak-optimize-receipt/v1","lever_id":"metal.command-buffer-amortization","engine_id":"mock","trace_path":"testdata/tau2/tau2-smoke.json","trace_sha256":"` + strings.Repeat("a", 64) + `","baseline_tok_s":3,"candidate_tok_s":3,"delta_pct":0,"reps":1,"measured_at":"2026-11-07T00:00:00Z"}`
	cand := func(tokS float64) string {
		return `{"schema":"fak-optimize-receipt/v1","lever_id":"metal.command-buffer-amortization","engine_id":"mock","trace_path":"testdata/tau2/tau2-smoke.json","trace_sha256":"` + strings.Repeat("a", 64) + `","baseline_tok_s":3,"candidate_tok_s":` + strconv.FormatFloat(tokS, 'f', -1, 64) + `,"delta_pct":0,"reps":1,"measured_at":"2026-11-07T00:00:00Z"}`
	}
	if err := os.WriteFile(basePath, []byte(base), 0o644); err != nil {
		t.Fatal(err)
	}

	t.Run("accept", func(t *testing.T) {
		if err := os.WriteFile(candPath, []byte(cand(6)), 0o644); err != nil {
			t.Fatal(err)
		}
		code, out, errb := runOptimizeTest("verify", "--baseline", basePath, "--candidate", candPath, "--json")
		if code != 0 {
			t.Fatalf("exit=%d stderr=%s stdout=%s", code, errb, out)
		}
		var v struct {
			Verdict string `json:"verdict"`
			Quality string `json:"quality"`
		}
		if err := json.Unmarshal([]byte(out), &v); err != nil {
			t.Fatalf("OptimizeVerdict json: %v\n%s", err, out)
		}
		if v.Verdict != "accept" || v.Quality != "absent" {
			t.Fatalf("OptimizeVerdict = %+v", v)
		}
	})
	t.Run("reject below floor", func(t *testing.T) {
		if err := os.WriteFile(candPath, []byte(cand(4)), 0o644); err != nil {
			t.Fatal(err)
		}
		code, out, _ := runOptimizeTest("verify", "--baseline", basePath, "--candidate", candPath, "--json")
		if code != 1 {
			t.Fatalf("exit=%d, want 1\nstdout=%s", code, out)
		}
		var v struct {
			Verdict string `json:"verdict"`
		}
		if err := json.Unmarshal([]byte(out), &v); err != nil {
			t.Fatalf("OptimizeVerdict json: %v\n%s", err, out)
		}
		if v.Verdict != "reject" {
			t.Fatalf("OptimizeVerdict = %+v", v)
		}
	})
}

// Unknown subcommands and bare invocations are usage errors (exit 2).
func TestOptimizeUsageErrors(t *testing.T) {
	for _, args := range [][]string{{}, {"frobnicate"}, {"measure"}, {"verify"}} {
		code, _, _ := runOptimizeTest(args...)
		if code != 2 {
			t.Fatalf("args=%v: exit=%d, want 2", args, code)
		}
	}
}
