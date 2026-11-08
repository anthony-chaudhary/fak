package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
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

const rtx5090EnvelopeID = "qwen38-27b-q4km-rtx5090-p1-decode"

// rtx5090ReceiptPairFile writes a minimal valid baseline/candidate nativeperf
// receipt pair for a committed RTX 5090 lever as one JSON array file.
func rtx5090ReceiptPairFile(t *testing.T, dir, name, lever string, baseTPS, candTPS float64) string {
	t.Helper()
	mk := func(role string, tps float64) nativeperf.ExperimentReceipt {
		r, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), lever)
		if err != nil {
			t.Fatal(err)
		}
		r.Role = role
		r.Revision = "rev-" + role
		r.Machine.ScrubbedID = "lab-node-class-a"
		r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
		r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
		r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r1+gaaaaaaa"}}
		r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
		r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
		for i := range r.Repetitions {
			r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: tps}
		}
		if role == nativeperf.RoleCandidate {
			r.ChangedAxes = []string{"lever:" + lever}
		}
		return r
	}
	pair := []nativeperf.ExperimentReceipt{mk(nativeperf.RoleBaseline, baseTPS), mk(nativeperf.RoleCandidate, candTPS)}
	raw, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// plan --envelope restricts selection to the pinned envelope and emits the
// follow-on commands with the --envelope flag carried through.
func TestOptimizePlanEnvelopeJSON(t *testing.T) {
	code, out, errb := runOptimizeTest("plan", "--envelope", rtx5090EnvelopeID, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var plan struct {
		LeverID         string `json:"lever_id"`
		Envelope        string `json:"envelope"`
		DependencyReady bool   `json:"dependency_ready"`
		MeasureCmd      string `json:"measure_cmd"`
		VerifyCmd       string `json:"verify_cmd"`
	}
	if err := json.Unmarshal([]byte(out), &plan); err != nil {
		t.Fatalf("json: %v\n%s", err, out)
	}
	if plan.LeverID != "rtx5090.q8_1-activation-quant" || plan.Envelope != rtx5090EnvelopeID || !plan.DependencyReady {
		t.Fatalf("envelope-filtered plan drifted: %+v", plan)
	}
	if !strings.Contains(plan.MeasureCmd, "--envelope "+rtx5090EnvelopeID) || !strings.Contains(plan.VerifyCmd, "--envelope "+rtx5090EnvelopeID) {
		t.Fatalf("follow-on commands lost the envelope pin: measure=%q verify=%q", plan.MeasureCmd, plan.VerifyCmd)
	}
}

// plan with an envelope no graph row or lever carries refuses typed (exit 1,
// the plan sub-verb's typed-refusal code).
func TestOptimizePlanUnknownEnvelope(t *testing.T) {
	code, out, _ := runOptimizeTest("plan", "--envelope", "no-such-envelope", "--json")
	if code != 1 {
		t.Fatalf("exit=%d, want 1\nstdout=%s", code, out)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("refusal json: %v\n%s", err, out)
	}
	if body.Error != "unknown_envelope" {
		t.Fatalf("refusal class = %q, want unknown_envelope", body.Error)
	}
}

// loop --envelope dry-run plans only inside the pinned envelope.
func TestOptimizeLoopEnvelopeDryRun(t *testing.T) {
	code, out, errb := runOptimizeTest("loop", "--max-iters", "1", "--dry-run", "--envelope", rtx5090EnvelopeID, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) != 1 {
		t.Fatalf("want one ledger line, got %d: %s", len(lines), out)
	}
	var line struct {
		LeverID string `json:"lever_id"`
		Verdict string `json:"verdict"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &line); err != nil {
		t.Fatalf("ledger line json: %v\n%s", err, lines[0])
	}
	if line.LeverID != "rtx5090.q8_1-activation-quant" || line.Verdict != "planned" {
		t.Fatalf("ledger line = %+v", line)
	}
}

// measure --trace and --receipt together refuse typed (exit 2).
func TestOptimizeMeasureReceiptMutualExclusion(t *testing.T) {
	code, out, _ := runOptimizeTest("measure", "--lever", "rtx5090.q8_1-activation-quant", "--trace", "t.json", "--receipt", "r.json", "--json")
	if code != 2 {
		t.Fatalf("exit=%d, want 2\nstdout=%s", code, out)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("refusal json: %v\n%s", err, out)
	}
	if body.Error != "invalid_input" {
		t.Fatalf("refusal class = %q, want invalid_input", body.Error)
	}
}

// measure --receipt folds a matched nativeperf A/B pair into the receipt.
func TestOptimizeMeasureReceiptAcceptsPair(t *testing.T) {
	dir := t.TempDir()
	path := rtx5090ReceiptPairFile(t, dir, "pair.json", "rtx5090.q8_1-activation-quant", 100, 105)
	code, out, errb := runOptimizeTest("measure", "--lever", "rtx5090.q8_1-activation-quant", "--receipt", path, "--json")
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	var rec struct {
		Schema        string  `json:"schema"`
		LeverID       string  `json:"lever_id"`
		EngineID      string  `json:"engine_id"`
		BaselineTokS  float64 `json:"baseline_tok_s"`
		CandidateTokS float64 `json:"candidate_tok_s"`
		DeltaPct      float64 `json:"delta_pct"`
		Reps          int     `json:"reps"`
		Revision      string  `json:"revision"`
		ArtifactSHA   string  `json:"artifact_sha256"`
		ForwardPath   string  `json:"forward_path"`
	}
	if err := json.Unmarshal([]byte(out), &rec); err != nil {
		t.Fatalf("receipt json: %v\n%s", err, out)
	}
	if rec.Schema != "fak-optimize-receipt/v1" || rec.LeverID != "rtx5090.q8_1-activation-quant" || rec.EngineID != "fak-native" {
		t.Fatalf("receipt identity drifted: %+v", rec)
	}
	if rec.BaselineTokS != 100 || rec.CandidateTokS != 105 || rec.DeltaPct != 5 || rec.Reps != 3 {
		t.Fatalf("arm means drifted: %+v", rec)
	}
	if rec.Revision != "rev-candidate" || rec.ArtifactSHA != "issue-8635-pinned-artifact" || rec.ForwardPath != "cuda/q4km-p1-decode" {
		t.Fatalf("execution identity drifted: %+v", rec)
	}
}

// measure --receipt against a pair that targets another lever refuses typed.
func TestOptimizeMeasureReceiptLeverMismatch(t *testing.T) {
	dir := t.TempDir()
	path := rtx5090ReceiptPairFile(t, dir, "pair.json", "rtx5090.dp4a-q4k-mmvq", 100, 105)
	code, out, _ := runOptimizeTest("measure", "--lever", "rtx5090.q8_1-activation-quant", "--receipt", path, "--json")
	if code != 2 {
		t.Fatalf("exit=%d, want 2\nstdout=%s", code, out)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("refusal json: %v\n%s", err, out)
	}
	if body.Error != "receipt_lever_mismatch" {
		t.Fatalf("refusal class = %q, want receipt_lever_mismatch", body.Error)
	}
}

// measure --receipt against a pair stamped with another envelope refuses typed.
func TestOptimizeMeasureReceiptEnvelopeMismatch(t *testing.T) {
	dir := t.TempDir()
	lever := "rtx5090.q8_1-activation-quant"
	mk := func(role string, tps float64) nativeperf.ExperimentReceipt {
		r, err := nativeperf.BaselineTemplate(nativeperf.ActiveGraph(), lever)
		if err != nil {
			t.Fatal(err)
		}
		r.Role = role
		r.EnvelopeID = "qwen38-27b-q4k-a100-p1-decode"
		r.Revision = "rev-" + role
		r.Machine.ScrubbedID = "lab-node-class-a"
		r.Memory = nativeperf.MemoryMetrics{PeakBytes: 1000, ResidentBytes: 900}
		r.Quality = nativeperf.QualityMetric{Name: "exact_match", Score: 1, HigherIsBetter: true}
		r.ModuleVersions = []nativeperf.ModuleRevision{{Module: "internal/model", Revision: "r1+gaaaaaaa"}}
		r.Commands = []string{"fak run-model --native --receipt-out receipt.json"}
		r.ProfilerArtifacts = []nativeperf.ArtifactRef{{Path: "profiles/run.json", SHA256: strings.Repeat("a", 64)}}
		for i := range r.Repetitions {
			r.Repetitions[i] = nativeperf.Repetition{EndToEndMilliseconds: 100, TokensPerSecond: tps}
		}
		if role == nativeperf.RoleCandidate {
			r.ChangedAxes = []string{"lever:" + lever}
		}
		return r
	}
	pair := []nativeperf.ExperimentReceipt{mk(nativeperf.RoleBaseline, 100), mk(nativeperf.RoleCandidate, 105)}
	raw, err := json.Marshal(pair)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "pair.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	code, out, _ := runOptimizeTest("measure", "--lever", lever, "--receipt", path, "--json")
	if code != 2 {
		t.Fatalf("exit=%d, want 2\nstdout=%s", code, out)
	}
	var body struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &body); err != nil {
		t.Fatalf("refusal json: %v\n%s", err, out)
	}
	if body.Error != "receipt_envelope_mismatch" {
		t.Fatalf("refusal class = %q, want receipt_envelope_mismatch", body.Error)
	}
}
