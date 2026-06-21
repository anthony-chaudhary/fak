package swebench

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetectEvalCapabilityShape(t *testing.T) {
	cap := DetectEvalCapability("")
	// Runnable iff both Docker and the harness are present; otherwise a Reason.
	if cap.Runnable != (cap.DockerPresent && cap.HarnessPresent) {
		t.Errorf("Runnable inconsistent: %+v", cap)
	}
	if !cap.Runnable && cap.Reason == "" {
		t.Errorf("non-runnable capability must carry a Reason: %+v", cap)
	}
}

func TestEvalCommandHint(t *testing.T) {
	got := EvalCommandHint("/x/preds.json", "run7", 8)
	for _, want := range []string{"run_evaluation", "--predictions_path /x/preds.json", "--run_id run7", "--max_workers 8", "SWE-bench_Verified"} {
		if !strings.Contains(got, want) {
			t.Errorf("hint missing %q: %s", want, got)
		}
	}
}

func TestParseEvalReport(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "report.json")
	os.WriteFile(p, []byte(`{"resolved_ids":["a__b-1","a__b-2"],"total_instances":5}`), 0o644)
	resolved, total, ids, ok := parseEvalReport(p)
	if !ok || resolved != 2 || total != 5 || len(ids) != 2 {
		t.Errorf("parse got resolved=%d total=%d ids=%v ok=%v", resolved, total, ids, ok)
	}
	// missing file -> not ok
	if _, _, _, ok := parseEvalReport(filepath.Join(dir, "nope.json")); ok {
		t.Errorf("missing report should not parse ok")
	}
}

func TestParseEvalReportDenominatorMirrorsBench(t *testing.T) {
	dir := t.TempDir()
	// No total_instances -> total = len(resolved)+len(unresolved), mirroring bench's
	// _count_report; never a fabricated total=resolved.
	p := filepath.Join(dir, "r.json")
	os.WriteFile(p, []byte(`{"resolved_ids":["a__b-1"],"unresolved_ids":["a__b-2","a__b-3"]}`), 0o644)
	resolved, total, _, ok := parseEvalReport(p)
	if !ok || resolved != 1 || total != 3 {
		t.Errorf("denominator from resolved+unresolved wrong: resolved=%d total=%d", resolved, total)
	}
	// resolved-only with no total and no unresolved -> bench's formula yields
	// total = resolved+unresolved = resolved (a real grade always carries both).
	p2 := filepath.Join(dir, "r2.json")
	os.WriteFile(p2, []byte(`{"resolved_ids":["a__b-1"]}`), 0o644)
	_, total2, _, _ := parseEvalReport(p2)
	if total2 != 1 {
		t.Errorf("resolved-only denominator should mirror bench (=resolved), got %d", total2)
	}
	// truly empty report -> total stays 0 (no false 100% from a fabricated total).
	p3 := filepath.Join(dir, "r3.json")
	os.WriteFile(p3, []byte(`{}`), 0o644)
	_, total3, _, ok3 := parseEvalReport(p3)
	if ok3 || total3 != 0 {
		t.Errorf("empty report: ok=%v total=%d (want ok=false total=0)", ok3, total3)
	}
}

// TestRunEvalGatedHere exercises the honest gating: when the box lacks Docker or
// the harness, RunEval returns Available=false with a Reason and the exact
// command for a Docker-capable box — it never fabricates a resolve-rate. Skipped
// if this box can actually run the harness (then the gated branch isn't taken).
func TestRunEvalGatedHere(t *testing.T) {
	if DetectEvalCapability("").Runnable {
		t.Skip("box can run the harness; gated branch not exercised")
	}
	res, err := RunEval(EvalConfig{PredictionsPath: "/tmp/does-not-matter/preds.json"})
	if err != nil {
		t.Fatalf("gated RunEval should not error: %v", err)
	}
	if res.Available {
		t.Errorf("expected gated (unavailable) result, got %+v", res)
	}
	if res.Reason == "" || res.Command == "" {
		t.Errorf("gated result needs Reason + Command: %+v", res)
	}
}
