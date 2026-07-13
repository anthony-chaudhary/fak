package productscorecard

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// goodProvenance is a fully-populated evidence record. Individual tests blank a
// field to exercise the fail-closed path.
func goodProvenance() Provenance {
	return Provenance{
		Model: "fak-tiny-1", Tokenizer: "bpe-v3", Engine: "fak-engine/v1",
		Seed: "deterministic-oracle:golden-42", Revision: "abc123def456",
		Baseline: "nightly-baseline@2026-07-10",
	}
}

// plantedDefectCase is the representative defect the witness plants: the
// aggregate mean RISES (three slices improve) while one critical slice
// collapses past tolerance. A mean-only gate would wave this through.
func plantedDefectCase() Case {
	return Case{
		ID:         "slice-regress/model-context-lang-task-engine",
		Tier:       TierPR,
		CostNote:   "~1ms CPU, deterministic oracle, no GPU/key/network",
		Provenance: goodProvenance(),
		Slices: []Slice{
			{Name: "task:codegen", Critical: false, Baseline: 0.60, Candidate: 0.92, Tolerance: 0.05, Measured: true},
			{Name: "lang:python", Critical: false, Baseline: 0.60, Candidate: 0.95, Tolerance: 0.05, Measured: true},
			{Name: "engine:speculative", Critical: true, Baseline: 0.90, Candidate: 0.40, Tolerance: 0.03, Measured: true}, // planted collapse
			{Name: "context:128k", Critical: false, Baseline: 0.60, Candidate: 0.98, Tolerance: 0.05, Measured: true},
		},
	}
}

// fixedCase is the same case after the fix: the critical slice is restored to
// within tolerance. The mean is identical-shaped (still an improvement), so the
// only thing that changed is the critical cohort's health.
func fixedCase() Case {
	c := plantedDefectCase()
	c.Slices[2].Candidate = 0.89 // 0.90 -> 0.89 is a 0.01 drop, within 0.03 tolerance
	return c
}

// TestGateBlocksHiddenCriticalRegression is the core acceptance witness: a
// higher candidate mean must NOT rescue a catastrophic critical-slice loss.
func TestGateBlocksHiddenCriticalRegression(t *testing.T) {
	c := plantedDefectCase()
	r := Gate(c)

	if r.CandidateMean <= r.BaselineMean {
		t.Fatalf("test invalid: candidate mean %.4f should exceed baseline mean %.4f to prove the aggregate hides the loss", r.CandidateMean, r.BaselineMean)
	}
	if r.Pass {
		t.Fatalf("gate PASSED a case with a catastrophic critical-slice loss (mean rose %.4f->%.4f) - aggregate hid the regression", r.BaselineMean, r.CandidateMean)
	}
	if r.Verdict != SliceRegression {
		t.Fatalf("verdict = %q, want %q", r.Verdict, SliceRegression)
	}
	if r.FirstDivergence == nil || r.FirstDivergence.Slice != "engine:speculative" {
		t.Fatalf("first divergence = %+v, want the engine:speculative critical slice", r.FirstDivergence)
	}
}

// TestGatePassesAfterFix is the other half of the witness: the same case passes
// once the critical slice is restored, in a freshly built (independent) case.
func TestGatePassesAfterFix(t *testing.T) {
	r := Gate(fixedCase())
	if !r.Pass {
		t.Fatalf("fixed case did not pass: verdict=%q reason=%q", r.Verdict, r.Reason)
	}
	if r.Verdict != SliceOK {
		t.Fatalf("verdict = %q, want %q", r.Verdict, SliceOK)
	}
	if r.FirstDivergence != nil {
		t.Fatalf("fixed case reported a divergence: %+v", r.FirstDivergence)
	}
}

// TestGateFailsClosedOnMissingEvidence proves missing/inconclusive evidence is
// never a pass, across every fail-closed path.
func TestGateFailsClosedOnMissingEvidence(t *testing.T) {
	cases := map[string]func(*Case){
		"missing provenance": func(c *Case) { c.Provenance.Tokenizer = "" },
		"unknown tier":       func(c *Case) { c.Tier = Tier("someday") },
		"no cost note":       func(c *Case) { c.CostNote = "" },
		"no slices":          func(c *Case) { c.Slices = nil },
		"unmeasured slice":   func(c *Case) { c.Slices[2].Measured = false },
		"no critical slice": func(c *Case) {
			for i := range c.Slices {
				c.Slices[i].Critical = false
			}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := fixedCase() // starts from a would-be pass
			mutate(&c)
			r := Gate(c)
			if r.Pass {
				t.Fatalf("gate PASSED with %s - inconclusive evidence must never pass", name)
			}
			if r.Verdict != SliceInconclusive {
				t.Fatalf("verdict = %q, want %q (%s)", r.Verdict, SliceInconclusive, name)
			}
		})
	}
}

// TestFirstDivergenceIsDeterministic proves the reported divergence is the FIRST
// actionable one in caller-priority order when several critical slices regress.
func TestFirstDivergenceIsDeterministic(t *testing.T) {
	c := plantedDefectCase()
	c.Slices[0].Critical, c.Slices[0].Candidate = true, 0.10 // an earlier critical collapse
	r := Gate(c)
	if r.FirstDivergence == nil || r.FirstDivergence.Slice != "task:codegen" {
		t.Fatalf("first divergence = %+v, want the earliest critical slice task:codegen", r.FirstDivergence)
	}
}

// TestReplayArtifactIsScrubbedAndCaptured proves a failure emits a scrubbed,
// re-runnable replay artifact: it names the divergence, redacts a planted host
// secret, and round-trips as JSON. This is the captured proof written to disk.
func TestReplayArtifactIsScrubbedAndCaptured(t *testing.T) {
	c := plantedDefectCase()
	// Plant a host secret in the baseline-provenance field.
	c.Provenance.Baseline = `baseline from /home/test-user\secrets\key.txt token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123`
	r := Gate(c)

	if r.Pass {
		t.Fatal("planted-defect case unexpectedly passed")
	}
	blob, err := r.MarshalReplay()
	if err != nil {
		t.Fatalf("MarshalReplay: %v", err)
	}
	text := string(blob)
	for _, leaked := range []string{`/home/test-user`, "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123", "secrets"} {
		if strings.Contains(text, leaked) {
			t.Fatalf("replay artifact leaked secret %q:\n%s", leaked, text)
		}
	}
	if !strings.Contains(text, "[redacted-path]") || !strings.Contains(text, "[redacted-secret]") {
		t.Fatalf("replay artifact was not scrubbed:\n%s", text)
	}

	// Emit + re-read the artifact from a clean temp dir (independent replay).
	path := filepath.Join(t.TempDir(), "replay.json")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatalf("write replay: %v", err)
	}
	var back Replay
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read replay: %v", err)
	}
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("replay artifact is not valid JSON: %v", err)
	}
	if back.Schema != SliceRegressionSchema {
		t.Fatalf("replay schema = %q, want %q", back.Schema, SliceRegressionSchema)
	}
	if back.FirstDivergence == nil || back.FirstDivergence.Slice != "engine:speculative" {
		t.Fatalf("replay did not localize the divergence: %+v", back.FirstDivergence)
	}
	if !back.Scrubbed {
		t.Fatal("replay artifact not marked scrubbed")
	}
}

// TestGateAllBatchFailsClosed proves a batch is green only if every case is OK
// and an empty batch fails closed.
func TestGateAllBatchFailsClosed(t *testing.T) {
	if _, ok := GateAll(nil); ok {
		t.Fatal("empty batch passed - a batch with no evidence must fail closed")
	}
	if _, ok := GateAll([]Case{plantedDefectCase(), fixedCase()}); ok {
		t.Fatal("batch containing a regression passed")
	}
	results, ok := GateAll([]Case{fixedCase()})
	if !ok {
		t.Fatalf("batch of one passing case failed: %+v", results)
	}
}
