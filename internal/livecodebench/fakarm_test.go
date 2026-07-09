package livecodebench

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
)

func fakStubSampler(ev FakSampleEvidence, u RawSampleUsage) FakArmSampler {
	return func(_ context.Context, p Problem, i int) (string, RawSampleUsage, FakSampleEvidence, error) {
		return fmt.Sprintf("%s#%d", p.QuestionID, i), u, ev, nil
	}
}

// TestRunFakArmMirrorsRawIdentityAndFoldsEvidence witnesses acceptance #1: the
// fak arm records the same run identity shape as the raw arm (problems, model,
// n, temperature, release) and folds per-sample adjudication evidence.
func TestRunFakArmMirrorsRawIdentityAndFoldsEvidence(t *testing.T) {
	probs := rawArmProblems(2)
	cfg := RawArmConfig{Model: "glm-4.6", Endpoint: "http://127.0.0.1:18080/v1", N: 2, Temperature: 0.8, Concurrency: 3}

	ev := FakSampleEvidence{Adjudicated: true, Adjudications: 2, Denied: 1, SafeResolves: 1, ResultAdmissions: 3}
	rep, err := RunFakArm(context.Background(), cfg, "release_v6", probs, fakStubSampler(ev, RawSampleUsage{PromptTokens: 10, CompletionTokens: 5, CachedPromptTokens: 4}))
	if err != nil {
		t.Fatalf("RunFakArm: %v", err)
	}
	if rep.Arm != FakArmName {
		t.Fatalf("arm = %q, want %q", rep.Arm, FakArmName)
	}
	if rep.Model != "glm-4.6" || rep.N != 2 || rep.Temperature != 0.8 || rep.Release != "release_v6" {
		t.Fatalf("run identity not recorded: %+v", rep)
	}
	if len(rep.Problems) != 2 {
		t.Fatalf("want 2 problems, got %d", len(rep.Problems))
	}
	for pi, p := range rep.Problems {
		if p.QuestionID != fmt.Sprintf("q%d", pi) {
			t.Fatalf("problem %d out of order: %q", pi, p.QuestionID)
		}
		if p.PromptSHA256 == "" {
			t.Fatalf("problem %q missing prompt_sha256 (SamePromptHash needs it)", p.QuestionID)
		}
		want := []string{fmt.Sprintf("q%d#0", pi), fmt.Sprintf("q%d#1", pi)}
		if len(p.Completions) != 2 || p.Completions[0] != want[0] || p.Completions[1] != want[1] {
			t.Fatalf("problem %d completions = %v, want %v", pi, p.Completions, want)
		}
	}
	// 4 samples total, evidence folded per sample.
	wantAdj := FakArmAdjudication{AdjudicatedSamples: 4, Adjudications: 8, Denied: 4, SafeResolves: 4, ResultAdmissions: 12}
	if rep.Adjudication != wantAdj {
		t.Fatalf("adjudication fold = %+v, want %+v", rep.Adjudication, wantAdj)
	}
	if rep.Usage.Samples != 4 || rep.Usage.CachedPromptTokens != 16 {
		t.Fatalf("usage fold wrong: %+v", rep.Usage)
	}
}

func TestRunFakArmSamplerErrorAborts(t *testing.T) {
	probs := rawArmProblems(2)
	cfg := RawArmConfig{Model: "m", Endpoint: "e", N: 1, Concurrency: 1}
	boom := errors.New("gateway 500")
	sample := func(_ context.Context, p Problem, _ int) (string, RawSampleUsage, FakSampleEvidence, error) {
		return "", RawSampleUsage{}, FakSampleEvidence{}, boom
	}
	if _, err := RunFakArm(context.Background(), cfg, "release_v6", probs, sample); err == nil || !errors.Is(err, boom) {
		t.Fatalf("want wrapped sampler error, got %v", err)
	}
	if _, err := RunFakArm(context.Background(), cfg, "release_v6", probs, nil); err == nil {
		t.Fatalf("want error for nil sampler")
	}
}

// compareFixture runs both arms over the same problems / config / release and
// returns the reports, so the comparison is asserted over real arm artifacts.
func compareFixture(t *testing.T) (RawArmReport, FakArmReport) {
	t.Helper()
	probs := rawArmProblems(3)
	cfg := RawArmConfig{Model: "glm-4.6", Endpoint: "http://127.0.0.1:8080/v1", N: 2, Temperature: 0.2, Concurrency: 2}

	rawRes, err := RunRawArmCached(context.Background(), cfg, "release_v6", probs,
		func(_ context.Context, p Problem, i int) (string, RawSampleUsage, error) {
			return "raw", RawSampleUsage{PromptTokens: 10, CompletionTokens: 5}, nil
		}, nil, nil)
	if err != nil {
		t.Fatalf("RunRawArmCached: %v", err)
	}
	fak, err := RunFakArm(context.Background(), cfg, "release_v6", probs,
		fakStubSampler(FakSampleEvidence{Adjudicated: true, Adjudications: 1}, RawSampleUsage{PromptTokens: 12, CompletionTokens: 5}))
	if err != nil {
		t.Fatalf("RunFakArm: %v", err)
	}
	return rawRes.Report, fak
}

// TestCompareArmsAssertsParityTrue witnesses acceptance #2: over two arms that
// genuinely ran the same problems / model / n / temperature / release, the
// comparison asserts SameProblemIDs and SamePromptHash true, emits per-arm
// summaries + deltas, and never allows a result claim (acceptance #3).
func TestCompareArmsAssertsParityTrue(t *testing.T) {
	raw, fak := compareFixture(t)
	c := CompareArms(raw, fak)

	if c.Schema != ABComparisonSchema {
		t.Fatalf("schema = %q", c.Schema)
	}
	if !c.SameProblemIDs || !c.SamePromptHash {
		t.Fatalf("parity assertions must hold: %+v (mismatches: %v)", c, c.Mismatches)
	}
	if !c.SameModel || !c.SameN || !c.SameTemperature || !c.SameRelease {
		t.Fatalf("identity assertions must hold: %+v (mismatches: %v)", c, c.Mismatches)
	}
	if len(c.Mismatches) != 0 {
		t.Fatalf("unexpected mismatches: %v", c.Mismatches)
	}
	if c.Raw.Arm != RawArmName || c.Fak.Arm != FakArmName || c.Raw.Problems != 3 || c.Fak.Problems != 3 {
		t.Fatalf("per-arm summaries wrong: raw=%+v fak=%+v", c.Raw, c.Fak)
	}
	// fak minus raw: 6 samples × (12-10) prompt tokens.
	if c.UsageDelta.PromptTokens != 12 || c.UsageDelta.Samples != 0 {
		t.Fatalf("usage delta wrong: %+v", c.UsageDelta)
	}
	if c.FakAdjudication.AdjudicatedSamples != 6 || c.FakAdjudication.Adjudications != 6 {
		t.Fatalf("fak adjudication evidence wrong: %+v", c.FakAdjudication)
	}
	// Acceptance #3: no pass-rate claim without graded evidence.
	if c.ResultClaimAllowed {
		t.Fatalf("result_claim_allowed must be false on an ungraded comparison")
	}
	if c.PassRateDelta != PassRateDeltaUngraded {
		t.Fatalf("pass_rate_delta = %q, want the ungraded sentinel", c.PassRateDelta)
	}
}

func TestCompareArmsDetectsMismatches(t *testing.T) {
	raw, fak := compareFixture(t)

	t.Run("different problems", func(t *testing.T) {
		f := fak
		f.Problems = append([]RawArmProblem(nil), fak.Problems[:2]...)
		c := CompareArms(raw, f)
		if c.SameProblemIDs {
			t.Fatalf("SameProblemIDs must be false when the fak arm ran fewer problems")
		}
		if len(c.Mismatches) == 0 {
			t.Fatalf("mismatch detail required")
		}
	})

	t.Run("tampered prompt hash", func(t *testing.T) {
		f := fak
		f.Problems = append([]RawArmProblem(nil), fak.Problems...)
		f.Problems[1].PromptSHA256 = "deadbeef"
		c := CompareArms(raw, f)
		if c.SamePromptHash {
			t.Fatalf("SamePromptHash must be false on a differing hash")
		}
	})

	t.Run("missing prompt hash is never a silent pass", func(t *testing.T) {
		r := raw
		r.Problems = append([]RawArmProblem(nil), raw.Problems...)
		r.Problems[0].PromptSHA256 = ""
		c := CompareArms(r, fak)
		if c.SamePromptHash {
			t.Fatalf("SamePromptHash must be false when a report carries no hash")
		}
		found := false
		for _, m := range c.Mismatches {
			if strings.Contains(m, "prompt hash missing") {
				found = true
			}
		}
		if !found {
			t.Fatalf("want a 'prompt hash missing' mismatch, got %v", c.Mismatches)
		}
	})

	t.Run("different identity", func(t *testing.T) {
		f := fak
		f.Model, f.Temperature, f.Release = "other", 0.9, "release_v5"
		c := CompareArms(raw, f)
		if c.SameModel || c.SameTemperature || c.SameRelease {
			t.Fatalf("identity mismatches not detected: %+v", c)
		}
		if c.ResultClaimAllowed {
			t.Fatalf("result_claim_allowed must stay false")
		}
	})
}
