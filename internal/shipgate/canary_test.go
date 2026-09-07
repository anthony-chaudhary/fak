package shipgate

// canary_test.go is issue #4580's named acceptance witness. It proves the
// quality-aware canary adjudicator (1) simulates PROMOTE, HOLD (inconclusive),
// and ROLLBACK, (2) fails against a planted representative defect and passes
// after the fix in a clean, independently replayed environment (the JSON
// replay artifact round-trips to the same verdict), (3) never passes missing
// or inconclusive evidence, and (4) scrubs host paths, emails, and
// secret-shaped values out of the captured artifact.

import (
	"encoding/json"
	"strings"
	"testing"
)

// fullProvenance is a complete evidence record — every required field set.
func fullProvenance() CanaryProvenance {
	return CanaryProvenance{
		Model:     "fak-sim-7b@q4",
		Tokenizer: "fak-sim-tok@v2",
		Engine:    "fak-engine/decode@ep8",
		Seed:      "oracle:fixed-cases-v1",
		Revision:  "rev:0123abcd",
		Baseline:  "baseline:nightly-suite-v1 tolerances:release-floor-v1",
	}
}

// healthyCase is a fully-evidenced case that should PROMOTE as written.
func healthyCase() CanaryCase {
	return CanaryCase{
		ID: "canary-4580", Tier: CanaryTierPR,
		CostNote:   "~1ms CPU, no GPU, no model call",
		MinSamples: 50, Provenance: fullProvenance(),
		Slices: []QualitySlice{
			{Name: "safety-critical-prompts", Critical: true, Baseline: 0.94, Candidate: 0.95, Tolerance: 0.02, Samples: 200, Measured: true},
			{Name: "long-context-recall", Critical: false, Baseline: 0.81, Candidate: 0.84, Tolerance: 0.05, Samples: 120, Measured: true},
		},
	}
}

// TestCanarySimulationCoversPromoteHoldRollback proves the simulation exercises
// all three outcomes with full per-case provenance, an explicit tier, and a
// documented cost — the issue's first and fourth acceptance criteria.
func TestCanarySimulationCoversPromoteHoldRollback(t *testing.T) {
	results := SimulateCanary()
	want := []string{"PROMOTE", "HOLD", "ROLLBACK"}
	if len(results) != len(want) {
		t.Fatalf("simulation returned %d results, want %d", len(results), len(want))
	}
	for i, r := range results {
		if r.Verdict != want[i] {
			t.Fatalf("simulation[%d] verdict=%s want %s (reason: %s)", i, r.Verdict, want[i], r.Reason)
		}
		if r.Promoted != (want[i] == "PROMOTE") {
			t.Fatalf("simulation[%d] promoted=%v inconsistent with verdict %s", i, r.Promoted, r.Verdict)
		}
		if miss := r.Replay.Provenance.missing(); len(miss) > 0 {
			t.Fatalf("simulation[%d] replay lost provenance fields %v", i, miss)
		}
		if !r.Replay.Tier.known() {
			t.Fatalf("simulation[%d] replay has unassigned tier %q", i, r.Replay.Tier)
		}
		if strings.TrimSpace(r.Replay.CostNote) == "" {
			t.Fatalf("simulation[%d] replay does not document runtime/resource cost", i)
		}
		if !r.Replay.Scrubbed {
			t.Fatalf("simulation[%d] replay is not marked scrubbed", i)
		}
	}
	// The simulated rollback is the quality-aware core rule: the aggregate mean
	// RISES while the critical slice craters — the mean must not rescue it.
	rb := results[2]
	if rb.CandidateMean <= rb.BaselineMean {
		t.Fatalf("simulated rollback should have a rising mean (got %.4f -> %.4f) to witness that the mean cannot rescue a critical breach",
			rb.BaselineMean, rb.CandidateMean)
	}
	if rb.FirstDivergence == nil || rb.FirstDivergence.Slice != "safety-critical-prompts" {
		t.Fatalf("simulated rollback did not name the first actionable divergence: %+v", rb.FirstDivergence)
	}
}

// TestCanaryPlantedDefectRollsBackThenFixedPromotes is the issue's witness: a
// planted representative defect (a cratered critical slice) FAILS the gate, the
// captured replay artifact reproduces the failure when independently replayed
// from its JSON bytes alone, and after the fix the same pipeline PASSES — also
// replayed independently. The artifact is verified scrubbed of a planted host
// path, secret, and email.
func TestCanaryPlantedDefectRollsBackThenFixedPromotes(t *testing.T) {
	planted := healthyCase()
	// The representative defect: the candidate degrades the critical safety
	// slice past tolerance (drop 0.06, 3x the 0.02 floor) while boosting the
	// other slice enough that the aggregate mean still rises.
	planted.Slices[0].Candidate = 0.88
	planted.Slices[1].Candidate = 0.99
	// Host-secret-shaped values that must NOT survive into the artifact.
	planted.Provenance.Baseline = "baseline from /home/test-user/baselines/nightly-v3.json"
	planted.Provenance.Engine = "fak-engine/decode@ep8 api_key: sk_live1234567890abcd"
	planted.CostNote = "~1ms CPU; owner alice@example.com"

	res := AdjudicateCanary(planted)
	if res.Verdict != "ROLLBACK" || res.Promoted {
		t.Fatalf("planted defect did not roll back: verdict=%s promoted=%v reason=%s", res.Verdict, res.Promoted, res.Reason)
	}
	if res.FirstDivergence == nil || res.FirstDivergence.Slice != "safety-critical-prompts" {
		t.Fatalf("rollback did not name the first actionable divergence: %+v", res.FirstDivergence)
	}
	if res.CandidateMean <= res.BaselineMean {
		t.Fatalf("test setup lost its point: the mean should rise (%.4f -> %.4f) while the critical slice fails",
			res.BaselineMean, res.CandidateMean)
	}

	// Capture the artifact and verify it is scrubbed.
	artifact, err := res.MarshalReplay()
	if err != nil {
		t.Fatalf("MarshalReplay: %v", err)
	}
	blob := string(artifact)
	for _, leaked := range []string{"test-user/baselines", "sk_live1234567890abcd", "alice@example.com"} {
		if strings.Contains(blob, leaked) {
			t.Fatalf("replay artifact leaked %q:\n%s", leaked, blob)
		}
	}
	for _, marker := range []string{"[redacted-path]", "[redacted-secret]", "[redacted-email]"} {
		if !strings.Contains(blob, marker) {
			t.Fatalf("replay artifact missing scrub marker %q:\n%s", marker, blob)
		}
	}

	// Independent replay: rebuild the case from the artifact's JSON bytes alone
	// (the clean-environment stand-in) and re-adjudicate — same verdict, same
	// first divergence.
	var replayed CanaryReplay
	if err := json.Unmarshal(artifact, &replayed); err != nil {
		t.Fatalf("unmarshal replay artifact: %v", err)
	}
	res2 := AdjudicateCanary(ReplayCase(replayed))
	if res2.Verdict != "ROLLBACK" {
		t.Fatalf("independent replay verdict=%s want ROLLBACK (reason: %s)", res2.Verdict, res2.Reason)
	}
	if res2.FirstDivergence == nil || res2.FirstDivergence.Slice != res.FirstDivergence.Slice {
		t.Fatalf("independent replay diverged on the divergence: %+v vs %+v", res2.FirstDivergence, res.FirstDivergence)
	}

	// The fix: restore the critical slice. The same pipeline now PROMOTEs, and
	// its own artifact independently replays to PROMOTE.
	fixed := planted
	fixed.Slices = append([]QualitySlice(nil), planted.Slices...)
	fixed.Slices[0].Candidate = 0.95
	fres := AdjudicateCanary(fixed)
	if fres.Verdict != "PROMOTE" || !fres.Promoted {
		t.Fatalf("fixed candidate did not promote: verdict=%s reason=%s", fres.Verdict, fres.Reason)
	}
	fart, err := fres.MarshalReplay()
	if err != nil {
		t.Fatalf("MarshalReplay(fixed): %v", err)
	}
	var freplayed CanaryReplay
	if err := json.Unmarshal(fart, &freplayed); err != nil {
		t.Fatalf("unmarshal fixed replay artifact: %v", err)
	}
	if fres2 := AdjudicateCanary(ReplayCase(freplayed)); fres2.Verdict != "PROMOTE" {
		t.Fatalf("independent replay of the fix verdict=%s want PROMOTE (reason: %s)", fres2.Verdict, fres2.Reason)
	}
}

// TestCanaryInconclusiveEvidenceNeverPromotes proves the fail-closed rule: every
// way evidence can be missing or inconclusive HOLDs — never a pass, and never an
// automated rollback from unattributable evidence.
func TestCanaryInconclusiveEvidenceNeverPromotes(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*CanaryCase)
	}{
		{"missing model", func(c *CanaryCase) { c.Provenance.Model = "" }},
		{"missing tokenizer", func(c *CanaryCase) { c.Provenance.Tokenizer = " " }},
		{"missing engine", func(c *CanaryCase) { c.Provenance.Engine = "" }},
		{"missing seed/oracle", func(c *CanaryCase) { c.Provenance.Seed = "" }},
		{"missing revision", func(c *CanaryCase) { c.Provenance.Revision = "" }},
		{"missing baseline provenance", func(c *CanaryCase) { c.Provenance.Baseline = "" }},
		{"unassigned tier", func(c *CanaryCase) { c.Tier = "weekly" }},
		{"undocumented cost", func(c *CanaryCase) { c.CostNote = "  " }},
		{"no evidence floor", func(c *CanaryCase) { c.MinSamples = 0 }},
		{"no slices", func(c *CanaryCase) { c.Slices = nil }},
		{"unmeasured slice", func(c *CanaryCase) { c.Slices[1].Measured = false }},
		{"under-sampled slice", func(c *CanaryCase) { c.Slices[1].Samples = 3 }},
		{"no critical slice", func(c *CanaryCase) {
			c.Slices[0].Critical = false
		}},
		{"negative aggregate delta without critical breach", func(c *CanaryCase) {
			c.Slices[1].Candidate = 0.60 // non-critical loss drags the mean down
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := healthyCase()
			tc.mutate(&c)
			res := AdjudicateCanary(c)
			if res.Verdict != "HOLD" || res.Promoted {
				t.Fatalf("%s: verdict=%s promoted=%v want HOLD (reason: %s)", tc.name, res.Verdict, res.Promoted, res.Reason)
			}
			if strings.TrimSpace(res.Reason) == "" {
				t.Fatalf("%s: HOLD carries no reason", tc.name)
			}
			if !res.Replay.Scrubbed {
				t.Fatalf("%s: HOLD did not emit a scrubbed replay artifact", tc.name)
			}
		})
	}

	// Sanity: the un-mutated case really does promote, so each HOLD above is
	// attributable to its single planted gap.
	if res := AdjudicateCanary(healthyCase()); res.Verdict != "PROMOTE" {
		t.Fatalf("healthy control case verdict=%s want PROMOTE (reason: %s)", res.Verdict, res.Reason)
	}
}

// TestCanaryDirectionalUncertainty proves issue #10278:
// 1. Sparse critical slices with point drop within tolerance roll back if the adverse bound breaches.
// 2. Ample samples with the same drop clear the adverse bound and promote.
// 3. Thin apparent aggregate gains at sparse sample counts hold (do not false-green promote).
// 4. Uncertainty fields round-trip through replay artifacts.
func TestCanaryDirectionalUncertainty(t *testing.T) {
	t.Run("sparse_adverse_breach_rolls_back", func(t *testing.T) {
		// Point drop: candidate 0.93 vs baseline 0.94 -> drop = 0.01 (within tolerance 0.02).
		// But at Samples = 50, margin is ~0.055, adverse drop is ~0.065 > 0.02 tolerance!
		c := healthyCase()
		c.Slices[0].Samples = 50
		c.Slices[0].Candidate = 0.93
		c.Slices[1].Samples = 5000
		c.Slices[1].Candidate = 0.99 // boost aggregate mean so mean rises

		res := AdjudicateCanary(c)
		if res.Verdict != "ROLLBACK" {
			t.Fatalf("sparse adverse breach verdict=%s want ROLLBACK (reason: %s)", res.Verdict, res.Reason)
		}
		if res.FirstDivergence == nil {
			t.Fatalf("expected divergence recorded on adverse rollback")
		}
		if res.FirstDivergence.UncertaintyMargin <= 0 {
			t.Fatalf("expected positive uncertainty margin, got %f", res.FirstDivergence.UncertaintyMargin)
		}
		if !strings.Contains(res.FirstDivergence.Reason, "adverse drop") {
			t.Fatalf("expected reason to mention adverse drop, got %q", res.FirstDivergence.Reason)
		}
	})

	t.Run("ample_samples_clears_adverse_bound_promotes", func(t *testing.T) {
		// Same drop: candidate 0.93 vs baseline 0.94 -> drop = 0.01.
		// At Samples = 10,000, margin is ~0.004, adverse drop is ~0.014 <= 0.02 tolerance!
		c := healthyCase()
		c.Slices[0].Samples = 10000
		c.Slices[0].Candidate = 0.93
		c.Slices[1].Samples = 10000
		c.Slices[1].Candidate = 0.90 // aggregate mean delta: (0.93+0.90)/2 - (0.94+0.81)/2 = 0.915 - 0.875 = +0.04 > 0

		res := AdjudicateCanary(c)
		if res.Verdict != "PROMOTE" {
			t.Fatalf("ample samples verdict=%s want PROMOTE (reason: %s)", res.Verdict, res.Reason)
		}
		if c.Slices[0].UncertaintyMargin <= 0 || c.Slices[0].UncertaintyMargin >= 0.01 {
			t.Fatalf("expected tight margin at 10k samples, got %f", c.Slices[0].UncertaintyMargin)
		}
	})

	t.Run("sparse_thin_improvement_holds_not_promotes", func(t *testing.T) {
		// Candidate mean is slightly higher (+0.001), but with sparse samples (50),
		// adverse lower bound of aggregate improvement is negative.
		// Use generous tolerance so critical slices don't roll back first.
		c := healthyCase()
		c.Slices[0].Tolerance = 0.10
		c.Slices[0].Samples = 50
		c.Slices[0].Candidate = 0.941 // +0.001
		c.Slices[1].Tolerance = 0.10
		c.Slices[1].Samples = 50
		c.Slices[1].Candidate = 0.811 // +0.001

		res := AdjudicateCanary(c)
		if res.Verdict != "HOLD" {
			t.Fatalf("sparse thin improvement verdict=%s want HOLD (reason: %s)", res.Verdict, res.Reason)
		}
		if !strings.Contains(res.Reason, "adverse bound") {
			t.Fatalf("expected HOLD reason to mention adverse bound, got %q", res.Reason)
		}
	})

	t.Run("replay_artifact_preserves_uncertainty_fields", func(t *testing.T) {
		c := healthyCase()
		res := AdjudicateCanary(c)
		artifact, err := res.MarshalReplay()
		if err != nil {
			t.Fatalf("MarshalReplay failed: %v", err)
		}

		var replayed CanaryReplay
		if err := json.Unmarshal(artifact, &replayed); err != nil {
			t.Fatalf("Unmarshal replay failed: %v", err)
		}

		if len(replayed.Slices) == 0 || replayed.Slices[0].UncertaintyMargin <= 0 {
			t.Fatalf("replay lost uncertainty margin: %+v", replayed.Slices)
		}
		if replayed.Slices[0].IntervalMethod != DefaultIntervalMethod {
			t.Fatalf("replay interval method=%q want %q", replayed.Slices[0].IntervalMethod, DefaultIntervalMethod)
		}
	})
}
