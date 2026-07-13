package bench

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestContaminationAudit is the re-runnable witness for issue #4571. It exercises
// BOTH halves of the acceptance witness on the same run:
//
//   - the FIXED corpus (DefaultContaminationCorpus) audits CLEAN — every case is
//     admitted to the confirmatory claim set; and
//   - the same corpus with three planted representative defects (a duplicate, a
//     pre-cutoff-exposed case, and a case with missing provenance) audits as
//     CONTAMINATION RISK — each defect is flagged, named with its first actionable
//     divergence, and excluded from the confirmatory claim set.
//
// The clean report is pinned to a committed golden (the scrubbed, independently
// replayable artifact).
func TestContaminationAudit(t *testing.T) {
	// --- passes after the fix: the clean corpus admits everything ---
	clean := AuditContamination(DefaultContaminationCorpus())
	if clean.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("provenance = %q, want %q", clean.Provenance.Kind, ProvenanceSimulated)
	}
	if clean.Verdict != VerdictContaminationClean {
		t.Fatalf("clean corpus verdict = %q, want %q", clean.Verdict, VerdictContaminationClean)
	}
	if clean.Cases != 3 || len(clean.ConfirmatoryEligible) != 3 || len(clean.Excluded) != 0 {
		t.Fatalf("clean corpus cases/eligible/excluded = %d/%d/%d, want 3/3/0",
			clean.Cases, len(clean.ConfirmatoryEligible), len(clean.Excluded))
	}
	// Tier cost is documented per tier, in escalating order, for every run tier.
	wantTiers := []struct {
		tier  string
		cases int
		cost  int
	}{{"pr", 1, 120}, {"nightly", 1, 800}, {"release", 1, 5000}}
	if len(clean.TierCosts) != len(wantTiers) {
		t.Fatalf("tier_costs = %+v, want %d tiers", clean.TierCosts, len(wantTiers))
	}
	for i, w := range wantTiers {
		got := clean.TierCosts[i]
		if got.Tier != w.tier || got.Cases != w.cases || got.RuntimeCostMS != w.cost {
			t.Errorf("tier_costs[%d] = %+v, want {%s %d %d}", i, got, w.tier, w.cases, w.cost)
		}
	}

	// --- fails against a planted defect: three representative contaminations ---
	planted := append(DefaultContaminationCorpus(),
		// (A) duplicate: same content hash as "sampling-ci-holdout".
		ContaminationCase{
			ID: "sampling-ci-dupe", Suite: "sampling",
			ContentHash: "sha256:sample-02", Model: "fak-decode-ref", Tokenizer: "bpe-v3",
			Engine: "cpu-q8", SeedOrOracle: "oracle:closed-form-binomial", CodeRevision: "rev-abc123",
			Baseline: "CI=[0.48,0.52] vs analytic mean", Holdout: true,
			Tier: "nightly", RuntimeCostMS: 800,
		},
		// (B) contaminated: published BEFORE the model's train cutoff, not holdout.
		ContaminationCase{
			ID: "public-leaderboard-leaked", Suite: "public",
			ContentHash: "sha256:leak-04", Model: "fak-decode-ref", Tokenizer: "bpe-v3",
			Engine: "cpu-q8", SeedOrOracle: "seed:1", CodeRevision: "rev-abc123",
			Baseline:    "tol=1e-6 vs published-answer-key",
			PublishedAt: "2023-05-01", TrainCutoff: "2024-10-01", Holdout: false,
			Tier: "pr", RuntimeCostMS: 90,
		},
		// (C) incomplete: missing the seed/deterministic oracle — inconclusive.
		ContaminationCase{
			ID: "missing-oracle", Suite: "decode",
			ContentHash: "sha256:miss-05", Model: "fak-decode-ref", Tokenizer: "bpe-v3",
			Engine: "cpu-q8", SeedOrOracle: "", CodeRevision: "rev-abc123",
			Baseline:    "tol=1e-6 vs ref-golden",
			PublishedAt: "2025-04-01", TrainCutoff: "2024-10-01", Holdout: false,
			Tier: "pr", RuntimeCostMS: 100,
		},
	)
	risk := AuditContamination(planted)
	if risk.Verdict != VerdictContaminationRisk {
		t.Fatalf("planted-defect verdict = %q, want %q", risk.Verdict, VerdictContaminationRisk)
	}
	// The three original cases stay eligible; none of the three defects leaks into
	// the confirmatory claim set.
	if len(risk.ConfirmatoryEligible) != 3 {
		t.Fatalf("eligible after planting = %v, want the 3 clean cases only", risk.ConfirmatoryEligible)
	}
	for _, bad := range []string{"sampling-ci-dupe", "public-leaderboard-leaked", "missing-oracle"} {
		for _, ok := range risk.ConfirmatoryEligible {
			if ok == bad {
				t.Errorf("planted defect %q leaked into the confirmatory claim set", bad)
			}
		}
	}
	byCase := map[string]ReplayArtifact{}
	for _, a := range risk.Excluded {
		byCase[a.Case] = a
	}
	if a := byCase["sampling-ci-dupe"]; a.Status != StatusDuplicate || a.DuplicateOf != "sampling-ci-holdout" {
		t.Errorf("dupe artifact = %+v, want status=%s duplicate_of=sampling-ci-holdout", a, StatusDuplicate)
	}
	if a := byCase["public-leaderboard-leaked"]; a.Status != StatusContaminated {
		t.Errorf("leaked artifact status = %q, want %q", a.Status, StatusContaminated)
	}
	if a := byCase["missing-oracle"]; a.Status != StatusIncomplete {
		t.Errorf("incomplete artifact status = %q, want %q", a.Status, StatusIncomplete)
	}
	// Every excluded case names a first actionable divergence — never a bare fail.
	for _, a := range risk.Excluded {
		if a.Divergence == "" {
			t.Errorf("excluded case %q has no first_divergence", a.Case)
		}
	}

	// The clean report marshals to stable JSON and matches the committed golden
	// (the scrubbed, re-derivable replay artifact). Regenerate with UPDATE_GOLDEN=1.
	got, err := clean.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	golden := filepath.Join("testdata", "contamination_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("clean report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}

// TestContaminationNeverPassesOnInconclusive pins the fail-closed rule (the issue's
// "missing or inconclusive evidence is never pass"): for EACH required provenance
// field, dropping it must move the case out of the confirmatory claim set with an
// incomplete verdict and a divergence that names the missing field — never a silent
// admit.
func TestContaminationNeverPassesOnInconclusive(t *testing.T) {
	base := DefaultContaminationCorpus()[0] // a fully-evidenced admitted case
	if r := AuditContamination([]ContaminationCase{base}); r.Verdict != VerdictContaminationClean {
		t.Fatalf("base case is not clean to start: %q", r.Verdict)
	}

	drops := []struct {
		name   string
		mutate func(*ContaminationCase)
	}{
		{"model", func(c *ContaminationCase) { c.Model = "" }},
		{"tokenizer", func(c *ContaminationCase) { c.Tokenizer = "" }},
		{"engine", func(c *ContaminationCase) { c.Engine = "" }},
		{"seed_or_oracle", func(c *ContaminationCase) { c.SeedOrOracle = "" }},
		{"code_revision", func(c *ContaminationCase) { c.CodeRevision = "" }},
		{"baseline", func(c *ContaminationCase) { c.Baseline = "" }},
		{"content_hash", func(c *ContaminationCase) { c.ContentHash = "" }},
		{"tier", func(c *ContaminationCase) { c.Tier = "someday" }},
		{"runtime_cost", func(c *ContaminationCase) { c.RuntimeCostMS = 0 }},
		{"published_at", func(c *ContaminationCase) { c.PublishedAt = ""; c.Holdout = false }},
		{"train_cutoff", func(c *ContaminationCase) { c.TrainCutoff = ""; c.Holdout = false }},
	}
	for _, d := range drops {
		c := base
		d.mutate(&c)
		r := AuditContamination([]ContaminationCase{c})
		if r.Verdict != VerdictContaminationRisk {
			t.Errorf("dropping %s: verdict = %q, want %q (inconclusive must never pass)", d.name, r.Verdict, VerdictContaminationRisk)
		}
		if len(r.ConfirmatoryEligible) != 0 {
			t.Errorf("dropping %s: case stayed eligible %v, want excluded", d.name, r.ConfirmatoryEligible)
		}
		if len(r.Excluded) != 1 || r.Excluded[0].Status != StatusIncomplete || r.Excluded[0].Divergence == "" {
			t.Errorf("dropping %s: excluded = %+v, want one incomplete with a named divergence", d.name, r.Excluded)
		}
	}
}

// TestContaminationHoldoutAdmittedWithoutDates pins that a fresh private holdout is
// admitted even with no publish/cutoff dates: a holdout is unexposed by
// construction, so it does not need exposure dates to clear the gate.
func TestContaminationHoldoutAdmittedWithoutDates(t *testing.T) {
	hold := DefaultContaminationCorpus()[1] // the holdout case
	if !hold.Holdout || hold.PublishedAt != "" || hold.TrainCutoff != "" {
		t.Fatalf("fixture drift: expected a dateless holdout, got %+v", hold)
	}
	r := AuditContamination([]ContaminationCase{hold})
	if r.Verdict != VerdictContaminationClean || len(r.ConfirmatoryEligible) != 1 {
		t.Fatalf("dateless holdout audit = %q eligible=%v, want clean + 1 eligible", r.Verdict, r.ConfirmatoryEligible)
	}
}

// TestContaminationScrubbed pins that no artifact ever carries raw eval content:
// the audit input has no raw-text field and the artifact exposes only the content
// hash and metadata, so the emitted report is publishable without leaking the eval.
func TestContaminationScrubbed(t *testing.T) {
	r := AuditContamination(DefaultContaminationCorpus())
	js, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	// The hash prefix is the only stand-in for content; assert it is present and no
	// case ever gained a raw prompt/expected field by regression.
	if !bytes.Contains(js, []byte("sha256:decode-01")) {
		t.Fatalf("report is missing the content-hash stand-in for scrubbed replay")
	}
	for _, banned := range []string{`"prompt"`, `"expected"`, `"raw"`, `"completion"`} {
		if bytes.Contains(js, []byte(banned)) {
			t.Errorf("report leaked a raw-content field %s — replay artifact is not scrubbed", banned)
		}
	}
}
