package modelroute

import (
	"reflect"
	"strings"
	"testing"
)

func clusterReceipt(provider, family, harness, effort string, verdict CrossAuditVerdict, reason string) IssueAuditReceipt {
	return IssueAuditReceipt{
		Author:  AuditIdentity{Provider: provider, Family: family, Harness: harness, ReasoningPosture: effort},
		Verdict: verdict,
		Reason:  reason,
	}
}

func findClusterRow(t *testing.T, res AuditClusterResult, mech MechanismClass, provider, family string) AuditClusterRow {
	t.Helper()
	for _, row := range res.Rows {
		if row.Mechanism == mech && row.Provider == provider && row.Family == family {
			return row
		}
	}
	t.Fatalf("no cluster row for (%s, %s, %s); rows=%+v", mech, provider, family, res.Rows)
	return AuditClusterRow{}
}

// TestAuditFailureClusters is the acceptance gate for #3855: synthetic receipts
// with known clusters produce stable rows; low-N and confounded groups are
// marked insufficient; and removing author provenance prevents a model-specific
// claim.
func TestAuditFailureClusters(t *testing.T) {
	const (
		silentReason     = "silently omitted the required migration file"
		incompleteReason = "incomplete fix; did not fully satisfy the acceptance criteria"
		staleReason      = "overwrote a peer's stale block during the merge"
	)

	t.Run("known_clusters_are_stable", func(t *testing.T) {
		var rs []IssueAuditReceipt
		// anthropic/opus: 3 silent-omission findings across DIVERSE harnesses
		// (not confoundable) plus 3 clean audits.
		rs = append(rs,
			clusterReceipt("anthropic", "opus", "claude-code", "", CrossAuditRefute, silentReason),
			clusterReceipt("anthropic", "opus", "cursor", "", CrossAuditRefute, silentReason),
			clusterReceipt("anthropic", "opus", "aider", "", CrossAuditRefute, silentReason),
			clusterReceipt("anthropic", "opus", "claude-code", "", CrossAuditPass, "clean"),
			clusterReceipt("anthropic", "opus", "cursor", "", CrossAuditPass, "clean"),
			clusterReceipt("anthropic", "opus", "aider", "", CrossAuditPass, "clean"),
		)
		// anthropic/sonnet: 4 clean audits, no findings — base-rate contrast.
		for i := 0; i < 4; i++ {
			rs = append(rs, clusterReceipt("anthropic", "sonnet", "claude-code", "", CrossAuditPass, "clean"))
		}

		res := ClusterAuditFailures(rs, DefaultClusterConfig())
		if res.TotalReceipts != 10 || res.TotalFindings != 3 {
			t.Fatalf("totals: receipts=%d findings=%d, want 10/3", res.TotalReceipts, res.TotalFindings)
		}
		row := findClusterRow(t, res, MechanismSilentOmission, "anthropic", "opus")
		if row.Findings != 3 || row.ProvenanceAudits != 6 {
			t.Fatalf("opus counts findings=%d audits=%d, want 3/6", row.Findings, row.ProvenanceAudits)
		}
		if row.GroupRatePermille != 500 || row.BaseRatePermille != 300 {
			t.Fatalf("opus rates group=%d base=%d, want 500/300", row.GroupRatePermille, row.BaseRatePermille)
		}
		if row.BaseFindings != 3 || row.BaseAudits != 10 {
			t.Fatalf("opus base findings=%d audits=%d, want 3/10", row.BaseFindings, row.BaseAudits)
		}
		if row.Insufficient || row.Confounded {
			t.Fatalf("opus row should be sufficient & unconfounded: %+v", row)
		}
		if got := row.Harnesses; !reflect.DeepEqual(got, []string{"aider", "claude-code", "cursor"}) {
			t.Fatalf("opus harnesses = %v", got)
		}
		// Determinism: a re-fold of the same receipts yields identical output.
		if !reflect.DeepEqual(res, ClusterAuditFailures(rs, DefaultClusterConfig())) {
			t.Fatal("fold is not deterministic across identical inputs")
		}
		// A sufficient, above-base-rate group emits exactly one route proposal.
		if len(res.Proposals) != 1 || res.Proposals[0].Family != "opus" {
			t.Fatalf("proposals = %+v, want one for opus", res.Proposals)
		}
	})

	t.Run("low_n_group_is_insufficient", func(t *testing.T) {
		var rs []IssueAuditReceipt
		rs = append(rs, clusterReceipt("mistral", "mistral-lg", "vllm", "", CrossAuditRefute, staleReason))
		for i := 0; i < 5; i++ {
			rs = append(rs, clusterReceipt("mistral", "mistral-lg", "vllm", "", CrossAuditPass, "clean"))
		}
		res := ClusterAuditFailures(rs, DefaultClusterConfig())
		row := findClusterRow(t, res, MechanismStaleContext, "mistral", "mistral-lg")
		if !row.Insufficient {
			t.Fatalf("single-finding group must be insufficient: %+v", row)
		}
		if !reflect.DeepEqual(row.InsufficientReasons, []string{ClusterInsufficientLowSample}) {
			t.Fatalf("reasons = %v, want only LOW_SAMPLE", row.InsufficientReasons)
		}
		if len(res.Proposals) != 0 {
			t.Fatalf("insufficient group must not propose: %+v", res.Proposals)
		}
	})

	t.Run("confounded_group_is_insufficient", func(t *testing.T) {
		var rs []IssueAuditReceipt
		// openai/gpt: 4 incomplete-fix findings, ALL on harness "codex" at effort
		// "high" — enough samples, but the axis co-varies perfectly with the model.
		for i := 0; i < 4; i++ {
			rs = append(rs, clusterReceipt("openai", "gpt", "codex", "high", CrossAuditRefute, incompleteReason))
		}
		rs = append(rs, clusterReceipt("openai", "gpt", "codex", "high", CrossAuditPass, "clean"))
		// meta/llama: 3 incomplete-fix findings across diverse harnesses — the
		// second family that makes "codex" unique to gpt.
		rs = append(rs,
			clusterReceipt("meta", "llama", "ollama", "", CrossAuditRefute, incompleteReason),
			clusterReceipt("meta", "llama", "llamacpp", "", CrossAuditRefute, incompleteReason),
			clusterReceipt("meta", "llama", "vllm", "", CrossAuditRefute, incompleteReason),
		)
		for i := 0; i < 3; i++ {
			rs = append(rs, clusterReceipt("meta", "llama", "ollama", "", CrossAuditPass, "clean"))
		}

		res := ClusterAuditFailures(rs, DefaultClusterConfig())
		gpt := findClusterRow(t, res, MechanismIncompleteFix, "openai", "gpt")
		if !gpt.Confounded {
			t.Fatalf("gpt group must be confounded (codex unique): %+v", gpt)
		}
		if gpt.Findings < DefaultClusterConfig().MinFindings || gpt.ProvenanceAudits < DefaultClusterConfig().MinAudits {
			t.Fatalf("gpt sample floors should be met so CONFOUNDED is the sole reason: %+v", gpt)
		}
		if !reflect.DeepEqual(gpt.InsufficientReasons, []string{ClusterInsufficientConfounded}) {
			t.Fatalf("gpt reasons = %v, want only CONFOUNDED", gpt.InsufficientReasons)
		}
		if len(gpt.ConfounderNotes) != 2 {
			t.Fatalf("gpt should note harness + effort confounds: %v", gpt.ConfounderNotes)
		}
		// A confounded group is never proposed as a route change.
		for _, p := range res.Proposals {
			if p.Provider == "openai" && p.Family == "gpt" {
				t.Fatalf("confounded gpt group must not be proposed: %+v", res.Proposals)
			}
		}
	})

	// A PARTIALLY redacted receipt — one provenance axis stripped, the other
	// kept — must not support a model-specific claim either. Otherwise a
	// redaction that drops only the provider still lets a family-keyed claim
	// escape, and the proposal renders a dangling "Provenance /opus".
	t.Run("partial_provenance_is_insufficient", func(t *testing.T) {
		// build makes a group that clears every other floor: 3 findings across
		// diverse harnesses over 6 audits, above a base rate diluted by a fully
		// attributed contrast family.
		build := func(provider, family string) []IssueAuditReceipt {
			var rs []IssueAuditReceipt
			rs = append(rs,
				clusterReceipt(provider, family, "claude-code", "", CrossAuditRefute, silentReason),
				clusterReceipt(provider, family, "cursor", "", CrossAuditRefute, silentReason),
				clusterReceipt(provider, family, "aider", "", CrossAuditRefute, silentReason),
			)
			for i := 0; i < 3; i++ {
				rs = append(rs, clusterReceipt(provider, family, "claude-code", "", CrossAuditPass, "clean"))
			}
			for i := 0; i < 4; i++ {
				rs = append(rs, clusterReceipt("anthropic", "sonnet", "claude-code", "", CrossAuditPass, "clean"))
			}
			return rs
		}

		for _, c := range []struct{ name, provider, family string }{
			{"family_only", "", "opus"},
			{"provider_only", "anthropic", ""},
		} {
			t.Run(c.name, func(t *testing.T) {
				res := ClusterAuditFailures(build(c.provider, c.family), DefaultClusterConfig())
				row := findClusterRow(t, res, MechanismSilentOmission, c.provider, c.family)
				if row.Findings < DefaultClusterConfig().MinFindings || row.ProvenanceAudits < DefaultClusterConfig().MinAudits {
					t.Fatalf("fixture should clear the sample floors so provenance is the sole reason: %+v", row)
				}
				if !row.Insufficient {
					t.Fatalf("partially attributed row must be insufficient: %+v", row)
				}
				if !containsStr(row.InsufficientReasons, ClusterInsufficientPartialProvenance) {
					t.Fatalf("reasons = %v, want %s", row.InsufficientReasons, ClusterInsufficientPartialProvenance)
				}
				for _, p := range res.Proposals {
					if p.Provider == c.provider && p.Family == c.family {
						t.Fatalf("partial provenance must not yield a model-specific proposal: %+v", p)
					}
				}
			})
		}
	})

	t.Run("stripping_provenance_prevents_model_claim", func(t *testing.T) {
		// findFamily carries the findings; contrastFamily adds clean audits that
		// dilute the base rate so an attributed group clears it. Stripping
		// provenance collapses BOTH into a single unattributed bucket.
		build := func(findProvider, findFamily, contrastProvider, contrastFamily string) []IssueAuditReceipt {
			var rs []IssueAuditReceipt
			rs = append(rs,
				clusterReceipt(findProvider, findFamily, "claude-code", "", CrossAuditRefute, silentReason),
				clusterReceipt(findProvider, findFamily, "cursor", "", CrossAuditRefute, silentReason),
				clusterReceipt(findProvider, findFamily, "aider", "", CrossAuditRefute, silentReason),
			)
			for i := 0; i < 3; i++ {
				rs = append(rs, clusterReceipt(findProvider, findFamily, "claude-code", "", CrossAuditPass, "clean"))
			}
			for i := 0; i < 4; i++ {
				rs = append(rs, clusterReceipt(contrastProvider, contrastFamily, "claude-code", "", CrossAuditPass, "clean"))
			}
			return rs
		}

		attributed := ClusterAuditFailures(build("anthropic", "opus", "anthropic", "sonnet"), DefaultClusterConfig())
		if len(attributed.Proposals) == 0 {
			t.Fatal("attributed corpus should yield a model-specific proposal")
		}

		stripped := ClusterAuditFailures(build("", "", "", ""), DefaultClusterConfig())
		for _, row := range stripped.Rows {
			if row.Provider != "" || row.Family != "" {
				t.Fatalf("stripped corpus produced an attributed row: %+v", row)
			}
			if !row.Insufficient || !containsStr(row.InsufficientReasons, ClusterInsufficientNoProvenance) {
				t.Fatalf("unattributed row must be insufficient NO_PROVENANCE: %+v", row)
			}
		}
		if len(stripped.Proposals) != 0 {
			t.Fatalf("stripped corpus must make no proposal: %+v", stripped.Proposals)
		}
		// The rendered report carries no sufficient (claimable) cluster.
		report := RenderAuditClusterReport(stripped)
		if !strings.Contains(report, "### Sufficient clusters\n\n_none") {
			t.Fatalf("stripped report should show no sufficient clusters:\n%s", report)
		}
	})
}

// TestAuditClusterIntentVocabularyLint is the prose half of the acceptance gate:
// the reusable linter flags intent-attribution language, and a report rendered
// from receipts whose reasons are steeped in intent prose is itself intent-free
// because rows never echo the free-text reason.
func TestAuditClusterIntentVocabularyLint(t *testing.T) {
	viol := IntentVocabularyViolations("The model deliberately sabotaged the fix and lied about the omission.")
	for _, want := range []string{"deliberately", "lied", "sabotaged"} {
		if !containsStr(viol, want) {
			t.Fatalf("linter missed %q in %v", want, viol)
		}
	}
	if len(IntentVocabularyViolations("silently omitted a required migration file")) != 0 {
		t.Fatal("mechanism-only prose must not be flagged as intent")
	}

	// Receipts whose reasons are drenched in intent-attribution language.
	poison := "silently omitted the file; the author deliberately and maliciously sabotaged it — a dishonest omission"
	var rs []IssueAuditReceipt
	for i := 0; i < 3; i++ {
		rs = append(rs, clusterReceipt("anthropic", "opus", "claude-code", "", CrossAuditRefute, poison))
	}
	for i := 0; i < 3; i++ {
		rs = append(rs, clusterReceipt("anthropic", "opus", "claude-code", "", CrossAuditPass, "clean"))
	}
	report := RenderAuditClusterReport(ClusterAuditFailures(rs, DefaultClusterConfig()))
	if got := IntentVocabularyViolations(report); len(got) != 0 {
		t.Fatalf("generated report leaked intent vocabulary %v:\n%s", got, report)
	}
	if !strings.Contains(report, string(MechanismSilentOmission)) {
		t.Fatalf("report should still surface the mechanism class:\n%s", report)
	}
	if !strings.Contains(report, "correlation, not causation") {
		t.Fatalf("report must carry the correlation-not-causation fence:\n%s", report)
	}
}

func TestNormalizeMechanism(t *testing.T) {
	cases := []struct {
		reason string
		want   MechanismClass
	}{
		{"silently omitted the migration", MechanismSilentOmission},
		{"incomplete fix, criteria unaddressed", MechanismIncompleteFix},
		{"cited a test that does not exist", MechanismFabricatedEvidence},
		{"overwrote a peer's stale block", MechanismStaleContext},
		{"made an unrelated change out of scope", MechanismScopeViolation},
		{"regressed a passing test", MechanismRegressionIntroduced},
		{"claimed done with no witness", MechanismUnverifiedClaim},
		{"", MechanismUnclassified},
		{"some reason we have never classified", MechanismUnclassified},
	}
	for _, c := range cases {
		if got := NormalizeMechanism(c.reason); got != c.want {
			t.Errorf("NormalizeMechanism(%q) = %s, want %s", c.reason, got, c.want)
		}
	}
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}
