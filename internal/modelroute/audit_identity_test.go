package modelroute

import (
	"strings"
	"testing"
)

func TestAuditIdentityNormalizesAliasesAndPreservesUnknownAxes(t *testing.T) {
	policy := DefaultAuditIndependencePolicy()
	policy.RequiredAxes = []AuditIdentityAxis{AuditAxisProvider, AuditAxisFamily, AuditAxisModel, AuditAxisWeights, AuditAxisHarness, AuditAxisEffort, AuditAxisProvenance}
	policy.Aliases = []AuditIdentityAlias{
		{Alias: "claude-author", CanonicalModel: "claude-opus-4-6", Provider: "anthropic", Family: "claude", WeightsRevision: "claude-w46", ProvenanceSource: "registry:claude"},
		{Alias: " GPT-Frontier ", CanonicalModel: "gpt-5.6-sol", Provider: " OpenAI ", Family: " GPT ", WeightsRevision: "weights-2026-07", ProvenanceSource: "registry:gpt"},
	}
	decision := EvaluateAuditIndependence(
		AuditIdentity{Provider: "anthropic", Family: "claude", Model: "claude-author", Harness: "claude-code", ReasoningPosture: "high"},
		AuditIdentity{Model: "gpt-frontier", Harness: " CODEX ", ReasoningPosture: " XHIGH "},
		policy,
	)
	if decision.Verdict != AuditIndependenceUnknown || decision.Reason != AuditReasonUnknownRequiredAxis {
		t.Fatalf("decision = %+v", decision)
	}
	got := decision.Auditor
	if got.Model != "gpt-5.6-sol" || got.Provider != "openai" || got.Family != "gpt" || got.WeightsRevision != "weights-2026-07" {
		t.Fatalf("canonical auditor = %+v", got)
	}
	if got.Harness != "codex" || got.ReasoningPosture != "xhigh" || got.ProvenanceSource == "" {
		t.Fatalf("canonical axes were lost: %+v", got)
	}
	if got.EndpointClass != "" || got.AccountClass != "" {
		t.Fatalf("unknown axes were guessed: %+v", got)
	}
}

func TestEvaluateAuditIndependence(t *testing.T) {
	baseAuthor := fullAuditIdentity("claude-author", "anthropic", "claude", "claude-w46", "claude-code", "remote", "subscription", "high", "session:claude-author")
	baseAuditor := fullAuditIdentity("gpt-auditor", "openai", "gpt", "gpt-w56", "codex", "remote", "subscription", "xhigh", "session:codex-auditor")
	local := fullAuditIdentity("qwen-local", "local", "qwen", "qwen-w35", "fak-local", "local", "local", "high", "weights:qwen")
	sameWeights := fullAuditIdentity("renamed-auditor", "other-provider", "renamed-family", "claude-w46", "other", "remote", "api", "high", "registry:renamed")
	openrouterAuthor := fullAuditIdentity("openrouter-claude", "openrouter", "claude", "claude-w46", "gateway", "remote", "api", "high", "registry:or-claude")
	openrouterAuditor := fullAuditIdentity("openrouter-gpt", "openrouter", "gpt", "gpt-w56", "gateway", "remote", "api", "xhigh", "registry:or-gpt")
	aliases := []AuditIdentityAlias{
		auditAlias("claude-author", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
		auditAlias("gpt-auditor", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
		auditAlias("qwen-local", "qwen3.5-27b", "local", "qwen", "qwen-w35"),
		auditAlias("renamed-auditor", "renamed-model", "other-provider", "renamed-family", "claude-w46"),
		auditAlias("openrouter-claude", "claude-opus-4-6", "openrouter", "claude", "claude-w46"),
		auditAlias("openrouter-gpt", "gpt-5.6-sol", "openrouter", "gpt", "gpt-w56"),
	}
	defaultPolicy := DefaultAuditIndependencePolicy()
	defaultPolicy.Aliases = aliases
	highPolicy := HighRiskAuditIndependencePolicy()
	highPolicy.Aliases = aliases

	tests := []struct {
		name    string
		author  AuditIdentity
		auditor AuditIdentity
		policy  AuditIndependencePolicy
		verdict AuditIndependenceVerdict
		reason  AuditIndependenceReason
	}{
		{"claude-to-gpt", baseAuthor, baseAuditor, defaultPolicy, AuditIndependenceAdmit, AuditReasonAdmitIndependent},
		{"claude-to-local", baseAuthor, local, defaultPolicy, AuditIndependenceAdmit, AuditReasonAdmitIndependent},
		{"same-weights-different-name-family-provider", baseAuthor, sameWeights, defaultPolicy, AuditIndependenceRefuse, AuditReasonRefuseSameWeights},
		{"unknown-author", AuditIdentity{Provider: "unknown", Model: "unregistered-author"}, baseAuditor, defaultPolicy, AuditIndependenceUnknown, AuditReasonUnknownAliasUnresolved},
		{"default-allows-same-provider-different-family", openrouterAuthor, openrouterAuditor, defaultPolicy, AuditIndependenceAdmit, AuditReasonAdmitIndependent},
		{"high-risk-refuses-same-provider", openrouterAuthor, openrouterAuditor, highPolicy, AuditIndependenceRefuse, AuditReasonRefuseSameProviderHighRisk},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EvaluateAuditIndependence(tt.author, tt.auditor, tt.policy)
			if got.Verdict != tt.verdict || got.Reason != tt.reason {
				t.Fatalf("EvaluateAuditIndependence = %s/%s missing=%v, want %s/%s", got.Verdict, got.Reason, got.MissingAxes, tt.verdict, tt.reason)
			}
			if !got.Verdict.Valid() || !got.Reason.Valid() || got.PolicyDigest == "" {
				t.Fatalf("decision escaped closed/bound contract: %+v", got)
			}
		})
	}
}

func TestEvaluateAuditIndependenceRefusesRelabeledSameFamilyAliases(t *testing.T) {
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = []AuditIdentityAlias{
		auditAlias("gpt-author", "gpt-5.6-sol", "openai", "gpt", "weights-gpt-56"),
		auditAlias("totally-different-reviewer", "gpt-5.6-sol", "openai", "gpt", "weights-gpt-56"),
	}
	got := EvaluateAuditIndependence(AuditIdentity{Model: "gpt-author"}, AuditIdentity{Model: "totally-different-reviewer"}, policy)
	if got.Verdict != AuditIndependenceRefuse || got.Reason != AuditReasonRefuseSameWeights {
		t.Fatalf("alias laundering decision = %+v", got)
	}
	if got.Author.Model != got.Auditor.Model || got.Author.Family != "gpt" {
		t.Fatalf("aliases did not canonicalize onto same lineage: %+v", got)
	}
}

func TestEvaluateAuditIndependenceRefusesAliasLineageConflict(t *testing.T) {
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = []AuditIdentityAlias{
		auditAlias("claude-author", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
		auditAlias("gpt-alt", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
	}
	got := EvaluateAuditIndependence(
		AuditIdentity{Provider: "anthropic", Family: "claude", Model: "claude-author"},
		AuditIdentity{Provider: "anthropic", Family: "claude", Model: "gpt-alt"},
		policy,
	)
	if got.Verdict != AuditIndependenceRefuse || got.Reason != AuditReasonRefuseAliasConflict {
		t.Fatalf("conflicting alias decision = %+v", got)
	}
}

func TestAuditIdentityRequiredAxesStayUnknown(t *testing.T) {
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = []AuditIdentityAlias{
		auditAlias("claude-author", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
		{Alias: "gpt-auditor", CanonicalModel: "gpt-5.6-sol", Provider: "openai", Family: "gpt", WeightsRevision: "gpt-w56"},
	}
	got := EvaluateAuditIndependence(
		fullAuditIdentity("claude-author", "anthropic", "claude", "claude-w46", "claude-code", "remote", "subscription", "high", "session:a"),
		fullAuditIdentity("gpt-auditor", "openai", "gpt", "gpt-w56", "codex", "remote", "subscription", "", ""),
		policy,
	)
	if got.Verdict != AuditIndependenceUnknown || got.Reason != AuditReasonUnknownRequiredAxis {
		t.Fatalf("required-axis decision = %+v", got)
	}
	missing := strings.Join(got.MissingAxes, ",")
	if !strings.Contains(missing, "auditor.effort") || !strings.Contains(missing, "auditor.provenance_source") {
		t.Fatalf("missing axes = %v", got.MissingAxes)
	}
}

func TestAuditIdentityDriverRequiresRosterAndExactCanonicalFamily(t *testing.T) {
	aliases := []AuditIdentityAlias{
		auditAlias("claude-prod", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
		auditAlias("gpt-prod", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
	}
	if _, err := ValidateAuditDriverIdentity("claude", AuditIdentity{Provider: "anthropic", Family: "claude", Model: "claude-prod"}, nil); err == nil {
		t.Fatal("unregistered claude identity should not cross the driver boundary")
	}
	if _, err := ValidateAuditDriverIdentity("codex", AuditIdentity{Provider: "openai", Family: "gpt-fake", Model: "gpt-prod"}, aliases); err == nil {
		t.Fatal("gpt-fake declaration should conflict with the roster")
	}
	if got, err := ValidateAuditDriverIdentity("codex", AuditIdentity{Provider: "openai", Family: "gpt", Model: "gpt-prod"}, aliases); err != nil || got.Driver != "codex" || got.Model != "gpt-5.6-sol" {
		t.Fatalf("canonical codex identity = %+v err=%v", got, err)
	}
	if got, err := ValidateAuditDriverIdentity(" HTTP ", AuditIdentity{Provider: "openai", Family: "gpt", Model: "gpt-prod"}, aliases); err != nil || got.Driver != "http" || got.Model != "gpt-5.6-sol" {
		t.Fatalf("canonical HTTP identity = %+v err=%v", got, err)
	}
}

func TestAuditIdentityObservedHTTPRequiresRosterBackedLineage(t *testing.T) {
	expected := AuditIdentity{Provider: "openai", Family: "gpt", Model: "gpt-prod", WeightsRevision: "gpt-w56"}
	unknown := VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "gpt-prod"}, nil)
	if unknown.Verdict != AuditIndependenceUnknown || unknown.Reason != AuditReasonUnknownObservedIdentity {
		t.Fatalf("model-only HTTP response = %+v", unknown)
	}
	aliases := []AuditIdentityAlias{
		auditAlias("gpt-prod", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
		auditAlias("claude-prod", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
	}
	matched := VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "gpt-prod"}, aliases)
	if matched.Verdict != AuditIndependenceAdmit {
		t.Fatalf("roster-backed HTTP response = %+v", matched)
	}
	for _, observed := range []AuditIdentity{{}, {Model: "unmapped-model"}} {
		unknown = VerifyObservedAuditIdentity(expected, observed, aliases)
		if unknown.Verdict != AuditIndependenceUnknown || unknown.Reason != AuditReasonUnknownObservedIdentity {
			t.Fatalf("missing/unmapped HTTP response identity = %+v", unknown)
		}
	}
	mismatch := VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "claude-prod"}, aliases)
	if mismatch.Verdict != AuditIndependenceRefuse || mismatch.Reason != AuditReasonRefuseObservedMismatch {
		t.Fatalf("HTTP declared-family mismatch = %+v", mismatch)
	}
}

func TestAuditIdentityRejectsUnknownPolicyRiskAndAxis(t *testing.T) {
	author := fullAuditIdentity("claude-author", "anthropic", "claude", "claude-w46", "claude-code", "remote", "subscription", "high", "session:a")
	auditor := fullAuditIdentity("gpt-auditor", "openai", "gpt", "gpt-w56", "codex", "remote", "subscription", "xhigh", "session:b")
	aliases := []AuditIdentityAlias{
		auditAlias("claude-author", "claude-opus-4-6", "anthropic", "claude", "claude-w46"),
		auditAlias("gpt-auditor", "gpt-5.6-sol", "openai", "gpt", "gpt-w56"),
	}
	badRisk := DefaultAuditIndependencePolicy()
	badRisk.Risk = "critical"
	badRisk.Aliases = aliases
	if got := EvaluateAuditIndependence(author, auditor, badRisk); got.Verdict != AuditIndependenceUnknown || got.Reason != AuditReasonUnknownPolicyRisk {
		t.Fatalf("unknown risk = %+v", got)
	}
	badAxis := DefaultAuditIndependencePolicy()
	badAxis.RequiredAxes = append(badAxis.RequiredAxes, "made-up")
	badAxis.Aliases = aliases
	if got := EvaluateAuditIndependence(author, auditor, badAxis); got.Verdict != AuditIndependenceUnknown || got.Reason != AuditReasonUnknownPolicyAxis {
		t.Fatalf("unknown axis = %+v", got)
	}
}

func TestAuditIdentityPolicyDigestBindsRosterIndependentOfOrder(t *testing.T) {
	a := auditAlias("a", "model-a", "openai", "gpt", "wa")
	b := auditAlias("b", "model-b", "anthropic", "claude", "wb")
	p1 := DefaultAuditIndependencePolicy()
	p1.Aliases = []AuditIdentityAlias{a, b}
	p2 := DefaultAuditIndependencePolicy()
	p2.Aliases = []AuditIdentityAlias{b, a}
	if p1.Digest() == "" || p1.Digest() != p2.Digest() {
		t.Fatalf("policy digests differ by roster order: %q != %q", p1.Digest(), p2.Digest())
	}
	p2.Aliases[0].WeightsRevision = "changed"
	if p1.Digest() == p2.Digest() {
		t.Fatal("policy digest did not bind roster weights")
	}
}

func TestEvaluateAuditIndependenceConclusiveRefusalsDominateUnknownAxes(t *testing.T) {
	aliases := []AuditIdentityAlias{
		auditAlias("gpt-a", "gpt-5.4", "openai", "gpt", "wa"),
		auditAlias("gpt-b", "gpt-5.6-sol", "openai", "gpt", "wb"),
		auditAlias("other-same-weights", "other-model", "other", "other-family", "wa"),
		auditAlias("same-provider-other-family", "other-2", "openai", "qwen", "wq"),
	}
	policy := DefaultAuditIndependencePolicy()
	policy.Aliases = aliases
	tests := []struct {
		name    string
		author  AuditIdentity
		auditor AuditIdentity
		policy  AuditIndependencePolicy
		reason  AuditIndependenceReason
	}{
		{"same-family-missing-effort", AuditIdentity{Model: "gpt-a"}, AuditIdentity{Model: "gpt-b"}, policy, AuditReasonRefuseSameFamily},
		{"same-weights-missing-harness", AuditIdentity{Model: "gpt-a"}, AuditIdentity{Model: "other-same-weights"}, policy, AuditReasonRefuseSameWeights},
		{"unresolved-explicit-same-family", AuditIdentity{Model: "unregistered-a", Family: "gpt"}, AuditIdentity{Model: "unregistered-b", Family: "gpt"}, policy, AuditReasonRefuseSameFamily},
	}
	high := policy
	high.Risk = AuditRiskHigh
	tests = append(tests, struct {
		name    string
		author  AuditIdentity
		auditor AuditIdentity
		policy  AuditIndependencePolicy
		reason  AuditIndependenceReason
	}{"high-same-provider-missing-account", AuditIdentity{Model: "gpt-a"}, AuditIdentity{Model: "same-provider-other-family"}, high, AuditReasonRefuseSameProviderHighRisk})
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := EvaluateAuditIndependence(tt.author, tt.auditor, tt.policy)
			if got.Verdict != AuditIndependenceRefuse || got.Reason != tt.reason {
				t.Fatalf("decision = %+v, want REFUSE/%s", got, tt.reason)
			}
		})
	}
}

func TestAuditIdentityRosterRejectsMissingAndConflictingProvenance(t *testing.T) {
	missing := AuditIdentityRoster{Schema: AuditIdentityRosterSchema, Aliases: []AuditIdentityAlias{{Alias: "a", CanonicalModel: "m", Provider: "p", Family: "f", WeightsRevision: "w"}}}
	if err := missing.Validate(); err == nil || !strings.Contains(err.Error(), "provenance_source") {
		t.Fatalf("missing provenance error = %v", err)
	}
	conflict := AuditIdentityRoster{Schema: AuditIdentityRosterSchema, Aliases: []AuditIdentityAlias{
		auditAlias("a", "m1", "p", "f", "w1"),
		auditAlias("a", "m2", "p", "f", "w2"),
	}}
	if err := conflict.Validate(); err == nil || !strings.Contains(err.Error(), "conflicting") {
		t.Fatalf("conflicting alias error = %v", err)
	}
}

func TestAuditIdentityCanonicalObservationIsRosterOrderDeterministic(t *testing.T) {
	a := auditAlias("gpt-primary", "gpt-5.6-sol", "openai", "gpt", "w56")
	b := auditAlias("gpt-secondary", "gpt-5.6-sol", "openai", "gpt", "w56")
	b.ProvenanceSource = a.ProvenanceSource
	expected := AuditIdentity{Model: "gpt-primary", Provider: "openai", Family: "gpt", WeightsRevision: "w56"}
	d1 := VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "gpt-5.6-sol"}, []AuditIdentityAlias{a, b})
	d2 := VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "gpt-5.6-sol"}, []AuditIdentityAlias{b, a})
	if d1.Verdict != d2.Verdict || d1.Reason != d2.Reason || d1.Auditor != d2.Auditor || d1.PolicyDigest != d2.PolicyDigest {
		t.Fatalf("roster order changed observation: d1=%+v d2=%+v", d1, d2)
	}
	b.ProvenanceSource = "registry:different"
	d1 = VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "gpt-5.6-sol"}, []AuditIdentityAlias{a, b})
	d2 = VerifyObservedAuditIdentity(expected, AuditIdentity{Model: "gpt-5.6-sol"}, []AuditIdentityAlias{b, a})
	if d1.Verdict != AuditIndependenceUnknown || d1.Reason != AuditReasonUnknownAliasAmbiguous || d1.Verdict != d2.Verdict || d1.Reason != d2.Reason {
		t.Fatalf("conflicting provenance was order-sensitive: d1=%+v d2=%+v", d1, d2)
	}
}

func fullAuditIdentity(model, provider, family, weights, harness, endpoint, account, effort, provenance string) AuditIdentity {
	return AuditIdentity{
		Model: model, Provider: provider, Family: family, WeightsRevision: weights,
		Harness: harness, EndpointClass: endpoint, AccountClass: account,
		ReasoningPosture: effort, ProvenanceSource: provenance,
	}
}

func auditAlias(alias, canonical, provider, family, weights string) AuditIdentityAlias {
	return AuditIdentityAlias{
		Alias: alias, CanonicalModel: canonical, Provider: provider, Family: family,
		WeightsRevision: weights, ProvenanceSource: "registry:" + alias,
	}
}
