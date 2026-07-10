package modelroute

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func auditRouteV1Alias(alias, canonical, provider, family, weights string) AuditIdentityAlias {
	return AuditIdentityAlias{
		Alias: alias, CanonicalModel: canonical, Provider: provider, Family: family,
		WeightsRevision: weights, ProvenanceSource: "registry:" + alias,
	}
}

func auditRouteV1Identity(model, provider, family, weights, harness, endpoint, account, effort string) AuditIdentity {
	driver := "http"
	if provider == "anthropic" && family == "claude" {
		driver = "claude"
	}
	if provider == "openai" && family == "gpt" {
		driver = "codex"
	}
	return AuditIdentity{
		Model: model, Provider: provider, Family: family, WeightsRevision: weights,
		Harness: harness, EndpointClass: endpoint, AccountClass: account, ReasoningPosture: effort, Driver: driver,
	}
}

func auditRouteV1Fixtures() (map[string]AuditIdentity, map[string]AuditAuditorConfig, []AuditIdentityAlias) {
	aliases := []AuditIdentityAlias{
		auditRouteV1Alias("claude-author", "claude-author-canonical", "anthropic", "claude", "claude-wa"),
		auditRouteV1Alias("gpt-author", "gpt-author-canonical", "openai", "gpt", "gpt-wa"),
		auditRouteV1Alias("local-author", "local-author-canonical", "local", "qwen", "qwen-wa"),
		auditRouteV1Alias("gpt-review", "gpt-review-configured", "openai", "gpt", "gpt-wr"),
		auditRouteV1Alias("gpt-alias", "gpt-alias-configured", "openai", "gpt", "gpt-wr2"),
		auditRouteV1Alias("claude-review", "claude-review-configured", "anthropic", "claude", "claude-wr"),
		auditRouteV1Alias("local-review", "local-open-review", "local", "qwen", "qwen-wr"),
	}
	authors := map[string]AuditIdentity{
		"claude":  auditRouteV1Identity("claude-author", "anthropic", "claude", "claude-wa", "claude-code", "hosted", "subscription", "high"),
		"gpt":     auditRouteV1Identity("gpt-author", "openai", "gpt", "gpt-wa", "codex", "hosted", "subscription", "xhigh"),
		"local":   auditRouteV1Identity("local-author", "local", "qwen", "qwen-wa", "local-runner", "local", "local", "high"),
		"unknown": {Model: "unregistered-author"},
	}
	candidates := map[string]AuditAuditorConfig{
		"gpt-primary": {
			ID: "gpt-primary", Identity: auditRouteV1Identity("gpt-review", "openai", "gpt", "gpt-wr", "codex", "hosted", "subscription", "xhigh"),
			Capability: TierT0, CapabilitySource: "benchmark:gpt-frontier", Price: AuditRoutePrice{InputMicrosPerMillionTokens: 1_000_000, OutputMicrosPerMillionTokens: 2_000_000},
		},
		"gpt-alias": {
			ID: "gpt-alias", Identity: auditRouteV1Identity("gpt-alias", "openai", "gpt", "gpt-wr2", "codex", "hosted", "subscription", "xhigh"),
			Capability: TierT0, CapabilitySource: "benchmark:gpt-frontier-alt", Price: AuditRoutePrice{InputMicrosPerMillionTokens: 3_000_000, OutputMicrosPerMillionTokens: 6_000_000},
		},
		"claude-primary": {
			ID: "claude-primary", Identity: auditRouteV1Identity("claude-review", "anthropic", "claude", "claude-wr", "claude-code", "hosted", "subscription", "xhigh"),
			Capability: TierT0, CapabilitySource: "benchmark:claude-frontier", Price: AuditRoutePrice{InputMicrosPerMillionTokens: 2_000_000, OutputMicrosPerMillionTokens: 4_000_000},
		},
		"local-open": {
			ID: "local-open", Identity: auditRouteV1Identity("local-review", "local", "qwen", "qwen-wr", "local-runner", "local", "local", "high"),
			Capability: TierT1, CapabilitySource: "benchmark:local-implementation", Price: AuditRoutePrice{},
		},
	}
	return authors, candidates, aliases
}

func auditRouteV1Roster(ids ...string) AuditRouteRoster {
	_, candidates, aliases := auditRouteV1Fixtures()
	roster := AuditRouteRoster{
		Schema: AuditRouteRosterSchema, Aliases: aliases, UnknownAuthorQuorum: 2,
		EstimatedInputTokens: 1_000, EstimatedOutputTokens: 500,
	}
	for _, id := range ids {
		roster.Candidates = append(roster.Candidates, candidates[id])
	}
	return roster
}

func auditRouteV1Health(roster AuditRouteRoster) AuditRouteHealth {
	health := AuditRouteHealth{
		Schema: AuditRouteHealthSchema,
		Providers: map[string]AuditProviderHealthStatus{
			"openai": AuditProviderHealthy, "anthropic": AuditProviderHealthy, "local": AuditProviderHealthy,
		},
		Capacity: map[string]AuditCapacityStatus{},
		Cooldown: map[string]AuditCooldownStatus{},
	}
	for _, candidate := range roster.Candidates {
		health.Capacity[candidate.ID] = AuditCapacityAvailable
		health.Cooldown[candidate.ID] = AuditCooldownReady
	}
	return health
}

func auditRouteV1CandidateIDs(plan AuditIssuePlan) []string {
	ids := make([]string, 0, len(plan.Candidates))
	for _, candidate := range plan.Candidates {
		ids = append(ids, candidate.CandidateID)
	}
	return ids
}

func auditRouteV1Considered(plan AuditIssuePlan, id string) (AuditRouteCandidateDecision, bool) {
	for _, candidate := range plan.Considered {
		if candidate.CandidateID == id {
			return candidate, true
		}
	}
	return AuditRouteCandidateDecision{}, false
}

func TestPlanIssueAuditReciprocalMatrix(t *testing.T) {
	authors, _, _ := auditRouteV1Fixtures()
	roster := auditRouteV1Roster("gpt-primary", "gpt-alias", "claude-primary", "local-open")
	health := auditRouteV1Health(roster)
	tests := []struct {
		name             string
		author           AuditIdentity
		risk             AuditRisk
		wantPrimary      string
		wantFallback     string
		wantQuorum       int
		wantUnknown      bool
		wantIndependence AuditIndependenceVerdict
	}{
		{name: "claude-to-configured-gpt-xhigh-then-local", author: authors["claude"], risk: AuditRiskDefault, wantPrimary: "gpt-primary", wantFallback: "local-open", wantQuorum: 1, wantIndependence: AuditIndependenceAdmit},
		{name: "gpt-codex-to-claude-then-local", author: authors["gpt"], risk: AuditRiskDefault, wantPrimary: "claude-primary", wantFallback: "local-open", wantQuorum: 1, wantIndependence: AuditIndependenceAdmit},
		{name: "local-to-frontier", author: authors["local"], risk: AuditRiskDefault, wantPrimary: "gpt-primary", wantFallback: "claude-primary", wantQuorum: 1, wantIndependence: AuditIndependenceAdmit},
		{name: "unknown-to-diversified-quorum", author: authors["unknown"], risk: AuditRiskDefault, wantPrimary: "local-open", wantFallback: "gpt-primary", wantQuorum: 2, wantUnknown: true, wantIndependence: AuditIndependenceUnknown},
		{name: "high-risk-requires-frontier-xhigh", author: authors["claude"], risk: AuditRiskHigh, wantPrimary: "gpt-primary", wantFallback: "gpt-alias", wantQuorum: 1, wantIndependence: AuditIndependenceAdmit},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, err := PlanIssueAudit(tt.author, tt.risk, roster, health)
			if err != nil {
				t.Fatalf("PlanIssueAudit: %v plan=%+v", err, plan)
			}
			ids := auditRouteV1CandidateIDs(plan)
			if len(ids) < 2 || ids[0] != tt.wantPrimary || !contains(ids, tt.wantFallback) {
				t.Fatalf("candidate order = %v, want primary %s and fallback %s", ids, tt.wantPrimary, tt.wantFallback)
			}
			if plan.RequiredQuorum != tt.wantQuorum || plan.AuthorUnknown != tt.wantUnknown {
				t.Fatalf("quorum/unknown = %d/%v, want %d/%v", plan.RequiredQuorum, plan.AuthorUnknown, tt.wantQuorum, tt.wantUnknown)
			}
			if plan.Candidates[0].Independence.Verdict != tt.wantIndependence || !plan.Candidates[0].Independence.Reason.Valid() {
				t.Fatalf("primary independence = %+v", plan.Candidates[0].Independence)
			}
			if plan.Candidates[0].ActualEffort == "" || plan.Candidates[0].EstimatedCostMicrosUSD < 0 || plan.PolicyDigest == "" {
				t.Fatalf("plan lost effort/cost/policy: %+v", plan.Candidates[0])
			}
			if tt.wantUnknown {
				seenProviders, seenFamilies := map[string]bool{}, map[string]bool{}
				for _, candidate := range plan.Candidates[:plan.RequiredQuorum] {
					if seenProviders[candidate.Identity.Provider] || seenFamilies[candidate.Identity.Family] {
						t.Fatalf("quorum is not diversified: %+v", plan.Candidates[:plan.RequiredQuorum])
					}
					seenProviders[candidate.Identity.Provider] = true
					seenFamilies[candidate.Identity.Family] = true
				}
			}
		})
	}
}

func TestPlanIssueAuditSkipsSameFamilyAliasesAndUnhealthyProviders(t *testing.T) {
	authors, _, _ := auditRouteV1Fixtures()
	roster := auditRouteV1Roster("gpt-primary", "gpt-alias", "claude-primary", "local-open")
	health := auditRouteV1Health(roster)

	gptPlan, err := PlanIssueAudit(authors["gpt"], AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"gpt-primary", "gpt-alias"} {
		decision, ok := auditRouteV1Considered(gptPlan, id)
		if !ok || decision.Admitted || decision.SkipReason != AuditSkipReciprocalRule {
			t.Fatalf("same-family alias %s was not skipped: %+v", id, decision)
		}
	}

	health.Providers["openai"] = AuditProviderUnhealthy
	claudePlan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatal(err)
	}
	if got := claudePlan.Candidates[0].CandidateID; got != "local-open" {
		t.Fatalf("unhealthy OpenAI provider primary = %q, want local-open", got)
	}
	for _, id := range []string{"gpt-primary", "gpt-alias"} {
		decision, _ := auditRouteV1Considered(claudePlan, id)
		if decision.SkipReason != AuditSkipProviderUnhealthy {
			t.Fatalf("unhealthy provider decision %s = %+v", id, decision)
		}
	}
}

func TestPlanIssueAuditHealthCapacityCostAndRiskFloors(t *testing.T) {
	authors, candidates, aliases := auditRouteV1Fixtures()
	t.Run("cost-orders-same-preference", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-alias", "gpt-primary", "local-open")
		plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		if err != nil || plan.Candidates[0].CandidateID != "gpt-primary" {
			t.Fatalf("cost-aware order = %v err=%v", auditRouteV1CandidateIDs(plan), err)
		}
	})
	t.Run("capacity-fallback", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary", "local-open")
		health := auditRouteV1Health(roster)
		health.Capacity["gpt-primary"] = AuditCapacitySaturated
		plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
		if err != nil || plan.Candidates[0].CandidateID != "local-open" {
			t.Fatalf("capacity fallback = %v err=%v", auditRouteV1CandidateIDs(plan), err)
		}
	})
	t.Run("high-risk-effort-floor", func(t *testing.T) {
		candidate := candidates["gpt-primary"]
		candidate.Identity.ReasoningPosture = "high"
		roster := AuditRouteRoster{
			Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate},
			EstimatedInputTokens: 100, EstimatedOutputTokens: 100,
		}
		_, err := PlanIssueAudit(authors["claude"], AuditRiskHigh, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteEffortFloor {
			t.Fatalf("high-risk effort error = %#v", err)
		}
	})
	t.Run("high-risk-tier-floor", func(t *testing.T) {
		roster := auditRouteV1Roster("local-open")
		_, err := PlanIssueAudit(authors["claude"], AuditRiskHigh, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteTierFloor {
			t.Fatalf("high-risk tier error = %#v", err)
		}
	})
}

func TestPlanIssueAuditTypedNoRouteRefusal(t *testing.T) {
	authors, _, _ := auditRouteV1Fixtures()
	roster := auditRouteV1Roster("gpt-primary", "gpt-alias")
	plan, err := PlanIssueAudit(authors["gpt"], AuditRiskDefault, roster, auditRouteV1Health(roster))
	var noRoute *AuditNoRouteError
	if !errors.As(err, &noRoute) || !IsAuditNoRoute(err) || !noRoute.Reason.Valid() || noRoute.Reason != AuditNoRouteNoIndependent {
		t.Fatalf("no-route error = %#v", err)
	}
	if len(plan.Considered) != 2 || len(noRoute.Plan.Considered) != 2 {
		t.Fatalf("no-route lost considered diagnostics: plan=%+v error=%+v", plan, noRoute)
	}
}

func TestPlanIssueAuditPartialAuthorStillRefusesKnownCollisions(t *testing.T) {
	_, candidates, aliases := auditRouteV1Fixtures()
	tests := []struct {
		name, weights string
		wantReason    AuditIndependenceReason
	}{
		{name: "same-weights", weights: "gpt-wr", wantReason: AuditReasonRefuseSameWeights},
		{name: "same-family", weights: "other-gpt-weights", wantReason: AuditReasonRefuseSameFamily},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			partialAlias := auditRouteV1Alias("partial-author", "partial-author-canonical", "other-provider", "gpt", tt.weights)
			roster := AuditRouteRoster{
				Schema: AuditRouteRosterSchema, Aliases: append(append([]AuditIdentityAlias(nil), aliases...), partialAlias),
				Candidates: []AuditAuditorConfig{candidates["gpt-primary"]}, UnknownAuthorQuorum: 2,
				EstimatedInputTokens: 100, EstimatedOutputTokens: 100,
			}
			author := AuditIdentity{Model: "partial-author", Family: "gpt", WeightsRevision: tt.weights}
			plan, err := PlanIssueAudit(author, AuditRiskDefault, roster, auditRouteV1Health(roster))
			if err == nil {
				t.Fatalf("partial author collision routed: %+v", plan)
			}
			decision, ok := auditRouteV1Considered(plan, "gpt-primary")
			if !ok || decision.SkipReason != AuditSkipIndependence || decision.Independence.Verdict != AuditIndependenceRefuse || decision.Independence.Reason != tt.wantReason {
				t.Fatalf("partial collision decision = %+v", decision)
			}
		})
	}
}

func TestPlanIssueAuditQuorumSearchFindsDiversifiedSubset(t *testing.T) {
	aliases := []AuditIdentityAlias{
		auditRouteV1Alias("a", "model-a", "p1", "f1", "w1"),
		auditRouteV1Alias("b", "model-b", "p1", "f2", "w2"),
		auditRouteV1Alias("c", "model-c", "p2", "f1", "w3"),
		auditRouteV1Alias("d", "model-d", "p2", "f2", "w4"),
	}
	candidate := func(id, provider, family, weights string, price int64) AuditAuditorConfig {
		return AuditAuditorConfig{
			ID: id, Identity: auditRouteV1Identity(id, provider, family, weights, "http-reviewer", "hosted", "api", "high"),
			Capability: TierT1, CapabilitySource: "benchmark:" + id,
			Price: AuditRoutePrice{InputMicrosPerMillionTokens: price, OutputMicrosPerMillionTokens: price},
		}
	}
	roster := AuditRouteRoster{
		Schema: AuditRouteRosterSchema, Aliases: aliases, UnknownAuthorQuorum: 2,
		Candidates:           []AuditAuditorConfig{candidate("a", "p1", "f1", "w1", 1), candidate("d", "p2", "f2", "w4", 100), candidate("b", "p1", "f2", "w2", 2), candidate("c", "p2", "f1", "w3", 3)},
		EstimatedInputTokens: 1_000_000,
	}
	health := AuditRouteHealth{
		Schema:    AuditRouteHealthSchema,
		Providers: map[string]AuditProviderHealthStatus{"p1": AuditProviderHealthy, "p2": AuditProviderHealthy},
		Capacity:  map[string]AuditCapacityStatus{"a": AuditCapacityAvailable, "b": AuditCapacityAvailable, "c": AuditCapacityAvailable, "d": AuditCapacityAvailable},
		Cooldown:  map[string]AuditCooldownStatus{"a": AuditCooldownReady, "b": AuditCooldownReady, "c": AuditCooldownReady, "d": AuditCooldownReady},
	}
	plan, err := PlanIssueAudit(AuditIdentity{Model: "unknown"}, AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatalf("backtracking quorum: %v plan=%+v", err, plan)
	}
	if got := auditRouteV1CandidateIDs(plan); len(got) != 4 || !reflect.DeepEqual(got[:2], []string{"b", "c"}) {
		t.Fatalf("diversified primary+fallback order = %v, want [b c] primary group with truthful alternatives", got)
	}
	if len(plan.QuorumGroups) < 2 || !reflect.DeepEqual(plan.QuorumGroups[0].CandidateIDs, []string{"b", "c"}) {
		t.Fatalf("quorum groups = %+v", plan.QuorumGroups)
	}
	for _, id := range []string{"a", "d"} {
		decision, _ := auditRouteV1Considered(plan, id)
		if !decision.Admitted || decision.SkipReason != AuditSkipNone {
			t.Fatalf("truthful fallback %s = %+v", id, decision)
		}
	}

	sameWeights := roster
	sameWeights.Aliases = []AuditIdentityAlias{
		auditRouteV1Alias("a", "model-a", "p1", "f1", "same"),
		auditRouteV1Alias("b", "model-b", "p2", "f2", "same"),
	}
	sameWeights.Candidates = []AuditAuditorConfig{candidate("a", "p1", "f1", "same", 1), candidate("b", "p2", "f2", "same", 2)}
	sameHealth := AuditRouteHealth{
		Schema:    AuditRouteHealthSchema,
		Providers: map[string]AuditProviderHealthStatus{"p1": AuditProviderHealthy, "p2": AuditProviderHealthy},
		Capacity:  map[string]AuditCapacityStatus{"a": AuditCapacityAvailable, "b": AuditCapacityAvailable},
		Cooldown:  map[string]AuditCooldownStatus{"a": AuditCooldownReady, "b": AuditCooldownReady},
	}
	if _, err := PlanIssueAudit(AuditIdentity{Model: "unknown"}, AuditRiskDefault, sameWeights, sameHealth); err == nil {
		t.Fatal("same-weights aliases formed a fake diversified quorum")
	}
}

func TestPlanIssueAuditQuorumRanksHealthThenPriorityThenCost(t *testing.T) {
	aliases := []AuditIdentityAlias{}
	var candidates []AuditAuditorConfig
	for i, spec := range []struct {
		id, provider, family string
		price, priority      int64
	}{
		{id: "a", provider: "p1", family: "f1", price: 1, priority: 5},
		{id: "b", provider: "p2", family: "f2", price: 1, priority: 5},
		{id: "c", provider: "p3", family: "f3", price: 10, priority: 0},
		{id: "d", provider: "p4", family: "f4", price: 10, priority: 0},
	} {
		weights := "w" + spec.id
		aliases = append(aliases, auditRouteV1Alias(spec.id, "model-"+spec.id, spec.provider, spec.family, weights))
		candidates = append(candidates, AuditAuditorConfig{
			ID: spec.id, Identity: auditRouteV1Identity(spec.id, spec.provider, spec.family, weights, "http-reviewer", "hosted", "api", "high"),
			Capability: TierT1, CapabilitySource: "benchmark:" + spec.id, Priority: int(spec.priority),
			Price: AuditRoutePrice{InputMicrosPerMillionTokens: spec.price, OutputMicrosPerMillionTokens: spec.price},
		})
		_ = i
	}
	roster := AuditRouteRoster{
		Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: candidates, UnknownAuthorQuorum: 2, EstimatedInputTokens: 1_000_000,
	}
	health := AuditRouteHealth{
		Schema: AuditRouteHealthSchema,
		Providers: map[string]AuditProviderHealthStatus{
			"p1": AuditProviderDegraded, "p2": AuditProviderDegraded, "p3": AuditProviderHealthy, "p4": AuditProviderHealthy,
		},
		Capacity: map[string]AuditCapacityStatus{"a": AuditCapacityAvailable, "b": AuditCapacityAvailable, "c": AuditCapacityAvailable, "d": AuditCapacityAvailable},
		Cooldown: map[string]AuditCooldownStatus{"a": AuditCooldownReady, "b": AuditCooldownReady, "c": AuditCooldownReady, "d": AuditCooldownReady},
	}
	plan, err := PlanIssueAudit(AuditIdentity{Model: "unknown"}, AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.QuorumGroups[0].CandidateIDs; !reflect.DeepEqual(got, []string{"c", "d"}) {
		t.Fatalf("healthy quorum did not beat cheap degraded quorum: %+v", plan.QuorumGroups)
	}

	for provider := range health.Providers {
		health.Providers[provider] = AuditProviderHealthy
	}
	plan, err = PlanIssueAudit(AuditIdentity{Model: "unknown"}, AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.QuorumGroups[0].CandidateIDs; !reflect.DeepEqual(got, []string{"c", "d"}) {
		t.Fatalf("priority did not beat cheaper group: %+v", plan.QuorumGroups)
	}

	for i := range roster.Candidates {
		roster.Candidates[i].Priority = 0
	}
	plan, err = PlanIssueAudit(AuditIdentity{Model: "unknown"}, AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatal(err)
	}
	if got := plan.QuorumGroups[0].CandidateIDs; !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("cost did not break equal health/priority tie: %+v", plan.QuorumGroups)
	}
}

func TestPlanIssueAuditRejectsDriverTierQuorumAndHealthAmbiguity(t *testing.T) {
	authors, candidates, aliases := auditRouteV1Fixtures()
	t.Run("driver-mismatch", func(t *testing.T) {
		candidate := candidates["gpt-primary"]
		candidate.Identity.Driver = "claude"
		roster := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}, EstimatedInputTokens: 1}
		plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		if err == nil {
			t.Fatal("driver mismatch routed")
		}
		decision, _ := auditRouteV1Considered(plan, candidate.ID)
		if decision.SkipReason != AuditSkipDriverIdentityInvalid {
			t.Fatalf("driver mismatch decision = %+v", decision)
		}
	})
	t.Run("missing-capability-source", func(t *testing.T) {
		candidate := candidates["gpt-primary"]
		candidate.CapabilitySource = ""
		roster := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}}
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidRoster {
			t.Fatalf("missing capability source error = %#v", err)
		}
	})
	t.Run("invalid-quorum", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary")
		roster.UnknownAuthorQuorum = 1
		_, err := PlanIssueAudit(authors["unknown"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidRoster {
			t.Fatalf("invalid quorum error = %#v", err)
		}
	})
	t.Run("ambiguous-health-key", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary")
		health := auditRouteV1Health(roster)
		health.Providers["OpenAI"] = AuditProviderUnhealthy
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidHealth {
			t.Fatalf("ambiguous health error = %#v", err)
		}
	})
	t.Run("local-endpoint-serving-provider", func(t *testing.T) {
		candidate := candidates["local-open"]
		candidate.Identity.Provider = "openai"
		candidate.Identity.Driver = "http"
		roster := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}}
		plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		if err == nil {
			t.Fatal("local endpoint with remote serving provider routed")
		}
		decision, _ := auditRouteV1Considered(plan, candidate.ID)
		if decision.SkipReason != AuditSkipIdentityConflict {
			t.Fatalf("local endpoint provider decision = %+v", decision)
		}
	})
	t.Run("local-provider-hosted-endpoint", func(t *testing.T) {
		candidate := candidates["local-open"]
		candidate.Identity.EndpointClass = "hosted"
		roster := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}}
		plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		if err == nil {
			t.Fatal("local provider with hosted endpoint routed")
		}
		decision, _ := auditRouteV1Considered(plan, candidate.ID)
		if decision.SkipReason != AuditSkipIdentityConflict {
			t.Fatalf("hosted local-provider decision = %+v", decision)
		}
	})
	t.Run("partial-author-locality-contradiction", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary", "claude-primary")
		author := AuditIdentity{Model: "unregistered-local", Provider: "local", EndpointClass: "hosted"}
		_, err := PlanIssueAudit(author, AuditRiskDefault, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteAuthorConflict {
			t.Fatalf("partial locality contradiction error = %#v", err)
		}
	})
	t.Run("candidate-bound-and-priority-domain", func(t *testing.T) {
		base := candidates["gpt-primary"]
		many := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases}
		for i := 0; i < MaxAuditRouteCandidates+1; i++ {
			candidate := base
			candidate.ID = "candidate-" + string(rune('a'+i))
			many.Candidates = append(many.Candidates, candidate)
		}
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, many, AuditRouteHealth{Schema: AuditRouteHealthSchema})
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidRoster {
			t.Fatalf("17-candidate error = %#v", err)
		}
		for _, priority := range []int{-1, MaxAuditRoutePriority + 1} {
			candidate := base
			candidate.Priority = priority
			roster := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}}
			_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
			if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidRoster {
				t.Fatalf("priority %d error = %#v", priority, err)
			}
		}
		decisions := make([]AuditRouteCandidateDecision, MaxAuditRouteCandidates)
		indexes := make([]int, MaxAuditRouteCandidates)
		for i := range decisions {
			decisions[i].Priority = MaxAuditRoutePriority
			indexes[i] = i
		}
		if got, want := auditQuorumGroupPriority(decisions, indexes), int64(MaxAuditRouteCandidates*MaxAuditRoutePriority); got != want {
			t.Fatalf("bounded priority sum = %d, want %d", got, want)
		}
		decisions = []AuditRouteCandidateDecision{{EstimatedCostMicrosUSD: int64(^uint64(0) >> 1)}, {EstimatedCostMicrosUSD: 1}}
		if got := auditQuorumGroupCost(decisions, []int{0, 1}); got != int64(^uint64(0)>>1) {
			t.Fatalf("overflow group cost = %d, want saturation", got)
		}
	})
}

func TestPlanIssueAuditCooldownAndPermutationDeterminism(t *testing.T) {
	authors, _, _ := auditRouteV1Fixtures()
	roster := auditRouteV1Roster("gpt-primary", "local-open")
	health := auditRouteV1Health(roster)
	health.Cooldown["gpt-primary"] = AuditCooldownActive
	plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
	if err != nil || plan.Candidates[0].CandidateID != "local-open" {
		t.Fatalf("cooldown fallback = %v err=%v", auditRouteV1CandidateIDs(plan), err)
	}
	decision, _ := auditRouteV1Considered(plan, "gpt-primary")
	if decision.SkipReason != AuditSkipCooldownActive {
		t.Fatalf("cooldown decision = %+v", decision)
	}

	health = auditRouteV1Health(roster)
	want, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
	if err != nil {
		t.Fatal(err)
	}
	reversed := roster
	reversed.Candidates = []AuditAuditorConfig{roster.Candidates[1], roster.Candidates[0]}
	got, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, reversed, health)
	if err != nil {
		t.Fatal(err)
	}
	wantJSON, _ := json.Marshal(want)
	gotJSON, _ := json.Marshal(got)
	if string(wantJSON) != string(gotJSON) {
		t.Fatalf("roster order changed plan:\nwant %s\ngot  %s", wantJSON, gotJSON)
	}
}

func TestPlanIssueAuditHighRiskProviderDiversityAndRootCauseRefusals(t *testing.T) {
	t.Run("high-risk-same-provider", func(t *testing.T) {
		aliases := []AuditIdentityAlias{
			auditRouteV1Alias("shared-author", "shared-author-canonical", "shared", "claude", "author-w"),
			auditRouteV1Alias("shared-review", "shared-review-canonical", "shared", "gpt", "review-w"),
		}
		author := auditRouteV1Identity("shared-author", "shared", "claude", "author-w", "author-harness", "hosted", "api", "xhigh")
		candidate := AuditAuditorConfig{
			ID: "shared-review", Identity: auditRouteV1Identity("shared-review", "shared", "gpt", "review-w", "http-reviewer", "hosted", "api", "xhigh"),
			Capability: TierT0, CapabilitySource: "benchmark:shared-review",
		}
		roster := AuditRouteRoster{Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}}
		health := AuditRouteHealth{
			Schema:    AuditRouteHealthSchema,
			Providers: map[string]AuditProviderHealthStatus{"shared": AuditProviderHealthy},
			Capacity:  map[string]AuditCapacityStatus{"shared-review": AuditCapacityAvailable},
			Cooldown:  map[string]AuditCooldownStatus{"shared-review": AuditCooldownReady},
		}
		plan, err := PlanIssueAudit(author, AuditRiskHigh, roster, health)
		if err == nil {
			t.Fatal("high-risk same-provider route admitted")
		}
		decision, _ := auditRouteV1Considered(plan, "shared-review")
		if decision.Independence.Verdict != AuditIndependenceRefuse || decision.Independence.Reason != AuditReasonRefuseSameProviderHighRisk {
			t.Fatalf("same-provider decision = %+v", decision)
		}
	})

	authors, candidates, aliases := auditRouteV1Fixtures()
	t.Run("unknown-all-unhealthy", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary", "claude-primary")
		health := auditRouteV1Health(roster)
		health.Providers["openai"] = AuditProviderUnhealthy
		health.Providers["anthropic"] = AuditProviderUnhealthy
		_, err := PlanIssueAudit(authors["unknown"], AuditRiskDefault, roster, health)
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteNoHealthyProvider {
			t.Fatalf("unknown unhealthy root cause = %#v", err)
		}
	})
	t.Run("missing-capacity", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary")
		health := auditRouteV1Health(roster)
		delete(health.Capacity, "gpt-primary")
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteNoCapacity {
			t.Fatalf("missing capacity root cause = %#v", err)
		}
	})
	t.Run("unknown-provider-health", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary")
		health := auditRouteV1Health(roster)
		health.Providers["openai"] = AuditProviderUnknown
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteNoHealthyProvider {
			t.Fatalf("unknown provider root cause = %#v", err)
		}
	})
	t.Run("unknown-cooldown", func(t *testing.T) {
		roster := auditRouteV1Roster("gpt-primary")
		health := auditRouteV1Health(roster)
		delete(health.Cooldown, "gpt-primary")
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, health)
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteCooldown {
			t.Fatalf("unknown cooldown root cause = %#v", err)
		}
	})
	t.Run("cost-overflow", func(t *testing.T) {
		candidate := candidates["gpt-primary"]
		candidate.Price.InputMicrosPerMillionTokens = 1_000_001
		roster := AuditRouteRoster{
			Schema: AuditRouteRosterSchema, Aliases: aliases, Candidates: []AuditAuditorConfig{candidate}, EstimatedInputTokens: int64(^uint64(0) >> 1),
		}
		_, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidRoster {
			t.Fatalf("cost overflow error = %#v", err)
		}
	})
	t.Run("quorum-cost-overflow", func(t *testing.T) {
		gpt := candidates["gpt-primary"]
		claude := candidates["claude-primary"]
		gpt.Price = AuditRoutePrice{InputMicrosPerMillionTokens: 500_000}
		claude.Price = AuditRoutePrice{InputMicrosPerMillionTokens: 500_000}
		roster := AuditRouteRoster{
			Schema:               AuditRouteRosterSchema,
			Aliases:              aliases,
			Candidates:           []AuditAuditorConfig{gpt, claude},
			UnknownAuthorQuorum:  2,
			EstimatedInputTokens: int64(^uint64(0) >> 1),
		}
		plan, err := PlanIssueAudit(authors["unknown"], AuditRiskDefault, roster, auditRouteV1Health(roster))
		var noRoute *AuditNoRouteError
		if !errors.As(err, &noRoute) || noRoute.Reason != AuditNoRouteInvalidRoster {
			t.Fatalf("quorum cost overflow error = %#v plan=%+v", err, plan)
		}
		if noRoute.Detail != "quorum cost overflows int64" {
			t.Fatalf("quorum cost overflow detail = %q", noRoute.Detail)
		}
	})
}

type auditRouteArtifactV1 struct {
	Schema                string                          `json:"schema"`
	Risk                  AuditRisk                       `json:"risk"`
	Author                AuditIdentity                   `json:"author"`
	AuthorUnknown         bool                            `json:"author_unknown"`
	RequiredTier          WorkTier                        `json:"required_tier"`
	RequiredEffort        string                          `json:"required_effort"`
	RequiredQuorum        int                             `json:"required_quorum"`
	PolicyVersion         string                          `json:"policy_version"`
	PolicyDigest          string                          `json:"policy_digest"`
	EstimatedInputTokens  int64                           `json:"estimated_input_tokens"`
	EstimatedOutputTokens int64                           `json:"estimated_output_tokens"`
	QuorumCostMicrosUSD   int64                           `json:"quorum_cost_micros_usd"`
	Candidates            []auditRouteArtifactCandidateV1 `json:"candidates"`
}

type auditRouteArtifactCandidateV1 struct {
	Rank                   int                       `json:"rank"`
	CandidateID            string                    `json:"candidate_id"`
	Identity               AuditIdentity             `json:"identity"`
	Capability             WorkTier                  `json:"capability"`
	CapabilitySource       string                    `json:"capability_source"`
	ActualEffort           string                    `json:"actual_effort"`
	ProviderHealth         AuditProviderHealthStatus `json:"provider_health"`
	Capacity               AuditCapacityStatus       `json:"capacity"`
	Cooldown               AuditCooldownStatus       `json:"cooldown"`
	EstimatedCostMicrosUSD int64                     `json:"estimated_cost_micros_usd"`
	IndependenceVerdict    AuditIndependenceVerdict  `json:"independence_verdict"`
	IndependenceReason     AuditIndependenceReason   `json:"independence_reason"`
	IndependenceDigest     string                    `json:"independence_digest"`
	MissingAxes            []string                  `json:"missing_axes,omitempty"`
}

func auditRouteArtifactFromPlanV1(plan AuditIssuePlan) auditRouteArtifactV1 {
	artifact := auditRouteArtifactV1{
		Schema: plan.Schema, Risk: plan.Risk, Author: plan.Author, AuthorUnknown: plan.AuthorUnknown,
		RequiredTier: plan.RequiredTier, RequiredEffort: plan.RequiredEffort, RequiredQuorum: plan.RequiredQuorum,
		PolicyVersion: plan.PolicyVersion, PolicyDigest: plan.PolicyDigest,
		EstimatedInputTokens: plan.EstimatedInputTokens, EstimatedOutputTokens: plan.EstimatedOutputTokens,
		QuorumCostMicrosUSD: plan.QuorumCostMicrosUSD,
	}
	for _, candidate := range plan.Candidates {
		artifact.Candidates = append(artifact.Candidates, auditRouteArtifactCandidateV1{
			Rank: candidate.Rank, CandidateID: candidate.CandidateID, Identity: candidate.Identity,
			Capability: candidate.Capability, CapabilitySource: candidate.CapabilitySource, ActualEffort: candidate.ActualEffort,
			ProviderHealth: candidate.ProviderHealth, Capacity: candidate.Capacity, Cooldown: candidate.Cooldown,
			EstimatedCostMicrosUSD: candidate.EstimatedCostMicrosUSD,
			IndependenceVerdict:    candidate.Independence.Verdict, IndependenceReason: candidate.Independence.Reason,
			IndependenceDigest: candidate.Independence.PolicyDigest, MissingAxes: candidate.Independence.MissingAxes,
		})
	}
	return artifact
}

func TestPlanIssueAuditCredentialFreeArtifact(t *testing.T) {
	authors, _, _ := auditRouteV1Fixtures()
	roster := auditRouteV1Roster("gpt-primary", "local-open")
	plan, err := PlanIssueAudit(authors["claude"], AuditRiskDefault, roster, auditRouteV1Health(roster))
	if err != nil {
		t.Fatal(err)
	}
	b, err := json.MarshalIndent(auditRouteArtifactFromPlanV1(plan), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	lower := strings.ToLower(string(b))
	for _, forbidden := range []string{"credential", "cred_env", "api_key", "base_url", "secret", "sk-"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("credential-free plan contains %q:\n%s", forbidden, b)
		}
	}
	t.Logf("credential-free route output:\n%s", b)
	want, err := os.ReadFile(filepath.Join("testdata", "audit_route_plan_v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(bytesTrimSpaceV1(b), bytesTrimSpaceV1(want)) {
		t.Fatalf("route artifact drifted; got:\n%s", b)
	}
}

func bytesTrimSpaceV1(b []byte) []byte {
	return []byte(strings.TrimSpace(string(b)))
}
