package modelroute

import (
	"strings"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// #615 — content-addressed, dos-verifiable routing decision (the pure half).
// ---------------------------------------------------------------------------

// TestDecisionDigestStableSameSubjectSameManifest proves the witness #615 names:
// the same subject under the same manifest always content-addresses the same way,
// across many runs — a witness, not a promise.
func TestDecisionDigestStableSameSubjectSameManifest(t *testing.T) {
	m := perAspectManifest()
	s := Subject{Aspect: AspectToolCall, Tool: "refund_payment"}
	first := m.Digest(s)
	if first == "" || !strings.HasPrefix(first, "sha256:") {
		t.Fatalf("digest should be a non-empty sha256: address, got %q", first)
	}
	for i := 0; i < 50; i++ {
		if got := m.Digest(s); got != first {
			t.Fatalf("digest not stable at %d: %q != %q", i, got, first)
		}
	}
}

// TestDecisionDigestDistinguishesRoute proves different routes content-address
// differently — a digest that collided two routes would be useless as a witness.
func TestDecisionDigestDistinguishesRoute(t *testing.T) {
	m := perAspectManifest()
	a := m.Digest(Subject{Aspect: AspectToolCall, Tool: "refund_payment"}) // ensemble
	b := m.Digest(Subject{Aspect: AspectToolCall, Tool: "search_kb"})      // small
	d := m.Digest(Subject{Aspect: AspectQuery})                            // default
	if a == b || a == d || b == d {
		t.Fatalf("distinct routes collided: refund=%q search=%q default=%q", a, b, d)
	}
}

// TestDecisionDigestBindsManifestVersion proves the version is part of the
// content address: a schema bump re-addresses the same decision.
func TestDecisionDigestBindsManifestVersion(t *testing.T) {
	d := Decision{Subject: Subject{Aspect: AspectRequest}, Plan: okPlan()}
	if d.Digest("fak-route/v1") == d.Digest("fak-route/v1.2") {
		t.Fatal("version should be part of the content address")
	}
}

// TestAuditRecordRoundTrips proves an audit record round-trips (digest in -> same
// digest out) and is dos-checkable via Verify.
func TestAuditRecordRoundTrips(t *testing.T) {
	m := perAspectManifest()
	rec := m.NewAuditRecord(Subject{Aspect: AspectStep, Complexity: ComplexityHigh})
	if err := rec.Verify(); err != nil {
		t.Fatalf("fresh record should verify: %v", err)
	}
	got, err := ParseAuditRecord(rec.JSON())
	if err != nil {
		t.Fatalf("parse round-trip: %v", err)
	}
	if got.Digest != rec.Digest {
		t.Fatalf("digest changed across round-trip: %q != %q", got.Digest, rec.Digest)
	}
}

// TestAuditRecordTamperFailsVerify proves a tampered record fails the dos check —
// the decision can be REPLAYED and bound to evidence, not merely trusted.
func TestAuditRecordTamperFailsVerify(t *testing.T) {
	m := perAspectManifest()
	rec := m.NewAuditRecord(Subject{Aspect: AspectQuery})
	rec.Decision.RuleName = "forged-rule" // edit the decision after the fact
	if err := rec.Verify(); err == nil {
		t.Fatal("tampered decision should fail digest verify")
	}
	// A flipped digest field also fails at the parse boundary.
	tampered := strings.Replace(string(m.NewAuditRecord(Subject{}).JSON()), "sha256:", "sha256:dead", 1)
	if _, err := ParseAuditRecord([]byte(tampered)); err == nil {
		t.Fatal("flipped digest should fail at parse")
	}
}

// ---------------------------------------------------------------------------
// #616 — logical model-id registry + residency-aware resolution.
// ---------------------------------------------------------------------------

func smallLargeRegistry(t *testing.T) *Registry {
	t.Helper()
	r, err := NewRegistry([]ResolvedModel{
		{Alias: "small", EngineID: "gpt-4o-mini", Provider: "openai", Remote: true},
		{Alias: "large", EngineID: "in-kernel-glm", Remote: false},
		{Alias: "guard-a", EngineID: "in-kernel-guard", Remote: false},
		{Alias: "guard-b", EngineID: "in-kernel-guard2", Remote: false},
		{Alias: "default", EngineID: "in-kernel-default", Remote: false},
	})
	if err != nil {
		t.Fatalf("registry build: %v", err)
	}
	return r
}

// TestRegistryValidatesAgainstKnownEngines proves a manifest of logical names
// validates against a configured registry + the known-engine-id set.
func TestRegistryValidatesAgainstKnownEngines(t *testing.T) {
	r := smallLargeRegistry(t)
	known := []string{"gpt-4o-mini", "in-kernel-glm", "in-kernel-guard", "in-kernel-guard2", "in-kernel-default"}
	if err := r.ValidateManifest(perAspectManifest(), known); err != nil {
		t.Fatalf("valid manifest rejected: %v", err)
	}
}

// TestRegistryUnknownModelFailsLoud proves an unknown model name in the manifest,
// and an unknown engine id in the registry, both fail loud.
func TestRegistryUnknownModelFailsLoud(t *testing.T) {
	// A manifest naming a model the registry does not know.
	r := smallLargeRegistry(t)
	bad := Manifest{Default: Plan{Members: []Member{{Model: "ghost"}}}}
	if err := r.ValidateManifest(bad, nil); err == nil {
		t.Fatal("unknown model name should fail manifest validation")
	}
	// A registry whose engine id is absent from the known-engine set.
	r2, err := NewRegistry([]ResolvedModel{{Alias: "x", EngineID: "no-such-engine", Remote: false}})
	if err != nil {
		t.Fatalf("registry build: %v", err)
	}
	if err := r2.ValidateEngines([]string{"gpt-4o-mini"}); err == nil {
		t.Fatal("unknown engine id should fail the closed-vocab check")
	}
}

// TestRegistryRejectsMalformed proves a duplicate/empty alias and a provider-less
// remote model fail at build time.
func TestRegistryRejectsMalformed(t *testing.T) {
	cases := [][]ResolvedModel{
		{{Alias: "", EngineID: "e"}},                                 // empty alias
		{{Alias: "a", EngineID: ""}},                                 // empty engine id
		{{Alias: "a", EngineID: "e1"}, {Alias: "a", EngineID: "e2"}}, // dup alias
		{{Alias: "a", EngineID: "e", Remote: true}},                  // remote, no provider
	}
	for i, models := range cases {
		if _, err := NewRegistry(models); err == nil {
			t.Errorf("case %d: malformed registry should fail to build", i)
		}
	}
}

// TestResolveExposesEngineAndLocality proves a resolved member exposes its engine
// id + remote/local flag to the dispatcher.
func TestResolveExposesEngineAndLocality(t *testing.T) {
	r := smallLargeRegistry(t)
	rm, err := r.Resolve(Subject{}, "small")
	if err != nil {
		t.Fatalf("resolve small: %v", err)
	}
	if rm.EngineID != "gpt-4o-mini" || !rm.Remote || rm.Provider != "openai" {
		t.Fatalf("small should resolve to a remote openai engine: %+v", rm)
	}
	lrm, _ := r.Resolve(Subject{}, "large")
	if lrm.Remote || lrm.EngineID != "in-kernel-glm" {
		t.Fatalf("large should resolve local: %+v", lrm)
	}
}

// TestSensitiveSubjectCannotResolveRemote proves the load-bearing floor: a
// sensitive subject cannot resolve to a remote member, but still resolves to a
// local one.
func TestSensitiveSubjectCannotResolveRemote(t *testing.T) {
	r := smallLargeRegistry(t)
	sens := Subject{Labels: map[string]string{SensitiveLabel: "true"}}
	_, err := r.Resolve(sens, "small") // small is remote
	if err == nil {
		t.Fatal("sensitive subject must not resolve to a remote model")
	}
	var re *ResolveError
	if !asResolveError(err, &re) || re.Reason != "sensitive_remote" {
		t.Fatalf("want sensitive_remote ResolveError, got %v", err)
	}
	// The same sensitive subject still resolves to a LOCAL member.
	if _, err := r.Resolve(sens, "large"); err != nil {
		t.Fatalf("sensitive subject should still resolve local: %v", err)
	}
}

// TestResolvePlanFailsFastOnIneligibleMember proves a plan with one ineligible
// member never partially resolves.
func TestResolvePlanFailsFastOnIneligibleMember(t *testing.T) {
	r := smallLargeRegistry(t)
	sens := Subject{Labels: map[string]string{SensitiveLabel: "yes"}}
	// large (local, ok) then small (remote, refused for sensitive).
	plan := Plan{Members: []Member{{Model: "large"}, {Model: "small"}}, Reduce: ReduceFirst}
	if _, err := r.ResolvePlan(sens, plan); err == nil {
		t.Fatal("a plan containing a sensitive-remote member should fail to resolve")
	}
}

func asResolveError(err error, target **ResolveError) bool {
	re, ok := err.(*ResolveError)
	if ok {
		*target = re
	}
	return ok
}

// ---------------------------------------------------------------------------
// #603 — routing observability (decision journal + Counts rollup; /metrics emit
// deferred to the peer-broken gateway).
// ---------------------------------------------------------------------------

// TestDecisionJournalCountsRollup proves the journal records per-aspect/-rule/
// -strategy decisions and Counts() folds them into the rollup a gateway exporter
// copies into fak_gateway_*.
func TestDecisionJournalCountsRollup(t *testing.T) {
	m := perAspectManifest()
	var j DecisionJournal
	subs := []Subject{
		{Aspect: AspectToolCall, Tool: "refund_payment"}, // ensemble, rule refund-guard
		{Aspect: AspectToolCall, Tool: "search_kb"},      // single, rule search-small
		{Aspect: AspectStep, Complexity: ComplexityHigh}, // single, rule hard-step
		{Aspect: AspectQuery},                            // default
	}
	for _, s := range subs {
		j.Record(m.Version, m.Route(s), 1*time.Microsecond)
	}
	if j.Len() != 4 {
		t.Fatalf("journal len = %d, want 4", j.Len())
	}
	c := j.Counts()
	if c.Total != 4 {
		t.Fatalf("total = %d, want 4", c.Total)
	}
	if c.EnsembleHits != 1 {
		t.Fatalf("ensemble hits = %d, want 1", c.EnsembleHits)
	}
	if c.ByStrategy[string(StrategyDefault)] != 1 || c.ByStrategy[string(StrategySingle)] != 2 || c.ByStrategy[string(StrategyEnsemble)] != 1 {
		t.Fatalf("strategy rollup wrong: %+v", c.ByStrategy)
	}
	if c.ByRule["refund-guard"] != 1 || c.ByRule["(default)"] != 1 {
		t.Fatalf("rule rollup wrong: %+v", c.ByRule)
	}
	if c.ByAspect["tool_call"] != 2 || c.ByAspect["step"] != 1 {
		t.Fatalf("aspect rollup wrong: %+v", c.ByAspect)
	}
	if c.TotalOverhead != 4*time.Microsecond {
		t.Fatalf("total overhead = %v, want 4us", c.TotalOverhead)
	}
}

// TestDecisionRecordCarriesScoutAndDigest proves a record carries the scout-call
// count and the #615 digest a /metrics exporter reads.
func TestDecisionRecordCarriesScoutAndDigest(t *testing.T) {
	m := Manifest{Default: Plan{Members: []Member{{Model: "default"}}}, Rules: []Rule{
		{Name: "scouted", Match: Match{Aspect: AspectRequest},
			Plan: Plan{Members: []Member{{Model: "big"}}, Scout: "cheap-classifier"}},
	}}
	rec := RecordDecision(m.Version, m.Route(Subject{Aspect: AspectRequest}), 0)
	if rec.ScoutCalls != 1 {
		t.Fatalf("scout calls = %d, want 1", rec.ScoutCalls)
	}
	if !strings.HasPrefix(rec.Digest, "sha256:") {
		t.Fatalf("record should carry the #615 digest, got %q", rec.Digest)
	}
	if rec.Strategy != StrategySingle || rec.RuleName != "scouted" {
		t.Fatalf("record strategy/rule wrong: %+v", rec)
	}
}

// TestCountsSortedRulesDeterministic proves the exporter's rule order is
// deterministic (descending count, ties by name).
func TestCountsSortedRulesDeterministic(t *testing.T) {
	c := Counts{ByRule: map[string]int{"a": 1, "b": 3, "c": 3}}
	got := c.SortedRules()
	want := []string{"b", "c", "a"} // 3,3 (b<c) then 1
	if len(got) != 3 || got[0] != want[0] || got[1] != want[1] || got[2] != want[2] {
		t.Fatalf("sorted rules = %v, want %v", got, want)
	}
}

// ---------------------------------------------------------------------------
// #604 — bridge ensemble roles to spec decode (drafter/verifier -> polymodel,
// via a closure; the live verify pass is deferred engine wiring).
// ---------------------------------------------------------------------------

// TestSpecRolesExtractsDrafterVerifier proves a roles-tagged Plan maps onto the
// drafter/verifier pair.
func TestSpecRolesExtractsDrafterVerifier(t *testing.T) {
	p := Plan{Members: []Member{
		{Model: "draft-7b", Role: "drafter"},
		{Model: "verify-70b", Role: "Verifier"}, // case-insensitive
	}, Reduce: ReduceFirst}
	if !p.IsSpecEnsemble() {
		t.Fatal("a 1-drafter/1-verifier plan should be a spec ensemble")
	}
	roles, err := p.SpecRoles()
	if err != nil {
		t.Fatalf("spec roles: %v", err)
	}
	if roles.Drafter != "draft-7b" || roles.Verifier != "verify-70b" {
		t.Fatalf("roles = %+v", roles)
	}
}

// TestSpecRolesRejectsNonSpecEnsemble proves a vote/best-of ensemble is NOT
// silently treated as a draft/verify pair.
func TestSpecRolesRejectsNonSpecEnsemble(t *testing.T) {
	vote := Plan{Members: []Member{{Model: "a"}, {Model: "b"}}, Reduce: ReduceVote}
	if vote.IsSpecEnsemble() {
		t.Fatal("a vote ensemble must not be a spec ensemble")
	}
	if _, err := vote.SpecRoles(); err == nil {
		t.Fatal("a role-less ensemble should fail SpecRoles")
	}
	// Two drafters is also not a spec pair.
	twoDraft := Plan{Members: []Member{{Model: "a", Role: "drafter"}, {Model: "b", Role: "drafter"}}}
	if _, err := twoDraft.SpecRoles(); err == nil {
		t.Fatal("two drafters should fail SpecRoles")
	}
}

// TestSpecAcceptPassesThroughBoundAcceptFunc proves the pass-through to a bound
// accept call (the caller binds polymodel.AcceptGreedy) drives the accept
// accounting WITHOUT modelroute importing polymodel. The fake accept here mimics
// polymodel.AcceptGreedy's contract: longest matching prefix + 1 correction.
func TestSpecAcceptPassesThroughBoundAcceptFunc(t *testing.T) {
	p := Plan{Members: []Member{
		{Model: "draft", Role: "drafter"},
		{Model: "verify", Role: "verifier"},
	}}
	// Stand-in for polymodel.AcceptGreedy(draft, targetArgmax) adapted to SpecAccept.
	fakeAcceptGreedy := func(draft, target []int) SpecAccept {
		k := len(draft)
		limit := k
		if len(target) < limit {
			limit = len(target)
		}
		acc := 0
		for acc < limit && draft[acc] == target[acc] {
			acc++
		}
		adv := acc
		if acc < len(target) {
			adv = acc + 1
		}
		return SpecAccept{Accepted: acc, Advance: adv, KeepKV: acc, EvictKV: k - acc}
	}
	draft := []int{10, 11, 12, 99}
	target := []int{10, 11, 12, 42, 7} // 3 match, divergence at 3, bonus at 4
	roles, res, err := p.SpecAccept(draft, target, fakeAcceptGreedy)
	if err != nil {
		t.Fatalf("spec accept: %v", err)
	}
	if roles.Drafter != "draft" || roles.Verifier != "verify" {
		t.Fatalf("roles = %+v", roles)
	}
	if res.Accepted != 3 || res.Advance != 4 || res.KeepKV != 3 || res.EvictKV != 1 {
		t.Fatalf("accept accounting = %+v", res)
	}
}

// TestSpecAcceptRefusesNonSpecAndNilFunc proves the bridge refuses to run a
// non-spec ensemble as spec-decode, and refuses a nil accept func.
func TestSpecAcceptRefusesNonSpecAndNilFunc(t *testing.T) {
	vote := Plan{Members: []Member{{Model: "a"}, {Model: "b"}}, Reduce: ReduceVote}
	if _, _, err := vote.SpecAccept(nil, nil, func(_, _ []int) SpecAccept { return SpecAccept{} }); err == nil {
		t.Fatal("a non-spec ensemble must not run as spec-decode")
	}
	spec := Plan{Members: []Member{{Model: "d", Role: "drafter"}, {Model: "v", Role: "verifier"}}}
	if _, _, err := spec.SpecAccept(nil, nil, nil); err == nil {
		t.Fatal("a nil accept func should be refused")
	}
}
