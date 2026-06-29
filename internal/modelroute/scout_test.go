package modelroute

import (
	"context"
	"errors"
	"testing"
	"time"
)

// scoutManifest is a routing policy whose ONLY rule fires on a high-complexity
// request: with the static signals alone a bare {Aspect: request} subject does NOT
// match it (complexity is unset, ComplexityAny) and falls to the fail-closed
// default; only AFTER a scout labels it ComplexityHigh does the rule fire. This is
// what lets the test prove the scout's label changed the matched rule.
func scoutManifest() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{Members: []Member{{Model: "small-default"}}, Reason: "no rule matched"},
		Rules: []Rule{
			{
				Name:  "hard-to-large",
				Match: Match{Aspect: AspectRequest, MinComplexity: ComplexityHigh},
				Plan:  Plan{Members: []Member{{Model: "large"}}, Reason: "scout-found hard work -> large model"},
			},
			{
				Name:  "billing-domain",
				Match: Match{Aspect: AspectRequest, Labels: map[string]string{"domain": "billing"}},
				Plan:  Plan{Members: []Member{{Model: "billing-specialist"}}, Reason: "billing domain -> specialist"},
			},
		},
	}
}

// fakeClassifier is the INJECTED stand-in for the cheap scout model: it records
// every call and returns a fixed label, so the test drives the live scout step
// without an engine. It is the test's witness that the scout seam is an interface,
// not a hardcoded model.
type fakeClassifier struct {
	label   ScoutLabel
	err     error
	calls   int
	gotSubj Subject
}

func (f *fakeClassifier) Classify(_ context.Context, s Subject) (ScoutLabel, error) {
	f.calls++
	f.gotSubj = s
	if f.err != nil {
		return ScoutLabel{}, f.err
	}
	return f.label, nil
}

// TestScoutRouteFillsComplexityAndChangesRoute is the core acceptance test: a bare
// subject the STATIC signals route to the default gets its Complexity filled from
// the injected classifier's label, after which it routes to a DIFFERENT rule — and
// the scout decision is observable (the outcome records both routes and the call
// count). This is the issue's acceptance: "a scout-driven manifest routes a subject
// the static signals alone would not (the scout's label changes the matched rule);
// the scout call count is observable."
func TestScoutRouteFillsComplexityAndChangesRoute(t *testing.T) {
	m := scoutManifest()
	subj := Subject{Aspect: AspectRequest} // no complexity, no labels

	// Witness the STATIC (no-scout) route first: it must fall to the default.
	staticDecision := m.Route(subj)
	if staticDecision.Matched {
		t.Fatalf("precondition: a bare subject must NOT match any rule, got rule %q", staticDecision.RuleName)
	}
	if staticDecision.Plan.Primary() != "small-default" {
		t.Fatalf("precondition: static route should be the default, got %q", staticDecision.Plan.Primary())
	}

	fake := &fakeClassifier{label: ScoutLabel{Complexity: ComplexityHigh}}
	out, err := m.ScoutRoute(context.Background(), fake, subj)
	if err != nil {
		t.Fatalf("ScoutRoute: %v", err)
	}

	// The scout was actually called exactly once (the priced extra model call).
	if fake.calls != 1 {
		t.Fatalf("classifier calls = %d, want exactly 1", fake.calls)
	}
	if out.ScoutCalls != 1 {
		t.Fatalf("outcome ScoutCalls = %d, want 1 (the scout overhead must be observable)", out.ScoutCalls)
	}

	// The scout's label populated the subject's Complexity (it was unset before).
	if out.Enriched.Complexity != ComplexityHigh {
		t.Fatalf("enriched complexity = %q, want high (scout label must fill the subject)", out.Enriched.Complexity)
	}

	// Before == the static route (the no-scout path is unchanged); After == the
	// scouted route, which now matches the high-complexity rule.
	if out.Before.RuleName != "" || out.Before.Matched {
		t.Fatalf("Before must be the static default route, got matched=%v rule=%q", out.Before.Matched, out.Before.RuleName)
	}
	if !out.After.Matched || out.After.RuleName != "hard-to-large" {
		t.Fatalf("After should match the hard-to-large rule, got matched=%v rule=%q", out.After.Matched, out.After.RuleName)
	}
	if out.After.Plan.Primary() != "large" {
		t.Fatalf("After plan should route to large, got %q", out.After.Plan.Primary())
	}

	// The scout DEMONSTRABLY changed the route — the witness the extra call earned
	// its cost.
	if !out.Changed() {
		t.Fatal("scout should have changed the matched rule (default -> hard-to-large)")
	}

	// The original caller subject is untouched (ApplyScoutLabel is additive/pure).
	if subj.Complexity != ComplexityAny {
		t.Fatalf("the caller's original subject must not be mutated, got complexity %q", subj.Complexity)
	}
}

// TestScoutRouteFillsLabelsAndRoutesByDomain proves the scout also enriches the
// OPEN Labels signal (not just Complexity): a scout that tags domain=billing routes
// a bare subject to the billing specialist the static signals would have missed.
func TestScoutRouteFillsLabelsAndRoutesByDomain(t *testing.T) {
	m := scoutManifest()
	subj := Subject{Aspect: AspectRequest}

	fake := &fakeClassifier{label: ScoutLabel{Labels: map[string]string{"domain": "billing"}}}
	out, err := m.ScoutRoute(context.Background(), fake, subj)
	if err != nil {
		t.Fatalf("ScoutRoute: %v", err)
	}
	if out.Enriched.Labels["domain"] != "billing" {
		t.Fatalf("scout label should populate domain, got %+v", out.Enriched.Labels)
	}
	if !out.After.Matched || out.After.RuleName != "billing-domain" {
		t.Fatalf("After should match billing-domain, got matched=%v rule=%q", out.After.Matched, out.After.RuleName)
	}
	if !out.Changed() {
		t.Fatal("a domain-tagging scout should change the route off the default")
	}
}

// TestScoutRouteNoOpWhenLabelEmpty proves the no-scout-effect path is honest: a
// classifier that returns an EMPTY label (it could not classify) leaves the subject
// unchanged and routes EXACTLY as the static route would — and Changed() is false,
// so the wasted extra call is visible, never hidden.
func TestScoutRouteNoOpWhenLabelEmpty(t *testing.T) {
	m := scoutManifest()
	subj := Subject{Aspect: AspectRequest}

	fake := &fakeClassifier{label: ScoutLabel{}} // empty: scout learned nothing
	out, err := m.ScoutRoute(context.Background(), fake, subj)
	if err != nil {
		t.Fatalf("ScoutRoute: %v", err)
	}
	if out.Changed() {
		t.Fatalf("an empty scout label must NOT change the route, but Before=%q After=%q", out.Before.RuleName, out.After.RuleName)
	}
	if out.After.Matched {
		t.Fatalf("with no label the scouted route must still fall to default, got rule %q", out.After.RuleName)
	}
	// The cost is still charged and observable even though the route did not move —
	// the honesty invariant: a scout that earns nothing is still priced.
	if out.ScoutCalls != 1 {
		t.Fatalf("the wasted scout call must still be counted, got ScoutCalls=%d", out.ScoutCalls)
	}
	// Identical digest to the static route proves the subject was truly unchanged.
	if got, want := out.After.Digest(m.Version), m.Digest(subj); got != want {
		t.Fatalf("no-op scout must route identically to the static route: %s vs %s", got, want)
	}
}

// TestApplyScoutLabelNeverDowngrades proves ApplyScoutLabel is additive: a subject
// that already carries a complexity / a label keeps its own value — a scout cannot
// clobber a signal the caller set deliberately.
func TestApplyScoutLabelNeverDowngrades(t *testing.T) {
	subj := Subject{
		Aspect:     AspectRequest,
		Complexity: ComplexityHigh,
		Labels:     map[string]string{"tenant": "acme"},
	}
	// Scout claims low complexity and a conflicting tenant; both must be ignored.
	got := ApplyScoutLabel(subj, ScoutLabel{
		Complexity: ComplexityLow,
		Labels:     map[string]string{"tenant": "intruder", "domain": "billing"},
	})
	if got.Complexity != ComplexityHigh {
		t.Fatalf("existing complexity must win over the scout, got %q", got.Complexity)
	}
	if got.Labels["tenant"] != "acme" {
		t.Fatalf("existing label must win over the scout, got tenant=%q", got.Labels["tenant"])
	}
	// But a NEW label the subject did not carry IS added.
	if got.Labels["domain"] != "billing" {
		t.Fatalf("scout should add a new label, got %+v", got.Labels)
	}
	// The input must be unmodified (pure).
	if _, ok := subj.Labels["domain"]; ok {
		t.Fatal("ApplyScoutLabel mutated the caller's input map")
	}
}

// TestScoutRouteRejectsNilClassifier proves the seam is required — there is no
// silent no-scout fall-through that would hide a missing injected model.
func TestScoutRouteRejectsNilClassifier(t *testing.T) {
	m := scoutManifest()
	if _, err := m.ScoutRoute(context.Background(), nil, Subject{Aspect: AspectRequest}); err == nil {
		t.Fatal("a nil classifier must be refused, not silently skipped")
	}
}

// TestScoutRouteFailsLoudOnClassifierError proves a scout error is surfaced, never
// swallowed into a quiet static route.
func TestScoutRouteFailsLoudOnClassifierError(t *testing.T) {
	m := scoutManifest()
	fake := &fakeClassifier{err: errors.New("scout model timed out")}
	if _, err := m.ScoutRoute(context.Background(), fake, Subject{Aspect: AspectRequest}); err == nil {
		t.Fatal("a classifier error must fail loud")
	}
}

// TestScoutRouteRejectsOutOfVocabComplexity proves the fail-closed discipline: a
// scout that invents an out-of-vocabulary complexity is refused, so it can never
// inject a value that silently never matches a MinComplexity floor.
func TestScoutRouteRejectsOutOfVocabComplexity(t *testing.T) {
	m := scoutManifest()
	fake := &fakeClassifier{label: ScoutLabel{Complexity: Complexity("extreme")}}
	if _, err := m.ScoutRoute(context.Background(), fake, Subject{Aspect: AspectRequest}); err == nil {
		t.Fatal("an out-of-vocabulary scout complexity must be refused")
	}
}

// TestScoutOutcomeRecordCountsTheScoutCall proves the scouted decision is OBSERVABLE
// through the existing decision-journal surface: the record carries ScoutCalls==1
// even when the After plan itself names no Scout, so a /metrics exporter prices the
// live scout overhead.
func TestScoutOutcomeRecordCountsTheScoutCall(t *testing.T) {
	m := scoutManifest()
	fake := &fakeClassifier{label: ScoutLabel{Complexity: ComplexityHigh}}
	out, err := m.ScoutRoute(context.Background(), fake, Subject{Aspect: AspectRequest})
	if err != nil {
		t.Fatalf("ScoutRoute: %v", err)
	}
	// The After plan (hard-to-large) names no Scout, yet a live scout call WAS made.
	if out.After.Plan.Scout != "" {
		t.Fatalf("precondition: After plan should name no Scout, got %q", out.After.Plan.Scout)
	}
	rec := out.Record(m.Version, 7*time.Microsecond)
	if rec.ScoutCalls != 1 {
		t.Fatalf("recorded ScoutCalls = %d, want 1 (the live scout call must be priced)", rec.ScoutCalls)
	}
	if rec.RuleName != "hard-to-large" {
		t.Fatalf("recorded rule = %q, want hard-to-large", rec.RuleName)
	}
	if rec.Overhead != 7*time.Microsecond {
		t.Fatalf("recorded overhead = %v, want 7us", rec.Overhead)
	}

	// And it folds into the journal rollup the gateway exporter reads.
	var j DecisionJournal
	j.Append(rec)
	if c := j.Counts(); c.ScoutCalls != 1 {
		t.Fatalf("journal scout-call rollup = %d, want 1", c.ScoutCalls)
	}
}
