package modelroute

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Route — per-aspect selection, first-match, fail-closed default, determinism.
// ---------------------------------------------------------------------------

// perAspectManifest routes three DIFFERENT aspects of one request to three
// different models — the generalization a request-only router cannot express.
func perAspectManifest() Manifest {
	return Manifest{
		Version: Version,
		Default: Plan{Members: []Member{{Model: "default", Role: "primary"}}},
		Rules: []Rule{
			{Name: "refund-guard", Match: Match{Aspect: AspectToolCall, Tool: "refund_payment"},
				Plan: Plan{Members: []Member{{Model: "guard-a"}, {Model: "guard-b"}}, Reduce: ReduceVote}},
			{Name: "search-small", Match: Match{Aspect: AspectToolCall, Tool: "search_*"},
				Plan: Plan{Members: []Member{{Model: "small"}}}},
			{Name: "hard-step", Match: Match{Aspect: AspectStep, MinComplexity: ComplexityHigh},
				Plan: Plan{Members: []Member{{Model: "large"}}}},
		},
	}
}

func TestRoutePerAspectWithinOneRequest(t *testing.T) {
	m := perAspectManifest()
	if err := m.Validate(); err != nil {
		t.Fatalf("manifest invalid: %v", err)
	}

	// A tool_call to refund_payment -> the two-model guard ensemble.
	d := m.Route(Subject{Aspect: AspectToolCall, Tool: "refund_payment"})
	if !d.Matched || d.RuleName != "refund-guard" {
		t.Fatalf("refund: matched=%v rule=%q", d.Matched, d.RuleName)
	}
	if !d.Plan.IsEnsemble() || d.Plan.Reduce != ReduceVote {
		t.Fatalf("refund: want vote ensemble, got %+v", d.Plan)
	}

	// A tool_call to search_kb -> the small model (prefix wildcard).
	d = m.Route(Subject{Aspect: AspectToolCall, Tool: "search_kb"})
	if d.RuleName != "search-small" || d.Plan.Primary() != "small" {
		t.Fatalf("search: rule=%q primary=%q", d.RuleName, d.Plan.Primary())
	}

	// A high-complexity reasoning step -> the large model.
	d = m.Route(Subject{Aspect: AspectStep, Complexity: ComplexityHigh})
	if d.RuleName != "hard-step" || d.Plan.Primary() != "large" {
		t.Fatalf("step: rule=%q primary=%q", d.RuleName, d.Plan.Primary())
	}

	// An unmatched aspect -> the fail-closed default.
	d = m.Route(Subject{Aspect: AspectQuery})
	if d.Matched || d.Plan.Primary() != "default" {
		t.Fatalf("default: matched=%v primary=%q", d.Matched, d.Plan.Primary())
	}
}

func TestRouteFirstMatchWins(t *testing.T) {
	m := Manifest{
		Default: Plan{Members: []Member{{Model: "default"}}},
		Rules: []Rule{
			{Name: "first", Match: Match{Aspect: AspectToolCall}, Plan: Plan{Members: []Member{{Model: "a"}}}},
			{Name: "second", Match: Match{Aspect: AspectToolCall, Tool: "x"}, Plan: Plan{Members: []Member{{Model: "b"}}}},
		},
	}
	d := m.Route(Subject{Aspect: AspectToolCall, Tool: "x"})
	if d.RuleName != "first" { // top-to-bottom: the broad rule listed first wins
		t.Fatalf("first-match: rule=%q", d.RuleName)
	}
}

func TestRouteDeterministic(t *testing.T) {
	m := perAspectManifest()
	s := Subject{Aspect: AspectToolCall, Tool: "refund_payment"}
	first := m.Route(s)
	for i := 0; i < 50; i++ {
		if got := m.Route(s); !reflect.DeepEqual(got, first) {
			t.Fatalf("route not deterministic at %d", i)
		}
	}
}

// ---------------------------------------------------------------------------
// Match — bounds, wildcards, complexity floor, labels.
// ---------------------------------------------------------------------------

func TestMatchToolPrefixWildcard(t *testing.T) {
	mt := Match{Tool: "git_*"}
	if !mt.Matches(Subject{Tool: "git_push"}) || mt.Matches(Subject{Tool: "search_kb"}) {
		t.Fatal("prefix wildcard mismatch")
	}
	if !(Match{Tool: "*"}).Matches(Subject{Tool: "anything"}) {
		t.Fatal("star should match any tool")
	}
	if !(Match{Tool: "exact"}).Matches(Subject{Tool: "exact"}) || (Match{Tool: "exact"}).Matches(Subject{Tool: "exactish"}) {
		t.Fatal("exact tool mismatch")
	}
}

func TestMatchTokenBounds(t *testing.T) {
	mt := Match{MinPromptTokens: 100, MaxPromptTokens: 1000}
	if mt.Matches(Subject{PromptTokens: 99}) || mt.Matches(Subject{PromptTokens: 1001}) {
		t.Fatal("out-of-band tokens matched")
	}
	if !mt.Matches(Subject{PromptTokens: 500}) {
		t.Fatal("in-band tokens did not match")
	}
	// MaxPromptTokens == 0 means unbounded.
	if !(Match{MinPromptTokens: 100}).Matches(Subject{PromptTokens: 1 << 20}) {
		t.Fatal("zero max should be unbounded")
	}
}

func TestMatchComplexityFloor(t *testing.T) {
	mt := Match{MinComplexity: ComplexityMedium}
	if mt.Matches(Subject{Complexity: ComplexityLow}) || mt.Matches(Subject{Complexity: ComplexityAny}) {
		t.Fatal("below-floor complexity matched")
	}
	if !mt.Matches(Subject{Complexity: ComplexityMedium}) || !mt.Matches(Subject{Complexity: ComplexityHigh}) {
		t.Fatal("at/above-floor complexity did not match")
	}
}

func TestMatchLabels(t *testing.T) {
	mt := Match{Labels: map[string]string{"domain": "legal", "lang": "go"}}
	if !mt.Matches(Subject{Labels: map[string]string{"domain": "legal", "lang": "go", "extra": "ok"}}) {
		t.Fatal("superset labels should match")
	}
	if mt.Matches(Subject{Labels: map[string]string{"domain": "legal"}}) {
		t.Fatal("missing label should not match")
	}
}

// ---------------------------------------------------------------------------
// Combine — the ensemble reduce half.
// ---------------------------------------------------------------------------

func votes(pairs ...[2]string) []Vote {
	out := make([]Vote, 0, len(pairs))
	for _, p := range pairs {
		out = append(out, Vote{Member: Member{Model: p[0]}, Output: p[1]})
	}
	return out
}

func TestCombineFirst(t *testing.T) {
	r, err := Combine(ReduceFirst, votes([2]string{"a", "x"}, [2]string{"b", "y"}))
	if err != nil || r.Output != "x" || r.Winner != "a" {
		t.Fatalf("first: %+v err=%v", r, err)
	}
	// "" defaults to first.
	r2, _ := Combine("", votes([2]string{"a", "x"}))
	if r2.Reduce != ReduceFirst || r2.Output != "x" {
		t.Fatalf("empty-reduce default: %+v", r2)
	}
}

func TestCombineConcat(t *testing.T) {
	r, err := Combine(ReduceConcat, votes([2]string{"a", "one"}, [2]string{"b", "two"}))
	if err != nil || r.Output != "one\ntwo" {
		t.Fatalf("concat: %+v err=%v", r, err)
	}
}

func TestCombineVoteMajority(t *testing.T) {
	r, err := Combine(ReduceVote, votes(
		[2]string{"a", "yes"}, [2]string{"b", "no"}, [2]string{"c", "yes"}))
	if err != nil || r.Output != "yes" {
		t.Fatalf("vote: %+v err=%v", r, err)
	}
	if r.Tally["yes"] != 2 || r.Tally["no"] != 1 {
		t.Fatalf("vote tally: %+v", r.Tally)
	}
	if r.Winner != "a" { // lexicographically-smallest model that voted "yes"
		t.Fatalf("vote winner: %q", r.Winner)
	}
}

func TestCombineVoteWeighted(t *testing.T) {
	// One heavy vote outweighs two light ones — and the tie-break stays deterministic.
	vs := []Vote{
		{Member: Member{Model: "big", Weight: 5}, Output: "A"},
		{Member: Member{Model: "s1", Weight: 1}, Output: "B"},
		{Member: Member{Model: "s2", Weight: 1}, Output: "B"},
	}
	r, err := Combine(ReduceVote, vs)
	if err != nil || r.Output != "A" {
		t.Fatalf("weighted vote: %+v err=%v", r, err)
	}
}

func TestCombineVoteTieBreakDeterministic(t *testing.T) {
	// Equal weight on "alpha" and "beta" -> the lexicographically-smaller output wins,
	// every time, regardless of vote order.
	for i := 0; i < 20; i++ {
		r, _ := Combine(ReduceVote, votes([2]string{"z", "beta"}, [2]string{"a", "alpha"}))
		if r.Output != "alpha" {
			t.Fatalf("tie-break not deterministic: %q", r.Output)
		}
	}
}

func TestCombineBestOf(t *testing.T) {
	vs := []Vote{
		{Member: Member{Model: "a"}, Output: "lo", Score: 0.2},
		{Member: Member{Model: "b"}, Output: "hi", Score: 0.9},
		{Member: Member{Model: "c"}, Output: "mid", Score: 0.5},
	}
	r, err := Combine(ReduceBestOf, vs)
	if err != nil || r.Output != "hi" || r.Winner != "b" {
		t.Fatalf("best_of: %+v err=%v", r, err)
	}
}

func TestCombineAllReduceMean(t *testing.T) {
	vs := []Vote{
		{Member: Member{Model: "a", Weight: 1}, Output: "2"},
		{Member: Member{Model: "b", Weight: 3}, Output: "6"},
	}
	r, err := Combine(ReduceAllReduce, vs) // (2*1 + 6*3)/(1+3) = 20/4 = 5
	if err != nil || r.Output != "5" {
		t.Fatalf("all_reduce: %+v err=%v", r, err)
	}
}

func TestCombineErrors(t *testing.T) {
	if _, err := Combine(ReduceFirst, nil); err == nil {
		t.Fatal("empty votes should error")
	}
	if _, err := Combine(ReduceAllReduce, votes([2]string{"a", "not-a-number"})); err == nil {
		t.Fatal("non-numeric all_reduce should error")
	}
	if _, err := Combine(Reduction("bogus"), votes([2]string{"a", "x"})); err == nil {
		t.Fatal("unknown reduction should error")
	}
}

// ---------------------------------------------------------------------------
// Validate — fail-loud on every malformed manifest.
// ---------------------------------------------------------------------------

func TestValidateRejectsBadManifests(t *testing.T) {
	cases := []struct {
		name string
		m    Manifest
	}{
		{"no default members", Manifest{Default: Plan{}}},
		{"empty model", Manifest{Default: Plan{Members: []Member{{Model: ""}}}}},
		{"negative weight", Manifest{Default: Plan{Members: []Member{{Model: "a", Weight: -1}}}}},
		{"ensemble without reduce", Manifest{Default: Plan{Members: []Member{{Model: "a"}, {Model: "b"}}}}},
		{"ensemble unknown reduce", Manifest{Default: Plan{Members: []Member{{Model: "a"}, {Model: "b"}}, Reduce: "bogus"}}},
		{"single unknown reduce", Manifest{Default: Plan{Members: []Member{{Model: "a"}}, Reduce: "bogus"}}},
		{"bad version", Manifest{Version: "fak-route/v2", Default: okPlan()}},
		{"dup rule name", Manifest{Default: okPlan(), Rules: []Rule{
			{Name: "r", Plan: okPlan()}, {Name: "r", Plan: okPlan()}}}},
		{"empty rule name", Manifest{Default: okPlan(), Rules: []Rule{{Name: "", Plan: okPlan()}}}},
		{"min>max", Manifest{Default: okPlan(), Rules: []Rule{
			{Name: "r", Match: Match{MinPromptTokens: 100, MaxPromptTokens: 10}, Plan: okPlan()}}}},
		{"unknown latency", Manifest{Default: okPlan(), Rules: []Rule{
			{Name: "r", Match: Match{Latency: "fast"}, Plan: okPlan()}}}},
		{"unknown complexity", Manifest{Default: okPlan(), Rules: []Rule{
			{Name: "r", Match: Match{MinComplexity: "extreme"}, Plan: okPlan()}}}},
	}
	for _, c := range cases {
		if err := c.m.Validate(); err == nil {
			t.Errorf("%s: expected validation error, got nil", c.name)
		}
	}
}

func okPlan() Plan { return Plan{Members: []Member{{Model: "a"}}} }

func TestValidateAcceptsOpenAspect(t *testing.T) {
	// Aspect is an OPEN set: a deployment may route its own named stage.
	m := Manifest{Default: okPlan(), Rules: []Rule{
		{Name: "custom", Match: Match{Aspect: Aspect("my_custom_stage")}, Plan: okPlan()}}}
	if err := m.Validate(); err != nil {
		t.Fatalf("open aspect rejected: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Manifest round-trip + unknown-field rejection (mirrors fak policy --dump|--check).
// ---------------------------------------------------------------------------

func TestDefaultManifestValidAndRoundTrips(t *testing.T) {
	m := DefaultManifest()
	if err := m.Validate(); err != nil {
		t.Fatalf("default manifest invalid: %v", err)
	}
	got, err := ParseManifest(m.JSON())
	if err != nil {
		t.Fatalf("re-parse: %v", err)
	}
	if !reflect.DeepEqual(got, m) {
		t.Fatalf("round-trip mismatch:\n want %+v\n got  %+v", m, got)
	}
}

func TestParseRejectsUnknownField(t *testing.T) {
	bad := []byte(`{"version":"fak-route/v1","default":{"members":[{"model":"a"}]},"rulez":[]}`)
	if _, err := ParseManifest(bad); err == nil || !strings.Contains(err.Error(), "parse manifest") {
		t.Fatalf("unknown field should fail loudly, got %v", err)
	}
}

func TestParseRejectsInvalidPlan(t *testing.T) {
	bad := []byte(`{"default":{"members":[]}}`)
	if _, err := ParseManifest(bad); err == nil {
		t.Fatal("empty default members should fail at parse")
	}
}

// ---------------------------------------------------------------------------
// Plan helpers.
// ---------------------------------------------------------------------------

func TestPlanHelpers(t *testing.T) {
	p := Plan{Members: []Member{{Model: "a"}, {Model: "primary-model", Role: "primary"}, {Model: "c"}}}
	if !p.IsEnsemble() {
		t.Fatal("3 members should be an ensemble")
	}
	if p.Primary() != "primary-model" {
		t.Fatalf("primary role should win: %q", p.Primary())
	}
	if got := p.Models(); !reflect.DeepEqual(got, []string{"a", "primary-model", "c"}) {
		t.Fatalf("models: %v", got)
	}
	// No explicit primary -> first member.
	if (Plan{Members: []Member{{Model: "x"}}}).Primary() != "x" {
		t.Fatal("first member should be primary fallback")
	}
	if (Plan{}).Primary() != "" {
		t.Fatal("empty plan primary should be empty")
	}
}

// Sanity: the on-disk JSON shape uses the documented snake_case keys.
func TestManifestJSONKeys(t *testing.T) {
	b := DefaultManifest().JSON()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatalf("json: %v", err)
	}
	for _, k := range []string{"version", "default", "rules"} {
		if _, ok := raw[k]; !ok {
			t.Errorf("missing top-level key %q", k)
		}
	}
}
