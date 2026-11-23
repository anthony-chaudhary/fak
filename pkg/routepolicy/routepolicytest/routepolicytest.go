// Package routepolicytest is the conformance kit for routepolicy.Policy. An
// alternative implementation proves it honours the routepolicy contract by
// running the same manifest + subject -> decision table the reference Manifest
// passes:
//
//	func TestConformance(t *testing.T) {
//		routepolicytest.Run(t, func(m routepolicy.Manifest) routepolicy.Policy { return myPolicyFrom(m) })
//	}
//
// build receives a validated Manifest and returns the Policy under test; the kit
// never inspects the Policy beyond Route.
package routepolicytest

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
	"github.com/anthony-chaudhary/fak/pkg/routepolicy/inputtrigger"
)

// Case is one conformance row: route Subject under Manifest and expect the
// Decision named by WantRule (empty means the fail-closed default) and WantPlan.
type Case struct {
	Name     string
	Manifest routepolicy.Manifest
	Subject  routepolicy.Subject
	WantRule string
	WantPlan routepolicy.Plan
}

func single(model string) routepolicy.Plan {
	return routepolicy.Plan{Members: []routepolicy.Member{{Model: model}}}
}

var fallback = routepolicy.Plan{
	Members: []routepolicy.Member{{Model: "default", Role: "primary"}},
	Reason:  "fail-closed default",
}

func manifest(rules ...routepolicy.Rule) routepolicy.Manifest {
	return routepolicy.Manifest{Version: routepolicy.Version, Default: fallback, Rules: rules}
}

func rule(name string, m routepolicy.Match, p routepolicy.Plan) routepolicy.Rule {
	return routepolicy.Rule{Name: name, Match: m, Plan: p}
}

var ensemble = routepolicy.Plan{
	Members: []routepolicy.Member{{Model: "guard-a", Weight: 1}, {Model: "guard-b", Weight: 2, Role: "verifier"}},
	Reduce:  routepolicy.ReduceVote,
	Scout:   "scout-tiny",
	Reason:  "two-model guard ensemble",
}

// Cases returns the conformance table. Each call returns a fresh copy so a
// caller may mutate it.
func Cases() []Case {
	ordered := manifest(
		rule("first", routepolicy.Match{Aspect: routepolicy.AspectToolCall}, single("a")),
		rule("second", routepolicy.Match{Aspect: routepolicy.AspectToolCall}, single("b")),
	)
	specificFirst := manifest(
		rule("specific", routepolicy.Match{Aspect: routepolicy.AspectToolCall, Tool: "git_push"}, single("guard")),
		rule("general", routepolicy.Match{Aspect: routepolicy.AspectToolCall}, single("small")),
	)
	tools := manifest(
		rule("git-any", routepolicy.Match{Tool: "git_*"}, single("git-model")),
		rule("exact", routepolicy.Match{Tool: "search"}, single("search-model")),
		rule("star", routepolicy.Match{Tool: "*"}, single("any-tool")),
	)
	tokens := manifest(
		rule("short", routepolicy.Match{MaxPromptTokens: 100}, single("small")),
		rule("long", routepolicy.Match{MinPromptTokens: 1000}, single("large")),
	)
	latency := manifest(rule("interactive", routepolicy.Match{Latency: routepolicy.LatencyInteractive}, single("fast")))
	complexity := manifest(rule("hard", routepolicy.Match{MinComplexity: routepolicy.ComplexityMedium}, single("large")))
	labels := manifest(rule("eu-legal", routepolicy.Match{Labels: map[string]string{"region": "eu", "domain": "legal"}}, single("eu-legal-model")))
	trigger := manifest(rule("tool-result", routepolicy.Match{InputTrigger: inputtrigger.ToolResult}, single("continuation")))
	ens := manifest(rule("guard-writes", routepolicy.Match{Aspect: routepolicy.AspectToolCall, Tool: "write_*"}, ensemble))

	return []Case{
		{Name: "empty rules fall through to default", Manifest: manifest(), Subject: routepolicy.Subject{Aspect: routepolicy.AspectRequest}, WantPlan: fallback},
		{Name: "first match wins over a later identical match", Manifest: ordered, Subject: routepolicy.Subject{Aspect: routepolicy.AspectToolCall}, WantRule: "first", WantPlan: single("a")},
		{Name: "specific rule listed first wins", Manifest: specificFirst, Subject: routepolicy.Subject{Aspect: routepolicy.AspectToolCall, Tool: "git_push"}, WantRule: "specific", WantPlan: single("guard")},
		{Name: "later general rule catches the rest", Manifest: specificFirst, Subject: routepolicy.Subject{Aspect: routepolicy.AspectToolCall, Tool: "read"}, WantRule: "general", WantPlan: single("small")},
		{Name: "aspect mismatch falls through to default", Manifest: specificFirst, Subject: routepolicy.Subject{Aspect: routepolicy.AspectQuery, Tool: "git_push"}, WantPlan: fallback},
		{Name: "open aspect value routes only on exact match", Manifest: manifest(rule("custom", routepolicy.Match{Aspect: "my_stage"}, single("custom"))), Subject: routepolicy.Subject{Aspect: "my_stage"}, WantRule: "custom", WantPlan: single("custom")},
		{Name: "tool prefix wildcard matches", Manifest: tools, Subject: routepolicy.Subject{Tool: "git_commit"}, WantRule: "git-any", WantPlan: single("git-model")},
		{Name: "tool prefix wildcard matches the bare prefix", Manifest: tools, Subject: routepolicy.Subject{Tool: "git_"}, WantRule: "git-any", WantPlan: single("git-model")},
		{Name: "exact tool name matches", Manifest: tools, Subject: routepolicy.Subject{Tool: "search"}, WantRule: "exact", WantPlan: single("search-model")},
		{Name: "exact tool name is not a prefix match", Manifest: tools, Subject: routepolicy.Subject{Tool: "searcher"}, WantRule: "star", WantPlan: single("any-tool")},
		{Name: "lone star matches the empty tool", Manifest: tools, Subject: routepolicy.Subject{}, WantRule: "star", WantPlan: single("any-tool")},
		{Name: "max token bound is inclusive", Manifest: tokens, Subject: routepolicy.Subject{PromptTokens: 100}, WantRule: "short", WantPlan: single("small")},
		{Name: "between token bounds falls through", Manifest: tokens, Subject: routepolicy.Subject{PromptTokens: 101}, WantPlan: fallback},
		{Name: "min token bound is inclusive", Manifest: tokens, Subject: routepolicy.Subject{PromptTokens: 1000}, WantRule: "long", WantPlan: single("large")},
		{Name: "latency matches exactly", Manifest: latency, Subject: routepolicy.Subject{Latency: routepolicy.LatencyInteractive}, WantRule: "interactive", WantPlan: single("fast")},
		{Name: "latency mismatch falls through", Manifest: latency, Subject: routepolicy.Subject{Latency: routepolicy.LatencyBatch}, WantPlan: fallback},
		{Name: "unset subject latency does not match a latency rule", Manifest: latency, Subject: routepolicy.Subject{}, WantPlan: fallback},
		{Name: "complexity above the floor matches", Manifest: complexity, Subject: routepolicy.Subject{Complexity: routepolicy.ComplexityHigh}, WantRule: "hard", WantPlan: single("large")},
		{Name: "complexity at the floor matches", Manifest: complexity, Subject: routepolicy.Subject{Complexity: routepolicy.ComplexityMedium}, WantRule: "hard", WantPlan: single("large")},
		{Name: "complexity below the floor falls through", Manifest: complexity, Subject: routepolicy.Subject{Complexity: routepolicy.ComplexityLow}, WantPlan: fallback},
		{Name: "unknown complexity ranks below every floor", Manifest: complexity, Subject: routepolicy.Subject{Complexity: "extreme"}, WantPlan: fallback},
		{Name: "every label pair must hold", Manifest: labels, Subject: routepolicy.Subject{Labels: map[string]string{"region": "eu", "domain": "legal", "tenant": "t1"}}, WantRule: "eu-legal", WantPlan: single("eu-legal-model")},
		{Name: "a missing label falls through", Manifest: labels, Subject: routepolicy.Subject{Labels: map[string]string{"region": "eu"}}, WantPlan: fallback},
		{Name: "a differing label falls through", Manifest: labels, Subject: routepolicy.Subject{Labels: map[string]string{"region": "us", "domain": "legal"}}, WantPlan: fallback},
		{Name: "input trigger matches exactly", Manifest: trigger, Subject: routepolicy.Subject{InputTrigger: inputtrigger.ToolResult}, WantRule: "tool-result", WantPlan: single("continuation")},
		{Name: "other input trigger falls through", Manifest: trigger, Subject: routepolicy.Subject{InputTrigger: inputtrigger.UserMessage}, WantPlan: fallback},
		{Name: "unclassified input trigger falls through", Manifest: trigger, Subject: routepolicy.Subject{}, WantPlan: fallback},
		{Name: "ensemble plan is returned whole", Manifest: ens, Subject: routepolicy.Subject{Aspect: routepolicy.AspectToolCall, Tool: "write_file"}, WantRule: "guard-writes", WantPlan: ensemble},
	}
}

// Run drives every conformance Case through the Policy build returns. Each case
// checks the matched rule, the Matched flag, the chosen Plan, that the Subject is
// echoed unchanged, and that routing the same Subject twice is deterministic.
func Run(t *testing.T, build func(routepolicy.Manifest) routepolicy.Policy) {
	t.Helper()
	for _, c := range Cases() {
		c := c
		t.Run(c.Name, func(t *testing.T) {
			if err := c.Manifest.Validate(); err != nil {
				t.Fatalf("conformance manifest is invalid: %v", err)
			}
			p := build(c.Manifest)
			if p == nil {
				t.Fatal("build returned a nil Policy")
			}
			got := p.Route(c.Subject)
			if got.RuleName != c.WantRule {
				t.Errorf("RuleName = %q, want %q", got.RuleName, c.WantRule)
			}
			if wantMatched := c.WantRule != ""; got.Matched != wantMatched {
				t.Errorf("Matched = %v, want %v", got.Matched, wantMatched)
			}
			if !reflect.DeepEqual(got.Plan, c.WantPlan) {
				t.Errorf("Plan = %+v, want %+v", got.Plan, c.WantPlan)
			}
			if !reflect.DeepEqual(got.Subject, c.Subject) {
				t.Errorf("Subject echo = %+v, want %+v", got.Subject, c.Subject)
			}
			if again := p.Route(c.Subject); !reflect.DeepEqual(again, got) {
				t.Errorf("Route is not deterministic: %+v then %+v", got, again)
			}
		})
	}
}
