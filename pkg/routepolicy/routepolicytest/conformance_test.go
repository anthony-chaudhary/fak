package routepolicytest_test

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
	"github.com/anthony-chaudhary/fak/pkg/routepolicy/routepolicytest"
)

// fak-test:runtime fast est=10ms lane=default
func TestRunReferenceManifestPassesAllCases(t *testing.T) {
	if len(routepolicytest.Cases()) == 0 {
		t.Fatal("Cases() returned empty conformance suite")
	}
	routepolicytest.Run(t, func(m routepolicy.Manifest) routepolicy.Policy { return m })
}

// fak-test:runtime fast est=10ms lane=default
func TestCasesAreUniqueValidAndFreshCopies(t *testing.T) {
	c1 := routepolicytest.Cases()
	if len(c1) < 20 {
		t.Fatalf("Cases() count = %d, expected comprehensive conformance suite", len(c1))
	}

	seen := make(map[string]bool, len(c1))
	for i, c := range c1 {
		if c.Name == "" {
			t.Errorf("case %d has empty Name", i)
		}
		if seen[c.Name] {
			t.Errorf("duplicate case Name %q", c.Name)
		}
		seen[c.Name] = true
		if err := c.Manifest.Validate(); err != nil {
			t.Errorf("case %q invalid manifest: %v", c.Name, err)
		}
	}

	// Mutating a returned slice and its manifests must not pollute subsequent Calls.
	origRule := c1[1].WantRule
	origRuleName := c1[1].Manifest.Rules[0].Name
	c1[1].WantRule = "tampered"
	c1[1].Manifest.Rules[0].Name = "tampered-rule"
	c1[0] = routepolicytest.Case{Name: "overwritten"}

	c2 := routepolicytest.Cases()
	if c2[1].WantRule != origRule || c2[1].Manifest.Rules[0].Name != origRuleName {
		t.Errorf("Cases() shared mutable state across calls: got (%q, %q), want (%q, %q)",
			c2[1].WantRule, c2[1].Manifest.Rules[0].Name, origRule, origRuleName)
	}
	if c2[0].Name == "overwritten" {
		t.Errorf("Cases() slice backing array mutated across calls")
	}
}

type wrongPolicyFunc func(routepolicy.Subject) routepolicy.Decision

func (f wrongPolicyFunc) Route(s routepolicy.Subject) routepolicy.Decision { return f(s) }

// fak-test:runtime fast est=10ms lane=default
func TestCasesDiscriminateBrokenPolicies(t *testing.T) {
	broken := map[string]func(routepolicy.Manifest) routepolicy.Policy{
		"always-default": func(m routepolicy.Manifest) routepolicy.Policy {
			return wrongPolicyFunc(func(s routepolicy.Subject) routepolicy.Decision {
				return routepolicy.Decision{Subject: s, Plan: m.Default}
			})
		},
		"ignore-labels": func(m routepolicy.Manifest) routepolicy.Policy {
			return wrongPolicyFunc(func(s routepolicy.Subject) routepolicy.Decision {
				s2 := s
				s2.Labels = nil
				d := m.Route(s2)
				d.Subject = s
				return d
			})
		},
		"ignore-input-trigger": func(m routepolicy.Manifest) routepolicy.Policy {
			return wrongPolicyFunc(func(s routepolicy.Subject) routepolicy.Decision {
				s2 := s
				s2.InputTrigger = ""
				d := m.Route(s2)
				d.Subject = s
				return d
			})
		},
		"wrong-matched-flag": func(m routepolicy.Manifest) routepolicy.Policy {
			return wrongPolicyFunc(func(s routepolicy.Subject) routepolicy.Decision {
				d := m.Route(s)
				d.Matched = true
				return d
			})
		},
		"corrupt-subject-echo": func(m routepolicy.Manifest) routepolicy.Policy {
			return wrongPolicyFunc(func(s routepolicy.Subject) routepolicy.Decision {
				d := m.Route(s)
				d.Subject = routepolicy.Subject{Tool: "mutated"}
				return d
			})
		},
	}

	cases := routepolicytest.Cases()
	for name, build := range broken {
		caught := false
		for _, c := range cases {
			got := build(c.Manifest).Route(c.Subject)
			wantMatched := c.WantRule != ""
			if got.RuleName != c.WantRule || got.Matched != wantMatched ||
				!reflect.DeepEqual(got.Plan, c.WantPlan) || !reflect.DeepEqual(got.Subject, c.Subject) {
				caught = true
				break
			}
		}
		if !caught {
			t.Errorf("conformance Cases() failed to detect broken policy %q", name)
		}
	}
}
