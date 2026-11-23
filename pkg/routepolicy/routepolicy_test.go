package routepolicy_test

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
	"github.com/anthony-chaudhary/fak/pkg/routepolicy/routepolicytest"
)

func TestManifestConformance(t *testing.T) {
	routepolicytest.Run(t, func(m routepolicy.Manifest) routepolicy.Policy { return m })
}

type lastMatchWins struct{ m routepolicy.Manifest }

func (p lastMatchWins) Route(s routepolicy.Subject) routepolicy.Decision {
	for i := len(p.m.Rules) - 1; i >= 0; i-- {
		if p.m.Rules[i].Match.Matches(s) {
			return routepolicy.Decision{Subject: s, Plan: p.m.Rules[i].Plan, RuleName: p.m.Rules[i].Name, Matched: true}
		}
	}
	return routepolicy.Decision{Subject: s, Plan: p.m.Default}
}

type noDefault struct{ m routepolicy.Manifest }

func (p noDefault) Route(s routepolicy.Subject) routepolicy.Decision {
	d := p.m.Route(s)
	if !d.Matched {
		d.Plan = routepolicy.Plan{}
	}
	return d
}

// The kit is only a witness if it can fail: each wrong policy must disagree with
// the expected decision on at least one case.
func TestKitRejectsWrongPolicies(t *testing.T) {
	for name, build := range map[string]func(routepolicy.Manifest) routepolicy.Policy{
		"last-match-wins": func(m routepolicy.Manifest) routepolicy.Policy { return lastMatchWins{m} },
		"no-default":      func(m routepolicy.Manifest) routepolicy.Policy { return noDefault{m} },
	} {
		caught := false
		for _, c := range routepolicytest.Cases() {
			got := build(c.Manifest).Route(c.Subject)
			if got.RuleName != c.WantRule || !reflect.DeepEqual(got.Plan, c.WantPlan) {
				caught = true
				break
			}
		}
		if !caught {
			t.Errorf("conformance kit passes the wrong policy %q", name)
		}
	}
}

func TestDefaultManifestRoundTripsThroughJSON(t *testing.T) {
	want := routepolicy.DefaultManifest()
	got, err := routepolicy.ParseManifest(want.JSON())
	if err != nil {
		t.Fatalf("ParseManifest(DefaultManifest().JSON()): %v", err)
	}
	if string(got.JSON()) != string(want.JSON()) {
		t.Fatalf("round trip changed the manifest:\n%s\nvs\n%s", got.JSON(), want.JSON())
	}
}

func TestParseManifestRefusesForeignMajorVersion(t *testing.T) {
	_, err := routepolicy.ParseManifest([]byte(`{"version":"fak-route/v2","default":{"members":[{"model":"m"}]}}`))
	const want = `modelroute: manifest version "fak-route/v2" is not fak-route/v1.x`
	if err == nil || err.Error() != want {
		t.Fatalf("err = %v, want %q", err, want)
	}
}
