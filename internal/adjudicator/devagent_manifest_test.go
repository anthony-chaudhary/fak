package adjudicator_test

import (
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// The shipped dev-agent manifest must be a faithful, loadable encoding of the
// in-code DevAgentPolicy — so `--policy examples/dev-agent-policy.json` deploys
// EXACTLY the preset (no fork, no drift). This is an external test package so it can
// import the policy loader (which imports adjudicator) without an import cycle.
func TestDevAgentManifestMatchesPreset(t *testing.T) {
	b, err := os.ReadFile("../../examples/dev-agent-policy.json")
	if err != nil {
		t.Fatalf("read manifest: %v", err)
	}
	got, err := policy.Parse(b)
	if err != nil {
		t.Fatalf("parse manifest: %v", err)
	}
	want := adjudicator.DevAgentPolicy()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("examples/dev-agent-policy.json does not round-trip to DevAgentPolicy()\n got=%+v\nwant=%+v", got, want)
	}
}
