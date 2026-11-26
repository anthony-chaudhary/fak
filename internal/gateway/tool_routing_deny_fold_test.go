package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// fak-test:runtime fast est=200ms lane=default
func TestCanonicalToolName_DenyMatchesAnyCasing(t *testing.T) {
	pol := &policy.Runtime{Adjudicator: adjudicator.Policy{
		Posture: adjudicator.PostureDefaultOpen,
		Deny:    map[string]abi.ReasonCode{"exfiltrate": abi.ReasonPolicyBlock},
	}}
	adj := adjudicator.New(pol.Adjudicator)
	for _, wire := range []string{
		"exfiltrate",
		"Exfiltrate",
		"mcp__srv__exfiltrate",
		"mcp__srv__Exfiltrate",
	} {
		canon := canonicalToolName(wire, pol)
		v := adj.Adjudicate(context.Background(), &abi.ToolCall{Tool: canon})
		if v.Kind != abi.VerdictDeny {
			t.Errorf("wire %q canonicalised to %q: verdict %v, want Deny", wire, canon, v.Kind)
		}
	}
}
