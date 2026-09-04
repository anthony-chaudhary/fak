package policy

import (
	"testing"
)

func TestPolicyIslandsWiring(t *testing.T) {
	t.Run("TieredEvaluator wired into Manifest", func(t *testing.T) {
		m := Manifest{
			Allow: []string{
				"custom_tool",
			},
		}

		eval := m.NewTieredEvaluator()
		if eval == nil {
			t.Fatal("expected non-nil TieredEvaluator from Manifest")
		}

		// Evaluate custom_tool on convenience surface
		dec := m.EvaluateAgainstTiers("custom_tool", `{}`)
		if !dec.Allowed {
			t.Fatalf("expected custom_tool to be allowed, got: %+v", dec)
		}
		if dec.Tier != TierConvenienceSurface {
			t.Fatalf("expected tier %q, got %q", TierConvenienceSurface, dec.Tier)
		}

		// Evaluate destructive operation against Frozen Core
		ssrfDec := m.EvaluateAgainstTiers("WebFetch", `{"url":"http://169.254.169.254/latest/meta-data"}`)
		if ssrfDec.Allowed {
			t.Fatalf("expected SSRF to be refused by Frozen Core, got: %+v", ssrfDec)
		}
		if ssrfDec.Tier != TierFrozenCore {
			t.Fatalf("expected tier %q, got %q", TierFrozenCore, ssrfDec.Tier)
		}
	})
}
