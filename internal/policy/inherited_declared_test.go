package policy

import (
	"slices"
	"strings"
	"testing"
)

// The #2358 secret floor strips a credential-shaped variable from every spawned child's
// environment. That is right for INCIDENTAL inheritance and wrong for the one variable a
// launch surface was configured to authenticate the child with — the floor cannot tell them
// apart, so the surface declares which is which. These tests pin both halves: the declared
// name passes, and nothing else the floor holds out starts passing with it.

// TestStripInheritedSecretsExceptPassesDeclaredCredential is the core fix. A bearer-token
// variable trips BOTH halves of the floor (its NAME contains TOKEN, and a vendor PAT VALUE can
// look secret-shaped), so a keep-set that only exempted one half would still strip it.
func TestStripInheritedSecretsExceptPassesDeclaredCredential(t *testing.T) {
	ambient := []string{
		"PATH=/usr/bin:/bin",
		"ANTHROPIC_BASE_URL=https://gateway.example.com/serving-endpoints/anthropic",
		"ANTHROPIC_AUTH_TOKEN=sk-vendor-tenant-pat", // sk- prefix => value-shaped too
		"GITHUB_TOKEN=ghp_unrelated_canary",
	}
	kept, stripped := StripInheritedSecretsExcept(ambient, []string{"ANTHROPIC_AUTH_TOKEN"})

	joined := strings.Join(kept, "\n")
	if !strings.Contains(joined, "ANTHROPIC_AUTH_TOKEN=sk-vendor-tenant-pat") {
		t.Fatalf("declared credential stripped anyway; the child would authenticate with nothing:\n%s", joined)
	}
	if slices.Contains(stripped, "ANTHROPIC_AUTH_TOKEN") {
		t.Error("declared credential reported as stripped; the audit would misdescribe the launch")
	}
	// Declaring one credential must not open the floor for the rest.
	if strings.Contains(joined, "GITHUB_TOKEN") {
		t.Error("an UNDECLARED credential passed the floor — the #2358 exfiltration path reopened")
	}
	if !slices.Contains(stripped, "GITHUB_TOKEN") {
		t.Errorf("GITHUB_TOKEN not recorded as stripped: %v", stripped)
	}
	if !strings.Contains(joined, "PATH=/usr/bin:/bin") || !strings.Contains(joined, "ANTHROPIC_BASE_URL=") {
		t.Errorf("non-secret config dropped: %v", kept)
	}
}

// TestStripInheritedSecretsUnchangedWithoutDeclaration is the no-regression assertion: the
// always-on floor with no keep-set must behave exactly as it did, so every existing spawn
// surface keeps its posture bit-for-bit.
func TestStripInheritedSecretsUnchangedWithoutDeclaration(t *testing.T) {
	ambient := []string{
		"PATH=/usr/bin",
		"CLAUDE_CONFIG_DIR=/home/agent/.claude",
		"ANTHROPIC_API_KEY=sk-ant-api-billing-survives",
		"CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat-held-out",
		"FAK_CANARY_SECRET=sk-proj-held-out",
		"MALFORMED_NO_EQUALS",
	}
	wantKept, wantStripped := StripInheritedSecrets(ambient)

	// nil and empty declarations are both "declare nothing".
	for name, declared := range map[string][]string{
		"nil":        nil,
		"empty":      {},
		"whitespace": {"", "   "},
	} {
		kept, stripped := StripInheritedSecretsExcept(ambient, declared)
		if !slices.Equal(kept, wantKept) {
			t.Errorf("declared=%s changed the kept set:\n got %v\nwant %v", name, kept, wantKept)
		}
		if !slices.Equal(stripped, wantStripped) {
			t.Errorf("declared=%s changed the stripped set:\n got %v\nwant %v", name, stripped, wantStripped)
		}
	}
	// Spot-check the historical contract itself, so this test fails loudly if the floor drifts.
	if !slices.Contains(wantStripped, "CLAUDE_CODE_OAUTH_TOKEN") || !slices.Contains(wantStripped, "FAK_CANARY_SECRET") {
		t.Errorf("the always-on floor stopped stripping a credential: %v", wantStripped)
	}
	if !slices.Contains(wantKept, "ANTHROPIC_API_KEY=sk-ant-api-billing-survives") {
		t.Errorf("provider API key no longer survives: %v", wantKept)
	}
	if !slices.Contains(wantKept, "MALFORMED_NO_EQUALS") {
		t.Errorf("malformed entry no longer preserved: %v", wantKept)
	}
}

// TestStripInheritedSecretsExceptMatchesNameExactly pins that the exemption is case-SENSITIVE.
// POSIX environment names are, so a case-folded keep-set would let a mistyped lowercase entry
// in a config file exempt the real credential — a typo must not widen a security boundary.
func TestStripInheritedSecretsExceptMatchesNameExactly(t *testing.T) {
	ambient := []string{"ANTHROPIC_AUTH_TOKEN=sk-real-credential"}
	for _, declared := range []string{"anthropic_auth_token", "Anthropic_Auth_Token", "ANTHROPIC_AUTH_TOKE"} {
		kept, stripped := StripInheritedSecretsExcept(ambient, []string{declared})
		if len(kept) != 0 {
			t.Errorf("declared %q exempted a name it does not equal: kept %v", declared, kept)
		}
		if !slices.Contains(stripped, "ANTHROPIC_AUTH_TOKEN") {
			t.Errorf("declared %q: expected the real name to still be stripped, got %v", declared, stripped)
		}
	}
	// Surrounding whitespace in a hand-edited config is forgiven, since it is not ambiguous.
	if kept, _ := StripInheritedSecretsExcept(ambient, []string{"  ANTHROPIC_AUTH_TOKEN  "}); len(kept) != 1 {
		t.Errorf("a padded declaration failed to match: kept %v", kept)
	}
}

// TestStripInheritedSecretsExceptDeclarationIsNotInjection pins that declaring is PERMISSION
// for a variable to pass, never an instruction to create one: a declared name absent from the
// ambient set must not appear in the output (an empty value would look like a real, blank
// credential to the child).
func TestStripInheritedSecretsExceptDeclarationIsNotInjection(t *testing.T) {
	kept, stripped := StripInheritedSecretsExcept([]string{"PATH=/usr/bin"}, []string{"ANTHROPIC_AUTH_TOKEN"})
	if !slices.Equal(kept, []string{"PATH=/usr/bin"}) {
		t.Fatalf("declaring an absent variable altered the environment: %v", kept)
	}
	if len(stripped) != 0 {
		t.Errorf("stripped = %v, want none", stripped)
	}
}

// TestStripInheritedSecretsExceptIsNotTransitive is the containment argument for the whole
// change: the exemption belongs to ONE launch. Re-running the floor on the resulting
// environment — which is what the next spawn surface down does with its own keep-set — strips
// the credential again, so a declared secret does not follow the child into its descendants.
func TestStripInheritedSecretsExceptIsNotTransitive(t *testing.T) {
	childEnv, _ := StripInheritedSecretsExcept(
		[]string{"PATH=/usr/bin", "ANTHROPIC_AUTH_TOKEN=sk-vendor-tenant-pat"},
		[]string{"ANTHROPIC_AUTH_TOKEN"})
	if !slices.Contains(childEnv, "ANTHROPIC_AUTH_TOKEN=sk-vendor-tenant-pat") {
		t.Fatalf("setup: the declared credential should have reached the child: %v", childEnv)
	}
	grandchildEnv, stripped := StripInheritedSecretsExcept(childEnv, nil)
	if slices.Contains(grandchildEnv, "ANTHROPIC_AUTH_TOKEN=sk-vendor-tenant-pat") {
		t.Fatal("the declared credential survived a second, undeclared hop — the exemption is transitive and #2358 is reopened")
	}
	if !slices.Contains(stripped, "ANTHROPIC_AUTH_TOKEN") {
		t.Errorf("second hop did not record the strip: %v", stripped)
	}
}
