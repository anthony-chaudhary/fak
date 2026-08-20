package cloudroute

import (
	"strings"
	"testing"
)

func TestDetectBedrockFromTheSelector(t *testing.T) {
	r, ok := Detect([]string{"PATH=/usr/bin", "CLAUDE_CODE_USE_BEDROCK=1", "AWS_PROFILE=corp-sso", "AWS_REGION=us-east-1"})
	if !ok {
		t.Fatal("Detect returned false for a Bedrock-selected environment")
	}
	if r.Kind != KindBedrock || r.Selector != "CLAUDE_CODE_USE_BEDROCK" {
		t.Fatalf("Route = %+v, want the bedrock kind named by its selector", r)
	}
	want := "AWS_PROFILE,AWS_REGION,CLAUDE_CODE_USE_BEDROCK"
	if got := strings.Join(r.Observed, ","); got != want {
		t.Fatalf("Observed = %q, want %q (sorted names only, so a refusal is safe to log)", got, want)
	}
}

func TestDetectVertex(t *testing.T) {
	r, ok := Detect([]string{"CLAUDE_CODE_USE_VERTEX=true", "GOOGLE_CLOUD_PROJECT=corp-ml"})
	if !ok || r.Kind != KindVertex {
		t.Fatalf("Route = %+v ok=%v, want the vertex kind", r, ok)
	}
}

// A subscription-routed box that merely HAS an AWS profile is not a cloud route.
// Detecting on credential presence rather than on the selector would refuse every
// developer who has ever run `aws configure`.
func TestDetectIgnoresIncidentalCloudCredentials(t *testing.T) {
	if r, ok := Detect([]string{"AWS_PROFILE=personal", "AWS_REGION=us-west-2", "GOOGLE_CLOUD_PROJECT=side-project"}); ok {
		t.Fatalf("Detect = %+v, want false: credentials in the environment are incidental, the SELECTOR is the declaration", r)
	}
}

func TestDetectTreatsAnEmptyOrFalseSelectorAsOff(t *testing.T) {
	for _, v := range []string{"", "0", "false", "no", "off"} {
		if r, ok := Detect([]string{"CLAUDE_CODE_USE_BEDROCK=" + v}); ok {
			t.Fatalf("selector %q detected %+v, want off", v, r)
		}
	}
	for _, v := range []string{"1", "true", "TRUE", "yes", "y", "on", " 1 "} {
		if _, ok := Detect([]string{"CLAUDE_CODE_USE_BEDROCK=" + v}); !ok {
			t.Fatalf("selector %q not detected, want on", v)
		}
	}
}

// Windows environment names are case-insensitive and a shell may well export a
// mixed-case name, so detection must fold. A missed detection there is the exact
// wrong-cause refusal this package exists to prevent.
func TestDetectFoldsNameCase(t *testing.T) {
	if _, ok := Detect([]string{"Claude_Code_Use_Bedrock=1"}); !ok {
		t.Fatal("mixed-case selector not detected")
	}
}

func TestDetectPrefersBedrockDeterministicallyWhenBothAreSet(t *testing.T) {
	r, _ := Detect([]string{"CLAUDE_CODE_USE_VERTEX=1", "CLAUDE_CODE_USE_BEDROCK=1"})
	if r.Kind != KindBedrock {
		t.Fatalf("Kind = %q, want a deterministic winner (bedrock) so the refusal is never ambiguous", r.Kind)
	}
}

func TestDetectReportsTheWaiver(t *testing.T) {
	r, ok := Detect([]string{"CLAUDE_CODE_USE_BEDROCK=1", WaiverKey + "=1"})
	if !ok || !r.Waived {
		t.Fatalf("Route = %+v ok=%v, want Waived so the caller warns instead of refusing", r, ok)
	}
	r, _ = Detect([]string{"CLAUDE_CODE_USE_BEDROCK=1"})
	if r.Waived {
		t.Fatal("Waived must default to false — refusal is the default, the waiver is opt-in")
	}
}

func TestNonSecretEnvNamesExcludesTheCredentialChain(t *testing.T) {
	for _, n := range NonSecretEnvNames() {
		if credentialShaped[n] {
			t.Fatalf("%s is credential-shaped and must not enter a static allow-list", n)
		}
	}
	var sawSelector, sawRegion bool
	for _, n := range NonSecretEnvNames() {
		switch n {
		case "CLAUDE_CODE_USE_BEDROCK":
			sawSelector = true
		case "AWS_REGION":
			sawRegion = true
		}
	}
	if !sawSelector || !sawRegion {
		t.Fatalf("NonSecretEnvNames = %v, want the route selectors and region config enumerated", NonSecretEnvNames())
	}
}

func TestCredentialNamesAreDeclaredOnlyWhenPresent(t *testing.T) {
	r, _ := Detect([]string{"CLAUDE_CODE_USE_BEDROCK=1", "AWS_SESSION_TOKEN=xyz", "AWS_PROFILE=corp"})
	got := r.CredentialNames([]string{"CLAUDE_CODE_USE_BEDROCK=1", "AWS_SESSION_TOKEN=xyz", "AWS_PROFILE=corp"})
	if strings.Join(got, ",") != "AWS_SESSION_TOKEN" {
		t.Fatalf("CredentialNames = %v, want only the credential-shaped name that is actually SET (declaring is permission, never invention)", got)
	}
	if got := r.CredentialNames(nil); len(got) != 0 {
		t.Fatalf("CredentialNames = %v, want none for an empty environment", got)
	}
}

func TestExplainNamesTheInertRepointAndCarriesNoValues(t *testing.T) {
	r, _ := Detect([]string{"CLAUDE_CODE_USE_BEDROCK=1", "AWS_PROFILE=corp-sso-9f3a"})
	got := r.Explain()
	for _, want := range []string{"AWS Bedrock", "CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_BASE_URL", "AWS_PROFILE"} {
		if !strings.Contains(got, want) {
			t.Fatalf("Explain() = %q, missing %q", got, want)
		}
	}
	if strings.Contains(got, "corp-sso-9f3a") {
		t.Fatalf("Explain() leaked a VALUE: %q", got)
	}
}

func TestEnvNamesIsCopiedNotAliased(t *testing.T) {
	got := EnvNames(KindBedrock)
	if len(got) == 0 {
		t.Fatal("EnvNames(bedrock) is empty")
	}
	got[0] = "MUTATED"
	if EnvNames(KindBedrock)[0] == "MUTATED" {
		t.Fatal("EnvNames handed out the package slice; a caller could rewrite the vocabulary")
	}
	if EnvNames(Kind("nope")) != nil {
		t.Fatal("EnvNames of an unknown kind must be nil")
	}
}
