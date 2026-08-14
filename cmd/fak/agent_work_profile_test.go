package main

import (
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestResolveAgentWorkProfileFailsClosedOnOriginal(t *testing.T) {
	if _, err := resolveAgentWorkProfile("ponytail:original:medium"); err == nil || !strings.Contains(err.Error(), "invalid --work-profile") {
		t.Fatalf("resolve original err = %v", err)
	}
}

func TestApplyAgentWorkProfileRestoresEnvironment(t *testing.T) {
	t.Setenv(syspromptmmu.WorkProfileEnvVar, "prior")
	profile, _ := resolveAgentWorkProfile("ponytail:high")
	restore, err := applyAgentWorkProfile(profile)
	if err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv(syspromptmmu.WorkProfileEnvVar); got != "ponytail:native:high" {
		t.Fatalf("env = %q", got)
	}
	restore()
	if got := os.Getenv(syspromptmmu.WorkProfileEnvVar); got != "prior" {
		t.Fatalf("restored env = %q", got)
	}
}

func TestAgentProfileCatalogShowsIndependentAxes(t *testing.T) {
	var out strings.Builder
	if err := printAgentOutputProfiles(&out, nil); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Response profiles", "Work profiles", "ponytail:medium", "ponytail:original:*", "--work-profile standard", "mix independently"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("catalog missing %q:\n%s", want, out.String())
		}
	}
}
