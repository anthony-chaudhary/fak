package main

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestResolveAgentOutputStyle(t *testing.T) {
	got, err := resolveAgentOutputStyle(" Caveman:Native:Medium ")
	if err != nil || got.Style != "caveman:native:medium" || got.Family != "caveman:native" || got.Intensity != "medium" || got.Level != 2 {
		t.Fatalf("resolve = %+v, err=%v", got, err)
	}
	original, err := resolveAgentOutputStyle("caveman:original:medium")
	if err != nil || original.Style != "caveman:original:medium" || original.Family != "caveman:original" || original.Intensity != "medium" {
		t.Fatalf("original profile resolve = %+v, err=%v", original, err)
	}
}

func TestApplyAgentOutputStyleRestoresEnvironment(t *testing.T) {
	const prior = "native:low"
	t.Setenv(syspromptmmu.StyleEnvVar, prior)
	style, _ := resolveAgentOutputStyle("caveman:native:high")
	restore, err := applyAgentOutputStyle(style)
	if err != nil {
		t.Fatal(err)
	}
	if got := os.Getenv(syspromptmmu.StyleEnvVar); got != "caveman:native:high" {
		t.Fatalf("active=%q", got)
	}
	restore()
	if got := os.Getenv(syspromptmmu.StyleEnvVar); got != prior {
		t.Fatalf("restored=%q", got)
	}
}

func TestAgentOutputProfilesUserReadout(t *testing.T) {
	var out bytes.Buffer
	if err := printAgentOutputProfiles(&out, nil); err != nil {
		t.Fatal(err)
	}
	text := out.String()
	for _, want := range []string{
		"default is caveman:medium",
		"caveman:original:*",
		"not-yet",
		"--output-style full",
		"independent axis",
		"--set-default adapt.output-style=caveman:medium",
		"--set-default adapt.output-style=full",
		"CLI --output-style > persisted preference > shipped default",
		"Sweep controls: fak agent profiles --sweep",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("profiles readout missing %q:\n%s", want, text)
		}
	}
}

func TestAgentOutputProfilesJSON(t *testing.T) {
	var out bytes.Buffer
	if err := printAgentOutputProfiles(&out, []string{"--json"}); err != nil {
		t.Fatal(err)
	}
	var got []agentOutputProfile
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got) < 8 || got[5].Selection != "caveman:medium" || got[5].Canonical != "caveman:native:medium" || got[5].Status != "default" || got[7].Status != "not-yet" {
		t.Fatalf("profile JSON = %+v", got)
	}
}

func TestAgentDefaultOutputStyleIsCavemanMedium(t *testing.T) {
	got, err := resolveAgentOutputStyle(agentDefaultOutputStyle)
	if err != nil {
		t.Fatal(err)
	}
	if got.Style != "caveman:native:medium" || got.Level != 2 {
		t.Fatalf("default style = %+v", got)
	}
}
