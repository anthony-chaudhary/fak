package main

import (
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
	if _, err := resolveAgentOutputStyle("caveman:original:medium"); err == nil || !strings.Contains(err.Error(), "invalid --output-style") {
		t.Fatalf("original profile should fail closed, err=%v", err)
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
