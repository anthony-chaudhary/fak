//go:build wip_agent_systools

package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/systools"
)

func TestAgentSysToolsFlagDefaultsOn(t *testing.T) {
	fs, af := newAgentFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse empty args: %v", err)
	}
	if af.sysTools == nil || !*af.sysTools {
		t.Fatal("--sys-tools defaulted off for agent")
	}

	fs2, af2 := newAgentFlagSet()
	if err := fs2.Parse([]string{"--sys-tools=false"}); err != nil {
		t.Fatalf("Parse --sys-tools=false: %v", err)
	}
	if af2.sysTools == nil || *af2.sysTools {
		t.Fatal("--sys-tools=false did not disable for agent")
	}
}

func TestChatSysToolsFlagDefaultsOn(t *testing.T) {
	fs, cf := newChatFlagSet()
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse empty args: %v", err)
	}
	if cf.sysTools == nil || !*cf.sysTools {
		t.Fatal("--sys-tools defaulted off for chat")
	}

	fs2, cf2 := newChatFlagSet()
	if err := fs2.Parse([]string{"--sys-tools=false"}); err != nil {
		t.Fatalf("Parse --sys-tools=false: %v", err)
	}
	if cf2.sysTools == nil || *cf2.sysTools {
		t.Fatal("--sys-tools=false did not disable for chat")
	}
}

func TestArmSysToolsExposesExpectedToolsInCatalog(t *testing.T) {
	catalog, err := agent.ArmSysTools(systools.Config{})
	if err != nil {
		t.Fatalf("ArmSysTools: %v", err)
	}
	defer agent.DisarmSysTools()

	found := make(map[string]bool)
	for _, tool := range catalog {
		found[tool.Function.Name] = true
	}

	for _, want := range []string{"get_time", "fetch_web", "web_search"} {
		if !found[want] {
			t.Errorf("expected tool %q in catalog, got %v", want, catalog)
		}
	}
}
