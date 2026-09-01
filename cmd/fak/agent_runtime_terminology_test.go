package main

import (
	"strings"
	"testing"
)

const agentRuntimeBoundaryDefinition = "fak is an agent runtime: the operator-controlled boundary for cache and context, model routing, tool authority, memory, observability, and native inference"

// TestAgentRuntimeFrontDoorsKeepTheCategoryHierarchy deliberately checks a
// small allow-list of public positioning surfaces. Technical and historical
// uses of "agent kernel" elsewhere are valid and must not be globally replaced.
func TestAgentRuntimeFrontDoorsKeepTheCategoryHierarchy(t *testing.T) {
	frontDoors := []string{
		"llms.txt",
		"docs/index.md",
		"docs/adoption/naming.md",
		"docs/adoption/pitch-ladder.md",
		"docs/launch/positioning-brief.md",
		"docs/explainers/agent-runtime.md",
	}

	for _, path := range frontDoors {
		t.Run(path, func(t *testing.T) {
			body := normalizeTerminology(readRepoFile(t, path))
			at := strings.Index(body, agentRuntimeBoundaryDefinition)
			if at < 0 {
				t.Fatalf("%s must define the public agent-runtime category as %q", path, agentRuntimeBoundaryDefinition)
			}
			if at > 2500 {
				t.Errorf("%s hides the public category definition after byte %d; keep it in prominent lead messaging", path, at)
			}
			if !strings.Contains(body, "Fused Agent Kernel") {
				t.Errorf("%s must retain Fused Agent Kernel as the technical name", path)
			}
			if !strings.Contains(body, "agent kernel") {
				t.Errorf("%s must retain agent kernel as the technical architecture term", path)
			}
		})
	}
}

func TestAgentRuntimeNamingKeepsTheShippedScopeFence(t *testing.T) {
	body := normalizeTerminology(readRepoFile(t, "docs/adoption/naming.md"))
	for _, required := range []string{
		"does not claim every possible orchestration or runtime feature is shipped",
		"CLAIMS.md",
	} {
		if !strings.Contains(body, required) {
			t.Errorf("docs/adoption/naming.md must keep the category/status fence %q", required)
		}
	}
}

func normalizeTerminology(s string) string {
	s = strings.NewReplacer("\r", " ", "\n", " ", "**", "", "`", "", ">", " ").Replace(s)
	return strings.Join(strings.Fields(s), " ")
}
