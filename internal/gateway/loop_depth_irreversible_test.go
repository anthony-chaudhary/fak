package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

func toolDefNamed(name string) agent.ToolDef {
	return agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: name}}
}

// fak-test:runtime fast est=2ms lane=default
func TestLoopDepthIrreversibleFloorSurvivesPolicyCapAtEffortNone(t *testing.T) {
	capped := &policy.LoopDepthRule{MaxLoopDepth: 1}

	irreversible := ChatRequest{Model: "looped-stub", ReasoningEffort: "none", Tools: []agent.ToolDef{toolDefNamed("bash")}}
	got, msg := resolveLoopDepth(irreversible, 8, 3, capped)
	if msg != "" || got == nil || got.Loops != 3 {
		t.Fatalf("irreversible tool scope at effort none: receipt %+v msg %q, want Loops 3", got, msg)
	}

	reversible := ChatRequest{Model: "looped-stub", ReasoningEffort: "none", Tools: []agent.ToolDef{toolDefNamed("read_file")}}
	got, msg = resolveLoopDepth(reversible, 8, 3, capped)
	if msg != "" || got == nil || got.Loops != 1 {
		t.Fatalf("reversible tool scope under cap 1: receipt %+v msg %q, want Loops 1", got, msg)
	}
}
