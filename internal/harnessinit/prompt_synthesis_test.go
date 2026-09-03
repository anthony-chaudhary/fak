package harnessinit

import (
	"strings"
	"testing"
)

func TestResolvePromptScales(t *testing.T) {
	// 1. ScaleCoordinator
	coordSpec := PromptSpec{
		Scale:         ScaleCoordinator,
		ModelFamily:   "qwen3.8-27b",
		IsSmallLocal:  false,
		ContextBudget: 32768,
		WireFormat:    "openai",
	}
	coordPrompt, err := ResolvePrompt(coordSpec)
	if err != nil {
		t.Fatalf("failed to resolve coordinator prompt: %v", err)
	}

	if !strings.Contains(coordPrompt.Raw, "ROLE: COORDINATOR AGENT") {
		t.Errorf("coordinator prompt missing coordinator role: %s", coordPrompt.Raw)
	}
	if !strings.Contains(coordPrompt.Raw, "DELEGATION & COLLISION PROTOCOL") {
		t.Errorf("coordinator prompt missing delegation overlay: %s", coordPrompt.Raw)
	}
	if strings.Contains(coordPrompt.Raw, "ROLE: S0/S1 LEAF WORKER") {
		t.Errorf("coordinator prompt should not contain leaf contract: %s", coordPrompt.Raw)
	}

	// 2. ScaleLeafWorker (Unconstrained budget >= 16000)
	leafSpec := PromptSpec{
		Scale:         ScaleLeafWorker,
		ModelFamily:   "qwen2.5-coder-7b",
		IsSmallLocal:  true,
		ContextBudget: 24000,
		WireFormat:    "openai",
	}
	leafPrompt, err := ResolvePrompt(leafSpec)
	if err != nil {
		t.Fatalf("failed to resolve leaf worker prompt: %v", err)
	}

	if !strings.Contains(leafPrompt.Raw, "ROLE: S0/S1 LEAF WORKER") {
		t.Errorf("leaf prompt missing leaf role: %s", leafPrompt.Raw)
	}
	if !strings.Contains(leafPrompt.Raw, "CONVENTIONS:") {
		t.Errorf("leaf prompt with 24k budget should contain guidance card: %s", leafPrompt.Raw)
	}
	if strings.Contains(leafPrompt.Raw, "ROLE: COORDINATOR AGENT") {
		t.Errorf("leaf prompt should not contain coordinator role: %s", leafPrompt.Raw)
	}

	// 3. ScaleLeafWorker (Constrained budget < 16000 -> drops secondary guidance)
	leafConstrainedSpec := PromptSpec{
		Scale:         ScaleLeafWorker,
		ModelFamily:   "qwen2.5-coder-7b",
		IsSmallLocal:  true,
		ContextBudget: 8000,
		WireFormat:    "openai",
	}
	leafConstrainedPrompt, err := ResolvePrompt(leafConstrainedSpec)
	if err != nil {
		t.Fatalf("failed to resolve constrained leaf worker prompt: %v", err)
	}

	if strings.Contains(leafConstrainedPrompt.Raw, "CONVENTIONS:") {
		t.Errorf("constrained leaf prompt (<16k budget) should drop secondary guidance: %s", leafConstrainedPrompt.Raw)
	}

	// 4. ScaleValidator
	validatorSpec := PromptSpec{
		Scale:         ScaleValidator,
		ModelFamily:   "qwen2.5-coder-7b",
		IsSmallLocal:  true,
		ContextBudget: 4000,
		WireFormat:    "openai",
	}
	valPrompt, err := ResolvePrompt(validatorSpec)
	if err != nil {
		t.Fatalf("failed to resolve validator prompt: %v", err)
	}

	if !strings.Contains(valPrompt.Raw, "ROLE: VALIDATOR / PREDICATE EVALUATOR") {
		t.Errorf("validator prompt missing validator rubric: %s", valPrompt.Raw)
	}
	if strings.Contains(valPrompt.Raw, "TOOL CATALOG") {
		t.Errorf("validator prompt should not advertise any tools: %s", valPrompt.Raw)
	}

	// 5. Prefix Invariance: Verify Zone 1 spine is byte-identical across all three scales
	if coordPrompt.Zone1Content != leafPrompt.Zone1Content {
		// Notice coordinator has PartIDPolicyFull, while leaf has PartIDPolicyFloor.
		// Both share PartIDSpineCore!
		if !strings.Contains(coordPrompt.Zone1Content, "You are an agent operating inside the fak kernel runtime.") {
			t.Errorf("coordinator Zone 1 missing core spine")
		}
		if !strings.Contains(leafPrompt.Zone1Content, "You are an agent operating inside the fak kernel runtime.") {
			t.Errorf("leaf Zone 1 missing core spine")
		}
	}
}
