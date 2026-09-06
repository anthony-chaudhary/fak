package agent

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/syspromptmmu"
)

func TestAgentDescriptor_ParseFrontmatter_Standard(t *testing.T) {
	content := `---
name: explore
description: Fast read-only codebase discovery and inspection agent
mode: subagent
model: tier2
variant: high
max_turns: 15
capabilities:
  tools:
    - glob
    - grep
    - read
  paths:
    - internal/**
    - cmd/**
  allow_mutation: false
---
# Explore Persona

You are an expert read-only code exploration agent.
Analyze the codebase thoroughly and report findings.
`
	desc, err := ParseAgentDescriptor([]byte(content), "internal/agent/testdata/explore.md")
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if desc.Name != "explore" {
		t.Errorf("Name = %q, want %q", desc.Name, "explore")
	}
	if desc.Description != "Fast read-only codebase discovery and inspection agent" {
		t.Errorf("Description = %q", desc.Description)
	}
	if desc.Mode != AgentModeSubagent {
		t.Errorf("Mode = %q, want %q", desc.Mode, AgentModeSubagent)
	}
	if desc.Model != ModelTier2 {
		t.Errorf("Model = %q, want %q", desc.Model, ModelTier2)
	}
	if desc.Variant != VariantHigh {
		t.Errorf("Variant = %q, want %q", desc.Variant, VariantHigh)
	}
	if desc.MaxTurns != 15 {
		t.Errorf("MaxTurns = %d, want 15", desc.MaxTurns)
	}
	if desc.Capabilities.AllowMutation != false {
		t.Errorf("AllowMutation = true, want false")
	}

	expectedTools := []string{"glob", "grep", "read"}
	if len(desc.Capabilities.Tools) != len(expectedTools) {
		t.Fatalf("Tools len = %d, want %d", len(desc.Capabilities.Tools), len(expectedTools))
	}
	for i, tool := range expectedTools {
		if desc.Capabilities.Tools[i] != tool {
			t.Errorf("Tools[%d] = %q, want %q", i, desc.Capabilities.Tools[i], tool)
		}
	}

	expectedPaths := []string{"internal/**", "cmd/**"}
	if len(desc.Capabilities.Paths) != len(expectedPaths) {
		t.Fatalf("Paths len = %d, want %d", len(desc.Capabilities.Paths), len(expectedPaths))
	}
	for i, p := range expectedPaths {
		if desc.Capabilities.Paths[i] != p {
			t.Errorf("Paths[%d] = %q, want %q", i, desc.Capabilities.Paths[i], p)
		}
	}

	if !strings.Contains(desc.Prompt, "You are an expert read-only code exploration agent.") {
		t.Errorf("Prompt body missing expected text: %s", desc.Prompt)
	}
}

func TestAgentDescriptor_ParseFrontmatter_InlineAndDefaults(t *testing.T) {
	content := `---
capabilities:
  tools: [glob, grep, read]
  paths: ["internal/**"]
  allow_mutation: true
---
Default agent instructions here.
`
	desc, err := ParseAgentDescriptor([]byte(content), "path/to/custom-agent.md")
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	// Name should default to base filename without extension.
	if desc.Name != "custom-agent" {
		t.Errorf("Name = %q, want %q", desc.Name, "custom-agent")
	}
	// Mode should default to subagent.
	if desc.Mode != AgentModeSubagent {
		t.Errorf("Mode = %q, want %q", desc.Mode, AgentModeSubagent)
	}
	// Model should default to tier1.
	if desc.Model != ModelTier1 {
		t.Errorf("Model = %q, want %q", desc.Model, ModelTier1)
	}
	// Variant should default to default.
	if desc.Variant != VariantDefault {
		t.Errorf("Variant = %q, want %q", desc.Variant, VariantDefault)
	}
	// MaxTurns should default to DefaultDescriptorMaxTurns (10).
	if desc.MaxTurns != DefaultDescriptorMaxTurns {
		t.Errorf("MaxTurns = %d, want %d", desc.MaxTurns, DefaultDescriptorMaxTurns)
	}
	if desc.Capabilities.AllowMutation != true {
		t.Errorf("AllowMutation = false, want true")
	}
	if len(desc.Capabilities.Tools) != 3 || desc.Capabilities.Tools[0] != "glob" {
		t.Errorf("Tools = %+v", desc.Capabilities.Tools)
	}
	if len(desc.Capabilities.Paths) != 1 || desc.Capabilities.Paths[0] != "internal/**" {
		t.Errorf("Paths = %+v", desc.Capabilities.Paths)
	}
}

func TestAgentDescriptor_ParseFrontmatter_FlatCapabilities(t *testing.T) {
	content := `---
name: patcher
mode: primary
model: tier3
variant: adaptive
max-turns: 25
tools:
  - edit
  - write
paths:
  - internal/agent/**
allow-mutation: true
---
Patcher persona instructions.
`
	desc, err := ParseAgentDescriptor([]byte(content), "patcher.md")
	if err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}

	if desc.Name != "patcher" {
		t.Errorf("Name = %q, want %q", desc.Name, "patcher")
	}
	if desc.Mode != AgentModePrimary {
		t.Errorf("Mode = %q, want %q", desc.Mode, AgentModePrimary)
	}
	if desc.Model != ModelTier3 {
		t.Errorf("Model = %q, want %q", desc.Model, ModelTier3)
	}
	if desc.Variant != VariantAdaptive {
		t.Errorf("Variant = %q, want %q", desc.Variant, VariantAdaptive)
	}
	if desc.MaxTurns != 25 {
		t.Errorf("MaxTurns = %d, want 25", desc.MaxTurns)
	}
	if !desc.Capabilities.AllowMutation {
		t.Errorf("AllowMutation = false, want true")
	}
	if len(desc.Capabilities.Tools) != 2 || desc.Capabilities.Tools[0] != "edit" {
		t.Errorf("Tools = %+v", desc.Capabilities.Tools)
	}
	if len(desc.Capabilities.Paths) != 1 || desc.Capabilities.Paths[0] != "internal/agent/**" {
		t.Errorf("Paths = %+v", desc.Capabilities.Paths)
	}
}

func TestAgentDescriptor_ParseFrontmatter_Errors(t *testing.T) {
	_, err := ParseAgentDescriptor([]byte(""), "empty.md")
	if err == nil {
		t.Errorf("expected error on empty content")
	}

	_, err = ParseAgentDescriptor([]byte("no frontmatter here"), "no-fm.md")
	if err == nil {
		t.Errorf("expected error on missing opening ---")
	}

	_, err = ParseAgentDescriptor([]byte("---\nname: test\n"), "unclosed.md")
	if err == nil {
		t.Errorf("expected error on unclosed ---")
	}
}

func TestAgentDescriptor_Discovery(t *testing.T) {
	tmpDir := t.TempDir()

	fakAgentsDir := filepath.Join(tmpDir, ".fak", "agents")
	agentsDir := filepath.Join(tmpDir, ".agents")
	if err := os.MkdirAll(fakAgentsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(agentsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// 1. .fak/agents/explore.md
	exploreContent := `---
name: explore
description: Discovery agent from .fak/agents
mode: subagent
model: tier1
---
Explore from .fak
`
	if err := os.WriteFile(filepath.Join(fakAgentsDir, "explore.md"), []byte(exploreContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// 2. .fak/agents/writer.md
	writerContent := `---
name: writer
description: Writer agent from .fak/agents
mode: subagent
model: tier2
---
Writer instructions
`
	if err := os.WriteFile(filepath.Join(fakAgentsDir, "writer.md"), []byte(writerContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// 3. .agents/reviewer.md
	reviewerContent := `---
name: reviewer
description: Reviewer agent from .agents
mode: subagent
model: tier2
---
Reviewer instructions
`
	if err := os.WriteFile(filepath.Join(agentsDir, "reviewer.md"), []byte(reviewerContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// 4. .agents/explore.md (Duplicate name - .fak/agents should take precedence)
	exploreDupContent := `---
name: explore
description: Duplicate explore agent from .agents
mode: subagent
model: tier3
---
Duplicate explore
`
	if err := os.WriteFile(filepath.Join(agentsDir, "explore.md"), []byte(exploreDupContent), 0o644); err != nil {
		t.Fatal(err)
	}

	// 5. .agents/README.md (Should be skipped)
	if err := os.WriteFile(filepath.Join(agentsDir, "README.md"), []byte("# Agents Readme"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 6. .agents/corrupt.md (Invalid frontmatter, should be skipped gracefully)
	if err := os.WriteFile(filepath.Join(agentsDir, "corrupt.md"), []byte("not frontmatter"), 0o644); err != nil {
		t.Fatal(err)
	}

	discovered, err := DiscoverAgentDescriptors(tmpDir)
	if err != nil {
		t.Fatalf("DiscoverAgentDescriptors failed: %v", err)
	}

	// Verify FindWorkspaceAgentDescriptors alias returns identical count
	discoveredAlias, err := FindWorkspaceAgentDescriptors(tmpDir)
	if err != nil || len(discoveredAlias) != len(discovered) {
		t.Fatalf("FindWorkspaceAgentDescriptors mismatch: err=%v, count=%d", err, len(discoveredAlias))
	}

	if len(discovered) != 3 {
		t.Fatalf("discovered count = %d, want 3", len(discovered))
	}

	// Check sorted order by Name: explore, reviewer, writer
	if discovered[0].Name != "explore" || discovered[1].Name != "reviewer" || discovered[2].Name != "writer" {
		t.Errorf("unexpected discovered order: %s, %s, %s",
			discovered[0].Name, discovered[1].Name, discovered[2].Name)
	}

	// Precedence check: explore should have description from .fak/agents
	if discovered[0].Description != "Discovery agent from .fak/agents" {
		t.Errorf("explore precedence failed: got %q", discovered[0].Description)
	}
}

func TestAgentDescriptor_MonotonicNarrowing(t *testing.T) {
	parent := AgentCapabilities{
		AllowMutation: false,
		Tools:         []string{"glob", "grep", "read"},
		Paths:         []string{"internal/**"},
	}

	// Case 1: Child tries to widen mutation authority
	childWidenMutation := AgentCapabilities{
		AllowMutation: true,
		Tools:         []string{"read"},
		Paths:         []string{"internal/agent/**"},
	}
	if err := ValidateChildCapabilities(parent, childWidenMutation); !errors.Is(err, ErrAuthorityWidened) {
		t.Errorf("expected ErrAuthorityWidened for mutation widening, got: %v", err)
	}
	narrowed := IntersectCapabilities(parent, childWidenMutation)
	if narrowed.AllowMutation != false {
		t.Errorf("IntersectCapabilities should have forced AllowMutation=false")
	}

	// Case 2: Child tries to request tool not in parent
	childWidenTool := AgentCapabilities{
		AllowMutation: false,
		Tools:         []string{"read", "bash"},
		Paths:         []string{"internal/agent/**"},
	}
	if err := ValidateChildCapabilities(parent, childWidenTool); !errors.Is(err, ErrAuthorityWidened) {
		t.Errorf("expected ErrAuthorityWidened for tool widening, got: %v", err)
	}
	narrowed = IntersectCapabilities(parent, childWidenTool)
	if len(narrowed.Tools) != 1 || narrowed.Tools[0] != "read" {
		t.Errorf("narrowed tools = %+v, want [read]", narrowed.Tools)
	}

	// Case 3: Child tries to request path outside parent scope
	childWidenPath := AgentCapabilities{
		AllowMutation: false,
		Tools:         []string{"read"},
		Paths:         []string{"cmd/fak/**"},
	}
	if err := ValidateChildCapabilities(parent, childWidenPath); !errors.Is(err, ErrAuthorityWidened) {
		t.Errorf("expected ErrAuthorityWidened for path widening, got: %v", err)
	}
	narrowed = IntersectCapabilities(parent, childWidenPath)
	if len(narrowed.Paths) != 0 {
		t.Errorf("narrowed paths = %+v, want empty", narrowed.Paths)
	}

	// Case 4: Child is completely within parent authority
	childValid := AgentCapabilities{
		AllowMutation: false,
		Tools:         []string{"glob", "read"},
		Paths:         []string{"internal/agent/**"},
	}
	if err := ValidateChildCapabilities(parent, childValid); err != nil {
		t.Errorf("expected valid child authority, got error: %v", err)
	}
	narrowed = IntersectCapabilities(parent, childValid)
	if len(narrowed.Tools) != 2 || len(narrowed.Paths) != 1 {
		t.Errorf("narrowed capabilities unexpected: %+v", narrowed)
	}

	// Case 5: Descriptor Narrow method with turn budget
	desc := &AgentDescriptor{
		Name:         "sub-worker",
		MaxTurns:     20,
		Capabilities: childWidenTool,
	}
	capped := desc.Narrow(parent, 8)
	if capped.MaxTurns != 8 {
		t.Errorf("capped MaxTurns = %d, want 8", capped.MaxTurns)
	}
	if len(capped.Capabilities.Tools) != 1 || capped.Capabilities.Tools[0] != "read" {
		t.Errorf("capped tools = %+v, want [read]", capped.Capabilities.Tools)
	}
}

func TestAgentDescriptor_PromptMMU_Integration(t *testing.T) {
	desc := &AgentDescriptor{
		Name:        "tester",
		Description: "Verification and testing agent",
		Mode:        AgentModeSubagent,
		Model:       ModelTier1,
		Variant:     VariantDefault,
		MaxTurns:    10,
		Capabilities: AgentCapabilities{
			Tools:         []string{"glob", "read"},
			Paths:         []string{"internal/**"},
			AllowMutation: false,
		},
		Prompt: "Run test suite and report results without editing files.",
	}

	// 1. FormatPrompt
	prompt := desc.FormatPrompt()
	if !strings.Contains(prompt, "# Agent Persona: tester (subagent)") {
		t.Errorf("FormatPrompt missing header: %s", prompt)
	}
	if !strings.Contains(prompt, "Authorized Tools: glob, read") {
		t.Errorf("FormatPrompt missing tools: %s", prompt)
	}
	if !strings.Contains(prompt, "Allow Mutation: false") {
		t.Errorf("FormatPrompt missing mutation flag: %s", prompt)
	}
	if !strings.Contains(prompt, "Run test suite and report results without editing files.") {
		t.Errorf("FormatPrompt missing instructions: %s", prompt)
	}

	// 2. PromptOverlayBytes
	bytes := desc.PromptOverlayBytes()
	if len(bytes) == 0 || string(bytes) != prompt {
		t.Errorf("PromptOverlayBytes mismatch")
	}

	// 3. AsPromptEdit
	edit := desc.AsPromptEdit()
	if edit.Op != syspromptmmu.EditAdd {
		t.Errorf("edit.Op = %v, want EditAdd", edit.Op)
	}
	if edit.Tier != syspromptmmu.TierOverlay {
		t.Errorf("edit.Tier = %v, want TierOverlay", edit.Tier)
	}
	if string(edit.Content) != prompt {
		t.Errorf("edit.Content mismatch")
	}

	// 4. BuildSystemBlock
	// Witness always admits valid overlays.
	witness := func(e syspromptmmu.BaseEdit) bool { return true }
	block := desc.BuildSystemBlock(nil, witness)
	if !block.CacheStable() {
		t.Errorf("block.CacheStable() = false, want true")
	}
	if block.Overlays != 1 {
		t.Errorf("block.Overlays = %d, want 1", block.Overlays)
	}
	if !strings.Contains(string(block.Value), "tester") {
		t.Errorf("block.Value does not contain persona overlay")
	}
}

func TestAgentDescriptor_Registry(t *testing.T) {
	reg := NewAgentDescriptorRegistry()
	if reg.Len() != 0 {
		t.Errorf("initial Len = %d, want 0", reg.Len())
	}

	d1 := &AgentDescriptor{Name: "Alpha", Mode: AgentModePrimary}
	d2 := &AgentDescriptor{Name: "Beta", Mode: AgentModeSubagent}

	reg.Register(d1)
	reg.Register(d2)

	if reg.Len() != 2 {
		t.Errorf("Len = %d, want 2", reg.Len())
	}

	// Case-insensitive lookup
	got, ok := reg.Get("alpha")
	if !ok || got.Name != "Alpha" {
		t.Errorf("Get('alpha') failed: ok=%v, got=%+v", ok, got)
	}

	list := reg.List()
	if len(list) != 2 || list[0].Name != "Alpha" || list[1].Name != "Beta" {
		t.Errorf("List() unexpected: %+v", list)
	}
}

func TestAgentDescriptor_ParseString_And_SyspromptmmuPlan(t *testing.T) {
	content := `---
name: plan-worker
description: Plan execution worker
mode: subagent
model: tier2
variant: adaptive
max_turns: 12
capabilities:
  tools: [read, edit]
  paths: [internal/**]
  allow_mutation: true
---
# Plan Worker Instructions
Execute tasks according to plan.
`
	// Test ParseAgentDescriptor with string directly
	desc, err := ParseAgentDescriptor(content)
	if err != nil {
		t.Fatalf("ParseAgentDescriptor(string) failed: %v", err)
	}
	if desc.Name != "plan-worker" {
		t.Errorf("Name = %q, want %q", desc.Name, "plan-worker")
	}
	if desc.MaxTurns != 12 {
		t.Errorf("MaxTurns = %d, want 12", desc.MaxTurns)
	}
	if !desc.Capabilities.AllowMutation {
		t.Errorf("AllowMutation = false, want true")
	}

	// Test AsOverlaySegment and syspromptmmu.NewOverlaySegment
	overlaySeg := desc.AsOverlaySegment()
	if overlaySeg.Tier != syspromptmmu.TierOverlay {
		t.Errorf("overlaySeg.Tier = %v, want TierOverlay", overlaySeg.Tier)
	}
	if overlaySeg.Tokens <= 0 {
		t.Errorf("overlaySeg.Tokens = %d, want > 0", overlaySeg.Tokens)
	}

	// Test ApplyToPlan
	basePlan := syspromptmmu.BaseContext()
	witness := func(e syspromptmmu.BaseEdit) bool { return true }
	newPlan, verdict := desc.ApplyToPlan(basePlan, witness)
	if !verdict.Applied || verdict.Reason != syspromptmmu.EditOK {
		t.Fatalf("ApplyToPlan failed: applied=%t, reason=%s", verdict.Applied, verdict.Reason)
	}
	if len(newPlan) != len(basePlan)+1 {
		t.Errorf("newPlan length = %d, want %d", len(newPlan), len(basePlan)+1)
	}
	if newPlan[len(newPlan)-1].Tier != syspromptmmu.TierOverlay {
		t.Errorf("appended segment tier = %v, want TierOverlay", newPlan[len(newPlan)-1].Tier)
	}
}
