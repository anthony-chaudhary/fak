package main

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type memoryCapturePlanner struct {
	messages []agent.Message
}

func (p *memoryCapturePlanner) Model() string { return "memory-capture" }

func (p *memoryCapturePlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.messages = append([]agent.Message(nil), messages...)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "acknowledged"}}, nil
}

func buildTestMemoryStore(t *testing.T, storeDir string) {
	t.Helper()
	if err := os.MkdirAll(storeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	memIndex := "# Memory Index\n\n" +
		"- [Fresh Rule](fresh.md) — active project convention\n" +
		"- [Stale Rule](stale.md) — obsolete artifact claim\n" +
		"- [Prose Rule](prose.md) — general orientation\n"
	if err := os.WriteFile(filepath.Join(storeDir, "MEMORY.md"), []byte(memIndex), 0o644); err != nil {
		t.Fatal(err)
	}
	freshNote := "---\nname: fresh-rule\ndescription: active convention\nmetadata:\n  type: feedback\n---\n\nAlways verify memory before injection: cmd/fak/agent_memory.go.\n"
	if err := os.WriteFile(filepath.Join(storeDir, "fresh.md"), []byte(freshNote), 0o644); err != nil {
		t.Fatal(err)
	}
	staleNote := "---\nname: stale-rule\ndescription: gone artifact\nmetadata:\n  type: project\n---\n\nThe old component was at internal/nonexistentpkg/missing.go.\n"
	if err := os.WriteFile(filepath.Join(storeDir, "stale.md"), []byte(staleNote), 0o644); err != nil {
		t.Fatal(err)
	}
	proseNote := "---\nname: prose-rule\ndescription: prose rule\nmetadata:\n  type: user\n---\n\nKeep answers concise and clear.\n"
	if err := os.WriteFile(filepath.Join(storeDir, "prose.md"), []byte(proseNote), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestAgentMemoryFlagsDefaults verifies the --memory and --memory-store flag defaults
// and CLI parsing on both cmd/fak/agent.go and cmd/fak/chat.go.
func TestAgentMemoryFlagsDefaults(t *testing.T) {
	// Agent flag set defaults
	fsAgent, af := newAgentFlagSet()
	if err := fsAgent.Parse([]string{}); err != nil {
		t.Fatalf("Parse agent empty args: %v", err)
	}
	if af.memory == nil || !*af.memory {
		t.Fatal("--memory defaulted to false for agent, want true")
	}
	if af.memoryStore == nil || *af.memoryStore != "" {
		t.Fatalf("--memory-store defaulted to %q for agent, want empty string", *af.memoryStore)
	}

	// Agent flag set explicit overrides
	fsAgent2, af2 := newAgentFlagSet()
	if err := fsAgent2.Parse([]string{"--memory=false", "--memory-store=/custom/store"}); err != nil {
		t.Fatalf("Parse agent explicit args: %v", err)
	}
	if af2.memory == nil || *af2.memory {
		t.Fatal("--memory=false did not disable for agent")
	}
	if af2.memoryStore == nil || *af2.memoryStore != "/custom/store" {
		t.Fatalf("--memory-store was not parsed for agent: %q", *af2.memoryStore)
	}

	// Chat flag set defaults
	fsChat, cf := newChatFlagSet()
	if err := fsChat.Parse([]string{}); err != nil {
		t.Fatalf("Parse chat empty args: %v", err)
	}
	if cf.memory == nil || !*cf.memory {
		t.Fatal("--memory defaulted to false for chat, want true")
	}
	if cf.memoryStore == nil || *cf.memoryStore != "" {
		t.Fatalf("--memory-store defaulted to %q for chat, want empty string", *cf.memoryStore)
	}

	// Chat flag set explicit overrides
	fsChat2, cf2 := newChatFlagSet()
	if err := fsChat2.Parse([]string{"--memory=false", "--memory-store=/custom/chat_store"}); err != nil {
		t.Fatalf("Parse chat explicit args: %v", err)
	}
	if cf2.memory == nil || *cf2.memory {
		t.Fatal("--memory=false did not disable for chat")
	}
	if cf2.memoryStore == nil || *cf2.memoryStore != "/custom/chat_store" {
		t.Fatalf("--memory-store was not parsed for chat: %q", *cf2.memoryStore)
	}
}

// TestResolveAgentMemoryOptionDiscovery verifies auto-discovery of .fak/memory,
// .claude/memory, and MEMORY.md in the workspace root in priority order.
func TestResolveAgentMemoryOptionDiscovery(t *testing.T) {
	// 1. Auto-discover .fak/memory
	wsFak := t.TempDir()
	buildTestMemoryStore(t, filepath.Join(wsFak, ".fak", "memory"))
	optFak, storeFak := resolveAgentMemoryOption(true, "", wsFak)
	if optFak == nil {
		t.Fatal("expected non-nil RunOption for discovered .fak/memory")
	}
	if !strings.Contains(storeFak, filepath.Join(".fak", "memory")) {
		t.Fatalf("discovered store = %q, want containing .fak/memory", storeFak)
	}

	// 2. Auto-discover .claude/memory when .fak/memory is absent
	wsClaude := t.TempDir()
	buildTestMemoryStore(t, filepath.Join(wsClaude, ".claude", "memory"))
	optClaude, storeClaude := resolveAgentMemoryOption(true, "", wsClaude)
	if optClaude == nil {
		t.Fatal("expected non-nil RunOption for discovered .claude/memory")
	}
	if !strings.Contains(storeClaude, filepath.Join(".claude", "memory")) {
		t.Fatalf("discovered store = %q, want containing .claude/memory", storeClaude)
	}

	// 3. Auto-discover root MEMORY.md when directories are absent
	wsRoot := t.TempDir()
	buildTestMemoryStore(t, wsRoot)
	optRoot, storeRoot := resolveAgentMemoryOption(true, "", wsRoot)
	if optRoot == nil {
		t.Fatal("expected non-nil RunOption for discovered root MEMORY.md")
	}
	if !strings.HasSuffix(storeRoot, "MEMORY.md") && storeRoot != wsRoot {
		t.Fatalf("discovered store = %q, want root MEMORY.md or wsRoot", storeRoot)
	}

	// 4. Precedence: .fak/memory > .claude/memory > root MEMORY.md
	wsMulti := t.TempDir()
	buildTestMemoryStore(t, filepath.Join(wsMulti, ".fak", "memory"))
	buildTestMemoryStore(t, filepath.Join(wsMulti, ".claude", "memory"))
	buildTestMemoryStore(t, wsMulti)
	_, storeMulti := resolveAgentMemoryOption(true, "", wsMulti)
	if !strings.Contains(storeMulti, filepath.Join(".fak", "memory")) {
		t.Fatalf("precedence check failed: got %q, want .fak/memory", storeMulti)
	}

	// 5. Absent memory store yields nil option and no injection
	wsEmpty := t.TempDir()
	optEmpty, storeEmpty := resolveAgentMemoryOption(true, "", wsEmpty)
	if optEmpty != nil || storeEmpty != "" {
		t.Fatalf("empty workspace should return nil option and empty store, got opt=%v store=%q", optEmpty, storeEmpty)
	}

	// 6. --memory=false disables even when memory store exists
	optDisabled, _ := resolveAgentMemoryOption(false, "", wsFak)
	if optDisabled != nil {
		t.Fatal("--memory=false must return nil RunOption")
	}
}

// TestResolveAgentMemoryOptionCustomStore verifies explicit --memory-store paths
// (both directory and direct file).
func TestResolveAgentMemoryOptionCustomStore(t *testing.T) {
	ws := t.TempDir()
	customDir := filepath.Join(t.TempDir(), "arbitrary_memory_dir")
	buildTestMemoryStore(t, customDir)

	// Directory path
	optDir, storeDir := resolveAgentMemoryOption(true, customDir, ws)
	if optDir == nil {
		t.Fatal("custom directory store returned nil option")
	}
	if storeDir != customDir {
		t.Fatalf("resolved store = %q, want %q", storeDir, customDir)
	}

	// Direct MEMORY.md file path
	customFile := filepath.Join(customDir, "MEMORY.md")
	optFile, storeFile := resolveAgentMemoryOption(true, customFile, ws)
	if optFile == nil {
		t.Fatal("custom file store returned nil option")
	}
	if storeFile != customFile {
		t.Fatalf("resolved store = %q, want %q", storeFile, customFile)
	}
}

// TestMemoryPromptAssemblyVerifiedFreshOnly tests that verified notes enter prompt
// while stale notes are withheld, and verifies the prompt structure sent to the model.
func TestMemoryPromptAssemblyVerifiedFreshOnly(t *testing.T) {
	ws := t.TempDir()
	buildTestMemoryStore(t, filepath.Join(ws, ".fak", "memory"))

	memOpt, _ := resolveAgentMemoryOption(true, "", ws)
	if memOpt == nil {
		t.Fatal("resolveAgentMemoryOption returned nil")
	}

	planner := &memoryCapturePlanner{}
	_, err := agent.RunArm(context.Background(), planner, "run integration test", true, 1, nil, memOpt)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	if len(planner.messages) < 3 {
		t.Fatalf("expected at least 3 messages (system, memory digest, task), got %d: %+v", len(planner.messages), planner.messages)
	}

	// Message 0: loop's system prompt
	if planner.messages[0].Role != agent.RoleSystem {
		t.Fatalf("messages[0] role = %q, want %q", planner.messages[0].Role, agent.RoleSystem)
	}

	// Message 1: memory digest
	memMsg := planner.messages[1]
	if memMsg.Role != agent.RoleSystem {
		t.Fatalf("messages[1] role = %q, want %q", memMsg.Role, agent.RoleSystem)
	}

	// Fresh note must be rendered in body
	if !strings.Contains(memMsg.Content, "## Fresh Rule (fresh.md)") ||
		!strings.Contains(memMsg.Content, "Always verify memory before injection: cmd/fak/agent_memory.go.") {
		t.Fatalf("fresh note body missing from memory prompt:\n%s", memMsg.Content)
	}

	// Stale note body must NEVER be rendered
	if strings.Contains(memMsg.Content, "The old component was at internal/nonexistentpkg/missing.go.") {
		t.Fatalf("stale note body leaked into memory prompt:\n%s", memMsg.Content)
	}

	// Stale note must be listed under withheld footer with evidence
	if !strings.Contains(memMsg.Content, "withheld (never injected as fact):") ||
		!strings.Contains(memMsg.Content, "stale.md") {
		t.Fatalf("stale note was not withheld with evidence:\n%s", memMsg.Content)
	}

	// Message 2: task
	if planner.messages[2].Role != agent.RoleUser || planner.messages[2].Content != "run integration test" {
		t.Fatalf("messages[2] = %+v, want user task", planner.messages[2])
	}
}

// TestMemoryPromptAssemblyDisabledViaFlag tests that no memory digest is injected
// when memory is disabled.
func TestMemoryPromptAssemblyDisabledViaFlag(t *testing.T) {
	ws := t.TempDir()
	buildTestMemoryStore(t, filepath.Join(ws, ".fak", "memory"))

	memOpt, _ := resolveAgentMemoryOption(false, "", ws)
	if memOpt != nil {
		t.Fatal("expected nil RunOption when memory is false")
	}

	planner := &memoryCapturePlanner{}
	var runOpts []agent.RunOption
	if memOpt != nil {
		runOpts = append(runOpts, memOpt)
	}
	_, err := agent.RunArm(context.Background(), planner, "task with no memory", true, 1, nil, runOpts...)
	if err != nil {
		t.Fatalf("RunArm: %v", err)
	}

	if len(planner.messages) != 2 {
		t.Fatalf("expected exactly 2 messages (system, task), got %d: %+v", len(planner.messages), planner.messages)
	}
	for _, m := range planner.messages {
		if strings.Contains(m.Content, "Fresh Rule") || strings.Contains(m.Content, "Fleet memory") {
			t.Fatalf("unexpected memory content in message: %s", m.Content)
		}
	}
}

// TestChatHeadlessMemoryIntegration verifies end-to-end integration with chat headless execution.
func TestChatHeadlessMemoryIntegration(t *testing.T) {
	ws := t.TempDir()
	buildTestMemoryStore(t, filepath.Join(ws, ".fak", "memory"))

	// 1. With memory enabled
	memOpt, _ := resolveAgentMemoryOption(true, "", ws)
	plannerWithMem := &memoryCapturePlanner{}
	var out bytes.Buffer
	err := runChatHeadless(&out, plannerWithMem, "hello chat", 1, false, "", "", memOpt)
	if err != nil {
		t.Fatalf("runChatHeadless with memory: %v", err)
	}
	if len(plannerWithMem.messages) < 3 {
		t.Fatalf("expected at least 3 messages with memory, got %d", len(plannerWithMem.messages))
	}
	if !strings.Contains(plannerWithMem.messages[1].Content, "Fresh Rule") {
		t.Fatalf("memory message not found in chat prompt:\n%+v", plannerWithMem.messages)
	}

	// 2. With memory disabled
	memOptDisabled, _ := resolveAgentMemoryOption(false, "", ws)
	plannerNoMem := &memoryCapturePlanner{}
	var out2 bytes.Buffer
	var optsNoMem []agent.RunOption
	if memOptDisabled != nil {
		optsNoMem = append(optsNoMem, memOptDisabled)
	}
	err = runChatHeadless(&out2, plannerNoMem, "hello chat no mem", 1, false, "", "", optsNoMem...)
	if err != nil {
		t.Fatalf("runChatHeadless without memory: %v", err)
	}
	if len(plannerNoMem.messages) != 2 {
		t.Fatalf("expected 2 messages without memory, got %d", len(plannerNoMem.messages))
	}
}
