package gateway

import (
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestDeterministicPromptCanonicalizer_FourTierOrdering verifies that disordered prompt sections
// are canonicalized into strict 4-tier ordering:
// Tier 1 (Static System Instructions) -> Tier 2 (Static Tool Schemas) ->
// Tier 3 (Durable Context: repomap, project rules, file tree) -> Tier 4 (Dynamic Turn History).
func TestDeterministicPromptCanonicalizer_FourTierOrdering(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	tools := []responsesTool{
		{Name: "search_files", Type: "function", Description: "Search files"},
	}

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "Can you inspect the project setup?"},                           // Tier 4 (Dynamic History)
		{Role: agent.RoleSystem, Content: "You are an autonomous AI software engineer."},                // Tier 1 (Static System)
		{Role: agent.RoleSystem, Content: "# Workspace Context\nDirectory Structure:\ncmd/\ninternal/"}, // Tier 3 (Durable Context - repo tree)
		{Role: "tools", Content: "Available Tools:\n- search_files"},                                    // Tier 2 (Tool Schemas)
		{Role: agent.RoleAssistant, Content: "I will inspect the workspace."},                           // Tier 4 (Dynamic History)
		{Role: agent.RoleSystem, Content: "# Project Rules\nStrictly adhere to AGENTS.md conventions."}, // Tier 3 (Durable Context - project rules)
	}

	outMsgs, sortedTools, stats := c.CanonicalizePromptMessages(messages, tools)

	if len(outMsgs) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(outMsgs))
	}
	if len(sortedTools) != 1 {
		t.Fatalf("expected 1 sorted tool, got %d", len(sortedTools))
	}

	expectedTiers := []PromptTier{
		Tier1SystemInstructions,
		Tier2ToolDeclarations,
		Tier3ProjectRules,
		Tier4RepoContext,
		Tier5DynamicTurns,
		Tier5DynamicTurns,
	}

	for i, wantTier := range expectedTiers {
		gotTier := c.ClassifyMessage(outMsgs[i])
		if gotTier != wantTier {
			t.Errorf("msg[%d] (role=%s, content=%q): got tier %v, want %v",
				i, outMsgs[i].Role, outMsgs[i].Content, gotTier, wantTier)
		}
	}

	wantPresent := []PromptTier{
		Tier1SystemInstructions,
		Tier2ToolDeclarations,
		Tier3ProjectRules,
		Tier4RepoContext,
		Tier5DynamicTurns,
	}
	if len(stats.TiersPresent) != len(wantPresent) {
		t.Fatalf("stats.TiersPresent = %v, want %v", stats.TiersPresent, wantPresent)
	}
	for i, want := range wantPresent {
		if stats.TiersPresent[i] != want {
			t.Errorf("stats.TiersPresent[%d] = %v, want %v", i, stats.TiersPresent[i], want)
		}
	}

	if stats.ToolsSorted != 1 {
		t.Errorf("stats.ToolsSorted = %d, want 1", stats.ToolsSorted)
	}
	if stats.StablePrefixHash == "" {
		t.Errorf("stats.StablePrefixHash is empty")
	}

	// Verify CanonicalizePromptOrder produces the exact same message ordering
	orderedMsgs := CanonicalizePromptOrder(messages)
	if len(orderedMsgs) != len(outMsgs) {
		t.Fatalf("CanonicalizePromptOrder len = %d, want %d", len(orderedMsgs), len(outMsgs))
	}
	for i := range orderedMsgs {
		if orderedMsgs[i].Role != outMsgs[i].Role || orderedMsgs[i].Content != outMsgs[i].Content {
			t.Errorf("CanonicalizePromptOrder msg[%d] mismatch: got (role=%s, content=%q), want (role=%s, content=%q)",
				i, orderedMsgs[i].Role, orderedMsgs[i].Content, outMsgs[i].Role, outMsgs[i].Content)
		}
	}
}

// TestDeterministicPromptCanonicalizer_ToolSorting verifies tools in reverse or random order
// are deterministically sorted alphabetically by name for both responsesTool and agent.ToolDef.
func TestDeterministicPromptCanonicalizer_ToolSorting(t *testing.T) {
	c := DeterministicPromptCanonicalizer{}

	tools := []responsesTool{
		{Name: "zebra_tool", Description: "Zebra"},
		{Name: "alpha_tool", Description: "Alpha"},
		{Name: "mike_tool", Description: "Mike"},
		{Name: "bravo_tool", Description: "Bravo"},
	}

	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "System prompt"},
	}

	_, sortedTools, stats := c.CanonicalizePromptMessages(messages, tools)

	if len(sortedTools) != 4 {
		t.Fatalf("expected 4 sorted tools, got %d", len(sortedTools))
	}
	wantOrder := []string{"alpha_tool", "bravo_tool", "mike_tool", "zebra_tool"}
	for i, want := range wantOrder {
		if sortedTools[i].Name != want {
			t.Errorf("tool[%d] = %q, want %q", i, sortedTools[i].Name, want)
		}
	}
	if stats.ToolsSorted != 4 {
		t.Errorf("stats.ToolsSorted = %d, want 4", stats.ToolsSorted)
	}

	// Test CanonicalizeToolDefs for agent.ToolDef
	agentTools := []agent.ToolDef{
		{Function: agent.ToolDefFunction{Name: "zebra_tool"}},
		{Function: agent.ToolDefFunction{Name: "alpha_tool"}},
		{Function: agent.ToolDefFunction{Name: "mike_tool"}},
		{Function: agent.ToolDefFunction{Name: "bravo_tool"}},
	}
	sortedAgentTools := CanonicalizeToolDefs(agentTools)
	for i, want := range wantOrder {
		if sortedAgentTools[i].Function.Name != want {
			t.Errorf("agentTool[%d] = %q, want %q", i, sortedAgentTools[i].Function.Name, want)
		}
	}

	// Test CanonicalizePromptMessagesWithToolDefs
	_, sortedWithDefs, defStats := c.CanonicalizePromptMessagesWithToolDefs(messages, agentTools)
	if defStats.ToolsSorted != 4 {
		t.Errorf("defStats.ToolsSorted = %d, want 4", defStats.ToolsSorted)
	}
	for i, want := range wantOrder {
		if sortedWithDefs[i].Function.Name != want {
			t.Errorf("sortedWithDefs[%d] = %q, want %q", i, sortedWithDefs[i].Function.Name, want)
		}
	}
}

// TestDeterministicPromptCanonicalizer_CacheBreakpointPrefixBoundary verifies that an explicit
// provider cache breakpoint (cache_control: {"type": "ephemeral"}) is set at the boundary of prefix tiers.
func TestDeterministicPromptCanonicalizer_CacheBreakpointPrefixBoundary(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "Hello assistant!"},                                                 // Tier 5
		{Role: agent.RoleSystem, Content: "You are an AI assistant."},                                       // Tier 1
		{Role: agent.RoleSystem, Content: "# Durable Context\n<repomap>\nmain.go\n</repomap>"},              // Tier 4 (1st)
		{Role: "tools", Content: "Available Tools:\n- tool_a"},                                              // Tier 2
		{Role: agent.RoleSystem, Content: "# Memory Cards\n- User prefers Go code.\n- Follow house style."}, // Tier 4 (2nd - boundary)
		{Role: agent.RoleAssistant, Content: "Hello! How can I help you today?"},                            // Tier 5
	}

	outMsgs, _, stats := c.CanonicalizePromptMessages(messages, nil)

	if len(outMsgs) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(outMsgs))
	}

	// Message index 3 should be the last message of Tier 4 (boundary of prefix)
	tier4BoundaryIdx := 3
	if stats.CacheBreakpointIndex != tier4BoundaryIdx {
		t.Errorf("stats.CacheBreakpointIndex = %d, want %d", stats.CacheBreakpointIndex, tier4BoundaryIdx)
	}
	if stats.CacheBreakpointTier != Tier4RepoContext {
		t.Errorf("stats.CacheBreakpointTier = %v, want %v", stats.CacheBreakpointTier, Tier4RepoContext)
	}

	// Verify only the boundary message carries the cache_control breakpoint
	for i, msg := range outMsgs {
		if i == tier4BoundaryIdx {
			if msg.CacheControl == nil {
				t.Fatalf("msg[%d] should have CacheControl, got nil", i)
			}
			if msg.CacheControl.Type != "ephemeral" {
				t.Errorf("msg[%d].CacheControl.Type = %q, want %q", i, msg.CacheControl.Type, "ephemeral")
			}
		} else {
			if msg.CacheControl != nil {
				t.Errorf("msg[%d] should NOT have CacheControl, got %+v", i, msg.CacheControl)
			}
		}
	}

	// Fallback test: when Tier 4 is absent, breakpoint lands on Tier 3
	noTier4Msgs := []agent.Message{
		{Role: agent.RoleUser, Content: "User query"},
		{Role: agent.RoleSystem, Content: "Base system instructions"},
		{Role: "tools", Content: "Tools declarations"},
		{Role: agent.RoleSystem, Content: "# Project Rules\nRule 1"},
	}
	outNoTier4, _, statsNoTier4 := c.CanonicalizePromptMessages(noTier4Msgs, nil)
	if statsNoTier4.CacheBreakpointIndex != 2 {
		t.Errorf("fallback to Tier 3: stats.CacheBreakpointIndex = %d, want 2", statsNoTier4.CacheBreakpointIndex)
	}
	if statsNoTier4.CacheBreakpointTier != Tier3ProjectRules {
		t.Errorf("fallback to Tier 3: stats.CacheBreakpointTier = %v, want %v", statsNoTier4.CacheBreakpointTier, Tier3ProjectRules)
	}
	if outNoTier4[2].CacheControl == nil || outNoTier4[2].CacheControl.Type != "ephemeral" {
		t.Errorf("fallback to Tier 3: outNoTier4[2].CacheControl = %+v, want ephemeral", outNoTier4[2].CacheControl)
	}

	// Fallback test: when Tier 4 and Tier 3 are absent, breakpoint lands on Tier 2
	noTier3Msgs := []agent.Message{
		{Role: agent.RoleUser, Content: "User query"},
		{Role: agent.RoleSystem, Content: "Base system instructions"},
		{Role: "tools", Content: "Tools declarations"},
	}
	outNoTier3, _, statsNoTier3 := c.CanonicalizePromptMessages(noTier3Msgs, nil)
	if statsNoTier3.CacheBreakpointIndex != 1 {
		t.Errorf("fallback to Tier 2: stats.CacheBreakpointIndex = %d, want 1", statsNoTier3.CacheBreakpointIndex)
	}
	if statsNoTier3.CacheBreakpointTier != Tier2ToolDeclarations {
		t.Errorf("fallback to Tier 2: stats.CacheBreakpointTier = %v, want %v", statsNoTier3.CacheBreakpointTier, Tier2ToolDeclarations)
	}
	if outNoTier3[1].CacheControl == nil || outNoTier3[1].CacheControl.Type != "ephemeral" {
		t.Errorf("fallback to Tier 2: outNoTier3[1].CacheControl = %+v, want ephemeral", outNoTier3[1].CacheControl)
	}

	// Fallback test: when Tier 4, Tier 3 and Tier 2 are absent, breakpoint lands on Tier 1
	tier1OnlyMsgs := []agent.Message{
		{Role: agent.RoleUser, Content: "User query"},
		{Role: agent.RoleSystem, Content: "Base system instructions"},
	}
	outTier1Only, _, statsTier1Only := c.CanonicalizePromptMessages(tier1OnlyMsgs, nil)
	if statsTier1Only.CacheBreakpointIndex != 0 {
		t.Errorf("fallback to Tier 1: stats.CacheBreakpointIndex = %d, want 0", statsTier1Only.CacheBreakpointIndex)
	}
	if statsTier1Only.CacheBreakpointTier != Tier1SystemInstructions {
		t.Errorf("fallback to Tier 1: stats.CacheBreakpointTier = %v, want %v", statsTier1Only.CacheBreakpointTier, Tier1SystemInstructions)
	}
	if outTier1Only[0].CacheControl == nil || outTier1Only[0].CacheControl.Type != "ephemeral" {
		t.Errorf("fallback to Tier 1: outTier1Only[0].CacheControl = %+v, want ephemeral", outTier1Only[0].CacheControl)
	}
}

// TestDeterministicPromptCanonicalizer_Idempotence verifies that executing canonicalization twice
// on the same input produces byte-for-byte and structure-for-structure identical output.
func TestDeterministicPromptCanonicalizer_Idempotence(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	tools := []responsesTool{
		{Name: "read_file", Description: "Read"},
		{Name: "write_file", Description: "Write"},
		{Name: "execute_cmd", Description: "Execute"},
	}

	inputMessages := []agent.Message{
		{Role: agent.RoleUser, Content: "Please review my PR."},
		{
			Role: agent.RoleSystem,
			Content: "You are a code reviewer.\n" +
				"Today's date is Monday, Sep 07, 2026.\n" +
				"Session ID: sess-9999-aaaa",
		},
		{Role: agent.RoleSystem, Content: "# Project Rules\n1. Run tests before landing.\n2. Keep commits atomic."},
		{Role: "tools", Content: "Available Tools: read_file, write_file, execute_cmd"},
		{Role: agent.RoleAssistant, Content: "I will review your code changes."},
	}

	// First pass
	out1, tools1, stats1 := c.CanonicalizePromptMessages(inputMessages, tools)

	// Second pass on the result of the first pass
	out2, tools2, stats2 := c.CanonicalizePromptMessages(out1, tools1)

	if len(out1) != len(out2) {
		t.Fatalf("idempotence failed: out1 len = %d, out2 len = %d", len(out1), len(out2))
	}

	for i := range out1 {
		if out1[i].Role != out2[i].Role {
			t.Errorf("msg[%d].Role mismatch: out1=%q, out2=%q", i, out1[i].Role, out2[i].Role)
		}
		if out1[i].Content != out2[i].Content {
			t.Errorf("msg[%d].Content mismatch:\nout1=%q\nout2=%q", i, out1[i].Content, out2[i].Content)
		}
		if !reflect.DeepEqual(out1[i].CacheControl, out2[i].CacheControl) {
			t.Errorf("msg[%d].CacheControl mismatch: out1=%+v, out2=%+v", i, out1[i].CacheControl, out2[i].CacheControl)
		}
	}

	if !reflect.DeepEqual(tools1, tools2) {
		t.Errorf("tools mismatch across passes:\ntools1=%+v\ntools2=%+v", tools1, tools2)
	}

	if stats1.StablePrefixHash != stats2.StablePrefixHash {
		t.Errorf("StablePrefixHash mismatch across passes:\nstats1=%q\nstats2=%q", stats1.StablePrefixHash, stats2.StablePrefixHash)
	}
	if stats1.CacheBreakpointIndex != stats2.CacheBreakpointIndex {
		t.Errorf("CacheBreakpointIndex mismatch across passes: %d vs %d", stats1.CacheBreakpointIndex, stats2.CacheBreakpointIndex)
	}
	if stats2.VolatileElementsHoisted != 0 {
		t.Errorf("pass 2 hoisted %d elements, want 0 (already hoisted)", stats2.VolatileElementsHoisted)
	}

	// Also verify CanonicalizePromptOrder idempotence
	order1 := CanonicalizePromptOrder(inputMessages)
	order2 := CanonicalizePromptOrder(order1)
	if len(order1) != len(order2) {
		t.Fatalf("CanonicalizePromptOrder idempotence len mismatch: %d vs %d", len(order1), len(order2))
	}
	for i := range order1 {
		if order1[i].Role != order2[i].Role || order1[i].Content != order2[i].Content {
			t.Errorf("CanonicalizePromptOrder msg[%d] mismatch:\norder1=%+v\norder2=%+v", i, order1[i], order2[i])
		}
	}
}

// TestDeterministicPromptCanonicalizer_ConcurrentRace tests concurrent execution of canonicalization
// across multiple goroutines to verify race freedom under go test -race.
func TestDeterministicPromptCanonicalizer_ConcurrentRace(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	tools := []responsesTool{
		{Name: "grep_search", Description: "Search regex"},
		{Name: "read_file", Description: "Read contents"},
		{Name: "bash_exec", Description: "Execute shell"},
	}

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "Can you fix the bug?"},
		{
			Role: agent.RoleSystem,
			Content: "You are an autonomous engineering agent.\n" +
				"Today's date is Monday, Sep 07, 2026.\n" +
				"Session ID: sess-race-1234",
		},
		{Role: agent.RoleSystem, Content: "# Durable Context\n<repomap>\ncmd/fak/\ninternal/gateway/\n</repomap>"},
		{Role: "tools", Content: "Tools: grep_search, read_file, bash_exec"},
		{Role: agent.RoleAssistant, Content: "I am investigating the issue."},
	}

	const goroutines = 64
	var wg sync.WaitGroup
	wg.Add(goroutines)

	expectedHash := c.ComputeStablePrefixHash(messages, tools)

	for i := 0; i < goroutines; i++ {
		go func(iter int) {
			defer wg.Done()

			// Run CanonicalizePromptMessages
			outMsgs, sortedTools, stats := c.CanonicalizePromptMessages(messages, tools)
			if len(outMsgs) != 5 {
				t.Errorf("goroutine %d: outMsgs len = %d, want 5", iter, len(outMsgs))
			}
			if len(sortedTools) != 3 {
				t.Errorf("goroutine %d: sortedTools len = %d, want 3", iter, len(sortedTools))
			}
			if stats.StablePrefixHash != expectedHash {
				t.Errorf("goroutine %d: hash = %q, want %q", iter, stats.StablePrefixHash, expectedHash)
			}
			if stats.CacheBreakpointIndex != 2 {
				t.Errorf("goroutine %d: breakpoint index = %d, want 2", iter, stats.CacheBreakpointIndex)
			}

			// Run CanonicalizePromptOrder
			ordered := CanonicalizePromptOrder(messages)
			if len(ordered) != 5 {
				t.Errorf("goroutine %d: ordered len = %d, want 5", iter, len(ordered))
			}
		}(i)
	}

	wg.Wait()
}

// TestDeterministicPromptCanonicalizer_StablePrefixHash verifies that two sessions with different
// dynamic conversation turns (Tier 4) or varying timestamp strings produce identical StablePrefixHash for Tiers 1-3.
func TestDeterministicPromptCanonicalizer_StablePrefixHash(t *testing.T) {
	c := DeterministicPromptCanonicalizer{
		StripTimestamps: true,
		StripSessionIDs: true,
	}

	tools := []responsesTool{
		{
			Name:        "edit_file",
			Type:        "function",
			Description: "Edit existing file",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
		{
			Name:        "read_file",
			Type:        "function",
			Description: "Read file contents",
			Parameters:  json.RawMessage(`{"type":"object"}`),
		},
	}

	// Session A: has date A, session ID A, dynamic turn A
	sessionA := []agent.Message{
		{
			Role: agent.RoleSystem,
			Content: "You are a senior systems engineer.\n" +
				"Today's date is Monday, Sep 07, 2026.\n" +
				"Session ID: sess-uuid-aaa-1111",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Project Rules\nAlways run tests before committing code. AGENTS.md rules apply.",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Workspace Context\nDirectory Structure:\ncmd/\n  fak/\ninternal/\n  gateway/",
		},
		{
			Role:    agent.RoleUser,
			Content: "How do I implement prompt prefix caching in Go?",
		},
		{
			Role:    agent.RoleAssistant,
			Content: "Prompt prefix caching relies on stabilizing invariant tokens at the start of context.",
		},
	}

	// Session B: has date B, session ID B, completely different dynamic turns in Tier 4
	// and tools in reverse order
	toolsReverse := []responsesTool{tools[1], tools[0]}
	sessionB := []agent.Message{
		{
			Role: agent.RoleSystem,
			Content: "You are a senior systems engineer.\n" +
				"Today's date is Tuesday, Oct 14, 2026.\n" +
				"Session ID: sess-uuid-bbb-9999",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Project Rules\nAlways run tests before committing code. AGENTS.md rules apply.",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Workspace Context\nDirectory Structure:\ncmd/\n  fak/\ninternal/\n  gateway/",
		},
		{
			Role:    agent.RoleUser,
			Content: "What is the capital of Japan?",
		},
		{
			Role:    agent.RoleAssistant,
			Content: "The capital of Japan is Tokyo.",
		},
		{
			Role:    agent.RoleUser,
			Content: "Tell me about Mount Fuji.",
		},
		{
			Role:    agent.RoleAssistant,
			Content: "Mount Fuji is the tallest mountain in Japan.",
		},
	}

	hashA := c.ComputeStablePrefixHash(sessionA, tools)
	hashB := c.ComputeStablePrefixHash(sessionB, toolsReverse)

	if hashA == "" {
		t.Fatalf("hashA is empty")
	}
	if hashA != hashB {
		t.Fatalf("StablePrefixHash mismatch:\nhashA = %s\nhashB = %s", hashA, hashB)
	}

	// Verify through CanonicalizePromptMessages stats as well
	outA, _, statsA := c.CanonicalizePromptMessages(sessionA, tools)
	outB, _, statsB := c.CanonicalizePromptMessages(sessionB, toolsReverse)

	if statsA.StablePrefixHash != statsB.StablePrefixHash {
		t.Fatalf("stats.StablePrefixHash mismatch:\nstatsA = %s\nstatsB = %s", statsA.StablePrefixHash, statsB.StablePrefixHash)
	}
	if statsA.StablePrefixHash != hashA {
		t.Fatalf("statsA.StablePrefixHash = %s, want %s", statsA.StablePrefixHash, hashA)
	}

	if statsA.VolatileElementsHoisted == 0 {
		t.Errorf("statsA.VolatileElementsHoisted = 0, want > 0")
	}
	if statsB.VolatileElementsHoisted == 0 {
		t.Errorf("statsB.VolatileElementsHoisted = 0, want > 0")
	}

	// Verify volatile elements were hoisted into Tier 4 (user message)
	var foundVolatileA, foundVolatileB bool
	for _, m := range outA {
		if c.ClassifyMessage(m) == Tier4DynamicHistory && strings.Contains(m.Content, "Today's date is") {
			foundVolatileA = true
			break
		}
	}
	for _, m := range outB {
		if c.ClassifyMessage(m) == Tier4DynamicHistory && strings.Contains(m.Content, "Today's date is") {
			foundVolatileB = true
			break
		}
	}
	if !foundVolatileA {
		t.Errorf("volatile timestamp not found in Tier 4 for session A")
	}
	if !foundVolatileB {
		t.Errorf("volatile timestamp not found in Tier 4 for session B")
	}
}

// TestDeterministicPromptCanonicalizer_ConversationPreservation verifies that user/assistant
// conversation turn order in Tier 4 is strictly preserved even when jumbled with prefix tiers.
func TestDeterministicPromptCanonicalizer_ConversationPreservation(t *testing.T) {
	c := DeterministicPromptCanonicalizer{}

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "Turn 1: User prompt"},
		{Role: agent.RoleAssistant, Content: "Turn 1: Assistant reply"},
		{Role: agent.RoleSystem, Content: "# Project Rules\nRule A: Be concise."}, // Tier 3
		{Role: agent.RoleUser, Content: "Turn 2: User follow-up"},
		{Role: agent.RoleAssistant, Content: "Turn 2: Assistant follow-up"},
		{Role: agent.RoleSystem, Content: "Base system prompt: You are helpful."}, // Tier 1
		{Role: agent.RoleUser, Content: "Turn 3: User conclusion"},
	}

	outMsgs, _, _ := c.CanonicalizePromptMessages(messages, nil)

	var tier4Contents []string
	for _, m := range outMsgs {
		if c.ClassifyMessage(m) == Tier4DynamicHistory {
			tier4Contents = append(tier4Contents, m.Content)
		}
	}

	wantTier4 := []string{
		"Turn 1: User prompt",
		"Turn 1: Assistant reply",
		"Turn 2: User follow-up",
		"Turn 2: Assistant follow-up",
		"Turn 3: User conclusion",
	}

	if len(tier4Contents) != len(wantTier4) {
		t.Fatalf("tier 4 message count = %d, want %d", len(tier4Contents), len(wantTier4))
	}
	for i, want := range wantTier4 {
		if tier4Contents[i] != want {
			t.Errorf("tier4[%d] = %q, want %q", i, tier4Contents[i], want)
		}
	}
}

// TestDeterministicPromptCanonicalizer_EnforceBlockAlignment verifies that 1024-token
// boundary padding ensures PrefixTokensEstimate is an exact multiple of 1024.
func TestDeterministicPromptCanonicalizer_EnforceBlockAlignment(t *testing.T) {
	cAligned := DeterministicPromptCanonicalizer{
		EnforceBlockAlignment: true,
	}
	cUnaligned := DeterministicPromptCanonicalizer{
		EnforceBlockAlignment: false,
	}

	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "Short system prompt that is well under 1024 tokens."},
	}

	_, _, statsUnaligned := cUnaligned.CanonicalizePromptMessages(messages, nil)
	if statsUnaligned.PrefixTokensEstimate%1024 == 0 {
		t.Fatalf("unaligned prefix unexpectedly aligned to 1024 tokens: %d", statsUnaligned.PrefixTokensEstimate)
	}

	outAligned, _, statsAligned := cAligned.CanonicalizePromptMessages(messages, nil)
	if statsAligned.PrefixTokensEstimate%1024 != 0 {
		t.Fatalf("aligned prefix not a multiple of 1024 tokens: %d", statsAligned.PrefixTokensEstimate)
	}
	if statsAligned.PrefixTokensEstimate < 1024 {
		t.Fatalf("aligned prefix tokens < 1024: %d", statsAligned.PrefixTokensEstimate)
	}
	// Verify that padding was added to the prefix message
	if len(outAligned[0].Content) <= len(messages[0].Content) {
		t.Fatalf("padding was not appended to prefix message: len got %d, want > %d",
			len(outAligned[0].Content), len(messages[0].Content))
	}
}

// TestPromptOrdering_Tiers1Through5 verifies that messages arriving out of order are sorted into canonical Tiers 1-5:
// Tier 1 (Static System Instructions) -> Tier 2 (Static Tool Declarations) ->
// Tier 3 (Static Project Rules) -> Tier 4 (Static Repo Context) -> Tier 5 (Ephemeral Dynamic Turns).
func TestPromptOrdering_Tiers1Through5(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	tools := []responsesTool{
		{Name: "git_diff", Type: "function", Description: "Show git diff"},
	}

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "Can you analyze the failure in CI?"},                           // Tier 5
		{Role: agent.RoleSystem, Content: "# Workspace Context\nDirectory Structure:\ncmd/\ninternal/"}, // Tier 4
		{Role: agent.RoleSystem, Content: "You are an autonomous systems engineering agent."},           // Tier 1
		{Role: agent.RoleAssistant, Content: "Checking the CI logs now."},                               // Tier 5
		{Role: "tools", Content: "Available Tools:\n- git_diff"},                                        // Tier 2
		{Role: agent.RoleSystem, Content: "# Project Rules\nStrictly adhere to AGENTS.md conventions."}, // Tier 3
	}

	outMsgs, sortedTools, stats := c.CanonicalizePromptMessages(messages, tools)

	if len(outMsgs) != 6 {
		t.Fatalf("expected 6 messages, got %d", len(outMsgs))
	}
	if len(sortedTools) != 1 {
		t.Fatalf("expected 1 sorted tool, got %d", len(sortedTools))
	}

	expectedTiers := []PromptTier{
		Tier1SystemInstructions,
		Tier2ToolDeclarations,
		Tier3ProjectRules,
		Tier4RepoContext,
		Tier5DynamicTurns,
		Tier5DynamicTurns,
	}

	for i, wantTier := range expectedTiers {
		gotTier := c.ClassifyMessage(outMsgs[i])
		if gotTier != wantTier {
			t.Errorf("msg[%d] (role=%s, content=%q): got tier %v (%d), want %v (%d)",
				i, outMsgs[i].Role, outMsgs[i].Content, gotTier, gotTier, wantTier, wantTier)
		}
	}

	wantPresent := []PromptTier{
		Tier1SystemInstructions,
		Tier2ToolDeclarations,
		Tier3ProjectRules,
		Tier4RepoContext,
		Tier5DynamicTurns,
	}
	if len(stats.TiersPresent) != len(wantPresent) {
		t.Fatalf("stats.TiersPresent = %v, want %v", stats.TiersPresent, wantPresent)
	}
	for i, want := range wantPresent {
		if stats.TiersPresent[i] != want {
			t.Errorf("stats.TiersPresent[%d] = %v, want %v", i, stats.TiersPresent[i], want)
		}
	}

	// CanonicalizePromptOrder produces the exact same message ordering
	orderedMsgs := CanonicalizePromptOrder(messages)
	if len(orderedMsgs) != len(outMsgs) {
		t.Fatalf("CanonicalizePromptOrder len = %d, want %d", len(orderedMsgs), len(outMsgs))
	}
	for i := range orderedMsgs {
		if orderedMsgs[i].Role != outMsgs[i].Role || orderedMsgs[i].Content != outMsgs[i].Content {
			t.Errorf("CanonicalizePromptOrder msg[%d] mismatch: got (role=%s, content=%q), want (role=%s, content=%q)",
				i, orderedMsgs[i].Role, orderedMsgs[i].Content, outMsgs[i].Role, outMsgs[i].Content)
		}
	}
}

// TestPromptOrdering_PreservesConversationChronology verifies that conversation turns within
// Tier 5 preserve their exact chronological sequence even when interleaved with prefix tiers.
func TestPromptOrdering_PreservesConversationChronology(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	messages := []agent.Message{
		{Role: agent.RoleUser, Content: "Turn 1: User prompt"},
		{Role: agent.RoleAssistant, Content: "Turn 1: Assistant reply"},
		{Role: agent.RoleSystem, Content: "# Project Rules\nRule A: Be concise."}, // Tier 3
		{Role: agent.RoleUser, Content: "Turn 2: User follow-up"},
		{Role: agent.RoleAssistant, Content: "Turn 2: Assistant follow-up"},
		{Role: agent.RoleSystem, Content: "You are an assistant."},                                      // Tier 1
		{Role: agent.RoleSystem, Content: "# Workspace Context\nDirectory Structure:\ncmd/\ninternal/"}, // Tier 4
		{Role: agent.RoleUser, Content: "Turn 3: Run command"},
		{Role: "tool", Content: "Turn 3: tool result OK", ToolCallID: "call_123"}, // Tier 5 tool output
		{Role: agent.RoleAssistant, Content: "Turn 3: Assistant conclusion"},
		{Role: agent.RoleUser, Content: "Turn 4: Final user acknowledgment"},
	}

	outMsgs, _, _ := c.CanonicalizePromptMessages(messages, nil)

	var tier5Msgs []agent.Message
	for _, m := range outMsgs {
		if c.ClassifyMessage(m) == Tier5DynamicTurns {
			tier5Msgs = append(tier5Msgs, m)
		}
	}

	wantTurns := []struct {
		role    string
		content string
	}{
		{role: agent.RoleUser, content: "Turn 1: User prompt"},
		{role: agent.RoleAssistant, content: "Turn 1: Assistant reply"},
		{role: agent.RoleUser, content: "Turn 2: User follow-up"},
		{role: agent.RoleAssistant, content: "Turn 2: Assistant follow-up"},
		{role: agent.RoleUser, content: "Turn 3: Run command"},
		{role: "tool", content: "Turn 3: tool result OK"},
		{role: agent.RoleAssistant, content: "Turn 3: Assistant conclusion"},
		{role: agent.RoleUser, content: "Turn 4: Final user acknowledgment"},
	}

	if len(tier5Msgs) != len(wantTurns) {
		t.Fatalf("tier 5 message count = %d, want %d", len(tier5Msgs), len(wantTurns))
	}

	for i, want := range wantTurns {
		if tier5Msgs[i].Role != want.role {
			t.Errorf("tier5[%d].Role = %q, want %q", i, tier5Msgs[i].Role, want.role)
		}
		if tier5Msgs[i].Content != want.content {
			t.Errorf("tier5[%d].Content = %q, want %q", i, tier5Msgs[i].Content, want.content)
		}
	}
}

// TestPromptOrdering_PrefixHashStability verifies that two sessions with different dynamic conversation
// turns produce the exact same StablePrefixHash for Tiers 1-4.
func TestPromptOrdering_PrefixHashStability(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	toolsA := []responsesTool{
		{Name: "fetch_data", Type: "function", Description: "Fetch data"},
		{Name: "compute_metric", Type: "function", Description: "Compute metric"},
	}

	// Tools in reverse order for Session B
	toolsB := []responsesTool{
		toolsA[1],
		toolsA[0],
	}

	sharedStaticTiers := []agent.Message{
		{Role: agent.RoleSystem, Content: "You are a performance optimization expert."},
		{Role: "tools", Content: "Available Tools:\n- compute_metric\n- fetch_data"},
		{Role: agent.RoleSystem, Content: "# Project Rules\n1. Deterministic output.\n2. No allocations in hot path."},
		{Role: agent.RoleSystem, Content: "# Workspace Context\nDirectory Structure:\ncmd/\ninternal/engine/"},
	}

	// Session 1 has dynamic turns about caching
	session1 := append([]agent.Message{}, sharedStaticTiers...)
	session1 = append(session1,
		agent.Message{Role: agent.RoleUser, Content: "How do prefix cache hits work?"},
		agent.Message{Role: agent.RoleAssistant, Content: "They reuse the KV-cache of invariant prefix tokens."},
	)

	// Session 2 has dynamic turns about kernel scheduling
	session2 := append([]agent.Message{}, sharedStaticTiers...)
	session2 = append(session2,
		agent.Message{Role: agent.RoleUser, Content: "Explain vDSO system calls."},
		agent.Message{Role: agent.RoleAssistant, Content: "vDSO provides fast user-space execution of kernel routines."},
		agent.Message{Role: agent.RoleUser, Content: "Which calls use vDSO?"},
		agent.Message{Role: agent.RoleAssistant, Content: "gettimeofday and clock_gettime."},
	)

	hash1 := c.ComputeStablePrefixHash(session1, toolsA)
	hash2 := c.ComputeStablePrefixHash(session2, toolsB)

	if hash1 == "" {
		t.Fatalf("hash1 is empty")
	}
	if hash1 != hash2 {
		t.Fatalf("StablePrefixHash mismatch across sessions:\nhash1 = %s\nhash2 = %s", hash1, hash2)
	}

	_, _, stats1 := c.CanonicalizePromptMessages(session1, toolsA)
	_, _, stats2 := c.CanonicalizePromptMessages(session2, toolsB)

	if stats1.StablePrefixHash != hash1 {
		t.Fatalf("stats1.StablePrefixHash = %s, want %s", stats1.StablePrefixHash, hash1)
	}
	if stats2.StablePrefixHash != hash2 {
		t.Fatalf("stats2.StablePrefixHash = %s, want %s", stats2.StablePrefixHash, hash2)
	}
}

// TestPromptOrdering_ToolAlphabeticalCanonicalization verifies that tools declared in arbitrary order
// are sorted canonically by name.
func TestPromptOrdering_ToolAlphabeticalCanonicalization(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	tools := []responsesTool{
		{Name: "zebra_tool", Description: "Zebra"},
		{Name: "beta_tool", Description: "Beta"},
		{Name: "alpha_tool", Description: "Alpha"},
		{Name: "omega_tool", Description: "Omega"},
		{Name: "delta_tool", Description: "Delta"},
	}

	wantOrder := []string{"alpha_tool", "beta_tool", "delta_tool", "omega_tool", "zebra_tool"}

	// 1. Direct CanonicalizeTools
	sorted := CanonicalizeTools(tools)
	if len(sorted) != len(wantOrder) {
		t.Fatalf("sorted tools len = %d, want %d", len(sorted), len(wantOrder))
	}
	for i, want := range wantOrder {
		if sorted[i].Name != want {
			t.Errorf("sorted[%d] = %q, want %q", i, sorted[i].Name, want)
		}
	}

	// 2. CanonicalizePromptMessages
	msgs := []agent.Message{
		{Role: agent.RoleSystem, Content: "You are an assistant."},
	}
	_, sortedMsgsTools, stats := c.CanonicalizePromptMessages(msgs, tools)
	if stats.ToolsSorted != len(wantOrder) {
		t.Errorf("stats.ToolsSorted = %d, want %d", stats.ToolsSorted, len(wantOrder))
	}
	for i, want := range wantOrder {
		if sortedMsgsTools[i].Name != want {
			t.Errorf("sortedMsgsTools[%d] = %q, want %q", i, sortedMsgsTools[i].Name, want)
		}
	}

	// 3. ToolDef canonicalization
	toolDefs := []agent.ToolDef{
		{Function: agent.ToolDefFunction{Name: "zebra_tool"}},
		{Function: agent.ToolDefFunction{Name: "beta_tool"}},
		{Function: agent.ToolDefFunction{Name: "alpha_tool"}},
		{Function: agent.ToolDefFunction{Name: "omega_tool"}},
		{Function: agent.ToolDefFunction{Name: "delta_tool"}},
	}
	sortedDefs := CanonicalizeToolDefs(toolDefs)
	for i, want := range wantOrder {
		if sortedDefs[i].Function.Name != want {
			t.Errorf("sortedDefs[%d] = %q, want %q", i, sortedDefs[i].Function.Name, want)
		}
	}

	_, sortedWithDefs, defStats := c.CanonicalizePromptMessagesWithToolDefs(msgs, toolDefs)
	if defStats.ToolsSorted != len(wantOrder) {
		t.Errorf("defStats.ToolsSorted = %d, want %d", defStats.ToolsSorted, len(wantOrder))
	}
	for i, want := range wantOrder {
		if sortedWithDefs[i].Function.Name != want {
			t.Errorf("sortedWithDefs[%d] = %q, want %q", i, sortedWithDefs[i].Function.Name, want)
		}
	}
}

// TestPromptOrdering_TimestampHoisting verifies that injected dynamic timestamps (Today is 2026-09-07...)
// are stripped/hoisted from Tier 1 so prefix cache hit rate is preserved.
func TestPromptOrdering_TimestampHoisting(t *testing.T) {
	c := DefaultDeterministicPromptCanonicalizer()

	sessionA := []agent.Message{
		{
			Role: agent.RoleSystem,
			Content: "You are an autonomous engineering agent.\n" +
				"Today is 2026-09-07.\n" +
				"Session ID: sess-20260907-001",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Project Rules\nAlways verify with tests.",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Workspace Context\nDirectory Structure:\ncmd/\ninternal/",
		},
		{
			Role:    agent.RoleUser,
			Content: "Please review the pull request.",
		},
	}

	sessionB := []agent.Message{
		{
			Role: agent.RoleSystem,
			Content: "You are an autonomous engineering agent.\n" +
				"Today is 2026-10-15.\n" +
				"Session ID: sess-20261015-999",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Project Rules\nAlways verify with tests.",
		},
		{
			Role:    agent.RoleSystem,
			Content: "# Workspace Context\nDirectory Structure:\ncmd/\ninternal/",
		},
		{
			Role:    agent.RoleUser,
			Content: "Please review the pull request.",
		},
	}

	outA, _, statsA := c.CanonicalizePromptMessages(sessionA, nil)
	outB, _, statsB := c.CanonicalizePromptMessages(sessionB, nil)

	// Tier 1 in both sessions must have timestamps stripped
	tier1A := outA[0].Content
	tier1B := outB[0].Content

	if strings.Contains(tier1A, "Today is 2026-09-07") || strings.Contains(tier1A, "Session ID:") {
		t.Errorf("session A Tier 1 still contains volatile ephemera: %q", tier1A)
	}
	if strings.Contains(tier1B, "Today is 2026-10-15") || strings.Contains(tier1B, "Session ID:") {
		t.Errorf("session B Tier 1 still contains volatile ephemera: %q", tier1B)
	}

	// Tier 1 must now be byte-identical between session A and session B
	if tier1A != tier1B {
		t.Errorf("Tier 1 not identical between sessions:\nTier1A: %q\nTier1B: %q", tier1A, tier1B)
	}

	// Timestamps must be hoisted to Tier 5 (user message)
	lastMsgA := outA[len(outA)-1].Content
	lastMsgB := outB[len(outB)-1].Content

	if !strings.Contains(lastMsgA, "Today is 2026-09-07") {
		t.Errorf("session A user turn missing hoisted timestamp: %q", lastMsgA)
	}
	if !strings.Contains(lastMsgB, "Today is 2026-10-15") {
		t.Errorf("session B user turn missing hoisted timestamp: %q", lastMsgB)
	}

	if statsA.VolatileElementsHoisted == 0 {
		t.Errorf("statsA.VolatileElementsHoisted = 0, want > 0")
	}
	if statsB.VolatileElementsHoisted == 0 {
		t.Errorf("statsB.VolatileElementsHoisted = 0, want > 0")
	}

	// Prefix hashes over Tiers 1-4 must be identical despite different dates/sessions
	if statsA.StablePrefixHash == "" {
		t.Fatalf("statsA.StablePrefixHash is empty")
	}
	if statsA.StablePrefixHash != statsB.StablePrefixHash {
		t.Fatalf("StablePrefixHash mismatch:\nstatsA: %s\nstatsB: %s", statsA.StablePrefixHash, statsB.StablePrefixHash)
	}
}
