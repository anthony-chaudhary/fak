package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestCompactMessagesUnderBudget(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "You are a helpful assistant."},
		{Role: RoleUser, Content: "Hello"},
		{Role: RoleAssistant, Content: "Hi there!"},
	}
	out, oc := CompactMessages(messages, 1000)
	if oc.Reason != CompactReasonUnderBudget {
		t.Fatalf("expected under_budget, got %s", oc.Reason)
	}
	if len(out) != len(messages) {
		t.Fatalf("expected %d messages, got %d", len(messages), len(out))
	}
}

func TestCompactMessagesTooFew(t *testing.T) {
	messages := []Message{
		{Role: RoleUser, Content: "Hello"},
		{Role: RoleAssistant, Content: "Hi"},
	}
	out, oc := CompactMessages(messages, 10)
	if oc.Reason != CompactReasonTooFewMsgs {
		t.Fatalf("expected too_few_msgs, got %s", oc.Reason)
	}
	if len(out) != len(messages) {
		t.Fatalf("expected %d messages, got %d", len(messages), len(out))
	}
}

func TestCompactMessagesPreservesSystemPrefix(t *testing.T) {
	sysContent := "SYSTEM: You are an autonomous coding agent operating in the workspace."
	messages := []Message{
		{Role: RoleSystem, Content: sysContent},
		{Role: RoleUser, Content: "Task 1: Inspect the workspace layout and list all files in cmd/fak."},
		{Role: RoleAssistant, Content: "Understood, inspecting cmd/fak directory now."},
		{Role: RoleUser, Content: "Task 2: Read the configuration file config.toml and verify keys."},
		{Role: RoleAssistant, Content: "Read complete, all configuration keys validated successfully."},
		{Role: RoleUser, Content: "Task 3: Execute build and run unit test suite."},
		{Role: RoleAssistant, Content: "Build succeeded, running tests now."},
		{Role: RoleUser, Content: "Task 4: What is the current status of the build?"},
	}
	// Small budget to force compaction of older turns
	out, oc := CompactMessages(messages, 60)
	if oc.Reason != CompactReasonNone {
		t.Fatalf("expected compaction to fire (CompactReasonNone), got %q", oc.Reason)
	}
	if oc.Dropped == 0 || oc.ShedTokens == 0 {
		t.Fatalf("expected dropped > 0 and shedTokens > 0, got dropped=%d, shed=%d", oc.Dropped, oc.ShedTokens)
	}
	// Check system message at index 0 is bit-for-bit identical
	if out[0].Role != RoleSystem || out[0].Content != sysContent {
		t.Fatalf("system prompt at index 0 corrupted! got role=%s content=%q", out[0].Role, out[0].Content)
	}
	// Last turn should be kept
	if out[len(out)-1].Content != messages[len(messages)-1].Content {
		t.Fatalf("latest user message not kept: got %q", out[len(out)-1].Content)
	}
}

func TestCompactMessagesTombstonesOriginatingTask(t *testing.T) {
	originating := "OrigTask: Implement comprehensive prompt shrink support for in-kernel wire."
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt"},
		{Role: RoleUser, Content: originating},
		{Role: RoleAssistant, Content: "Working on task step 1."},
		{Role: RoleUser, Content: "Proceed to step 2."},
		{Role: RoleAssistant, Content: "Working on task step 2."},
		{Role: RoleUser, Content: "Proceed to step 3."},
		{Role: RoleAssistant, Content: "Working on task step 3."},
		{Role: RoleUser, Content: "Verify everything."},
	}
	stashed := make(map[string]string)
	opts := CompactOptions{
		Budget: 40,
		RestoreStash: func(id, excerpt string, body []byte) {
			stashed[id] = string(body)
		},
	}
	out, oc := CompactMessagesWithOptions(messages, opts)
	if oc.Reason != CompactReasonNone {
		t.Fatalf("compaction failed: %s", oc.Reason)
	}
	if oc.RestoreID == "" {
		t.Fatal("expected non-empty RestoreID for dropped originating task")
	}
	if stashed[oc.RestoreID] != originating {
		t.Fatalf("stashed body = %q, want %q", stashed[oc.RestoreID], originating)
	}
	// Verify stub message contains tombstone and restore_id
	stubFound := false
	for _, m := range out {
		if strings.Contains(m.Content, compactStubPrefix) {
			stubFound = true
			if !strings.Contains(m.Content, compactTombstonePrefix) {
				t.Fatalf("stub message missing tombstone prefix: %q", m.Content)
			}
			if !strings.Contains(m.Content, oc.RestoreID) {
				t.Fatalf("stub message missing restore_id: %q", m.Content)
			}
		}
	}
	if !stubFound {
		t.Fatal("compact stub message not found in output")
	}
}

func TestCompactMessagesHoistsGoalPin(t *testing.T) {
	goalText := "[fak:goal] Maintain zero boundary violations between fak and fak-private at all times."
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt"},
		{Role: RoleUser, Content: "Initial prompt"},
		{Role: RoleAssistant, Content: "Understood"},
		{Role: RoleUser, Content: goalText},
		{Role: RoleAssistant, Content: "Goal recorded and pinned."},
		{Role: RoleUser, Content: "Intermediate work turn A"},
		{Role: RoleAssistant, Content: "Finished turn A"},
		{Role: RoleUser, Content: "Intermediate work turn B"},
		{Role: RoleAssistant, Content: "Finished turn B"},
		{Role: RoleUser, Content: "Final status check"},
	}
	out, oc := CompactMessages(messages, 50)
	if oc.Reason != CompactReasonNone {
		t.Fatalf("compaction failed: %s", oc.Reason)
	}
	goalFound := false
	for _, m := range out {
		if strings.Contains(m.Content, goalText) {
			goalFound = true
			break
		}
	}
	if !goalFound {
		t.Fatalf("pinned goal was not hoisted out of drop range: %+v", out)
	}
}

func TestCompactMessagesToolPairingNotOrphaned(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt"},
		{Role: RoleUser, Content: "Long initial user query that fills some token budget..."},
		{Role: RoleAssistant, Content: "Initial plan description"},
		{Role: RoleUser, Content: "Execute tool call"},
		{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{ID: "call_abc", Type: "function", Function: Func{Name: "Read", Arguments: `{"file_path":"test.go"}`}},
			},
		},
		{Role: RoleTool, ToolCallID: "call_abc", Content: "file content here"},
		{Role: RoleAssistant, Content: "I have read test.go"},
		{Role: RoleUser, Content: "Now verify"},
	}
	out, oc := CompactMessages(messages, 60)
	if oc.Reason != CompactReasonNone {
		t.Fatalf("compaction failed: %s", oc.Reason)
	}
	// Verify that if RoleTool is in out, its preceding assistant ToolCall is also in out
	toolIdx := -1
	asstIdx := -1
	for i, m := range out {
		if m.Role == RoleTool && m.ToolCallID == "call_abc" {
			toolIdx = i
		}
		if m.Role == RoleAssistant && len(m.ToolCalls) > 0 && m.ToolCalls[0].ID == "call_abc" {
			asstIdx = i
		}
	}
	if toolIdx != -1 && asstIdx == -1 {
		t.Fatalf("tool result was orphaned: toolIdx=%d but assistant tool call was dropped", toolIdx)
	}
	if toolIdx != -1 && asstIdx != -1 && asstIdx >= toolIdx {
		t.Fatalf("assistant tool call must precede tool result: asst=%d, tool=%d", asstIdx, toolIdx)
	}
}

func TestElideStaleReadMessages(t *testing.T) {
	bigContent := strings.Repeat("stale read content line\n", 20)
	messages := []Message{
		{Role: RoleUser, Content: "Inspect file"},
		{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{ID: "read-1", Type: "function", Function: Func{Name: "Read", Arguments: `{"file_path":"internal/app.go"}`}},
			},
		},
		{Role: RoleTool, ToolCallID: "read-1", Content: bigContent},
		{
			Role: RoleAssistant,
			ToolCalls: []ToolCall{
				{ID: "edit-1", Type: "function", Function: Func{Name: "Edit", Arguments: `{"file_path":"internal/app.go","old":"a","new":"b"}`}},
			},
		},
		{Role: RoleTool, ToolCallID: "edit-1", Content: "edited successfully"},
		{Role: RoleAssistant, Content: "Done editing"},
		{Role: RoleUser, Content: "Tail 1"},
		{Role: RoleAssistant, Content: "Tail 2"},
		{Role: RoleUser, Content: "Tail 3"},
		{Role: RoleAssistant, Content: "Tail 4"},
	}
	stashed := make(map[string]string)
	out := ElideStaleReadMessages(messages, func(id, excerpt string, body []byte) {
		stashed[id] = string(body)
	})
	if out[2].Content == bigContent {
		t.Fatal("stale read content was not elided")
	}
	if !strings.Contains(out[2].Content, "superseded by a later in-session edit") {
		t.Fatalf("elided message missing superseded marker: %q", out[2].Content)
	}
	if len(stashed) == 0 {
		t.Fatal("stale read body was not stashed")
	}
}

func TestDeferColdToolDefs(t *testing.T) {
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Bash", Description: "Execute shell command"}},
		{Type: "function", Function: ToolDefFunction{Name: "Read", Description: "Read file"}},
		{Type: "function", Function: ToolDefFunction{Name: "custom_cold_analyzer", Description: "Heavy cold analyzer"}},
		{Type: "function", Function: ToolDefFunction{Name: "database_migrator", Description: "Migrate database"}},
	}
	hot, coldCount := DeferColdToolDefs(tools)
	if coldCount != 2 {
		t.Fatalf("expected 2 cold tools deferred, got %d", coldCount)
	}
	hasSearch := false
	for _, tdef := range hot {
		if tdef.Function.Name == "custom_cold_analyzer" || tdef.Function.Name == "database_migrator" {
			t.Fatalf("cold tool %q leaked into hot set", tdef.Function.Name)
		}
		if tdef.Function.Name == "ToolSearch" {
			hasSearch = true
		}
	}
	if !hasSearch {
		t.Fatal("ToolSearch tool definition was not injected")
	}
}

func TestInKernelPlannerApplyPromptShrink(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	m := model.NewSynthetic(cfg)
	tok := loadProbeTok(t)
	p := NewInKernelPlanner(m, tok, "tiny-test", false, nil, false)
	p.quant = false
	p.maxNew = 4
	// Enable prompt-shrink levers
	p.SetPromptShrinkLevers(80, true, true)

	bigRead := strings.Repeat("huge file contents to read line\n", 30)
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt invariant instructions."},
		{Role: RoleUser, Content: "Read the file first."},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "r1", Type: "function", Function: Func{Name: "Read", Arguments: `{"file_path":"pkg.go"}`}}}},
		{Role: RoleTool, ToolCallID: "r1", Content: bigRead},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "e1", Type: "function", Function: Func{Name: "Edit", Arguments: `{"file_path":"pkg.go"}`}}}},
		{Role: RoleTool, ToolCallID: "e1", Content: "ok"},
		{Role: RoleAssistant, Content: "File edited."},
		{Role: RoleUser, Content: "Middle turn 1"},
		{Role: RoleAssistant, Content: "Middle reply 1"},
		{Role: RoleUser, Content: "Middle turn 2"},
		{Role: RoleAssistant, Content: "Middle reply 2"},
		{Role: RoleUser, Content: "Latest query: what is the final state?"},
	}
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Bash", Description: "Run bash"}},
		{Type: "function", Function: ToolDefFunction{Name: "cold_tool_x", Description: "Cold tool"}},
	}

	shrunkMsgs, shrunkTools, outcome := p.ApplyPromptShrink(context.Background(), messages, tools)
	if !outcome.Compacted {
		t.Fatalf("expected compacted=true, outcome=%+v", outcome)
	}
	if outcome.ColdToolsDeferred != 1 {
		t.Fatalf("expected 1 cold tool deferred, got %d", outcome.ColdToolsDeferred)
	}
	if len(shrunkMsgs) >= len(messages) {
		t.Fatalf("shrunk messages length %d not less than original %d", len(shrunkMsgs), len(messages))
	}
	// Verify system message preserved at index 0
	if shrunkMsgs[0].Role != RoleSystem || shrunkMsgs[0].Content != messages[0].Content {
		t.Fatalf("system prompt not preserved bit-for-bit: got %q", shrunkMsgs[0].Content)
	}
	// Verify hot tools contains Bash and ToolSearch
	hasToolSearch := false
	for _, td := range shrunkTools {
		if td.Function.Name == "ToolSearch" {
			hasToolSearch = true
		}
	}
	if !hasToolSearch {
		t.Fatal("shrunk tools missing ToolSearch")
	}

	// Now complete a turn through InKernelPlanner.Complete and verify execution
	comp, err := p.Complete(context.Background(), messages, tools)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if comp == nil {
		t.Fatal("expected non-nil completion")
	}
}

func TestInKernelPlannerPromptShrinkSampleOpts(t *testing.T) {
	cfg := tinyConcurrencyConfig()
	m := model.NewSynthetic(cfg)
	tok := loadProbeTok(t)
	// Create planner with levers disabled by default
	p := NewInKernelPlanner(m, tok, "tiny-test-opt", false, nil, false)
	p.quant = false
	p.maxNew = 4

	messages := []Message{
		{Role: RoleSystem, Content: "System instructions."},
		{Role: RoleUser, Content: "Turn 1"},
		{Role: RoleAssistant, Content: "Reply 1"},
		{Role: RoleUser, Content: "Turn 2"},
		{Role: RoleAssistant, Content: "Reply 2"},
		{Role: RoleUser, Content: "Turn 3"},
		{Role: RoleAssistant, Content: "Reply 3"},
		{Role: RoleUser, Content: "Turn 4"},
	}
	tools := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "Bash", Description: "Run bash"}},
		{Type: "function", Function: ToolDefFunction{Name: "cold_db_tool", Description: "Database tool"}},
	}

	// Without opts: no prompt shrink
	shrunkMsgs1, shrunkTools1, oc1 := p.ApplyPromptShrink(context.Background(), messages, tools)
	if oc1.Compacted || oc1.ColdToolsDeferred != 0 {
		t.Fatalf("expected no shrink without options, got oc1=%+v", oc1)
	}
	if len(shrunkMsgs1) != len(messages) || len(shrunkTools1) != len(tools) {
		t.Fatalf("expected untouched messages and tools")
	}

	// With per-request options: prompt shrink active!
	shrunkMsgs2, shrunkTools2, oc2 := p.ApplyPromptShrink(context.Background(), messages, tools,
		WithCompactHistoryBudget(20),
		WithDeferColdTools(true),
	)
	if !oc2.Compacted {
		t.Fatalf("expected compacted=true with SampleOpt, got %+v", oc2)
	}
	if oc2.CompactOutcome.Dropped < 2 {
		t.Fatalf("expected at least 2 dropped messages, got %d", oc2.CompactOutcome.Dropped)
	}
	if oc2.ColdToolsDeferred != 1 {
		t.Fatalf("expected 1 cold tool deferred with SampleOpt, got %d", oc2.ColdToolsDeferred)
	}
	for _, td := range shrunkTools2 {
		if td.Function.Name == "cold_db_tool" {
			t.Fatalf("cold tool was not deferred with SampleOpt")
		}
	}
	if len(shrunkMsgs2) >= len(messages) {
		t.Fatalf("messages were not compacted: got %d, want < %d", len(shrunkMsgs2), len(messages))
	}
}
