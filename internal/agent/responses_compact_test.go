package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

func TestResponsesCompact_PrefixAnchorPreserved(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "You are an autonomous agent system prompt with extensive developer guidelines."},
		{Role: "developer", Content: "Tools available: [Read, Edit, Write, Bash, ContextRestore]."},
		{Role: RoleUser, Content: "Initial user instruction: bootstrap the new service architecture."},
		{Role: RoleAssistant, Content: strings.Repeat("Detailed intermediate plan step 1: initialize components. ", 20)},
		{Role: RoleUser, Content: strings.Repeat("User feedback on step 1: looks good, continue with step 2. ", 20)},
		{Role: RoleAssistant, Content: strings.Repeat("Detailed intermediate plan step 2: wire dependencies. ", 20)},
		{Role: RoleUser, Content: strings.Repeat("User feedback on step 2: please proceed to testing. ", 20)},
		{Role: RoleAssistant, Content: "Recent assistant turn: preparing test suite execution."},
		{Role: RoleUser, Content: "Recent user turn: execute the tests now."},
	}

	origTokens := EstimateResponsesMessagesTokens(messages)
	budget := origTokens / 2 // force aggressive compaction

	compacted, outcome, err := CompactResponsesMessages(messages, budget, nil)
	if err != nil {
		t.Fatalf("CompactResponsesMessages returned error: %v", err)
	}

	if !outcome.PrefixAnchorPreserved {
		t.Fatalf("expected PrefixAnchorPreserved == true, got false")
	}

	if outcome.ShedTurns == 0 {
		t.Fatalf("expected middle turns to be shed, got ShedTurns == 0")
	}

	// Verify leading prefix anchor messages (0, 1, 2) are 100% untouched
	if len(compacted) < 3 {
		t.Fatalf("compacted messages length %d too short", len(compacted))
	}
	for i := 0; i <= 2; i++ {
		if compacted[i].Role != messages[i].Role {
			t.Errorf("prefix message %d role mutated: got %q, want %q", i, compacted[i].Role, messages[i].Role)
		}
		if compacted[i].Content != messages[i].Content {
			t.Errorf("prefix message %d content mutated: got %q, want %q", i, compacted[i].Content, messages[i].Content)
		}
	}
}

func TestResponsesCompact_MiddleTurnsShedToBudget(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt instructions for Responses wire."},
		{Role: RoleUser, Content: "First user message establishing the thread."},
		{Role: RoleAssistant, Content: strings.Repeat("Middle assistant turn A with considerable context length. ", 25)},
		{Role: RoleUser, Content: strings.Repeat("Middle user turn B with considerable context length. ", 25)},
		{Role: RoleAssistant, Content: strings.Repeat("Middle assistant turn C with considerable context length. ", 25)},
		{Role: RoleUser, Content: strings.Repeat("Middle user turn D with considerable context length. ", 25)},
		{Role: RoleAssistant, Content: "Recent suffix assistant turn."},
		{Role: RoleUser, Content: "Recent suffix user turn."},
	}

	origTokens := EstimateResponsesMessagesTokens(messages)
	budget := origTokens / 2

	compacted, outcome, err := CompactResponsesMessages(messages, budget, nil)
	if err != nil {
		t.Fatalf("CompactResponsesMessages returned error: %v", err)
	}

	if outcome.OriginalTokens != origTokens {
		t.Errorf("OriginalTokens = %d, want %d", outcome.OriginalTokens, origTokens)
	}

	if outcome.CompactedTokens > budget {
		t.Errorf("CompactedTokens = %d, exceeded budget %d", outcome.CompactedTokens, budget)
	}

	if outcome.ShedTurns == 0 {
		t.Errorf("expected ShedTurns > 0, got 0")
	}

	if outcome.ShedTokens <= 0 {
		t.Errorf("expected ShedTokens > 0, got %d", outcome.ShedTokens)
	}

	if outcome.ShedTokens != outcome.OriginalTokens-outcome.CompactedTokens {
		t.Errorf("ShedTokens %d != OriginalTokens %d - CompactedTokens %d",
			outcome.ShedTokens, outcome.OriginalTokens, outcome.CompactedTokens)
	}

	// Verify recent suffix turns (last 2 messages) are kept intact
	if len(compacted) < 2 {
		t.Fatalf("compacted length %d too short", len(compacted))
	}
	lastCompacted := compacted[len(compacted)-1]
	lastOriginal := messages[len(messages)-1]
	secondLastCompacted := compacted[len(compacted)-2]
	secondLastOriginal := messages[len(messages)-2]

	if lastCompacted.Content != lastOriginal.Content || lastCompacted.Role != lastOriginal.Role {
		t.Errorf("last suffix message mutated: got %+v, want %+v", lastCompacted, lastOriginal)
	}
	if secondLastCompacted.Content != secondLastOriginal.Content || secondLastCompacted.Role != secondLastOriginal.Role {
		t.Errorf("second to last suffix message mutated: got %+v, want %+v", secondLastCompacted, secondLastOriginal)
	}
}

func TestResponsesCompact_GoalsNeverDropped(t *testing.T) {
	// Injects [fak:goal] deploy production service in turn 3; verifies it survives compaction even when surrounding turns are dropped.
	goalText := "[fak:goal] deploy production service"
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt for continuous integration."},
		{Role: RoleUser, Content: "First user turn starting the session."},
		{Role: RoleAssistant, Content: strings.Repeat("Volatile intermediate turn 2 with large payload. ", 30)},
		{Role: RoleUser, Content: goalText}, // Turn 3: Goal
		{Role: RoleAssistant, Content: strings.Repeat("Volatile intermediate turn 4 with large payload. ", 30)},
		{Role: RoleUser, Content: strings.Repeat("Volatile intermediate turn 5 with large payload. ", 30)},
		{Role: RoleAssistant, Content: "Recent suffix assistant message."},
		{Role: RoleUser, Content: "Recent suffix user message."},
	}

	origTokens := EstimateResponsesMessagesTokens(messages)
	budget := origTokens / 2

	compacted, outcome, err := CompactResponsesMessages(messages, budget, nil)
	if err != nil {
		t.Fatalf("CompactResponsesMessages returned error: %v", err)
	}

	if outcome.GoalsPreserved < 1 {
		t.Errorf("expected GoalsPreserved >= 1, got %d", outcome.GoalsPreserved)
	}

	// Verify turn 3 survived compaction verbatim
	goalFound := false
	for _, m := range compacted {
		if strings.Contains(m.Content, goalText) {
			goalFound = true
			if m.Role != RoleUser {
				t.Errorf("goal message role changed to %q, want %q", m.Role, RoleUser)
			}
			break
		}
	}
	if !goalFound {
		t.Fatalf("goal %q was dropped during compaction", goalText)
	}

	// Verify surrounding turns (turn 2, 4, 5) were shed
	if outcome.ShedTurns < 2 {
		t.Errorf("expected surrounding turns to be shed, got ShedTurns = %d", outcome.ShedTurns)
	}
}

func TestResponsesCompact_CASRestoreStubsCreated(t *testing.T) {
	casStore := make(map[string][]byte)
	casPut := func(b []byte) string {
		h := sha256.Sum256(b)
		digest := hex.EncodeToString(h[:])
		casStore[digest] = b
		return digest
	}

	turn2Content := strings.Repeat("Important turn 2 intermediate data to preserve in CAS. ", 20)
	turn3Content := strings.Repeat("Important turn 3 intermediate data to preserve in CAS. ", 20)

	messages := []Message{
		{Role: RoleSystem, Content: "System prompt."},
		{Role: RoleUser, Content: "First user request."},
		{Role: RoleAssistant, Content: turn2Content},
		{Role: RoleUser, Content: turn3Content},
		{Role: RoleAssistant, Content: "Recent suffix assistant."},
		{Role: RoleUser, Content: "Recent suffix user."},
	}

	origTokens := EstimateResponsesMessagesTokens(messages)
	budget := origTokens / 2

	compacted, outcome, err := CompactResponsesMessages(messages, budget, casPut)
	if err != nil {
		t.Fatalf("CompactResponsesMessages returned error: %v", err)
	}

	if outcome.RestoreStubsCreated == 0 {
		t.Fatalf("expected RestoreStubsCreated > 0, got 0")
	}

	if outcome.ShedTurns != outcome.RestoreStubsCreated {
		t.Errorf("ShedTurns %d != RestoreStubsCreated %d", outcome.ShedTurns, outcome.RestoreStubsCreated)
	}

	if len(casStore) == 0 {
		t.Fatalf("casStore was not populated by casPut")
	}

	// Check stubs in compacted messages
	stubsChecked := 0
	for _, m := range compacted {
		if strings.HasPrefix(m.Content, "[fak:compacted") {
			stubsChecked++
			if m.Role != RoleAssistant {
				t.Errorf("stub message role = %q, want %q", m.Role, RoleAssistant)
			}
			ref, size, turns, ok := ParseResponsesCompactStub(m.Content)
			if !ok {
				t.Fatalf("failed to parse stub: %q", m.Content)
			}
			if len(ref) != 64 {
				t.Errorf("stub ref %q is not a 64-char sha256 hex digest", ref)
			}
			if turns != 1 {
				t.Errorf("stub turns = %d, want 1", turns)
			}
			stored, exists := casStore[ref]
			if !exists {
				t.Errorf("stub ref %q not found in CAS store", ref)
			}
			if len(stored) != size {
				t.Errorf("stub size %d != stored byte length %d", size, len(stored))
			}
			if !strings.Contains(m.Content, "hint=\"fak_context_restore\"") {
				t.Errorf("stub missing hint=\"fak_context_restore\": %q", m.Content)
			}
		}
	}

	if stubsChecked != outcome.RestoreStubsCreated {
		t.Errorf("found %d stubs, want %d", stubsChecked, outcome.RestoreStubsCreated)
	}
}

func TestResponsesCompact_UnderBudgetUntouched(t *testing.T) {
	messages := []Message{
		{Role: RoleSystem, Content: "System prompt."},
		{Role: RoleUser, Content: "User instruction."},
		{Role: RoleAssistant, Content: "Short assistant reply."},
		{Role: RoleUser, Content: "Short user follow-up."},
		{Role: RoleAssistant, Content: "Recent assistant."},
		{Role: RoleUser, Content: "Recent user."},
	}

	origTokens := EstimateResponsesMessagesTokens(messages)
	budget := origTokens + 500 // well under budget

	compacted, outcome, err := CompactResponsesMessages(messages, budget, nil)
	if err != nil {
		t.Fatalf("CompactResponsesMessages returned error: %v", err)
	}

	if outcome.ShedTurns != 0 {
		t.Errorf("expected ShedTurns == 0, got %d", outcome.ShedTurns)
	}
	if outcome.RestoreStubsCreated != 0 {
		t.Errorf("expected RestoreStubsCreated == 0, got %d", outcome.RestoreStubsCreated)
	}
	if outcome.ShedTokens != 0 {
		t.Errorf("expected ShedTokens == 0, got %d", outcome.ShedTokens)
	}
	if outcome.CompactedTokens != outcome.OriginalTokens {
		t.Errorf("CompactedTokens %d != OriginalTokens %d", outcome.CompactedTokens, outcome.OriginalTokens)
	}
	if !outcome.PrefixAnchorPreserved {
		t.Errorf("expected PrefixAnchorPreserved == true")
	}

	if len(compacted) != len(messages) {
		t.Fatalf("compacted length %d != original length %d", len(compacted), len(messages))
	}
	for i := range messages {
		if compacted[i].Role != messages[i].Role || compacted[i].Content != messages[i].Content {
			t.Errorf("message %d was modified under budget: got %+v, want %+v", i, compacted[i], messages[i])
		}
	}
}

func TestResponsesCompact_GoalPrefixVariants(t *testing.T) {
	tests := []struct {
		name    string
		content string
		isGoal  bool
	}{
		{"bracket goal", "[fak:goal] migrate database schema", true},
		{"colon goal", "Goal: migrate database schema", true},
		{"multiline goal", "Context line 1\n[fak:goal] ship release", true},
		{"regular message", "I will work on the task now", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := Message{Role: RoleUser, Content: tt.content}
			if got := isResponsesGoalMessage(m); got != tt.isGoal {
				t.Errorf("isResponsesGoalMessage(%q) = %v, want %v", tt.content, got, tt.isGoal)
			}
		})
	}
}
