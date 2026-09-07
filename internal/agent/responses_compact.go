package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
)

// responses_compact.go — port lossless cache-prefix-preserving compaction to the OpenAI Responses wire (/v1/responses).
//
// Invariant 1: Leading Prefix Anchor (all initial system messages, tool declarations, and the first user message)
//              must NEVER be shed or modified (PrefixAnchorPreserved = true).
// Invariant 2: Recent Suffix Turns (the last K >= 2 conversation turns) must be kept intact so immediate context is preserved.
// Invariant 3: Goal Invariant: messages containing goal markers like [fak:goal] or Goal: must NEVER be dropped.
//
// Middle-turn shedding:
// Drops older intermediate conversation turns from the middle until total tokens <= tokenBudget.
// For each shed turn:
//   - If casPut != nil: store the full message content in CAS, obtain SHA-256 digest ref.
//   - Replace the dropped turn with a compact stub Message:
//     Role: "assistant", Content: fmt.Sprintf("[fak:compacted id=%s size=%d turns=%d hint=\"fak_context_restore\"]", ref, len(content), 1)

// CompactResponsesOutcome is an alias to CompactOutcome for Responses compaction callers.
type CompactResponsesOutcome = CompactOutcome

const defaultResponsesSuffixKeepTurns = 2

// CompactResponsesMessages compacts a conversation history for the OpenAI Responses wire (/v1/responses)
// down to tokenBudget while strictly preserving the leading prefix anchor (system messages, tool declarations,
// first user turn), recent suffix turns (last K >= 2 conversation turns), and standing goals ([fak:goal] or Goal:).
// Dropped intermediate turns are replaced by compact content-addressed restore stubs via CAS.
func CompactResponsesMessages(messages []Message, tokenBudget int, casPut func([]byte) string) ([]Message, CompactOutcome, error) {
	if len(messages) == 0 {
		return messages, CompactOutcome{PrefixAnchorPreserved: true}, nil
	}

	origTokens := EstimateResponsesMessagesTokens(messages)
	initialGoals := countResponsesGoals(messages)

	// If already within budget or budget is non-positive (unbounded), keep all messages intact.
	if tokenBudget <= 0 || origTokens <= tokenBudget {
		return messages, CompactOutcome{
			OriginalTokens:        origTokens,
			CompactedTokens:       origTokens,
			ShedTokens:            0,
			ShedTurns:             0,
			PrefixAnchorPreserved: true,
			GoalsPreserved:        initialGoals,
			RestoreStubsCreated:   0,
		}, nil
	}

	anchorEnd := findResponsesPrefixAnchorEnd(messages)
	suffixStart := len(messages) - defaultResponsesSuffixKeepTurns
	if suffixStart < 0 {
		suffixStart = 0
	}

	// Do not orphan tool calls / tool results in recent suffix.
	for suffixStart > anchorEnd+1 && (messages[suffixStart].Role == RoleTool || messages[suffixStart].Role == "tool" || messages[suffixStart].ToolCallID != "") {
		suffixStart--
	}

	// If there are no intermediate turns between prefix anchor and suffix turns, we cannot shed anything.
	if anchorEnd+1 >= suffixStart {
		return messages, CompactOutcome{
			OriginalTokens:        origTokens,
			CompactedTokens:       origTokens,
			ShedTokens:            0,
			ShedTurns:             0,
			PrefixAnchorPreserved: true,
			GoalsPreserved:        initialGoals,
			RestoreStubsCreated:   0,
		}, nil
	}

	compacted := append([]Message(nil), messages...)
	currentTokens := origTokens
	shedTurns := 0
	restoreStubsCreated := 0

	// Middle-turn shedding: drop older intermediate conversation turns from the middle until total tokens <= tokenBudget.
	for i := anchorEnd + 1; i < suffixStart; i++ {
		if currentTokens <= tokenBudget {
			break
		}
		if isResponsesGoalMessage(compacted[i]) {
			// Invariant 3: Goal Invariant — messages containing goal markers must NEVER be dropped.
			continue
		}

		content := compacted[i].Content
		body := []byte(content)
		ref := ""
		if casPut != nil {
			ref = casPut(body)
		}
		if ref == "" {
			h := sha256.Sum256(body)
			ref = hex.EncodeToString(h[:])
		}

		stub := Message{
			Role:    RoleAssistant,
			Content: fmt.Sprintf("[fak:compacted id=%s size=%d turns=%d hint=\"fak_context_restore\"]", ref, len(content), 1),
		}

		origTurnTokens := EstimateResponsesMessageTokens(compacted[i])
		stubTokens := EstimateResponsesMessageTokens(stub)

		compacted[i] = stub
		currentTokens = currentTokens - origTurnTokens + stubTokens
		shedTurns++
		restoreStubsCreated++
	}

	compactedTokens := EstimateResponsesMessagesTokens(compacted)
	shedTokens := origTokens - compactedTokens
	if shedTokens < 0 {
		shedTokens = 0
	}

	outcome := CompactOutcome{
		OriginalTokens:        origTokens,
		CompactedTokens:       compactedTokens,
		ShedTokens:            shedTokens,
		ShedTurns:             shedTurns,
		PrefixAnchorPreserved: true,
		GoalsPreserved:        countResponsesGoals(compacted),
		RestoreStubsCreated:   restoreStubsCreated,
	}

	return compacted, outcome, nil
}

// EstimateResponsesMessageTokens estimates the token count of a single message (~4 chars/token heuristic).
func EstimateResponsesMessageTokens(m Message) int {
	return EstimateMessageTokens(m)
}

// EstimateResponsesMessagesTokens estimates total tokens for a slice of messages.
func EstimateResponsesMessagesTokens(messages []Message) int {
	total := 0
	for _, m := range messages {
		total += EstimateResponsesMessageTokens(m)
	}
	return total
}

// findResponsesPrefixAnchorEnd finds the inclusive end index of the leading prefix anchor.
// Invariant 1: all initial system messages, tool declarations, and the first user message.
func findResponsesPrefixAnchorEnd(messages []Message) int {
	if len(messages) == 0 {
		return -1
	}
	for i, m := range messages {
		if m.Role == RoleUser || m.Role == "user" {
			return i
		}
	}
	lastSystem := -1
	for i, m := range messages {
		role := strings.ToLower(m.Role)
		if role == RoleSystem || role == "system" || role == "developer" || role == "tools" || role == "tool" {
			lastSystem = i
		} else {
			break
		}
	}
	return lastSystem
}

// isResponsesGoalMessage reports whether a message contains goal markers like [fak:goal] or Goal:.
// Invariant 3: messages containing goal markers must NEVER be dropped.
func isResponsesGoalMessage(m Message) bool {
	if strings.Contains(m.Content, "[fak:goal]") || strings.Contains(m.Content, "Goal:") {
		return true
	}
	text := strings.TrimSpace(m.Content)
	if strings.HasPrefix(text, "goal:") || strings.HasPrefix(text, "Goal:") {
		return true
	}
	if strings.Contains(strings.ToLower(text), "[fak:goal]") {
		return true
	}
	return isCompactGoalText(m.Content)
}

func countResponsesGoals(messages []Message) int {
	count := 0
	for _, m := range messages {
		if isResponsesGoalMessage(m) {
			count++
		}
	}
	return count
}

// ParseResponsesCompactStub parses a compact restore stub emitted during Responses compaction.
func ParseResponsesCompactStub(content string) (ref string, size int, turns int, ok bool) {
	content = strings.TrimSpace(content)
	if strings.HasPrefix(content, "[fak:compacted") && strings.HasSuffix(content, "]") {
		inner := strings.TrimPrefix(content, "[fak:compacted")
		inner = strings.TrimSuffix(inner, "]")
		for _, part := range strings.Fields(inner) {
			if strings.HasPrefix(part, "id=") {
				ref = strings.TrimPrefix(part, "id=")
				ref = strings.Trim(ref, `"'`)
			} else if strings.HasPrefix(part, "size=") {
				fmt.Sscanf(strings.TrimPrefix(part, "size="), "%d", &size)
			} else if strings.HasPrefix(part, "turns=") {
				fmt.Sscanf(strings.TrimPrefix(part, "turns="), "%d", &turns)
			}
		}
		if ref != "" {
			return ref, size, turns, true
		}
	}

	if strings.HasPrefix(content, "{") && strings.Contains(content, "_paged") {
		var m struct {
			Paged bool   `json:"_paged"`
			Ref   string `json:"ref"`
			Size  int    `json:"size"`
			Turns int    `json:"turns"`
		}
		if json.Unmarshal([]byte(content), &m) == nil && m.Paged {
			return m.Ref, m.Size, m.Turns, true
		}
	}
	return "", 0, 0, false
}
