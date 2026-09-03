package ctxmmu

import (
	"strings"
	"unicode"
)

// TurnRecord represents a single turn in an agent interaction trajectory.
type TurnRecord struct {
	Role         string `json:"role"`
	Content      string `json:"content"`
	ToolCallName string `json:"tool_call_name,omitempty"`
	ToolCallArgs string `json:"tool_call_args,omitempty"`
	IsFailure    bool   `json:"is_failure,omitempty"`
	Affordance   string `json:"affordance,omitempty"`
	VerifiedFact string `json:"verified_fact,omitempty"`
}

// PositiveCompactedHistory represents the compacted trajectory retaining positive residue.
type PositiveCompactedHistory struct {
	OriginalGoal             string       `json:"original_goal"`
	RetainedTurns            []TurnRecord `json:"retained_turns"`
	VerifiedFacts            []string     `json:"verified_facts"`
	LatestAffordance         string       `json:"latest_affordance"`
	ShedTurnsCount           int          `json:"shed_turns_count"`
	TotalTokensSavedEstimate int          `json:"total_tokens_saved_estimate"`
}

var apologyPhrases = []string{
	"apolog",
	"sorry",
	"my mistake",
	"that's my mistake",
	"that was my mistake",
	"my error",
	"that's my error",
	"that was my error",
	"i was wrong",
	"excuse the error",
	"excuse my error",
	"excuse this error",
	"my fault",
	"my bad",
	"pardon",
	"made a mistake",
	"made an error",
	"stand corrected",
	"stood corrected",
}

var refusalTokens = []string{
	"policy_block",
	"policy_deny",
	"arbiter_refuse",
	"lane_drained",
	"claim_unwitnessed",
	"permission denied",
	"access denied",
	"command failed",
	"execution failed",
	"tool execution failed",
	"action refused",
}

// CompactPositiveState sheds rejected tool call attempts, error banners, and apology
// clutter while preserving the original goal, verified facts, and the latest active affordance.
func CompactPositiveState(turns []TurnRecord, originalGoal string) *PositiveCompactedHistory {
	goal := strings.TrimSpace(originalGoal)
	if goal == "" {
		for _, t := range turns {
			if strings.EqualFold(t.Role, "user") && strings.TrimSpace(t.Content) != "" {
				goal = strings.TrimSpace(t.Content)
				break
			}
		}
	}

	facts := make([]string, 0)
	seenFacts := make(map[string]bool)
	for _, t := range turns {
		fact := strings.TrimSpace(t.VerifiedFact)
		if fact != "" && !seenFacts[fact] {
			seenFacts[fact] = true
			facts = append(facts, fact)
		}
	}

	var latestAffordance string
	for _, t := range turns {
		if aff := strings.TrimSpace(t.Affordance); aff != "" {
			latestAffordance = aff
		}
	}

	n := len(turns)
	if n == 0 {
		return &PositiveCompactedHistory{
			OriginalGoal:             goal,
			RetainedTurns:            make([]TurnRecord, 0),
			VerifiedFacts:            facts,
			LatestAffordance:         latestAffordance,
			ShedTurnsCount:           0,
			TotalTokensSavedEstimate: 0,
		}
	}

	shed := make([]bool, n)

	// Step 1: Flag explicit failures, tool failures, and pure apology/error turns.
	for i := 0; i < n; i++ {
		t := turns[i]
		if t.IsFailure || isToolFailure(t) || isPureErrorOrApology(t) {
			shed[i] = true
		}
	}

	// Step 2: Correlate assistant tool calls with tool results.
	// A tool call attempt that failed or was refused causes both the call and result to be shed.
	changed := true
	for changed {
		changed = false
		for i := 0; i < n; i++ {
			if !strings.EqualFold(turns[i].Role, "assistant") || turns[i].ToolCallName == "" {
				continue
			}

			// Find corresponding tool result.
			for j := i + 1; j < n; j++ {
				if strings.EqualFold(turns[j].Role, "tool") {
					if turns[j].ToolCallName != "" && turns[j].ToolCallName != turns[i].ToolCallName {
						continue
					}
					// If the tool result failed, the assistant tool call is a rejected attempt.
					if shed[j] && !shed[i] {
						shed[i] = true
						changed = true
					}
					// If the assistant call failed, the tool result is shed.
					if shed[i] && !shed[j] {
						shed[j] = true
						changed = true
					}
					break
				}
				if strings.EqualFold(turns[j].Role, "assistant") || strings.EqualFold(turns[j].Role, "user") {
					break
				}
			}
		}
	}

	// Step 3: Check remaining non-user turns for content that degrades to empty after cleaning.
	for i := 0; i < n; i++ {
		if shed[i] || strings.EqualFold(turns[i].Role, "user") {
			continue
		}
		cleaned := cleanNegativeText(turns[i].Content)
		if strings.TrimSpace(cleaned) == "" && turns[i].ToolCallName == "" {
			shed[i] = true
		}
	}

	// Step 4: Build retained turns and compute token savings.
	retained := make([]TurnRecord, 0, n)
	shedCount := 0
	tokensSaved := 0

	for i := 0; i < n; i++ {
		t := turns[i]
		if shed[i] {
			shedCount++
			tokensSaved += estimateTurnTokens(t)
			continue
		}

		if !strings.EqualFold(t.Role, "user") {
			cleaned := cleanNegativeText(t.Content)
			if cleaned != t.Content {
				origTok := EstimateTokens([]byte(t.Content))
				cleanTok := EstimateTokens([]byte(cleaned))
				if origTok > cleanTok {
					tokensSaved += (origTok - cleanTok)
				}
				t.Content = cleaned
			}
		}
		retained = append(retained, t)
	}

	return &PositiveCompactedHistory{
		OriginalGoal:             goal,
		RetainedTurns:            retained,
		VerifiedFacts:            facts,
		LatestAffordance:         latestAffordance,
		ShedTurnsCount:           shedCount,
		TotalTokensSavedEstimate: tokensSaved,
	}
}

func estimateTurnTokens(t TurnRecord) int {
	tok := EstimateTokens([]byte(t.Content))
	if t.ToolCallName != "" {
		tok += EstimateTokens([]byte(t.ToolCallName))
	}
	if t.ToolCallArgs != "" {
		tok += EstimateTokens([]byte(t.ToolCallArgs))
	}
	return tok
}

func isToolFailure(t TurnRecord) bool {
	if t.IsFailure {
		return true
	}
	if !strings.EqualFold(t.Role, "tool") {
		return false
	}
	trimmed := strings.TrimSpace(t.Content)
	if trimmed == "" {
		return false
	}
	if isErrorBannerLine(trimmed) {
		return true
	}
	lower := strings.ToLower(trimmed)
	for _, token := range refusalTokens {
		if strings.Contains(lower, token) {
			return true
		}
	}
	return false
}

func isPureErrorOrApology(t TurnRecord) bool {
	if strings.EqualFold(t.Role, "user") {
		return false
	}
	if t.ToolCallName != "" {
		return false
	}
	trimmed := strings.TrimSpace(t.Content)
	if trimmed == "" {
		return false
	}
	cleaned := cleanNegativeText(trimmed)
	return strings.TrimSpace(cleaned) == ""
}

func isErrorBannerLine(line string) bool {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return false
	}
	lower := strings.ToLower(trimmed)

	prefixes := []string{
		"[error",
		"error:",
		"[refusal",
		"refusal:",
		"[failed",
		"failed:",
		"[fatal",
		"fatal:",
		"panic:",
		"traceback (most recent call last):",
		"policy_block",
		"policy_deny",
		"arbiter_refuse",
		"lane_drained",
		"claim_unwitnessed",
		"permission denied",
		"access denied",
		"command failed",
		"execution failed",
		"exit status ",
		"exit code ",
	}
	for _, p := range prefixes {
		if strings.HasPrefix(lower, p) {
			return true
		}
	}

	if strings.HasPrefix(trimmed, "===") || strings.HasPrefix(trimmed, "---") || strings.HasPrefix(trimmed, "***") {
		if strings.Contains(lower, "error") || strings.Contains(lower, "fail") || strings.Contains(lower, "refus") || strings.Contains(lower, "deny") || strings.Contains(lower, "end") || strings.Contains(lower, "banner") {
			return true
		}
	}

	return false
}

func hasApologyPhrase(text string) bool {
	lower := strings.ToLower(text)
	for _, p := range apologyPhrases {
		if strings.Contains(lower, p) {
			return true
		}
	}
	return false
}

func cleanNegativeText(text string) string {
	if text == "" {
		return ""
	}
	lines := strings.Split(text, "\n")
	var keptLines []string

	for _, line := range lines {
		if isErrorBannerLine(line) {
			continue
		}
		cleanedLine := cleanApologiesFromLine(line)
		if strings.TrimSpace(cleanedLine) != "" {
			keptLines = append(keptLines, cleanedLine)
		}
	}

	return strings.Join(keptLines, "\n")
}

func cleanApologiesFromLine(line string) string {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return ""
	}
	if !hasApologyPhrase(trimmed) {
		return trimmed
	}

	sentences := splitSentences(trimmed)
	var kept []string
	for _, s := range sentences {
		cleaned := cleanSentenceApology(s)
		if strings.TrimSpace(cleaned) != "" {
			kept = append(kept, cleaned)
		}
	}
	return strings.Join(kept, " ")
}

func cleanSentenceApology(sentence string) string {
	trimmed := strings.TrimSpace(sentence)
	if trimmed == "" {
		return ""
	}
	if !hasApologyPhrase(trimmed) {
		return trimmed
	}

	// If there are comma-separated clauses, attempt to strip leading apology clauses.
	commaIdx := strings.Index(trimmed, ",")
	if commaIdx != -1 {
		prefix := strings.TrimSpace(trimmed[:commaIdx])
		suffix := strings.TrimSpace(trimmed[commaIdx+1:])
		if hasApologyPhrase(prefix) {
			cleanedSuffix := cleanSentenceApology(suffix)
			if cleanedSuffix != "" {
				runes := []rune(cleanedSuffix)
				if len(runes) > 0 && unicode.IsLower(runes[0]) {
					runes[0] = unicode.ToUpper(runes[0])
					cleanedSuffix = string(runes)
				}
				return cleanedSuffix
			}
			return ""
		}
	}

	return ""
}

func splitSentences(line string) []string {
	var sentences []string
	var current strings.Builder
	runes := []rune(line)
	n := len(runes)

	for i := 0; i < n; i++ {
		r := runes[i]
		current.WriteRune(r)
		if r == '.' || r == '!' || r == '?' {
			if i+1 == n || unicode.IsSpace(runes[i+1]) {
				s := strings.TrimSpace(current.String())
				if s != "" {
					sentences = append(sentences, s)
				}
				current.Reset()
			}
		}
	}

	if rem := strings.TrimSpace(current.String()); rem != "" {
		sentences = append(sentences, rem)
	}
	return sentences
}
