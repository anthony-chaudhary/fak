package trajectory

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	degenerateCycleMinSpanTokens   = 16
	degenerateCycleMinRepeatCount  = 4
	degenerateCycleMinOverlapRatio = 0.5
	degenerateCycleMaxExemplars    = 16
)

// AuditDegenerateCycle records one verbatim repetition cycle detected inside
// an assistant turn's reasoning or content channel.
type AuditDegenerateCycle struct {
	TurnIndex   int    `json:"turn_index"`
	Channel     string `json:"channel"`
	SpanTokens  int    `json:"span_tokens"`
	RepeatCount int    `json:"repeat_count"`
	FirstOffset int    `json:"first_offset"`
}

type cycleToken struct {
	text   string
	offset int
}

// tokenizeForCycle splits text on whitespace and breaks long non-whitespace
// runs into 4-byte UTF-8 chunks so both spaced prose and spaceless strings
// map consistently to the >=16-token / >=64-char cycle threshold.
func tokenizeForCycle(text string) []cycleToken {
	if text == "" {
		return nil
	}
	var out []cycleToken
	i := 0
	for i < len(text) {
		r, size := utf8.DecodeRuneInString(text[i:])
		if unicode.IsSpace(r) {
			i += size
			continue
		}
		wordStart := i
		for i < len(text) {
			r2, s2 := utf8.DecodeRuneInString(text[i:])
			if unicode.IsSpace(r2) {
				break
			}
			i += s2
		}
		word := text[wordStart:i]
		if len(word) <= 4 {
			out = append(out, cycleToken{text: word, offset: wordStart})
			continue
		}
		for pos := 0; pos < len(word); {
			end := pos + 4
			if end >= len(word) {
				end = len(word)
			} else {
				for end < len(word) && !utf8.RuneStart(word[end]) {
					end++
				}
			}
			out = append(out, cycleToken{text: word[pos:end], offset: wordStart + pos})
			pos = end
		}
	}
	return out
}

func fnv1aTokens(tokens []cycleToken) uint64 {
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	for _, tok := range tokens {
		for i := 0; i < len(tok.text); i++ {
			h ^= uint64(tok.text[i])
			h *= prime64
		}
		h ^= 0xff
		h *= prime64
	}
	return h
}

func tokenSlicesEqual(a, b []cycleToken) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].text != b[i].text {
			return false
		}
	}
	return true
}

// DetectDegenerateCycle scans a single assistant channel ("reasoning" or
// "content") for any contiguous span of >= 16 tokens repeating >= 4 consecutive
// times verbatim, or >= 4 non-overlapping exact occurrences covering > 50% of
// the channel's tokens.
func DetectDegenerateCycle(turnIndex int, channel, text string) (AuditDegenerateCycle, bool) {
	tokens := tokenizeForCycle(text)
	n := len(tokens)
	if n < degenerateCycleMinSpanTokens*degenerateCycleMinRepeatCount {
		return AuditDegenerateCycle{}, false
	}

	positionsByHash := make(map[uint64][]int, n)
	for i := 0; i <= n-degenerateCycleMinSpanTokens; i++ {
		h := fnv1aTokens(tokens[i : i+degenerateCycleMinSpanTokens])
		positionsByHash[h] = append(positionsByHash[h], i)
	}

	// Pass 1: consecutive verbatim repeats (>= 4x).
	for i := 0; i <= n-degenerateCycleMinSpanTokens*degenerateCycleMinRepeatCount; i++ {
		h := fnv1aTokens(tokens[i : i+degenerateCycleMinSpanTokens])
		positions := positionsByHash[h]
		if len(positions) < degenerateCycleMinRepeatCount {
			continue
		}
		for _, j := range positions {
			if j <= i {
				continue
			}
			span := j - i
			if span < degenerateCycleMinSpanTokens || i+span*degenerateCycleMinRepeatCount > n {
				continue
			}
			base := tokens[i : i+span]
			repeats := 1
			for pos := i + span; pos+span <= n; pos += span {
				if !tokenSlicesEqual(base, tokens[pos:pos+span]) {
					break
				}
				repeats++
			}
			if repeats >= degenerateCycleMinRepeatCount {
				return AuditDegenerateCycle{
					TurnIndex:   turnIndex,
					Channel:     channel,
					SpanTokens:  span,
					RepeatCount: repeats,
					FirstOffset: tokens[i].offset,
				}, true
			}
		}
	}

	// Pass 2: non-consecutive exact occurrences (>= 4x) with coverage > 0.5.
	for i := 0; i <= n-degenerateCycleMinSpanTokens; i++ {
		h := fnv1aTokens(tokens[i : i+degenerateCycleMinSpanTokens])
		positions := positionsByHash[h]
		if len(positions) < degenerateCycleMinRepeatCount {
			continue
		}
		maxSpan := n / degenerateCycleMinRepeatCount
		for span := degenerateCycleMinSpanTokens; span <= maxSpan; span++ {
			if i+span > n {
				break
			}
			base := tokens[i : i+span]
			count := 1
			nextAllowed := i + span
			for _, pos := range positions {
				if pos < nextAllowed || pos+span > n {
					continue
				}
				if tokenSlicesEqual(base, tokens[pos:pos+span]) {
					count++
					nextAllowed = pos + span
				}
			}
			if count >= degenerateCycleMinRepeatCount && float64(count*span)/float64(n) > degenerateCycleMinOverlapRatio {
				return AuditDegenerateCycle{
					TurnIndex:   turnIndex,
					Channel:     channel,
					SpanTokens:  span,
					RepeatCount: count,
					FirstOffset: tokens[i].offset,
				}, true
			}
		}
	}

	return AuditDegenerateCycle{}, false
}

// ScanDegenerateCycles checks both reasoning and content channels for one
// assistant turn and returns any detected cycles.
func ScanDegenerateCycles(turnIndex int, reasoning, content string) []AuditDegenerateCycle {
	var cycles []AuditDegenerateCycle
	if c, ok := DetectDegenerateCycle(turnIndex, "reasoning", reasoning); ok {
		cycles = append(cycles, c)
	}
	if c, ok := DetectDegenerateCycle(turnIndex, "content", content); ok {
		cycles = append(cycles, c)
	}
	return cycles
}

func observeAuditAssistantTurn(reasoning, content string, state *auditParseState) {
	observeAuditAssistantTurnWithID("", reasoning, content, state)
}

func observeAuditAssistantTurnWithID(turnID, reasoning, content string, state *auditParseState) {
	if strings.TrimSpace(reasoning) == "" && strings.TrimSpace(content) == "" {
		return
	}
	var turnIndex int
	if turnID != "" {
		if state.assistantTurnByID == nil {
			state.assistantTurnByID = map[string]int{}
		}
		if existing, ok := state.assistantTurnByID[turnID]; ok {
			turnIndex = existing
		} else {
			turnIndex = state.assistantTurnIndex
			state.assistantTurnIndex++
			state.assistantTurnByID[turnID] = turnIndex
		}
	} else {
		turnIndex = state.assistantTurnIndex
		state.assistantTurnIndex++
	}
	cycles := ScanDegenerateCycles(turnIndex, reasoning, content)
	if len(cycles) == 0 {
		return
	}
	if state.seenCycleKeys == nil {
		state.seenCycleKeys = map[string]struct{}{}
	}
	for _, c := range cycles {
		key := fmt.Sprintf("%d:%s", c.TurnIndex, c.Channel)
		if _, seen := state.seenCycleKeys[key]; seen {
			continue
		}
		state.seenCycleKeys[key] = struct{}{}
		state.degenerateCyclesCount++
		if len(state.degenerateCycles) < degenerateCycleMaxExemplars {
			state.degenerateCycles = append(state.degenerateCycles, c)
		}
	}
}

func extractClaudeAssistantChannels(content any) (reasoning, text string) {
	switch v := content.(type) {
	case string:
		return "", v
	case []any:
		var rBuf, tBuf strings.Builder
		for _, raw := range v {
			block, _ := raw.(map[string]any)
			if block == nil {
				continue
			}
			switch block["type"] {
			case "thinking", "reasoning":
				s, _ := block["thinking"].(string)
				if s == "" {
					s, _ = block["text"].(string)
				}
				if s != "" {
					if rBuf.Len() > 0 {
						rBuf.WriteByte('\n')
					}
					rBuf.WriteString(s)
				}
			case "text":
				if s, ok := block["text"].(string); ok && s != "" {
					if tBuf.Len() > 0 {
						tBuf.WriteByte('\n')
					}
					tBuf.WriteString(s)
				}
			}
		}
		return rBuf.String(), tBuf.String()
	}
	return "", ""
}
