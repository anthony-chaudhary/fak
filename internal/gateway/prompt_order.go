package gateway

import (
	"bytes"
	"encoding/json"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

var (
	// reCompactionSummary matches <summary>...</summary> or <compaction_summary>...</compaction_summary> blocks.
	reCompactionSummary = regexp.MustCompile(`(?is)<(?:summary|compaction_summary)>.*?</(?:summary|compaction_summary)>`)

	// reVolatileLine matches standalone lines carrying ephemeral or dynamic session metadata.
	reVolatileLine = regexp.MustCompile(`(?im)^\s*(?:Today(?:'s date)? is|Current (?:date|time)(?:\s+is|:)?|Date:|Timestamp:|Turn(?: count)?:|Current turn:|Session(?: ID)?:\s*[0-9a-zA-Z_-]+)[^\n]*$`)

	// reVolatileInline matches inline clauses or sentences carrying dynamic timestamps or session info.
	reVolatileInline = regexp.MustCompile(`(?i)\b(?:Today(?:'s date)? is|Current (?:date|time)(?: is|:)|Date:|Timestamp:|Turn(?: count)?:|Current turn:|Session(?: ID)?:\s*[0-9a-zA-Z_-]+)[^\n\.\;]+[\.\;]?`)
)

// CanonicalizeTools sorts responses tools canonically (alphabetically by function/tool name).
func CanonicalizeTools(tools []responsesTool) []responsesTool {
	if len(tools) <= 1 {
		return tools
	}
	sorted := make([]responsesTool, len(tools))
	copy(sorted, tools)
	sort.SliceStable(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})
	return sorted
}

// ExtractCompactionSummary finds and extracts any <summary> or <compaction_summary> block.
func ExtractCompactionSummary(s string) (string, string) {
	loc := reCompactionSummary.FindStringIndex(s)
	if loc == nil {
		return s, ""
	}
	summary := strings.TrimSpace(s[loc[0]:loc[1]])
	cleaned := strings.TrimSpace(s[:loc[0]] + "\n" + s[loc[1]:])
	cleaned = cleanWhitespace(cleaned)
	return cleaned, summary
}

// ExtractVolatileMetadata extracts ephemeral session metadata (timestamps, turn counters, session IDs)
// from standing instructions or system prompt text, returning the cleaned text and extracted items.
func ExtractVolatileMetadata(s string) (string, []string) {
	var extracted []string

	lines := strings.Split(s, "\n")
	var remainingLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			remainingLines = append(remainingLines, line)
			continue
		}
		if reVolatileLine.MatchString(trimmed) {
			extracted = append(extracted, trimmed)
			continue
		}
		inlineMatches := reVolatileInline.FindAllString(trimmed, -1)
		if len(inlineMatches) > 0 {
			cleanedLine := trimmed
			for _, m := range inlineMatches {
				mTrim := strings.TrimSpace(m)
				if mTrim != "" {
					extracted = append(extracted, mTrim)
					cleanedLine = strings.Replace(cleanedLine, m, "", 1)
				}
			}
			cleanedLine = strings.TrimSpace(cleanedLine)
			if cleanedLine != "" {
				remainingLines = append(remainingLines, cleanedLine)
			}
		} else {
			remainingLines = append(remainingLines, line)
		}
	}

	cleaned := cleanWhitespace(strings.Join(remainingLines, "\n"))
	return cleaned, extracted
}

func cleanWhitespace(s string) string {
	for strings.Contains(s, "\n\n\n") {
		s = strings.ReplaceAll(s, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(s)
}

// RenderInvariantPrefix serializes Tier 0 (invariant standing instructions + canonically sorted tools)
// into a deterministic byte stream starting from Byte Offset 0.
func RenderInvariantPrefix(instructions string, tools []responsesTool) []byte {
	var buf bytes.Buffer
	buf.WriteString("instructions:\n")
	buf.WriteString(instructions)
	buf.WriteString("\ntools:\n")
	sorted := CanonicalizeTools(tools)
	for i, t := range sorted {
		if i > 0 {
			buf.WriteString("\n---\n")
		}
		raw, err := json.Marshal(t)
		if err == nil {
			buf.Write(raw)
		} else {
			buf.WriteString(t.Name)
		}
	}
	return buf.Bytes()
}

// EstimateTokens provides a token estimate using the standard ~4 chars/token heuristic.
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return (len(text) + 3) / 4
}

// EstimateTokensBytes provides a token estimate for raw byte slices.
func EstimateTokensBytes(b []byte) int {
	if len(b) == 0 {
		return 0
	}
	return (len(b) + 3) / 4
}

// ComputePrefixBlockAlignment computes the proportion of 1024-token blocks in post-compaction
// prefix that align with 1024-token block boundaries of pre-compaction prefix.
func ComputePrefixBlockAlignment(preCompactionPrefix, postCompactionPrefix []byte) float64 {
	tokPost := EstimateTokensBytes(postCompactionPrefix)
	if tokPost == 0 {
		return 0.0
	}
	if !bytes.Equal(preCompactionPrefix, postCompactionPrefix) {
		// If prefixes differ at byte offset 0, find length of common prefix
		matchingBytes := 0
		limit := len(preCompactionPrefix)
		if len(postCompactionPrefix) < limit {
			limit = len(postCompactionPrefix)
		}
		for i := 0; i < limit; i++ {
			if preCompactionPrefix[i] != postCompactionPrefix[i] {
				break
			}
			matchingBytes++
		}
		matchingTokens := EstimateTokensBytes(preCompactionPrefix[:matchingBytes])
		alignedBlocks := matchingTokens / 1024
		return float64(alignedBlocks*1024) / float64(tokPost)
	}
	if tokPost >= 1024 {
		aligned := (tokPost / 1024) * 1024
		return float64(aligned) / float64(tokPost)
	}
	return 1.0
}

// CanonicalizeResponsesPrefix stabilizes the prefix for Responses requests across compaction boundaries:
// Tier 0 (Invariant Prefix at Byte Offset 0):
//   - Base developer / system instructions (stripped of volatile metadata and compaction summary)
//   - Alphabetically sorted canonical tool definitions
//
// Tier 1 (Pruned History & Summary):
//   - Post-compaction summary block anchored strictly after Tier 0 as root of pruned history
//   - Preserved recent turn sequence
//   - Volatile session metadata relocated to suffix of latest user message
//
// Returns (stabilizedInstructions, sortedTools, stabilizedMessages, prefixReuseRatio).
func CanonicalizeResponsesPrefix(
	instructions string,
	tools []responsesTool,
	messages []agent.Message,
) (string, []responsesTool, []agent.Message, float64) {
	sortedTools := CanonicalizeTools(tools)

	var allVolatile []string
	var summaryBlock string

	cleanedInstructions := instructions
	if cleanedInstructions != "" {
		var extractedV []string
		cleanedInstructions, summaryBlock = ExtractCompactionSummary(cleanedInstructions)
		cleanedInstructions, extractedV = ExtractVolatileMetadata(cleanedInstructions)
		allVolatile = append(allVolatile, extractedV...)
	}

	outMessages := make([]agent.Message, 0, len(messages)+2)
	hasSystemLeading := len(messages) > 0 && (messages[0].Role == agent.RoleSystem || messages[0].Role == "developer")
	startIdx := 0

	if hasSystemLeading {
		sysContent := messages[0].Content
		if summaryBlock == "" {
			sysContent, summaryBlock = ExtractCompactionSummary(sysContent)
		}
		var extractedV []string
		sysContent, extractedV = ExtractVolatileMetadata(sysContent)
		allVolatile = append(allVolatile, extractedV...)

		if cleanedInstructions == "" {
			cleanedInstructions = sysContent
		} else {
			sysContent = cleanedInstructions
		}
		outMessages = append(outMessages, agent.Message{
			Role:    messages[0].Role,
			Content: sysContent,
		})
		startIdx = 1
	} else if cleanedInstructions != "" {
		outMessages = append(outMessages, agent.Message{
			Role:    agent.RoleSystem,
			Content: cleanedInstructions,
		})
	}

	for i := startIdx; i < len(messages); i++ {
		msg := messages[i]
		if summaryBlock == "" && reCompactionSummary.MatchString(msg.Content) {
			cleaned, s := ExtractCompactionSummary(msg.Content)
			summaryBlock = s
			if cleaned == "" {
				continue
			}
			msg.Content = cleaned
		}
		outMessages = append(outMessages, msg)
	}

	if summaryBlock != "" {
		summaryMsg := agent.Message{
			Role:    agent.RoleSystem,
			Content: summaryBlock,
		}
		if len(outMessages) > 0 && (outMessages[0].Role == agent.RoleSystem || outMessages[0].Role == "developer") {
			head := outMessages[:1]
			tail := append([]agent.Message{summaryMsg}, outMessages[1:]...)
			outMessages = append(head, tail...)
		} else {
			outMessages = append([]agent.Message{summaryMsg}, outMessages...)
		}
	}

	if len(allVolatile) > 0 {
		volatileText := strings.Join(allVolatile, "\n")
		lastUserIdx := -1
		for i := len(outMessages) - 1; i >= 0; i-- {
			if outMessages[i].Role == agent.RoleUser {
				lastUserIdx = i
				break
			}
		}
		if lastUserIdx >= 0 {
			if strings.TrimSpace(outMessages[lastUserIdx].Content) == "" {
				outMessages[lastUserIdx].Content = volatileText
			} else {
				outMessages[lastUserIdx].Content = strings.TrimRight(outMessages[lastUserIdx].Content, "\r\n") + "\n\n" + volatileText
			}
		} else {
			outMessages = append(outMessages, agent.Message{
				Role:    agent.RoleUser,
				Content: volatileText,
			})
		}
	}

	prefixBytes := RenderInvariantPrefix(cleanedInstructions, sortedTools)
	prefixTokens := EstimateTokensBytes(prefixBytes)

	var reuseRatio float64
	if prefixTokens >= 1024 {
		aligned := (prefixTokens / 1024) * 1024
		reuseRatio = float64(aligned) / float64(prefixTokens)
	} else if prefixTokens > 0 {
		totalTokens := prefixTokens
		for _, m := range outMessages {
			totalTokens += EstimateTokens(m.Content)
		}
		if totalTokens > 0 {
			reuseRatio = float64(prefixTokens) / float64(totalTokens)
		}
	}

	return cleanedInstructions, sortedTools, outMessages, reuseRatio
}
