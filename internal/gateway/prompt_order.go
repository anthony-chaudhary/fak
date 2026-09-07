package gateway

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

var (
	// reCompactionSummary matches <summary>...</summary> or <compaction_summary>...</compaction_summary> blocks.
	reCompactionSummary = regexp.MustCompile(`(?is)<(?:summary|compaction_summary)>.*?</(?:summary|compaction_summary)>`)

	// reVolatileLine matches standalone lines carrying ephemeral or dynamic session metadata.
	reVolatileLine = regexp.MustCompile(`(?im)^\s*(?:Today(?:'s date)?(?:\s+is)?|Current (?:date|time)(?:\s+is|:)?|Date:|Timestamp:|Turn(?: count)?:|Current turn:|Session(?: ID)?:\s*[0-9a-zA-Z_-]+|PID:|Process(?: ID)?:\s*\d+)[^\n]*$`)

	// reVolatileInline matches inline clauses or sentences carrying dynamic timestamps or session info.
	reVolatileInline = regexp.MustCompile(`(?i)\b(?:Today(?:'s date)?(?:\s+is)?|Current (?:date|time)(?: is|:)|Date:|Timestamp:|Turn(?: count)?:|Current turn:|Session(?: ID)?:\s*[0-9a-zA-Z_-]+|PID:|Process(?: ID)?:\s*\d+)[^\n\.\;]+[\.\;]?`)

	// reIsoTimestampLine matches standalone lines containing an ISO 8601 timestamp.
	reIsoTimestampLine = regexp.MustCompile(`(?im)^\s*\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\s*$`)

	// reIsoTimestampInline matches inline ISO 8601 timestamps.
	reIsoTimestampInline = regexp.MustCompile(`\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b`)

	// reTimestampLine matches standalone lines carrying timestamps or dates.
	reTimestampLine = regexp.MustCompile(`(?im)^\s*(?:Today(?:'s date)?(?:\s+is)?|Current (?:date|time)(?:\s+is|:)?|Date:|Timestamp:)[^\n]*$`)

	// reTimestampInline matches inline timestamp clauses.
	reTimestampInline = regexp.MustCompile(`(?i)\b(?:Today(?:'s date)?(?:\s+is)?|Current (?:date|time)(?: is|:)|Date:|Timestamp:)[^\n\.\;]+[\.\;]?`)

	// reSessionLine matches standalone lines carrying session, trace, run IDs or turn counters.
	reSessionLine = regexp.MustCompile(`(?im)^\s*(?:Session(?: ID)?|Trace(?: ID)?|Run(?: ID)?|Turn(?: count)?|Current turn|PID|Process(?: ID)?):\s*[0-9a-zA-Z_\-\.]+[^\n]*$`)

	// reSessionInline matches inline session clauses.
	reSessionInline = regexp.MustCompile(`(?i)\b(?:Session(?: ID)?|Trace(?: ID)?|Run(?: ID)?|Turn(?: count)?|Current turn|PID|Process(?: ID)?):\s*[0-9a-zA-Z_\-\.]+[\.\;]?`)

	// reWorkingDirLine matches standalone lines carrying working directory or workspace root.
	reWorkingDirLine = regexp.MustCompile(`(?im)^\s*(?:Working directory|Current directory|Workspace root|CWD):\s*[^\n]+$`)

	// reWorkingDirInline matches inline working directory clauses.
	reWorkingDirInline = regexp.MustCompile(`(?i)\b(?:Working directory|Current directory|Workspace root|CWD):\s*[^\n\.\;]+[\.\;]?`)
)

// CanonicalizeToolDeclarations deterministically sorts tool descriptors alphabetically by name.
func CanonicalizeToolDeclarations[T any](tools []T, getName func(T) string) []T {
	if len(tools) <= 1 {
		return tools
	}
	sorted := make([]T, len(tools))
	copy(sorted, tools)
	sort.SliceStable(sorted, func(i, j int) bool {
		if getName == nil {
			return false
		}
		return getName(sorted[i]) < getName(sorted[j])
	})
	return sorted
}

// CanonicalizeTools sorts responses tools canonically (alphabetically by function/tool name).
func CanonicalizeTools(tools []responsesTool) []responsesTool {
	return CanonicalizeToolDeclarations(tools, func(t responsesTool) string {
		return t.Name
	})
}

// CanonicalizeToolDefs sorts agent tool definitions deterministically (alphabetically by function name).
func CanonicalizeToolDefs(tools []agent.ToolDef) []agent.ToolDef {
	return CanonicalizeToolDeclarations(tools, func(t agent.ToolDef) string {
		return t.Function.Name
	})
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
		if reVolatileLine.MatchString(trimmed) || reIsoTimestampLine.MatchString(trimmed) {
			extracted = append(extracted, trimmed)
			continue
		}
		inlineMatches := reVolatileInline.FindAllString(trimmed, -1)
		inlineIso := reIsoTimestampInline.FindAllString(trimmed, -1)
		allInline := append(inlineMatches, inlineIso...)
		if len(allInline) > 0 {
			cleanedLine := trimmed
			for _, m := range allInline {
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

// PromptTier represents the canonical prompt prefix tier for prefix cache optimization.
type PromptTier int

const (
	TierUnknown             PromptTier = 0
	Tier1SystemInstructions            = 1 // Static persona, instructions
	Tier2ToolDeclarations              = 2 // Static tool schemas, alphabetically sorted
	Tier3ProjectRules                  = 3 // Static workspace / repository guidelines
	Tier4RepoContext                   = 4 // Static code skeleton, file trees
	Tier5DynamicTurns                  = 5 // Ephemeral conversation turns, user queries

	// Backward-compatibility aliases for 4-tier and 5-tier references
	Tier3DurableContext   PromptTier = Tier3ProjectRules
	Tier4WorkspaceContext PromptTier = Tier4RepoContext
	Tier4DynamicHistory   PromptTier = Tier5DynamicTurns
	Tier5DynamicHistory   PromptTier = Tier5DynamicTurns
)

// String returns the canonical name of the prompt tier.
func (t PromptTier) String() string {
	switch t {
	case Tier1SystemInstructions:
		return "Tier1SystemInstructions"
	case Tier2ToolDeclarations:
		return "Tier2ToolDeclarations"
	case Tier3ProjectRules:
		return "Tier3ProjectRules"
	case Tier4RepoContext:
		return "Tier4RepoContext"
	case Tier5DynamicTurns:
		return "Tier5DynamicTurns"
	default:
		return "TierUnknown"
	}
}

// MarshalText implements encoding.TextMarshaler for PromptTier.
func (t PromptTier) MarshalText() ([]byte, error) {
	return []byte(t.String()), nil
}

// CanonicalizeReport contains metrics from prompt canonicalization.
type CanonicalizeReport struct {
	OriginalCount         int         `json:"original_count"`
	CanonicalCount        int         `json:"canonical_count"`
	TierCounts            map[int]int `json:"tier_counts"`
	StablePrefixHash      string      `json:"stable_prefix_hash"`
	VolatileTokensHoisted int         `json:"volatile_tokens_hoisted"`
}

// CanonicalizeStats captures metrics from prompt canonicalization.
type CanonicalizeStats struct {
	TiersPresent            []PromptTier `json:"tiers_present"`
	ToolsSorted             int          `json:"tools_sorted"`
	VolatileElementsHoisted int          `json:"volatile_elements_hoisted"`
	PrefixTokensEstimate    int          `json:"prefix_tokens_estimate"`
	StablePrefixHash        string       `json:"stable_prefix_hash"`
	CacheBreakpointIndex    int          `json:"cache_breakpoint_index"`
	CacheBreakpointTier     PromptTier   `json:"cache_breakpoint_tier"`
}

// DeterministicPromptCanonicalizer reorganizes prompt messages and tool definitions into
// canonical tiers (Tiers 1-4 prefix, Tier 5 dynamic history) to maximize prefix cache reuse.
type DeterministicPromptCanonicalizer struct {
	StripVolatileEphemera bool
	NormalizeTools        bool

	StripTimestamps        bool
	StripSessionIDs        bool
	EnforceBlockAlignment  bool
	InsertCacheBreakpoint  bool
	DisableCacheBreakpoint bool
}

// DefaultDeterministicPromptCanonicalizer returns a canonicalizer with default prefix caching configuration.
func DefaultDeterministicPromptCanonicalizer() DeterministicPromptCanonicalizer {
	return DeterministicPromptCanonicalizer{
		StripVolatileEphemera: true,
		NormalizeTools:        true,
		StripTimestamps:       true,
		StripSessionIDs:       true,
		InsertCacheBreakpoint: true,
	}
}

func (c DeterministicPromptCanonicalizer) shouldInsertCacheBreakpoint() bool {
	if c.DisableCacheBreakpoint {
		return false
	}
	return true
}

// ClassifyPromptSection categorizes a prompt message into canonical tiers 1 to 5
// based on message role and content markers.
func ClassifyPromptSection(content string, role string) int {
	r := strings.ToLower(strings.TrimSpace(role))
	c := strings.ToLower(strings.TrimSpace(content))

	// Conversational turns (user queries, assistant replies, tool execution outputs) are strictly Tier 5.
	if r == "user" || r == "assistant" || r == "tool" || r == "function" {
		return int(Tier5DynamicTurns)
	}

	// Explicit tool declaration roles
	if r == "tools" || r == "tool_declarations" || r == "tool_schemas" {
		return int(Tier2ToolDeclarations)
	}

	// Tool declaration content prefixes
	if strings.HasPrefix(c, "<tools>") || strings.HasPrefix(c, "<tool_declarations>") ||
		strings.HasPrefix(c, "<tool_schemas>") || strings.HasPrefix(c, "available tools:") ||
		strings.HasPrefix(c, "# available tools") || strings.HasPrefix(c, "tool declarations:") ||
		strings.HasPrefix(c, "# tool declarations") || strings.HasPrefix(c, "tool definitions:") ||
		strings.HasPrefix(c, "# tool definitions") || strings.HasPrefix(c, "# tools") ||
		strings.HasPrefix(c, "tools:") || strings.HasPrefix(c, "tool schemas:") ||
		strings.HasPrefix(c, "# tool schemas") {
		return int(Tier2ToolDeclarations)
	}

	// Explicit workspace / repo skeleton roles -> Tier 4
	if r == "workspace_context" || r == "workspace" || r == "file_tree" ||
		r == "repo_skeleton" || r == "directory_index" || r == "repomap" || r == "context" ||
		r == "memory" || r == "memory_cards" || r == "memory_card" || r == "durable_context" ||
		r == "stable_context" || r == "documents" || r == "document" {
		return int(Tier4RepoContext)
	}

	// Explicit project rules roles -> Tier 3
	if r == "project_rules" || r == "rules" || r == "guidance" {
		return int(Tier3ProjectRules)
	}

	// Context messages with workspace / repo skeleton prefixes -> Tier 4
	if strings.HasPrefix(c, "<file_tree>") || strings.HasPrefix(c, "<workspace_context>") ||
		strings.HasPrefix(c, "<directory_index>") || strings.HasPrefix(c, "<repo_skeleton>") ||
		strings.HasPrefix(c, "<repomap>") || strings.HasPrefix(c, "<repo_map>") ||
		strings.HasPrefix(c, "<memory>") || strings.HasPrefix(c, "<memories>") ||
		strings.HasPrefix(c, "<memory_cards>") || strings.HasPrefix(c, "<memory_card>") ||
		strings.HasPrefix(c, "<durable_context>") || strings.HasPrefix(c, "<stable_context>") ||
		strings.HasPrefix(c, "<document>") || strings.HasPrefix(c, "<documents>") ||
		strings.HasPrefix(c, "<workspace>") || strings.HasPrefix(c, "<context>") {
		return int(Tier4RepoContext)
	}
	if strings.HasPrefix(c, "file tree") || strings.HasPrefix(c, "# file tree") ||
		strings.HasPrefix(c, "directory structure") || strings.HasPrefix(c, "# directory structure") ||
		strings.HasPrefix(c, "workspace context") || strings.HasPrefix(c, "# workspace context") ||
		strings.HasPrefix(c, "repository skeleton") || strings.HasPrefix(c, "# repository skeleton") ||
		strings.HasPrefix(c, "directory index") || strings.HasPrefix(c, "repo skeleton") ||
		strings.HasPrefix(c, "workspace:") || strings.HasPrefix(c, "repomap") ||
		strings.HasPrefix(c, "# repomap") || strings.HasPrefix(c, "## repomap") ||
		strings.HasPrefix(c, "repo map") || strings.HasPrefix(c, "memory cards") ||
		strings.HasPrefix(c, "# memory cards") || strings.HasPrefix(c, "# memory") ||
		strings.HasPrefix(c, "memory card") || strings.HasPrefix(c, "durable context") ||
		strings.HasPrefix(c, "# durable context") || strings.HasPrefix(c, "stable context") ||
		strings.HasPrefix(c, "# stable context") || strings.HasPrefix(c, "document context") ||
		strings.HasPrefix(c, "# document context") {
		return int(Tier4RepoContext)
	}

	// Project rules prefixes -> Tier 3
	if strings.HasPrefix(c, "<project_rules>") || strings.HasPrefix(c, "<rules>") || strings.HasPrefix(c, "<guidance>") ||
		strings.HasPrefix(c, "project rules") || strings.HasPrefix(c, "# project rules") ||
		strings.HasPrefix(c, "## project rules") || strings.HasPrefix(c, "repository rules") ||
		strings.HasPrefix(c, "# repository rules") || strings.HasPrefix(c, "coding guidelines") ||
		strings.HasPrefix(c, "repository guidelines") || strings.HasPrefix(c, "project guidelines") ||
		strings.HasPrefix(c, "agents.md") || strings.HasPrefix(c, "# agents.md") ||
		strings.HasPrefix(c, "claude.md") || strings.HasPrefix(c, "# claude.md") ||
		strings.HasPrefix(c, "standing rules") {
		return int(Tier3ProjectRules)
	}

	// Static persona / base instructions prefixes -> Tier 1
	isPersonaOrBaseInstructions := strings.HasPrefix(c, "you are ") ||
		strings.HasPrefix(c, "you are an ") ||
		strings.HasPrefix(c, "you are a ") ||
		strings.HasPrefix(c, "you're ") ||
		strings.HasPrefix(c, "<system>") ||
		strings.HasPrefix(c, "<system_instructions>") ||
		strings.HasPrefix(c, "<developer_instructions>") ||
		strings.HasPrefix(c, "# system instructions") ||
		strings.HasPrefix(c, "system instructions") ||
		strings.HasPrefix(c, "instructions:") ||
		strings.HasPrefix(c, "developer instructions")

	if isPersonaOrBaseInstructions {
		return int(Tier1SystemInstructions)
	}

	// System messages containing project rules markers -> Tier 3
	if strings.Contains(c, "project rules") || strings.Contains(c, "repository rules") ||
		strings.Contains(c, "coding guidelines") || strings.Contains(c, "repository guidelines") ||
		strings.Contains(c, "project guidelines") || strings.Contains(content, "AGENTS.md") ||
		strings.Contains(content, "CLAUDE.md") || strings.Contains(c, "rules:") ||
		strings.Contains(c, "# project rules") || strings.Contains(c, "## project rules") {
		return int(Tier3ProjectRules)
	}

	// System messages containing repo / workspace markers -> Tier 4
	if strings.Contains(c, "directory") || strings.Contains(c, "file tree") ||
		strings.Contains(c, "skeleton") || strings.Contains(c, "repomap") ||
		strings.Contains(c, "memory card") || strings.Contains(c, "memory cards") ||
		strings.Contains(c, "durable context") || strings.Contains(c, "stable context") ||
		strings.Contains(content, "├──") || strings.Contains(content, "└──") {
		return int(Tier4RepoContext)
	}

	// System messages containing "instructions" -> Tier 1
	if strings.Contains(c, "instructions") || strings.Contains(c, "persona") {
		return int(Tier1SystemInstructions)
	}

	// Default for system / developer role: Tier 1
	if r == "system" || r == "developer" {
		return int(Tier1SystemInstructions)
	}

	// Default for everything else: Tier 5
	return int(Tier5DynamicTurns)
}

// ClassifyMessage categorizes an agent message into its appropriate PromptTier.
func (c DeterministicPromptCanonicalizer) ClassifyMessage(msg agent.Message) PromptTier {
	r := strings.ToLower(strings.TrimSpace(msg.Role))
	if r == "user" || r == "assistant" || r == "tool" || r == "function" ||
		msg.ToolCallID != "" || len(msg.ToolCalls) > 0 || msg.FunctionCall != nil {
		return Tier5DynamicTurns
	}

	if isToolDeclarationsMessage(msg) {
		return Tier2ToolDeclarations
	}
	if isWorkspaceContextMessage(msg) || isRepomapOrMemoryContext(msg) {
		return Tier4RepoContext
	}
	if isProjectRulesMessage(msg) {
		return Tier3ProjectRules
	}
	return PromptTier(ClassifyPromptSection(msg.Content, msg.Role))
}

func isToolDeclarationsMessage(msg agent.Message) bool {
	r := strings.ToLower(strings.TrimSpace(msg.Role))
	if r == "user" || r == "assistant" || r == "tool" || r == "function" ||
		msg.ToolCallID != "" || len(msg.ToolCalls) > 0 || msg.FunctionCall != nil {
		return false
	}
	if r == "tools" || r == "tool_declarations" || r == "tool_schemas" {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(msg.Name))
	if name == "tools" || name == "tool_declarations" || name == "tool_schemas" {
		return true
	}
	content := strings.TrimSpace(msg.Content)
	lower := strings.ToLower(content)
	if strings.HasPrefix(lower, "<tools>") || strings.HasPrefix(lower, "<tool_declarations>") || strings.HasPrefix(lower, "<tool_schemas>") {
		return true
	}
	if strings.HasPrefix(lower, "available tools:") || strings.HasPrefix(lower, "# available tools") ||
		strings.HasPrefix(lower, "tool declarations:") || strings.HasPrefix(lower, "# tool declarations") ||
		strings.HasPrefix(lower, "tool definitions:") || strings.HasPrefix(lower, "# tools") ||
		strings.HasPrefix(lower, "tools:") || strings.HasPrefix(lower, "tool schemas:") || strings.HasPrefix(lower, "# tool schemas") {
		return true
	}
	return false
}

// isDurableContextMessage determines whether a message carries durable context
// (repomap, memory cards, project rules, stable document context, file tree).
func isDurableContextMessage(msg agent.Message) bool {
	return isProjectRulesMessage(msg) || isWorkspaceContextMessage(msg) || isRepomapOrMemoryContext(msg)
}

func isRepomapOrMemoryContext(msg agent.Message) bool {
	r := strings.ToLower(strings.TrimSpace(msg.Role))
	if r == "user" || r == "assistant" || r == "tool" || r == "function" ||
		msg.ToolCallID != "" || len(msg.ToolCalls) > 0 || msg.FunctionCall != nil {
		return false
	}
	if r == "repomap" || r == "memory" || r == "memory_cards" || r == "memory_card" ||
		r == "context" || r == "durable_context" || r == "stable_context" || r == "documents" || r == "document" {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(msg.Name))
	if name == "repomap" || name == "memory" || name == "memory_cards" || name == "memory_card" ||
		name == "context" || name == "durable_context" || name == "stable_context" || name == "documents" || name == "document" {
		return true
	}
	content := strings.TrimSpace(msg.Content)
	lower := strings.ToLower(content)
	if strings.HasPrefix(lower, "<repomap>") || strings.HasPrefix(lower, "<repo_map>") ||
		strings.HasPrefix(lower, "<memory>") || strings.HasPrefix(lower, "<memories>") ||
		strings.HasPrefix(lower, "<memory_cards>") || strings.HasPrefix(lower, "<memory_card>") ||
		strings.HasPrefix(lower, "<durable_context>") || strings.HasPrefix(lower, "<stable_context>") ||
		strings.HasPrefix(lower, "<document>") || strings.HasPrefix(lower, "<documents>") ||
		strings.HasPrefix(lower, "<context>") {
		return true
	}
	if strings.HasPrefix(lower, "repomap") || strings.HasPrefix(lower, "# repomap") ||
		strings.HasPrefix(lower, "## repomap") || strings.HasPrefix(lower, "repo map") ||
		strings.HasPrefix(lower, "memory cards") || strings.HasPrefix(lower, "# memory cards") ||
		strings.HasPrefix(lower, "# memory") || strings.HasPrefix(lower, "memory card") ||
		strings.HasPrefix(lower, "durable context") || strings.HasPrefix(lower, "# durable context") ||
		strings.HasPrefix(lower, "stable context") || strings.HasPrefix(lower, "# stable context") ||
		strings.HasPrefix(lower, "document context") || strings.HasPrefix(lower, "# document context") {
		return true
	}
	if r == agent.RoleSystem || r == "developer" || r == "system" {
		if strings.Contains(lower, "repomap") || strings.Contains(lower, "memory cards") ||
			strings.Contains(lower, "memory card") || strings.Contains(lower, "durable context") {
			if !strings.HasPrefix(lower, "you are ") && !strings.HasPrefix(lower, "you are an ") && !strings.HasPrefix(lower, "you are a ") {
				return true
			}
		}
	}
	return false
}

func isProjectRulesMessage(msg agent.Message) bool {
	r := strings.ToLower(strings.TrimSpace(msg.Role))
	if r == "user" || r == "assistant" || r == "tool" || r == "function" ||
		msg.ToolCallID != "" || len(msg.ToolCalls) > 0 || msg.FunctionCall != nil {
		return false
	}
	if r == "project_rules" || r == "rules" || r == "guidance" {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(msg.Name))
	if name == "project_rules" || name == "rules" || name == "guidance" || name == "agents.md" || name == "claude.md" {
		return true
	}
	content := strings.TrimSpace(msg.Content)
	lower := strings.ToLower(content)
	if strings.HasPrefix(lower, "<project_rules>") || strings.HasPrefix(lower, "<rules>") || strings.HasPrefix(lower, "<guidance>") {
		return true
	}
	if strings.HasPrefix(lower, "project rules") || strings.HasPrefix(lower, "# project rules") ||
		strings.HasPrefix(lower, "## project rules") || strings.HasPrefix(lower, "repository rules") ||
		strings.HasPrefix(lower, "# repository rules") || strings.HasPrefix(lower, "coding guidelines") ||
		strings.HasPrefix(lower, "repository guidelines") || strings.HasPrefix(lower, "project guidelines") ||
		strings.HasPrefix(lower, "agents.md") || strings.HasPrefix(lower, "# agents.md") ||
		strings.HasPrefix(lower, "claude.md") || strings.HasPrefix(lower, "# claude.md") ||
		strings.HasPrefix(lower, "standing rules") {
		return true
	}
	if r == agent.RoleSystem || r == "developer" || r == "system" {
		if strings.Contains(content, "AGENTS.md") || strings.Contains(content, "CLAUDE.md") ||
			strings.Contains(lower, "project rules") || strings.Contains(lower, "coding guidelines") {
			if !strings.HasPrefix(lower, "you are ") && !strings.HasPrefix(lower, "you are an ") && !strings.HasPrefix(lower, "you are a ") {
				return true
			}
		}
	}
	return false
}

func isWorkspaceContextMessage(msg agent.Message) bool {
	r := strings.ToLower(strings.TrimSpace(msg.Role))
	if r == "user" || r == "assistant" || r == "tool" || r == "function" ||
		msg.ToolCallID != "" || len(msg.ToolCalls) > 0 || msg.FunctionCall != nil {
		return false
	}
	if r == "workspace_context" || r == "workspace" || r == "file_tree" {
		return true
	}
	name := strings.ToLower(strings.TrimSpace(msg.Name))
	if name == "workspace_context" || name == "file_tree" || name == "directory_index" || name == "repo_skeleton" || name == "workspace" {
		return true
	}
	content := strings.TrimSpace(msg.Content)
	lower := strings.ToLower(content)
	if strings.HasPrefix(lower, "<file_tree>") || strings.HasPrefix(lower, "<workspace_context>") ||
		strings.HasPrefix(lower, "<directory_index>") || strings.HasPrefix(lower, "<repo_skeleton>") ||
		strings.HasPrefix(lower, "<workspace>") {
		return true
	}
	if strings.HasPrefix(lower, "file tree") || strings.HasPrefix(lower, "# file tree") ||
		strings.HasPrefix(lower, "directory structure") || strings.HasPrefix(lower, "# directory structure") ||
		strings.HasPrefix(lower, "workspace context") || strings.HasPrefix(lower, "# workspace context") ||
		strings.HasPrefix(lower, "repository skeleton") || strings.HasPrefix(lower, "# repository skeleton") ||
		strings.HasPrefix(lower, "directory index") || strings.HasPrefix(lower, "repo skeleton") ||
		strings.HasPrefix(lower, "workspace:") {
		return true
	}
	if r == agent.RoleSystem || r == "developer" || r == "system" {
		if strings.Contains(content, "├──") || strings.Contains(content, "└──") {
			return true
		}
		if strings.Contains(lower, "file tree") || strings.Contains(lower, "directory index") || strings.Contains(lower, "repo skeleton") {
			return true
		}
	}
	return false
}

func (c DeterministicPromptCanonicalizer) extractVolatileFromText(s string) (string, []string) {
	if !c.StripTimestamps && !c.StripSessionIDs {
		return s, nil
	}
	var extracted []string
	lines := strings.Split(s, "\n")
	var remainingLines []string

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			remainingLines = append(remainingLines, line)
			continue
		}

		lower := strings.ToLower(trimmed)
		isTimestamp := c.StripTimestamps && (reTimestampLine.MatchString(trimmed) ||
			reIsoTimestampLine.MatchString(trimmed) ||
			strings.HasPrefix(lower, "date:") || strings.HasPrefix(lower, "timestamp:") ||
			strings.HasPrefix(lower, "today"))

		isSession := c.StripSessionIDs && (reSessionLine.MatchString(trimmed) ||
			strings.HasPrefix(lower, "session id:") || strings.HasPrefix(lower, "session:") ||
			strings.HasPrefix(lower, "trace id:") || strings.HasPrefix(lower, "run id:") ||
			strings.HasPrefix(lower, "turn:") || strings.HasPrefix(lower, "current turn:"))

		isWorkingDir := (c.StripTimestamps || c.StripSessionIDs) && reWorkingDirLine.MatchString(trimmed)

		if isTimestamp || isSession || isWorkingDir {
			extracted = append(extracted, trimmed)
			continue
		}

		cleanedLine := trimmed
		matchedInline := false
		if c.StripTimestamps {
			for _, m := range reTimestampInline.FindAllString(cleanedLine, -1) {
				mTrim := strings.TrimSpace(m)
				if mTrim != "" {
					extracted = append(extracted, mTrim)
					cleanedLine = strings.Replace(cleanedLine, m, "", 1)
					matchedInline = true
				}
			}
			for _, m := range reIsoTimestampInline.FindAllString(cleanedLine, -1) {
				mTrim := strings.TrimSpace(m)
				if mTrim != "" {
					extracted = append(extracted, mTrim)
					cleanedLine = strings.Replace(cleanedLine, m, "", 1)
					matchedInline = true
				}
			}
		}
		if c.StripSessionIDs {
			for _, m := range reSessionInline.FindAllString(cleanedLine, -1) {
				mTrim := strings.TrimSpace(m)
				if mTrim != "" {
					extracted = append(extracted, mTrim)
					cleanedLine = strings.Replace(cleanedLine, m, "", 1)
					matchedInline = true
				}
			}
		}
		if c.StripTimestamps || c.StripSessionIDs {
			for _, m := range reWorkingDirInline.FindAllString(cleanedLine, -1) {
				mTrim := strings.TrimSpace(m)
				if mTrim != "" {
					extracted = append(extracted, mTrim)
					cleanedLine = strings.Replace(cleanedLine, m, "", 1)
					matchedInline = true
				}
			}
		}

		if matchedInline {
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

func computePrefixTokens(prefixMsgs []agent.Message, tools []responsesTool) int {
	total := 0
	for _, m := range prefixMsgs {
		total += EstimateTokens(m.Content)
	}
	if len(tools) > 0 {
		for _, t := range tools {
			total += EstimateTokens(t.Name + " " + t.Description + " " + string(t.Parameters))
		}
	}
	return total
}

func computeStablePrefixHashFromTiers(prefixMsgs []agent.Message, sortedTools []responsesTool) string {
	h := sha256.New()
	h.Write([]byte("TIER2_TOOLS_START\n"))
	for _, t := range sortedTools {
		raw, err := json.Marshal(t)
		if err == nil {
			h.Write(raw)
		} else {
			h.Write([]byte(fmt.Sprintf("%s:%s", t.Type, t.Name)))
		}
		h.Write([]byte("\n"))
	}
	h.Write([]byte("TIER2_TOOLS_END\n"))

	h.Write([]byte("TIER_PREFIX_MSGS_START\n"))
	for _, m := range prefixMsgs {
		h.Write([]byte(fmt.Sprintf("role:%s|name:%s|content:%s\n", m.Role, m.Name, m.Content)))
	}
	h.Write([]byte("TIER_PREFIX_MSGS_END\n"))

	return hex.EncodeToString(h.Sum(nil))
}

func normalizeToolDeclarationText(s string) string {
	lines := strings.Split(s, "\n")
	if len(lines) <= 2 {
		return s
	}
	var headerLines []string
	var toolLines []string
	inTools := false
	for _, l := range lines {
		trimmed := strings.TrimSpace(l)
		if strings.HasPrefix(trimmed, "- ") || strings.HasPrefix(trimmed, "* ") {
			inTools = true
			toolLines = append(toolLines, l)
		} else if !inTools {
			headerLines = append(headerLines, l)
		} else if trimmed == "" {
			continue
		} else {
			toolLines = append(toolLines, l)
		}
	}
	if len(toolLines) > 1 {
		sort.Strings(toolLines)
		var res []string
		res = append(res, headerLines...)
		res = append(res, toolLines...)
		return strings.Join(res, "\n")
	}
	return s
}

func computeStablePrefixHash(tier1, tier2, tier3, tier4 []agent.Message) string {
	h := sha256.New()
	writeTier := func(name string, msgs []agent.Message) {
		h.Write([]byte(name + "_START\n"))
		for _, m := range msgs {
			h.Write([]byte(fmt.Sprintf("role:%s|name:%s|content:%s\n", m.Role, m.Name, m.Content)))
		}
		h.Write([]byte(name + "_END\n"))
	}
	writeTier("TIER1", tier1)
	writeTier("TIER2", tier2)
	writeTier("TIER3", tier3)
	writeTier("TIER4", tier4)
	return hex.EncodeToString(h.Sum(nil))
}

// CanonicalizeMessages sorts messages into canonical 5 tiers (Tier 1 System -> Tier 2 Tools ->
// Tier 3 Project Rules -> Tier 4 Repo Context -> Tier 5 Dynamic Turns), preserves the chronological
// order of messages within Tier 5, hoists volatile ephemera out of Tiers 1-4 into Tier 5,
// and returns the canonicalized messages along with a detailed CanonicalizeReport.
func (c *DeterministicPromptCanonicalizer) CanonicalizeMessages(messages []agent.Message) ([]agent.Message, CanonicalizeReport) {
	report := CanonicalizeReport{
		OriginalCount: len(messages),
		TierCounts:    make(map[int]int),
	}

	var canon DeterministicPromptCanonicalizer
	if c != nil {
		canon = *c
	} else {
		canon = DefaultDeterministicPromptCanonicalizer()
	}

	outMsgs, _, stats := canon.CanonicalizePromptMessages(messages, nil)

	for _, msg := range outMsgs {
		t := int(canon.ClassifyMessage(msg))
		report.TierCounts[t]++
	}

	report.CanonicalCount = len(outMsgs)
	report.StablePrefixHash = stats.StablePrefixHash
	report.VolatileTokensHoisted = stats.VolatileElementsHoisted

	return outMsgs, report
}

// CanonicalizePromptMessages canonicalizes messages and tools into prompt prefix tiers (1-4)
// followed by dynamic history (Tier 5), sorting tools and hoisting volatile metadata,
// and setting an explicit provider cache breakpoint at the prefix boundary (or longest stable prefix).
func (c DeterministicPromptCanonicalizer) CanonicalizePromptMessages(
	messages []agent.Message,
	tools []responsesTool,
) ([]agent.Message, []responsesTool, CanonicalizeStats) {
	sortedTools := CanonicalizeTools(tools)
	stats := CanonicalizeStats{
		ToolsSorted:          len(sortedTools),
		CacheBreakpointIndex: -1,
		CacheBreakpointTier:  TierUnknown,
	}

	var tier1Msgs []agent.Message
	var tier2Msgs []agent.Message
	var tier3Msgs []agent.Message
	var tier4Msgs []agent.Message
	var tier5Msgs []agent.Message

	var allHoisted []string

	for _, msg := range messages {
		tier := c.ClassifyMessage(msg)
		switch tier {
		case Tier1SystemInstructions:
			cleaned, hoisted := c.extractVolatileFromText(msg.Content)
			allHoisted = append(allHoisted, hoisted...)
			msg.Content = cleaned
			if strings.TrimSpace(msg.Content) != "" || (cleaned == "" && len(hoisted) == 0) {
				tier1Msgs = append(tier1Msgs, msg)
			}
		case Tier2ToolDeclarations:
			cleaned, hoisted := c.extractVolatileFromText(msg.Content)
			allHoisted = append(allHoisted, hoisted...)
			msg.Content = cleaned
			if c.NormalizeTools {
				msg.Content = normalizeToolDeclarationText(msg.Content)
			}
			if strings.TrimSpace(msg.Content) != "" || (cleaned == "" && len(hoisted) == 0) {
				tier2Msgs = append(tier2Msgs, msg)
			}
		case Tier3ProjectRules:
			cleaned, hoisted := c.extractVolatileFromText(msg.Content)
			allHoisted = append(allHoisted, hoisted...)
			msg.Content = cleaned
			if strings.TrimSpace(msg.Content) != "" || (cleaned == "" && len(hoisted) == 0) {
				tier3Msgs = append(tier3Msgs, msg)
			}
		case Tier4RepoContext:
			cleaned, hoisted := c.extractVolatileFromText(msg.Content)
			allHoisted = append(allHoisted, hoisted...)
			msg.Content = cleaned
			if strings.TrimSpace(msg.Content) != "" || (cleaned == "" && len(hoisted) == 0) {
				tier4Msgs = append(tier4Msgs, msg)
			}
		case Tier5DynamicTurns:
			tier5Msgs = append(tier5Msgs, msg)
		default:
			tier5Msgs = append(tier5Msgs, msg)
		}
	}

	if c.NormalizeTools && len(tier2Msgs) > 1 {
		sort.SliceStable(tier2Msgs, func(i, j int) bool {
			return tier2Msgs[i].Content < tier2Msgs[j].Content
		})
	}

	stats.VolatileElementsHoisted = len(allHoisted)

	// Hoist volatile elements into Tier 5
	if len(allHoisted) > 0 {
		hoistedText := strings.Join(allHoisted, "\n")
		lastUserIdx := -1
		for i := len(tier5Msgs) - 1; i >= 0; i-- {
			if tier5Msgs[i].Role == agent.RoleUser {
				lastUserIdx = i
				break
			}
		}
		if lastUserIdx >= 0 {
			if strings.TrimSpace(tier5Msgs[lastUserIdx].Content) == "" {
				tier5Msgs[lastUserIdx].Content = hoistedText
			} else {
				tier5Msgs[lastUserIdx].Content = strings.TrimRight(tier5Msgs[lastUserIdx].Content, "\r\n") + "\n\n" + hoistedText
			}
		} else {
			tier5Msgs = append(tier5Msgs, agent.Message{
				Role:    agent.RoleUser,
				Content: hoistedText,
			})
		}
	}

	prefixMsgs := make([]agent.Message, 0, len(tier1Msgs)+len(tier2Msgs)+len(tier3Msgs)+len(tier4Msgs))
	prefixMsgs = append(prefixMsgs, tier1Msgs...)
	prefixMsgs = append(prefixMsgs, tier2Msgs...)
	prefixMsgs = append(prefixMsgs, tier3Msgs...)
	prefixMsgs = append(prefixMsgs, tier4Msgs...)

	// Reset any existing cache control flags on input to ensure exactly one deterministic breakpoint.
	for i := range prefixMsgs {
		prefixMsgs[i].CacheControl = nil
	}
	for i := range tier5Msgs {
		tier5Msgs[i].CacheControl = nil
	}

	// Place explicit provider cache breakpoint at the prefix boundary (or longest stable prefix).
	if c.shouldInsertCacheBreakpoint() && len(prefixMsgs) > 0 {
		bpIdx := len(prefixMsgs) - 1
		prefixMsgs[bpIdx].CacheControl = &agent.CacheControl{Type: "ephemeral"}
		stats.CacheBreakpointIndex = bpIdx

		if len(tier4Msgs) > 0 {
			stats.CacheBreakpointTier = Tier4RepoContext
		} else if len(tier3Msgs) > 0 {
			stats.CacheBreakpointTier = Tier3ProjectRules
		} else if len(tier2Msgs) > 0 {
			stats.CacheBreakpointTier = Tier2ToolDeclarations
		} else if len(tier1Msgs) > 0 {
			stats.CacheBreakpointTier = Tier1SystemInstructions
		}
	}

	prefixTokens := computePrefixTokens(prefixMsgs, sortedTools)

	if c.EnforceBlockAlignment && len(prefixMsgs) > 0 && prefixTokens > 0 {
		if rem := prefixTokens % 1024; rem != 0 {
			needed := 1024 - rem
			lastIdx := len(prefixMsgs) - 1
			if needed > 1 {
				prefixMsgs[lastIdx].Content += strings.Repeat(" ", (needed-1)*4)
			}
			for computePrefixTokens(prefixMsgs, sortedTools)%1024 != 0 {
				prefixMsgs[lastIdx].Content += " "
			}
			prefixTokens = computePrefixTokens(prefixMsgs, sortedTools)
		}
	}
	stats.PrefixTokensEstimate = prefixTokens

	var tiersPresent []PromptTier
	if len(tier1Msgs) > 0 {
		tiersPresent = append(tiersPresent, Tier1SystemInstructions)
	}
	if len(tier2Msgs) > 0 || len(sortedTools) > 0 {
		tiersPresent = append(tiersPresent, Tier2ToolDeclarations)
	}
	if len(tier3Msgs) > 0 {
		tiersPresent = append(tiersPresent, Tier3ProjectRules)
	}
	if len(tier4Msgs) > 0 {
		tiersPresent = append(tiersPresent, Tier4RepoContext)
	}
	if len(tier5Msgs) > 0 {
		tiersPresent = append(tiersPresent, Tier5DynamicTurns)
	}
	stats.TiersPresent = tiersPresent

	stats.StablePrefixHash = computeStablePrefixHashFromTiers(prefixMsgs, sortedTools)

	outMessages := make([]agent.Message, 0, len(prefixMsgs)+len(tier5Msgs))
	outMessages = append(outMessages, prefixMsgs...)
	outMessages = append(outMessages, tier5Msgs...)

	return outMessages, sortedTools, stats
}

// CanonicalizePromptMessagesWithToolDefs canonicalizes messages and agent tool definitions into
// canonical tiers, sorting tool definitions alphabetically by function name.
func (c DeterministicPromptCanonicalizer) CanonicalizePromptMessagesWithToolDefs(
	messages []agent.Message,
	tools []agent.ToolDef,
) ([]agent.Message, []agent.ToolDef, CanonicalizeStats) {
	sortedTools := CanonicalizeToolDefs(tools)
	rTools := make([]responsesTool, len(sortedTools))
	for i, t := range sortedTools {
		rTools[i] = responsesTool{
			Type:        t.Type,
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		}
	}
	outMsgs, _, stats := c.CanonicalizePromptMessages(messages, rTools)
	stats.ToolsSorted = len(sortedTools)
	return outMsgs, sortedTools, stats
}

// CanonicalizePromptOrder canonicalizes a slice of messages into the canonical
// order (Tier 1 System -> Tier 2 Tools -> Tier 3 Project Rules -> Tier 4 Repo Context -> Tier 5 Dynamic Turns),
// hoists volatile session metadata from static tiers to the dynamic suffix, and marks
// an explicit provider cache breakpoint at the boundary of the stable prefix.
func CanonicalizePromptOrder(messages []agent.Message) []agent.Message {
	c := DefaultDeterministicPromptCanonicalizer()
	out, _, _ := c.CanonicalizePromptMessages(messages, nil)
	return out
}

// CanonicalizePromptOrder canonicalizes a slice of messages into the canonical
// order using the receiver's configuration.
func (c DeterministicPromptCanonicalizer) CanonicalizePromptOrder(messages []agent.Message) []agent.Message {
	out, _, _ := c.CanonicalizePromptMessages(messages, nil)
	return out
}

// ComputeStablePrefixHash computes a deterministic SHA-256 hash over prefix tiers only.
func (c DeterministicPromptCanonicalizer) ComputeStablePrefixHash(
	messages []agent.Message,
	tools []responsesTool,
) string {
	_, _, stats := c.CanonicalizePromptMessages(messages, tools)
	return stats.StablePrefixHash
}
