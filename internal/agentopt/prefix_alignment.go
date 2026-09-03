package agentopt

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Family 8: KV buffer & prefix reuse.
//
// Prefix alignment stabilizes static prompt blocks and tool schemas into
// canonical representations to maximize prefix reuse across multi-turn agent
// interactions.

// TurnInteraction records input, output, and tool invocations for one turn.
type TurnInteraction struct {
	TurnIndex int               `json:"turn_index"`
	Prompt    string            `json:"prompt,omitempty"`
	Input     string            `json:"input,omitempty"`
	Output    string            `json:"output,omitempty"`
	ToolCalls []ToolCall        `json:"tool_calls,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

// AlignedPrefix represents a canonical, deterministically ordered prefix.
type AlignedPrefix struct {
	SystemPrompt   string       `json:"system_prompt"`
	SystemBlocks   []string     `json:"system_blocks"`
	ToolSchemas    []ToolSchema `json:"tool_schemas"`
	PrefixHash     string       `json:"prefix_hash"`
	CanonicalBytes []byte       `json:"canonical_bytes,omitempty"`
	RadixBlocks    []string     `json:"radix_blocks,omitempty"`
}

// PrefixStabilizerConfig defines parameters for prefix normalization.
type PrefixStabilizerConfig struct {
	BlockDelimiter      string `json:"block_delimiter"`
	NormalizeWhitespace bool   `json:"normalize_whitespace"`
	SortBlocks          bool   `json:"sort_blocks"`
	SortTools           bool   `json:"sort_tools"`
}

// DefaultPrefixStabilizerConfig returns standard stabilization settings.
func DefaultPrefixStabilizerConfig() PrefixStabilizerConfig {
	return PrefixStabilizerConfig{
		BlockDelimiter:      "\n\n",
		NormalizeWhitespace: true,
		SortBlocks:          true,
		SortTools:           true,
	}
}

// PrefixStabilizer enforces deterministic lexicographical ordering of tool
// schemas and static prompt blocks to guarantee invariant prefix hashes.
type PrefixStabilizer struct {
	cfg PrefixStabilizerConfig
}

// NewPrefixStabilizer creates a stabilizer with optional custom settings.
func NewPrefixStabilizer(cfgs ...PrefixStabilizerConfig) *PrefixStabilizer {
	c := DefaultPrefixStabilizerConfig()
	if len(cfgs) > 0 {
		c = cfgs[0]
		if c.BlockDelimiter == "" {
			c.BlockDelimiter = "\n\n"
		}
	}
	return &PrefixStabilizer{cfg: c}
}

// StabilizePrefix canonicalizes system prompt blocks and tool schemas into an AlignedPrefix.
func (s *PrefixStabilizer) StabilizePrefix(systemPrompt string, toolSchemas []ToolSchema) AlignedPrefix {
	cfg := DefaultPrefixStabilizerConfig()
	if s != nil && s.cfg.BlockDelimiter != "" {
		cfg = s.cfg
	}

	// 1. Process and stabilize system prompt blocks.
	blocks := extractPromptBlocks(systemPrompt, cfg.BlockDelimiter, cfg.NormalizeWhitespace)
	if cfg.SortBlocks && len(blocks) > 1 {
		sort.Strings(blocks)
	}
	canonicalPrompt := strings.Join(blocks, cfg.BlockDelimiter)

	// 2. Canonicalize and sort tool schemas.
	var sortedTools []ToolSchema
	if len(toolSchemas) > 0 {
		sortedTools = make([]ToolSchema, len(toolSchemas))
		for i, ts := range toolSchemas {
			sortedTools[i] = canonicalizeToolSchema(ts)
		}
		if cfg.SortTools && len(sortedTools) > 1 {
			sort.SliceStable(sortedTools, func(i, j int) bool {
				if sortedTools[i].Name != sortedTools[j].Name {
					return sortedTools[i].Name < sortedTools[j].Name
				}
				return sortedTools[i].Description < sortedTools[j].Description
			})
		}
	} else {
		sortedTools = []ToolSchema{}
	}

	// 3. Assemble canonical byte representation.
	var buf bytes.Buffer
	buf.WriteString("PREFIX_V1\n")
	buf.WriteString("SYSTEM:\n")
	for i, blk := range blocks {
		buf.WriteString(fmt.Sprintf("[%d]:%s\n", i, blk))
	}
	buf.WriteString("TOOLS:\n")
	for i, tool := range sortedTools {
		toolRaw, _ := json.Marshal(tool)
		buf.WriteString(fmt.Sprintf("[%d]:%s:%s\n", i, tool.Name, string(toolRaw)))
	}
	canonicalBytes := buf.Bytes()

	hasher := sha256.New()
	hasher.Write(canonicalBytes)
	prefixHash := hex.EncodeToString(hasher.Sum(nil))

	// 4. Construct radix block identifiers.
	var radixBlocks []string
	for i, blk := range blocks {
		bh := sha256.Sum256([]byte(blk))
		radixBlocks = append(radixBlocks, fmt.Sprintf("sys:%d:%s", i, hex.EncodeToString(bh[:])[:16]))
	}
	for _, tool := range sortedTools {
		toolRaw, _ := json.Marshal(tool)
		th := sha256.Sum256(toolRaw)
		radixBlocks = append(radixBlocks, fmt.Sprintf("tool:%s:%s", tool.Name, hex.EncodeToString(th[:])[:16]))
	}

	return AlignedPrefix{
		SystemPrompt:   canonicalPrompt,
		SystemBlocks:   blocks,
		ToolSchemas:    sortedTools,
		PrefixHash:     prefixHash,
		CanonicalBytes: canonicalBytes,
		RadixBlocks:    radixBlocks,
	}
}

// extractPromptBlocks splits a system prompt into distinct blocks, normalizing line breaks.
func extractPromptBlocks(prompt, delimiter string, normalizeWhitespace bool) []string {
	if strings.TrimSpace(prompt) == "" {
		return nil
	}

	normalized := strings.ReplaceAll(prompt, "\r\n", "\n")
	normalized = strings.ReplaceAll(normalized, "\r", "\n")

	parts := strings.Split(normalized, delimiter)
	var blocks []string
	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if trimmed == "" {
			continue
		}
		if normalizeWhitespace {
			// Normalize internal spacing of lines within a block
			lines := strings.Split(trimmed, "\n")
			var cleanedLines []string
			for _, line := range lines {
				cleanedLines = append(cleanedLines, strings.TrimRight(line, " \t"))
			}
			trimmed = strings.Join(cleanedLines, "\n")
		}
		blocks = append(blocks, trimmed)
	}
	return blocks
}

// canonicalizeToolSchema clones and sanitizes a tool schema to ensure deterministic encoding.
func canonicalizeToolSchema(ts ToolSchema) ToolSchema {
	out := ToolSchema{
		Name:                 strings.TrimSpace(ts.Name),
		Description:          strings.TrimSpace(ts.Description),
		AdditionalProperties: ts.AdditionalProperties,
	}
	if len(ts.Required) > 0 {
		req := make([]string, len(ts.Required))
		copy(req, ts.Required)
		sort.Strings(req)
		out.Required = req
	}
	if ts.Properties != nil {
		out.Properties = make(map[string]PropertySchema, len(ts.Properties))
		for k, v := range ts.Properties {
			out.Properties[k] = canonicalizePropertySchema(v)
		}
	}
	return out
}

// canonicalizePropertySchema clones and sanitizes nested property schemas.
func canonicalizePropertySchema(ps PropertySchema) PropertySchema {
	out := PropertySchema{
		Type: ps.Type,
	}
	if len(ps.Enum) > 0 {
		out.Enum = append([]string(nil), ps.Enum...)
	}
	if len(ps.Required) > 0 {
		req := append([]string(nil), ps.Required...)
		sort.Strings(req)
		out.Required = req
	}
	if ps.Items != nil {
		item := canonicalizePropertySchema(*ps.Items)
		out.Items = &item
	}
	if ps.Properties != nil {
		out.Properties = make(map[string]PropertySchema, len(ps.Properties))
		for k, v := range ps.Properties {
			out.Properties[k] = canonicalizePropertySchema(v)
		}
	}
	return out
}

// HashTurn computes a deterministic digest for a single turn interaction.
func HashTurn(turn TurnInteraction) string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("turn_index:%d\n", turn.TurnIndex)))
	inputText := turn.Prompt
	if inputText == "" {
		inputText = turn.Input
	}
	h.Write([]byte("input:" + strings.TrimSpace(inputText) + "\n"))
	h.Write([]byte("output:" + strings.TrimSpace(turn.Output) + "\n"))
	if len(turn.ToolCalls) > 0 {
		h.Write([]byte("tool_calls:\n"))
		// Sort tool calls by ID then Name to preserve deterministic order
		calls := make([]ToolCall, len(turn.ToolCalls))
		copy(calls, turn.ToolCalls)
		sort.SliceStable(calls, func(i, j int) bool {
			if calls[i].ID != calls[j].ID {
				return calls[i].ID < calls[j].ID
			}
			return calls[i].Name < calls[j].Name
		})
		for _, tc := range calls {
			argsRaw, _ := json.Marshal(tc.Args)
			h.Write([]byte(fmt.Sprintf("%s:%s:%t:%s\n", tc.ID, tc.Name, tc.ReadOnly, string(argsRaw))))
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ChainHash computes the sequential prefix hash folding in a parent hash and a turn interaction.
func ChainHash(parentHash string, turn TurnInteraction) string {
	h := sha256.New()
	h.Write([]byte(parentHash))
	h.Write([]byte(":"))
	h.Write([]byte(HashTurn(turn)))
	return hex.EncodeToString(h.Sum(nil))
}

// BuildPrefixChain constructs the sequence of prefix hashes [H0, H1, ... Hn]
// across consecutive turn interactions. H0 is the base prefix hash.
func (p AlignedPrefix) BuildPrefixChain(history []TurnInteraction) []string {
	chain := make([]string, 0, len(history)+1)
	curr := p.PrefixHash
	chain = append(chain, curr)
	for _, turn := range history {
		curr = ChainHash(curr, turn)
		chain = append(chain, curr)
	}
	return chain
}

// ChainHash computes the cumulative prefix hash for the base prefix followed by history turns.
func (p AlignedPrefix) ChainHash(history ...TurnInteraction) string {
	curr := p.PrefixHash
	for _, turn := range history {
		curr = ChainHash(curr, turn)
	}
	return curr
}

// ExtendTurn returns a new AlignedPrefix with PrefixHash chained with the given turn.
func (p AlignedPrefix) ExtendTurn(turn TurnInteraction) AlignedPrefix {
	newHash := ChainHash(p.PrefixHash, turn)
	turnDigest := HashTurn(turn)
	newRadixBlocks := make([]string, len(p.RadixBlocks), len(p.RadixBlocks)+1)
	copy(newRadixBlocks, p.RadixBlocks)
	newRadixBlocks = append(newRadixBlocks, fmt.Sprintf("turn:%d:%s", turn.TurnIndex, turnDigest[:16]))

	return AlignedPrefix{
		SystemPrompt:   p.SystemPrompt,
		SystemBlocks:   p.SystemBlocks,
		ToolSchemas:    p.ToolSchemas,
		PrefixHash:     newHash,
		CanonicalBytes: p.CanonicalBytes,
		RadixBlocks:    newRadixBlocks,
	}
}

// StabilizeTurn aligns the base prefix and folds in historical turn interactions
// into a chained prefix representation.
func (s *PrefixStabilizer) StabilizeTurn(systemPrompt string, toolSchemas []ToolSchema, history []TurnInteraction) AlignedPrefix {
	base := s.StabilizePrefix(systemPrompt, toolSchemas)
	if len(history) == 0 {
		return base
	}
	res := base
	for _, turn := range history {
		res = res.ExtendTurn(turn)
	}
	return res
}

// RadixPrefixNode represents a node in the radix prefix tree.
type RadixPrefixNode struct {
	Hash     string
	Children map[string]*RadixPrefixNode
	Hits     int64
}

// RadixPrefixTree maintains prefix hash hierarchies for multi-turn reuse.
type RadixPrefixTree struct {
	mu        sync.RWMutex
	root      *RadixPrefixNode
	nodeCount int
}

// NewRadixPrefixTree initializes an empty radix prefix tree.
func NewRadixPrefixTree() *RadixPrefixTree {
	return &RadixPrefixTree{
		root: &RadixPrefixNode{
			Children: make(map[string]*RadixPrefixNode),
		},
	}
}

// Insert adds a sequence of prefix hashes into the radix tree.
func (t *RadixPrefixTree) Insert(chain []string) int {
	if len(chain) == 0 {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	curr := t.root
	inserted := 0
	for _, h := range chain {
		child, exists := curr.Children[h]
		if !exists {
			child = &RadixPrefixNode{
				Hash:     h,
				Children: make(map[string]*RadixPrefixNode),
			}
			curr.Children[h] = child
			t.nodeCount++
			inserted++
		}
		child.Hits++
		curr = child
	}
	return inserted
}

// MatchLength returns the longest common prefix match length against the radix tree.
func (t *RadixPrefixTree) MatchLength(chain []string) int {
	if len(chain) == 0 {
		return 0
	}
	t.mu.RLock()
	defer t.mu.RUnlock()

	curr := t.root
	matched := 0
	for _, h := range chain {
		child, exists := curr.Children[h]
		if !exists {
			break
		}
		matched++
		curr = child
	}
	return matched
}

// MatchRatio computes the ratio of matched prefix steps relative to chain length.
func (t *RadixPrefixTree) MatchRatio(chain []string) float64 {
	if len(chain) == 0 {
		return 0.0
	}
	matched := t.MatchLength(chain)
	return float64(matched) / float64(len(chain))
}

// NodeCount returns the total number of prefix nodes stored in the tree.
func (t *RadixPrefixTree) NodeCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.nodeCount
}
