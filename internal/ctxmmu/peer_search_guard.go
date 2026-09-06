package ctxmmu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	// MaxPeerSearchTokens is the strict token budget (500 tokens) across all hits in a peer search result.
	MaxPeerSearchTokens = 500

	// MaxPeerSearchBytes is the strict byte limit (2KB = 2048 bytes) across all hits in a peer search result.
	MaxPeerSearchBytes = 2048

	// DefaultPeerSearchContextLines is the default number of surrounding context lines to extract around matches.
	DefaultPeerSearchContextLines = 2
)

// taintRank maps to the real lattice: Trusted (0) < Tainted (1) < Quarantined (2).
func taintRank(t abi.TaintLabel) int {
	switch t {
	case abi.TaintTrusted:
		return 0
	case abi.TaintTainted:
		return 1
	case abi.TaintQuarantined:
		return 2
	default:
		return 1 // fail-closed: unknown taint treated as tainted
	}
}

// maxTaint returns the lattice supremum of two taint labels.
func maxTaint(a, b abi.TaintLabel) abi.TaintLabel {
	if taintRank(a) >= taintRank(b) {
		return a
	}
	return b
}

// isShareableScope reports whether a scope is authorized to be shared across agents (ScopeFleet or ScopeTenant).
// ScopeAgent (0, fail-closed default) is strictly private to an agent and must never cross sibling boundaries.
func isShareableScope(scope abi.ShareScope) bool {
	return scope == abi.ScopeFleet || scope == abi.ScopeTenant
}

// PeerItemKind identifies the category of a peer context item.
type PeerItemKind string

const (
	PeerItemTurn     PeerItemKind = "turn"
	PeerItemMemory   PeerItemKind = "memory"
	PeerItemVariable PeerItemKind = "variable"
	PeerItemTool     PeerItemKind = "tool"
	PeerItemGeneric  PeerItemKind = "generic"
)

// PeerField represents an individual named field, variable, or metadata item with its own scope and taint.
type PeerField struct {
	Name        string         `json:"name"`
	Value       string         `json:"value"`
	Scope       abi.ShareScope `json:"scope"`
	Taint       abi.TaintLabel `json:"taint"`
	Quarantined bool           `json:"quarantined,omitempty"`
}

// PeerTurn represents a conversation turn, user prompt, assistant reasoning, or tool execution.
type PeerTurn struct {
	TurnIndex   int                  `json:"turn_index,omitempty"`
	Role        string               `json:"role"`
	Content     string               `json:"content"`
	ToolName    string               `json:"tool_name,omitempty"`
	ToolOutput  []byte               `json:"tool_output,omitempty"`
	Scope       abi.ShareScope       `json:"scope"`
	Taint       abi.TaintLabel       `json:"taint"`
	Quarantined bool                 `json:"quarantined,omitempty"`
	Ref         *abi.Ref             `json:"ref,omitempty"`
	Fields      map[string]PeerField `json:"fields,omitempty"`
	Metadata    map[string]string    `json:"metadata,omitempty"`
}

// PeerMemoryEntry represents a key-value or factual memory entry held by a peer.
type PeerMemoryEntry struct {
	Key         string               `json:"key"`
	Content     string               `json:"content"`
	Scope       abi.ShareScope       `json:"scope"`
	Taint       abi.TaintLabel       `json:"taint"`
	Quarantined bool                 `json:"quarantined,omitempty"`
	Ref         *abi.Ref             `json:"ref,omitempty"`
	Fields      map[string]PeerField `json:"fields,omitempty"`
	Metadata    map[string]string    `json:"metadata,omitempty"`
}

// PeerVariable represents an environment, configuration, or scratchpad variable.
type PeerVariable struct {
	Name        string         `json:"name"`
	Value       string         `json:"value"`
	Scope       abi.ShareScope `json:"scope"`
	Taint       abi.TaintLabel `json:"taint"`
	Quarantined bool           `json:"quarantined,omitempty"`
	Ref         *abi.Ref       `json:"ref,omitempty"`
}

// PeerContextItem represents a discrete fragment of context or tool output in a peer's context store.
type PeerContextItem struct {
	ID          string               `json:"id,omitempty"`
	Kind        PeerItemKind         `json:"kind"`
	Key         string               `json:"key,omitempty"`
	Content     string               `json:"content,omitempty"`
	RawBytes    []byte               `json:"raw_bytes,omitempty"`
	Scope       abi.ShareScope       `json:"scope"`
	Taint       abi.TaintLabel       `json:"taint"`
	Quarantined bool                 `json:"quarantined,omitempty"`
	Ref         *abi.Ref             `json:"ref,omitempty"`
	Fields      map[string]PeerField `json:"fields,omitempty"`
	Metadata    map[string]string    `json:"metadata,omitempty"`
}

// PeerContext aggregates a peer agent's live context including turns, memory, variables, and raw items.
type PeerContext struct {
	PeerID    string
	Turns     []PeerTurn
	Memory    []PeerMemoryEntry
	Variables []PeerVariable
	Items     []PeerContextItem
}

// NewPeerContext allocates an empty PeerContext container.
func NewPeerContext(peerID string) *PeerContext {
	return &PeerContext{
		PeerID: peerID,
	}
}

// AddTurn appends a turn to the peer context.
func (pc *PeerContext) AddTurn(t PeerTurn) {
	pc.Turns = append(pc.Turns, t)
}

// AddMemory appends a memory entry to the peer context.
func (pc *PeerContext) AddMemory(m PeerMemoryEntry) {
	pc.Memory = append(pc.Memory, m)
}

// AddVariable appends a variable to the peer context.
func (pc *PeerContext) AddVariable(v PeerVariable) {
	pc.Variables = append(pc.Variables, v)
}

// AddItem appends a discrete context item to the peer context.
func (pc *PeerContext) AddItem(item PeerContextItem) {
	pc.Items = append(pc.Items, item)
}

// AddToolOutput appends a raw tool execution output to the peer context.
func (pc *PeerContext) AddToolOutput(toolName string, output []byte, scope abi.ShareScope, taint abi.TaintLabel) {
	pc.Items = append(pc.Items, PeerContextItem{
		Kind:        PeerItemTool,
		Key:         toolName,
		RawBytes:    output,
		Content:     string(output),
		Scope:       scope,
		Taint:       taint,
		Quarantined: taint == abi.TaintQuarantined,
	})
}

// PeerSearchHit represents a single matched, scope-filtered, and clamped snippet from peer context.
type PeerSearchHit struct {
	SourceID    string               `json:"source_id,omitempty"`
	Kind        PeerItemKind         `json:"kind"`
	Key         string               `json:"key,omitempty"`
	Snippet     string               `json:"snippet"`
	Tokens      int                  `json:"tokens"`
	Bytes       int                  `json:"bytes"`
	Scope       abi.ShareScope       `json:"scope"`
	Taint       abi.TaintLabel       `json:"taint"`
	Quarantined bool                 `json:"quarantined"`
	Reference   abi.Ref              `json:"reference"`
	Ref         abi.Ref              `json:"ref"`
	Fields      map[string]PeerField `json:"fields,omitempty"`
	Metadata    map[string]string    `json:"metadata,omitempty"`
}

// PeerSearchResult contains the aggregate, budget-clamped search results across all matching hits.
type PeerSearchResult struct {
	Query       string          `json:"query"`
	Hits        []PeerSearchHit `json:"hits"`
	TotalTokens int             `json:"total_tokens"`
	TotalBytes  int             `json:"total_bytes"`
	Taint       abi.TaintLabel  `json:"taint"`
	Quarantined bool            `json:"quarantined"`
	Scope       abi.ShareScope  `json:"scope"`
	Reference   abi.Ref         `json:"reference"`
	Ref         abi.Ref         `json:"ref"`
}

// SearchRef returns the aggregate addressable reference for the search result.
func (r *PeerSearchResult) SearchRef() abi.Ref {
	return r.Reference
}

// RedactScopeAgentFields returns a new fields map with any ScopeAgent (or non-shareable) fields omitted.
func RedactScopeAgentFields(fields map[string]PeerField) map[string]PeerField {
	if len(fields) == 0 {
		return nil
	}
	out := make(map[string]PeerField)
	for k, v := range fields {
		if isShareableScope(v.Scope) {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// PeerSearchGuard enforces token budgets, scope exclusion, and taint preservation over peer context searches.
type PeerSearchGuard struct {
	MaxTokens    int
	MaxBytes     int
	ContextLines int
}

// NewPeerSearchGuard constructs a PeerSearchGuard with canonical default budgets (500 tokens / 2KB).
func NewPeerSearchGuard() *PeerSearchGuard {
	return &PeerSearchGuard{
		MaxTokens:    MaxPeerSearchTokens,
		MaxBytes:     MaxPeerSearchBytes,
		ContextLines: DefaultPeerSearchContextLines,
	}
}

// Search executes a query against peer context, applying strict scope redaction, line extraction,
// output budget clamping (<=500 tokens / 2KB total), and taint preservation.
func (g *PeerSearchGuard) Search(peerCtx *PeerContext, query string) *PeerSearchResult {
	if g == nil {
		g = NewPeerSearchGuard()
	}
	if peerCtx == nil {
		return g.emptyResult(query)
	}

	var items []PeerContextItem

	// 1. Process Turns
	for idx, t := range peerCtx.Turns {
		scope := t.Scope
		if t.Ref != nil && !isShareableScope(t.Ref.Scope) {
			continue
		}
		if !isShareableScope(scope) {
			continue
		}

		taint := t.Taint
		if t.Ref != nil {
			taint = maxTaint(taint, t.Ref.Taint)
		}
		if t.Quarantined {
			taint = abi.TaintQuarantined
		}

		content := t.Content
		var rawBytes []byte
		if len(t.ToolOutput) > 0 {
			rawBytes = t.ToolOutput
			content = string(t.ToolOutput)
		}

		redactedFields := RedactScopeAgentFields(t.Fields)

		items = append(items, PeerContextItem{
			ID:          fmt.Sprintf("turn-%d", idx),
			Kind:        PeerItemTurn,
			Key:         t.ToolName,
			Content:     content,
			RawBytes:    rawBytes,
			Scope:       scope,
			Taint:       taint,
			Quarantined: taint == abi.TaintQuarantined,
			Ref:         t.Ref,
			Fields:      redactedFields,
			Metadata:    t.Metadata,
		})
	}

	// 2. Process Memory entries
	for _, m := range peerCtx.Memory {
		scope := m.Scope
		if m.Ref != nil && !isShareableScope(m.Ref.Scope) {
			continue
		}
		if !isShareableScope(scope) {
			continue
		}

		taint := m.Taint
		if m.Ref != nil {
			taint = maxTaint(taint, m.Ref.Taint)
		}
		if m.Quarantined {
			taint = abi.TaintQuarantined
		}

		redactedFields := RedactScopeAgentFields(m.Fields)

		items = append(items, PeerContextItem{
			ID:          m.Key,
			Kind:        PeerItemMemory,
			Key:         m.Key,
			Content:     m.Content,
			Scope:       scope,
			Taint:       taint,
			Quarantined: taint == abi.TaintQuarantined,
			Ref:         m.Ref,
			Fields:      redactedFields,
			Metadata:    m.Metadata,
		})
	}

	// 3. Process Variables
	for _, v := range peerCtx.Variables {
		scope := v.Scope
		if v.Ref != nil && !isShareableScope(v.Ref.Scope) {
			continue
		}
		if !isShareableScope(scope) {
			continue
		}

		taint := v.Taint
		if v.Ref != nil {
			taint = maxTaint(taint, v.Ref.Taint)
		}
		if v.Quarantined {
			taint = abi.TaintQuarantined
		}

		items = append(items, PeerContextItem{
			ID:          v.Name,
			Kind:        PeerItemVariable,
			Key:         v.Name,
			Content:     fmt.Sprintf("%s = %s", v.Name, v.Value),
			Scope:       scope,
			Taint:       taint,
			Quarantined: taint == abi.TaintQuarantined,
			Ref:         v.Ref,
		})
	}

	// 4. Process discrete items
	for idx, it := range peerCtx.Items {
		scope := it.Scope
		if it.Ref != nil && !isShareableScope(it.Ref.Scope) {
			continue
		}
		if !isShareableScope(scope) {
			continue
		}

		taint := it.Taint
		if it.Ref != nil {
			taint = maxTaint(taint, it.Ref.Taint)
		}
		if it.Quarantined {
			taint = abi.TaintQuarantined
		}

		id := it.ID
		if id == "" {
			id = fmt.Sprintf("item-%d", idx)
		}

		redactedFields := RedactScopeAgentFields(it.Fields)

		items = append(items, PeerContextItem{
			ID:          id,
			Kind:        it.Kind,
			Key:         it.Key,
			Content:     it.Content,
			RawBytes:    it.RawBytes,
			Scope:       scope,
			Taint:       taint,
			Quarantined: taint == abi.TaintQuarantined,
			Ref:         it.Ref,
			Fields:      redactedFields,
			Metadata:    it.Metadata,
		})
	}

	return g.FilterAndClamp(items, query)
}

// FilterAndClamp filters discrete items by scope, extracts matching lines, clamps to output budget,
// and preserves taint across sibling boundaries.
func (g *PeerSearchGuard) FilterAndClamp(items []PeerContextItem, query string) *PeerSearchResult {
	if g == nil {
		g = NewPeerSearchGuard()
	}

	q := strings.TrimSpace(query)
	qLower := strings.ToLower(q)

	maxTokens := g.MaxTokens
	if maxTokens <= 0 {
		maxTokens = MaxPeerSearchTokens
	}
	maxBytes := g.MaxBytes
	if maxBytes <= 0 {
		maxBytes = MaxPeerSearchBytes
	}
	contextLines := g.ContextLines
	if contextLines < 0 {
		contextLines = DefaultPeerSearchContextLines
	}

	var hits []PeerSearchHit
	accumTokens := 0
	accumBytes := 0
	maxObservedTaint := abi.TaintTrusted
	quarantinedActive := false
	effectiveScope := abi.ScopeFleet

	for _, item := range items {
		// 1. Scope exclusion invariant: Filter out ScopeAgent or non-shared private scope
		if !isShareableScope(item.Scope) {
			continue
		}
		if item.Ref != nil && !isShareableScope(item.Ref.Scope) {
			continue
		}

		// 2. Taint calculation for item: calculate lattice supremum
		itemTaint := item.Taint
		if item.Ref != nil {
			itemTaint = maxTaint(itemTaint, item.Ref.Taint)
		}
		if item.Quarantined {
			itemTaint = abi.TaintQuarantined
		}

		// 3. Redact ScopeAgent fields before search matching so private fields can never be matched or leaked
		cleanFields := RedactScopeAgentFields(item.Fields)

		content := item.Content
		if len(item.RawBytes) > 0 {
			content = string(item.RawBytes)
		}

		matched := false
		var snippet string

		if q == "" {
			matched = true
			snippet = content
		} else {
			keyMatch := strings.Contains(strings.ToLower(item.Key), qLower)
			metaMatch := false
			for _, v := range item.Metadata {
				if strings.Contains(strings.ToLower(v), qLower) {
					metaMatch = true
					break
				}
			}
			fieldMatch := false
			for _, f := range cleanFields {
				if strings.Contains(strings.ToLower(f.Name), qLower) || strings.Contains(strings.ToLower(f.Value), qLower) {
					fieldMatch = true
					break
				}
			}

			extracted := extractMatchingSnippet(content, q, contextLines)
			if extracted != "" {
				matched = true
				snippet = extracted
			} else if keyMatch || metaMatch || fieldMatch {
				matched = true
				snippet = content
			}
		}

		if !matched {
			continue
		}

		// 4. Check remaining output budget across all hits
		remTokens := maxTokens - accumTokens
		remBytes := maxBytes - accumBytes
		if remTokens <= 0 || remBytes <= 0 {
			break
		}

		clampedBytes := clampSnippet([]byte(snippet), remTokens, remBytes)
		if len(clampedBytes) == 0 && len(snippet) > 0 && (remTokens <= 0 || remBytes <= 0) {
			break
		}

		clampedSnippet := string(clampedBytes)
		hitTokens := EstimateTokens(clampedBytes)
		hitBytes := len(clampedBytes)

		accumTokens += hitTokens
		accumBytes += hitBytes

		maxObservedTaint = maxTaint(maxObservedTaint, itemTaint)
		if itemTaint == abi.TaintQuarantined {
			quarantinedActive = true
		}
		if len(hits) == 0 {
			effectiveScope = item.Scope
		} else if item.Scope < effectiveScope {
			effectiveScope = item.Scope
		}

		hitRef := abi.Ref{
			Kind:   abi.RefInline,
			Inline: clampedBytes,
			Len:    int64(hitBytes),
			Taint:  itemTaint,
			Scope:  item.Scope,
			Digest: sha256Hex(clampedBytes),
		}

		hits = append(hits, PeerSearchHit{
			SourceID:    item.ID,
			Kind:        item.Kind,
			Key:         item.Key,
			Snippet:     clampedSnippet,
			Tokens:      hitTokens,
			Bytes:       hitBytes,
			Scope:       item.Scope,
			Taint:       itemTaint,
			Quarantined: itemTaint == abi.TaintQuarantined,
			Reference:   hitRef,
			Ref:         hitRef,
			Fields:      cleanFields,
			Metadata:    item.Metadata,
		})
	}

	// 5. Aggregate combined reference across all hits, strictly clamped to total budget
	var aggBuf bytes.Buffer
	for i, h := range hits {
		if i > 0 {
			aggBuf.WriteString("\n")
		}
		aggBuf.WriteString(h.Snippet)
	}
	aggBytes := aggBuf.Bytes()
	if len(aggBytes) > maxBytes || EstimateTokens(aggBytes) > maxTokens {
		aggBytes = clampSnippet(aggBytes, maxTokens, maxBytes)
	}

	totalTokens := EstimateTokens(aggBytes)
	totalBytes := len(aggBytes)

	combinedRef := abi.Ref{
		Kind:   abi.RefInline,
		Inline: aggBytes,
		Len:    int64(totalBytes),
		Taint:  maxObservedTaint,
		Scope:  effectiveScope,
		Digest: sha256Hex(aggBytes),
	}

	return &PeerSearchResult{
		Query:       query,
		Hits:        hits,
		TotalTokens: totalTokens,
		TotalBytes:  totalBytes,
		Taint:       maxObservedTaint,
		Quarantined: quarantinedActive,
		Scope:       effectiveScope,
		Reference:   combinedRef,
		Ref:         combinedRef,
	}
}

// ClampToolOutput is a convenience method for clamping a single large tool output
// to the peer search budget with scope validation and taint preservation.
func (g *PeerSearchGuard) ClampToolOutput(output []byte, query string, scope abi.ShareScope, taint abi.TaintLabel) (*PeerSearchHit, error) {
	if !isShareableScope(scope) {
		return nil, fmt.Errorf("ctxmmu: cannot search private scope %v", scope)
	}
	item := PeerContextItem{
		Kind:        PeerItemTool,
		Key:         "tool_output",
		RawBytes:    output,
		Content:     string(output),
		Scope:       scope,
		Taint:       taint,
		Quarantined: taint == abi.TaintQuarantined,
	}
	res := g.FilterAndClamp([]PeerContextItem{item}, query)
	if len(res.Hits) == 0 {
		return nil, fmt.Errorf("ctxmmu: no matches for query %q", query)
	}
	return &res.Hits[0], nil
}

// SearchPeerContext is a package-level helper executing a peer context search with default guard limits.
func SearchPeerContext(peerCtx *PeerContext, query string) *PeerSearchResult {
	return NewPeerSearchGuard().Search(peerCtx, query)
}

func (g *PeerSearchGuard) emptyResult(query string) *PeerSearchResult {
	ref := abi.Ref{
		Kind:   abi.RefInline,
		Inline: nil,
		Len:    0,
		Taint:  abi.TaintTrusted,
		Scope:  abi.ScopeFleet,
	}
	return &PeerSearchResult{
		Query:       query,
		Hits:        nil,
		TotalTokens: 0,
		TotalBytes:  0,
		Taint:       abi.TaintTrusted,
		Quarantined: false,
		Scope:       abi.ScopeFleet,
		Reference:   ref,
		Ref:         ref,
	}
}

// extractMatchingSnippet finds matching lines for query in content and returns matching lines
// with surrounding context lines, merging overlapping ranges.
func extractMatchingSnippet(content string, query string, contextLines int) string {
	if query == "" {
		return content
	}
	lines := strings.Split(content, "\n")
	qLower := strings.ToLower(query)

	type lineRange struct {
		start int
		end   int
	}
	var ranges []lineRange

	for i, line := range lines {
		if strings.Contains(strings.ToLower(line), qLower) {
			start := i - contextLines
			if start < 0 {
				start = 0
			}
			end := i + contextLines
			if end >= len(lines) {
				end = len(lines) - 1
			}
			ranges = append(ranges, lineRange{start: start, end: end})
		}
	}

	if len(ranges) == 0 {
		if strings.Contains(strings.ToLower(content), qLower) {
			return content
		}
		return ""
	}

	// Merge overlapping or adjacent ranges
	merged := []lineRange{ranges[0]}
	for _, r := range ranges[1:] {
		last := &merged[len(merged)-1]
		if r.start <= last.end+1 {
			if r.end > last.end {
				last.end = r.end
			}
		} else {
			merged = append(merged, r)
		}
	}

	var sb strings.Builder
	for idx, r := range merged {
		if idx > 0 {
			sb.WriteString("\n...\n")
		}
		for lineIdx := r.start; lineIdx <= r.end; lineIdx++ {
			if lineIdx > r.start {
				sb.WriteString("\n")
			}
			sb.WriteString(lines[lineIdx])
		}
	}
	return sb.String()
}

// clampSnippet clamps byte slice b to at most maxTokens (via EstimateTokens) and maxBytes,
// preserving valid UTF-8 boundary.
func clampSnippet(b []byte, maxTokens, maxBytes int) []byte {
	if maxTokens <= 0 || maxBytes <= 0 || len(b) == 0 {
		return nil
	}
	if len(b) > maxBytes {
		b = b[:maxBytes]
	}
	// In EstimateTokens(b): n = (len(b) + 3) / 4. For n <= maxTokens, len(b) <= 4 * maxTokens.
	maxLenFromTokens := 4 * maxTokens
	if len(b) > maxLenFromTokens {
		b = b[:maxLenFromTokens]
	}
	for len(b) > 0 && EstimateTokens(b) > maxTokens {
		b = b[:len(b)-1]
	}
	// Ensure valid UTF-8 ending
	for len(b) > 0 && !utf8.Valid(b) {
		b = b[:len(b)-1]
	}
	return b
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}
