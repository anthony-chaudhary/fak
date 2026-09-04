package ctxmmu

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

// CellKind identifies the structural role of a virtual context memory cell.
type CellKind uint8

const (
	CellKindUnknown            CellKind = iota
	CellKindSystemInstructions          // Immutable platform prompt and instructions (prefix anchor)
	CellKindUserConstraints             // Operator intent, pinned constraints, requirements (zero-loss)
	CellKindToolOutput                  // Tool execution outputs (bash, read, grep) subject to demand-paging
	CellKindNegativeFinding             // Verified absence, 0 matches, or failed check (zero-loss)
	CellKindAssistantTurn               // Assistant reasoning and conversational turns
	CellKindUserTurn                    // Interactive conversational user input
)

// String returns the human-readable name of the cell kind.
func (k CellKind) String() string {
	switch k {
	case CellKindSystemInstructions:
		return "system_instructions"
	case CellKindUserConstraints:
		return "user_constraints"
	case CellKindToolOutput:
		return "tool_output"
	case CellKindNegativeFinding:
		return "negative_finding"
	case CellKindAssistantTurn:
		return "assistant_turn"
	case CellKindUserTurn:
		return "user_turn"
	default:
		return "unknown"
	}
}

// IsPrefixAnchor reports whether the cell kind belongs to the immutable prefix anchor.
func (k CellKind) IsPrefixAnchor() bool {
	return k == CellKindSystemInstructions || k == CellKindUserConstraints
}

// IsZeroLoss reports whether the cell kind must never be dropped or lossily compacted.
func (k CellKind) IsZeroLoss() bool {
	return k == CellKindSystemInstructions || k == CellKindUserConstraints || k == CellKindNegativeFinding
}

// ContextCell represents one typed, content-addressed unit of virtual context memory.
type ContextCell struct {
	ID             uint64            `json:"id"`
	Turn           int               `json:"turn"`
	Kind           CellKind          `json:"kind"`
	Role           string            `json:"role"` // "system", "user", "assistant", "tool"
	ToolName       string            `json:"tool_name,omitempty"`
	Query          string            `json:"query,omitempty"` // Search pattern or checked symbol
	Content        []byte            `json:"-"`
	Digest         [32]byte          `json:"-"`
	DigestHex      string            `json:"digest_hex"`
	Tokens         int               `json:"tokens"`
	OriginalBytes  int               `json:"original_bytes,omitempty"`
	OriginalTokens int               `json:"original_tokens,omitempty"`
	Pinned         bool              `json:"pinned"`
	PagedOut       bool              `json:"paged_out"`
	Summary        string            `json:"summary,omitempty"`
	Metadata       map[string]string `json:"metadata,omitempty"`
}

// computeDigest ensures SHA256 digest, hex string, and token estimations are populated.
func (c *ContextCell) computeDigest() {
	if c.Digest == [32]byte{} && len(c.Content) > 0 {
		c.Digest = sha256.Sum256(c.Content)
	}
	if c.DigestHex == "" && c.Digest != [32]byte{} {
		c.DigestHex = hex.EncodeToString(c.Digest[:])
	}
	if c.Tokens <= 0 {
		c.Tokens = EstimateTokens(c.Content)
	}
	if c.OriginalBytes <= 0 {
		c.OriginalBytes = len(c.Content)
	}
	if c.OriginalTokens <= 0 {
		c.OriginalTokens = c.Tokens
	}
}

// FormatDigestCard constructs a deterministic, digest-bound reference card for a paged-out cell.
func FormatDigestCard(tool string, digestHex string, origBytes int, origTokens int, query string, summary string) []byte {
	var sb strings.Builder
	sb.WriteString("[PAGED_OUT_TOOL_OUTPUT tool=")
	if tool != "" {
		sb.WriteString(tool)
	} else {
		sb.WriteString("tool")
	}
	sb.WriteString(" digest=sha256:")
	sb.WriteString(digestHex)
	fmt.Fprintf(&sb, " bytes=%d tokens=%d", origBytes, origTokens)
	if query != "" {
		sb.WriteString(" query=\"")
		sb.WriteString(query)
		sb.WriteString("\"")
	}
	sb.WriteString("]\n")
	sb.WriteString("Ref: cas://sha256:")
	sb.WriteString(digestHex)
	sb.WriteString("\n")
	if summary != "" {
		sb.WriteString("Summary: ")
		sb.WriteString(summary)
		sb.WriteString("\n")
	}
	sb.WriteString("Demand-Page: fault into context via PageIn(digest)\n")
	sb.WriteString("[/PAGED_OUT_TOOL_OUTPUT]")
	return []byte(sb.String())
}

// summarizeContent extracts a short, deterministic single-line preview of content.
func summarizeContent(content []byte, maxLen int) string {
	if len(content) == 0 {
		return ""
	}
	if maxLen <= 0 {
		maxLen = 96
	}
	s := strings.TrimSpace(string(content))
	if idx := strings.IndexByte(s, '\n'); idx != -1 && idx < maxLen {
		s = s[:idx]
	} else if len(s) > maxLen {
		s = s[:maxLen] + "..."
	}
	return strings.TrimSpace(s)
}

// ProjectedContext represents the rendered, budget-constrained projection of virtual context memory.
type ProjectedContext struct {
	Cells             []ContextCell `json:"cells"`
	TotalTokens       int           `json:"total_tokens"`
	PrefixTokens      int           `json:"prefix_tokens"`
	PrefixBytes       []byte        `json:"prefix_bytes"`
	RenderedBytes     []byte        `json:"rendered_bytes"`
	PinnedConstraints []ContextCell `json:"pinned_constraints"`
	NegativeFindings  []ContextCell `json:"negative_findings"`
	PagedOutCount     int           `json:"paged_out_count"`
	ResidentCount     int           `json:"resident_count"`
}

// String returns the full rendered text of the projected context.
func (p ProjectedContext) String() string {
	return string(p.RenderedBytes)
}

// Bytes returns the raw rendered bytes.
func (p ProjectedContext) Bytes() []byte {
	return p.RenderedBytes
}

// HasPrefixAnchor reports whether the projected bytes strictly start with the prefix bytes.
func (p ProjectedContext) HasPrefixAnchor(prefix []byte) bool {
	return bytes.HasPrefix(p.RenderedBytes, prefix)
}

// PinnedConstraintCount returns the count of preserved pinned constraints.
func (p ProjectedContext) PinnedConstraintCount() int {
	return len(p.PinnedConstraints)
}

// NegativeFindingCount returns the count of preserved negative search findings.
func (p ProjectedContext) NegativeFindingCount() int {
	return len(p.NegativeFindings)
}

// Default constants for VirtualContext.
const (
	DefaultRecentTurnsResident    = 3
	DefaultToolPageOutThreshold   = 64 // tokens; tool outputs older than recent window exceeding this are paged out
	DefaultInitialVirtualCapacity = 64
)

// VirtualContext coordinates structured virtual memory segmentation, demand-paging,
// and prefix-cache-invariant context projection over lossy harness compaction.
type VirtualContext struct {
	mu          sync.RWMutex
	cells       []ContextCell
	casStore    map[string][]byte // digestHex -> full content
	nextID      uint64
	prefixBytes []byte // frozen byte-prefix of immutable anchors
	recentTurns int    // interactive turns to keep resident in full fidelity
	maxTurn     int
}

// NewVirtualContext creates a new VirtualContext with default settings.
func NewVirtualContext() *VirtualContext {
	return NewVirtualContextWithConfig(DefaultRecentTurnsResident)
}

// NewVirtualContextWithConfig creates a VirtualContext with a custom recent-turn retention window.
func NewVirtualContextWithConfig(recentTurns int) *VirtualContext {
	if recentTurns < 1 {
		recentTurns = DefaultRecentTurnsResident
	}
	return &VirtualContext{
		cells:       make([]ContextCell, 0, DefaultInitialVirtualCapacity),
		casStore:    make(map[string][]byte),
		recentTurns: recentTurns,
	}
}

// AddCell inserts a ContextCell into virtual memory, indexing its SHA256 digest into the CAS store.
func (v *VirtualContext) AddCell(c ContextCell) uint64 {
	v.mu.Lock()
	defer v.mu.Unlock()

	id := atomic.AddUint64(&v.nextID, 1)
	c.ID = id
	c.computeDigest()

	// Enforce zero-loss invariants on typed cells
	if c.Kind.IsZeroLoss() {
		c.Pinned = true
	}

	// Index raw content into CAS store
	if len(c.Content) > 0 {
		cp := make([]byte, len(c.Content))
		copy(cp, c.Content)
		v.casStore[c.DigestHex] = cp
	}

	if c.Turn > v.maxTurn {
		v.maxTurn = c.Turn
	}

	v.cells = append(v.cells, c)

	// If a new prefix anchor cell is registered at turn 0, invalidate cached prefix bytes
	// so the initial prefix anchor is rendered completely. Once turns advance past 0,
	// the prefix anchor is strictly immutable.
	if c.Kind.IsPrefixAnchor() && c.Turn == 0 {
		v.prefixBytes = nil
	}

	return id
}

// AddSystemInstructions inserts an immutable system prompt into the prefix anchor.
func (v *VirtualContext) AddSystemInstructions(content []byte) uint64 {
	return v.AddCell(ContextCell{
		Turn:    0,
		Kind:    CellKindSystemInstructions,
		Role:    "system",
		Content: content,
		Pinned:  true,
	})
}

// AddUserConstraint inserts a pinned user constraint into virtual context memory.
func (v *VirtualContext) AddUserConstraint(content []byte, pinned bool) uint64 {
	return v.AddCell(ContextCell{
		Turn:    0,
		Kind:    CellKindUserConstraints,
		Role:    "user",
		Content: content,
		Pinned:  pinned,
	})
}

// AddToolOutput inserts a tool execution result into virtual memory.
func (v *VirtualContext) AddToolOutput(turn int, toolName string, content []byte) uint64 {
	return v.AddCell(ContextCell{
		Turn:     turn,
		Kind:     CellKindToolOutput,
		Role:     "tool",
		ToolName: toolName,
		Content:  content,
	})
}

// AddNegativeFinding records a verified absence or negative search result with zero-loss guarantee.
func (v *VirtualContext) AddNegativeFinding(turn int, query string, finding []byte) uint64 {
	return v.AddCell(ContextCell{
		Turn:     turn,
		Kind:     CellKindNegativeFinding,
		Role:     "tool",
		ToolName: "verifier",
		Query:    query,
		Content:  finding,
		Pinned:   true,
	})
}

// AddAssistantTurn records an assistant reasoning turn.
func (v *VirtualContext) AddAssistantTurn(turn int, content []byte) uint64 {
	return v.AddCell(ContextCell{
		Turn:    turn,
		Kind:    CellKindAssistantTurn,
		Role:    "assistant",
		Content: content,
	})
}

// AddUserTurn records a conversational user turn.
func (v *VirtualContext) AddUserTurn(turn int, content []byte) uint64 {
	return v.AddCell(ContextCell{
		Turn:    turn,
		Kind:    CellKindUserTurn,
		Role:    "user",
		Content: content,
	})
}

// PageIn resolves paged-out content from the content-addressed store by SHA256 digest.
func (v *VirtualContext) PageIn(digestHex string) ([]byte, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	digestHex = strings.TrimPrefix(digestHex, "sha256:")
	content, ok := v.casStore[digestHex]
	if !ok {
		return nil, fmt.Errorf("ctxmmu: page fault: digest %s not found in CAS store", digestHex)
	}
	cp := make([]byte, len(content))
	copy(cp, content)
	return cp, nil
}

// formatCell renders a single ContextCell into its deterministic wire format.
func formatCell(c *ContextCell) []byte {
	var sb strings.Builder
	switch c.Kind {
	case CellKindSystemInstructions:
		sb.WriteString("[SYSTEM_INSTRUCTIONS]\n")
		sb.Write(c.Content)
		sb.WriteString("\n[/SYSTEM_INSTRUCTIONS]")
	case CellKindUserConstraints:
		fmt.Fprintf(&sb, "[USER_CONSTRAINT id=%d]\n", c.ID)
		sb.Write(c.Content)
		sb.WriteString("\n[/USER_CONSTRAINT]")
	case CellKindNegativeFinding:
		fmt.Fprintf(&sb, "[NEGATIVE_FINDING turn=%d query=\"%s\"]\n", c.Turn, c.Query)
		sb.Write(c.Content)
		sb.WriteString("\n[/NEGATIVE_FINDING]")
	case CellKindToolOutput:
		if c.PagedOut {
			sb.Write(c.Content) // already formatted as digest card
		} else {
			fmt.Fprintf(&sb, "[TOOL_OUTPUT turn=%d tool=%s]\n", c.Turn, c.ToolName)
			sb.Write(c.Content)
			sb.WriteString("\n[/TOOL_OUTPUT]")
		}
	case CellKindAssistantTurn:
		if c.PagedOut {
			fmt.Fprintf(&sb, "[PAGED_OUT_ASSISTANT turn=%d digest=sha256:%s bytes=%d]\n", c.Turn, c.DigestHex, c.OriginalBytes)
			sb.WriteString("Ref: cas://sha256:")
			sb.WriteString(c.DigestHex)
			sb.WriteString("\n[/PAGED_OUT_ASSISTANT]")
		} else {
			fmt.Fprintf(&sb, "[ASSISTANT turn=%d]\n", c.Turn)
			sb.Write(c.Content)
			sb.WriteString("\n[/ASSISTANT]")
		}
	case CellKindUserTurn:
		fmt.Fprintf(&sb, "[USER turn=%d]\n", c.Turn)
		sb.Write(c.Content)
		sb.WriteString("\n[/USER]")
	default:
		sb.Write(c.Content)
	}
	return []byte(sb.String())
}

// ProjectView renders a budget-constrained context view, projecting older tool outputs
// to digest-bound cards while preserving immutable prefix anchors and negative findings.
func (v *VirtualContext) ProjectView(maxTokens int) ProjectedContext {
	v.mu.Lock()
	defer v.mu.Unlock()

	if maxTokens <= 0 {
		maxTokens = 1024 * 1024 // unconstrained
	}

	// 1. Partition cells into Prefix Anchors vs Remainder
	var prefixCells []ContextCell
	var remainderCells []ContextCell
	var pinnedConstraints []ContextCell
	var negativeFindings []ContextCell

	for _, c := range v.cells {
		if c.Kind.IsPrefixAnchor() && c.Turn == 0 {
			prefixCells = append(prefixCells, c)
			if c.Kind == CellKindUserConstraints || (c.Pinned && c.Kind != CellKindSystemInstructions && c.Kind != CellKindNegativeFinding) {
				pinnedConstraints = append(pinnedConstraints, c)
			}
		} else {
			remainderCells = append(remainderCells, c)
			if c.Kind == CellKindUserConstraints || (c.Pinned && c.Kind != CellKindSystemInstructions && c.Kind != CellKindNegativeFinding) {
				pinnedConstraints = append(pinnedConstraints, c)
			}
			if c.Kind == CellKindNegativeFinding {
				negativeFindings = append(negativeFindings, c)
			}
		}
	}

	// 2. Render Prefix Anchor deterministically
	if len(v.prefixBytes) == 0 {
		var pBuf bytes.Buffer
		for i, c := range prefixCells {
			if i > 0 {
				pBuf.WriteString("\n\n")
			}
			pBuf.Write(formatCell(&c))
		}
		v.prefixBytes = pBuf.Bytes()
	}

	prefixTokens := EstimateTokens(v.prefixBytes)

	// 3. Clone remainder cells to evaluate projection
	projectedRemainder := make([]ContextCell, len(remainderCells))
	copy(projectedRemainder, remainderCells)

	// Determine active turn boundary
	recentWindowFloor := v.maxTurn - v.recentTurns + 1
	if recentWindowFloor < 1 {
		recentWindowFloor = 1
	}

	// Helper to calculate total tokens of current projection
	calcTokens := func() int {
		tok := prefixTokens
		for i := range projectedRemainder {
			tok += projectedRemainder[i].Tokens
		}
		return tok
	}

	// Pass A: If over budget, page out older tool outputs outside the recent window
	if calcTokens() > maxTokens {
		for i := range projectedRemainder {
			c := &projectedRemainder[i]
			if c.Kind == CellKindToolOutput && !c.PagedOut && c.Turn < recentWindowFloor {
				sum := c.Summary
				if sum == "" {
					sum = summarizeContent(c.Content, 96)
				}
				card := FormatDigestCard(c.ToolName, c.DigestHex, c.OriginalBytes, c.OriginalTokens, c.Query, sum)
				c.PagedOut = true
				c.Content = card
				c.Tokens = EstimateTokens(card)
				if calcTokens() <= maxTokens {
					break
				}
			}
		}
	}

	// Pass B: If still over budget, page out remaining tool outputs from oldest to newest
	if calcTokens() > maxTokens {
		for i := range projectedRemainder {
			c := &projectedRemainder[i]
			if c.Kind == CellKindToolOutput && !c.PagedOut {
				sum := c.Summary
				if sum == "" {
					sum = summarizeContent(c.Content, 96)
				}
				card := FormatDigestCard(c.ToolName, c.DigestHex, c.OriginalBytes, c.OriginalTokens, c.Query, sum)
				c.PagedOut = true
				c.Content = card
				c.Tokens = EstimateTokens(card)

				if calcTokens() <= maxTokens {
					break
				}
			}
		}
	}

	// Pass C: If still over budget, page out older assistant turns (excluding pinned / negative findings)
	if calcTokens() > maxTokens {
		for i := range projectedRemainder {
			c := &projectedRemainder[i]
			if c.Kind == CellKindAssistantTurn && !c.PagedOut && c.Turn < recentWindowFloor {
				c.PagedOut = true
				c.Tokens = EstimateTokens([]byte(fmt.Sprintf("[PAGED_OUT_ASSISTANT turn=%d digest=sha256:%s bytes=%d]\n", c.Turn, c.DigestHex, c.OriginalBytes)))
				if calcTokens() <= maxTokens {
					break
				}
			}
		}
	}

	// 4. Render the Full Projected Context
	var outBuf bytes.Buffer
	outBuf.Write(v.prefixBytes)

	allProjectedCells := make([]ContextCell, 0, len(prefixCells)+len(projectedRemainder))
	allProjectedCells = append(allProjectedCells, prefixCells...)
	allProjectedCells = append(allProjectedCells, projectedRemainder...)

	pagedCount := 0
	residentCount := 0

	for i := range projectedRemainder {
		outBuf.WriteString("\n\n")
		outBuf.Write(formatCell(&projectedRemainder[i]))
		if projectedRemainder[i].PagedOut {
			pagedCount++
		} else {
			residentCount++
		}
	}

	// Count prefix cells as resident
	residentCount += len(prefixCells)

	rendered := outBuf.Bytes()
	totalTokens := EstimateTokens(rendered)

	return ProjectedContext{
		Cells:             allProjectedCells,
		TotalTokens:       totalTokens,
		PrefixTokens:      prefixTokens,
		PrefixBytes:       v.prefixBytes,
		RenderedBytes:     rendered,
		PinnedConstraints: pinnedConstraints,
		NegativeFindings:  negativeFindings,
		PagedOutCount:     pagedCount,
		ResidentCount:     residentCount,
	}
}
