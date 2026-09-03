package agentopt

import (
	"fmt"
	"strings"
)

// Family 5: Context-window management & compression.
//
// Pinned sliding-window context reduction preserves immutable top-of-prompt
// context (system prompt, registered tool schemas, objective declaration)
// and recent active dialog turns (tail context). When combined token
// consumption exceeds the configured budget ceiling, intermediate turn bodies
// are folded into structured summary receipts (FoldedTurnReceipt).

const (
	// DefaultTailSize defines the default number of recent turns pinned at the tail.
	DefaultTailSize = 4
	// DefaultMinTailSize defines the minimum number of recent turns pinned under pressure.
	DefaultMinTailSize = 1
	// DefaultSummaryBudget defines the token budget target for folded turn receipts.
	DefaultSummaryBudget = 200
)

// estimateToolSchemaTokens estimates token weight for a tool specification.
func estimateToolSchemaTokens(ts ToolSchema) int {
	tokens := EstimateTokens(ts.Name) + EstimateTokens(ts.Description)
	for propName, prop := range ts.Properties {
		tokens += EstimateTokens(propName) + EstimateTokens(string(prop.Type))
		for _, e := range prop.Enum {
			tokens += EstimateTokens(e)
		}
	}
	for _, req := range ts.Required {
		tokens += EstimateTokens(req)
	}
	return tokens
}

// PinnedPreamble encapsulates the immutable top-of-prompt context that must
// remain pinned across all reduction passes: system prompt, tool schemas,
// and objective declarations.
type PinnedPreamble struct {
	SystemPrompt         string            `json:"system_prompt"`
	ToolSchemas          []string          `json:"tool_schemas,omitempty"`
	Tools                []ToolSchema      `json:"tools,omitempty"`
	ObjectiveDeclaration string            `json:"objective_declaration,omitempty"`
	Objective            string            `json:"objective,omitempty"`
	Tokens               int               `json:"tokens,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
}

// GetObjective returns the declared objective string.
func (h PinnedPreamble) GetObjective() string {
	if h.ObjectiveDeclaration != "" {
		return h.ObjectiveDeclaration
	}
	return h.Objective
}

// TotalTokens calculates the combined token count for the head context.
func (h PinnedPreamble) TotalTokens() int {
	if h.Tokens > 0 {
		return h.Tokens
	}
	tokens := EstimateTokens(h.SystemPrompt)
	if obj := h.GetObjective(); obj != "" {
		tokens += EstimateTokens(obj)
	}
	for _, s := range h.ToolSchemas {
		tokens += EstimateTokens(s)
	}
	for _, t := range h.Tools {
		tokens += estimateToolSchemaTokens(t)
	}
	return tokens
}

// ConversationTurn represents a single dialog turn or tool execution event.
type ConversationTurn struct {
	Index      int               `json:"index"`
	Role       string            `json:"role"`
	Content    string            `json:"content"`
	ToolCalls  []string          `json:"tool_calls,omitempty"`
	ToolResult string            `json:"tool_result,omitempty"`
	Tokens     int               `json:"tokens,omitempty"`
	Timestamp  string            `json:"timestamp,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// TotalTokens computes or estimates token consumption for this conversation turn.
func (t ConversationTurn) TotalTokens() int {
	if t.Tokens > 0 {
		return t.Tokens
	}
	tokens := EstimateTokens(t.Role) + EstimateTokens(t.Content)
	if t.ToolResult != "" {
		tokens += EstimateTokens(t.ToolResult)
	}
	for _, tc := range t.ToolCalls {
		tokens += EstimateTokens(tc)
	}
	return tokens
}

// FoldedTurnReceipt encapsulates the structured summary of a contiguous sequence
// of intermediate conversation turns that have been reduced to conserve context budget.
type FoldedTurnReceipt struct {
	StartIndex       int      `json:"start_index"`
	EndIndex         int      `json:"end_index"`
	TurnCount        int      `json:"turn_count"`
	Summary          string   `json:"summary"`
	KeyEvents        []string `json:"key_events,omitempty"`
	ToolsInvoked     []string `json:"tools_invoked,omitempty"`
	OriginalTokens   int      `json:"original_tokens"`
	FoldedTokens     int      `json:"folded_tokens"`
	TokensSaved      int      `json:"tokens_saved"`
	CompressionRatio float64  `json:"compression_ratio"`
}

// FormatContent formats the receipt into a legible text representation.
func (r FoldedTurnReceipt) FormatContent() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("[Folded Turn Receipt: turns %d..%d (%d turns folded, %d -> %d tokens)]\n",
		r.StartIndex, r.EndIndex, r.TurnCount, r.OriginalTokens, r.FoldedTokens))
	if r.Summary != "" {
		sb.WriteString("Summary: ")
		sb.WriteString(r.Summary)
		sb.WriteString("\n")
	}
	if len(r.ToolsInvoked) > 0 {
		sb.WriteString(fmt.Sprintf("Tools invoked: %s\n", strings.Join(r.ToolsInvoked, ", ")))
	}
	if len(r.KeyEvents) > 0 {
		sb.WriteString("Key events:\n")
		for _, ev := range r.KeyEvents {
			sb.WriteString(fmt.Sprintf("- %s\n", ev))
		}
	}
	return strings.TrimSpace(sb.String())
}

// TotalTokens returns the estimated tokens used by the receipt.
func (r FoldedTurnReceipt) TotalTokens() int {
	if r.FoldedTokens > 0 {
		return r.FoldedTokens
	}
	return EstimateTokens(r.FormatContent())
}

// AsTurn converts the receipt into a synthetic ConversationTurn for active prompt assembly.
func (r FoldedTurnReceipt) AsTurn() ConversationTurn {
	content := r.FormatContent()
	tokens := r.FoldedTokens
	if tokens <= 0 {
		tokens = EstimateTokens(content)
	}
	return ConversationTurn{
		Index:   r.StartIndex,
		Role:    "system",
		Content: content,
		Tokens:  tokens,
		Metadata: map[string]string{
			"folded_receipt": "true",
			"start_index":    fmt.Sprintf("%d", r.StartIndex),
			"end_index":      fmt.Sprintf("%d", r.EndIndex),
			"turn_count":     fmt.Sprintf("%d", r.TurnCount),
		},
	}
}

// ReducedWindowResult captures the outcome of context window reduction,
// holding the immutable head, pinned tail turns, folded intermediate turn receipts,
// and the active turns ready for dispatch.
type ReducedWindowResult struct {
	Head              PinnedPreamble      `json:"head"`
	PinnedTail        []ConversationTurn  `json:"pinned_tail"`
	FoldedReceipts    []FoldedTurnReceipt `json:"folded_receipts,omitempty"`
	IntermediateTurns []ConversationTurn  `json:"intermediate_turns,omitempty"`
	ActiveTurns       []ConversationTurn  `json:"active_turns"`
	TotalTokens       int                 `json:"total_tokens"`
	HeadTokens        int                 `json:"head_tokens"`
	TailTokens        int                 `json:"tail_tokens"`
	FoldedTokens      int                 `json:"folded_tokens"`
	TokensSaved       int                 `json:"tokens_saved"`
	OriginalTokens    int                 `json:"original_tokens"`
	ReductionApplied  bool                `json:"reduction_applied"`
	TurnsPreserved    int                 `json:"turns_preserved"`
	TurnsFolded       int                 `json:"turns_folded"`
}

// WindowReductionResult is an alias for ReducedWindowResult.
type WindowReductionResult = ReducedWindowResult

// SlidingWindowReducer manages pinned sliding-window context reduction.
type SlidingWindowReducer struct {
	TailSize         int
	MinTailSize      int
	SummaryBudget    int
	CustomSummarizer func(turns []ConversationTurn, targetBudget int) FoldedTurnReceipt
}

// ReducerOption configures options for SlidingWindowReducer.
type ReducerOption func(*SlidingWindowReducer)

// WithTailSize configures the desired number of recent turns to pin at the tail.
func WithTailSize(n int) ReducerOption {
	return func(r *SlidingWindowReducer) {
		if n > 0 {
			r.TailSize = n
		}
	}
}

// WithMinTailSize configures the minimum turns to retain under extreme context pressure.
func WithMinTailSize(n int) ReducerOption {
	return func(r *SlidingWindowReducer) {
		if n > 0 {
			r.MinTailSize = n
		}
	}
}

// WithSummaryBudget sets the target token budget for folded turn receipts.
func WithSummaryBudget(tokens int) ReducerOption {
	return func(r *SlidingWindowReducer) {
		if tokens > 0 {
			r.SummaryBudget = tokens
		}
	}
}

// WithSummarizer configures a custom summarizer function for intermediate turns.
func WithSummarizer(fn func(turns []ConversationTurn, targetBudget int) FoldedTurnReceipt) ReducerOption {
	return func(r *SlidingWindowReducer) {
		r.CustomSummarizer = fn
	}
}

// NewSlidingWindowReducer creates a new SlidingWindowReducer.
func NewSlidingWindowReducer(tailSize int, opts ...ReducerOption) *SlidingWindowReducer {
	if tailSize <= 0 {
		tailSize = DefaultTailSize
	}
	r := &SlidingWindowReducer{
		TailSize:      tailSize,
		MinTailSize:   DefaultMinTailSize,
		SummaryBudget: DefaultSummaryBudget,
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

// ReduceWindow applies pinned sliding-window reduction to the context window.
// It guarantees that the immutable PinnedPreamble (system prompt, tool schemas,
// objective declaration) and recent tail dialog turns remain pinned.
// Intermediate turns are folded into structured FoldedTurnReceipts when
// total tokens exceed maxTokens.
func (r *SlidingWindowReducer) ReduceWindow(head PinnedPreamble, turns []ConversationTurn, maxTokens int) ReducedWindowResult {
	if r == nil {
		r = NewSlidingWindowReducer(DefaultTailSize)
	}
	tailSize := r.TailSize
	if tailSize <= 0 {
		tailSize = DefaultTailSize
	}
	minTailSize := r.MinTailSize
	if minTailSize <= 0 {
		minTailSize = DefaultMinTailSize
	}
	if minTailSize > tailSize {
		minTailSize = tailSize
	}

	headTokens := head.TotalTokens()
	totalTurnTokens := 0
	for _, t := range turns {
		totalTurnTokens += t.TotalTokens()
	}
	originalTokens := headTokens + totalTurnTokens

	if len(turns) == 0 {
		return ReducedWindowResult{
			Head:             head,
			HeadTokens:       headTokens,
			TotalTokens:      headTokens,
			OriginalTokens:   headTokens,
			ReductionApplied: false,
		}
	}

	// If maxTokens is positive and total tokens already fit within budget, no reduction needed.
	if maxTokens > 0 && originalTokens <= maxTokens {
		tailCount := tailSize
		if len(turns) < tailCount {
			tailCount = len(turns)
		}
		splitIdx := len(turns) - tailCount
		intermediate := turns[:splitIdx]
		tail := turns[splitIdx:]

		tailTokens := 0
		for _, t := range tail {
			tailTokens += t.TotalTokens()
		}

		return ReducedWindowResult{
			Head:              head,
			PinnedTail:        tail,
			IntermediateTurns: intermediate,
			ActiveTurns:       turns,
			TotalTokens:       originalTokens,
			HeadTokens:        headTokens,
			TailTokens:        tailTokens,
			OriginalTokens:    originalTokens,
			ReductionApplied:  false,
			TurnsPreserved:    len(turns),
			TurnsFolded:       0,
		}
	}

	// Budget exceeded (or forced reduction):
	// Determine how many tail turns to pin. Under extreme context pressure,
	// if headTokens + tailTokens > maxTokens, shrink tail down toward minTailSize.
	tailCount := tailSize
	if len(turns) < tailCount {
		tailCount = len(turns)
	}

	if maxTokens > 0 {
		for tailCount > minTailSize {
			curTailTokens := 0
			for _, t := range turns[len(turns)-tailCount:] {
				curTailTokens += t.TotalTokens()
			}
			// Minimal receipt ~20 tokens
			if headTokens+curTailTokens+20 <= maxTokens {
				break
			}
			tailCount--
		}
	}

	splitIdx := len(turns) - tailCount
	if splitIdx <= 0 {
		// All turns are within the pinned tail.
		tail := turns
		tailTokens := totalTurnTokens
		return ReducedWindowResult{
			Head:             head,
			PinnedTail:       tail,
			ActiveTurns:      tail,
			TotalTokens:      originalTokens,
			HeadTokens:       headTokens,
			TailTokens:       tailTokens,
			OriginalTokens:   originalTokens,
			ReductionApplied: false,
			TurnsPreserved:   len(turns),
			TurnsFolded:      0,
		}
	}

	intermediate := turns[:splitIdx]
	tail := turns[splitIdx:]

	tailTokens := 0
	for _, t := range tail {
		tailTokens += t.TotalTokens()
	}

	summaryBudget := r.SummaryBudget
	if summaryBudget <= 0 {
		summaryBudget = DefaultSummaryBudget
	}
	if maxTokens > 0 {
		availBudget := maxTokens - headTokens - tailTokens
		if availBudget > 0 && availBudget < summaryBudget {
			summaryBudget = availBudget
		}
	}

	receipt := r.foldIntermediateTurns(intermediate, summaryBudget)

	receiptTurn := receipt.AsTurn()
	activeTurns := make([]ConversationTurn, 0, 1+len(tail))
	activeTurns = append(activeTurns, receiptTurn)
	activeTurns = append(activeTurns, tail...)

	foldedTokens := receiptTurn.TotalTokens()
	newTotalTokens := headTokens + foldedTokens + tailTokens
	tokensSaved := originalTokens - newTotalTokens
	if tokensSaved < 0 {
		tokensSaved = 0
	}

	return ReducedWindowResult{
		Head:              head,
		PinnedTail:        tail,
		FoldedReceipts:    []FoldedTurnReceipt{receipt},
		IntermediateTurns: nil,
		ActiveTurns:       activeTurns,
		TotalTokens:       newTotalTokens,
		HeadTokens:        headTokens,
		TailTokens:        tailTokens,
		FoldedTokens:      foldedTokens,
		TokensSaved:       tokensSaved,
		OriginalTokens:    originalTokens,
		ReductionApplied:  true,
		TurnsPreserved:    len(tail),
		TurnsFolded:       len(intermediate),
	}
}

// ShrinkWindow reduces the conversation context window to fit maxTokens,
// returning WindowReductionResult (an alias for ReducedWindowResult).
func (r *SlidingWindowReducer) ShrinkWindow(head PinnedPreamble, turns []ConversationTurn, maxTokens int) WindowReductionResult {
	return r.ReduceWindow(head, turns, maxTokens)
}

// ReduceWindow applies sliding-window reduction using default reducer settings.
func ReduceWindow(head PinnedPreamble, turns []ConversationTurn, maxTokens int) ReducedWindowResult {
	return NewSlidingWindowReducer(DefaultTailSize).ReduceWindow(head, turns, maxTokens)
}

// ShrinkWindow applies sliding-window reduction using default reducer settings.
func ShrinkWindow(head PinnedPreamble, turns []ConversationTurn, maxTokens int) WindowReductionResult {
	return NewSlidingWindowReducer(DefaultTailSize).ShrinkWindow(head, turns, maxTokens)
}

func (r *SlidingWindowReducer) foldIntermediateTurns(turns []ConversationTurn, targetBudget int) FoldedTurnReceipt {
	if len(turns) == 0 {
		return FoldedTurnReceipt{}
	}
	if r.CustomSummarizer != nil {
		return r.CustomSummarizer(turns, targetBudget)
	}

	startIndex := turns[0].Index
	endIndex := turns[len(turns)-1].Index
	turnCount := len(turns)

	origTokens := 0
	for _, t := range turns {
		origTokens += t.TotalTokens()
	}

	toolSet := make(map[string]bool)
	var toolsInvoked []string
	for _, t := range turns {
		for _, tc := range t.ToolCalls {
			name := extractToolName(tc)
			if name != "" && !toolSet[name] {
				toolSet[name] = true
				toolsInvoked = append(toolsInvoked, name)
			}
		}
		if t.Role == "tool" {
			if tn, ok := t.Metadata["tool_name"]; ok && tn != "" && !toolSet[tn] {
				toolSet[tn] = true
				toolsInvoked = append(toolsInvoked, tn)
			}
		}
	}

	var keyEvents []string
	for _, t := range turns {
		if isFoldedReceiptTurn(t) {
			keyEvents = append(keyEvents, fmt.Sprintf("Prior folded block (turns %s..%s)", t.Metadata["start_index"], t.Metadata["end_index"]))
			continue
		}
		switch t.Role {
		case "user":
			snippet := truncateText(cleanLine(t.Content), 80)
			if snippet != "" {
				keyEvents = append(keyEvents, fmt.Sprintf("User: %q", snippet))
			}
		case "assistant":
			if len(t.ToolCalls) > 0 {
				keyEvents = append(keyEvents, fmt.Sprintf("Assistant invoked %s", strings.Join(t.ToolCalls, ", ")))
			} else {
				snippet := truncateText(cleanLine(t.Content), 80)
				if snippet != "" {
					keyEvents = append(keyEvents, fmt.Sprintf("Assistant: %q", snippet))
				}
			}
		case "tool":
			resSnippet := truncateText(cleanLine(t.ToolResult+t.Content), 60)
			if resSnippet != "" {
				keyEvents = append(keyEvents, fmt.Sprintf("Tool result: %s", resSnippet))
			}
		}
	}

	var summaryParts []string
	summaryParts = append(summaryParts, fmt.Sprintf("Folded %d intermediate turns spanning indexes %d..%d.", turnCount, startIndex, endIndex))
	if len(toolsInvoked) > 0 {
		summaryParts = append(summaryParts, fmt.Sprintf("Tools executed: %s.", strings.Join(toolsInvoked, ", ")))
	}
	summary := strings.Join(summaryParts, " ")

	receipt := FoldedTurnReceipt{
		StartIndex:     startIndex,
		EndIndex:       endIndex,
		TurnCount:      turnCount,
		Summary:        summary,
		KeyEvents:      keyEvents,
		ToolsInvoked:   toolsInvoked,
		OriginalTokens: origTokens,
	}

	if targetBudget > 0 {
		currentTokens := EstimateTokens(receipt.FormatContent())
		for len(receipt.KeyEvents) > 0 && currentTokens > targetBudget {
			receipt.KeyEvents = receipt.KeyEvents[:len(receipt.KeyEvents)-1]
			currentTokens = EstimateTokens(receipt.FormatContent())
		}
		if currentTokens > targetBudget {
			receipt.KeyEvents = nil
			receipt.Summary = fmt.Sprintf("Folded %d intermediate turns (%d..%d).", turnCount, startIndex, endIndex)
		}
	}

	receipt.FoldedTokens = EstimateTokens(receipt.FormatContent())
	saved := origTokens - receipt.FoldedTokens
	if saved < 0 {
		saved = 0
	}
	receipt.TokensSaved = saved
	if origTokens > 0 {
		receipt.CompressionRatio = float64(saved) / float64(origTokens)
	}

	return receipt
}

func extractToolName(raw string) string {
	raw = strings.TrimSpace(raw)
	if idx := strings.IndexAny(raw, "({ \t\n"); idx != -1 {
		raw = raw[:idx]
	}
	return strings.Trim(raw, `"'`)
}

func truncateText(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen <= 3 {
		return s[:maxLen]
	}
	return s[:maxLen-3] + "..."
}

func cleanLine(s string) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.Join(strings.Fields(s), " ")
	return strings.TrimSpace(s)
}

func isFoldedReceiptTurn(t ConversationTurn) bool {
	if t.Metadata != nil && t.Metadata["folded_receipt"] == "true" {
		return true
	}
	return strings.HasPrefix(t.Content, "[Folded Turn Receipt:")
}
