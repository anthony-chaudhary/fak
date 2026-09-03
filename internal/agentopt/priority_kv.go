package agentopt

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"
)

// Priority defines retention priority levels for KV blocks.
type Priority int

const (
	// StringParameter and IntegerParameter provide parameter type compatibility for tool schemas.
	StringParameter  = TypeString
	IntegerParameter = TypeInteger
)

const (
	// PriorityOutput is low retention priority for large model or tool outputs.
	PriorityOutput Priority = 10
	// PriorityTurn is normal retention priority for conversational turns.
	PriorityTurn Priority = 20
	// PriorityTool is high retention priority for shared tool definitions.
	PriorityTool Priority = 30
	// PriorityRoot is highest retention priority for system instructions.
	PriorityRoot Priority = 40
	// PrioritySystem is an alias for PriorityRoot.
	PrioritySystem Priority = PriorityRoot
)

// String returns a readable representation of the priority level.
func (p Priority) String() string {
	switch p {
	case PriorityRoot:
		return "root"
	case PriorityTool:
		return "tool"
	case PriorityTurn:
		return "turn"
	case PriorityOutput:
		return "output"
	default:
		return fmt.Sprintf("priority(%d)", p)
	}
}

// KVBlock represents a resident key-value block holding tokens and retention attributes.
type KVBlock struct {
	BlockID    string    `json:"block_id"`
	TokenCount int       `json:"token_count"`
	Priority   Priority  `json:"priority"`
	LastAccess time.Time `json:"last_access"`
	Pinned     bool      `json:"pinned"`
	Kind       string    `json:"kind,omitempty"`
	Content    string    `json:"content,omitempty"`
}

// NewRootBlock creates a pinned system root instruction block with highest priority.
func NewRootBlock(blockID string, tokenCount int) KVBlock {
	return KVBlock{
		BlockID:    blockID,
		TokenCount: tokenCount,
		Priority:   PriorityRoot,
		LastAccess: time.Now(),
		Pinned:     true,
		Kind:       "root_instruction",
	}
}

// NewToolBlock creates a tool definition block with high priority.
func NewToolBlock(blockID string, tokenCount int, pinned bool) KVBlock {
	return KVBlock{
		BlockID:    blockID,
		TokenCount: tokenCount,
		Priority:   PriorityTool,
		LastAccess: time.Now(),
		Pinned:     pinned,
		Kind:       "tool_definition",
	}
}

// NewTurnBlock creates a conversational turn block with normal priority.
func NewTurnBlock(blockID string, tokenCount int) KVBlock {
	return KVBlock{
		BlockID:    blockID,
		TokenCount: tokenCount,
		Priority:   PriorityTurn,
		LastAccess: time.Now(),
		Pinned:     false,
		Kind:       "turn",
	}
}

// NewOutputBlock creates a tool or model output block with low priority.
func NewOutputBlock(blockID string, tokenCount int) KVBlock {
	return KVBlock{
		BlockID:    blockID,
		TokenCount: tokenCount,
		Priority:   PriorityOutput,
		LastAccess: time.Now(),
		Pinned:     false,
		Kind:       "output",
	}
}

// KVPruneReport provides token-level and block-level accounting when pruning KV blocks.
type KVPruneReport struct {
	InitialTokens   int       `json:"initial_tokens"`
	PrunedTokens    int       `json:"pruned_tokens"`
	RemainingTokens int       `json:"remaining_tokens"`
	MaxTokens       int       `json:"max_tokens"`
	InitialBlocks   int       `json:"initial_blocks"`
	PrunedBlocks    []KVBlock `json:"pruned_blocks"`
	RemainingBlocks int       `json:"remaining_blocks"`
	RemovedBlockIDs []string  `json:"removed_block_ids"`
}

// PriorityKVTable manages priority-aware KV blocks subject to token budget limits.
type PriorityKVTable struct {
	mu           sync.RWMutex
	blocks       map[string]KVBlock
	totalTokens  int
	lastPruned   []KVBlock
	lastReport   PruneReport
	lastKVReport KVPruneReport
}

// PriorityKVManager is an alias for PriorityKVTable.
type PriorityKVManager = PriorityKVTable

// NewPriorityKVTable creates a new empty PriorityKVTable.
func NewPriorityKVTable() *PriorityKVTable {
	return &PriorityKVTable{
		blocks: make(map[string]KVBlock),
	}
}

// NewPriorityKVManager creates a new PriorityKVManager instance.
func NewPriorityKVManager() *PriorityKVTable {
	return NewPriorityKVTable()
}

// AddBlock adds or updates a KVBlock in the table.
// If the block has PriorityRoot, it is automatically marked pinned.
func (t *PriorityKVTable) AddBlock(block KVBlock) error {
	if block.BlockID == "" {
		return errors.New("block ID cannot be empty")
	}
	if block.TokenCount < 0 {
		return errors.New("token count cannot be negative")
	}
	if block.LastAccess.IsZero() {
		block.LastAccess = time.Now()
	}
	if block.Priority >= PriorityRoot {
		block.Pinned = true
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if existing, found := t.blocks[block.BlockID]; found {
		t.totalTokens -= existing.TokenCount
	}
	t.blocks[block.BlockID] = block
	t.totalTokens += block.TokenCount
	return nil
}

// TouchBlock marks a resident block as recently accessed, updating its LastAccess timestamp.
func (t *PriorityKVTable) TouchBlock(blockID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	b, ok := t.blocks[blockID]
	if !ok {
		return false
	}
	b.LastAccess = time.Now()
	t.blocks[blockID] = b
	return true
}

// PinBlock pins a resident block, making it immune to budget pruning.
func (t *PriorityKVTable) PinBlock(blockID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	b, ok := t.blocks[blockID]
	if !ok {
		return false
	}
	b.Pinned = true
	t.blocks[blockID] = b
	return true
}

// UnpinBlock unpins a block. Blocks with PriorityRoot remain pinned.
func (t *PriorityKVTable) UnpinBlock(blockID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	b, ok := t.blocks[blockID]
	if !ok {
		return false
	}
	if b.Priority >= PriorityRoot {
		return false
	}
	b.Pinned = false
	t.blocks[blockID] = b
	return true
}

// DiscardBlock manually removes an unpinned block from the table.
func (t *PriorityKVTable) DiscardBlock(blockID string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	b, ok := t.blocks[blockID]
	if !ok {
		return false
	}
	if b.Pinned || b.Priority >= PriorityRoot {
		return false
	}
	t.totalTokens -= b.TokenCount
	delete(t.blocks, blockID)
	return true
}

// GetBlock returns a copy of a resident block by ID.
func (t *PriorityKVTable) GetBlock(blockID string) (KVBlock, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	b, ok := t.blocks[blockID]
	return b, ok
}

// HasBlock checks whether a block ID is currently resident.
func (t *PriorityKVTable) HasBlock(blockID string) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	_, ok := t.blocks[blockID]
	return ok
}

// TotalTokens returns the total token count of all resident blocks.
func (t *PriorityKVTable) TotalTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.totalTokens
}

// BlockCount returns the number of resident blocks.
func (t *PriorityKVTable) BlockCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.blocks)
}

// Blocks returns a snapshot of all currently resident blocks.
func (t *PriorityKVTable) Blocks() []KVBlock {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]KVBlock, 0, len(t.blocks))
	for _, b := range t.blocks {
		out = append(out, b)
	}
	return out
}

// PrunedBlocks returns the blocks discarded during the most recent budget enforcement.
func (t *PriorityKVTable) PrunedBlocks() []KVBlock {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]KVBlock, len(t.lastPruned))
	copy(out, t.lastPruned)
	return out
}

// PrunedBlockIDs returns the block IDs discarded during the most recent budget enforcement.
func (t *PriorityKVTable) PrunedBlockIDs() []string {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]string, len(t.lastReport.RemovedNodes))
	copy(out, t.lastReport.RemovedNodes)
	return out
}

// LastPruneReport returns the summary report of the most recent budget enforcement.
func (t *PriorityKVTable) LastPruneReport() PruneReport {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.lastReport
}

// LastKVPruneReport returns the detailed KV metrics from the most recent budget enforcement.
func (t *PriorityKVTable) LastKVPruneReport() KVPruneReport {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.lastKVReport
}

// EnforceBudget prunes resident blocks until total tokens fall within maxTokens.
// Pinned blocks and root instructions are immune to pruning.
// Discards intermediate conversational turns and large tool outputs first.
func (t *PriorityKVTable) EnforceBudget(maxTokens int) PruneReport {
	t.mu.Lock()
	defer t.mu.Unlock()

	initialTokens := t.totalTokens
	initialBlocks := len(t.blocks)

	if maxTokens < 0 {
		maxTokens = 0
	}

	if t.totalTokens <= maxTokens {
		report := PruneReport{
			OriginalNodesCount:  initialBlocks,
			PrunedNodesCount:    0,
			RemainingNodesCount: initialBlocks,
			RemovedNodes:        nil,
			ExecutionHopsSaved:  0,
			InitialUtility:      float64(initialTokens),
			FinalUtility:        float64(initialTokens),
		}
		t.lastPruned = nil
		t.lastReport = report
		t.lastKVReport = KVPruneReport{
			InitialTokens:   initialTokens,
			PrunedTokens:    0,
			RemainingTokens: initialTokens,
			MaxTokens:       maxTokens,
			InitialBlocks:   initialBlocks,
			PrunedBlocks:    nil,
			RemainingBlocks: initialBlocks,
			RemovedBlockIDs: nil,
		}
		return report
	}

	// Identify pruneable candidates: unpinned and Priority < PriorityRoot
	candidates := make([]KVBlock, 0, len(t.blocks))
	for _, b := range t.blocks {
		if !b.Pinned && b.Priority < PriorityRoot {
			candidates = append(candidates, b)
		}
	}

	// Sort candidates by discard precedence:
	// 1. Lower priority discarded first (PriorityOutput < PriorityTurn < PriorityTool).
	// 2. Tie-break within PriorityOutput: larger token count discarded first.
	// 3. Tie-break for others: older LastAccess (LRU) discarded first.
	// 4. Secondary tie-break: larger token count discarded first.
	// 5. Final tie-break: deterministic by BlockID.
	sort.Slice(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if a.Priority != b.Priority {
			return a.Priority < b.Priority
		}
		if a.Priority == PriorityOutput {
			if a.TokenCount != b.TokenCount {
				return a.TokenCount > b.TokenCount
			}
			if !a.LastAccess.Equal(b.LastAccess) {
				return a.LastAccess.Before(b.LastAccess)
			}
			return a.BlockID < b.BlockID
		}
		if !a.LastAccess.Equal(b.LastAccess) {
			return a.LastAccess.Before(b.LastAccess)
		}
		if a.TokenCount != b.TokenCount {
			return a.TokenCount > b.TokenCount
		}
		return a.BlockID < b.BlockID
	})

	var prunedBlocks []KVBlock
	var prunedIDs []string

	for _, cand := range candidates {
		if t.totalTokens <= maxTokens {
			break
		}
		delete(t.blocks, cand.BlockID)
		t.totalTokens -= cand.TokenCount
		prunedBlocks = append(prunedBlocks, cand)
		prunedIDs = append(prunedIDs, cand.BlockID)
	}

	prunedTokens := initialTokens - t.totalTokens
	remainingBlocks := len(t.blocks)

	report := PruneReport{
		OriginalNodesCount:  initialBlocks,
		PrunedNodesCount:    len(prunedIDs),
		RemainingNodesCount: remainingBlocks,
		RemovedNodes:        prunedIDs,
		ExecutionHopsSaved:  prunedTokens,
		InitialUtility:      float64(initialTokens),
		FinalUtility:        float64(t.totalTokens),
	}

	kvReport := KVPruneReport{
		InitialTokens:   initialTokens,
		PrunedTokens:    prunedTokens,
		RemainingTokens: t.totalTokens,
		MaxTokens:       maxTokens,
		InitialBlocks:   initialBlocks,
		PrunedBlocks:    prunedBlocks,
		RemainingBlocks: remainingBlocks,
		RemovedBlockIDs: prunedIDs,
	}

	t.lastPruned = prunedBlocks
	t.lastReport = report
	t.lastKVReport = kvReport

	return report
}

// EnforceBudgetDetailed runs EnforceBudget and returns the detailed KVPruneReport.
func (t *PriorityKVTable) EnforceBudgetDetailed(maxTokens int) KVPruneReport {
	t.EnforceBudget(maxTokens)
	return t.LastKVPruneReport()
}
