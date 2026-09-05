package speculative

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gym"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// BranchStatus represents the lifecycle state of a speculative hypothesis execution.
type BranchStatus string

const (
	Active    BranchStatus = "Active"
	Committed BranchStatus = "Committed"
	Aborted   BranchStatus = "Aborted"

	StatusActive    = Active
	StatusCommitted = Committed
	StatusAborted   = Aborted
)

func (s BranchStatus) String() string {
	return string(s)
}

var (
	branchSeq   uint64
	treeLocksMu sync.Mutex
	treeLocks   = make(map[*radixkv.Tree]*sync.Mutex)
)

func getTreeLock(t *radixkv.Tree) *sync.Mutex {
	treeLocksMu.Lock()
	defer treeLocksMu.Unlock()
	l, ok := treeLocks[t]
	if !ok {
		l = &sync.Mutex{}
		treeLocks[t] = l
	}
	return l
}

// Branch represents a speculative hypothesis execution environment backed by a lockstep
// CoW VFS workspace and Radix KV-cache prompt prefix branch.
type Branch struct {
	mu                sync.Mutex
	ID                string
	ParentArena       *gym.Arena
	ForkedArena       *gym.Arena
	KVTree            *radixkv.Tree
	BaseTokens        []int
	SpeculativeTokens []int
	Status            BranchStatus
	LeaseOwner        string
	fullTokens        []int
}

// Fork spawns a speculative hypothesis branch:
//   - Forks parent arena into a fresh child arena using parent.Fork (<5ms).
//   - Inserts speculativeTokens under baseTokens in tree using tree.Lookup + tree.Insert to establish
//     the speculative prompt prefix branch.
//   - Returns active *Branch.
func Fork(ctx context.Context, parent *gym.Arena, tree *radixkv.Tree, baseTokens []int, speculativeTokens []int) (*Branch, error) {
	if parent == nil {
		return nil, errors.New("speculative: parent arena is required")
	}
	if tree == nil {
		return nil, errors.New("speculative: radixkv tree is required")
	}

	branchID := fmt.Sprintf("spec-%d-%d", time.Now().UnixNano(), atomic.AddUint64(&branchSeq, 1))

	// 1. Fork parent arena into fresh child arena (<5ms)
	childArena, err := parent.Fork(ctx, branchID)
	if err != nil {
		return nil, fmt.Errorf("speculative: failed to fork child arena: %w", err)
	}

	// 2. Prepare token sequences
	baseCopy := append([]int(nil), baseTokens...)
	specCopy := append([]int(nil), speculativeTokens...)
	fullTokens := make([]int, 0, len(baseCopy)+len(specCopy))
	fullTokens = append(fullTokens, baseCopy...)
	fullTokens = append(fullTokens, specCopy...)

	// 3. Establish prompt prefix branch in KV tree under tree lock
	treeLock := getTreeLock(tree)
	treeLock.Lock()
	defer treeLock.Unlock()

	// Ensure baseTokens prefix boundary is established in the tree
	if len(baseCopy) > 0 {
		b, matched := tree.Lookup(baseCopy)
		if matched < len(baseCopy) {
			b = tree.Insert(b, baseCopy[matched:], nil)
		}
		tree.Done(b)
	}

	// Insert speculativeTokens under baseTokens
	if len(specCopy) > 0 {
		b, matched := tree.Lookup(fullTokens)
		if matched < len(fullTokens) {
			leaf := tree.Insert(b, fullTokens[matched:], nil)
			tree.Done(leaf)
		} else {
			tree.Done(b)
		}
	}

	return &Branch{
		ID:                branchID,
		ParentArena:       parent,
		ForkedArena:       childArena,
		KVTree:            tree,
		BaseTokens:        baseCopy,
		SpeculativeTokens: specCopy,
		Status:            Active,
		fullTokens:        fullTokens,
	}, nil
}

// Path returns the unified overlay workspace path of the forked arena.
func (b *Branch) Path() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.ForkedArena != nil {
		return b.ForkedArena.Path()
	}
	return ""
}

// Execute dispatches a command inside the forked arena under isolated containment.
func (b *Branch) Execute(ctx context.Context, req sandbox.ExecutionRequest) (sandbox.ExecutionResult, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.ForkedArena == nil {
		return sandbox.ExecutionResult{}, errors.New("speculative: forked arena is nil")
	}
	return b.ForkedArena.Execute(ctx, req)
}

// SpeculativeManager coordinates speculative hypothesis branches over a parent arena and KV cache.
type SpeculativeManager struct {
	mu         sync.Mutex
	parent     *gym.Arena
	tree       *radixkv.Tree
	baseTokens []int
	branches   map[string]*Branch
}

// NewManager creates a new SpeculativeManager.
func NewManager(parent *gym.Arena, tree *radixkv.Tree, baseTokens []int) *SpeculativeManager {
	return &SpeculativeManager{
		parent:     parent,
		tree:       tree,
		baseTokens: append([]int(nil), baseTokens...),
		branches:   make(map[string]*Branch),
	}
}

// Fork creates a new speculative hypothesis branch under this manager.
func (m *SpeculativeManager) Fork(ctx context.Context, speculativeTokens []int) (*Branch, error) {
	b, err := Fork(ctx, m.parent, m.tree, m.baseTokens, speculativeTokens)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	m.branches[b.ID] = b
	m.mu.Unlock()
	return b, nil
}

// GetBranch retrieves a branch by ID.
func (m *SpeculativeManager) GetBranch(id string) *Branch {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.branches[id]
}

// ActiveBranches returns all currently active branches.
func (m *SpeculativeManager) ActiveBranches() []*Branch {
	m.mu.Lock()
	defer m.mu.Unlock()
	var active []*Branch
	for _, b := range m.branches {
		b.mu.Lock()
		if b.Status == Active {
			active = append(active, b)
		}
		b.mu.Unlock()
	}
	return active
}
