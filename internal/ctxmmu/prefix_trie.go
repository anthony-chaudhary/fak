package ctxmmu

import (
	"sync"
)

// TrieNode represents a single token state in the prefix trie.
type TrieNode struct {
	Token    int
	Entry    *PromptEntry
	Children map[int]*TrieNode
	Parent   *TrieNode
	Depth    int
}

// PromptTrie is a thread-safe prefix tree indexing cached prompt sequences by token IDs.
// It supports exact match, longest prefix match, and trimmable match search.
type PromptTrie struct {
	mu   sync.RWMutex
	root *TrieNode
	size int
}

// NewPromptTrie constructs an empty prefix trie.
func NewPromptTrie() *PromptTrie {
	return &PromptTrie{
		root: &TrieNode{
			Children: make(map[int]*TrieNode),
			Depth:    0,
		},
	}
}

// CollectDescendantEntries returns all prompt entries stored in the subtree rooted at n.
func (n *TrieNode) CollectDescendantEntries() []*PromptEntry {
	if n == nil {
		return nil
	}
	var entries []*PromptEntry
	var walk func(curr *TrieNode)
	walk = func(curr *TrieNode) {
		if curr == nil {
			return
		}
		if curr.Entry != nil {
			entries = append(entries, curr.Entry)
		}
		for _, child := range curr.Children {
			walk(child)
		}
	}
	walk(n)
	return entries
}

// FindBestDescendantEntry finds the descendant entry in the subtree with highest retention priority,
// breaking ties by most recent access.
func (n *TrieNode) FindBestDescendantEntry() *PromptEntry {
	entries := n.CollectDescendantEntries()
	if len(entries) == 0 {
		return nil
	}
	best := entries[0]
	for _, e := range entries[1:] {
		if e.Role.RetentionPriority() > best.Role.RetentionPriority() {
			best = e
		} else if e.Role.RetentionPriority() == best.Role.RetentionPriority() {
			if e.LastAccess.After(best.LastAccess) {
				best = e
			}
		}
	}
	return best
}

// Insert inserts a token sequence into the trie and associates it with the entry.
func (t *PromptTrie) Insert(tokens []int, entry *PromptEntry) *TrieNode {
	t.mu.Lock()
	defer t.mu.Unlock()

	curr := t.root
	for _, tok := range tokens {
		next, exists := curr.Children[tok]
		if !exists {
			next = &TrieNode{
				Token:    tok,
				Children: make(map[int]*TrieNode),
				Parent:   curr,
				Depth:    curr.Depth + 1,
			}
			curr.Children[tok] = next
		}
		curr = next
	}

	if curr.Entry == nil {
		t.size++
	}
	curr.Entry = entry
	if entry != nil {
		entry.node = curr
	}
	return curr
}

// Delete removes the entry for the given token sequence from the trie.
func (t *PromptTrie) Delete(tokens []int) bool {
	t.mu.Lock()
	defer t.mu.Unlock()

	curr := t.root
	for _, tok := range tokens {
		next, exists := curr.Children[tok]
		if !exists {
			return false
		}
		curr = next
	}

	if curr.Entry == nil {
		return false
	}

	curr.Entry.node = nil
	curr.Entry = nil
	t.size--

	// Prune childless unindexed nodes up the branch
	for curr != t.root && curr.Entry == nil && len(curr.Children) == 0 {
		parent := curr.Parent
		if parent != nil {
			delete(parent.Children, curr.Token)
		}
		curr = parent
	}

	return true
}

// ExactMatch returns the entry associated with the exact token sequence, or nil if none exists.
func (t *PromptTrie) ExactMatch(tokens []int) *PromptEntry {
	t.mu.RLock()
	defer t.mu.RUnlock()

	curr := t.root
	for _, tok := range tokens {
		next, exists := curr.Children[tok]
		if !exists {
			return nil
		}
		curr = next
	}
	return curr.Entry
}

// LongestPrefix returns the entry corresponding to the longest prefix of tokens present in the trie,
// along with the matched prefix length.
func (t *PromptTrie) LongestPrefix(tokens []int) (*PromptEntry, int) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	curr := t.root
	var bestEntry *PromptEntry
	bestLen := 0

	for i, tok := range tokens {
		next, exists := curr.Children[tok]
		if !exists {
			break
		}
		curr = next
		if curr.Entry != nil {
			bestEntry = curr.Entry
			bestLen = i + 1
		}
	}
	return bestEntry, bestLen
}

// FindTrimmablePrefix finds the best cached sequence matching an incoming token sequence.
// It searches for:
//  1. An exact match (matchedLen == len(tokens), trimLen == 0).
//  2. A descendant entry sharing the longest common prefix (K tokens) with the incoming tokens.
//     If the descendant entry has length L > K, it can be trimmed by L - K tokens to reuse the K tokens.
//  3. An ancestor entry matching an exact prefix of length A <= len(tokens) (trimLen == 0).
//
// It returns (entry, matchedLen, trimLen, ok).
func (t *PromptTrie) FindTrimmablePrefix(tokens []int) (entry *PromptEntry, matchedLen int, trimLen int, ok bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if len(tokens) == 0 {
		return nil, 0, 0, false
	}

	curr := t.root
	depthReached := 0

	// Track deepest ancestor entry along the matched path
	var ancestorEntry *PromptEntry
	ancestorLen := 0

	for i, tok := range tokens {
		next, exists := curr.Children[tok]
		if !exists {
			break
		}
		curr = next
		depthReached = i + 1
		if curr.Entry != nil {
			ancestorEntry = curr.Entry
			ancestorLen = depthReached
		}
	}

	if depthReached == 0 {
		return nil, 0, 0, false
	}

	// Case 1: The reached node itself has an entry
	if curr.Entry != nil {
		return curr.Entry, depthReached, 0, true
	}

	// Case 2: Check for a descendant entry under the deepest reached node.
	// If a descendant entry has length L > depthReached, its first depthReached tokens
	// match tokens[:depthReached] identically. Trimming L - depthReached tokens allows reusing them.
	descendant := curr.FindBestDescendantEntry()
	if descendant != nil && len(descendant.Tokens) > depthReached {
		trim := len(descendant.Tokens) - depthReached
		return descendant, depthReached, trim, true
	}

	// Case 3: Fall back to longest ancestor entry found along the path
	if ancestorEntry != nil {
		return ancestorEntry, ancestorLen, 0, true
	}

	return nil, 0, 0, false
}

// MatchTrimmable is an alias for FindTrimmablePrefix.
func (t *PromptTrie) MatchTrimmable(tokens []int) (*PromptEntry, int, int, bool) {
	return t.FindTrimmablePrefix(tokens)
}

// Size returns the number of indexed prompt entries in the trie.
func (t *PromptTrie) Size() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.size
}

// Root returns the root node of the trie.
func (t *PromptTrie) Root() *TrieNode {
	return t.root
}
