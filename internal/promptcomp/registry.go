package promptcomp

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Registry is a thread-safe catalog of content-addressed prompt fragments.
type Registry struct {
	mu    sync.RWMutex
	parts map[string]PromptPart
}

// NewRegistry initializes an empty fragment registry.
func NewRegistry() *Registry {
	return &Registry{
		parts: make(map[string]PromptPart),
	}
}

// Register adds a validated prompt fragment to the registry.
// It enforces that part.ID and part.Content are non-empty, part.Digest matches SHA-256(part.Content),
// and part.ID is not already registered (detecting duplicate IDs).
func (r *Registry) Register(part PromptPart) error {
	trimmedID := strings.TrimSpace(part.ID)
	if trimmedID == "" {
		return ErrMissingID
	}
	if strings.TrimSpace(part.Content) == "" {
		return ErrEmptyContent
	}
	expectedDigest := ComputeDigest(part.Content)
	if part.Digest == "" || !strings.EqualFold(part.Digest, expectedDigest) {
		return fmt.Errorf("%w: declared %q != computed %q for %q", ErrDigestMismatch, part.Digest, expectedDigest, part.ID)
	}
	part.ID = trimmedID

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.parts[part.ID]; exists {
		return fmt.Errorf("%w: %q", ErrDuplicateID, part.ID)
	}
	r.parts[part.ID] = part
	return nil
}

// Get retrieves a prompt fragment by ID.
func (r *Registry) Get(id string) (PromptPart, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	part, ok := r.parts[id]
	return part, ok
}

// List returns all registered prompt fragments ordered deterministically:
// first by Kind ascending, second by Rank ascending, third by ID lexicographically.
func (r *Registry) List() []PromptPart {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]PromptPart, 0, len(r.parts))
	for _, p := range r.parts {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Kind != out[j].Kind {
			return out[i].Kind < out[j].Kind
		}
		if out[i].Rank != out[j].Rank {
			return out[i].Rank < out[j].Rank
		}
		return out[i].ID < out[j].ID
	})
	return out
}

// Len returns the number of fragments currently registered.
func (r *Registry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.parts)
}
