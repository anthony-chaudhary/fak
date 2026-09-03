package promptcomp

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// SegmentKind defines the tier of a prompt fragment relative to the cache breakpoint.
type SegmentKind uint8

const (
	// KindSpine represents immutable root attention sink / universal identity (Zone 1).
	KindSpine SegmentKind = iota
	// KindSafety represents system floor, security gates, and tool safety (Zone 1).
	KindSafety
	// KindContract represents role-specific instructions and behavioral expectations (Zone 2).
	KindContract
	// KindTools represents active tool schemas and tool calling grammar (Zone 2).
	KindTools
	// KindOverlay represents dynamic paged working-set rules and capability cards (Zone 2).
	KindOverlay
)

func (k SegmentKind) String() string {
	switch k {
	case KindSpine:
		return "spine"
	case KindSafety:
		return "safety"
	case KindContract:
		return "contract"
	case KindTools:
		return "tools"
	case KindOverlay:
		return "overlay"
	default:
		return fmt.Sprintf("unknown(%d)", k)
	}
}

// IsPrefixZone returns true if the segment belongs to Zone 1 (immutable warm prefix before breakpoint).
func (k SegmentKind) IsPrefixZone() bool {
	return k == KindSpine || k == KindSafety
}

// PromptPart is an atomic, content-addressed prompt building block.
type PromptPart struct {
	ID            string         // Canonical identifier (e.g. "core.spine", "contract.leaf-s1")
	Digest        string         // Hex SHA-256 of Content (computed automatically if empty)
	Content       string         // Raw UTF-8 fragment content
	Kind          SegmentKind    // Segment placement tier
	Rank          int            // Secondary tie-breaker ordering key within same Kind
	DependsOn     []string       // Required prerequisites that must be active
	ConflictsWith []string       // Mutually exclusive part IDs
	Predicate     func(Env) bool // Dynamic activation condition (nil means unconditionally active)
}

// Env provides contextual runtime parameters evaluated by fragment predicates.
type Env struct {
	ModelFamily   string         // e.g. "qwen3.8", "qwen2.5-coder", "claude-3-7"
	IsSmallLocal  bool           // True for 7B-14B models requiring concise contracts
	AgentTier     string         // "coordinator", "leaf", "validator"
	ContextBudget int            // Remaining context tokens available
	WireFormat    string         // "openai", "gguf", "anthropic"
	Extra         map[string]any // Extensible metadata
}

// Standard errors emitted by promptcomp operations.
var (
	ErrMissingID            = errors.New("promptcomp: fragment ID is required")
	ErrEmptyContent         = errors.New("promptcomp: fragment content is empty")
	ErrDuplicateID          = errors.New("promptcomp: duplicate fragment ID")
	ErrNotFound             = errors.New("promptcomp: fragment not found")
	ErrMissingDependency    = errors.New("promptcomp: required dependency is missing")
	ErrConflictingFragments = errors.New("promptcomp: conflicting fragments detected")
	ErrCyclicDependency     = errors.New("promptcomp: cyclic dependency detected")
	ErrDigestMismatch       = errors.New("promptcomp: content digest mismatch")
	ErrZoneInversion        = errors.New("promptcomp: zone 1 prefix segment placed after zone 2 segment")
)

// ComputeDigest returns the SHA-256 hex digest of the given content bytes.
func ComputeDigest(content string) string {
	h := sha256.Sum256([]byte(content))
	return hex.EncodeToString(h[:])
}

// Validate checks internal consistency of a PromptPart.
func (p *PromptPart) Validate() error {
	p.ID = strings.TrimSpace(p.ID)
	if p.ID == "" {
		return ErrMissingID
	}
	if strings.TrimSpace(p.Content) == "" {
		return ErrEmptyContent
	}
	computed := ComputeDigest(p.Content)
	if p.Digest == "" {
		p.Digest = computed
	} else if !strings.EqualFold(p.Digest, computed) {
		return fmt.Errorf("%w: declared %s != computed %s for %s", ErrDigestMismatch, p.Digest, computed, p.ID)
	}
	return nil
}

// Registry is a thread-safe catalog of immutable prompt fragments.
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
func (r *Registry) Register(part PromptPart) error {
	if err := part.Validate(); err != nil {
		return err
	}
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

// List returns a sorted slice of all registered prompt fragments.
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

// CompiledPrompt contains the deterministic assembled prompt and cache boundary metadata.
type CompiledPrompt struct {
	Raw          string       // Assembled prompt text
	Parts        []PromptPart // Ordered fragments in topological order
	PrefixBytes  int          // Byte length of Zone 1 (Spine + Policy)
	Zone1Content string       // Zone 1 content (tokens 0..K)
	Zone2Content string       // Zone 2 content (tokens K+1..N)
	Zone1Digest  string       // SHA-256 of Zone 1 content
	TotalDigest  string       // SHA-256 of full raw prompt
}

// Compile compiles all matching active fragments from the registry against the environment.
func (r *Registry) Compile(env Env) (*CompiledPrompt, error) {
	r.mu.RLock()
	allParts := make([]PromptPart, 0, len(r.parts))
	for _, p := range r.parts {
		allParts = append(allParts, p)
	}
	r.mu.RUnlock()

	return CompileParts(allParts, env)
}

// CompileParts resolves dependencies, checks conflicts, topologically sorts, and serializes parts.
func CompileParts(candidates []PromptPart, env Env) (*CompiledPrompt, error) {
	// 1. Evaluate predicates to find active set
	activeMap := make(map[string]PromptPart)
	for _, p := range candidates {
		if err := p.Validate(); err != nil {
			return nil, err
		}
		if p.Predicate == nil || p.Predicate(env) {
			if _, exists := activeMap[p.ID]; exists {
				return nil, fmt.Errorf("%w: %q", ErrDuplicateID, p.ID)
			}
			activeMap[p.ID] = p
		}
	}

	// 2. Validate dependencies and conflicts
	for id, p := range activeMap {
		for _, dep := range p.DependsOn {
			dep = strings.TrimSpace(dep)
			if dep == "" {
				continue
			}
			depPart, ok := activeMap[dep]
			if !ok {
				return nil, fmt.Errorf("%w: fragment %q requires active %q", ErrMissingDependency, id, dep)
			}
			// Invariant: Zone 1 cannot depend on Zone 2
			if p.Kind.IsPrefixZone() && !depPart.Kind.IsPrefixZone() {
				return nil, fmt.Errorf("%w: zone 1 fragment %q cannot depend on zone 2 fragment %q", ErrZoneInversion, id, dep)
			}
		}
		for _, conf := range p.ConflictsWith {
			conf = strings.TrimSpace(conf)
			if conf == "" {
				continue
			}
			if _, exists := activeMap[conf]; exists {
				return nil, fmt.Errorf("%w: fragment %q conflicts with active %q", ErrConflictingFragments, id, conf)
			}
		}
	}

	// 3. Build Dependency Graph for Kahn's Algorithm
	inDegree := make(map[string]int, len(activeMap))
	dependents := make(map[string][]string, len(activeMap))
	for id := range activeMap {
		inDegree[id] = 0
	}

	for id, p := range activeMap {
		for _, dep := range p.DependsOn {
			dep = strings.TrimSpace(dep)
			if dep == "" {
				continue
			}
			inDegree[id]++
			dependents[dep] = append(dependents[dep], id)
		}
	}

	// Helper to sort candidates deterministically: Kind ASC, Rank ASC, ID ASC
	partLess := func(a, b PromptPart) bool {
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		if a.Rank != b.Rank {
			return a.Rank < b.Rank
		}
		return a.ID < b.ID
	}

	// Initial ready queue (inDegree == 0)
	var ready []PromptPart
	for id, deg := range inDegree {
		if deg == 0 {
			ready = append(ready, activeMap[id])
		}
	}
	sort.Slice(ready, func(i, j int) bool { return partLess(ready[i], ready[j]) })

	ordered := make([]PromptPart, 0, len(activeMap))
	for len(ready) > 0 {
		// Pop first element
		curr := ready[0]
		ready = ready[1:]
		ordered = append(ordered, curr)

		// Reduce inDegree for all dependents
		var newReady []PromptPart
		for _, depID := range dependents[curr.ID] {
			inDegree[depID]--
			if inDegree[depID] == 0 {
				newReady = append(newReady, activeMap[depID])
			}
		}

		if len(newReady) > 0 {
			ready = append(ready, newReady...)
			sort.Slice(ready, func(i, j int) bool { return partLess(ready[i], ready[j]) })
		}
	}

	// Cycle detection: if ordered length < activeMap length, a cycle exists
	if len(ordered) < len(activeMap) {
		return nil, ErrCyclicDependency
	}

	// 4. Verify Zone ordering (No Zone 1 element may appear after a Zone 2 element)
	seenZone2 := false
	for _, p := range ordered {
		if !p.Kind.IsPrefixZone() {
			seenZone2 = true
		} else if seenZone2 {
			return nil, fmt.Errorf("%w: prefix fragment %q ordered after non-prefix fragment", ErrZoneInversion, p.ID)
		}
	}

	// 5. Partition into Zone 1 and Zone 2 content
	var zone1Parts, zone2Parts []string
	for _, p := range ordered {
		trimmed := strings.TrimSpace(p.Content)
		if trimmed == "" {
			continue
		}
		if p.Kind.IsPrefixZone() {
			zone1Parts = append(zone1Parts, trimmed)
		} else {
			zone2Parts = append(zone2Parts, trimmed)
		}
	}

	zone1Text := strings.Join(zone1Parts, "\n\n")
	zone2Text := strings.Join(zone2Parts, "\n\n")

	var raw string
	prefixBytes := 0
	if zone1Text != "" && zone2Text != "" {
		raw = zone1Text + "\n\n" + zone2Text
		prefixBytes = len(zone1Text)
	} else if zone1Text != "" {
		raw = zone1Text
		prefixBytes = len(zone1Text)
	} else {
		raw = zone2Text
		prefixBytes = 0
	}

	return &CompiledPrompt{
		Raw:          raw,
		Parts:        ordered,
		PrefixBytes:  prefixBytes,
		Zone1Content: zone1Text,
		Zone2Content: zone2Text,
		Zone1Digest:  ComputeDigest(zone1Text),
		TotalDigest:  ComputeDigest(raw),
	}, nil
}
