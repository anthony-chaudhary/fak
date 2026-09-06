package promptcomp

import (
	"fmt"
	"sort"
	"strings"
)

// Assembler resolves and synthesizes prompt fragments into deterministic output strings.
type Assembler struct {
	Registry *Registry
	Parts    []PromptPart
}

// NewAssembler creates an Assembler. It accepts an optional *Registry, Registry, or []PromptPart.
func NewAssembler(src ...any) *Assembler {
	a := &Assembler{}
	if len(src) == 0 {
		return a
	}
	switch v := src[0].(type) {
	case *Registry:
		a.Registry = v
	case Registry:
		a.Registry = &v
	case []PromptPart:
		a.Parts = make([]PromptPart, len(v))
		copy(a.Parts, v)
	}
	return a
}

// NewAssemblerWithParts creates an Assembler initialized with an explicit list of parts.
func NewAssemblerWithParts(parts []PromptPart) *Assembler {
	return &Assembler{Parts: parts}
}

// Resolve filters active parts against the environment, validates conflicts and dependencies,
// detects cycles via DFS, and returns deterministically ordered prompt parts:
// first by Kind (KindSpine -> KindPolicy -> KindContract -> KindTools -> KindOverlay),
// second by Rank ascending, and third by ID lexicographically.
func (a *Assembler) Resolve(env Env) ([]PromptPart, error) {
	if a == nil {
		return nil, nil
	}

	var candidates []PromptPart
	if a.Registry != nil {
		candidates = append(candidates, a.Registry.List()...)
	}
	if len(a.Parts) > 0 {
		candidates = append(candidates, a.Parts...)
	}

	// 1. Evaluate predicates to determine active fragment set
	activeMap := make(map[string]PromptPart)
	var activeList []PromptPart
	for _, p := range candidates {
		if p.Predicate != nil && !p.Predicate(env) {
			continue
		}
		if _, exists := activeMap[p.ID]; exists {
			return nil, fmt.Errorf("%w: duplicate active fragment %q", ErrDuplicateID, p.ID)
		}
		activeMap[p.ID] = p
		activeList = append(activeList, p)
	}

	// 2. Validate conflicts: if part A conflicts with part B and both match predicates, return ErrConflict
	for _, p := range activeList {
		for _, confID := range p.ConflictsWith {
			confID = strings.TrimSpace(confID)
			if confID == "" {
				continue
			}
			if _, exists := activeMap[confID]; exists {
				return nil, fmt.Errorf("%w: fragment %q conflicts with active fragment %q", ErrConflict, p.ID, confID)
			}
		}
	}

	// 3. Validate dependencies: if part A depends on part B and B is missing or predicate false, return ErrMissingDependency
	for _, p := range activeList {
		for _, depID := range p.DependsOn {
			depID = strings.TrimSpace(depID)
			if depID == "" {
				continue
			}
			if _, exists := activeMap[depID]; !exists {
				return nil, fmt.Errorf("%w: fragment %q requires active dependency %q", ErrMissingDependency, p.ID, depID)
			}
		}
	}

	// 4. Detect cycles using DFS traversal on dependency edges
	// Node states: 0 = unvisited, 1 = visiting (on recursion stack), 2 = visited
	color := make(map[string]int, len(activeMap))
	var hasCycle func(id string) bool
	hasCycle = func(id string) bool {
		color[id] = 1
		p := activeMap[id]
		for _, depID := range p.DependsOn {
			depID = strings.TrimSpace(depID)
			if depID == "" {
				continue
			}
			if _, exists := activeMap[depID]; exists {
				if color[depID] == 1 {
					return true
				}
				if color[depID] == 0 {
					if hasCycle(depID) {
						return true
					}
				}
			}
		}
		color[id] = 2
		return false
	}

	// Deterministic iteration order for cycle detection
	sortedIDs := make([]string, 0, len(activeMap))
	for id := range activeMap {
		sortedIDs = append(sortedIDs, id)
	}
	sort.Strings(sortedIDs)

	for _, id := range sortedIDs {
		if color[id] == 0 {
			if hasCycle(id) {
				return nil, fmt.Errorf("%w: cycle detected involving fragment %q", ErrCyclicDependency, id)
			}
		}
	}

	// 5. Order active parts deterministically:
	// Kind ASC (KindSpine -> KindPolicy -> KindContract -> KindTools -> KindOverlay),
	// then Rank ASC, then ID lexicographically.
	resolved := make([]PromptPart, len(activeList))
	copy(resolved, activeList)

	sort.Slice(resolved, func(i, j int) bool {
		if resolved[i].Kind != resolved[j].Kind {
			return resolved[i].Kind < resolved[j].Kind
		}
		if resolved[i].Rank != resolved[j].Rank {
			return resolved[i].Rank < resolved[j].Rank
		}
		return resolved[i].ID < resolved[j].ID
	})

	return resolved, nil
}

// Assemble resolves active fragments against the environment and concatenates their Content
// with newline separators, guaranteeing byte-level determinism.
func (a *Assembler) Assemble(env Env) (string, error) {
	resolved, err := a.Resolve(env)
	if err != nil {
		return "", err
	}
	if len(resolved) == 0 {
		return "", nil
	}
	contents := make([]string, len(resolved))
	for i, p := range resolved {
		contents[i] = p.Content
	}
	return strings.Join(contents, "\n"), nil
}

// CompiledPrompt contains the deterministic assembled prompt and cache boundary metadata.
type CompiledPrompt struct {
	Raw          string       // Assembled prompt text
	Parts        []PromptPart // Ordered fragments in topological/kind order
	PrefixBytes  int          // Byte length of Zone 1 (Spine + Policy)
	Zone1Content string       // Zone 1 content (tokens 0..K)
	Zone2Content string       // Zone 2 content (tokens K+1..N)
	Zone1Digest  string       // SHA-256 of Zone 1 content
	TotalDigest  string       // SHA-256 of full raw prompt
}

// Compile compiles all matching active fragments from the registry against the environment.
func (r *Registry) Compile(env Env) (*CompiledPrompt, error) {
	return CompileParts(r.List(), env)
}

// CompileParts resolves dependencies, checks conflicts, verifies zone ordering, and serializes parts.
// Maintained for seamless compatibility with harnessinit.
func CompileParts(candidates []PromptPart, env Env) (*CompiledPrompt, error) {
	// Auto-compute digest if missing for candidates passed directly
	normalized := make([]PromptPart, len(candidates))
	for i, p := range candidates {
		if p.Digest == "" && p.Content != "" {
			p.Digest = ComputeDigest(p.Content)
		}
		normalized[i] = p
	}

	asm := NewAssemblerWithParts(normalized)
	resolved, err := asm.Resolve(env)
	if err != nil {
		return nil, err
	}

	// Verify Zone ordering: no Zone 1 fragment may depend on a Zone 2 fragment
	resolvedMap := make(map[string]PromptPart, len(resolved))
	for _, p := range resolved {
		resolvedMap[p.ID] = p
	}
	for _, p := range resolved {
		for _, depID := range p.DependsOn {
			depID = strings.TrimSpace(depID)
			if depID == "" {
				continue
			}
			if depPart, ok := resolvedMap[depID]; ok {
				if p.Kind.IsPrefixZone() && !depPart.Kind.IsPrefixZone() {
					return nil, fmt.Errorf("%w: zone 1 fragment %q cannot depend on zone 2 fragment %q", ErrZoneInversion, p.ID, depID)
				}
			}
		}
	}

	// Partition into Zone 1 and Zone 2 content
	var zone1Parts, zone2Parts []string
	for _, p := range resolved {
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
		Parts:        resolved,
		PrefixBytes:  prefixBytes,
		Zone1Content: zone1Text,
		Zone2Content: zone2Text,
		Zone1Digest:  ComputeDigest(zone1Text),
		TotalDigest:  ComputeDigest(raw),
	}, nil
}
