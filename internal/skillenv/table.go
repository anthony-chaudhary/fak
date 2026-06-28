package skillenv

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// Table is a versioned page table for skills - the "virtual skill environment".
// It maps skill names to their active versions, enabling hot-swap (remap) and
// rollback (inverse remap) without reloading or disturbing in-flight invocations.
//
// Multiple versions of a skill may be resident simultaneously in the underlying
// procedural cache (contextq.SkillContextRecord.Version already re-keys on version),
// so this table only tracks which version is "active" for new invocations.
// In-flight invocations keep their pinned frame (the ctxmmu CAS pin guarantees
// a pinned page survives eviction).
//
// Hot-swap = remap: flip the page-table entry v1 → v2. New invocations resolve
// the new version; existing ones continue using their pinned frame.
// Rollback = inverse remap: flip back to the prior version.
//
// Blast radius is reported before a swap via ctxresidency.Query, telling the
// operator exactly what will be invalidated by the flip.
type Table struct {
	mu             sync.RWMutex
	versions       map[string]string // skill name → active version
	resolver       Resolver          // how to resolve a skill name to a version
	mmu            *ctxmmu.MMU        // for blast-radius reads
	kvctx          *kvmmu.Context     // for blast-radius reads
}

// Resolver abstracts version resolution: pinned (explicit version), latest
// (A/B test or latest tag), or a custom policy. A nil resolver defaults to
// "latest" (the version the skill declares in its SKILL.md frontmatter).
type Resolver interface {
	// ResolveVersion returns the version string for a skill name.
	// If the skill is unknown, returns ("", false).
	ResolveVersion(skillName string) (version string, ok bool)
}

// DefaultResolver is the "latest" policy: it reads the version from the skill's
// SKILL.md frontmatter. The actual implementation is deferred to C1 (internal/capindex);
// this stub always returns an empty version.
type DefaultResolver struct{}

func (r *DefaultResolver) ResolveVersion(skillName string) (string, bool) {
	// Deferred to C1 (internal/capindex) - the skill resolver over .claude/skills/.
	// For now, return empty (no-op).
	return "", false
}

// New builds a versioned page table with the given resolver.
// If resolver is nil, DefaultResolver is used.
// If mmu or kvctx is nil, blast-radius reads are skipped (pre-flip advisory is unavailable).
func New(resolver Resolver, mmu *ctxmmu.MMU, kvctx *kvmmu.Context) *Table {
	if resolver == nil {
		resolver = &DefaultResolver{}
	}
	return &Table{
		versions: make(map[string]string),
		resolver: resolver,
		mmu:      mmu,
		kvctx:    kvctx,
	}
}

// ActiveVersion returns the currently active version for a skill.
// If the skill is not pinned in the table, it resolves via the resolver.
// If the skill is unknown, returns ("", false).
func (t *Table) ActiveVersion(skillName string) (string, bool) {
	t.mu.RLock()
	if v, ok := t.versions[skillName]; ok {
		t.mu.RUnlock()
		return v, true
	}
	t.mu.RUnlock()

	// Not pinned - resolve via the resolver.
	return t.resolver.ResolveVersion(skillName)
}

// Pin binds a skill name to an explicit version - a page-table entry.
// This is the "hot-swap" entry point: Pin sets the active version, and a
// subsequent Pin to a different version remaps (hot-swaps) it.
//
// The blast radius of a swap is reported before the flip (via ctxresidency.Query),
// so the operator knows exactly what will be invalidated. If mmu or kvctx is nil,
// blast radius is reported as zero.
//
// Returns the previous version (if any) and the blast radius of the swap.
func (t *Table) Pin(skillName, version string) (prevVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot pin empty skill name")
	}
	if version == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot pin empty version")
	}

	// Read blast radius BEFORE the flip (so the operator knows what will be invalidated).
	blastRadius = t.blastRadius()

	t.mu.Lock()
	defer t.mu.Unlock()

	prevVersion = t.versions[skillName]
	t.versions[skillName] = version

	return prevVersion, blastRadius, nil
}

// Unpin removes an explicit pin, reverting to resolver policy.
// This is the inverse of a Pin: it clears the page-table entry, so future
// ActiveVersion calls resolve via the resolver instead.
//
// Returns the version that was unpinned, and the blast radius of the rollback.
func (t *Table) Unpin(skillName string) (unpinnedVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot unpin empty skill name")
	}

	// Read blast radius BEFORE the rollback.
	blastRadius = t.blastRadius()

	t.mu.Lock()
	defer t.mu.Unlock()

	unpinnedVersion, ok := t.versions[skillName]
	if !ok {
		return "", blastRadius, nil // No-op: skill was not pinned.
	}
	delete(t.versions, skillName)

	return unpinnedVersion, blastRadius, nil
}

// Swap remaps one version to another - a convenience wrapper over Pin+Unpin.
// It unsets the old version and pins the new one atomically.
//
// Returns the previous version and the blast radius of the swap.
func (t *Table) Swap(skillName, fromVersion, toVersion string) (prevVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot swap empty skill name")
	}
	if fromVersion == "" || toVersion == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot swap to/from empty version")
	}

	// Read blast radius BEFORE the swap.
	blastRadius = t.blastRadius()

	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.versions[skillName]
	if ok && current != fromVersion {
		// The skill is pinned, but not to the expected version.
		// This is a consistency guard - the swap is refused to avoid a silent drift.
		return "", blastRadius, fmt.Errorf("skillenv: swap refused: skill %s pinned to %s, not %s", skillName, current, fromVersion)
	}

	t.versions[skillName] = toVersion
	return current, blastRadius, nil
}

// blastRadius reports the eviction blast radius via ctxresidency.Query.
// This is the "what would this swap invalidate?" read. If mmu or kvctx is nil,
// returns zero.
func (t *Table) blastRadius() ctxresidency.BlastRadius {
	if t.mmu == nil || t.kvctx == nil {
		return ctxresidency.BlastRadius{}
	}
	snap := ctxresidency.Query(t.kvctx, t.mmu)
	// The blast radius of a version swap is the sum of all evictable spans'
	// blast radii. This is a conservative upper bound; a real swap only
	// invalidates spans whose procedural views reference the swapped version.
	//
	// A precise blast radius would require scanning the ViewCache for all
	// procedural views (ViewProcedure) whose labels[version] matches the old
	// version. That's C6's witness surface; for now we report the total
	// evictable cost as a safe upper bound.
	tokens := 0
	deps := 0
	for _, span := range snap.Spans {
		if span.State == ctxresidency.StateEvictable {
			tokens += span.EvictBlastRadius.Tokens
			deps += span.EvictBlastRadius.DependentEntries
		}
	}
	return ctxresidency.BlastRadius{Tokens: tokens, DependentEntries: deps}
}

// List returns all pinned skills and their versions.
// This is the page-table snapshot - the complete mapping of skill → active version.
func (t *Table) List() map[string]string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	out := make(map[string]string, len(t.versions))
	for skill, version := range t.versions {
		out[skill] = version
	}
	return out
}