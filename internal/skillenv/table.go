package skillenv

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// Table maintains active skill versions and coordinates remap hot-swaps and rollbacks.
type Table struct {
	mu       sync.RWMutex
	versions map[string]string
	resolver Resolver
	mmu      *ctxmmu.MMU
	kvctx    *kvmmu.Context
}

// Resolver determines the active version for an unpinned skill name.
type Resolver interface {
	ResolveVersion(skillName string) (version string, ok bool)
}

// DefaultResolver serves unpinned skill resolutions using workspace defaults.
type DefaultResolver struct{}

// ResolveVersion yields empty resolution for unbound skills.
func (r *DefaultResolver) ResolveVersion(skillName string) (string, bool) {
	return "", false
}

// New constructs a version table bound to optional residency monitors.
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

// ActiveVersion retrieves the pinned version or delegates to the resolver.
func (t *Table) ActiveVersion(skillName string) (string, bool) {
	t.mu.RLock()
	if v, ok := t.versions[skillName]; ok {
		t.mu.RUnlock()
		return v, true
	}
	t.mu.RUnlock()

	return t.resolver.ResolveVersion(skillName)
}

// Pin binds a skill name to an explicit version and evaluates the eviction blast radius.
func (t *Table) Pin(skillName, version string) (prevVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot pin empty skill name")
	}
	if version == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot pin empty version")
	}

	blastRadius = t.blastRadius()

	t.mu.Lock()
	defer t.mu.Unlock()

	prevVersion = t.versions[skillName]
	t.versions[skillName] = version

	return prevVersion, blastRadius, nil
}

// Unpin removes an explicit pin and calculates the rollback blast radius.
func (t *Table) Unpin(skillName string) (unpinnedVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot unpin empty skill name")
	}

	blastRadius = t.blastRadius()

	t.mu.Lock()
	defer t.mu.Unlock()

	unpinnedVersion, ok := t.versions[skillName]
	if !ok {
		return "", blastRadius, nil
	}
	delete(t.versions, skillName)

	return unpinnedVersion, blastRadius, nil
}

// Swap atomically remaps an expected pinned version to a target version.
func (t *Table) Swap(skillName, fromVersion, toVersion string) (prevVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot swap empty skill name")
	}
	if fromVersion == "" || toVersion == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot swap to/from empty version")
	}

	blastRadius = t.blastRadius()

	t.mu.Lock()
	defer t.mu.Unlock()

	current, ok := t.versions[skillName]
	if ok && current != fromVersion {
		return "", blastRadius, fmt.Errorf("skillenv: swap refused: skill %s pinned to %s, not %s", skillName, current, fromVersion)
	}

	t.versions[skillName] = toVersion
	return current, blastRadius, nil
}

func (t *Table) blastRadius() ctxresidency.BlastRadius {
	if t.mmu == nil || t.kvctx == nil {
		return ctxresidency.BlastRadius{}
	}
	snap := ctxresidency.Query(t.kvctx, t.mmu)
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

// List takes a point-in-time snapshot of all pinned skill versions.
func (t *Table) List() map[string]string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	out := make(map[string]string, len(t.versions))
	for skill, version := range t.versions {
		out[skill] = version
	}
	return out
}
