package skillenv

import (
	"fmt"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxresidency"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// Table is the lock-free skill page table: active skill versions live in an
// immutable snapshot published via atomic.Pointer[map]. Reads (ActiveVersion,
// List) are a single atomic load — no lock, no teardown, zero-downtime under
// hot-swap. Writes (Pin/Unpin/Swap) build a fresh snapshot with the intended
// mutation applied and publish it with a CAS loop; a lost race re-derives the
// mutation from the freshest snapshot, so concurrent flips are last-write-wins
// and the from-version guard in Swap is re-checked per iteration (never a
// stale check under a held lock). A successful write is linearized at its
// publish, and the returned previous version is the pre-state at that point.
type Table struct {
	versions atomic.Pointer[map[string]string]
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
	t := &Table{
		resolver: resolver,
		mmu:      mmu,
		kvctx:    kvctx,
	}
	empty := map[string]string{}
	t.versions.Store(&empty)
	return t
}

// ActiveVersion retrieves the pinned version or delegates to the resolver.
func (t *Table) ActiveVersion(skillName string) (string, bool) {
	if v, ok := (*t.versions.Load())[skillName]; ok {
		return v, true
	}

	return t.resolver.ResolveVersion(skillName)
}

// casUpdate publishes a snapshot derived from the current one via mut, retrying
// on a lost CAS race so every retry re-derives against the freshest snapshot.
// It returns the base snapshot of the PUBLISHED version (the linearization
// pre-state) so callers can report prev values without a racy re-read. mut
// returns ok=false to refuse the whole update, which surfaces as an error.
func (t *Table) casUpdate(mut func(base, next map[string]string) bool) (map[string]string, error) {
	for {
		old := t.versions.Load()
		base := *old
		next := make(map[string]string, len(base)+1)
		for k, v := range base {
			next[k] = v
		}
		if !mut(base, next) {
			return base, fmt.Errorf("skillenv: page-table update refused")
		}
		if t.versions.CompareAndSwap(old, &next) {
			return base, nil
		}
	}
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
	base, err := t.casUpdate(func(_, next map[string]string) bool {
		next[skillName] = version
		return true
	})
	if err != nil {
		return "", blastRadius, err
	}

	return base[skillName], blastRadius, nil
}

// Unpin removes an explicit pin and calculates the rollback blast radius.
func (t *Table) Unpin(skillName string) (unpinnedVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot unpin empty skill name")
	}

	blastRadius = t.blastRadius()
	base, err := t.casUpdate(func(_, next map[string]string) bool {
		delete(next, skillName)
		return true
	})
	if err != nil {
		return "", blastRadius, err
	}

	// Absent in the linearization pre-state means nothing was unpinned; this is
	// not an error (matches the historical contract) — the resolver governs.
	return base[skillName], blastRadius, nil
}

// Swap atomically remaps an expected pinned version to a target version. The
// from-guard is evaluated against the same snapshot the CAS publishes from, so
// a concurrent flip that moved the skill is re-checked on the retry loop, not
// observed stale under a lock.
func (t *Table) Swap(skillName, fromVersion, toVersion string) (prevVersion string, blastRadius ctxresidency.BlastRadius, err error) {
	if skillName == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot swap empty skill name")
	}
	if fromVersion == "" || toVersion == "" {
		return "", ctxresidency.BlastRadius{}, fmt.Errorf("skillenv: cannot swap to/from empty version")
	}

	blastRadius = t.blastRadius()
	base, err := t.casUpdate(func(base, next map[string]string) bool {
		if current, ok := base[skillName]; ok && current != fromVersion {
			return false
		}
		next[skillName] = toVersion
		return true
	})
	if err != nil {
		current := base[skillName]
		if current != "" {
			return "", blastRadius, fmt.Errorf("skillenv: swap refused: skill %s pinned to %s, not %s", skillName, current, fromVersion)
		}
		return "", blastRadius, err
	}

	return base[skillName], blastRadius, nil
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
	out := make(map[string]string, len(*t.versions.Load()))
	for skill, version := range *t.versions.Load() {
		out[skill] = version
	}
	return out
}
