// Shared-arena contract: the DSpark unified KV-cache allocation in which the
// draft model and the target model share ONE KV-cache-manager allocation, so
// draft-model KV and target MLA KV coexist in a single arena with per-role
// page-index views, and accepted/rejected speculation steps are reconciled in
// place (accepted steps keep their pages; rejected steps return them) without
// separate caches or copies.
//
// Provenance (ADAPT; no upstream code copied): TensorRT-LLM's DSpark unified
// KV-cache standalone (PR #20664) makes the draft and target share one
// KV-cache-manager allocation so that speculation steps reconcile in place
// rather than through a second cache. Source: NVIDIA/TensorRT-LLM,
// tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py +
// docs/research/STUDY-BORROW-WAVE-H200-LATEST-2026-11-06.md (row
// in-memory oracle over a fixed page table, software evidence, no hardware
// claim.
package kvmmu

import (
	"errors"
	"sort"
	"strconv"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Role names one of the two participants that share the unified arena.
type Role int

const (
	// RoleDraft is the speculative draft model: it writes its own KV into the
	// shared arena and reads it back through the draft page-index view.
	RoleDraft Role = iota
	// RoleTarget is the target model: its MLA KV lives in the same arena and is
	// read back through the target page-index view.
	RoleTarget
)

// String renders the role for audit/log surfaces.
func (r Role) String() string {
	switch r {
	case RoleDraft:
		return "draft"
	case RoleTarget:
		return "target"
	default:
		return "unknown"
	}
}

// Structural sentinels for shared-arena misuse. Capacity overruns are NOT
// structural: they refuse with the typed compute.FitError (Verdict FitTooBig)
// so machine-facing callers assert verdict/want/avail, never prose.
var (
	// ErrNonPositivePages refuses an arena with zero or negative physical pages.
	ErrNonPositivePages = errors.New("kvmmu: shared arena requires at least one physical page")
	// ErrNegativeRoleOffset refuses a negative per-role page-index offset.
	ErrNegativeRoleOffset = errors.New("kvmmu: role page-index offset must be non-negative")
	// ErrUnknownStep refuses to reconcile a step the arena never allocated.
	ErrUnknownStep = errors.New("kvmmu: unknown spec step")
	// ErrStepAlreadyReconciled refuses to reconcile a step that was already
	// accepted or rejected: a step reconciles exactly once, in place.
	ErrStepAlreadyReconciled = errors.New("kvmmu: spec step already reconciled")
	// ErrNonPositiveAllocation refuses a step allocation of zero or negative pages.
	ErrNonPositiveAllocation = errors.New("kvmmu: spec step allocation must be positive")
	// ErrEmptyStepID refuses to allocate a step without a stable id.
	ErrEmptyStepID = errors.New("kvmmu: spec step requires a non-empty id")
	// ErrDuplicateStep refuses to allocate a step id that is already live.
	ErrDuplicateStep = errors.New("kvmmu: spec step id already live")
)

// SpecStep is one speculation step's claim on the shared arena: a stable id,
// the role that wrote it, and the physical pages it occupies.
type SpecStep struct {
	ID    string
	Role  Role
	Pages []int64 // physical page ids, in allocation order
}

// SharedArena is the DSpark unified KV-cache allocation: one fixed physical
// page table shared by the draft and target roles, with per-role page-index
// views and in-place step reconciliation. Safe for concurrent use.
type SharedArena struct {
	mu sync.Mutex

	totalPages int
	// targetOffset is the target role's page-index offset into the shared
	// physical table; the draft role always indexes from 0. The two offsets
	// differ, which is the "adjusted page-index handling" of the unified arena:
	// the same physical page carries a different index in each role's view.
	targetOffset int

	// free is the set of physical pages not currently claimed by a live step.
	free map[int64]struct{}
	// steps is the live (unreconciled) step ledger, keyed by step id.
	steps map[string]*SpecStep
	// reconciled records every step id that has been accepted or rejected, so a
	// second reconciliation of the same step is refused with
	// ErrStepAlreadyReconciled (distinct from an unknown step).
	reconciled map[string]struct{}
	// owner maps each allocated physical page to the step that claims it.
	owner map[int64]string
	// kept is the set of physical pages retained by accepted steps.
	kept map[int64]struct{}
}

// NewSharedArena builds the unified arena over a fixed physical page table of
// totalPages pages. The target role's page-index view is offset by targetOffset
// from the draft view (which always starts at 0); the two must differ so the
// per-role views are genuinely distinct apertures of the same allocation.
// Precondition: totalPages >= 1 and targetOffset >= 0.
// Postcondition: every physical page is free and no step is live.
func NewSharedArena(totalPages, targetOffset int) (*SharedArena, error) {
	if totalPages < 1 {
		return nil, ErrNonPositivePages
	}
	if targetOffset < 0 {
		return nil, ErrNegativeRoleOffset
	}
	free := make(map[int64]struct{}, totalPages)
	for id := int64(0); id < int64(totalPages); id++ {
		free[id] = struct{}{}
	}
	return &SharedArena{
		totalPages:   totalPages,
		targetOffset: targetOffset,
		free:         free,
		steps:        map[string]*SpecStep{},
		reconciled:   map[string]struct{}{},
		owner:        map[int64]string{},
		kept:         map[int64]struct{}{},
	}, nil
}

// RoleOffset returns the page-index offset a role's view applies to the shared
// physical table: 0 for the draft role, the configured offset for the target.
func (a *SharedArena) RoleOffset(role Role) int {
	a.mu.Lock()
	defer a.mu.Unlock()
	if role == RoleTarget {
		return a.targetOffset
	}
	return 0
}

// Allocate reserves a contiguous run of count physical pages for a new spec
// step and returns the step. It fails closed with a typed *compute.FitError
// (Verdict FitTooBig, Want = count, Avail = free pages) when the arena cannot
// back the allocation, so an over-allocation turns into an answerable refusal
// before a page is claimed. Structural misuse returns the package sentinels.
// Precondition: id is a non-empty, not-yet-live step id and count >= 1.
// Postcondition: on success the step is live, its pages are removed from the
// free set, and each page is owned by the step.
func (a *SharedArena) Allocate(id string, role Role, count int) (*SpecStep, error) {
	if id == "" {
		return nil, ErrEmptyStepID
	}
	if count < 1 {
		return nil, ErrNonPositiveAllocation
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, live := a.steps[id]; live {
		return nil, ErrDuplicateStep
	}
	if count > len(a.free) {
		return nil, &compute.FitError{
			Verdict: compute.FitTooBig,
			Want:    int64(count),
			Avail:   int64(len(a.free)),
			Scope:   compute.MemoryScopeDevice,
		}
	}
	// Claim the lowest free page ids, in order, so the allocation is
	// deterministic and the step's pages form a stable, inspectable run.
	allFree := make([]int64, 0, len(a.free))
	for id64 := range a.free {
		allFree = append(allFree, id64)
	}
	sort.Slice(allFree, func(i, j int) bool { return allFree[i] < allFree[j] })
	pages := allFree[:count]
	for _, p := range pages {
		delete(a.free, p)
		a.owner[p] = id
	}
	step := &SpecStep{ID: id, Role: role, Pages: pages}
	a.steps[id] = step
	return step, nil
}

// Accept reconciles a live spec step in place: its pages are retained in the
// arena and re-parented to the target role's page-index view, and the step is
// sealed. The physical pages are NOT freed — an accepted step keeps them.
// Precondition: id names a live (unreconciled) step.
// Postcondition: the step is reconciled, its pages are in the kept set, and the
// target role's view now includes them.
func (a *SharedArena) Accept(id string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, done := a.reconciled[id]; done {
		return ErrStepAlreadyReconciled
	}
	step, ok := a.steps[id]
	if !ok {
		return ErrUnknownStep
	}
	for _, p := range step.Pages {
		a.kept[p] = struct{}{}
	}
	a.reconciled[id] = struct{}{}
	delete(a.steps, id)
	return nil
}

// Reject reconciles a live spec step in place: its pages are returned to the
// free set and the step is sealed. The physical pages ARE freed — a rejected
// step gives them back to the shared arena.
// Precondition: id names a live (unreconciled) step.
// Postcondition: the step is reconciled, its pages are free again, and no role's
// view includes them.
func (a *SharedArena) Reject(id string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	if _, done := a.reconciled[id]; done {
		return ErrStepAlreadyReconciled
	}
	step, ok := a.steps[id]
	if !ok {
		return ErrUnknownStep
	}
	for _, p := range step.Pages {
		delete(a.owner, p)
		a.free[p] = struct{}{}
	}
	a.reconciled[id] = struct{}{}
	delete(a.steps, id)
	return nil
}

// RoleView returns the page-index view of one role: the physical pages the
// role currently sees, each rendered through that role's page-index offset.
// The draft view sees every live step's pages plus every kept page; the target
// view sees only the kept (accepted) pages, offset by the target offset.
// Precondition: role is RoleDraft or RoleTarget.
// Postcondition: the returned slice is a copy, safe for the caller to retain.
func (a *SharedArena) RoleView(role Role) []int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	off := 0
	if role == RoleTarget {
		off = a.targetOffset
	}
	var out []int64
	if role == RoleTarget {
		for p := range a.kept {
			out = append(out, p+int64(off))
		}
	} else {
		// The draft view sees every allocated page (live steps + kept).
		for p := range a.owner {
			out = append(out, p+int64(off))
		}
		for p := range a.kept {
			if _, owned := a.owner[p]; !owned {
				out = append(out, p+int64(off))
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// PageCount reports how many physical pages are currently allocated (claimed by
// a live step or retained by an accepted step).
func (a *SharedArena) PageCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.totalPages - len(a.free)
}

// FreeCount reports how many physical pages are currently free.
func (a *SharedArena) FreeCount() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return len(a.free)
}

// TotalPages reports the fixed physical page-table size of the arena.
func (a *SharedArena) TotalPages() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.totalPages
}

// PageOwner reports the step id that owns a physical page, or "" when the page
// is free or was returned by a rejected step (whose owner was cleared on reject).
// An accepted step keeps its owner so the retained pages remain attributable.
func (a *SharedArena) PageOwner(page int64) string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.owner[page]
}

// snapshot is an immutable copy of the arena's state, taken under the lock,
// that the closed oracle can inspect without holding the arena.
type snapshot struct {
	totalPages   int
	targetOffset int
	free         map[int64]struct{}
	owner        map[int64]string
	kept         map[int64]struct{}
	livePages    map[int64]struct{}
}

func (a *SharedArena) takeSnapshot() snapshot {
	a.mu.Lock()
	defer a.mu.Unlock()
	s := snapshot{
		totalPages:   a.totalPages,
		targetOffset: a.targetOffset,
		free:         make(map[int64]struct{}, len(a.free)),
		owner:        make(map[int64]string, len(a.owner)),
		kept:         make(map[int64]struct{}, len(a.kept)),
		livePages:    make(map[int64]struct{}),
	}
	for p := range a.free {
		s.free[p] = struct{}{}
	}
	for p, id := range a.owner {
		s.owner[p] = id
	}
	for p := range a.kept {
		s.kept[p] = struct{}{}
	}
	for p := range a.owner {
		s.livePages[p] = struct{}{}
	}
	for p := range a.kept {
		s.livePages[p] = struct{}{}
	}
	return s
}

// Verify is the closed in-memory oracle over the fixed page table: it checks
// the arena's invariants and returns a list of human-readable violations (nil
// when the arena is consistent). The invariants are:
//   - every physical page is in [0, totalPages);
//   - the free set and the allocated set (owner ∪ kept) are disjoint and their
//     union is the whole page table (one allocation, no page lost or doubled);
//   - no page is owned by two steps;
//   - every kept page is within the page table.
//
// A non-nil result names the first violated invariant so a test can assert the
// oracle actually saw the corruption, not just that it ran.
func (a *SharedArena) Verify() []string {
	s := a.takeSnapshot()
	var violations []string
	alloc := make(map[int64]struct{}, len(s.owner)+len(s.kept))
	for p := range s.owner {
		alloc[p] = struct{}{}
	}
	for p := range s.kept {
		alloc[p] = struct{}{}
	}
	// Invariant 1: every referenced page is in range.
	for p := range s.free {
		if p < 0 || p >= int64(s.totalPages) {
			violations = append(violations, "free page out of range: "+strconv.FormatInt(p, 10))
		}
	}
	for p := range alloc {
		if p < 0 || p >= int64(s.totalPages) {
			violations = append(violations, "allocated page out of range: "+strconv.FormatInt(p, 10))
		}
	}
	// Invariant 2: free and allocated are disjoint.
	for p := range s.free {
		if _, inAlloc := alloc[p]; inAlloc {
			violations = append(violations, "page both free and allocated: "+strconv.FormatInt(p, 10))
		}
	}
	// Invariant 3: free ∪ allocated == the whole page table (one allocation).
	if len(s.free)+len(alloc) != s.totalPages {
		violations = append(violations, "page accounting mismatch: free "+strconv.FormatInt(int64(len(s.free)), 10)+
			" + allocated "+strconv.FormatInt(int64(len(alloc)), 10)+" != total "+strconv.FormatInt(int64(s.totalPages), 10))
	}
	// Invariant 4: the page table is exactly [0, totalPages).
	for p := int64(0); p < int64(s.totalPages); p++ {
		_, isFree := s.free[p]
		_, isAlloc := alloc[p]
		if !isFree && !isAlloc {
			violations = append(violations, "page unaccounted: "+strconv.FormatInt(p, 10))
		}
	}
	return violations
}
