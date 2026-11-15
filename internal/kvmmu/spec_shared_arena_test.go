package kvmmu

import (
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestSpecSharedArena pins the DSpark unified-KV-arena contract: one shared
// allocation, per-role page-index views, in-place step reconciliation
// (accepted steps keep their pages, rejected steps return them), and a
// fail-closed refusal on an over-allocation. Every assertion inspects real
// page-table state through the closed in-memory oracle (Verify) and the
// arena's own views, never prose.
func TestSpecSharedArena(t *testing.T) {
	// A fixed page table of 8 physical pages; the target role's page-index
	// view is offset by 2 from the draft view (which always starts at 0).
	const total, targetOff = 8, 2

	t.Run("single shared allocation", func(t *testing.T) {
		a, err := NewSharedArena(total, targetOff)
		if err != nil {
			t.Fatalf("NewSharedArena(%d,%d) = %v, want ok", total, targetOff, err)
		}
		// One allocation: the whole page table is one arena, not two caches.
		if a.TotalPages() != total {
			t.Fatalf("TotalPages = %d, want %d", a.TotalPages(), total)
		}
		if a.FreeCount() != total || a.PageCount() != 0 {
			t.Fatalf("fresh arena = free %d / allocated %d, want %d / 0",
				a.FreeCount(), a.PageCount(), total)
		}
		// The oracle must see a consistent, fully-accounted page table.
		if v := a.Verify(); len(v) != 0 {
			t.Fatalf("fresh arena Verify = %v, want no violations", v)
		}
		// A single allocation of 3 pages claims exactly 3 of the 8.
		step, err := a.Allocate("s1", RoleDraft, 3)
		if err != nil {
			t.Fatalf("Allocate(s1,draft,3) = %v, want ok", err)
		}
		if len(step.Pages) != 3 {
			t.Fatalf("step pages = %v, want 3 physical pages", step.Pages)
		}
		if a.PageCount() != 3 || a.FreeCount() != total-3 {
			t.Fatalf("after allocate = allocated %d / free %d, want 3 / %d",
				a.PageCount(), a.FreeCount(), total-3)
		}
		// Every claimed page is owned by the step; the oracle stays clean.
		for _, p := range step.Pages {
			if a.PageOwner(p) != "s1" {
				t.Fatalf("PageOwner(%d) = %q, want s1", p, a.PageOwner(p))
			}
		}
		if v := a.Verify(); len(v) != 0 {
			t.Fatalf("post-allocate Verify = %v, want no violations", v)
		}
	})

	t.Run("per-role page-index views", func(t *testing.T) {
		a, err := NewSharedArena(total, targetOff)
		if err != nil {
			t.Fatalf("NewSharedArena = %v", err)
		}
		// The two roles index the SAME physical table with different offsets:
		// draft from 0, target from targetOff. That is the adjusted
		// page-index handling of the unified arena.
		if a.RoleOffset(RoleDraft) != 0 {
			t.Fatalf("draft offset = %d, want 0", a.RoleOffset(RoleDraft))
		}
		if a.RoleOffset(RoleTarget) != targetOff {
			t.Fatalf("target offset = %d, want %d", a.RoleOffset(RoleTarget), targetOff)
		}

		// Allocate 2 pages for the draft, then accept them so they enter the
		// target's view. The same physical page must carry a different index
		// in each role's view.
		step, err := a.Allocate("s1", RoleDraft, 2)
		if err != nil {
			t.Fatalf("Allocate = %v", err)
		}
		if err := a.Accept("s1"); err != nil {
			t.Fatalf("Accept(s1) = %v", err)
		}
		draftView := a.RoleView(RoleDraft)
		targetView := a.RoleView(RoleTarget)
		// Both views see the same two physical pages, but the target view is
		// shifted by targetOff.
		if len(draftView) != 2 || len(targetView) != 2 {
			t.Fatalf("views = draft %v / target %v, want 2 pages each", draftView, targetView)
		}
		for i, p := range step.Pages {
			if draftView[i] != p {
				t.Fatalf("draft view[%d] = %d, want physical page %d", i, draftView[i], p)
			}
			if targetView[i] != p+int64(targetOff) {
				t.Fatalf("target view[%d] = %d, want physical page %d + offset %d",
					i, targetView[i], p, targetOff)
			}
		}
		// The views are genuinely distinct apertures of one allocation.
		if reflect.DeepEqual(draftView, targetView) {
			t.Fatalf("draft and target views are identical (%v); the offsets must differ", draftView)
		}
		if v := a.Verify(); len(v) != 0 {
			t.Fatalf("post-accept Verify = %v, want no violations", v)
		}
	})

	t.Run("accepted step keeps pages", func(t *testing.T) {
		a, err := NewSharedArena(total, targetOff)
		if err != nil {
			t.Fatalf("NewSharedArena = %v", err)
		}
		step, err := a.Allocate("keep", RoleDraft, 3)
		if err != nil {
			t.Fatalf("Allocate = %v", err)
		}
		freeBefore := a.FreeCount()
		if err := a.Accept("keep"); err != nil {
			t.Fatalf("Accept(keep) = %v", err)
		}
		// An accepted step KEEPS its pages: nothing is returned to the free set.
		if a.FreeCount() != freeBefore {
			t.Fatalf("free after accept = %d, want %d (accepted step keeps its pages)",
				a.FreeCount(), freeBefore)
		}
		if a.PageCount() != 3 {
			t.Fatalf("allocated after accept = %d, want 3 (kept)", a.PageCount())
		}
		// The kept pages are still owned by the step and present in the target view.
		for _, p := range step.Pages {
			if a.PageOwner(p) != "keep" {
				t.Fatalf("PageOwner(%d) = %q after accept, want keep", p, a.PageOwner(p))
			}
		}
		targetView := a.RoleView(RoleTarget)
		if len(targetView) != 3 {
			t.Fatalf("target view after accept = %v, want the 3 kept pages", targetView)
		}
		// Reconciling the same step again is refused: a step reconciles once.
		if err := a.Accept("keep"); !errors.Is(err, ErrStepAlreadyReconciled) {
			t.Fatalf("second Accept = %v, want ErrStepAlreadyReconciled", err)
		}
		if v := a.Verify(); len(v) != 0 {
			t.Fatalf("post-accept Verify = %v, want no violations", v)
		}
	})

	t.Run("rejected step returns pages", func(t *testing.T) {
		a, err := NewSharedArena(total, targetOff)
		if err != nil {
			t.Fatalf("NewSharedArena = %v", err)
		}
		step, err := a.Allocate("drop", RoleDraft, 3)
		if err != nil {
			t.Fatalf("Allocate = %v", err)
		}
		// A rejected step RETURNS its pages to the shared arena.
		if err := a.Reject("drop"); err != nil {
			t.Fatalf("Reject(drop) = %v", err)
		}
		if a.FreeCount() != total {
			t.Fatalf("free after reject = %d, want %d (rejected step returns its pages)",
				a.FreeCount(), total)
		}
		if a.PageCount() != 0 {
			t.Fatalf("allocated after reject = %d, want 0 (all returned)", a.PageCount())
		}
		// The returned pages are no longer owned by the step and are gone from
		// the target view.
		for _, p := range step.Pages {
			if a.PageOwner(p) != "" {
				t.Fatalf("PageOwner(%d) = %q after reject, want free", p, a.PageOwner(p))
			}
		}
		if targetView := a.RoleView(RoleTarget); len(targetView) != 0 {
			t.Fatalf("target view after reject = %v, want empty", targetView)
		}
		// The returned pages are reusable: a fresh allocation of the same size
		// succeeds and the oracle stays clean.
		if _, err := a.Allocate("again", RoleDraft, 3); err != nil {
			t.Fatalf("re-allocate after reject = %v, want ok", err)
		}
		if v := a.Verify(); len(v) != 0 {
			t.Fatalf("post-reject Verify = %v, want no violations", v)
		}
	})

	t.Run("fail-closed over-allocation", func(t *testing.T) {
		a, err := NewSharedArena(total, targetOff)
		if err != nil {
			t.Fatalf("NewSharedArena = %v", err)
		}
		// Claim 5 of the 8 pages, then ask for 4 more: only 3 are free, so the
		// arena must refuse with a typed FitTooBig, not allocate into the void.
		if _, err := a.Allocate("big", RoleDraft, 5); err != nil {
			t.Fatalf("Allocate(big,5) = %v, want ok", err)
		}
		_, err = a.Allocate("over", RoleTarget, 4)
		if err == nil {
			t.Fatalf("Allocate(over,4) with only 3 free = nil, want a typed refusal")
		}
		var fit *compute.FitError
		if !errors.As(err, &fit) {
			t.Fatalf("over-allocation error = %v, want *compute.FitError", err)
		}
		if fit.Verdict != compute.FitTooBig {
			t.Fatalf("FitError verdict = %v, want FitTooBig", fit.Verdict)
		}
		if fit.Want != 4 || fit.Avail != 3 {
			t.Fatalf("FitError want/avail = %d/%d, want 4/3", fit.Want, fit.Avail)
		}
		// The refusal is fail-closed: nothing was allocated, the page table is
		// unchanged, and the oracle is clean.
		if a.PageCount() != 5 || a.FreeCount() != 3 {
			t.Fatalf("after refused over-allocation = allocated %d / free %d, want 5 / 3",
				a.PageCount(), a.FreeCount())
		}
		if v := a.Verify(); len(v) != 0 {
			t.Fatalf("post-refusal Verify = %v, want no violations", v)
		}
	})

	t.Run("structural refusals", func(t *testing.T) {
		// Zero-page arena is refused structurally.
		if _, err := NewSharedArena(0, 0); !errors.Is(err, ErrNonPositivePages) {
			t.Fatalf("NewSharedArena(0,0) = %v, want ErrNonPositivePages", err)
		}
		// Negative target offset is refused structurally.
		if _, err := NewSharedArena(total, -1); !errors.Is(err, ErrNegativeRoleOffset) {
			t.Fatalf("NewSharedArena(%d,-1) = %v, want ErrNegativeRoleOffset", total, err)
		}
		a, err := NewSharedArena(total, targetOff)
		if err != nil {
			t.Fatalf("NewSharedArena = %v", err)
		}
		// Zero-page allocation is refused structurally.
		if _, err := a.Allocate("z", RoleDraft, 0); !errors.Is(err, ErrNonPositiveAllocation) {
			t.Fatalf("Allocate(z,0) = %v, want ErrNonPositiveAllocation", err)
		}
		// Empty step id is refused structurally.
		if _, err := a.Allocate("", RoleDraft, 1); !errors.Is(err, ErrEmptyStepID) {
			t.Fatalf("Allocate(\"\") = %v, want ErrEmptyStepID", err)
		}
		// Reconciling an unknown step is refused.
		if err := a.Accept("nope"); !errors.Is(err, ErrUnknownStep) {
			t.Fatalf("Accept(nope) = %v, want ErrUnknownStep", err)
		}
		if err := a.Reject("nope"); !errors.Is(err, ErrUnknownStep) {
			t.Fatalf("Reject(nope) = %v, want ErrUnknownStep", err)
		}
	})
}
