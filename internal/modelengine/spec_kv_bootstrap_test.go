package modelengine

import (
	"errors"
	"testing"
)

// TestSpecKVBootstrap pins the ported request-owned speculative-KV ledger
// (SGLang PR #38978 @3c6a6c80): per-request bounded reservation against a
// shared pool, hold-under-pressure (zero grant instead of release-and-requeue),
// exactly-once release on the abort path, handoff on completion, and the
// allocation/release symmetry invariant Free + owned == PoolBlocks after every
// operation.
func TestSpecKVBootstrap(t *testing.T) {
	t.Run("rejects_nonpositive_configuration", func(t *testing.T) {
		for _, cfg := range [][2]int{{0, 4}, {8, 0}, {-1, 4}, {8, -1}} {
			if _, err := NewSpecKVBootstrap(cfg[0], cfg[1]); err == nil {
				t.Fatalf("NewSpecKVBootstrap(%d, %d) accepted, want error", cfg[0], cfg[1])
			}
		}
	})

	t.Run("admit_reserve_bound_by_cap_and_pool", func(t *testing.T) {
		b, err := NewSpecKVBootstrap(8, 4)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		if err := b.Admit("a"); err != nil {
			t.Fatalf("Admit(a): %v", err)
		}
		if err := b.Admit("b"); err != nil {
			t.Fatalf("Admit(b): %v", err)
		}
		if err := b.Admit("a"); !errors.Is(err, ErrSpecKVDuplicateAdmission) {
			t.Fatalf("duplicate Admit(a) err = %v, want ErrSpecKVDuplicateAdmission", err)
		}

		// Cap-bound draw: asks 6, owns at most 4.
		if got, err := b.Reserve("a", 6); err != nil || got != 4 {
			t.Fatalf("Reserve(a,6) = %d/%v, want 4/nil", got, err)
		}
		if got, err := b.Owned("a"); err != nil || got != 4 {
			t.Fatalf("Owned(a) = %d/%v, want 4/nil", got, err)
		}
		if got := b.Free(); got != 4 {
			t.Fatalf("Free = %d, want 4", got)
		}

		// Exact-fit draw for the second request empties the pool.
		if got, err := b.Reserve("b", 4); err != nil || got != 4 {
			t.Fatalf("Reserve(b,4) = %d/%v, want 4/nil", got, err)
		}
		if got := b.Free(); got != 0 {
			t.Fatalf("Free = %d, want 0", got)
		}

		// At cap and at pool floor, further draws grant zero without error.
		if got, err := b.Reserve("b", 1); err != nil || got != 0 {
			t.Fatalf("Reserve(b,1) at cap = %d/%v, want 0/nil", got, err)
		}
		if got, err := b.Reserve("a", 1); err != nil || got != 0 {
			t.Fatalf("Reserve(a,1) at pool floor = %d/%v, want 0/nil", got, err)
		}
		if got, err := b.Reserve("a", 0); err != nil || got != 0 {
			t.Fatalf("Reserve(a,0) = %d/%v, want 0/nil", got, err)
		}
		if got, err := b.Reserve("a", -3); err != nil || got != 0 {
			t.Fatalf("Reserve(a,-3) = %d/%v, want 0/nil", got, err)
		}
	})

	t.Run("pool_pressure_grants_zero_without_releasing", func(t *testing.T) {
		b, err := NewSpecKVBootstrap(4, 8)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		if err := b.Admit("a"); err != nil {
			t.Fatalf("Admit(a): %v", err)
		}
		if err := b.Admit("b"); err != nil {
			t.Fatalf("Admit(b): %v", err)
		}
		if got, err := b.Reserve("a", 4); err != nil || got != 4 {
			t.Fatalf("Reserve(a,4) = %d/%v, want 4/nil", got, err)
		}
		// The pool is exhausted by a's admission-time draw; b's bootstrap
		// wait must not force a to release and re-queue.
		if got, err := b.Reserve("b", 2); err != nil || got != 0 {
			t.Fatalf("Reserve(b,2) under pressure = %d/%v, want 0/nil", got, err)
		}
		if got, err := b.Owned("a"); err != nil || got != 4 {
			t.Fatalf("Owned(a) after pressure = %d/%v, want 4/nil", got, err)
		}
		if got, err := b.Owned("b"); err != nil || got != 0 {
			t.Fatalf("Owned(b) after pressure = %d/%v, want 0/nil", got, err)
		}
		if got := b.Free(); got != 0 {
			t.Fatalf("Free = %d, want 0", got)
		}
	})

	t.Run("abort_releases_exactly_once", func(t *testing.T) {
		b, err := NewSpecKVBootstrap(8, 4)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		if err := b.Admit("a"); err != nil {
			t.Fatalf("Admit(a): %v", err)
		}
		if err := b.Admit("b"); err != nil {
			t.Fatalf("Admit(b): %v", err)
		}
		if got, err := b.Reserve("a", 3); err != nil || got != 3 {
			t.Fatalf("Reserve(a,3) = %d/%v, want 3/nil", got, err)
		}
		if got, err := b.Reserve("b", 2); err != nil || got != 2 {
			t.Fatalf("Reserve(b,2) = %d/%v, want 2/nil", got, err)
		}
		if got := b.Free(); got != 3 {
			t.Fatalf("Free = %d, want 3", got)
		}

		released, err := b.Abort("a")
		if err != nil || released != 3 {
			t.Fatalf("Abort(a) = %d/%v, want 3/nil", released, err)
		}
		if got, err := b.Owned("a"); err != nil || got != 0 {
			t.Fatalf("Owned(a) after abort = %d/%v, want 0/nil", got, err)
		}
		if got := b.Free(); got != 6 {
			t.Fatalf("Free after abort = %d, want 6", got)
		}

		// Second abort finds nothing left: no double release.
		if released, err := b.Abort("a"); err != nil || released != 0 {
			t.Fatalf("second Abort(a) = %d/%v, want 0/nil", released, err)
		}
		if got := b.Free(); got != 6 {
			t.Fatalf("Free after double abort = %d, want 6", got)
		}

		// Freed blocks are immediately reusable by other admitted requests:
		// no churn wait between a's abort and b's next draw.
		if got, err := b.Reserve("b", 4); err != nil || got != 2 {
			t.Fatalf("Reserve(b,4) after abort = %d/%v, want 2/nil (cap headroom)", got, err)
		}
		if got := b.Free(); got != 4 {
			t.Fatalf("Free = %d, want 4", got)
		}
	})

	t.Run("aborted_id_is_terminal", func(t *testing.T) {
		b, err := NewSpecKVBootstrap(8, 4)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		if err := b.Admit("a"); err != nil {
			t.Fatalf("Admit(a): %v", err)
		}
		if _, err := b.Reserve("a", 2); err != nil {
			t.Fatalf("Reserve(a,2): %v", err)
		}
		if _, err := b.Abort("a"); err != nil {
			t.Fatalf("Abort(a): %v", err)
		}
		if _, err := b.Reserve("a", 1); !errors.Is(err, ErrSpecKVAborted) {
			t.Fatalf("Reserve after abort err = %v, want ErrSpecKVAborted", err)
		}
		if _, err := b.Complete("a"); !errors.Is(err, ErrSpecKVAborted) {
			t.Fatalf("Complete after abort err = %v, want ErrSpecKVAborted", err)
		}
		if err := b.Admit("a"); !errors.Is(err, ErrSpecKVDuplicateAdmission) {
			t.Fatalf("re-Admit after abort err = %v, want ErrSpecKVDuplicateAdmission", err)
		}
	})

	t.Run("complete_hands_off_and_frees_identity", func(t *testing.T) {
		b, err := NewSpecKVBootstrap(8, 4)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		if err := b.Admit("c"); err != nil {
			t.Fatalf("Admit(c): %v", err)
		}
		if got, err := b.Reserve("c", 5); err != nil || got != 4 {
			t.Fatalf("Reserve(c,5) = %d/%v, want 4/nil", got, err)
		}
		if released, err := b.Complete("c"); err != nil || released != 4 {
			t.Fatalf("Complete(c) = %d/%v, want 4/nil", released, err)
		}
		if got := b.Free(); got != 8 {
			t.Fatalf("Free after complete = %d, want 8", got)
		}

		// Handed off: the id no longer resolves.
		if _, err := b.Owned("c"); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Owned after complete err = %v, want ErrSpecKVUnknownRequest", err)
		}
		if _, err := b.Complete("c"); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("second Complete err = %v, want ErrSpecKVUnknownRequest", err)
		}
		if _, err := b.Reserve("c", 1); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Reserve after complete err = %v, want ErrSpecKVUnknownRequest", err)
		}

		// The identity is free for a fresh admission.
		if err := b.Admit("c"); err != nil {
			t.Fatalf("re-Admit(c) after complete: %v", err)
		}
		if got, err := b.Reserve("c", 2); err != nil || got != 2 {
			t.Fatalf("fresh Reserve(c,2) = %d/%v, want 2/nil", got, err)
		}
	})

	t.Run("unknown_ids_report_the_sentinel", func(t *testing.T) {
		b, err := NewSpecKVBootstrap(8, 4)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		if _, err := b.Reserve("ghost", 1); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Reserve(ghost) err = %v, want ErrSpecKVUnknownRequest", err)
		}
		if _, err := b.Abort("ghost"); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Abort(ghost) err = %v, want ErrSpecKVUnknownRequest", err)
		}
		if _, err := b.Complete("ghost"); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Complete(ghost) err = %v, want ErrSpecKVUnknownRequest", err)
		}
		if _, err := b.Owned("ghost"); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Owned(ghost) err = %v, want ErrSpecKVUnknownRequest", err)
		}
		if err := b.Admit(""); !errors.Is(err, ErrSpecKVUnknownRequest) {
			t.Fatalf("Admit(\"\") err = %v, want ErrSpecKVUnknownRequest", err)
		}
	})

	t.Run("allocation_release_symmetry", func(t *testing.T) {
		const pool, cap = 10, 3
		b, err := NewSpecKVBootstrap(pool, cap)
		if err != nil {
			t.Fatalf("NewSpecKVBootstrap: %v", err)
		}
		ids := []string{"r1", "r2", "r3", "r4"}
		for _, id := range ids {
			if err := b.Admit(id); err != nil {
				t.Fatalf("Admit(%s): %v", id, err)
			}
		}
		live := map[string]bool{}
		for _, id := range ids {
			live[id] = true
		}
		symmetry := func(step string) {
			t.Helper()
			total := b.Free()
			for id := range live {
				owned, err := b.Owned(id)
				if err != nil {
					t.Fatalf("%s: Owned(%s): %v", step, id, err)
				}
				total += owned
			}
			if total != pool {
				t.Fatalf("%s: Free + owned = %d, want pool %d", step, total, pool)
			}
			if got := b.Free(); got < 0 || got > pool {
				t.Fatalf("%s: Free = %d outside [0,%d]", step, got, pool)
			}
		}

		// Exact-grant trace over pool=10 cap=3: cap clamps, pool pressure
		// partially funds, abort/handoff returns, and every block comes back.
		type reserveOp struct {
			id          string
			want        int
			expectGrant int
		}
		reserves := []reserveOp{
			{id: "r1", want: 3, expectGrant: 3},
			{id: "r2", want: 5, expectGrant: 3}, // cap-bounds to 3
			{id: "r3", want: 4, expectGrant: 3}, // cap-bounds to 3
			{id: "r4", want: 2, expectGrant: 1}, // pool pressure: one left
		}
		for i, op := range reserves {
			step := "reserve" + string(rune('A'+i))
			got, err := b.Reserve(op.id, op.want)
			if err != nil {
				t.Fatalf("%s: Reserve(%s,%d): %v", step, op.id, op.want, err)
			}
			if got != op.expectGrant {
				t.Fatalf("%s: Reserve(%s,%d) granted %d, want %d", step, op.id, op.want, got, op.expectGrant)
			}
			symmetry(step)
		}

		// State now: r1/r2/r3 own 3 each, r4 owns 1, free = 0.
		if released, err := b.Abort("r1"); err != nil || released != 3 {
			t.Fatalf("Abort(r1) = %d/%v, want 3/nil", released, err)
		}
		delete(live, "r1")
		symmetry("abort-r1")

		// r4's next draw is funded by r1's abort: no churn wait.
		if got, err := b.Reserve("r4", 2); err != nil || got != 2 {
			t.Fatalf("Reserve(r4,2) after abort = %d/%v, want 2/nil", got, err)
		}
		symmetry("reserve-r4-reuse")

		// r2 sits at its cap: zero grant without error.
		if got, err := b.Reserve("r2", 1); err != nil || got != 0 {
			t.Fatalf("Reserve(r2,1) at cap = %d/%v, want 0/nil", got, err)
		}
		symmetry("reserve-r2-cap")

		if released, err := b.Complete("r3"); err != nil || released != 3 {
			t.Fatalf("Complete(r3) = %d/%v, want 3/nil", released, err)
		}
		delete(live, "r3")
		symmetry("complete-r3")

		if released, err := b.Abort("r2"); err != nil || released != 3 {
			t.Fatalf("Abort(r2) = %d/%v, want 3/nil", released, err)
		}
		delete(live, "r2")
		symmetry("abort-r2")

		if released, err := b.Complete("r4"); err != nil || released != 3 {
			t.Fatalf("Complete(r4) = %d/%v, want 3/nil", released, err)
		}
		delete(live, "r4")
		symmetry("complete-r4")
		if got := b.Free(); got != pool {
			t.Fatalf("Free after all terminals = %d, want pool %d", got, pool)
		}
	})
}
