package gateway

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// kvmmu_slot_reclaim_test.go is the #915 witness: a drain/stop SlotEvent, projected onto the
// gateway's wire-neutral SlotFreed, drives a REAL KV free (kvmmu.Context.EvictColdest ->
// model.KVCache.Evict) for that session's residency — the load-bearing drain/stop↔evict edge
// of #912. It proves three things the acceptance names: (1) a terminal transition frees real
// positions a waiting sequence can allocate; (2) the freed-slot→freed-KV edge fires once per
// terminal transition and never on a HOLD; (3) flag-off is byte-identical (no reclaim at all).

// newSlotReclaimServer is the minimal Server the slot-freed consumer needs: only logf and the
// kvReclaimer seam are exercised, so a bare struct keeps the witness isolated from the gateway's
// engine/abi global registration (no cross-test races on shared registries).
func newSlotReclaimServer() *Server {
	return &Server{logf: func(string, ...any) {}}
}

// liveResidencyReclaimer wraps a real kvmmu.Context and, on a terminal slot-freed event, frees
// the WHOLE residency the ended session held — EvictUnderBudget(0) means "evict everything
// unpinned" via the existing EvictColdest path. This is the existing reclaim ROUTED by the
// slot-freed edge, not a new policy: it drives the proven model.KVCache.Evict (re-RoPE +
// renumber) under the hood, exactly what the issue points EvictColdest/Evict at.
type liveResidencyReclaimer struct{ ctx *kvmmu.Context }

func (r liveResidencyReclaimer) ReclaimResidency(trace string) int {
	freed := 0
	for _, sp := range r.ctx.EvictUnderBudget(0) {
		freed += sp.Positions
	}
	return freed
}

// recordingReclaimer counts ReclaimResidency calls (the edge firings) so a test can assert the
// freed-slot→freed-KV edge fires EXACTLY once per terminal transition and never on a hold.
type recordingReclaimer struct {
	mu    sync.Mutex
	calls []string
}

func (r *recordingReclaimer) ReclaimResidency(trace string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, trace)
	return 3 // a fixed non-zero "positions freed" so the edge's freed return is checkable
}

func (r *recordingReclaimer) snapshot() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.calls...)
}

// TestSlotFreedDrivesRealKVReclaim is the #915 deliverable: with the in-kernel KVMMU flag on, a
// Draining slot-freed event drives a REAL EvictColdest/Evict over a live residency, and the
// model KV cache shrinks — the positions a waiting sequence can now allocate.
func TestSlotFreedDrivesRealKVReclaim(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "on")

	m := model.NewSynthetic(kvmmuSynthCfg())
	sess := m.NewSession()
	ctx := kvmmu.New(sess)
	// Append two NON-system (unpinned, hence evictable) spans of real K/V — the residency a
	// running session would hold. EvictColdest excludes pinned spans, so use user/tool roles.
	ctx.Append("seg-user", "user", []int{1, 2, 3, 4})
	ctx.Append("seg-tool", "fetch", []int{5, 6, 7})

	before := sess.Cache.Len()
	if before == 0 {
		t.Fatalf("residency prefill produced no KV positions; nothing to reclaim")
	}

	srv := newSlotReclaimServer()
	srv.SetKVResidencyReclaimer(liveResidencyReclaimer{ctx: ctx})

	freed, fired := srv.ReclaimKVOnSlotFreed(SlotFreed{TraceID: "trace-915", Cause: lifecycle.TokenDraining, Rev: 7})
	if !fired {
		t.Fatalf("a Draining transition with the flag on must fire the KV-free edge; fired=false")
	}
	if freed <= 0 {
		t.Fatalf("the reclaim freed %d positions, want > 0 (the residency must be evicted)", freed)
	}
	after := sess.Cache.Len()
	if after >= before {
		t.Fatalf("KV residency not freed by the slot-freed edge: Len before=%d after=%d", before, after)
	}
	if got := before - after; got != freed {
		t.Fatalf("reported freed=%d disagrees with the real KV shrink %d (before=%d after=%d)", freed, got, before, after)
	}
	t.Logf("#915: Draining SlotEvent -> real EvictColdest freed %d KV positions (Len %d->%d)", freed, before, after)
}

// TestSlotFreedEdgeFiresOncePerTerminalTransition is the once-per-boundary witness: each of the
// two terminal causes drives EXACTLY one reclaim, and the two HOLD causes (paused,
// budget-exhausted) drive none — the slot is free for scheduling but its warm KV is kept.
func TestSlotFreedEdgeFiresOncePerTerminalTransition(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "on")

	rec := &recordingReclaimer{}
	srv := newSlotReclaimServer()
	srv.SetKVResidencyReclaimer(rec)

	// Terminal transitions free KV exactly once each.
	for _, cause := range []string{lifecycle.TokenDraining, lifecycle.TokenStopped} {
		freed, fired := srv.ReclaimKVOnSlotFreed(SlotFreed{TraceID: "t-" + cause, Cause: cause})
		if !fired || freed != 3 {
			t.Fatalf("terminal cause %q: got fired=%v freed=%d, want true/3", cause, fired, freed)
		}
	}
	// Holds never free KV (a paused session may resume and reuse its warm residency).
	for _, cause := range []string{lifecycle.TokenPaused, "budget-exhausted"} {
		freed, fired := srv.ReclaimKVOnSlotFreed(SlotFreed{TraceID: "t-" + cause, Cause: cause})
		if fired || freed != 0 {
			t.Fatalf("non-terminal HOLD cause %q must not free KV: got fired=%v freed=%d", cause, fired, freed)
		}
	}

	got := rec.snapshot()
	want := []string{"t-" + lifecycle.TokenDraining, "t-" + lifecycle.TokenStopped}
	if len(got) != len(want) {
		t.Fatalf("edge fired %d times, want %d (once per terminal transition): %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("edge firing %d: trace %q, want %q", i, got[i], want[i])
		}
	}
}

// TestSlotFreedFlagOffUnchanged is the posture guard: with FAK_INKERNEL_KVMMU off (the
// default), NO slot-freed cause — terminal or hold — drives a reclaim; the served path is
// byte-for-byte the pre-#915 behavior until an operator opts in.
func TestSlotFreedFlagOffUnchanged(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "") // explicitly off (the default)

	rec := &recordingReclaimer{}
	srv := newSlotReclaimServer()
	srv.SetKVResidencyReclaimer(rec)

	for _, cause := range []string{lifecycle.TokenDraining, lifecycle.TokenStopped, lifecycle.TokenPaused, "budget-exhausted"} {
		freed, fired := srv.ReclaimKVOnSlotFreed(SlotFreed{TraceID: "t", Cause: cause})
		if fired || freed != 0 {
			t.Fatalf("flag-off must be inert for cause %q: got fired=%v freed=%d", cause, fired, freed)
		}
	}
	if calls := rec.snapshot(); len(calls) != 0 {
		t.Fatalf("flag-off must NOT call the reclaimer; got %d call(s): %v", len(calls), calls)
	}
}

// TestSlotFreedNoReclaimerWiredIsNoOp proves the armed-but-unwired posture: flag on and a
// terminal cause, but no reclaimer injected yet, returns (0,false) rather than panicking — the
// edge is live but there is nothing to free.
func TestSlotFreedNoReclaimerWiredIsNoOp(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "on")

	srv := newSlotReclaimServer() // no SetKVResidencyReclaimer
	freed, fired := srv.ReclaimKVOnSlotFreed(SlotFreed{TraceID: "t", Cause: lifecycle.TokenStopped})
	if fired || freed != 0 {
		t.Fatalf("flag on + terminal cause + no reclaimer must be a no-op: got fired=%v freed=%d", fired, freed)
	}
}
