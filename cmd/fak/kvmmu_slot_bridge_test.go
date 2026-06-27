package main

// kvmmu_slot_bridge_test.go is the #915 LOAD-BEARING witness: a REAL internal/session Draining
// transition, delivered through the scheduler's SlotEvent and projected by the host bridge
// (wireSlotFreedKVReclaim), drives a REAL kvmmu.Context.EvictColdest -> model.KVCache.Evict over
// a live residency — the model KV cache shrinks, freeing positions a waiting sequence can
// allocate. The gateway's own unit test could only assert this from a hand-built SlotFreed; this
// closes the wire end-to-end, from a real RunState transition to real freed KV.

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/lifecycle"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// slotBridgeSynthCfg is a tiny synthetic model config — enough real layers/heads to hold a few
// KV positions a reclaim can free. Mirrors the gateway witness's kvmmuSynthCfg.
func slotBridgeSynthCfg() model.Config {
	return model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 260, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "llama",
	}
}

// liveResidencyReclaimer frees the WHOLE residency an ended session held via the existing
// EvictColdest path (EvictUnderBudget(0) == "evict everything unpinned"). It is the proven
// model.KVCache.Evict reclaim (re-RoPE + renumber) ROUTED by the slot-freed edge, not a new
// policy — exactly what the issue points EvictColdest/Evict at. A production host injects a
// reclaimer that maps the SlotFreed TraceID onto the live served residency; this witness holds a
// single residency, so trace->residency routing (the gateway seam's concern) is out of scope.
type liveResidencyReclaimer struct{ ctx *kvmmu.Context }

func (r liveResidencyReclaimer) ReclaimResidency(string) int {
	freed := 0
	for _, sp := range r.ctx.EvictUnderBudget(0) {
		freed += sp.Positions
	}
	return freed
}

// newSlotBridgeServer builds the minimal real gateway.Server the bridge routes into: the default
// in-kernel engine (registered by internal/registrations, blank-imported in main.go) with a
// silent log sink. New falls to the mock planner with no upstream/in-kernel model — fine, the
// witness exercises only the slot-freed KV-reclaim edge, not the chat surface.
func newSlotBridgeServer(t *testing.T) *gateway.Server {
	t.Helper()
	srv, err := gateway.New(gateway.Config{
		EngineID:     "inkernel",
		Model:        "slot-bridge-test",
		Invalidation: "global",
		Logf:         func(string, ...any) {},
	})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	t.Cleanup(func() { srv.Close() })
	return srv
}

// newLiveResidency builds a real synthetic model session with two unpinned (evictable) spans of
// real K/V — the residency a running session would hold — and returns the kvmmu.Context plus the
// session whose Cache.Len() the reclaim shrinks.
func newLiveResidency(t *testing.T) (*kvmmu.Context, *model.Session) {
	t.Helper()
	m := model.NewSynthetic(slotBridgeSynthCfg())
	sess := m.NewSession()
	ctx := kvmmu.New(sess)
	// EvictColdest excludes pinned (system) spans, so use user/tool roles — an evictable residency.
	ctx.Append("seg-user", "user", []int{1, 2, 3, 4})
	ctx.Append("seg-tool", "fetch", []int{5, 6, 7})
	if sess.Cache.Len() == 0 {
		t.Fatalf("residency prefill produced no KV positions; nothing to reclaim")
	}
	return ctx, sess
}

// TestSlotFreedBridgeDrivesRealKVReclaim is the #915 deliverable: with the in-kernel KVMMU flag
// on, a real Draining RunState transition — through the scheduler, the host bridge, and the
// gateway edge — drives a real EvictColdest and the model KV cache shrinks. A preceding HOLD
// (Paused) frees nothing (the warm residency is kept for a resume), proving the edge fires only
// at a terminal boundary.
func TestSlotFreedBridgeDrivesRealKVReclaim(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "on")

	ctx, sess := newLiveResidency(t)
	before := sess.Cache.Len()

	srv := newSlotBridgeServer(t)
	srv.SetKVResidencyReclaimer(liveResidencyReclaimer{ctx: ctx})

	tbl := session.NewTable()
	sched := session.NewScheduler(session.StrictPriority)
	sched.Attach(tbl, session.AttachOptions{})
	wireSlotFreedKVReclaim(sched, srv, nil)

	// A HOLD (paused) frees no KV — a paused session may resume and reuse its warm residency.
	tbl.Transition("sess-915", session.Paused, "")
	if held := sess.Cache.Len(); held != before {
		t.Fatalf("a Paused HOLD must keep warm KV: Len before=%d after=%d", before, held)
	}

	// A terminal Draining transition frees the residency's real KV positions.
	tbl.Transition("sess-915", session.Draining, "")
	after := sess.Cache.Len()
	if after >= before {
		t.Fatalf("Draining transition did not free KV through the bridge: Len before=%d after=%d", before, after)
	}
	t.Logf("#915: real Draining transition -> EvictColdest freed %d KV positions (Len %d->%d)", before-after, before, after)
}

// TestSlotFreedBridgeFlagOffUnchanged is the posture guard (acceptance: flag-off path proven
// unchanged): with FAK_INKERNEL_KVMMU off (the default), a real terminal Draining transition
// drives NO reclaim — the residency is byte-for-byte the pre-#915 served path until an operator
// opts in. The gateway owns the gate; the bridge inherits it.
func TestSlotFreedBridgeFlagOffUnchanged(t *testing.T) {
	t.Setenv("FAK_INKERNEL_KVMMU", "") // explicitly off (the default)

	ctx, sess := newLiveResidency(t)
	before := sess.Cache.Len()

	srv := newSlotBridgeServer(t)
	srv.SetKVResidencyReclaimer(liveResidencyReclaimer{ctx: ctx})

	tbl := session.NewTable()
	sched := session.NewScheduler(session.StrictPriority)
	sched.Attach(tbl, session.AttachOptions{})
	wireSlotFreedKVReclaim(sched, srv, nil)

	tbl.Transition("sess-915", session.Draining, "")
	if after := sess.Cache.Len(); after != before {
		t.Fatalf("flag-off must free no KV: Len before=%d after=%d", before, after)
	}
}

// TestSlotEventToSlotFreedProjection pins the projection contract the wire depends on: every
// SlotCause crosses as the exact lowercase token the gateway keys its KV-free edge on (the two
// terminal causes match internal/lifecycle's Draining/Stopped), and TraceID/Rev are carried.
func TestSlotEventToSlotFreedProjection(t *testing.T) {
	cases := []struct {
		cause     session.SlotCause
		wantToken string
		terminal  bool
	}{
		{session.CauseDraining, lifecycle.TokenDraining, true},
		{session.CauseStopped, lifecycle.TokenStopped, true},
		{session.CausePaused, lifecycle.TokenPaused, false},
		{session.CauseBudgetExhausted, "budget-exhausted", false},
	}
	for _, c := range cases {
		got := slotEventToSlotFreed(session.SlotEvent{TraceID: "trace-915", Cause: c.cause, Rev: 7})
		if got.Cause != c.wantToken {
			t.Fatalf("cause %v projected to %q, want %q", c.cause, got.Cause, c.wantToken)
		}
		if got.TraceID != "trace-915" || got.Rev != 7 {
			t.Fatalf("projection dropped identity: got %+v, want TraceID=trace-915 Rev=7", got)
		}
	}
}
