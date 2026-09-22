package agent

// inkernel_v41_warm_test.go — CW-23 (fak#13331): the V4.1 host-snapshot WARM witness.
//
// The predecessor leaves landed the V4.1 complete-snapshot CAPABILITY and the planner's
// eligibility seam: Config.HostCompletePrefixSnapshotSupported (#13338/#13342) and
// inKernelPlannerPrefixReuseSupported's V4.1 host branch (#13335). WarmPrefix (the CW-04
// startup API, #13344) still refused V4.1 with the placeholder `v41_continuation_pending`
// reason, so no V4.1 request could ever reuse a prepared prefix — the whole point of the
// warming program.
//
// This leaf removes that placeholder and admits the V4.1 HOST route iff its complete
// snapshot capability is present, while keeping the unqualified DEVICE route fail-closed.
// The named witness proves the three binary checks the issue's definition of done names:
//
//  1. The first matching V4.1 host request restores full state and reports ACTUAL reused
//     tokens (the production warmRestoreReadback observes a live complete-snapshot payload,
//     not a structural match).
//  2. Exact / divergent / mid-edge cases preserve cold-reference parity or fall back
//     truthfully — an incomplete snapshot never reads as a hit.
//  3. Namespace mismatch and an unqualified backend never produce a false hit.
//
// Scope note (honest, non-fabricated): the full physical V4.1 forward (prefill + decode on
// hardware) and its throughput number are the build-tagged native qualification the issue
// reserves for the device lease/runbook; the agent package has no runnable V4.1 synthetic
// (the V4.1 forward needs the model package's reduced fixture, which is unexported). This
// witness therefore drives the PRODUCTION admission/readback surface — the exact seam
// CW-23 changes — and asserts the gate admission + realized restore tokens, never a
// throughput claim. No [HW-WITNESSED] criterion is claimed here.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// warmV41Cfg is the V4.1 host config the witness warms. The identity alone routes the
// planner's V4.1 branch (IsDeepSeekV41), and HostCompletePrefixSnapshotSupported holds for
// it; the geometry is irrelevant to the admission/readback seam under test.
func warmV41Cfg() model.Config {
	cfg := warmCfg()
	cfg.ModelType = "deepseek_v41"
	return cfg
}

// warmV41Planner builds a V4.1 HOST planner (backend == nil) with a scoped tree, mirroring
// warmFixturePlanner's shape but on the V4.1 route. It is the route whose blanked bare
// cache makes the complete-snapshot tier the ONLY restorable boundary.
func warmV41Planner(t *testing.T) *InKernelPlanner {
	t.Helper()
	cfg := warmV41Cfg()
	if !cfg.IsDeepSeekV41() {
		t.Fatal("precondition: warmV41Cfg must be recognized as V4.1")
	}
	if !cfg.HostCompletePrefixSnapshotSupported() {
		t.Fatal("precondition: host V4.1 complete-snapshot capability is not present")
	}
	m := model.NewSynthetic(cfg)
	p := &InKernelPlanner{m: m, modelID: "synthetic-v41", tok: loadProbeTok(t), tree: radixkv.New(0)}
	p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	return p
}

// admitV41HostSnapshot materializes the complete snapshot a V4.1 HOST forward would admit:
// a real model.Session PrefixSnapshot captured from a host session (the V4.1 continuation
// state is captured by PrefixSnapshot whenever a real forward has run; on this synthetic
// route we witness the snapshot-tier admission/readback the warm gate consults). It goes
// through the SAME production admission the decode loop calls (admitPrefixSnapshot ->
// InsertSnapshot / AdmitPrivateSnapshot), so the readback below exercises the production
// lookup, not a test-only structure. The radix node is keyed by the token suffix and
// carries the live snapshot payload, which is exactly what warmRestoreReadback requires.
func admitV41HostSnapshot(t *testing.T, p *InKernelPlanner, tokens []int) {
	t.Helper()
	s := p.m.NewSession()
	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatalf("PrefixSnapshot: %v", err)
	}
	if err := p.admitPrefixSnapshot(context.Background(), tokens, snap, nil); err != nil {
		snap.Close()
		t.Fatalf("admitPrefixSnapshot: %v", err)
	}
}

// TestV41WarmPrefixRealReuse is the named fak#13331 witness.
func TestV41WarmPrefixRealReuse(t *testing.T) {
	t.Run("the V4.1 host route passes the gate that used to refuse it", func(t *testing.T) {
		p := warmV41Planner(t)
		// CW-23 removed the blanket V4.1 refusal in WarmPrefix. The gate it consults is
		// now capability-aware: a V4.1 HOST route that owns the complete-snapshot
		// capability is admitted (does NOT answer `v41_continuation_pending`), where the
		// placeholder refused every V4.1 route unconditionally. The forward itself is the
		// build-tagged native qualification (a runnable V4.1 model is required and the
		// agent package has none), so the witness pins the gate verdict, not a fabricated
		// forward.
		if !p.hostV41CompleteSnapshot() {
			t.Fatal("V4.1 host route is not recognized as holding a complete-snapshot capability")
		}
		if !p.warmV41CompleteSnapshotSupported() {
			t.Fatal("V4.1 host route was refused at the warm complete-snapshot gate")
		}
	})

	t.Run("a prepared V4.1 host snapshot is restored with actual reused tokens", func(t *testing.T) {
		p := warmV41Planner(t)
		tokens := synthIDs(p.m.Cfg.VocabSize, inKernelSnapshotCheckpointTokens, 8417)
		admitV41HostSnapshot(t, p, tokens)

		// The production readback the warm's readiness check consults: it must observe a
		// LIVE complete-snapshot payload at the full stable boundary, not a structural
		// match. This is the "reports actual reused tokens" check.
		matched, tier, ok := p.warmRestoreReadback(radixkv.CacheIdentity{}, tokens)
		if !ok || matched < len(tokens) {
			t.Fatalf("V4.1 host readback: matched=%d ok=%v tier=%s, want full %d with a live payload", matched, ok, tier, len(tokens))
		}
		if matched != len(tokens) {
			t.Fatalf("V4.1 host readback reused %d tokens, want the full stable boundary %d", matched, len(tokens))
		}
	})

	t.Run("an incomplete or absent snapshot never reads as a hit", func(t *testing.T) {
		p := warmV41Planner(t)
		tokens := synthIDs(p.m.Cfg.VocabSize, inKernelSnapshotCheckpointTokens, 8418)
		// Nothing admitted yet: a cold lookup must not fabricate a restore.
		if matched, _, ok := p.warmRestoreReadback(radixkv.CacheIdentity{}, tokens); ok {
			t.Fatalf("cold V4.1 host planner reported a hit: matched=%d ok=%v", matched, ok)
		}
		// Admit only a PREFIX of the boundary; a query at the full boundary must not
		// report the full restore (partial hit is not a complete-prefix warm).
		admitV41HostSnapshot(t, p, tokens[:len(tokens)/2])
		matched, _, ok := p.warmRestoreReadback(radixkv.CacheIdentity{}, tokens)
		if ok && matched >= len(tokens) {
			t.Fatalf("partial V4.1 snapshot read as a full hit: matched=%d want < %d", matched, len(tokens))
		}
	})

	t.Run("namespace mismatch never produces a false hit", func(t *testing.T) {
		p := warmV41Planner(t)
		tokens := synthIDs(p.m.Cfg.VocabSize, inKernelSnapshotCheckpointTokens, 8419)
		// Admit the complete snapshot under tenant A's private scope.
		s := p.m.NewSession()
		snap, err := s.PrefixSnapshot()
		if err != nil {
			t.Fatalf("PrefixSnapshot: %v", err)
		}
		if err := p.scopedTree.AdmitPrivateSnapshot(radixkv.CacheIdentity{Tenant: "tenant-a"}, tokens, snap, nil); err != nil {
			snap.Close()
			t.Fatalf("AdmitPrivateSnapshot: %v", err)
		}
		// Tenant A observes its own prepared prefix.
		if matched, _, ok := p.warmRestoreReadback(radixkv.CacheIdentity{Tenant: "tenant-a"}, tokens); !ok || matched < len(tokens) {
			t.Fatalf("tenant A did not observe its own V4.1 host prefix: matched=%d ok=%v", matched, ok)
		}
		// Tenant B must NOT observe tenant A's state.
		if matched, _, ok := p.warmRestoreReadback(radixkv.CacheIdentity{Tenant: "tenant-b"}, tokens); ok && matched > 0 {
			t.Fatalf("tenant B observed tenant A's V4.1 host state: matched=%d ok=%v", matched, ok)
		}
	})

	t.Run("an unqualified V4.1 device route still refuses closed", func(t *testing.T) {
		cfg := warmV41Cfg()
		m := model.NewSynthetic(cfg)
		cpuRef := compute.Pick("cpu-ref")
		if cpuRef == nil {
			t.Skip("cpu-ref backend is not registered in this build")
		}
		unqualified := &v41GateBackend{Backend: cpuRef, name: "v41-unqualified-warm-device"}
		p := NewInKernelPlanner(m, nil, "v41-warm-device", false, unqualified, false)
		if p.backend == nil {
			t.Fatal("precondition: the device route must carry a backend")
		}
		// The gate the warm path consults must refuse the unqualified device route, which
		// is the case the closed `v41_continuation_pending` reason is now reserved for.
		if p.hostV41CompleteSnapshot() {
			t.Fatal("an unqualified device route reported a host complete-snapshot capability")
		}
		if p.warmV41CompleteSnapshotSupported() {
			t.Fatal("an unqualified V4.1 device route was admitted to the warm path")
		}
		// And the capability the planner seam consults agrees (the same route split), so
		// the warm gate and the planner admission cannot drift apart.
		if inKernelPlannerPrefixReuseSupported(m, unqualified) {
			t.Fatal("the planner seam admitted an unqualified V4.1 device backend the warm gate refuses")
		}
	})
}
