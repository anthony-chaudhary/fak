package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// fak#12742 (scoped arm): on the host hybrid seam a KV-bearing match in one scope
// (here a short prompt another tenant promoted to fleet) can be shallower than a
// complete snapshot the requesting tenant admitted at a deeper checkpoint. The
// scoped lookup must restore the deeper snapshot instead of the shallow fleet KV,
// and greedy output must stay identical to a cache-disabled planner.
// fak-test:runtime fast est=1s lane=default
func TestInKernelScopedHostHybridPrefersDeeperSnapshotOverShallowKV(t *testing.T) {
	cfg := tinyHybridCfg()
	block := inKernelSnapshotCheckpointTokens
	common := synthIDs(cfg.VocabSize, 3*block, 12746)
	withSuffix := func(n int, seed uint64, first int) []int {
		s := synthIDs(cfg.VocabSize, n, seed)
		s[0] = first
		return append(append([]int(nil), common...), s...)
	}
	turn1 := withSuffix(20, 12747, 1)
	turn2 := withSuffix(20, 12748, 2)
	shallow := append([]int(nil), common[:40]...)
	deep := ((len(turn1) - 1) / block) * block
	if deep <= len(shallow) || deep > len(common) {
		t.Fatalf("precondition deep=%d shallow=%d common=%d", deep, len(shallow), len(common))
	}

	p := reusePlanner(true, false, cfg)
	p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	if !inKernelHostSnapshotReuse(p) || p.backend != nil {
		t.Fatal("precondition: scoped host-session snapshot seam")
	}
	ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
	ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "")
	run := func(p *InKernelPlanner, ctx context.Context, ids []int) (tokens []int, matched int, tier radixkv.SnapshotTier) {
		t.Helper()
		_, _, _, matched, tier, _, _, _, err := p.generateReusedContextWithBias(
			ctx, ids, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				tokens = append(tokens, id)
				return false
			})
		if err != nil {
			t.Fatalf("generate len=%d: %v", len(ids), err)
		}
		return
	}

	// Tenant A admits a complete snapshot at the deep grid checkpoint.
	if _, m, _ := run(p, ctxA, turn1); m != 0 {
		t.Fatalf("cold tenant-a turn reused %d, want 0", m)
	}
	// Tenant B admits an exact short KV leaf and promotes it to fleet visibility.
	if _, m, _ := run(p, ctxB, shallow); m != 0 {
		t.Fatalf("cold tenant-b shallow turn reused %d, want 0", m)
	}
	if err := p.scopedTree.Promote(radixkv.ScopeTenant, radixkv.CacheIdentity{Tenant: "tenant-b"}, shallow); err != nil {
		t.Fatalf("promote shallow prefix: %v", err)
	}
	owner := radixkv.CacheIdentity{Tenant: "tenant-a"}
	if kv, _, km, scope, err := p.scopedTree.Lookup(owner, turn2); err != nil || kv == nil || km != len(shallow) || scope != radixkv.ScopeFleet {
		t.Fatalf("precondition: shallow fleet KV lookup kv=%v matched=%d scope=%v err=%v, want fleet KV at %d", kv != nil, km, scope, err, len(shallow))
	}
	if snap, _, sm, _, _, err := p.scopedTree.LookupSnapshotTieredContext(context.Background(), owner, turn2); err != nil || snap == nil || sm < deep {
		t.Fatalf("precondition: tenant snapshot matched=%d err=%v, want >= %d", sm, err, deep)
	} else {
		snap.Close()
	}

	warm, m2, tier2 := run(p, ctxA, turn2)
	t.Logf("turn2 reused=%d tier=%s deep=%d shallow=%d", m2, tier2, deep, len(shallow))
	if m2 < deep || m2 > len(common) {
		t.Fatalf("turn2 reused=%d, want in [%d,%d] (shallow fleet KV=%d hid the deeper tenant snapshot)", m2, deep, len(common), len(shallow))
	}
	cold, cm, _ := run(reusePlanner(false, false, cfg), context.Background(), turn2)
	if cm != 0 || !eqInts(warm, cold) {
		t.Fatalf("restored greedy %v != cold %v (cold matched=%d)", warm, cold, cm)
	}
}
