package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestInKernelHostHybridSharedPrefixReuse is the regression witness for the Apple-silicon
// Qwen3.8-27B hybrid Metal path (p.backend == nil, p.metal == true). Before the fix a
// repeated-prefix turn reported `cacheable=Ktok reused=0tok` because a mid-edge split on a
// recurrent hybrid carries no KV (radixkv.go:624 requires CanEvict()==nil, which a GDN
// cache refuses), so only an EXACT full-prompt duplicate could reuse. The host-session
// path now materializes a complete PrefixSnapshot at the adaptive block boundary, so a
// sibling that shares that block restores it. The assertion is realized reuse > 0.
func TestInKernelHostHybridSharedPrefixReuse(t *testing.T) {
	cfg := tinyHybridCfg()
	if !cfg.IsQwen35Hybrid() {
		t.Fatal("precondition: tinyHybridCfg must be a Qwen35 hybrid")
	}
	p := reusePlanner(true, false, cfg)
	if p.backend != nil {
		t.Fatal("precondition: reusePlanner must run with a nil backend (host Metal seam)")
	}

	common := synthIDs(cfg.VocabSize, 192, 9001) // three whole 64-token snapshot blocks
	turn1 := append(append([]int(nil), common...), synthIDs(cfg.VocabSize, 20, 9002)...)
	turn2 := append(append([]int(nil), common...), synthIDs(cfg.VocabSize, 20, 9003)...)

	if _, m := decode(p, turn1, 1); m != 0 {
		t.Fatalf("cold turn reused %d tokens, want 0", m)
	}
	_, matched := decode(p, turn2, 1)
	if matched <= 0 {
		t.Fatalf("host hybrid shared-prefix turn reused %d tokens, want > 0 (cacheable>0 but reused must be non-zero)", matched)
	}
	if matched > len(common) {
		t.Fatalf("host hybrid reused %d tokens, want <= shared prefix %d", matched, len(common))
	}
	t.Logf("HOST HYBRID REUSE: shared prefix %d tokens, second turn reused %d", len(common), matched)
}

// TestInKernelHostHybridSharedPrefixReuseScoped is the same witness on the scoped (tenant)
// path the turnkey multi-session server uses.
func TestInKernelHostHybridSharedPrefixReuseScoped(t *testing.T) {
	cfg := tinyHybridCfg()
	p := reusePlanner(true, false, cfg)
	p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
	ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
	common := synthIDs(cfg.VocabSize, 192, 9101)
	turn1 := append(append([]int(nil), common...), synthIDs(cfg.VocabSize, 20, 9102)...)
	turn2 := append(append([]int(nil), common...), synthIDs(cfg.VocabSize, 20, 9103)...)

	run := func(ctx context.Context, ids []int) (matched int) {
		t.Helper()
		_, _, _, matched, _, _, _, _, err := p.generateReusedContextWithBias(
			ctx, ids, 1, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(int) bool { return false },
		)
		if err != nil {
			t.Fatalf("generate len=%d: %v", len(ids), err)
		}
		return matched
	}

	if m := run(ctxA, turn1); m != 0 {
		t.Fatalf("cold scoped turn reused %d tokens, want 0", m)
	}
	matched := run(ctxA, turn2)
	if matched <= 0 {
		t.Fatalf("scoped host hybrid shared-prefix turn reused %d tokens, want > 0", matched)
	}
	// A different tenant must NOT observe tenant A's snapshot.
	ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "")
	if m := run(ctxB, turn2); m != 0 {
		t.Fatalf("tenant B observed tenant A state: reused %d tokens, want 0", m)
	}
	t.Logf("SCOPED HOST HYBRID REUSE: shared prefix %d tokens, same-tenant reused %d", len(common), matched)
}

// TestInKernelHostHybridPrefixReuseGranularity pins the boundary of the fix: reuse is
// available at a COMPLETE snapshot block (64 tokens) and keeps failing open below one block,
// where no complete PrefixSnapshot can be materialized. This is the honest version of the
// old "hybrid split always falls back" behavior — narrower, not removed.
func TestInKernelHostHybridPrefixReuseGranularity(t *testing.T) {
	cfg := tinyHybridCfg()
	block := inKernelSnapshotCheckpointTokens

	for _, tc := range []struct {
		name      string
		prefixLen int
		wantGT0   bool
	}{
		{"below one block", block - 1, false},
		{"exactly one block", block, true},
		{"two blocks", 2 * block, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := reusePlanner(true, false, cfg)
			common := synthIDs(cfg.VocabSize, tc.prefixLen, 9201)
			// Keep both prompts longer than the shared prefix so a partial hit (not an
			// exact duplicate) is what is measured.
			turn1 := append(append([]int(nil), common...), synthIDs(cfg.VocabSize, 30, 9202)...)
			turn2 := append(append([]int(nil), common...), synthIDs(cfg.VocabSize, 30, 9203)...)
			decode(p, turn1, 1)
			_, matched := decode(p, turn2, 1)
			if tc.wantGT0 && matched <= 0 {
				t.Fatalf("prefix %d: reused %d, want > 0", tc.prefixLen, matched)
			}
			if !tc.wantGT0 && matched != 0 {
				t.Fatalf("prefix %d: reused %d, want 0 (no complete block to materialize)", tc.prefixLen, matched)
			}
			t.Logf("prefix=%d reused=%d", tc.prefixLen, matched)
		})
	}
}
