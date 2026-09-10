package agent

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestInKernelAdaptiveCheckpointMaterializesSharedDevicePrefix reproduces the
// multi-request hybrid-cache shape where every prompt is longer than one snapshot
// block after its common prefix. A fixed deepest-before-leaf checkpoint lands in
// each request's private suffix, so later siblings cannot restore the shared block.
func TestInKernelAdaptiveCheckpointMaterializesSharedDevicePrefix(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")

	cfg := tinyHybridCfg()
	common := synthIDs(cfg.VocabSize, 150, 16631)
	commonBlock := (len(common) / inKernelSnapshotCheckpointTokens) * inKernelSnapshotCheckpointTokens
	if commonBlock == 0 || commonBlock == len(common) {
		t.Fatalf("test common prefix length %d does not straddle a snapshot block", len(common))
	}
	prompt := func(first int, seed uint64) []int {
		suffix := synthIDs(cfg.VocabSize, inKernelSnapshotCheckpointTokens+26, seed)
		suffix[0] = first
		return append(append([]int(nil), common...), suffix...)
	}
	prompts := [][]int{
		prompt(1, 16632),
		prompt(2, 16633),
		prompt(3, 16634),
	}

	newPlanner := func(reuse bool) *InKernelPlanner {
		backend := &countingBackend{Backend: compute.Default(), deviceMemory: true}
		p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "adaptive-checkpoint-hybrid", false, backend, false)
		p.quant = false
		if !reuse {
			p.tree = nil
			p.scopedTree = nil
		}
		return p
	}
	run := func(p *InKernelPlanner, ctx context.Context, ids []int) (tokens []int, cacheable, matched int, tier radixkv.SnapshotTier) {
		t.Helper()
		_, _, cacheable, matched, tier, _, _, _, err := p.generateReusedContextWithBias(
			ctx, ids, 4, 0, 0, 0, nil, 0, 0, map[int]bool{}, func(id int) bool {
				tokens = append(tokens, id)
				return false
			},
		)
		if err != nil {
			t.Fatalf("generate prompt_len=%d: %v", len(ids), err)
		}
		return tokens, cacheable, matched, tier
	}

	p := newPlanner(true)
	if _, cacheable, matched, tier := run(p, context.Background(), prompts[0]); cacheable != 0 || matched != 0 || tier != radixkv.SnapshotTierMiss {
		t.Fatalf("cold A cacheable=%d matched=%d tier=%s, want 0/0/miss", cacheable, matched, tier)
	}
	if _, _, matched, _ := run(p, context.Background(), prompts[1]); matched != 0 {
		t.Fatalf("sibling B restored %d tokens before the shared block was materialized, want 0", matched)
	}

	boundary, snap, matchedBlock, tier, err := p.tree.LookupSnapshotTieredContext(context.Background(), common[:commonBlock])
	if err != nil {
		t.Fatalf("lookup materialized common block: %v", err)
	}
	if boundary != nil {
		p.tree.Done(boundary)
	}
	if snap == nil {
		t.Fatalf("sibling B did not materialize reusable common block %d", commonBlock)
	}
	snap.Close()
	if matchedBlock != commonBlock || tier != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("common block matched=%d tier=%s, want %d/%s", matchedBlock, tier, commonBlock, radixkv.SnapshotTierDeviceL1)
	}

	warmTokens, cacheable, matched, tier := run(p, context.Background(), prompts[2])
	if cacheable < len(common) {
		t.Fatalf("sibling C cacheable=%d, want at least literal common prefix %d", cacheable, len(common))
	}
	if matched != commonBlock || tier != radixkv.SnapshotTierDeviceL1 {
		t.Fatalf("sibling C restored=%d tier=%s, want %d/%s", matched, tier, commonBlock, radixkv.SnapshotTierDeviceL1)
	}
	coldTokens, _, coldMatched, coldTier := run(newPlanner(false), context.Background(), prompts[2])
	if coldMatched != 0 || coldTier != radixkv.SnapshotTierMiss {
		t.Fatalf("cache-disabled reference matched=%d tier=%s, want 0/miss", coldMatched, coldTier)
	}
	if !eqInts(warmTokens, coldTokens) {
		t.Fatalf("restored sibling output changed: warm=%v cold=%v", warmTokens, coldTokens)
	}

	t.Run("tenant scope", func(t *testing.T) {
		scoped := newPlanner(true)
		ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
		ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "")
		run(scoped, ctxA, prompts[0])
		run(scoped, ctxA, prompts[1])
		warmTenantTokens, _, matchedA, tierA := run(scoped, ctxA, prompts[2])
		_, _, matchedB, tierB := run(scoped, ctxB, prompts[2])
		if matchedA != commonBlock || tierA != radixkv.SnapshotTierDeviceL1 {
			t.Fatalf("same tenant restored=%d tier=%s, want %d/%s", matchedA, tierA, commonBlock, radixkv.SnapshotTierDeviceL1)
		}
		coldTenantTokens, _, coldTenantMatched, coldTenantTier := run(newPlanner(false), ctxA, prompts[2])
		if coldTenantMatched != 0 || coldTenantTier != radixkv.SnapshotTierMiss {
			t.Fatalf("cache-disabled tenant reference matched=%d tier=%s, want 0/miss", coldTenantMatched, coldTenantTier)
		}
		if !eqInts(warmTenantTokens, coldTenantTokens) {
			t.Fatalf("same-tenant partial restore skipped divergent suffix: warm=%v cold=%v", warmTenantTokens, coldTenantTokens)
		}
		if matchedB != 0 || tierB != radixkv.SnapshotTierMiss {
			t.Fatalf("tenant B observed tenant A snapshot: matched=%d tier=%s", matchedB, tierB)
		}
	})

	t.Logf("SW-VERIFIED adaptive hybrid snapshot block=%d common=%d suffix=%d; no hardware-performance claim", commonBlock, len(common), len(prompts[0])-len(common))
}
