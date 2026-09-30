package agent

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Each suffix puts the cold checkpoint beyond the common prefix. The second
// sibling discovers the structural boundary; the third must restore its complete
// recurrent and attention state, including tokens between snapshot-grid points.
// fak-test:runtime fast est=3s lane=default
func TestInKernelAdaptiveCheckpointMaterializesSharedDevicePrefix(t *testing.T) {
	t.Setenv("FAK_INKERNEL_RADIX", "on")
	for _, commonLen := range []int{65, 150, 191} {
		t.Run(fmt.Sprintf("common=%d", commonLen), func(t *testing.T) {
			cfg := tinyHybridCfg()
			common := synthIDs(cfg.VocabSize, commonLen, 16631)
			prompt := func(first int, seed uint64) []int {
				suffix := synthIDs(cfg.VocabSize, inKernelSnapshotCheckpointTokens+26, seed)
				suffix[0] = first
				return append(append([]int(nil), common...), suffix...)
			}
			prompts := [][]int{prompt(1, 16632), prompt(2, 16633), prompt(3, 16634)}
			newPlanner := func(reuse bool) *InKernelPlanner {
				backend := &countingBackend{Backend: compute.Default(), deviceMemory: true}
				p := NewInKernelPlanner(model.NewSynthetic(cfg), nil, "adaptive-checkpoint-hybrid", false, backend, false)
				p.quant = false
				if !reuse {
					p.tree, p.scopedTree = nil, nil
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
			primeTokens, cacheable, matched, tier := run(p, context.Background(), prompts[0])
			if cacheable != 0 || matched != 0 || tier != radixkv.SnapshotTierMiss {
				t.Fatalf("cold A cacheable=%d matched=%d tier=%s, want 0/0/miss", cacheable, matched, tier)
			}
			// Exact full-prompt snapshots remain usable before any sibling arrives.
			exactTokens, cacheable, matched, tier := run(p, context.Background(), prompts[0])
			if cacheable != len(prompts[0]) || matched != len(prompts[0]) || tier != radixkv.SnapshotTierDeviceL1 || !eqInts(exactTokens, primeTokens) {
				t.Fatalf("exact A cacheable=%d matched=%d tier=%s tokens=%v, want %d/%d/device-l1/%v", cacheable, matched, tier, exactTokens, len(prompts[0]), len(prompts[0]), primeTokens)
			}
			if _, _, matched, _ := run(p, context.Background(), prompts[1]); matched != 0 {
				t.Fatalf("sibling B restored %d tokens before the structural boundary was materialized, want 0", matched)
			}

			boundary, snap, matchedBoundary, tier, err := p.tree.LookupSnapshotTieredContext(context.Background(), common)
			if err != nil {
				t.Fatalf("lookup materialized common boundary: %v", err)
			}
			if boundary != nil {
				p.tree.Done(boundary)
			}
			if snap == nil {
				t.Fatalf("sibling B did not materialize complete common boundary %d", len(common))
			}
			snap.Close()
			if matchedBoundary != len(common) || tier != radixkv.SnapshotTierDeviceL1 {
				t.Fatalf("common boundary matched=%d tier=%s, want %d/%s", matchedBoundary, tier, len(common), radixkv.SnapshotTierDeviceL1)
			}

			warmTokens, cacheable, matched, tier := run(p, context.Background(), prompts[2])
			if cacheable != len(common) || matched != len(common) || tier != radixkv.SnapshotTierDeviceL1 {
				t.Fatalf("sibling C cacheable=%d restored=%d tier=%s, want %d/%d/%s", cacheable, matched, tier, len(common), len(common), radixkv.SnapshotTierDeviceL1)
			}
			accounting := p.nativeCacheAccountingFor(nativeCacheAccountingFacts{
				promptTokens: len(prompts[2]), cacheableTokens: cacheable, matchedTokens: matched, sourceTier: tier, generated: len(warmTokens),
			})
			if accounting.RestoredTokens != len(common) || accounting.ComputedTokens != len(prompts[2])-len(common) || accounting.Operation != model.NativeCachePartialHit {
				t.Fatalf("sibling C actual restore accounting: %+v", accounting)
			}
			coldPlanner := newPlanner(false)
			coldTokens, _, coldMatched, coldTier := run(coldPlanner, context.Background(), prompts[2])
			if coldMatched != 0 || coldTier != radixkv.SnapshotTierMiss || !eqInts(warmTokens, coldTokens) {
				t.Fatalf("cold reference matched=%d tier=%s warm=%v cold=%v", coldMatched, coldTier, warmTokens, coldTokens)
			}
			// Compare all prompt logits, rather than only the argmax used for greedy output.
			full, fullSnap, fullMatched, _, err := p.tree.LookupSnapshotTieredContext(context.Background(), prompts[2])
			if err != nil || fullSnap == nil || fullMatched != len(prompts[2]) {
				t.Fatalf("lookup sibling C full snapshot matched=%d snapshot=%v err=%v", fullMatched, fullSnap != nil, err)
			}
			warmLogits := full.Logits()
			p.tree.Done(full)
			fullSnap.Close()
			reference := coldPlanner.m.NewBackendSession(coldPlanner.backend)
			coldLogits := reference.Prefill(prompts[2])
			reference.Close()
			if len(warmLogits) != len(coldLogits) || len(warmLogits) != cfg.VocabSize {
				t.Fatalf("prompt logits widths warm=%d cold=%d vocab=%d", len(warmLogits), len(coldLogits), cfg.VocabSize)
			}
			for i := range coldLogits {
				if math.Float32bits(warmLogits[i]) != math.Float32bits(coldLogits[i]) {
					t.Fatalf("prompt logit[%d] warm=%g cold=%g", i, warmLogits[i], coldLogits[i])
				}
			}

			t.Run("tenant scope", func(t *testing.T) {
				scoped := newPlanner(true)
				ctxA := WithPrefixCacheIdentity(context.Background(), "tenant-a", "")
				ctxB := WithPrefixCacheIdentity(context.Background(), "tenant-b", "")
				run(scoped, ctxA, prompts[0])
				run(scoped, ctxA, prompts[1])
				warmTenantTokens, _, matchedA, tierA := run(scoped, ctxA, prompts[2])
				_, _, matchedB, tierB := run(scoped, ctxB, prompts[2])
				if matchedA != len(common) || tierA != radixkv.SnapshotTierDeviceL1 || !eqInts(warmTenantTokens, coldTokens) {
					t.Fatalf("same tenant restored=%d tier=%s tokens=%v, want %d/device-l1/%v", matchedA, tierA, warmTenantTokens, len(common), coldTokens)
				}
				if matchedB != 0 || tierB != radixkv.SnapshotTierMiss {
					t.Fatalf("tenant B observed tenant A snapshot: matched=%d tier=%s", matchedB, tierB)
				}
			})
			t.Logf("SW-VERIFIED complete hybrid boundary=%d suffix=%d; no hardware-performance claim", len(common), len(prompts[0])-len(common))
		})
	}
}
