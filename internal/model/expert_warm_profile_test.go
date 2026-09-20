package model

import (
	"bytes"
	"strings"
	"testing"
)

// expert_warm_profile_test.go — the CW-10 witnesses for #13349 (parent #12640, epic #12952).
//
// The leaf's claim is a SELECTION claim, so the tests are selection tests: given a recorded demand
// profile and a tier index, the plan must be deterministic, must fit the budget, must reject a
// stale/wrong-identity profile, and must never invent a projection the tier does not carry. The
// execution leaf #13343 owns the separate claim that a selected warm actually populates residency.
//
// expertWarmTestTier builds the same miniaturized MoE the checkpoint-tier tests build, so the plan
// is exercised against real indexed names and real per-expert strides rather than a mock index.

// expertWarmTestTier returns a model whose routed experts are checkpoint slabs, its tier, and one
// expert projection's byte stride.
func expertWarmTestTier(t *testing.T, hidden, experts, topK int, hostBytes int64) (*ExpertCheckpointTier, int64) {
	t.Helper()
	_, tier, stride := expertCheckpointTestModel(t, hidden, experts, topK, hostBytes)
	return tier, stride
}

// expertWarmV41Tier builds the same slabs under the native V4.1 arch, so the tier indexes the
// ffn.experts.<e>.{w1,w3,w2} spelling instead of the historical mlp.experts one. The bytes are the
// resident model's own, so the only difference from expertWarmTestTier is the indexed name.
func expertWarmV41Tier(t *testing.T, hidden, experts, topK int, hostBytes int64) (*ExpertCheckpointTier, int64) {
	t.Helper()
	resident := expertPrefetchModel(t, hidden, experts, topK)

	var blob []byte
	fused := make([]FusedExpertTensor, 0, 3)
	for _, proj := range []string{"gate", "up", "down"} {
		suffix := proj + "_proj.weight"
		offset := int64(len(blob))
		for e := 0; e < experts; e++ {
			blob = append(blob, resident.q4kw[expertName(0, e, suffix)].raw...)
		}
		fused = append(fused, FusedExpertTensor{
			Name:    "blk.0.ffn_" + proj + "_exps.weight",
			Layer:   0,
			Proj:    proj + "_proj",
			Arch:    "deepseek41",
			Quant:   ExpertCheckpointQ4K,
			Offset:  offset,
			Experts: experts,
			Rows:    hidden,
			Cols:    hidden,
		})
	}
	stride := int64(len(resident.q4kw[expertName(0, 0, "gate_proj.weight")].raw))
	tier := NewExpertCheckpointTier(hostBytes)
	if err := tier.AddShard(bytes.NewReader(blob), int64(len(blob)), fused); err != nil {
		t.Fatalf("AddShard over a V4.1-arch checkpoint: %v", err)
	}
	return tier, stride
}

func expertWarmTestIdentity() ExpertWarmProfileIdentity {
	return ExpertWarmProfileIdentity{Checkpoint: "rev-abc", Quantization: "Q2_K", Layout: "ffn.experts"}
}

// TestExpertCacheWarmProfile is the named witness required by #13349. It covers the four required
// cases (empty, duplicate, stale, over-budget) plus the deterministic-ordering and byte-ceiling
// invariants, all against a real tier index.
func TestExpertCacheWarmProfile(t *testing.T) {
	const H, E, K = 256, 8, 4
	id := expertWarmTestIdentity()

	t.Run("absent_or_empty_profile_is_empty", func(t *testing.T) {
		tier, _ := expertWarmTestTier(t, H, E, K, 1<<20)
		for _, profile := range []ExpertWarmProfile{
			{Identity: id},
			{Identity: id, Entries: nil},
		} {
			plan := tier.ExpertWarmPlanFor(profile, id, 1<<20)
			if !plan.Empty() {
				t.Fatalf("empty profile selected %d experts; want none", len(plan.Selected))
			}
			if plan.Reason != ExpertWarmReasonNoProfile {
				t.Fatalf("reason = %q; want %q", plan.Reason, ExpertWarmReasonNoProfile)
			}
		}
	})

	t.Run("budget_zero_declines", func(t *testing.T) {
		tier, _ := expertWarmTestTier(t, H, E, K, 0)
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{{Layer: 0, Expert: 0, Count: 9}}}
		plan := tier.ExpertWarmPlanFor(profile, id, 0)
		if !plan.Empty() || plan.Reason != ExpertWarmReasonBudgetZero {
			t.Fatalf("zero budget: selected=%d reason=%q; want empty/budget_zero", len(plan.Selected), plan.Reason)
		}
	})

	t.Run("stale_identity_is_rejected", func(t *testing.T) {
		tier, _ := expertWarmTestTier(t, H, E, K, 1<<20)
		profile := ExpertWarmProfile{
			Identity: ExpertWarmProfileIdentity{Checkpoint: "other", Quantization: "Q2_K", Layout: "ffn.experts"},
			Entries:  []ExpertWarmDemandEntry{{Layer: 0, Expert: 0, Count: 9}},
		}
		plan := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if !plan.Empty() || plan.Reason != ExpertWarmReasonStaleIdentity {
			t.Fatalf("stale profile: selected=%d reason=%q; want empty/stale_identity", len(plan.Selected), plan.Reason)
		}
	})

	t.Run("duplicate_rows_select_each_projection_once", func(t *testing.T) {
		tier, _ := expertWarmTestTier(t, H, E, K, 1<<20)
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{
			{Layer: 0, Expert: 0, Count: 5},
			{Layer: 0, Expert: 0, Count: 5},
			{Layer: 0, Expert: 0, Count: 5},
		}}
		plan := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if plan.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("reason = %q; want planned", plan.Reason)
		}
		seen := map[string]bool{}
		for _, s := range plan.Selected {
			if seen[s.Name] {
				t.Fatalf("projection %s selected twice", s.Name)
			}
			seen[s.Name] = true
		}
		if len(plan.Selected) != 3 {
			t.Fatalf("selected %d projections; want one per routed projection (3)", len(plan.Selected))
		}
	})

	t.Run("byte_budget_is_a_ceiling", func(t *testing.T) {
		tier, stride := expertWarmTestTier(t, H, E, K, 1<<20)
		// Demand every expert, then cap the budget at two projections' worth.
		var entries []ExpertWarmDemandEntry
		for e := 0; e < E; e++ {
			entries = append(entries, ExpertWarmDemandEntry{Layer: 0, Expert: e, Count: float64(E - e)})
		}
		profile := ExpertWarmProfile{Identity: id, Entries: entries}
		plan := tier.ExpertWarmPlanFor(profile, id, 2*stride)
		if plan.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("reason = %q; want planned", plan.Reason)
		}
		if plan.Bytes > 2*stride {
			t.Fatalf("plan bytes = %d; budget ceiling %d exceeded", plan.Bytes, 2*stride)
		}
		if len(plan.Selected) != 2 {
			t.Fatalf("selected %d projections under a 2-stride budget; want 2", len(plan.Selected))
		}
	})

	t.Run("oversized_first_row_cannot_starve_a_smaller_fit", func(t *testing.T) {
		tier, stride := expertWarmTestTier(t, H, E, K, 1<<20)
		// Rank expert 0 highest but give a budget smaller than one projection, then a budget that
		// fits exactly one: the selector must skip the oversized and take the fitting one.
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{
			{Layer: 0, Expert: 0, Count: 100},
			{Layer: 0, Expert: 1, Count: 1},
		}}
		plan := tier.ExpertWarmPlanFor(profile, id, stride)
		if plan.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("reason = %q; want planned", plan.Reason)
		}
		if plan.Bytes != stride || len(plan.Selected) != 1 {
			t.Fatalf("bytes=%d selected=%d; want one projection at %d bytes", plan.Bytes, len(plan.Selected), stride)
		}
		if plan.Selected[0].Expert != 0 {
			t.Fatalf("selected expert %d; want the highest-demand expert 0", plan.Selected[0].Expert)
		}
	})

	t.Run("nothing_fits_reports_no_fit", func(t *testing.T) {
		tier, stride := expertWarmTestTier(t, H, E, K, 1<<20)
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{{Layer: 0, Expert: 0, Count: 9}}}
		plan := tier.ExpertWarmPlanFor(profile, id, stride-1)
		if !plan.Empty() || plan.Reason != ExpertWarmReasonNoFit {
			t.Fatalf("selected=%d reason=%q; want empty/no_fit", len(plan.Selected), plan.Reason)
		}
	})

	t.Run("unknown_expert_is_not_invented", func(t *testing.T) {
		tier, _ := expertWarmTestTier(t, H, E, K, 1<<20)
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{
			{Layer: 0, Expert: E + 5, Count: 9},
			{Layer: 99, Expert: 0, Count: 9},
		}}
		plan := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if !plan.Empty() || plan.Reason != ExpertWarmReasonNoIndexedExpert {
			t.Fatalf("selected=%d reason=%q; want empty/no_indexed_expert", len(plan.Selected), plan.Reason)
		}
	})

	t.Run("ranking_is_deterministic_and_demand_ordered", func(t *testing.T) {
		tier, _ := expertWarmTestTier(t, H, E, K, 1<<20)
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{
			{Layer: 0, Expert: 3, Count: 2},
			{Layer: 0, Expert: 1, Count: 7},
			{Layer: 0, Expert: 2, Count: 5},
		}}
		first := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if first.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("reason = %q; want planned", first.Reason)
		}
		// Selected must be ordered demand-desc across the flattened projections.
		for i := 1; i < len(first.Selected); i++ {
			if first.Selected[i-1].Demand < first.Selected[i].Demand {
				t.Fatalf("selection not demand-ordered at %d: %v < %v",
					i, first.Selected[i-1].Demand, first.Selected[i].Demand)
			}
		}
		// Two runs over the same inputs are byte-identical (map order must not leak).
		second := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if first.String() != second.String() {
			t.Fatalf("plan not deterministic:\n %s\n %s", first.String(), second.String())
		}
	})

	t.Run("selected_names_are_canonical_v41_leaves", func(t *testing.T) {
		tier, _ := expertWarmV41Tier(t, H, E, K, 1<<20)
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{{Layer: 0, Expert: 0, Count: 1}}}
		plan := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if plan.Reason != ExpertWarmReasonPlanned {
			t.Fatalf("reason = %q; want planned", plan.Reason)
		}
		for _, s := range plan.Selected {
			if !strings.Contains(s.Name, ".ffn.experts.0.") || !strings.HasSuffix(s.Name, ".weight") {
				t.Fatalf("selected name %q is not a canonical V4.1 routed-expert name", s.Name)
			}
			if !tier.Has(s.Name) {
				t.Fatalf("selected name %q is not indexed by the tier", s.Name)
			}
			if s.Bytes <= 0 {
				t.Fatalf("selected %q carries non-positive bytes %d", s.Name, s.Bytes)
			}
		}
		if len(plan.Selected) != 3 {
			t.Fatalf("selected %d projections; want one per V4.1 routed leaf (3)", len(plan.Selected))
		}
	})

	t.Run("nil_tier_declines_without_panicking", func(t *testing.T) {
		var tier *ExpertCheckpointTier
		profile := ExpertWarmProfile{Identity: id, Entries: []ExpertWarmDemandEntry{{Layer: 0, Expert: 0, Count: 1}}}
		plan := tier.ExpertWarmPlanFor(profile, id, 1<<20)
		if !plan.Empty() || plan.Reason != ExpertWarmReasonNoIndexedExpert {
			t.Fatalf("nil tier: selected=%d reason=%q; want empty/no_indexed_expert", len(plan.Selected), plan.Reason)
		}
	})
}
