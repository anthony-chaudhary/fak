package fitplan

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
)

func loadDeploymentFixture(t *testing.T, name string) DeploymentInput {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	var in DeploymentInput
	if err := json.Unmarshal(raw, &in); err != nil {
		t.Fatal(err)
	}
	return in
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=Pin the tiered-refusal contract: a capacity refusal over known pools names next_tiers best-first with byte-proven shortfalls; unknown capacity proves no tier; eligible reports omit the key. when=changed:internal/fitplan/next_tiers*,internal/fitplan/deployment*
func TestDeploymentRefusalNamesNextTiers(t *testing.T) {
	t.Run("too-small-names-multi-device-only", func(t *testing.T) {
		r, err := EvaluatePlacement(loadDeploymentFixture(t, "ex-too-small.json"))
		if err != nil {
			t.Fatal(err)
		}
		if r.CapacityStatus != StatusRefused || r.Eligible {
			t.Fatalf("report=%+v", r)
		}
		// Independent recomputation from the report's own pool rows:
		// rank charges 17179869184 + 30064771072 = 47244640256 required vs
		// 4294967296 + 4294967296 = 8589934592 declared usable.
		const wantShortfall = int64(38654705664) // 47244640256 - 8589934592
		var recomputed int64
		for _, p := range r.Pools {
			if p.RequiredBytes > p.UsableBytes {
				recomputed += p.RequiredBytes - p.UsableBytes
			}
		}
		if recomputed != wantShortfall {
			t.Fatalf("recomputed shortfall=%d want=%d pools=%+v", recomputed, wantShortfall, r.Pools)
		}
		if len(r.NextTiers) != 1 {
			t.Fatalf("next_tiers=%+v", r.NextTiers)
		}
		step := r.NextTiers[0]
		if step.Tier != NextTierMultiDevice || step.ShortfallBytes != wantShortfall {
			t.Fatalf("step=%+v", step)
		}
		// Movable bytes are 30064771072 < shortfall and no host pool is
		// declared, so host_offload must be absent.
		for _, s := range r.NextTiers {
			if s.Tier == NextTierHostOffload {
				t.Fatalf("host_offload must be absent: %+v", r.NextTiers)
			}
		}
		if !strings.Contains(step.Note, strconv.FormatInt(wantShortfall, 10)) {
			t.Fatalf("note=%q", step.Note)
		}
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"next_tiers"`) || !strings.Contains(string(b), `"tier":"multi_device"`) || !strings.Contains(string(b), "38654705664") {
			t.Fatalf("json=%s", b)
		}
	})
	t.Run("host-pool-makes-host-offload-viable", func(t *testing.T) {
		r, err := EvaluatePlacement(loadDeploymentFixture(t, "ex-host-offload.json"))
		if err != nil {
			t.Fatal(err)
		}
		if r.CapacityStatus != StatusRefused || r.Eligible {
			t.Fatalf("report=%+v", r)
		}
		// Shortfall: (17179869184-8589934592) + (30064771072-25769803776).
		const wantShortfall = int64(12884901888)
		if len(r.NextTiers) != 2 {
			t.Fatalf("next_tiers=%+v", r.NextTiers)
		}
		if r.NextTiers[0].Tier != NextTierHostOffload || r.NextTiers[0].ShortfallBytes != wantShortfall {
			t.Fatalf("first=%+v", r.NextTiers[0])
		}
		if r.NextTiers[1].Tier != NextTierMultiDevice || r.NextTiers[1].ShortfallBytes != wantShortfall {
			t.Fatalf("second=%+v", r.NextTiers[1])
		}
	})
	t.Run("unknown-capacity-proves-no-tier", func(t *testing.T) {
		r, err := EvaluatePlacement(loadDeploymentFixture(t, "ex-unknown-capacity.json"))
		if err != nil {
			t.Fatal(err)
		}
		if r.CapacityStatus != StatusUnknown || r.Eligible {
			t.Fatalf("report=%+v", r)
		}
		found := false
		for _, reason := range r.Reasons {
			if reason == PlacementPoolUnknown {
				found = true
			}
		}
		if !found {
			t.Fatalf("reasons=%v", r.Reasons)
		}
		if r.NextTiers != nil {
			t.Fatalf("next_tiers=%+v", r.NextTiers)
		}
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), "next_tiers") {
			t.Fatalf("json=%s", b)
		}
	})
	t.Run("eligible-omits-next-tiers", func(t *testing.T) {
		r, err := EvaluatePlacement(loadDeploymentFixture(t, "dense-16g.json"))
		if err != nil {
			t.Fatal(err)
		}
		if r.CapacityStatus != StatusFits || !r.Eligible {
			t.Fatalf("report=%+v", r)
		}
		if r.NextTiers != nil {
			t.Fatalf("next_tiers=%+v", r.NextTiers)
		}
		b, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(b), "next_tiers") {
			t.Fatalf("json=%s", b)
		}
	})
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=Keep compare on the same tiered-refusal contract as plan/check by delegating to EvaluatePlacement. when=changed:internal/fitplan/next_tiers*,internal/fitplan/deployment*
func TestDeploymentCompareInheritsNextTiers(t *testing.T) {
	// 16+32 GiB pools: the asymmetric placement fits; the equal 4/4 split
	// charges 22 GiB per rank and overruns the smaller pool by 6 GiB.
	in := loadDeploymentFixture(t, "ex-asym-supported.json")
	equal := in.Placement
	equal.Name = "equal"
	equal.Ranks = append([]RankPlacement(nil), in.Placement.Ranks...)
	equal.Ranks[0].Shards = 4
	equal.Ranks[1].Shards = 4
	reports, err := ComparePlacements(DeploymentComparisonInput{Pools: in.Pools, Placements: []Placement{in.Placement, equal}})
	if err != nil {
		t.Fatal(err)
	}
	if !reports[0].Eligible || reports[0].NextTiers != nil {
		t.Fatalf("fits=%+v", reports[0])
	}
	if reports[1].CapacityStatus != StatusRefused || len(reports[1].NextTiers) != 1 || reports[1].NextTiers[0].Tier != NextTierMultiDevice || reports[1].NextTiers[0].ShortfallBytes != 6442450944 {
		t.Fatalf("refused=%+v", reports[1])
	}
}
