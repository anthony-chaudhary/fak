package hwutil

import (
	"encoding/json"
	"reflect"
	"testing"
)

func loadFrozen(t *testing.T, host string) *Snapshot {
	t.Helper()
	var s Snapshot
	if err := json.Unmarshal([]byte(frozenFixture), &s); err != nil {
		t.Fatalf("unmarshal frozen fixture: %v", err)
	}
	s.Host = host
	return &s
}

func summaryOf(t *testing.T, sums []RASDiagSummary, id RASDiagCheckID) RASDiagSummary {
	t.Helper()
	for _, s := range sums {
		if s.Check == id {
			return s
		}
	}
	t.Fatalf("no summary for %v", id)
	return RASDiagSummary{}
}

func TestRASDiagCatalog(t *testing.T) {
	t.Run("closed_vocabulary_matches_upstream_enum", func(t *testing.T) {
		// Ordinals from rasDiagnosticsCheckId in src/ras/diagnostics.h.
		want := map[RASDiagCheckID]string{
			0: "gpu_model", 1: "cuda_driver_version", 2: "ecc", 3: "nvlink",
			4: "nccl_env", 5: "rdma_topo", 6: "iommu_mode", 7: "ats",
			8: "xid_sxid", 9: "nvidia_driver_version", 10: "paths",
		}
		if int(RASDiagCheckCount) != 11 {
			t.Fatalf("check count = %d, want 11", RASDiagCheckCount)
		}
		cat := RASDiagCatalog()
		if len(cat) != len(want) {
			t.Fatalf("catalog len = %d", len(cat))
		}
		for i, c := range cat {
			if int(c.ID) != i || c.ID.String() != want[c.ID] || c.Collect == nil || c.Summarize == nil {
				t.Fatalf("entry %d = id %d %q collect=%v summarize=%v", i, c.ID, c.ID.String(), c.Collect != nil, c.Summarize != nil)
			}
		}
	})

	t.Run("frozen_fixture_single_node", func(t *testing.T) {
		sums := RunRASDiag([]*Snapshot{loadFrozen(t, "node-01")}, RASDiagOptions{})
		if len(sums) != int(RASDiagCheckCount) {
			t.Fatalf("summaries = %d", len(sums))
		}
		g := summaryOf(t, sums, RASDiagGPUModel)
		if g.Status != RASDiagOK || g.Reference != "1x NVIDIA RTX 5070 Ti" {
			t.Fatalf("gpu_model = %+v", g)
		}
		if e := summaryOf(t, sums, RASDiagECC); e.Status != RASDiagOK || len(e.Outliers) != 0 {
			t.Fatalf("ecc = %+v", e)
		}
		// The fixture has only a thunderbolt link and no NVLink counters.
		for _, id := range []RASDiagCheckID{RASDiagNVLink, RASDiagCUDADriverVersion, RASDiagNCCLEnv,
			RASDiagRDMATopo, RASDiagIOMMUMode, RASDiagATS, RASDiagXIDSXID, RASDiagNVIDIADriverVersion, RASDiagPaths} {
			s := summaryOf(t, sums, id)
			if s.Status != RASDiagUnobserved || !reflect.DeepEqual(s.Unobserved, []string{"node-01"}) {
				t.Fatalf("%v = %+v, want unobserved", id, s)
			}
		}
	})

	t.Run("gpu_model_mismatch_names_outlier_against_lowest_host", func(t *testing.T) {
		b := loadFrozen(t, "node-02")
		b.GPUs[0].Name = "NVIDIA H200"
		sums := RunRASDiag([]*Snapshot{b, loadFrozen(t, "node-01")}, RASDiagOptions{})
		g := summaryOf(t, sums, RASDiagGPUModel)
		if g.Status != RASDiagMismatch || g.Reference != "1x NVIDIA RTX 5070 Ti" || !reflect.DeepEqual(g.Outliers, []string{"node-02"}) {
			t.Fatalf("gpu_model = %+v", g)
		}
	})

	t.Run("ecc_uncorrected_and_threshold", func(t *testing.T) {
		b := loadFrozen(t, "node-02")
		one := int64(1)
		b.GPUs[0].ECCUncorrectedTotal = &one
		e := summaryOf(t, RunRASDiag([]*Snapshot{loadFrozen(t, "node-01"), b}, RASDiagOptions{}), RASDiagECC)
		if e.Status != RASDiagFault || !reflect.DeepEqual(e.Outliers, []string{"node-02"}) {
			t.Fatalf("ecc uncorrected = %+v", e)
		}
		v := loadFrozen(t, "node-04")
		zero := int64(0)
		v.GPUs[0].ECCUncorrectedTotal = &zero
		v.GPUs[0].ECCUncorrectableVolatile = &one
		if e := summaryOf(t, RunRASDiag([]*Snapshot{v}, RASDiagOptions{}), RASDiagECC); e.Status != RASDiagFault {
			t.Fatalf("volatile uncorrected must take precedence and fault: %+v", e)
		}
		c := loadFrozen(t, "node-03")
		five := int64(5)
		c.GPUs[0].ECCCorrectableVolatile = &five
		if e := summaryOf(t, RunRASDiag([]*Snapshot{c}, RASDiagOptions{}), RASDiagECC); e.Status != RASDiagOK {
			t.Fatalf("threshold 0 must disable corrected check: %+v", e)
		}
		if e := summaryOf(t, RunRASDiag([]*Snapshot{c}, RASDiagOptions{ECCCorrectedThreshold: 5}), RASDiagECC); e.Status != RASDiagFault {
			t.Fatalf("corrected at threshold must fault: %+v", e)
		}
	})

	t.Run("nvlink_inactive_link_faults", func(t *testing.T) {
		speed := 50.0
		a := loadFrozen(t, "node-01")
		a.Links = append(a.Links, Link{Kind: LinkNVLink, ID: "nvl0", SpeedGBps: &speed, State: LinkUp})
		b := loadFrozen(t, "node-02")
		b.Links = append(b.Links, Link{Kind: LinkNVLink, ID: "nvl0", State: LinkDown})
		n := summaryOf(t, RunRASDiag([]*Snapshot{a, b}, RASDiagOptions{}), RASDiagNVLink)
		if n.Status != RASDiagFault || !reflect.DeepEqual(n.Outliers, []string{"node-02"}) {
			t.Fatalf("nvlink = %+v", n)
		}
	})

	t.Run("nvlink_all_states_unreadable_is_unobserved", func(t *testing.T) {
		a := loadFrozen(t, "node-01")
		a.Links = append(a.Links, Link{Kind: LinkNVLink, ID: "nvl0", State: LinkStateUnknown})
		if n := summaryOf(t, RunRASDiag([]*Snapshot{a}, RASDiagOptions{}), RASDiagNVLink); n.Status != RASDiagUnobserved {
			t.Fatalf("nvlink = %+v, want unobserved", n)
		}
	})

	t.Run("no_snapshots_is_unobserved_not_pass", func(t *testing.T) {
		for _, s := range RunRASDiag(nil, RASDiagOptions{}) {
			if s.Status != RASDiagUnobserved {
				t.Fatalf("%v = %+v", s.Check, s)
			}
		}
	})
}
