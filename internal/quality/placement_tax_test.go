package quality

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/placementtax"
)

func TestReviewPlacementTaxDefaultsOffForLocalProposal(t *testing.T) {
	r, err := ReviewPlacementTax(nil)
	if err != nil {
		t.Fatal(err)
	}
	if r.Required || r.Report != nil {
		t.Fatalf("local proposal unexpectedly requires placement analysis: %+v", r)
	}
}

func TestReviewPlacementTaxRequiresCrossDomainLedger(t *testing.T) {
	c := reviewComparison()
	r, err := ReviewPlacementTax(&c)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Required || r.Report == nil {
		t.Fatalf("missing required report: %+v", r)
	}
	if !r.Report.Reference.Feasibility.Feasible || !r.Report.Candidate.Feasibility.Feasible {
		t.Fatalf("unexpected infeasibility: %+v", r.Report)
	}
	if len(r.Report.Candidate.Ledger) == 0 || r.Report.Relative == nil {
		t.Fatalf("missing ledger or ratios: %+v", r.Report)
	}
}

func TestReviewPlacementTaxRefusesUnmatchedEnvelope(t *testing.T) {
	c := reviewComparison()
	c.Workload.Quality.Target = ""
	r, err := ReviewPlacementTax(&c)
	if err == nil || !strings.Contains(err.Error(), "quality target") {
		t.Fatalf("err=%v", err)
	}
	if !r.Required || r.Report != nil {
		t.Fatalf("invalid cross-domain proposal lost requirement: %+v", r)
	}
}

func reviewComparison() placementtax.Comparison {
	return placementtax.Comparison{
		Workload:  placementtax.Workload{ID: "decode", Units: 10, Unit: "token", Quality: placementtax.QualityEnvelope{ModelID: "qwen38", Precision: "q4", SequenceLength: 1024, BatchSize: 1, Target: "same logits"}},
		Topology:  placementtax.Topology{Domains: []placementtax.Domain{{ID: "h1"}, {ID: "h2"}}, Nodes: []placementtax.Node{{ID: "a", DomainID: "h1", Capacity: placementtax.Capacity{MemoryBytes: 100, ComputeUnits: 1}}, {ID: "b", DomainID: "h2", Capacity: placementtax.Capacity{MemoryBytes: 100, ComputeUnits: 1}}}, Links: []placementtax.Link{{ID: "ab", FromNode: "a", ToNode: "b", BandwidthBytesPerSecond: 1000, Latency: time.Millisecond}}},
		Reference: placementtax.Placement{Name: "one", Allocations: []placementtax.Allocation{{NodeID: "a", Demand: placementtax.Capacity{MemoryBytes: 100, ComputeUnits: 1}}}, UsefulCompute: reviewCost(100 * time.Millisecond)},
		Candidate: placementtax.Placement{Name: "two", Allocations: []placementtax.Allocation{{NodeID: "a", Demand: placementtax.Capacity{MemoryBytes: 50, ComputeUnits: .5}}, {NodeID: "b", Demand: placementtax.Capacity{MemoryBytes: 50, ComputeUnits: .5}}}, UsefulCompute: reviewCost(60 * time.Millisecond), Transfers: []placementtax.Transfer{{LinkID: "ab", Bytes: 10, Messages: 1, Provenance: placementtax.ProvenanceEstimated}}},
	}
}

func reviewCost(d time.Duration) placementtax.ComponentCost {
	return placementtax.ComponentCost{Latency: d, Cycle: d, Provenance: placementtax.ProvenanceEstimated}
}
