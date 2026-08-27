package placementtax

import (
	"math"
	"strings"
	"testing"
	"time"
)

const gib = uint64(1 << 30)

func TestUnifiedMemorySingleHostBeatsTwoFiniteLinkHosts(t *testing.T) {
	c := baseComparison()
	c.Candidate = Placement{
		Name:          "unified-host",
		Allocations:   []Allocation{{NodeID: "large", Demand: Capacity{MemoryBytes: 80 * gib, ComputeUnits: 8}}},
		UsefulCompute: estimatedCost(95 * time.Millisecond),
	}
	c.Candidate.UsefulCompute.MonetaryUSD = ModeledValue{Value: 1, Modeled: true}
	c.Candidate.UsefulCompute.EnergyJoules = ModeledValue{Value: 2, Modeled: true}
	c.Reference = Placement{
		Name: "finite-link-pair",
		Allocations: []Allocation{
			{NodeID: "small-a", Demand: Capacity{MemoryBytes: 40 * gib, ComputeUnits: 4}},
			{NodeID: "small-b", Demand: Capacity{MemoryBytes: 40 * gib, ComputeUnits: 4}},
		},
		UsefulCompute: estimatedCost(55 * time.Millisecond),
		Transfers: []Transfer{{
			LinkID:     "a-to-b",
			Messages:   1,
			Bytes:      64_000_000,
			Provenance: ProvenanceEstimated,
		}},
	}
	c.Reference.UsefulCompute.MonetaryUSD = ModeledValue{Value: 1.5, Modeled: true}
	c.Reference.UsefulCompute.EnergyJoules = ModeledValue{Value: 2.5, Modeled: true}
	c.Topology.Links[0].MonetaryUSDPerByte = ModeledValue{Value: 1e-9, Modeled: true}
	c.Topology.Links[0].EnergyJoulesPerByte = ModeledValue{Value: 2e-9, Modeled: true}

	report, err := Analyze(c)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Candidate.Feasibility.Feasible || !report.Reference.Feasibility.Feasible {
		t.Fatalf("both placements should be feasible: candidate=%+v reference=%+v",
			report.Candidate.Feasibility, report.Reference.Feasibility)
	}
	if got, want := report.Reference.Communication[0].RawLatency, 65*time.Millisecond; got != want {
		t.Fatalf("finite-link raw communication = %v, want %v", got, want)
	}
	if report.Delta == nil || report.Delta.Latency >= 0 {
		t.Fatalf("unified candidate latency delta = %+v, want negative", report.Delta)
	}
	if report.Relative == nil || report.Relative.Latency.PlacementEfficiency <= 1 {
		t.Fatalf("unified candidate latency efficiency = %+v, want > 1", report.Relative)
	}
	if !report.Delta.MonetaryUSD.Modeled || !report.Delta.EnergyJoules.Modeled {
		t.Fatalf("modeled money and energy deltas were dropped: %+v", report.Delta)
	}
	if got, want := report.Reference.Metrics.MonetaryUSD.Value, 1.564; math.Abs(got-want) > 1e-12 {
		t.Fatalf("reference monetary cost = %.12g, want %.12g", got, want)
	}
	if got, want := report.Reference.Metrics.EnergyJoules.Value, 2.628; math.Abs(got-want) > 1e-12 {
		t.Fatalf("reference energy = %.12g, want %.12g", got, want)
	}
	if report.Delta.MonetaryUSD.Value == report.Delta.EnergyJoules.Value {
		t.Fatalf("money and energy were conflated: %+v", report.Delta)
	}
}

func TestComputeBoundDistributedPlacementWins(t *testing.T) {
	c := distributedComparison()
	c.Candidate.UsefulCompute = estimatedCost(40 * time.Millisecond)
	c.Candidate.Synchronization = estimatedCost(1 * time.Millisecond)
	c.Candidate.Transfers[0].Bytes = 10_000_000
	c.Candidate.Transfers[0].Overlap = Overlap{
		Latency: time.Millisecond,
		Cycle:   time.Millisecond,
	}
	c.Reference.UsefulCompute = estimatedCost(100 * time.Millisecond)

	report, err := Analyze(c)
	if err != nil {
		t.Fatal(err)
	}
	comm := report.Candidate.Communication[0]
	if got, want := comm.RawLatency, 11*time.Millisecond; got != want {
		t.Fatalf("raw communication = %v, want %v", got, want)
	}
	if got, want := comm.HiddenLatency, time.Millisecond; got != want {
		t.Fatalf("hidden communication = %v, want %v", got, want)
	}
	if got, want := comm.ExposedLatency, 10*time.Millisecond; got != want {
		t.Fatalf("exposed communication = %v, want %v", got, want)
	}
	if report.Delta == nil || report.Delta.Latency != -49*time.Millisecond {
		t.Fatalf("latency delta = %+v, want -49ms", report.Delta)
	}
	if report.Delta.Throughput <= 0 {
		t.Fatalf("throughput delta = %v, want positive", report.Delta.Throughput)
	}
	if !report.Relative.Latency.Available || report.Relative.Latency.PenaltyRatio >= 1 {
		t.Fatalf("latency relative = %+v, want candidate win", report.Relative.Latency)
	}
	if !report.Relative.Throughput.Available || report.Relative.Throughput.PlacementEfficiency <= 1 {
		t.Fatalf("throughput relative = %+v, want candidate win", report.Relative.Throughput)
	}
	if got := ledgerCost(report.Candidate.Ledger, ComponentExposedCommunication).Latency; got != 10*time.Millisecond {
		t.Fatalf("communication ledger latency = %v, want only 10ms exposed", got)
	}
}

func TestCommunicationBoundDistributedPlacementLoses(t *testing.T) {
	c := distributedComparison()
	c.Candidate.UsefulCompute = estimatedCost(40 * time.Millisecond)
	c.Candidate.Transfers[0].Bytes = 1_000_000_000
	c.Reference.UsefulCompute = estimatedCost(100 * time.Millisecond)

	report, err := Analyze(c)
	if err != nil {
		t.Fatal(err)
	}
	if report.Delta == nil || report.Delta.Latency <= 0 {
		t.Fatalf("latency delta = %+v, want communication-bound loss", report.Delta)
	}
	if report.Delta.Throughput >= 0 {
		t.Fatalf("throughput delta = %v, want negative", report.Delta.Throughput)
	}
	if got := report.Relative.Latency.PenaltyRatio; got <= 1 {
		t.Fatalf("latency penalty ratio = %v, want > 1", got)
	}
	if got := report.Relative.Throughput.PlacementEfficiency; got >= 1 {
		t.Fatalf("throughput placement efficiency = %v, want < 1", got)
	}
}

func TestInfeasibleSingleHostReferenceRefusesRatios(t *testing.T) {
	c := distributedComparison()
	c.Candidate.UsefulCompute = estimatedCost(70 * time.Millisecond)
	c.Reference.UsefulCompute = estimatedCost(100 * time.Millisecond)
	c.Reference.Allocations[0].Demand.MemoryBytes = 128 * gib

	report, err := Analyze(c)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Candidate.Feasibility.Feasible {
		t.Fatalf("candidate unexpectedly infeasible: %+v", report.Candidate.Feasibility)
	}
	if report.Reference.Feasibility.Feasible {
		t.Fatal("single-host reference should be infeasible")
	}
	if report.Reference.Metrics != nil || len(report.Reference.Ledger) != 0 {
		t.Fatalf("infeasible reference reported metrics: %+v", report.Reference)
	}
	if report.Delta != nil || report.Relative != nil {
		t.Fatalf("infeasible comparison must refuse deltas and ratios: delta=%+v relative=%+v",
			report.Delta, report.Relative)
	}
	if got := strings.Join(report.Reference.Feasibility.Reasons, "\n"); !strings.Contains(got, "demand exceeds capacity") {
		t.Fatalf("feasibility reason = %q, want capacity refusal", got)
	}
}

func TestValidationFailures(t *testing.T) {
	tests := []struct {
		name string
		edit func(*Comparison)
		want string
	}{
		{
			name: "quality envelope",
			edit: func(c *Comparison) { c.Workload.Quality.Target = "" },
			want: "quality target",
		},
		{
			name: "duplicate node",
			edit: func(c *Comparison) { c.Topology.Nodes = append(c.Topology.Nodes, c.Topology.Nodes[0]) },
			want: "duplicate node",
		},
		{
			name: "invalid link bandwidth",
			edit: func(c *Comparison) { c.Topology.Links[0].BandwidthBytesPerSecond = math.Inf(1) },
			want: "bandwidth",
		},
		{
			name: "unknown transfer link",
			edit: func(c *Comparison) { c.Candidate.Transfers[0].LinkID = "missing" },
			want: "unknown link",
		},
		{
			name: "negative component",
			edit: func(c *Comparison) { c.Reference.Synchronization.Latency = -time.Nanosecond },
			want: "durations must be non-negative",
		},
		{
			name: "unallocated link endpoint",
			edit: func(c *Comparison) { c.Candidate.Allocations = c.Candidate.Allocations[:1] },
			want: "endpoints must both be allocated",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := distributedComparison()
			c.Candidate.UsefulCompute = estimatedCost(50 * time.Millisecond)
			c.Reference.UsefulCompute = estimatedCost(100 * time.Millisecond)
			tt.edit(&c)
			_, err := Analyze(c)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Analyze() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func baseComparison() Comparison {
	return Comparison{
		Workload: Workload{
			ID:    "matched-inference",
			Units: 1,
			Unit:  "request",
			Quality: QualityEnvelope{
				ModelID:        "model-digest",
				Precision:      "bf16",
				SequenceLength: 4096,
				BatchSize:      1,
				Target:         "exact-output-envelope",
			},
		},
		Topology: Topology{
			Domains: []Domain{
				{ID: "unified-host"},
				{ID: "host-a"},
				{ID: "host-b"},
			},
			Nodes: []Node{
				{ID: "large", DomainID: "unified-host", Capacity: Capacity{MemoryBytes: 96 * gib, ComputeUnits: 16}},
				{ID: "small-a", DomainID: "host-a", Capacity: Capacity{MemoryBytes: 64 * gib, ComputeUnits: 8}},
				{ID: "small-b", DomainID: "host-b", Capacity: Capacity{MemoryBytes: 64 * gib, ComputeUnits: 8}},
			},
			Links: []Link{{
				ID:                      "a-to-b",
				FromNode:                "small-a",
				ToNode:                  "small-b",
				Latency:                 time.Millisecond,
				BandwidthBytesPerSecond: 1_000_000_000,
			}},
		},
	}
}

func distributedComparison() Comparison {
	c := baseComparison()
	c.Candidate = Placement{
		Name: "distributed",
		Allocations: []Allocation{
			{NodeID: "small-a", Demand: Capacity{MemoryBytes: 40 * gib, ComputeUnits: 4}},
			{NodeID: "small-b", Demand: Capacity{MemoryBytes: 40 * gib, ComputeUnits: 4}},
		},
		Transfers: []Transfer{{
			LinkID:     "a-to-b",
			Messages:   1,
			Bytes:      1,
			Provenance: ProvenanceEstimated,
		}},
	}
	c.Reference = Placement{
		Name:        "single-host",
		Allocations: []Allocation{{NodeID: "large", Demand: Capacity{MemoryBytes: 80 * gib, ComputeUnits: 8}}},
	}
	return c
}

func estimatedCost(d time.Duration) ComponentCost {
	return ComponentCost{
		Latency:    d,
		Cycle:      d,
		Provenance: ProvenanceEstimated,
	}
}

func ledgerCost(ledger []LedgerEntry, kind ComponentKind) ComponentCost {
	for _, entry := range ledger {
		if entry.Kind == kind {
			return entry.Cost
		}
	}
	return ComponentCost{}
}
