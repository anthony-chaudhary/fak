package modelperfobs

import (
	"reflect"
	"testing"
)

func TestTopologyModesDoNotNaivelyAggregateMemory(t *testing.T) {
	for _, tc := range []struct {
		name       string
		mode       TopologyMode
		wantCopies uint64
		wantMin    float64
		wantFits   bool
	}{
		{"replicas", IndependentReplicas, 4, 110, false},
		{"shared prefix", SharedPrefixRouting, 4, 110, false},
		{"sharded", ModelParallel, 1, 35, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := topologyFixture()
			in.Mode = tc.mode
			got, err := EstimateTopologyEconomics(in)
			if err != nil {
				t.Fatal(err)
			}
			if got.WeightCopies != tc.wantCopies {
				t.Fatalf("weight copies = %d, want %d", got.WeightCopies, tc.wantCopies)
			}
			if got.PerNodeMemoryBytes.Min != tc.wantMin {
				t.Fatalf("per-node memory = %g, want %g", got.PerNodeMemoryBytes.Min, tc.wantMin)
			}
			if got.FitsPerNode != tc.wantFits {
				t.Fatalf("per-node fit = %v, want %v; aggregate RAM must not decide fit", got.FitsPerNode, tc.wantFits)
			}
		})
	}
}

func TestReplicatedPeakKVConcentrationPreventsFalseAggregateFit(t *testing.T) {
	for _, mode := range []TopologyMode{IndependentReplicas, SharedPrefixRouting} {
		t.Run(string(mode), func(t *testing.T) {
			in := topologyFixture()
			in.Mode = mode
			in.PerNodeMemoryBytes = 50
			in.WeightsBytes = ClosedRange{Min: 10, Max: 10}
			in.ResidentKVBytes = ClosedRange{Min: 120, Max: 120}

			got, err := EstimateTopologyEconomics(in)
			if err != nil {
				t.Fatal(err)
			}
			if got.AggregatePhysicalMemory.Max > in.PerNodeMemoryBytes*float64(in.Nodes) {
				t.Fatalf("fixture aggregate memory %g exceeds total RAM %g", got.AggregatePhysicalMemory.Max, in.PerNodeMemoryBytes*float64(in.Nodes))
			}
			if got.PerNodeMemoryBytes.Max != 130 {
				t.Fatalf("peak per-node memory = %g, want 130", got.PerNodeMemoryBytes.Max)
			}
			if got.FitsPerNode {
				t.Fatal("reported fit from aggregate RAM despite peak-node KV concentration")
			}
		})
	}
}

func TestSharedPrefixRoutingDoesNotFabricateSavings(t *testing.T) {
	independent := topologyFixture()
	shared := independent
	shared.Mode = SharedPrefixRouting

	independentGot, err := EstimateTopologyEconomics(independent)
	if err != nil {
		t.Fatal(err)
	}
	sharedGot, err := EstimateTopologyEconomics(shared)
	if err != nil {
		t.Fatal(err)
	}
	independentGot.Mode = sharedGot.Mode
	if !reflect.DeepEqual(sharedGot, independentGot) {
		t.Fatalf("shared-prefix routing changed economics without different inputs:\nshared:      %+v\nindependent: %+v", sharedGot, independentGot)
	}
}

func TestNetworkSerializationUsesLinkRateNotSummedNodeBandwidth(t *testing.T) {
	for _, tc := range []struct {
		nodes uint64
		gbps  uint64
		want  float64
	}{
		{1, 10, 8}, {2, 25, 3.2}, {4, 100, .8},
	} {
		in := topologyFixture()
		in.Nodes = tc.nodes
		in.EthernetGbps = tc.gbps
		got, err := EstimateTopologyEconomics(in)
		if err != nil {
			t.Fatal(err)
		}
		if got.NetworkSerialization.Min != tc.want || got.NetworkSerialization.Max != tc.want {
			t.Fatalf("nodes=%d gbe=%d serialization=%v, want %g (must not divide by nodes)", tc.nodes, tc.gbps, got.NetworkSerialization, tc.want)
		}
	}
}

func TestFailedOrUnknownQualityIsNotQualified(t *testing.T) {
	for _, quality := range []QualityQualification{QualityFailed, QualityUnknown} {
		t.Run(string(quality), func(t *testing.T) {
			in := topologyFixture()
			in.Quality = quality
			got, err := EstimateTopologyEconomics(in)
			if err != nil {
				t.Fatal(err)
			}
			for _, comparison := range got.Comparisons {
				if comparison.TopologyCostPerJob.Bounds != nil || comparison.TopologyTimePerJob.Bounds != nil {
					t.Fatalf("%s quality produced qualified economics: %+v", quality, comparison)
				}
				if comparison.TopologyCheaper != nil || comparison.TopologyFaster != nil {
					t.Fatalf("%s quality produced break-even verdict: %+v", quality, comparison)
				}
			}
		})
	}
}

func TestQualifiedBreakEvenIntervalVerdicts(t *testing.T) {
	point := func(v float64) ClosedRange { return ClosedRange{Min: v, Max: v} }
	for _, tc := range []struct {
		name            string
		alternativeRate ClosedRange
		want            *bool
	}{
		{name: "strictly higher alternative", alternativeRate: point(3), want: boolPointer(true)},
		{name: "strictly lower alternative", alternativeRate: point(1), want: boolPointer(false)},
		{name: "overlapping alternative", alternativeRate: ClosedRange{Min: 1, Max: 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := topologyFixture()
			in.SingleNodeJobSeconds = point(8)
			in.KVStateMovementBytes = point(0)
			in.WeightTransferBytes = point(0)
			in.Synchronization = point(0)
			in.SchedulerImbalance = point(0)
			in.SetupRecovery = point(0)
			in.ReplayCompaction = point(0)
			in.IdleCapacity = point(0)
			in.ClusterCostPerSecond = point(1)

			alternativeTotal := ClosedRange{
				Min: tc.alternativeRate.Min * 8,
				Max: tc.alternativeRate.Max * 8,
			}
			in.LargerHost.CompletedJobs = 8
			in.LargerHost.TotalTime = alternativeTotal
			in.LargerHost.TotalCost = alternativeTotal
			in.APIBurst.CompletedJobs = 8
			in.APIBurst.TotalTime = alternativeTotal
			in.APIBurst.TotalCost = alternativeTotal

			got, err := EstimateTopologyEconomics(in)
			if err != nil {
				t.Fatal(err)
			}
			for _, comparison := range got.Comparisons {
				if !reflect.DeepEqual(comparison.TopologyCheaper, tc.want) {
					t.Errorf("%s TopologyCheaper = %v, want %v", comparison.Alternative, comparison.TopologyCheaper, tc.want)
				}
				if !reflect.DeepEqual(comparison.TopologyFaster, tc.want) {
					t.Errorf("%s TopologyFaster = %v, want %v", comparison.Alternative, comparison.TopologyFaster, tc.want)
				}
			}
		})
	}
}

func boolPointer(v bool) *bool {
	return &v
}

func TestQualifiedCompletedJobBreakEven(t *testing.T) {
	in := topologyFixture()
	got, err := EstimateTopologyEconomics(in)
	if err != nil {
		t.Fatal(err)
	}
	for _, comparison := range got.Comparisons {
		if comparison.TopologyCostPerJob.Bounds == nil || comparison.AlternativeCostPerJob.Bounds == nil {
			t.Fatalf("qualified jobs lack per-job economics: %+v", comparison)
		}
	}
}

func topologyFixture() TopologyInput {
	point := func(v float64) ClosedRange { return ClosedRange{Min: v, Max: v} }
	return TopologyInput{
		Mode: IndependentReplicas, Nodes: 4, EthernetGbps: 10,
		PerNodeMemoryBytes: 50, WeightsBytes: point(100), ResidentKVBytes: point(40), Jobs: 8,
		SingleNodeJobSeconds: point(10), ShardSpeedup: point(2), NetworkEfficiency: point(1),
		KVStateMovementBytes: point(10e9), WeightTransferBytes: point(0),
		Synchronization: point(1), SchedulerImbalance: point(.1), SetupRecovery: point(2),
		ReplayCompaction: point(1), IdleCapacity: point(.2), ClusterCostPerSecond: point(1),
		Quality:    QualityQualified,
		LargerHost: CompletedJobAlternative{Name: "larger-host", CompletedJobs: 8, TotalTime: point(100), TotalCost: point(100), Quality: QualityQualified},
		APIBurst:   CompletedJobAlternative{Name: "api-burst", CompletedJobs: 8, TotalTime: point(80), TotalCost: point(160), Quality: QualityQualified},
	}
}
