package modelperfobs

import "fmt"

const TopologyEconomicsSchema = "fak-topology-economics/1"

type TopologyMode string

const (
	IndependentReplicas TopologyMode = "independent-agent-replication"
	SharedPrefixRouting TopologyMode = "shared-prefix-routing"
	ModelParallel       TopologyMode = "model-parallel-sharded"
)

type QualityQualification string

const (
	QualityQualified QualityQualification = "qualified"
	QualityFailed    QualityQualification = "failed"
	QualityUnknown   QualityQualification = "unknown"
)

// QualifiedMetric is unknown unless the corresponding jobs passed the caller's
// task-specific quality gate. Throughput alone never establishes useful work.
type QualifiedMetric struct {
	Bounds  *ClosedRange `json:"bounds,omitempty"`
	Unknown string       `json:"unknown,omitempty"`
}

type CompletedJobAlternative struct {
	Name          string               `json:"name"`
	CompletedJobs uint64               `json:"completed_jobs"`
	TotalTime     ClosedRange          `json:"total_time_seconds"`
	TotalCost     ClosedRange          `json:"total_cost"`
	Quality       QualityQualification `json:"quality"`
}

type TopologyInput struct {
	Mode                 TopologyMode            `json:"mode"`
	Nodes                uint64                  `json:"nodes"`
	EthernetGbps         uint64                  `json:"ethernet_gbps"`
	PerNodeMemoryBytes   float64                 `json:"per_node_memory_bytes"`
	WeightsBytes         ClosedRange             `json:"weights_bytes"`
	ResidentKVBytes      ClosedRange             `json:"resident_kv_bytes"`
	Jobs                 uint64                  `json:"jobs"`
	SingleNodeJobSeconds ClosedRange             `json:"single_node_job_seconds"`
	ShardSpeedup         ClosedRange             `json:"shard_speedup"`
	NetworkEfficiency    ClosedRange             `json:"network_efficiency_fraction"`
	KVStateMovementBytes ClosedRange             `json:"kv_state_movement_bytes"`
	WeightTransferBytes  ClosedRange             `json:"weight_transfer_bytes"`
	Synchronization      ClosedRange             `json:"synchronization_seconds"`
	SchedulerImbalance   ClosedRange             `json:"scheduler_imbalance_fraction"`
	SetupRecovery        ClosedRange             `json:"setup_recovery_seconds"`
	ReplayCompaction     ClosedRange             `json:"replay_compaction_seconds"`
	IdleCapacity         ClosedRange             `json:"idle_capacity_fraction"`
	ClusterCostPerSecond ClosedRange             `json:"cluster_cost_per_second"`
	Quality              QualityQualification    `json:"quality"`
	LargerHost           CompletedJobAlternative `json:"larger_host"`
	APIBurst             CompletedJobAlternative `json:"api_burst"`
}

type BreakEvenComparison struct {
	Alternative           string          `json:"alternative"`
	TopologyCostPerJob    QualifiedMetric `json:"topology_cost_per_qualified_job"`
	AlternativeCostPerJob QualifiedMetric `json:"alternative_cost_per_qualified_job"`
	TopologyTimePerJob    QualifiedMetric `json:"topology_time_per_qualified_job_seconds"`
	AlternativeTimePerJob QualifiedMetric `json:"alternative_time_per_qualified_job_seconds"`
	TopologyCheaper       *bool           `json:"topology_cheaper,omitempty"`
	TopologyFaster        *bool           `json:"topology_faster,omitempty"`
}

type TopologyEconomics struct {
	Schema                  string                `json:"schema"`
	Mode                    TopologyMode          `json:"mode"`
	WeightCopies            uint64                `json:"weight_copies"`
	PerNodeMemoryBytes      ClosedRange           `json:"per_node_memory_bytes"`
	AggregatePhysicalMemory ClosedRange           `json:"aggregate_physical_memory_bytes"`
	FitsPerNode             bool                  `json:"fits_per_node"`
	FitUncertain            bool                  `json:"fit_uncertain"`
	NetworkBytes            ClosedRange           `json:"network_bytes"`
	NetworkSerialization    ClosedRange           `json:"network_serialization_seconds"`
	ProductiveSeconds       ClosedRange           `json:"productive_seconds"`
	TotalTime               ClosedRange           `json:"total_time_seconds"`
	TotalCost               ClosedRange           `json:"total_cost"`
	Quality                 QualityQualification  `json:"quality"`
	Comparisons             []BreakEvenComparison `json:"comparisons"`
	Limitations             []string              `json:"limitations"`
}

// EstimateTopologyEconomics composes analytical inputs; it does not claim
// measured scaling. Ethernet is a serialization constraint, never multiplied
// by node count as though every byte could use every link simultaneously.
func EstimateTopologyEconomics(in TopologyInput) (TopologyEconomics, error) {
	if err := validateTopologyInput(in); err != nil {
		return TopologyEconomics{}, err
	}

	copies := in.Nodes
	// Independent replicas and shared-prefix routing both keep one weight copy
	// per node. Neither mode implies balanced KV placement: the lower bound is a
	// perfect partition, while the upper bound allows all resident KV on the
	// peak node. Shared-prefix benefit must be supplied as lower state or
	// movement bounds, never as an automatic cache-saving multiplier.
	memory := ClosedRange{
		Min: in.WeightsBytes.Min + in.ResidentKVBytes.Min/float64(in.Nodes),
		Max: in.WeightsBytes.Max + in.ResidentKVBytes.Max,
	}
	parallelism := float64(minU64(in.Nodes, in.Jobs))
	if in.Mode == ModelParallel {
		copies = 1
		// Selecting ModelParallel explicitly asserts that weights and KV state
		// are sharded across the nodes.
		memory = scaleRange(addRange(in.WeightsBytes, in.ResidentKVBytes), 1/float64(in.Nodes))
		parallelism = 1
	}
	aggregateMemory := ClosedRange{
		Min: in.WeightsBytes.Min*float64(copies) + in.ResidentKVBytes.Min,
		Max: in.WeightsBytes.Max*float64(copies) + in.ResidentKVBytes.Max,
	}

	work := scaleRange(in.SingleNodeJobSeconds, float64(in.Jobs)/parallelism)
	if in.Mode == ModelParallel {
		work = divPositiveRange(scaleRange(in.SingleNodeJobSeconds, float64(in.Jobs)), in.ShardSpeedup)
	}
	// Lost capacity and imbalance stretch useful work multiplicatively; setup,
	// recovery, synchronization, and replay are elapsed-time additions.
	capacity := ClosedRange{Min: 1 - in.IdleCapacity.Max, Max: 1 - in.IdleCapacity.Min}
	work = divPositiveRange(work, capacity)
	work = mulRange(work, addScalar(in.SchedulerImbalance, 1))

	networkBytes := addRange(in.KVStateMovementBytes, in.WeightTransferBytes)
	wireBytesPerSecond := scaleRange(in.NetworkEfficiency, float64(in.EthernetGbps)*1e9/8)
	serialization := divPositiveRange(networkBytes, wireBytesPerSecond)
	totalTime := addRange(work, serialization)
	totalTime = addRange(totalTime, in.Synchronization)
	totalTime = addRange(totalTime, in.SetupRecovery)
	totalTime = addRange(totalTime, in.ReplayCompaction)
	totalCost := mulRange(totalTime, in.ClusterCostPerSecond)

	result := TopologyEconomics{
		Schema: TopologyEconomicsSchema, Mode: in.Mode, WeightCopies: copies,
		PerNodeMemoryBytes: memory, AggregatePhysicalMemory: aggregateMemory,
		FitsPerNode:  memory.Max <= in.PerNodeMemoryBytes,
		FitUncertain: memory.Min <= in.PerNodeMemoryBytes && memory.Max > in.PerNodeMemoryBytes,
		NetworkBytes: networkBytes, NetworkSerialization: serialization,
		ProductiveSeconds: work, TotalTime: totalTime, TotalCost: totalCost, Quality: in.Quality,
		Limitations: []string{
			"Inputs are analytical bounds, not benchmark measurements.",
			"A single bottleneck-link serialization bound does not model fabric topology, collectives, or overlap.",
			"Quality must be established by an external task-specific gate; unknown or failed quality has no qualified-job economics.",
			"Execution remains fak-native; external engines may only supply separately identified parity evidence.",
		},
	}
	result.Comparisons = []BreakEvenComparison{
		compareCompletedJobs(in.Jobs, totalTime, totalCost, in.Quality, in.LargerHost),
		compareCompletedJobs(in.Jobs, totalTime, totalCost, in.Quality, in.APIBurst),
	}
	return result, nil
}

func compareCompletedJobs(jobs uint64, time, cost ClosedRange, quality QualityQualification, alt CompletedJobAlternative) BreakEvenComparison {
	out := BreakEvenComparison{Alternative: alt.Name}
	out.TopologyCostPerJob = qualifiedPerJob(cost, jobs, quality)
	out.TopologyTimePerJob = qualifiedPerJob(time, jobs, quality)
	out.AlternativeCostPerJob = qualifiedPerJob(alt.TotalCost, alt.CompletedJobs, alt.Quality)
	out.AlternativeTimePerJob = qualifiedPerJob(alt.TotalTime, alt.CompletedJobs, alt.Quality)
	if out.TopologyCostPerJob.Bounds != nil && out.AlternativeCostPerJob.Bounds != nil {
		out.TopologyCheaper = strictIntervalVerdict(*out.TopologyCostPerJob.Bounds, *out.AlternativeCostPerJob.Bounds)
	}
	if out.TopologyTimePerJob.Bounds != nil && out.AlternativeTimePerJob.Bounds != nil {
		out.TopologyFaster = strictIntervalVerdict(*out.TopologyTimePerJob.Bounds, *out.AlternativeTimePerJob.Bounds)
	}
	return out
}

func strictIntervalVerdict(candidate, alternative ClosedRange) *bool {
	if candidate.Max < alternative.Min {
		verdict := true
		return &verdict
	}
	if candidate.Min > alternative.Max {
		verdict := false
		return &verdict
	}
	return nil
}

func qualifiedPerJob(total ClosedRange, jobs uint64, quality QualityQualification) QualifiedMetric {
	if quality != QualityQualified {
		return QualifiedMetric{Unknown: "quality is " + string(quality)}
	}
	if jobs == 0 {
		return QualifiedMetric{Unknown: "completed job count is unknown or zero"}
	}
	v := scaleRange(total, 1/float64(jobs))
	return QualifiedMetric{Bounds: &v}
}

func validateTopologyInput(in TopologyInput) error {
	if in.Nodes != 1 && in.Nodes != 2 && in.Nodes != 4 {
		return fmt.Errorf("nodes must be 1, 2, or 4")
	}
	if in.EthernetGbps != 10 && in.EthernetGbps != 25 && in.EthernetGbps != 100 {
		return fmt.Errorf("ethernet_gbps must be 10, 25, or 100")
	}
	if in.Mode != IndependentReplicas && in.Mode != SharedPrefixRouting && in.Mode != ModelParallel {
		return fmt.Errorf("unsupported topology mode %q", in.Mode)
	}
	if !finite(in.PerNodeMemoryBytes) || in.PerNodeMemoryBytes <= 0 || in.Jobs == 0 {
		return fmt.Errorf("per_node_memory_bytes and jobs must be positive")
	}
	for _, item := range []struct {
		name     string
		r        ClosedRange
		positive bool
		fraction bool
	}{
		{"weights_bytes", in.WeightsBytes, true, false}, {"resident_kv_bytes", in.ResidentKVBytes, false, false},
		{"single_node_job_seconds", in.SingleNodeJobSeconds, true, false}, {"shard_speedup", in.ShardSpeedup, true, false},
		{"network_efficiency_fraction", in.NetworkEfficiency, true, true}, {"kv_state_movement_bytes", in.KVStateMovementBytes, false, false},
		{"weight_transfer_bytes", in.WeightTransferBytes, false, false}, {"synchronization_seconds", in.Synchronization, false, false},
		{"scheduler_imbalance_fraction", in.SchedulerImbalance, false, true}, {"setup_recovery_seconds", in.SetupRecovery, false, false},
		{"replay_compaction_seconds", in.ReplayCompaction, false, false}, {"idle_capacity_fraction", in.IdleCapacity, false, true},
		{"cluster_cost_per_second", in.ClusterCostPerSecond, false, false},
	} {
		if err := validateRange(item.name, item.r, item.positive, item.fraction); err != nil {
			return err
		}
	}
	if in.IdleCapacity.Max >= 1 {
		return fmt.Errorf("idle_capacity_fraction must be less than 1")
	}
	if !validQuality(in.Quality) {
		return fmt.Errorf("invalid quality %q", in.Quality)
	}
	for _, alt := range []CompletedJobAlternative{in.LargerHost, in.APIBurst} {
		if !validQuality(alt.Quality) {
			return fmt.Errorf("invalid quality for %q", alt.Name)
		}
		if err := validateRange("alternative total_time_seconds", alt.TotalTime, false, false); err != nil {
			return err
		}
		if err := validateRange("alternative total_cost", alt.TotalCost, false, false); err != nil {
			return err
		}
	}
	return nil
}

func validQuality(q QualityQualification) bool {
	return q == QualityQualified || q == QualityFailed || q == QualityUnknown
}
func minU64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}
