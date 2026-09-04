package harnessbench

import "time"

const (
	SpawningInvariantSchema = "fak.harnessbench.spawning_invariant.v1"
	ThunderingHerdSchema    = "fak.harnessbench.thundering_herd.v1"
	ThermalSheddingSchema   = "fak.harnessbench.thermal_shedding.v1"
)

// SpawningInvariantReceipt records the latency and resource drift over high-churn agent spawn cycles.
type SpawningInvariantReceipt struct {
	Schema          string    `json:"schema"`
	TotalCycles     int       `json:"total_cycles"`
	Epoch1LatencyNS int64     `json:"epoch_1_latency_ns"`
	EpochNLatencyNS int64     `json:"epoch_n_latency_ns"`
	LatencyRatio    float64   `json:"latency_ratio"`
	InitialAllocB   uint64    `json:"initial_alloc_b"`
	FinalAllocB     uint64    `json:"final_alloc_b"`
	AllocDriftBytes int64     `json:"alloc_drift_bytes"`
	Pass            bool      `json:"pass"`
	CompletedAt     time.Time `json:"completed_at"`
}

// ThunderingHerdReceipt records admission cap and backpressure under burst saturation.
type ThunderingHerdReceipt struct {
	Schema         string    `json:"schema"`
	BurstSize      int       `json:"burst_size"`
	PoolWorkers    int       `json:"pool_workers"`
	QueueCap       int       `json:"queue_cap"`
	EnqueuedCount  int       `json:"enqueued_count"`
	Rejected429    int       `json:"rejected_429"`
	ProcessedCount int       `json:"processed_count"`
	InitialThreads int       `json:"initial_threads"`
	FinalThreads   int       `json:"final_threads"`
	CleanRecovery  bool      `json:"clean_recovery"`
	Pass           bool      `json:"pass"`
	CompletedAt    time.Time `json:"completed_at"`
}

// ThermalSheddingReceipt records concurrency throttling and recovery under host thermal stress.
type ThermalSheddingReceipt struct {
	Schema          string    `json:"schema"`
	InitialK        int       `json:"initial_k"`
	ThrottledK      int       `json:"throttled_k"`
	RestoredK       int       `json:"restored_k"`
	P3TasksPaused   bool      `json:"p3_tasks_paused"`
	PeakObservedCPU float64   `json:"peak_observed_cpu"`
	Pass            bool      `json:"pass"`
	CompletedAt     time.Time `json:"completed_at"`
}
