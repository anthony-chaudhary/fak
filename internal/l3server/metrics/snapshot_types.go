package metrics

import "sync/atomic"

type StartupState struct {
	ShardsReady          atomic.Int32
	ShardsTotal          atomic.Int32
	MemReservedBytes     atomic.Int64
	MemTotalBytes        atomic.Int64
	Ready                atomic.Bool
	AcceptingConnections atomic.Bool
	Phase                atomic.Int32
	PhaseLabel           atomic.Value
	AllocStartNanos      atomic.Int64
	AvgShardNanos        atomic.Int64
}

type SlabClassSnapshot struct {
	Size            uint64
	TotalSlots      uint64
	UsedSlots       uint64
	AllocCount      int64
	SlotUtilization float64
}

type SlabDetectionSnapshot struct {
	Detected            bool
	DetectedPageBytes   uint64
	SlotUtilization     float64
	ConfiguredPageBytes uint64
	ModelTotalSlots     uint64
	ModelClassSize      uint64
}

type SlabMetricsProvider func() (classes []SlabClassSnapshot, detection SlabDetectionSnapshot)

type CacheStateProvider func() (allocBytes int64, entries int64)

type VacuumSnapshot struct {
	RebalancesTotal    int64
	LastRebalanceEpoch int64
	PendingShards      int
	PressureEvals      int64
	PressureRebuilds   int64
	MaxDrift           float64
	RebalanceFailures  int64
}

type PressureClassSnapshot struct {
	Size          uint64
	Pressure      float64
	EvictionRate  float64
	AllocFailRate float64
	CurrentWeight float64
}

type PressureMetricsProvider func() []PressureClassSnapshot

type PollerSnapshot struct {
	Device                string
	ActiveConns           int32
	Completions           int64
	DispatchEnqueued      int64
	DispatchDropped       int64
	SendChDropped         int64
	DispatchWorkers       int
	DispatchQueueDepth    int
	DispatchQueueCap      int
	CleanupQueueDepth     int
	CleanupQueueCap       int
	DispatchSaturationPct float64
}

type PollerMetricsProvider func() []PollerSnapshot

type RDMAReadSnapshot struct {
	Issued             int64
	Confirmed          int64
	Failed             int64
	ForcedInline       int64
	LeaseDrops         int64
	MgetMigrationSkips int64
}

type RDMAReadMetricsProvider func() RDMAReadSnapshot

type SystemHealthSnapshot struct {
	MemAvailableBytes    int64
	MemPressureActive    bool
	MemPressureLevel     int32
	MemPSISomeBP         int64
	MemPSIFullBP         int64
	ProcessRSSBytes      int64
	ProcessHugetlbBytes  int64
	ProcessTotalRSSBytes int64
	FDCount              int64
	FDLimit              int64
	NICPortActive        map[string]bool
	NICHWErrors          map[string]map[string]int64

	CPUUserSeconds         float64
	CPUSystemSeconds       float64
	Goroutines             int
	Threads                int
	VoluntaryCtxSwitches   int64
	InvoluntaryCtxSwitches int64
}

type SystemHealthProvider func() SystemHealthSnapshot

type VacuumMetricsProvider func() VacuumSnapshot

type OpLatencySnapshot struct {
	ShardID                                        int
	AllP50Us, AllP99Us                             int64
	GetP50Us, GetP99Us                             int64
	SetP50Us, SetP99Us                             int64
	ExistsP50Us, ExistsP99Us                       int64
	QueueWaitP50Us, QueueWaitP99Us                 int64
	AllocDurP50Us, AllocDurP99Us                   int64
	QueueDepth, QueueCap                           int
}

type ShardPressureSnapshot struct {
	ShardID    int
	ClassSize  uint64
	AllocOps   int64
	AllocFails int64
	Evictions  int64
}

type ShardPressureProvider func() []ShardPressureSnapshot

type OpLatencyProvider func() []OpLatencySnapshot
