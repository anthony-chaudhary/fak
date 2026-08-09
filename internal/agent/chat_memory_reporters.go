package agent

// KVMemoryStats is an optional planner-owned snapshot of local KV-cache residency.
// It is separate from Usage cache-read counters: those count work saved on a turn,
// while this reports resident KV memory pressure in the local process. Planners that
// proxy an upstream model do not implement it; the gateway emits no resident-KV series
// for them rather than publishing a fake zero.
type KVMemoryStats struct {
	Enabled            bool   // true when a reusable local KV cache is active
	Backend            string // radixkv, device backend name, or empty when unknown
	MemoryClass        string // kv_cache
	Scope              string // host/device
	DType              string // storage dtype for the local KV rows, currently f32 for HAL KV
	BytesPerToken      int64  // bytes per resident KV position under this model layout
	ResidentTokens     int    // true resident prefix positions, not the LRU edge-token budget
	ResidentBytes      int64
	CapacityKnown      bool
	CapacityFreeKnown  bool
	CapacityTotalBytes int64
	CapacityFreeBytes  int64
	HeadroomRatio      float64
	FitBudgetBytes     int64
	FitMarginBytes     int64
	BudgetTokens       int // configured LRU budget metric; 0 means unbounded or unavailable
	LRUTokens          int // Σ edge lengths, the budget metric radixkv enforces
	MaxDepthTokens     int
	Nodes              int
	Leaves             int
	Evictions          int
	PolicyEvictions    int
	Splits             int
}

// KVMemoryReporter is the optional interface a local planner implements when it
// can report resident KV-cache memory state.
type KVMemoryReporter interface {
	KVMemoryStats() KVMemoryStats
}

// RequestMemoryDemand is one row from the most recent local request memory plan.
// It mirrors compute.MemoryDemand without making gateway depend on compute types.
type RequestMemoryDemand struct {
	Class  string
	Scope  string
	DType  string
	Bytes  int64
	Detail string
}

type RequestMemoryCapacity struct {
	Scope      string
	TotalBytes int64
	FreeBytes  int64
	Known      bool
	FreeKnown  bool
}

// RequestMemoryStats is the optional planner-owned snapshot of the last in-kernel
// request admission plan. It reports successful plans too, so request memory pressure
// is visible before an OOM happens.
type RequestMemoryStats struct {
	Observed      bool
	Backend       string
	PromptTokens  int
	MaxNewTokens  int
	PlannedTokens int
	HeadroomRatio float64
	MemoryPlan    []RequestMemoryDemand
	Capacities    []RequestMemoryCapacity
}

// RequestMemoryReporter is implemented by local planners that can report their last
// request-time memory plan. Proxy planners do not implement it, so the gateway emits no
// local request-memory series for upstream providers.
type RequestMemoryReporter interface {
	RequestMemoryStats() RequestMemoryStats
}

// InKernelOOMRetryClassStats is one bounded-label row for decode retries that were
// attempted after a local in-kernel device allocation OOM.
type InKernelOOMRetryClassStats struct {
	Class           string
	Attempts        uint64
	Successes       uint64
	Failures        uint64
	LastFailedBytes uint64
	LastSite        string
}

// InKernelOOMRetryStats is the optional planner-owned snapshot of idle-pool trim retries
// after in-kernel device allocation OOMs. It is intentionally class-bucketed; allocator
// sites stay out of Prometheus labels and are exposed only in debug output.
type InKernelOOMRetryStats struct {
	Backend string
	Rows    []InKernelOOMRetryClassStats
}

// InKernelOOMRetryReporter is implemented by local planners that can report in-kernel
// OOM retry attempts. Proxy planners do not implement it.
type InKernelOOMRetryReporter interface {
	InKernelOOMRetryStats() InKernelOOMRetryStats
}

// InKernelMemoryPressureTrimClassStats is one bounded-label row for proactive
// memory-pressure trims before a served in-kernel device decode enters allocation-heavy
// work. "resolved" means a capacity-precheck refusal fit after the trim.
type InKernelMemoryPressureTrimClassStats struct {
	Scope           string
	Class           string
	Reason          string
	Attempts        uint64
	Trimmed         uint64
	NoHooks         uint64
	Resolved        uint64
	LastWantBytes   uint64
	LastBudgetBytes uint64
	LastMarginBytes int64
}

// InKernelMemoryPressureTrimStats reports proactive idle-pool trims triggered by
// known request-memory pressure. It is separate from OOM retry stats: these happen
// before decode allocation, not after a recovered DeviceAllocError.
type InKernelMemoryPressureTrimStats struct {
	Backend string
	Rows    []InKernelMemoryPressureTrimClassStats
}

// InKernelMemoryPressureTrimReporter is implemented by local planners that can
// report proactive memory-pressure trims. Proxy planners do not implement it.
type InKernelMemoryPressureTrimReporter interface {
	InKernelMemoryPressureTrimStats() InKernelMemoryPressureTrimStats
}

// MoEResidencyReporter is implemented by local planners that can report activated-expert
// residency across the requests they served (R6, #5617). Proxy planners do not implement it, so
// the gateway emits no local MoE-residency series for upstream providers.
//
// Unlike the reporters above it has a second silence: a local planner whose operator declared no
// expert budget builds no ring, so its ledger stays at Requests==0 forever. Surfaces must render
// that as "not engaged" — an absent block — rather than as a ring reporting zero hits, which is
// what a row of zeros reads like on a dashboard.
type MoEResidencyReporter interface {
	MoEResidencyStats() MoEResidencyLedger
}
