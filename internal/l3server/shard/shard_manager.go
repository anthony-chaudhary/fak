package shard

import (
	"fmt"
	"log"
	"math"
	"math/bits"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// VacuumConfig controls the auto-rebalancer â€” the automatic trigger policy for slab rebalancing.
// Config keys use the `vacuum_` prefix for historical reasons.
type VacuumConfig struct {
	Enabled              bool
	IntervalSeconds      int     // evaluation interval (default 30)
	CooldownSeconds      int     // deprecated: submissions are now parallel; kept for TOML compatibility
	UtilizationThreshold float64 // trigger when dominant class > this (default 0.50)
	MinAgeSeconds        int     // don't rebalance shards younger than this (default 120)

	// Pressure-driven rebalancing (v0.13.0)
	PressureRebalancing bool    // enable pressure-based weight adjustment (default true)
	DampingFactor       float64 // 0.0-1.0: how fast weights adjust per cycle (default 0.3)
	DriftThreshold      float64 // min relative change to trigger rebuild (default 0.15)
	MinClassWeight      float64 // floor weight for any class (default 0.5)
	EvictionRateNorm    float64 // evictions/sec that maps to pressure=1.0 (default 10.0)

	// Proactive watermark (v0.24.0)
	WatermarkThreshold float64 // per-class util that triggers early-warning evaluation (default 0.50)
}

// VacuumStats exposes vacuum state for Prometheus metrics.
type VacuumStats struct {
	RebalancesTotal    int64
	LastRebalanceEpoch int64 // unix seconds, 0 = never
	PendingShards      int   // shards currently needing rebalance
	PressureEvals      int64 // total pressure evaluations performed
	PressureRebuilds   int64 // rebuilds triggered by pressure
	MaxDrift           float64 // current max relative weight drift across all shards
	RebalanceFailures  int64   // total rebalance failures (dispatch timeouts, etc.)
}

// rebalanceFailState tracks consecutive failures and backoff for a single shard.
type rebalanceFailState struct {
	consecutiveFails int
	nextEligible     time.Time
}

// Manager distributes operations across shards.
type Manager struct {
	shards    []*Shard
	numShards uint64
	mask      uint64 // numShards - 1 (power of 2)

	// Per-shard readiness tracking for early connection acceptance
	shardReady []atomic.Bool // len == numShards; set true after each shard is allocated+started
	allReady   atomic.Bool   // set when all shards allocated + vacuum started

	// Client-reported model page size hint (triggers slab rebuild)
	modelPageHint atomic.Uint64

	// Auto-rebalancer state
	vacuumCfg           VacuumConfig
	vacuumQuit          chan struct{}
	vacuumDone          chan struct{}
	vacuumRebalances    atomic.Int64
	vacuumLastRebalance atomic.Int64 // unix seconds
	startedAt           time.Time

	// Pressure rebalancing state
	pressureEvals    atomic.Int64 // total pressure evaluations
	pressureRebuilds atomic.Int64 // rebuilds triggered by pressure

	// Per-shard rebalance backoff (only accessed from vacuum goroutine â€” no lock needed)
	rebalanceBackoff       map[int]rebalanceFailState
	vacuumRebalanceFailures atomic.Int64

	vacuumTickCount     atomic.Uint64
	vacuumPendingShards atomic.Int32 // cached pending count from last vacuum tick

	migrateSem          chan struct{} // limits concurrent shard migrations
	verboseShardLogging bool

	// Aggregate warmup tracking (for summary logging when verbose=false)
	warmupCompleted atomic.Int32
}

// ManagerConfig holds config for the shard manager.
type ManagerConfig struct {
	NumShards         int
	MaxMemoryGB       int
	EvictionPolicy    string
	AllocatorMode     string // "slab" (default) or "offset"
	ModelPageBytes    uint64
	MaxLeaseDurMs     int64
	UseHugePages      bool // kept for convenience checks
	HugePageSizeKB    int  // 0 = disabled, 2048 = 2MB, 1048576 = 1GB
	IndexCapacity     uint64
	DispatchTimeoutMs int64
	WarmupOps           int
	AutoTuneSlabs       bool
	SlabDistribution    string             // "auto", "model", "uniform", "dedicated"
	InitialClassWeights map[uint64]float64 // static class weight overrides from config; nil = use defaults
	VerboseShardLogging      bool
	Vacuum                   VacuumConfig
	OnShardReady             func(id, total int, memBytes uint64) // called after each shard is allocated
	MaxConcurrentMigrations  int // max parallel shard migrations (default 2)
	MigrateBatchSize         int // entries per ZeroLatencyBalance batch (default 512)
	MigrateDrainBudget       int // max ops drained per migration cycle (default 64)

	// OOM graceful handling
	OOMRejectAfterFails int // consecutive alloc failures before fast-rejecting SETs; 0 = disabled

	// SystemOOMFlag is the syshealth Monitor's MemPressureActive flag.
	// Passed to each shard so checkOOM() can proactively reject SETs
	// when system memory is critically low.
	SystemOOMFlag *atomic.Bool

	// SystemPressureLevel is the syshealth Monitor's MemPressureLevel.
	// Provides tiered backpressure (0-4) for gradual write throttling.
	SystemPressureLevel *atomic.Int32

	// NUMA assignment â€” per-shard NUMA node IDs (len == NumShards); nil = no pinning
	NUMAAssignment []int

	// PreAllocCheck is called before each shard allocation with (shardIdx, totalShards, perShardBytes).
	// Return a non-nil error to abort shard creation (prevents OOM-killer during allocation).
	PreAllocCheck func(shardIdx, totalShards int, perShardBytes uint64) error
}

// NewManager creates a shard manager with N shards.
func NewManager(cfg ManagerConfig) (*Manager, error) {
	numShards := cfg.NumShards
	if numShards <= 0 {
		numShards = 16 // match cmd/server default; NumCPU() is unpredictable on high-core machines
	}
	// Round up to power of 2 for fast masking
	numShards = int(nextPow2Mgr(uint64(numShards)))

	totalMem := uint64(cfg.MaxMemoryGB) * 1024 * 1024 * 1024

	numShards = capShardsForMemory(numShards, totalMem)

	perShardMem := totalMem / uint64(numShards)

	idxCap := cfg.IndexCapacity
	if idxCap == 0 {
		idxCap = 131072
	}
	perShardIdxCap := idxCap // per-shard, as documented in TOML
	if perShardIdxCap < 1024 {
		perShardIdxCap = 1024
	}

	maxConcMig := cfg.MaxConcurrentMigrations
	if maxConcMig <= 0 {
		maxConcMig = 2
	}
	migrateSem := make(chan struct{}, maxConcMig)

	migrateBatch := cfg.MigrateBatchSize
	if migrateBatch <= 0 {
		migrateBatch = 512
	}

	m := &Manager{
		shards:              make([]*Shard, numShards),
		numShards:           uint64(numShards),
		mask:                uint64(numShards) - 1,
		shardReady:          make([]atomic.Bool, numShards),
		vacuumCfg:           cfg.Vacuum,
		startedAt:           time.Now(),
		migrateSem:          migrateSem,
		verboseShardLogging: cfg.VerboseShardLogging,
		rebalanceBackoff:    make(map[int]rebalanceFailState),
	}

	// Group shards by NUMA node so we can allocate across nodes in parallel.
	// Each NUMA node has its own hugepage pool and buddy allocator, so
	// parallel mmap+MAP_POPULATE across nodes actually runs concurrently in
	// the kernel. Within a node, shards are still sequential (same mem policy
	// + serialized by the node's allocator).
	type shardSlot struct {
		idx      int
		numaNode int
	}
	nodeGroups := make(map[int][]shardSlot) // numaNode â†’ shard indices
	for i := 0; i < numShards; i++ {
		node := -1
		if cfg.NUMAAssignment != nil && i < len(cfg.NUMAAssignment) {
			node = cfg.NUMAAssignment[i]
		}
		nodeGroups[node] = append(nodeGroups[node], shardSlot{idx: i, numaNode: node})
	}

	// If there are multiple NUMA node groups, allocate in parallel.
	// A shared atomic counter drives the OnShardReady progress callback.
	var shardsCompleted atomic.Int32
	var allocErr error
	var errOnce sync.Once

	buildShard := func(slot shardSlot) error {
		i := slot.idx
		if cfg.PreAllocCheck != nil {
			if err := cfg.PreAllocCheck(i, numShards, perShardMem); err != nil {
				return fmt.Errorf("pre-alloc check for shard %d: %w", i, err)
			}
		}

		staggeredWarmup := cfg.WarmupOps
		if cfg.WarmupOps > 0 && numShards > 1 {
			staggeredWarmup = cfg.WarmupOps + (cfg.WarmupOps*i)/numShards
		}

		shardCfg := ShardConfig{
			ID:                i,
			IndexCapacity:     perShardIdxCap,
			MaxMemoryBytes:    perShardMem,
			EvictionPolicy:    cfg.EvictionPolicy,
			AllocatorMode:     cfg.AllocatorMode,
			ModelPageBytes:    cfg.ModelPageBytes,
			MaxLeaseDurMs:     cfg.MaxLeaseDurMs,
			UseHugePages:      cfg.UseHugePages,
			HugePageSizeKB:    cfg.HugePageSizeKB,
			DispatchTimeoutMs:   cfg.DispatchTimeoutMs,
			WarmupOps:           staggeredWarmup,
			AutoTuneSlabs:       cfg.AutoTuneSlabs,
			SlabDistribution:    cfg.SlabDistribution,
			InitialClassWeights: cfg.InitialClassWeights,
			VerboseShardLogging: cfg.VerboseShardLogging,
			OnWarmupComplete:    m.NotifyWarmupComplete,
			MigrateBatchSize:    migrateBatch,
			MigrateDrainBudget:  cfg.MigrateDrainBudget,
			MigrateSem:          migrateSem,
			NUMANode:              slot.numaNode,
			OOMRejectAfterFails:   cfg.OOMRejectAfterFails,
			SystemOOMFlag:         cfg.SystemOOMFlag,
			SystemPressureLevel:   cfg.SystemPressureLevel,
		}
		s, err := New(shardCfg)
		if err != nil {
			return fmt.Errorf("failed to create shard %d: %w", i, err)
		}
		m.shards[i] = s
		m.shardReady[i].Store(true)

		if cfg.OnShardReady != nil {
			completed := int(shardsCompleted.Add(1))
			cfg.OnShardReady(completed, numShards, perShardMem)
		}
		return nil
	}

	if len(nodeGroups) > 1 {
		// Multi-NUMA: parallel across nodes, sequential within each node.
		// Each goroutine gets its own OS thread for set_mempolicy (handled
		// inside New() via LockOSThread).
		log.Printf("[shard] allocating %d shards across %d NUMA node(s) in parallel", numShards, len(nodeGroups))
		var wg sync.WaitGroup
		for node, slots := range nodeGroups {
			wg.Add(1)
			go func(node int, slots []shardSlot) {
				defer wg.Done()
				for _, slot := range slots {
					if allocErr != nil {
						return // another node group failed
					}
					if err := buildShard(slot); err != nil {
						errOnce.Do(func() { allocErr = err })
						return
					}
				}
			}(node, slots)
		}
		wg.Wait()
	} else {
		// Single-NUMA or no NUMA: sequential (same as before).
		for _, slots := range nodeGroups {
			for _, slot := range slots {
				if err := buildShard(slot); err != nil {
					allocErr = err
					break
				}
			}
		}
	}

	if allocErr != nil {
		for _, s := range m.shards {
			if s != nil {
				s.Stop()
			}
		}
		return nil, allocErr
	}

	return m, nil
}

// Start starts all shard goroutines and the vacuum coordinator (if enabled).
func (m *Manager) Start() {
	for _, s := range m.shards {
		s.Start()
	}
	m.startVacuum()
	m.allReady.Store(true)
}

// IsShardReady returns whether shard idx has completed allocation.
func (m *Manager) IsShardReady(idx int) bool {
	if idx < 0 || idx >= len(m.shardReady) {
		return false
	}
	return m.shardReady[idx].Load()
}

// IsReady returns whether all shards are allocated and the manager is started.
func (m *Manager) IsReady() bool {
	return m.allReady.Load()
}

// ReadyCount returns the number of shards that have completed allocation.
func (m *Manager) ReadyCount() int {
	count := 0
	for i := range m.shardReady {
		if m.shardReady[i].Load() {
			count++
		}
	}
	return count
}

// Mask returns the shard mask (numShards - 1) for external dispatch use.
func (m *Manager) Mask() uint64 {
	return m.mask
}

// NewManagerAsync creates a Manager shell with no allocated shards.
// The shards slice is pre-allocated (all nil), and numShards/mask are set.
// Call AllocateShards in a background goroutine to perform the actual allocation.
func NewManagerAsync(cfg ManagerConfig) *Manager {
	numShards := cfg.NumShards
	if numShards <= 0 {
		numShards = 16
	}
	numShards = int(nextPow2Mgr(uint64(numShards)))
	totalMem := uint64(cfg.MaxMemoryGB) * 1024 * 1024 * 1024
	numShards = capShardsForMemory(numShards, totalMem)

	maxConcMig := cfg.MaxConcurrentMigrations
	if maxConcMig <= 0 {
		maxConcMig = 2
	}

	return &Manager{
		shards:              make([]*Shard, numShards),
		numShards:           uint64(numShards),
		mask:                uint64(numShards) - 1,
		shardReady:          make([]atomic.Bool, numShards),
		vacuumCfg:           cfg.Vacuum,
		startedAt:           time.Now(),
		migrateSem:          make(chan struct{}, maxConcMig),
		verboseShardLogging: cfg.VerboseShardLogging,
		rebalanceBackoff:    make(map[int]rebalanceFailState),
	}
}

// AllocateShards performs the heavy shard allocation in the current goroutine.
// Designed to be called in a background goroutine after NewManagerAsync.
// After all shards are allocated, calls Start() to launch shard goroutines + vacuum.
func (m *Manager) AllocateShards(cfg ManagerConfig) error {
	numShards := int(m.numShards)
	totalMem := uint64(cfg.MaxMemoryGB) * 1024 * 1024 * 1024
	perShardMem := totalMem / uint64(numShards)

	idxCap := cfg.IndexCapacity
	if idxCap == 0 {
		idxCap = 131072
	}
	perShardIdxCap := idxCap
	if perShardIdxCap < 1024 {
		perShardIdxCap = 1024
	}

	migrateBatch := cfg.MigrateBatchSize
	if migrateBatch <= 0 {
		migrateBatch = 512
	}

	type shardSlot struct {
		idx      int
		numaNode int
	}
	nodeGroups := make(map[int][]shardSlot)
	for i := 0; i < numShards; i++ {
		node := -1
		if cfg.NUMAAssignment != nil && i < len(cfg.NUMAAssignment) {
			node = cfg.NUMAAssignment[i]
		}
		nodeGroups[node] = append(nodeGroups[node], shardSlot{idx: i, numaNode: node})
	}

	var shardsCompleted atomic.Int32
	var allocErr error
	var errOnce sync.Once

	buildShard := func(slot shardSlot) error {
		i := slot.idx
		if cfg.PreAllocCheck != nil {
			if err := cfg.PreAllocCheck(i, numShards, perShardMem); err != nil {
				return fmt.Errorf("pre-alloc check for shard %d: %w", i, err)
			}
		}

		staggeredWarmup := cfg.WarmupOps
		if cfg.WarmupOps > 0 && numShards > 1 {
			staggeredWarmup = cfg.WarmupOps + (cfg.WarmupOps*i)/numShards
		}

		shardCfg := ShardConfig{
			ID:                i,
			IndexCapacity:     perShardIdxCap,
			MaxMemoryBytes:    perShardMem,
			EvictionPolicy:    cfg.EvictionPolicy,
			AllocatorMode:     cfg.AllocatorMode,
			ModelPageBytes:    cfg.ModelPageBytes,
			MaxLeaseDurMs:     cfg.MaxLeaseDurMs,
			UseHugePages:      cfg.UseHugePages,
			HugePageSizeKB:    cfg.HugePageSizeKB,
			DispatchTimeoutMs:   cfg.DispatchTimeoutMs,
			WarmupOps:           staggeredWarmup,
			AutoTuneSlabs:       cfg.AutoTuneSlabs,
			SlabDistribution:    cfg.SlabDistribution,
			InitialClassWeights: cfg.InitialClassWeights,
			VerboseShardLogging: cfg.VerboseShardLogging,
			OnWarmupComplete:    m.NotifyWarmupComplete,
			MigrateBatchSize:    migrateBatch,
			MigrateDrainBudget:  cfg.MigrateDrainBudget,
			MigrateSem:          m.migrateSem,
			NUMANode:              slot.numaNode,
			OOMRejectAfterFails:   cfg.OOMRejectAfterFails,
			SystemOOMFlag:         cfg.SystemOOMFlag,
			SystemPressureLevel:   cfg.SystemPressureLevel,
		}
		s, err := New(shardCfg)
		if err != nil {
			return fmt.Errorf("failed to create shard %d: %w", i, err)
		}
		m.shards[i] = s
		m.shardReady[i].Store(true)

		if cfg.OnShardReady != nil {
			completed := int(shardsCompleted.Add(1))
			cfg.OnShardReady(completed, numShards, perShardMem)
		}
		return nil
	}

	if len(nodeGroups) > 1 {
		log.Printf("[shard] allocating %d shards across %d NUMA node(s) in parallel", numShards, len(nodeGroups))
		var wg sync.WaitGroup
		for node, slots := range nodeGroups {
			wg.Add(1)
			go func(node int, slots []shardSlot) {
				defer wg.Done()
				for _, slot := range slots {
					if allocErr != nil {
						return
					}
					if err := buildShard(slot); err != nil {
						errOnce.Do(func() { allocErr = err })
						return
					}
				}
			}(node, slots)
		}
		wg.Wait()
	} else {
		for _, slots := range nodeGroups {
			for _, slot := range slots {
				if err := buildShard(slot); err != nil {
					allocErr = err
					break
				}
			}
		}
	}

	if allocErr != nil {
		for _, s := range m.shards {
			if s != nil {
				s.Stop()
			}
		}
		return allocErr
	}

	// Start all shard goroutines + vacuum coordinator
	m.Start()
	return nil
}

// Stop stops the vacuum coordinator and all shard goroutines, then waits for them to finish.
func (m *Manager) Stop() {
	m.stopVacuum()
	// Signal all shards to stop
	for _, s := range m.shards {
		s.Stop()
	}
	// Wait for all shard goroutines to exit
	for _, s := range m.shards {
		<-s.Done()
	}
}

// Route returns the shard for the given key hash.
func (m *Manager) Route(keyHash uint64) *Shard {
	return m.shards[keyHash&m.mask]
}

// RouteKey hashes a key and returns the shard.
func (m *Manager) RouteKey(key []byte) (*Shard, uint64) {
	h := index.KeyHash(key)
	return m.Route(h), h
}

// Submit hashes the key, routes to the correct shard, and submits the op.
func (m *Manager) Submit(op ShardOp) OpResult {
	if op.KeyHash == 0 && len(op.Key) > 0 {
		op.KeyHash = index.KeyHash(op.Key)
	}
	shard := m.Route(op.KeyHash)
	return shard.Submit(op)
}

// Get is a convenience method for single key GET.
func (m *Manager) Get(key []byte) ([]byte, bool) {
	h := index.KeyHash(key)
	result := m.Route(h).Submit(ShardOp{
		Type:    OpGet,
		Key:     key,
		KeyHash: h,
		Result:  make(chan OpResult, 1),
	})
	return result.Value, result.Found
}

// Set is a convenience method for single key SET.
func (m *Manager) Set(key, value []byte, ttlMs int64) bool {
	h := index.KeyHash(key)
	result := m.Route(h).Submit(ShardOp{
		Type:    OpSet,
		Key:     key,
		KeyHash: h,
		Value:   value,
		TTLMs:   ttlMs,
		Result:  make(chan OpResult, 1),
	})
	return result.OK
}

// Delete is a convenience method for single key DELETE.
func (m *Manager) Delete(key []byte) bool {
	h := index.KeyHash(key)
	result := m.Route(h).Submit(ShardOp{
		Type:    OpDelete,
		Key:     key,
		KeyHash: h,
		Result:  make(chan OpResult, 1),
	})
	return result.OK
}

// NumShards returns the number of shards.
func (m *Manager) NumShards() int {
	return int(m.numShards)
}

// Shard returns shard at index i.
func (m *Manager) Shard(i int) *Shard {
	return m.shards[i]
}

// NotifyWarmupComplete is called by a shard when its warmup detection completes.
// When not in verbose mode, it emits aggregate progress summaries.
func (m *Manager) NotifyWarmupComplete(shardID int, dominantSize uint64, freqPercent float64) {
	n := m.warmupCompleted.Add(1)
	total := int32(m.numShards)
	elapsed := time.Since(m.startedAt).Seconds()
	if m.verboseShardLogging {
		return // per-shard logs already emitted by size_tracker
	}
	// Log first, midpoint, and final
	if n == 1 || n == total/2 || n == total {
		log.Printf("[cama] warmup: %d/%d shards detected dominant value size %d bytes (%.1f%%) [elapsed %.1fs]",
			n, total, dominantSize, freqPercent, elapsed)
	}
}

// RegisterAllocListener adds a listener for allocator swap events on all shards.
func (m *Manager) RegisterAllocListener(l AllocChangeListener) {
	for _, s := range m.shards {
		s.RegisterAllocListener(l)
	}
}

const (
	minPageHintBytes = 64                // minimum allowed page hint (64 B)
	maxPageHintBytes = 64 * 1024 * 1024  // maximum allowed page hint (64 MB)
)

// SetModelPageHint broadcasts a client-reported model page size to all shards.
// Deduplicates repeated calls with the same value. Rejects out-of-range values.
func (m *Manager) SetModelPageHint(pageBytes uint64) {
	if pageBytes < minPageHintBytes || pageBytes > maxPageHintBytes {
		log.Printf("[cama] ignoring out-of-range page-size hint: %d bytes (valid range: %dâ€“%d)",
			pageBytes, minPageHintBytes, maxPageHintBytes)
		return
	}
	if m.modelPageHint.Load() == pageBytes {
		return // already set to this value
	}
	m.modelPageHint.Store(pageBytes)
	log.Printf("[cama] model page-size hint received: %d bytes â€” broadcasting to %d shards", pageBytes, m.numShards)
	for i := 0; i < int(m.numShards); i++ {
		if !m.shards[i].SubmitAsync(ShardOp{
			Type:      OpPageSizeHint,
			HintBytes: pageBytes,
		}) {
			log.Printf("[cama] WARNING: PageSizeHint dropped for shard %d â€” queue full", i)
		}
	}
}

// startVacuum launches the vacuum coordinator goroutine if enabled.
func (m *Manager) startVacuum() {
	if !m.vacuumCfg.Enabled {
		return
	}
	m.vacuumQuit = make(chan struct{})
	m.vacuumDone = make(chan struct{})
	// Log effective vacuum norm (may be scaled for high shard counts)
	norm := m.vacuumCfg.EvictionRateNorm
	if norm <= 0 {
		norm = 10.0
	}
	if m.numShards > 8 {
		norm *= float64(m.numShards) / 8.0
	}
	log.Printf("[rebalance] vacuum started: interval=%ds, effective_eviction_rate_norm=%.1f (shards=%d)",
		m.vacuumCfg.IntervalSeconds, norm, m.numShards)
	go m.vacuumLoop()
}

// stopVacuum signals the vacuum coordinator to exit and waits for it.
func (m *Manager) stopVacuum() {
	if m.vacuumQuit == nil {
		return
	}
	close(m.vacuumQuit)
	<-m.vacuumDone
}

// vacuumLoop is the auto-rebalancer goroutine.
// It evaluates all shards periodically and rebalances all that need it.
func (m *Manager) vacuumLoop() {
	defer close(m.vacuumDone)

	interval := time.Duration(m.vacuumCfg.IntervalSeconds) * time.Second
	minAge := time.Duration(m.vacuumCfg.MinAgeSeconds) * time.Second

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	minAgeLogged := false
	for {
		select {
		case <-ticker.C:
			// Don't evaluate until the server has been running long enough
			if time.Since(m.startedAt) < minAge {
				if !minAgeLogged {
					log.Printf("[rebalance] waiting for min_age (%ds) before first evaluation", m.vacuumCfg.MinAgeSeconds)
					minAgeLogged = true
				}
				continue
			}
			m.vacuumTick()
		case <-m.vacuumQuit:
			return
		}
	}
}

// rebalanceTarget holds the evaluation result for a shard that needs rebalancing.
type rebalanceTarget struct {
	shardIdx int
	shard    *Shard
	weights  map[uint64]float64
	reason   string
}

// rebalanceResult holds the outcome of a parallel rebalance submission.
type rebalanceResult struct {
	shardIdx int
	ok       bool
	err      error
	weights  map[uint64]float64
}

// vacuumTick evaluates all shards and rebalances all that need it.
// Phase 1: sequential evaluation (PressureSnap + evaluateShardPressure).
// Phase 2: parallel submission via goroutines (migration semaphore limits actual concurrency).
// Phase 3: sequential aggregation of results.
func (m *Manager) vacuumTick() {
	// --- Phase 1: sequential evaluation ---
	if m.vacuumCfg.PressureRebalancing {
		for i := 0; i < int(m.numShards); i++ {
			m.shards[i].PressureSnap()
		}
	}

	var targets []rebalanceTarget
	warmupSkippedCount := 0
	backoffBaseDur := 10 * time.Second // fixed base for backoff calculation
	now := time.Now()

	for i := 0; i < int(m.numShards); i++ {
		s := m.shards[i]
		needsRebalance, weights, reason := m.evaluateShardPressure(s)
		if !needsRebalance {
			det := s.SizeDetectionSnapshot()
			if det.Status == "warming_up" {
				warmupSkippedCount++
			}
			continue
		}

		// Check per-shard backoff before adding to targets
		if bs, ok := m.rebalanceBackoff[i]; ok && now.Before(bs.nextEligible) {
			remainSec := int(bs.nextEligible.Sub(now).Seconds()) + 1
			log.Printf("[rebalance] shard %d: skipping (backing off %ds, %d consecutive failures)",
				i, remainSec, bs.consecutiveFails)
			continue
		}

		targets = append(targets, rebalanceTarget{
			shardIdx: i,
			shard:    s,
			weights:  weights,
			reason:   reason,
		})
	}

	// Cache pending count for VacuumStats (avoids re-evaluation on every scrape)
	m.vacuumPendingShards.Store(int32(len(targets)))

	if len(targets) == 0 {
		if warmupSkippedCount > 0 {
			log.Printf("[rebalance] tick: skipped %d/%d shards (warmup in progress)", warmupSkippedCount, m.numShards)
		} else {
			tick := m.vacuumTickCount.Add(1)
			if tick%10 == 0 {
				log.Printf("[rebalance] tick %d: evaluated %d shards, all healthy", tick, m.numShards)
			}
		}
		return
	}

	// --- Phase 2: parallel submission ---
	results := make([]rebalanceResult, len(targets))
	var wg sync.WaitGroup
	for idx, t := range targets {
		// Check quit signal before submitting
		select {
		case <-m.vacuumQuit:
			return
		default:
		}

		wg.Add(1)
		go func(idx int, t rebalanceTarget) {
			defer wg.Done()
			log.Printf("[rebalance] shard %d: triggering rebalance (%s)", t.shard.ID(), t.reason)
			result := t.shard.Submit(ShardOp{
				Type:         OpRebalance,
				ClassWeights: t.weights,
				Result:       make(chan OpResult, 1),
			})
			results[idx] = rebalanceResult{
				shardIdx: t.shardIdx,
				ok:       result.Err == nil,
				err:      result.Err,
				weights:  t.weights,
			}
		}(idx, t)
	}
	wg.Wait()

	// --- Phase 3: sequential aggregation ---
	rebalancedCount := 0
	for idx, r := range results {
		if r.err != nil {
			m.vacuumRebalanceFailures.Add(1)
			bs := m.rebalanceBackoff[r.shardIdx]
			bs.consecutiveFails++
			exp := bs.consecutiveFails
			if exp > 5 {
				exp = 5 // cap at 2^5 = 32x base (~5 min with 10s base)
			}
			backoffDur := backoffBaseDur * time.Duration(1<<uint(exp))
			bs.nextEligible = time.Now().Add(backoffDur)
			m.rebalanceBackoff[r.shardIdx] = bs
			log.Printf("[rebalance] shard %d: failed (%d consecutive), backing off %ds: %v",
				r.shardIdx, bs.consecutiveFails, int(backoffDur.Seconds()), r.err)
		} else {
			delete(m.rebalanceBackoff, r.shardIdx)
			rebalancedCount++
			m.vacuumRebalances.Add(1)
			m.vacuumLastRebalance.Store(time.Now().Unix())
			if r.weights != nil {
				m.pressureRebuilds.Add(1)
			}
			log.Printf("[rebalance] shard %d: rebalanced (%s)", targets[idx].shard.ID(), targets[idx].reason)
		}
	}

	if warmupSkippedCount > 0 {
		log.Printf("[rebalance] tick: skipped %d/%d shards (warmup in progress)", warmupSkippedCount, m.numShards)
	}
	if rebalancedCount > 0 {
		remaining := int32(len(targets)) - int32(rebalancedCount)
		if remaining < 0 {
			remaining = 0
		}
		m.vacuumPendingShards.Store(remaining)
		log.Printf("[rebalance] tick: rebalanced %d/%d shards", rebalancedCount, m.numShards)
	}
}

// evaluateShard determines if a shard needs rebalancing (read-only, no locks).
func (m *Manager) evaluateShard(s *Shard) bool {
	needs, _, _ := m.evaluateShardPressure(s)
	return needs
}

// evaluateShardPressure determines if a shard needs rebalancing and returns
// the proposed class weights (nil = use default auto-tune weights) and a
// human-readable reason string describing the trigger condition.
func (m *Manager) evaluateShardPressure(s *Shard) (bool, map[uint64]float64, string) {
	// Offset allocator handles all sizes natively â€” no rebalancing needed
	if s.Config().AllocatorMode == "offset" {
		return false, nil, ""
	}

	det := s.SizeDetectionSnapshot()

	// 0. Skip shards still in warmup â€” vacuum must not compete with warmup for the dispatch queue
	if det.Status == "warming_up" {
		return false, nil, ""
	}

	// 0.5 Skip shards under burst load â€” rebalancing (even with async allocator
	// construction) adds memory pressure and MR registration overhead during the
	// window when the shard is already struggling to keep up. Defer to next tick.
	queueDepth := len(s.ops)
	queueCap := cap(s.ops)
	if queueCap > 0 && float64(queueDepth) > float64(queueCap)*0.25 {
		return false, nil, ""
	}

	// Also skip if async allocator construction is already in progress
	if s.allocBuilding.Load() {
		return false, nil, ""
	}

	// 1. Not enough data yet â€” skip (unless pressure rebalancing is on and we have pressure data)
	hasDetection := det.Status == "detected" || det.Status == "rebuilt"

	// 2. Detected size differs from configured model_page_bytes â†’ immediate rebalance (no weights)
	// When configuredSize=0 (auto-detect mode), skip this check â€” the system is self-tuning
	// and should not retrigger after migration has already tuned the allocator.
	if hasDetection {
		configuredSize := s.Config().ModelPageBytes
		if configuredSize > 0 && det.DominantValueSize > 0 && det.DominantValueSize != configuredSize {
			return true, nil, fmt.Sprintf("size mismatch (detected=%d, configured=%d)", det.DominantValueSize, configuredSize)
		}
	}

	// 3. Pressure-based evaluation (when enabled)
	// Simplified: trigger rebalance when any class has evictions above the
	// rate threshold AND there is global free space to redistribute into.
	if m.vacuumCfg.PressureRebalancing {
		m.pressureEvals.Add(1)

		snap := s.LoadPressureSnapshot()
		if snap != nil && snap.WindowDuration > 0 && hasPressureActivity(snap) {
			windowSec := snap.WindowDuration.Seconds()
			if windowSec <= 0 {
				windowSec = 1.0
			}
			evictionThreshold := m.vacuumCfg.EvictionRateNorm
			if evictionThreshold <= 0 {
				evictionThreshold = 10.0
			}
			// Scale threshold by shard count relative to 8-shard reference
			if m.numShards > 8 {
				evictionThreshold *= float64(m.numShards) / 8.0
			}

			a := s.Allocator()
			utils := a.ClassUtilizations()
			var totalUsed, totalSlots uint64
			for _, u := range utils {
				totalUsed += u.UsedSlots
				totalSlots += u.TotalSlots
			}
			globalFreeRatio := float64(0)
			if totalSlots > 0 {
				globalFreeRatio = 1.0 - float64(totalUsed)/float64(totalSlots)
			}

			for i, u := range utils {
				if u.TotalSlots == 0 || i >= len(snap.Evictions) {
					continue
				}
				rate := float64(snap.Evictions[i]) / windowSec
				if rate > evictionThreshold && globalFreeRatio > 0.20 {
					return true, nil, fmt.Sprintf("pressure: class %d evicting %.1f/s (threshold %.1f), %.0f%% free globally",
						u.Size, rate, evictionThreshold, globalFreeRatio*100)
				}
			}
		}
	}

	// 3.5 Watermark with relative check: trigger if a class is above watermark
	// AND there's significant free space globally (meaning redistribution can help)
	if m.vacuumCfg.WatermarkThreshold > 0 {
		a := s.Allocator()
		utils := a.ClassUtilizations()
		var totalUsedW, totalSlotsW uint64
		var maxClassUtil float64
		for _, u := range utils {
			totalUsedW += u.UsedSlots
			totalSlotsW += u.TotalSlots
			if u.TotalSlots > 0 {
				cu := float64(u.UsedSlots) / float64(u.TotalSlots)
				if cu > maxClassUtil {
					maxClassUtil = cu
				}
			}
		}
		globalUtilW := float64(0)
		if totalSlotsW > 0 {
			globalUtilW = float64(totalUsedW) / float64(totalSlotsW)
		}
		globalFreeRatio := 1.0 - globalUtilW
		if maxClassUtil > m.vacuumCfg.WatermarkThreshold && globalFreeRatio > 0.20 {
			return true, nil, fmt.Sprintf("watermark: class at %.0f%% with %.0f%% free globally", maxClassUtil*100, globalFreeRatio*100)
		}
	}

	// 4. Legacy checks (only if detection has completed)
	if !hasDetection {
		return false, nil, ""
	}

	// Dominant class saturated (> utilization threshold)
	usedFrac, ok := s.DominantClassUtilization()
	if ok && usedFrac > m.vacuumCfg.UtilizationThreshold {
		return true, nil, fmt.Sprintf("dominant class saturated (%.0f%% > %.0f%%)", usedFrac*100, m.vacuumCfg.UtilizationThreshold*100)
	}

	// High eviction rate + high waste = allocator misconfigured
	sets := s.Metrics().Sets()
	if sets > 0 {
		evictionRate := float64(s.Metrics().Evictions()) / float64(sets)
		if evictionRate > 0.50 && det.CurrentSlotUtilization < 80 {
			return true, nil, fmt.Sprintf("high eviction rate %.0f%% with low slot utilization %.0f%%", evictionRate*100, det.CurrentSlotUtilization)
		}
	}

	return false, nil, ""
}

// ComputeClassPressure computes the pressure score for a single slab class.
// Uses relative utilization (class vs global average) so uniform load doesn't trigger rebalancing.
// Formula: relativeUtil*0.30 + min(evictRate/norm,1)*0.30 + allocFailRate*0.20 + min(promoFromRate/norm,1)*0.20
func ComputeClassPressure(classUtil, globalUtil, evictRate, allocFailRate, promoFromRate, norm float64) float64 {
	if norm <= 0 {
		norm = 10.0
	}
	relativeUtil := classUtil - globalUtil
	if relativeUtil < 0 {
		relativeUtil = 0
	}
	return relativeUtil*0.30 +
		math.Min(evictRate/norm, 1.0)*0.30 +
		allocFailRate*0.20 +
		math.Min(promoFromRate/norm, 1.0)*0.20
}

// VacuumStats returns a snapshot of vacuum coordinator state for metrics export.
// Uses a cached pending count from the last vacuum tick (O(1) atomics instead of
// O(numShards) evaluateShardPressure calls). Value may be up to one vacuum
// interval stale, which is acceptable for a Prometheus gauge.
func (m *Manager) VacuumStats() VacuumStats {
	return VacuumStats{
		RebalancesTotal:    m.vacuumRebalances.Load(),
		LastRebalanceEpoch: m.vacuumLastRebalance.Load(),
		PendingShards:      int(m.vacuumPendingShards.Load()),
		PressureEvals:      m.pressureEvals.Load(),
		PressureRebuilds:   m.pressureRebuilds.Load(),
		MaxDrift:           0, // deprecated: drift tracking removed in pressure rebalancing simplification
		RebalanceFailures:  m.vacuumRebalanceFailures.Load(),
	}
}

// --- On-demand maintenance API result types ---

// OnDemandVacuumResult holds the result of a client-triggered vacuum evaluation.
type OnDemandVacuumResult struct {
	ShardsEvaluated  int            `json:"shards_evaluated"`
	ShardsRebalanced []int          `json:"shards_rebalanced"`
	ShardsSkipped    map[int]string `json:"shards_skipped"`
	VacuumStats      VacuumStats    `json:"vacuum_stats"`
	DurationMs       int64          `json:"duration_ms"`
}

// OnDemandAutoTuneResult holds the result of a client-triggered auto-tune.
type OnDemandAutoTuneResult struct {
	ShardsRebuilt      []int                      `json:"shards_rebuilt"`
	ShardsSkipped      map[int]string             `json:"shards_skipped"`
	DetectionSnapshots map[int]DetectionSnapshot   `json:"detection_snapshots"`
	DurationMs         int64                      `json:"duration_ms"`
}

// MaintenanceStatusResult holds read-only vacuum + detection state.
type MaintenanceStatusResult struct {
	VacuumConfig    VacuumConfig          `json:"vacuum_config"`
	VacuumStats     VacuumStats           `json:"vacuum_stats"`
	ShardDetections []ShardDetectionEntry `json:"shard_detections"`
}

// ShardDetectionEntry pairs a shard ID with its detection snapshot.
type ShardDetectionEntry struct {
	ShardID   int               `json:"shard_id"`
	Detection DetectionSnapshot `json:"detection"`
}

// OnDemandVacuum evaluates target shards and rebalances those that need it.
// Unlike the background vacuum loop, this skips cooldown sleep between shards.
func (m *Manager) OnDemandVacuum(force bool, shardIDs []int) OnDemandVacuumResult {
	start := time.Now()
	targets := m.resolveShardIDs(shardIDs)

	result := OnDemandVacuumResult{
		ShardsSkipped: make(map[int]string),
	}

	// Take pressure snapshots if pressure rebalancing is enabled
	if m.vacuumCfg.PressureRebalancing {
		for _, s := range targets {
			s.PressureSnap()
		}
	}

	for _, s := range targets {
		result.ShardsEvaluated++
		id := s.ID()

		if !force && s.Config().AllocatorMode == "offset" {
			result.ShardsSkipped[id] = "offset allocator"
			continue
		}

		needsRebalance, weights, _ := m.evaluateShardPressure(s)
		if !needsRebalance && !force {
			result.ShardsSkipped[id] = "healthy"
			continue
		}

		opResult := s.Submit(ShardOp{
			Type:         OpRebalance,
			ClassWeights: weights,
			Result:       make(chan OpResult, 1),
		})
		if opResult.Err != nil {
			result.ShardsSkipped[id] = opResult.Err.Error()
			continue
		}

		result.ShardsRebalanced = append(result.ShardsRebalanced, id)
		m.vacuumRebalances.Add(1)
		m.vacuumLastRebalance.Store(time.Now().Unix())
		if weights != nil {
			m.pressureRebuilds.Add(1)
		}
	}

	result.VacuumStats = m.VacuumStats()
	result.DurationMs = time.Since(start).Milliseconds()
	return result
}

// OnDemandAutoTune forces auto-tune detection + rebuild on target shards.
func (m *Manager) OnDemandAutoTune(force bool, shardIDs []int) OnDemandAutoTuneResult {
	start := time.Now()
	targets := m.resolveShardIDs(shardIDs)

	result := OnDemandAutoTuneResult{
		ShardsSkipped:      make(map[int]string),
		DetectionSnapshots: make(map[int]DetectionSnapshot),
	}

	for _, s := range targets {
		id := s.ID()
		opResult := s.Submit(ShardOp{
			Type:        OpForceAutoTune,
			ForceDetect: force,
			Result:      make(chan OpResult, 1),
		})
		if opResult.Err != nil {
			result.ShardsSkipped[id] = opResult.Err.Error()
			continue
		}
		result.ShardsRebuilt = append(result.ShardsRebuilt, id)
		if opResult.Info != nil {
			if snap, ok := opResult.Info["detection"].(DetectionSnapshot); ok {
				result.DetectionSnapshots[id] = snap
			}
		}
	}

	result.DurationMs = time.Since(start).Milliseconds()
	return result
}

// MaintenanceStatus returns read-only vacuum config, stats, and per-shard detection state.
func (m *Manager) MaintenanceStatus() MaintenanceStatusResult {
	detections := make([]ShardDetectionEntry, int(m.numShards))
	for i := 0; i < int(m.numShards); i++ {
		detections[i] = ShardDetectionEntry{
			ShardID:   i,
			Detection: m.shards[i].SizeDetectionSnapshot(),
		}
	}
	return MaintenanceStatusResult{
		VacuumConfig:    m.vacuumCfg,
		VacuumStats:     m.VacuumStats(),
		ShardDetections: detections,
	}
}

// resolveShardIDs returns the target shards: all if shardIDs is nil/empty, else only the listed ones.
func (m *Manager) resolveShardIDs(shardIDs []int) []*Shard {
	if len(shardIDs) == 0 {
		return m.shards
	}
	targets := make([]*Shard, 0, len(shardIDs))
	for _, id := range shardIDs {
		if id >= 0 && id < int(m.numShards) {
			targets = append(targets, m.shards[id])
		}
	}
	return targets
}

// hasPressureActivity returns true if a pressure snapshot has any eviction or alloc failure
// activity. Pure utilization without evictions/failures means "full but happy" â€” no action needed.
func hasPressureActivity(snap *PressureSnapshot) bool {
	for _, e := range snap.Evictions {
		if e > 0 {
			return true
		}
	}
	for _, f := range snap.AllocFails {
		if f > 0 {
			return true
		}
	}
	for _, p := range snap.PromotionsFrom {
		if p > 0 {
			return true
		}
	}
	return false
}

// NextPow2 rounds v up to the next power of 2 (used for shard count calculation).
func NextPow2(v uint64) uint64 {
	return nextPow2Mgr(v)
}

func nextPow2Mgr(v uint64) uint64 {
	if v == 0 {
		return 1
	}
	v--
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	v |= v >> 32
	v++
	return v
}

// capShardsForMemory reduces numShards so that each shard gets at least 8 GB.
// With dedicated slab distribution (2 classes, 95/5 split), 8 GB/shard is ample.
// With auto mode (32 classes), the warmup concentrates memory on active classes.
// Hard cap at 8 GB; warning at 16 GB recommending dedicated distribution.
// When total memory is below 8 GB the cap is skipped (can't be satisfied).
func capShardsForMemory(numShards int, totalMem uint64) int {
	const minPerShardMem uint64 = 8 << 30  // 8 GB hard minimum
	const warnPerShardMem uint64 = 16 << 30 // 16 GB recommendation threshold
	if totalMem < minPerShardMem || totalMem/uint64(numShards) >= minPerShardMem {
		// Not capped â€” but warn if per-shard is below 16 GB recommendation
		if totalMem >= minPerShardMem && totalMem/uint64(numShards) < warnPerShardMem {
			log.Printf("[cama] WARNING: per-shard memory %.1f GB < 16 GB â€” consider slab_distribution=\"dedicated\" for optimal slot utilization",
				float64(totalMem/uint64(numShards))/(1<<30))
		}
		return numShards
	}
	capped := int(totalMem / minPerShardMem)
	if capped < 1 {
		capped = 1
	}
	// Round down to power of 2 for fast masking
	capped = int(1 << uint(63-bits.LeadingZeros64(uint64(capped))))
	log.Printf("[cama] reducing num_shards %d â†’ %d (per-shard memory %.1f GB < 8 GB minimum)",
		numShards, capped, float64(totalMem/uint64(numShards))/(1<<30))
	return capped
}
