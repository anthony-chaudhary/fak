package shard

import (
	"errors"
	"log"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
	"github.com/anthony-chaudhary/fak/internal/l3server/eviction"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/lease"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/numa"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
)

// OpType enumerates the operations a shard can handle.
type OpType uint8

const (
	OpGet OpType = iota
	OpSet
	OpDelete
	OpMGet
	OpMSet
	OpTest    // EXISTS check
	OpLease
	OpPin
	OpUnpin
	OpInfo
	OpMDel    // Batch delete
	OpKeys    // Key pattern scan
	OpFlush        // Flush all data
	OpRebalance      // ZeroLatencyBalance rebalance (internal only, not a wire op)
	OpPageSizeHint   // Client-reported model page size hint (triggers slab rebuild)
	OpForceAutoTune  // Client-requested auto-tune (force detect + rebuild)
	OpSnapshot       // Iterate all entries and return them for snapshot
	OpRestore        // Bulk-load entries from a snapshot
	OpMGetWithAlloc  // Batch GET returning AllocInfos per key (for RDMA batch)
	OpBatchLease     // Batch lease grant (multiple keyHashes in one op)
)

// ShardOp is a request submitted to a shard.
type ShardOp struct {
	Type         OpType
	Key          []byte
	KeyHash      uint64
	Value        []byte
	TTLMs        int64 // TTL in milliseconds; 0 = no expiry
	LeaseMs      int64
	Keys         [][]byte   // for MGET/MSET/MDEL
	KeyHashes    []uint64
	Values       [][]byte   // for MSET
	Pattern      []byte     // for KEYS scan
	ClassWeights map[uint64]float64 // pressure-derived weights for OpRebalance; nil = use defaults
	HintBytes      uint64             // for OpPageSizeHint: client-reported model page size
	ForceDetect    bool               // for OpForceAutoTune: force early detection
	RestoreEntries []snapshot.KVEntry // for OpRestore: entries to load
	EnqueuedAt     time.Time          // stamped before channel send for queue-wait measurement
	Result         chan OpResult
}

// AllocMeta describes the physical allocation of a value in the slab allocator.
// Used by the RDMA transport to locate the value's memory region for RDMA Read.
type AllocMeta struct {
	ClassIdx int
	Offset   uint64
	Size     uint64
}

// OpResult is the result returned from a shard operation.
type OpResult struct {
	Value       []byte
	Values      [][]byte  // for MGET
	Found       bool
	Founds      []bool    // for MGET/MTEST
	OK              bool
	Err             error
	Info            map[string]interface{}
	MatchedKeys     [][]byte           // for KEYS scan
	AllocInfo       *AllocMeta         // populated for GET when value is slab-backed
	AllocInfos      []*AllocMeta       // populated for OpMGetWithAlloc: per-key alloc metadata
	SnapshotEntries []snapshot.KVEntry // for OpSnapshot
	Loaded          int                // for OpRestore: number of entries loaded
	SetStatuses     []byte             // per-key status for MSET (0=ok, nonzero=error)
}

// ShardConfig holds configuration for a single shard.
type ShardConfig struct {
	ID                int
	IndexCapacity     uint64
	MaxMemoryBytes    uint64
	EvictionPolicy    string // "wtinylfu", "sieve", "lru"
	AllocatorMode     string // "slab" (default) or "offset"
	ModelPageBytes    uint64
	MaxLeaseDurMs     int64
	UseHugePages      bool // kept for convenience checks (derived from HugePageSizeKB > 0)
	HugePageSizeKB    int  // 0 = disabled, 2048 = 2MB, 1048576 = 1GB
	DispatchTimeoutMs int64 // 0 = use default (30s)
	WarmupOps           int   // SET ops before auto-detect; 0 = disabled
	AutoTuneSlabs       bool  // rebuild allocator on FLUSH with detected sizes
	SlabDistribution    string             // "auto", "model", "uniform", "dedicated"
	InitialClassWeights map[uint64]float64 // static class weight overrides from config; nil = use defaults
	VerboseShardLogging bool                                                       // emit per-shard log lines (default false â€” aggregates only)
	OnWarmupComplete    func(shardID int, dominantSize uint64, freqPercent float64) // aggregate warmup callback
	MigrateBatchSize    int                // entries per ZeroLatencyBalance batch (default 512)
	MigrateDrainBudget  int                // max ops drained per migration cycle (default 64)
	MigrateSem          chan struct{}       // shared semaphore limiting concurrent migrations (from Manager)

	// NUMA pinning â€” set by Manager when topology is detected
	NUMANode int // -1 = no pinning

	// OOM graceful handling â€” consecutive alloc failures before fast-rejecting SETs.
	// 0 = disabled (always attempt full eviction loop). Default 3.
	OOMRejectAfterFails int

	// SystemOOMFlag is a pointer to the syshealth Monitor's MemPressureActive flag.
	// When non-nil and set to true, checkOOM() rejects SETs proactively even if
	// this individual shard has not yet experienced allocation failures.
	SystemOOMFlag *atomic.Bool

	// SystemPressureLevel is a pointer to the syshealth Monitor's MemPressureLevel.
	// Provides tiered backpressure (0=normal, 1=elevated, 2=high, 3=critical,
	// 4=emergency). Supersedes SystemOOMFlag for gradual write throttling.
	SystemPressureLevel *atomic.Int32
}

// migrateState holds in-progress ZeroLatencyBalance state.
// Non-nil only while a migration is active.
type migrateState struct {
	oldAlloc     alloc.Allocator    // source allocator (reads for non-migrated entries)
	newAlloc     alloc.Allocator    // destination allocator
	cursor       uint64             // next slot index to resume iteration from
	batch        int                // entries per batch
	migrated     int                // count of entries migrated so far
	weights      map[uint64]float64 // nil = auto-tune, non-nil = pressure-driven
	freezeAfter  bool               // set sizeTracker.frozen=true on finalize (auto-detect only, not rebalance)
	startTime    time.Time          // when migration started
	totalEntries uint64             // snapshot of index count at start
	lastLogTime  time.Time          // last periodic progress log
	preRegDone   []<-chan struct{}   // signals from AllocPreRegisterer goroutines

	drainBudget    int               // drain budget override (0 = use config default)
}

// DefaultIndexCapacity is the default Swiss table capacity per shard (2^17).
const DefaultIndexCapacity uint64 = 131072

var errDispatchTimeout = errors.New("shard dispatch timeout")

// ErrOOM is a sentinel error returned when a SET is rejected due to memory pressure.
// The dispatch layer detects this and returns RespOOM with diagnostics instead of RespError.
var ErrOOM = errors.New("OOM: memory pressure â€” SET rejected")

// IsOOM reports whether err is the OOM sentinel.
func IsOOM(err error) bool {
	return errors.Is(err, ErrOOM)
}

// --- Panic recovery tracking ---

const (
	panicWindowSize  = 3                // panics to track
	panicWindowDur   = 60 * time.Second // window for rapid-panic detection
	consecutiveDropThreshold int32 = 100   // M2: drops before circuit breaker trips
	circuitCooldownBase              = 200 * time.Millisecond
	circuitCooldownMax               = 5 * time.Second
)

// panicTracker is a circular buffer of panic timestamps for rapid-panic detection.
type panicTracker struct {
	times [panicWindowSize]time.Time
	idx   int
}

func (pt *panicTracker) record(t time.Time) bool {
	pt.times[pt.idx] = t
	pt.idx = (pt.idx + 1) % panicWindowSize
	// Check if all slots are within the window
	oldest := pt.times[pt.idx] // next slot to be overwritten is the oldest
	return !oldest.IsZero() && t.Sub(oldest) < panicWindowDur
}

// circuitState tracks consecutive op drops for circuit-breaker logic.
type circuitState struct {
	consecutiveDrops int32
	cooldownUntil    time.Time
	cooldownDur      time.Duration
	halfOpen         bool
}

// --- Channel and timer pools for Submit() hot path ---

var resultChanPool = sync.Pool{
	New: func() any { return make(chan OpResult, 1) },
}

func getResultChan() chan OpResult {
	ch := resultChanPool.Get().(chan OpResult)
	// Drain any stale result (defensive â€” shouldn't happen in normal path)
	select {
	case <-ch:
	default:
	}
	return ch
}

func putResultChan(ch chan OpResult) {
	// Drain before returning (result may not have been consumed on timeout path)
	select {
	case <-ch:
	default:
	}
	resultChanPool.Put(ch)
}

var timerPool = sync.Pool{
	New: func() any { return time.NewTimer(time.Hour) },
}

func getTimer(d time.Duration) *time.Timer {
	t := timerPool.Get().(*time.Timer)
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
	t.Reset(d)
	return t
}

func putTimer(t *time.Timer) {
	if !t.Stop() {
		select {
		case <-t.C:
		default:
		}
	}
	timerPool.Put(t)
}

// xorshift64 is a fast, non-cryptographic PRNG for probabilistic rejection.
// Used by checkOOM for fair tiered backpressure. Single-threaded (shard goroutine only).
type xorshift64 struct {
	state uint64
}

func newXorshift64(seed uint64) xorshift64 {
	if seed == 0 {
		seed = 1 // xorshift64 cannot be seeded with 0
	}
	return xorshift64{state: seed}
}

func (x *xorshift64) next() uint64 {
	x.state ^= x.state << 13
	x.state ^= x.state >> 7
	x.state ^= x.state << 17
	return x.state
}

// allocBox wraps an alloc.Allocator so it can be stored in atomic.Pointer.
// Go's atomic.Pointer requires a concrete (non-interface) type parameter.
type allocBox struct {
	a alloc.Allocator
}

// AllocatorChange describes an allocator swap for listener notification.
type AllocatorChange struct {
	ShardID      int
	OldAllocator alloc.Allocator
	NewAllocator alloc.Allocator
}

// AllocChangeListener receives notifications when a shard's allocator is swapped.
// Implementations must not block â€” heavy work (e.g. MR deregistration) should be
// spawned in a goroutine.
type AllocChangeListener interface {
	OnAllocatorChanged(AllocatorChange)
}

// AllocPreRegisterer is optionally implemented by AllocChangeListeners to
// pre-register resources (e.g. RDMA MRs) for a new allocator during migration.
// This allows expensive registration to overlap with batch data migration,
// eliminating blocking during finalizeMigration.
type AllocPreRegisterer interface {
	PreRegisterAllocator(shardID int, newAlloc alloc.Allocator) <-chan struct{}
	DiscardPreRegistered(shardID int)
}

// pendingAllocResult holds an allocator constructed asynchronously in a background
// goroutine, awaiting pickup by the shard's run loop to begin migration.
type pendingAllocResult struct {
	newAlloc    alloc.Allocator
	weights     map[uint64]float64
	freezeAfter bool // set sizeTracker.frozen=true on finalize (auto-detect rebuilds, not vacuum)
}

// Shard is a shared-nothing unit of key-value storage.
// Exactly one goroutine services each shard â€” zero locks.
type Shard struct {
	id              int
	idx             *index.Table
	allocPtr        atomic.Pointer[allocBox]
	eviction        eviction.Policy
	leases          *lease.Table
	metrics         *metrics.ShardMetrics
	ops             chan ShardOp
	quit            chan struct{}
	done            chan struct{} // closed when run() exits
	config          ShardConfig
	dispatchTimeout time.Duration
	sizeTracker      *valueSizeTracker
	pressureTracker  *classPressureTracker
	latency          *opLatencyTrackers // per-op-category latency tracking
	allocListeners   []AllocChangeListener
	migrate          *migrateState // non-nil during ZeroLatencyBalance

	// Async allocator construction: background goroutine builds the allocator
	// and sends the result here. The shard's run loop picks it up to start migration,
	// avoiding a 30s+ mmap(MAP_POPULATE) stall in the shard's op goroutine.
	pendingAlloc  chan *pendingAllocResult // capacity 1
	allocBuilding atomic.Bool             // true while background goroutine is constructing

	// C1: Panic recovery with cooldown
	panics             *panicTracker
	panicCooldownUntil time.Time
	panicCooldownDur   time.Duration

	// M2: Circuit breaker
	circuit circuitState

	// L1: Adaptive TTL sweep
	ttlBatchSize int
	ttlInterval  time.Duration

	// H4: Rate-limited drop logging (inline atomic gate, 1s interval)
	dropLogLast atomic.Int64 // unix nanos of last drop warning

	// OOM admission control â€” fast-reject SETs after consecutive alloc failures
	consecAllocFails int // consecutive allocation failures (reset on success or free)
	oomActive        bool // true = reject SETs without attempting eviction
	oomThreshold     int  // from config.OOMRejectAfterFails; 0 = disabled

	// System-level memory pressure flag from syshealth Monitor.
	// Read-only from shard goroutine; written by health monitor goroutine.
	systemOOMFlag *atomic.Bool

	// Tiered system memory pressure level from syshealth Monitor.
	// 0-4 (PressureNone..PressureEmergency). Used for gradual write throttling.
	systemPressureLevel *atomic.Int32

	// PRNG for probabilistic tiered rejection in checkOOM.
	// Single-threaded (shard goroutine only) â€” no synchronization needed.
	rng xorshift64
}

// New creates a new Shard.
func New(cfg ShardConfig) (*Shard, error) {
	// Bind memory allocations to the shard's NUMA node so slab regions
	// are local to the CPUs that will process this shard's operations.
	//
	// CRITICAL: LockOSThread is required because set_mempolicy(2) is a
	// per-thread syscall. Without it, the Go scheduler can migrate this
	// goroutine to a different OS thread between SetMemPolicy and the
	// mmap inside NewSlabAllocator â€” causing the mmap to run with
	// MPOL_DEFAULT and pull hugepages from the wrong NUMA node.
	if cfg.NUMANode >= 0 {
		runtime.LockOSThread()
		if err := numa.SetMemPolicy(cfg.NUMANode); err != nil {
			log.Printf("[shard %d] NUMA membind to node %d failed: %v (using default policy)", cfg.ID, cfg.NUMANode, err)
			runtime.UnlockOSThread()
		} else {
			log.Printf("[shard %d] NUMA memory bound to node %d (MPOL_BIND) â€” slab allocations will be node-local", cfg.ID, cfg.NUMANode)
			defer func() {
				numa.ResetMemPolicy()
				runtime.UnlockOSThread()
			}()
		}
	}

	// Create index
	idxCap := cfg.IndexCapacity
	if idxCap == 0 {
		idxCap = DefaultIndexCapacity
	}
	idx := index.NewTable(idxCap)

	// Create allocator
	var a alloc.Allocator
	var allocErr error
	switch cfg.AllocatorMode {
	case "offset":
		maxAllocs := uint32(idxCap * 2)
		if maxAllocs < 4096 {
			maxAllocs = 4096
		}
		a, allocErr = alloc.NewOffsetAllocator(alloc.OffsetAllocatorConfig{
			MaxMemoryBytes: cfg.MaxMemoryBytes,
			HugePageSizeKB: cfg.HugePageSizeKB,
			MaxAllocations: maxAllocs,
		})
	default:
		a, allocErr = alloc.NewSlabAllocator(alloc.SlabConfig{
			MaxMemoryBytes: cfg.MaxMemoryBytes,
			HugePageSizeKB: cfg.HugePageSizeKB,
			ModelPageBytes: cfg.ModelPageBytes,
			ClassWeights:   cfg.InitialClassWeights,
			Dedicated:      cfg.SlabDistribution == "dedicated",
		})
	}
	if allocErr != nil {
		return nil, allocErr
	}

	// Create eviction engine with a closure that captures a pointer to the
	// real callback. This avoids creating the engine twice â€” the callback is
	// wired up after the shard is constructed.
	maxKeys := idxCap * index.MaxLoadNumerator / index.MaxLoadDenominator
	var onEvict func(uint64, uint16)
	evictTrampoline := func(keyHash uint64, keyLen uint16) { onEvict(keyHash, keyLen) }

	ev := eviction.NewPolicy(cfg.EvictionPolicy, maxKeys, evictTrampoline)

	// Create lease table
	maxLease := cfg.MaxLeaseDurMs
	if maxLease == 0 {
		maxLease = 30000
	}
	lt := lease.NewTable(maxLease)

	dispatchTimeout := time.Duration(cfg.DispatchTimeoutMs) * time.Millisecond
	if dispatchTimeout <= 0 {
		dispatchTimeout = 30 * time.Second
	}

	// L2: Set max capacity on index for hard cap enforcement
	idx.SetMaxCapacity(maxKeys)

	s := &Shard{
		id:              cfg.ID,
		idx:             idx,
		eviction:        ev,
		leases:          lt,
		metrics:         metrics.NewShardMetrics(cfg.ID),
		ops:             make(chan ShardOp, 4096),
		quit:            make(chan struct{}),
		done:            make(chan struct{}),
		config:          cfg,
		dispatchTimeout: dispatchTimeout,
		sizeTracker:      newValueSizeTracker(int64(cfg.WarmupOps), cfg.ID, cfg.VerboseShardLogging, cfg.OnWarmupComplete),
		pressureTracker:  newClassPressureTracker(a.NumClasses()),
		latency:          &opLatencyTrackers{},
		pendingAlloc:     make(chan *pendingAllocResult, 1),
		panics:           &panicTracker{},
		ttlBatchSize:     ttlSweepBatchSize,
		ttlInterval:      ttlSweepInterval,
		oomThreshold:        cfg.OOMRejectAfterFails,
		systemOOMFlag:       cfg.SystemOOMFlag,
		systemPressureLevel: cfg.SystemPressureLevel,
	}
	s.rng = newXorshift64(uint64(time.Now().UnixNano()) + uint64(cfg.ID))
	s.allocPtr.Store(&allocBox{a: a})

	// Wire up eviction callback now that the shard is constructed
	onEvict = func(keyHash uint64, keyLen uint16) {
		s.evictKey(keyHash, keyLen)
	}

	return s, nil
}

// Start launches the shard's goroutine, pinned to an OS thread.
func (s *Shard) Start() {
	go s.run()
}

// Submit sends an operation to the shard and waits for the result.
// Returns a timeout error if the shard doesn't accept or respond within dispatchTimeout.
func (s *Shard) Submit(op ShardOp) OpResult {
	ch := getResultChan()
	op.Result = ch
	// NO defer putResultChan â€” must handle per-path to avoid cross-contamination
	// on timeout (shard goroutine may still hold a reference to ch).

	timer := getTimer(s.dispatchTimeout)
	defer putTimer(timer)

	op.EnqueuedAt = time.Now()
	select {
	case s.ops <- op:
	case <-timer.C:
		putResultChan(ch) // safe: shard never received op
		return OpResult{Err: errDispatchTimeout}
	}
	// Reset timer for result wait
	if !timer.Stop() {
		select {
		case <-timer.C:
		default:
		}
	}
	timer.Reset(s.dispatchTimeout)
	select {
	case result := <-ch:
		putResultChan(ch) // safe: shard finished writing
		return result
	case <-timer.C:
		// DON'T recycle â€” shard still holds ch via op.Result.
		// Buffered chan (cap 1) absorbs the eventual write; both refs drop â†’ GC.
		return OpResult{Err: errDispatchTimeout}
	}
}

// SubmitAsync sends an operation without waiting.
// Returns true if the op was enqueued, false if the channel was full (dropped).
func (s *Shard) SubmitAsync(op ShardOp) bool {
	// M2: Circuit breaker â€” fast-reject when shard is in cooldown
	if s.circuit.consecutiveDrops >= consecutiveDropThreshold {
		now := time.Now()
		if now.Before(s.circuit.cooldownUntil) {
			// Still in cooldown â€” reject immediately
			s.metrics.IncrOpsDropped()
			if op.Result != nil {
				op.Result <- OpResult{Err: errDispatchTimeout}
			}
			return false
		}
		// Cooldown expired â€” enter half-open: allow one probe op through
		s.circuit.halfOpen = true
	}

	op.EnqueuedAt = time.Now()
	select {
	case s.ops <- op:
		if s.circuit.halfOpen {
			// Half-open probe succeeded â€” close circuit fully
			s.circuit.consecutiveDrops = 0
			s.circuit.cooldownDur = 0
			s.circuit.halfOpen = false
			log.Printf("[l3server] shard %d circuit breaker CLOSED (half-open probe succeeded)", s.id)
		} else {
			s.circuit.consecutiveDrops = 0 // successful enqueue resets drop counter
		}
		return true
	default:
		if s.circuit.halfOpen {
			// Half-open probe failed â€” escalate cooldown
			s.circuit.halfOpen = false
			s.circuit.cooldownDur *= 2
			if s.circuit.cooldownDur > circuitCooldownMax {
				s.circuit.cooldownDur = circuitCooldownMax
			}
			s.circuit.cooldownUntil = time.Now().Add(s.circuit.cooldownDur)
			s.metrics.IncrOpsDropped()
			if op.Result != nil {
				op.Result <- OpResult{Err: errDispatchTimeout}
			}
			log.Printf("[l3server] shard %d circuit breaker half-open probe FAILED â€” escalating cooldown to %v",
				s.id, s.circuit.cooldownDur)
			return false
		}
		// Channel full â€” drop to avoid blocking
		s.circuit.consecutiveDrops++
		if s.circuit.consecutiveDrops == consecutiveDropThreshold {
			s.circuit.cooldownDur = circuitCooldownBase
			s.circuit.cooldownUntil = time.Now().Add(s.circuit.cooldownDur)
			s.metrics.IncrCircuitTrips()
			log.Printf("[l3server] WARNING: shard %d circuit breaker TRIPPED â€” %d consecutive drops, cooldown %v",
				s.id, consecutiveDropThreshold, s.circuit.cooldownDur)
		}
		// H4: Rate-limited drop logging (1s gate)
		now := time.Now().UnixNano()
		if last := s.dropLogLast.Load(); now-last > int64(time.Second) {
			if s.dropLogLast.CompareAndSwap(last, now) {
				log.Printf("[l3server] WARNING: shard %d op queue full (cap=%d) â€” op dropped (type=%d)",
					s.id, cap(s.ops), op.Type)
			}
		}
		s.metrics.IncrOpsDropped()
		if op.Result != nil {
			op.Result <- OpResult{Err: errDispatchTimeout}
		}
		return false
	}
}

// Stop signals the shard goroutine to exit.
func (s *Shard) Stop() {
	close(s.quit)
}

// Done returns a channel that is closed when the shard goroutine has exited.
func (s *Shard) Done() <-chan struct{} {
	return s.done
}

// Metrics returns the shard's metrics collector.
func (s *Shard) Metrics() *metrics.ShardMetrics {
	return s.metrics
}

// ID returns the shard ID.
func (s *Shard) ID() int {
	return s.id
}

// OpLatencySnapshot returns p50/p99 latencies per op category and queue depth.
// Called from the metrics scrape goroutine â€” reads ring buffer entries which are
// int64 aligned (atomic on 64-bit platforms), so no lock is needed.
func (s *Shard) OpLatencySnapshot() (allP50, allP99, getP50, getP99, setP50, setP99, existsP50, existsP99, queueWaitP50, queueWaitP99, allocDurP50, allocDurP99 int64, queueDepth, queueCap int) {
	allP50 = s.latency.all.p50().Microseconds()
	allP99 = s.latency.all.p99().Microseconds()
	getP50 = s.latency.get.p50().Microseconds()
	getP99 = s.latency.get.p99().Microseconds()
	setP50 = s.latency.set.p50().Microseconds()
	setP99 = s.latency.set.p99().Microseconds()
	existsP50 = s.latency.exists.p50().Microseconds()
	existsP99 = s.latency.exists.p99().Microseconds()
	queueWaitP50 = s.latency.queueWait.p50().Microseconds()
	queueWaitP99 = s.latency.queueWait.p99().Microseconds()
	allocDurP50 = s.latency.allocDur.p50().Microseconds()
	allocDurP99 = s.latency.allocDur.p99().Microseconds()
	queueDepth = len(s.ops)
	queueCap = cap(s.ops)
	return
}

// Allocator returns the shard's allocator (slab or offset).
// Safe to call from any goroutine (returns current allocator via atomic load).
func (s *Shard) Allocator() alloc.Allocator {
	return s.allocPtr.Load().a
}

// RegisterAllocListener adds a listener for allocator swap events.
func (s *Shard) RegisterAllocListener(l AllocChangeListener) {
	s.allocListeners = append(s.allocListeners, l)
}

// Config returns the shard's configuration.
func (s *Shard) Config() ShardConfig {
	return s.config
}

// IndexCount returns the number of live entries in this shard's index.
func (s *Shard) IndexCount() uint64 {
	return s.idx.Count()
}

// SizeDetectionSnapshot returns the current value-size auto-detection state.
// Safe to call from any goroutine (reads only; shard goroutine is the only writer).
func (s *Shard) SizeDetectionSnapshot() DetectionSnapshot {
	return s.sizeTracker.snapshot()
}

// DominantClassUtilization returns the utilization fraction of the slab class
// matching the auto-detected dominant value size. Returns (0, false) if detection
// hasn't completed or no matching class exists.
// Safe to call from any goroutine (reads atomic/snapshot state only).
func (s *Shard) DominantClassUtilization() (usedFrac float64, detected bool) {
	det := s.sizeTracker.snapshot()
	if det.Status != "detected" && det.Status != "rebuilt" {
		return 0, false
	}
	a := s.allocPtr.Load().a
	idx := findAllocClassIn(a, det.DominantValueSize)
	if idx < 0 {
		return 0, false
	}
	utils := a.ClassUtilizations()
	if idx >= len(utils) {
		return 0, false
	}
	u := utils[idx]
	if u.TotalSlots == 0 {
		return 0, true
	}
	return float64(u.UsedSlots) / float64(u.TotalSlots), true
}

// PressureSnap takes a snapshot of pressure counters and publishes it atomically.
// Must be called from a context that can read the shard's allocator (e.g. vacuum coordinator).
func (s *Shard) PressureSnap() *PressureSnapshot {
	return s.pressureTracker.snapshot(s.allocPtr.Load().a)
}

// LoadPressureSnapshot returns the most recently published pressure snapshot, or nil.
func (s *Shard) LoadPressureSnapshot() *PressureSnapshot {
	return s.pressureTracker.loadSnapshot()
}

// HugepageSummary delegates to the allocator's hugepage summary.
func (s *Shard) HugepageSummary() (gotHuge, thpHinted, regular int) {
	return s.allocPtr.Load().a.HugepageSummary()
}

const ttlSweepInterval = 10 * time.Second
const ttlSweepBatchSize = 1000

// Utilization warning thresholds â€” log warnings as shard approaches capacity.
const (
	utilizationWarnInterval = 30 * time.Second // how often to check
	utilizationWarn85       = 0.85
	utilizationWarn95       = 0.95
)

// checkUtilizationWarnings logs warnings when any slab class approaches capacity.
// Called periodically from the shard run loop. lastWarnLevel tracks the highest
// threshold already warned about to avoid log spam.
func (s *Shard) checkUtilizationWarnings(lastWarnLevel *float64) {
	a := s.allocPtr.Load().a
	utils := a.ClassUtilizations()
	var maxUtil float64
	var maxClass uint64
	var maxUsed, maxTotal uint64
	for _, u := range utils {
		if u.TotalSlots == 0 {
			continue
		}
		util := float64(u.UsedSlots) / float64(u.TotalSlots)
		if util > maxUtil {
			maxUtil = util
			maxClass = u.Size
			maxUsed = u.UsedSlots
			maxTotal = u.TotalSlots
		}
	}

	// Only warn when crossing a new threshold (and don't re-warn if level hasn't changed)
	if maxUtil >= utilizationWarn95 && *lastWarnLevel < utilizationWarn95 {
		log.Printf("[shard %d] CRITICAL: slab class %d bytes at %.0f%% capacity (%d/%d slots) â€” allocation failures imminent",
			s.id, maxClass, maxUtil*100, maxUsed, maxTotal)
		*lastWarnLevel = utilizationWarn95
	} else if maxUtil >= utilizationWarn85 && *lastWarnLevel < utilizationWarn85 {
		log.Printf("[shard %d] WARNING: slab class %d bytes at %.0f%% capacity (%d/%d slots) â€” approaching saturation",
			s.id, maxClass, maxUtil*100, maxUsed, maxTotal)
		*lastWarnLevel = utilizationWarn85
	} else if maxUtil < utilizationWarn85 && *lastWarnLevel > 0 {
		// Reset: utilization dropped back below warning threshold
		*lastWarnLevel = 0
	}
}

func (s *Shard) run() {
	if s.config.NUMANode >= 0 {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
	}
	defer close(s.done)

	// Pin to NUMA node if topology was detected
	if s.config.NUMANode >= 0 {
		if err := numa.PinThread(s.config.NUMANode); err != nil {
			log.Printf("[shard %d] NUMA thread pin to node %d failed: %v (continuing unpinned)", s.id, s.config.NUMANode, err)
		} else {
			log.Printf("[shard %d] NUMA thread pinned to node %d (sched_setaffinity)", s.id, s.config.NUMANode)
		}
	}

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	ttlTicker := time.NewTicker(s.ttlInterval)
	defer ttlTicker.Stop()

	utilTicker := time.NewTicker(utilizationWarnInterval)
	defer utilTicker.Stop()

	var lastWarnLevel float64

	for {
		// C1: Panic recovery â€” runLoopBody recovers from panics.
		// Tickers and LockOSThread stay in outer run() (not re-created on restart).
		exit := s.runLoopBody(ticker, ttlTicker, utilTicker, &lastWarnLevel)
		if exit {
			return
		}
		// runLoopBody returned due to panic recovery â€” check if cooldown is needed
		if !s.panicCooldownUntil.IsZero() && time.Now().Before(s.panicCooldownUntil) {
			remaining := time.Until(s.panicCooldownUntil)
			log.Printf("[shard %d] entering panic cooldown for %v â€” draining ops with error", s.id, remaining.Truncate(time.Second))
			// During cooldown: drain ops with error, but stay alive
			cooldownTimer := time.NewTimer(remaining)
			for {
				select {
				case op := <-s.ops:
					if op.Result != nil {
						op.Result <- OpResult{Err: errors.New("shard in panic cooldown")}
					}
				case <-cooldownTimer.C:
					goto resumed
				case <-s.quit:
					cooldownTimer.Stop()
					s.drainOps()
					s.allocPtr.Load().a.Close()
					return
				}
			}
		resumed:
			s.panicCooldownUntil = time.Time{} // clear cooldown
			s.metrics.SetShardHalted(0)
			log.Printf("[shard %d] panic cooldown expired â€” resuming normal operation", s.id)
		}
	}
}

// runLoopBody is the inner loop of run(), wrapped in panic recovery.
// Returns true if the shard should exit (quit signal), false if it panicked and should retry.
func (s *Shard) runLoopBody(ticker *time.Ticker, ttlTicker *time.Ticker, utilTicker *time.Ticker, lastWarnLevel *float64) (exit bool) {
	defer func() {
		if r := recover(); r != nil {
			// Log the panic with full stack
			buf := make([]byte, 4096)
			n := runtime.Stack(buf, false)
			log.Printf("[shard %d] PANIC RECOVERED: %v\n%s", s.id, r, buf[:n])
			s.metrics.IncrPanics()

			// Abort any active migration â€” state may be inconsistent.
			// Wrapped in inner recover: abortMigration notifies alloc listeners
			// which may panic during shutdown (e.g., closed cleanup queue).
			if s.migrate != nil {
				func() {
					defer func() {
						if r2 := recover(); r2 != nil {
							log.Printf("[shard %d] abortMigration during panic recovery also panicked: %v (clearing migration state)", s.id, r2)
							s.migrate = nil
						}
					}()
					s.abortMigration()
				}()
			}

			// Drain pending ops with error so callers don't block
			s.drainOpsWithError(errors.New("shard panic recovered"))

			// Check rapid-panic threshold â€” exponential cooldown instead of permanent halt
			if s.panics.record(time.Now()) {
				if s.panicCooldownDur == 0 {
					s.panicCooldownDur = 30 * time.Second
				} else {
					s.panicCooldownDur *= 2
					if s.panicCooldownDur > 300*time.Second {
						s.panicCooldownDur = 300 * time.Second
					}
				}
				s.panicCooldownUntil = time.Now().Add(s.panicCooldownDur)
				s.metrics.SetShardHalted(1)
				log.Printf("[shard %d] CRITICAL: %d panics in %v â€” entering cooldown for %v",
					s.id, panicWindowSize, panicWindowDur, s.panicCooldownDur)
			}

			exit = false // continue loop in run()
		}
	}()

	for {
		if s.migrate != nil {
			// During migration: drain pending ops (capped) then do one batch.
			drainBudget := s.config.MigrateDrainBudget
			if drainBudget <= 0 {
				drainBudget = 64
			}
			if s.migrate.drainBudget > 0 {
				drainBudget = s.migrate.drainBudget
			}
			for i := 0; i < drainBudget; i++ {
				select {
				case op := <-s.ops:
					opStart := time.Now()
					if !op.EnqueuedAt.IsZero() {
						s.latency.queueWait.record(opStart.Sub(op.EnqueuedAt))
					}
					s.handleOp(op)
					s.latency.record(op.Type, time.Since(opStart))
				case <-s.quit:
					s.abortMigration()
					s.drainOps()
					s.allocPtr.Load().a.Close()
					return true
				default:
					i = drainBudget // break
				}
			}
			// Non-blocking: service lease cleanup and TTL sweep so they aren't starved
			select {
			case <-ticker.C:
				s.leases.Cleanup()
			default:
			}
			select {
			case <-ttlTicker.C:
				s.sweepExpired()
			default:
			}
			select {
			case <-utilTicker.C:
				s.checkUtilizationWarnings(lastWarnLevel)
			default:
			}
			// Drain any stale async allocator that arrived while migration is already active
			select {
			case pa := <-s.pendingAlloc:
				log.Printf("[rebalance] shard %d: discarding async allocator (migration already active)", s.id)
				pa.newAlloc.Close()
				s.allocBuilding.Store(false)
				s.releaseMigrateSem()
			default:
			}
			done := s.migrateBatch()
			if done {
				s.finalizeMigration()
			}
			runtime.Gosched() // yield to scheduler between batches
			continue
		}
		// Normal select (blocking) â€” no migration in progress
		select {
		case op := <-s.ops:
			opStart := time.Now()
			if !op.EnqueuedAt.IsZero() {
				s.latency.queueWait.record(opStart.Sub(op.EnqueuedAt))
			}
			s.handleOp(op)
			s.latency.record(op.Type, time.Since(opStart))
		case pa := <-s.pendingAlloc:
			s.commitMigration(pa)
		case <-ticker.C:
			s.leases.Cleanup()
		case <-ttlTicker.C:
			s.sweepExpired()
		case <-utilTicker.C:
			s.checkUtilizationWarnings(lastWarnLevel)
		case <-s.quit:
			s.drainOps()
			select {
			case pa := <-s.pendingAlloc:
				pa.newAlloc.Close()
				s.allocBuilding.Store(false)
				s.releaseMigrateSem()
			default:
			}
			s.allocPtr.Load().a.Close()
			return true
		}
	}
}

// sweepExpired scans the index for expired entries and removes them.
// Uses adaptive batch size: doubles on backlog, gradually returns to default.
func (s *Shard) sweepExpired() {
	now := time.Now().UnixMilli()
	swept := 0
	batchLimit := s.ttlBatchSize
	s.idx.Iter(func(_ uint64, e index.Entry) bool {
		if swept >= batchLimit {
			return false
		}
		if e.TTL > 0 && now > e.TTL {
			s.freeEntryKey(e)
			s.freeEntryValue(e)
			s.idx.Delete(e.KeyHash, e.KeyLen)
			s.eviction.Remove(e.KeyHash)
			s.metrics.IncrTTLExpirations()
			swept++
		}
		return true
	})

	// L1: Adaptive sweep â€” if batch limit was hit (backlog), speed up
	if swept >= batchLimit {
		// Double batch size (cap 10,000) and halve interval (floor 1s)
		if s.ttlBatchSize < 10000 {
			s.ttlBatchSize *= 2
			if s.ttlBatchSize > 10000 {
				s.ttlBatchSize = 10000
			}
		}
		if s.ttlInterval > time.Second {
			s.ttlInterval /= 2
			if s.ttlInterval < time.Second {
				s.ttlInterval = time.Second
			}
		}
	} else {
		// Gradually return to defaults
		if s.ttlBatchSize > ttlSweepBatchSize {
			s.ttlBatchSize = (s.ttlBatchSize + ttlSweepBatchSize) / 2
		}
		if s.ttlInterval < ttlSweepInterval {
			s.ttlInterval = (s.ttlInterval + ttlSweepInterval) / 2
		}
	}

	// Clear OOM state if sweep freed entries
	if swept > 0 {
		s.allocSucceeded()
	}
}

// drainOps sends error results to any pending ops so callers don't block forever.
func (s *Shard) drainOps() {
	for {
		select {
		case op := <-s.ops:
			if op.Result != nil {
				op.Result <- OpResult{Err: errors.New("shard shutting down")}
			}
		default:
			return
		}
	}
}

// drainOpsWithError sends a specific error to all pending ops (used by panic recovery).
func (s *Shard) drainOpsWithError(err error) {
	for {
		select {
		case op := <-s.ops:
			if op.Result != nil {
				op.Result <- OpResult{Err: err}
			}
		default:
			return
		}
	}
}

// evictKey is called by eviction engine to actually free index + allocator resources.
// keyLen is provided by the eviction engine so Lookup is O(1) â€” no linear scan needed.
func (s *Shard) evictKey(keyHash uint64, keyLen uint16) {
	// Check lease protection
	if s.leases.IsProtected(keyHash) {
		s.metrics.IncrEvictionsLeaseSkip()
		return // can't evict leased keys
	}

	entry, _, found := s.idx.Lookup(keyHash, keyLen)
	if !found {
		return
	}

	// Record value-class eviction for pressure tracking before freeing
	valClassIdx := int(entry.ValueClassIdx)
	if entry.Flags&index.FlagHasClassIdx == 0 {
		ea := s.allocForEntry(entry)
		valClassIdx = findAllocClassIn(ea, uint64(entry.ValueLen))
	}
	if valClassIdx >= 0 {
		s.pressureTracker.recordEviction(valClassIdx)
	}

	s.freeEntryKey(entry)
	s.freeEntryValue(entry)
	s.idx.Delete(keyHash, entry.KeyLen)
	s.metrics.IncrEvictions()
}

// freeEntryKey frees the allocation for an entry's key using the correct allocator.
func (s *Shard) freeEntryKey(e index.Entry) {
	s.freeEntryKeyFrom(e, s.allocForEntry(e))
}

// freeEntryValue frees the allocation for an entry's value using the correct allocator.
func (s *Shard) freeEntryValue(e index.Entry) {
	s.freeEntryValueFrom(e, s.allocForEntry(e))
}

// allClassesSaturated returns true if ALL slab classes are above the given
// utilization threshold. When false, some classes have free space â€” the failure
// is likely fragmentation, not exhaustion.
func (s *Shard) allClassesSaturated(threshold float64) bool {
	a := s.allocPtr.Load().a
	utils := a.ClassUtilizations()
	for _, u := range utils {
		if u.TotalSlots == 0 {
			continue
		}
		if float64(u.UsedSlots)/float64(u.TotalSlots) < threshold {
			return false
		}
	}
	return true
}

// findAllocClass returns the index of the smallest class that fits size in the current allocator.
func (s *Shard) findAllocClass(size uint64) int {
	return findAllocClassIn(s.allocPtr.Load().a, size)
}

// findAllocClassIn delegates to the allocator's FindClass method.
func findAllocClassIn(a alloc.Allocator, size uint64) int {
	return a.FindClass(size)
}

// allocForEntry returns the correct allocator for an entry during migration.
// If no migration is active, returns the current allocator.
// During migration, entries with FlagMigrated are in newAlloc; others are in oldAlloc.
func (s *Shard) allocForEntry(e index.Entry) alloc.Allocator {
	if s.migrate != nil && e.Flags&index.FlagMigrated != 0 {
		return s.migrate.newAlloc
	}
	return s.allocPtr.Load().a
}

// allocForEntryOr is like allocForEntry but accepts a pre-loaded default allocator
// to avoid repeated atomic loads in batch loops.
func (s *Shard) allocForEntryOr(e index.Entry, defaultAlloc alloc.Allocator) alloc.Allocator {
	if s.migrate != nil && e.Flags&index.FlagMigrated != 0 {
		return s.migrate.newAlloc
	}
	return defaultAlloc
}

// freeEntryKeyFrom frees the key allocation using the given allocator.
func (s *Shard) freeEntryKeyFrom(e index.Entry, a alloc.Allocator) {
	if e.KeyOffset == 0 {
		return
	}
	cls := int(e.KeyClassIdx)
	if e.Flags&index.FlagHasClassIdx == 0 {
		cls = findAllocClassIn(a, uint64(e.KeyLen))
	}
	if cls < 0 || cls >= a.NumClasses() {
		log.Printf("WARN: shard %d: no alloc class for key size %d (orphaned slot)", s.id, e.KeyLen)
		return
	}
	a.Free(alloc.Allocation{ClassIdx: cls, Offset: e.KeyOffset, Size: a.ClassSize(cls)})
}

// freeEntryValueFrom frees the value allocation using the given allocator.
func (s *Shard) freeEntryValueFrom(e index.Entry, a alloc.Allocator) {
	if e.ValueOffset == 0 {
		return
	}
	cls := int(e.ValueClassIdx)
	if e.Flags&index.FlagHasClassIdx == 0 {
		cls = findAllocClassIn(a, uint64(e.ValueLen))
	}
	if cls < 0 || cls >= a.NumClasses() {
		log.Printf("WARN: shard %d: no alloc class for value size %d (orphaned slot)", s.id, e.ValueLen)
		return
	}
	a.Free(alloc.Allocation{ClassIdx: cls, Offset: e.ValueOffset, Size: a.ClassSize(cls)})
}
