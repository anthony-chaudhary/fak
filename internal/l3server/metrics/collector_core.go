package metrics

import (
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Collector aggregates metrics from all shards and connections.
type Collector struct {
	shardsMu             sync.RWMutex
	shards               []*ShardMetrics
	startup              *StartupState
	connReg              *ConnRegistry
	ClientStatsReg       *ClientStatsRegistry
	startedAt            time.Time
	SlabMetrics          SlabMetricsProvider     // set after shard allocation
	CacheStateProvider   CacheStateProvider      // set after shard allocation
	VacuumMetrics        VacuumMetricsProvider   // set after manager starts
	PressureMetrics      PressureMetricsProvider // set after manager starts (pressure rebalancing)
	PollerMetrics        PollerMetricsProvider   // set after RDMA servers start
	RDMAReadMetrics      RDMAReadMetricsProvider // set after RDMA servers start
	InflightOpsFunc      func() int64            // set after transport servers start
	SystemHealth         SystemHealthProvider    // set after health monitor starts
	OpLatencyMetrics     OpLatencyProvider       // set after shard allocation
	ShardPressureMetrics ShardPressureProvider   // set after shard allocation
	ConnBufBytesFunc     func() int64            // RDMA connection buffer bytes (from A2)
	MaxKeysPerShard      uint64                  // configured max_keys per shard (for capacity metrics)
	NumShards            int                     // number of shards (for capacity metrics)
	ReplicationQueue     func() (depth, cap int) // cluster replication queue metrics (nil = no cluster)

	// Slab-aware readiness: returns per-shard detection status strings.
	// Set after shard allocation. Returns nil if not wired up.
	ShardDetectionFunc func() []string

	// Preflight metrics (point-in-time values captured once at startup)
	PreflightMemAvailGB  float64
	PreflightSwapUsedPct float64
	PreflightStaleProcs  int

	lastChurnWarn time.Time // rate-limit churn warnings

	// M4: Shutdown guard â€” 503 during shutdown prevents partial-state scrapes
	shuttingDown atomic.Bool
}

// NewCollector creates a metrics collector. Shard metrics are nil until SetShards is called.
func NewCollector(startup *StartupState, connReg *ConnRegistry, clientStatsReg *ClientStatsRegistry, startedAt time.Time) *Collector {
	return &Collector{startup: startup, connReg: connReg, ClientStatsReg: clientStatsReg, startedAt: startedAt}
}

// SetShards registers shard metrics once allocation is complete.
func (c *Collector) SetShards(shards []*ShardMetrics) {
	c.shardsMu.Lock()
	c.shards = shards
	c.shardsMu.Unlock()
}

// shardTotals holds aggregated counters across all shards.
type shardTotals struct {
	gets                   int64
	sets                   int64
	deletes                int64
	exists                 int64
	existsHits             int64
	existsMisses           int64
	hits                   int64
	misses                 int64
	evictions              int64
	ttlExpirations         int64
	evictionsKeyPressure   int64
	evictionsValuePressure int64
	evictionsFailed        int64
	evictionsLeaseSkip     int64
	evictionsRebalance     int64
	promotions             int64
	oomRejections          int64
	opsDropped             int64
	bytesIn                int64
	bytesOut               int64
	rdmaReadBytesOut       int64
	keyBytesIn             int64
	valueBytesIn           int64
	migrationsInProgress   int32
	panics                 int64
	circuitTrips           int64
	shardHalted            int32
}

// aggregateShards computes totals across all shards. Returns nil shardTotals if shards is nil.
func aggregateShards(shards []*ShardMetrics) *shardTotals {
	if shards == nil {
		return nil
	}
	t := &shardTotals{}
	for _, m := range shards {
		t.gets += m.Gets()
		t.sets += m.Sets()
		t.deletes += m.Deletes()
		t.exists += m.Exists()
		t.existsHits += m.ExistsHits()
		t.existsMisses += m.ExistsMisses()
		t.hits += m.Hits()
		t.misses += m.Misses()
		t.evictions += m.Evictions()
		t.ttlExpirations += m.TTLExpirations()
		t.bytesIn += m.BytesIn()
		t.bytesOut += m.BytesOut()
		t.rdmaReadBytesOut += m.RDMAReadBytesOut()
		t.keyBytesIn += m.KeyBytesIn()
		t.valueBytesIn += m.ValueBytesIn()
		t.evictionsKeyPressure += m.EvictionsKeyPressure()
		t.evictionsValuePressure += m.EvictionsValuePressure()
		t.evictionsFailed += m.EvictionsFailed()
		t.evictionsLeaseSkip += m.EvictionsLeaseSkip()
		t.evictionsRebalance += m.EvictionsRebalance()
		t.promotions += m.Promotions()
		t.oomRejections += m.OOMRejections()
		t.opsDropped += m.OpsDropped()
		t.migrationsInProgress += m.MigrationActive()
		t.panics += m.Panics()
		t.circuitTrips += m.CircuitTrips()
		t.shardHalted += m.ShardHalted()
	}
	return t
}

// SetShuttingDown handles Prometheus scrape requests.
// SetShuttingDown marks the collector as shutting down. Future scrapes return 503.
func (c *Collector) SetShuttingDown() { c.shuttingDown.Store(true) }

func (c *Collector) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// M4: Reject scrapes during shutdown to prevent partial-state reads
	if c.shuttingDown.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte("# L3 server shutting down\n"))
		return
	}

	var b strings.Builder

	c.shardsMu.RLock()
	shards := c.shards
	c.shardsMu.RUnlock()

	totals := aggregateShards(shards)
	ltTotals := aggregateLifetimeShards(shards)

	c.writeServerState(&b, ltTotals, totals, shards)
	c.writeAggregateOps(&b, ltTotals)
	c.writeEviction(&b, ltTotals)
	c.writePerShardOps(&b, shards, ltTotals)
	c.writeOpLatency(&b)
	c.writeShardPressure(&b)
	c.writeMigration(&b, shards, ltTotals)
	c.writeWireProtocol(&b, ltTotals)
	c.writeRDMA(&b)
	c.writeSystemHealth(&b)
	c.writeSlabAllocator(&b)
	c.writePerConnection(&b)
	c.writeVacuum(&b)
	c.writeStartup(&b)
	c.writeClientStats(&b)
	c.writeBottleneck(&b)

	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	w.Write([]byte(b.String()))
}
