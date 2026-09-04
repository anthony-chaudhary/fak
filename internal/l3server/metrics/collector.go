package metrics

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// Collector aggregates metrics from all shards and connections.
type Collector struct {
	shardsMu           sync.RWMutex
	shards             []*ShardMetrics
	startup            *StartupState
	connReg            *ConnRegistry
	ClientStatsReg     *ClientStatsRegistry
	startedAt          time.Time
	SlabMetrics        SlabMetricsProvider    // set after shard allocation
	CacheStateProvider CacheStateProvider     // set after shard allocation
	VacuumMetrics      VacuumMetricsProvider  // set after manager starts
	PressureMetrics    PressureMetricsProvider // set after manager starts (pressure rebalancing)
	PollerMetrics      PollerMetricsProvider   // set after RDMA servers start
	RDMAReadMetrics    RDMAReadMetricsProvider // set after RDMA servers start
	InflightOpsFunc    func() int64            // set after transport servers start
	SystemHealth       SystemHealthProvider    // set after health monitor starts
	OpLatencyMetrics      OpLatencyProvider          // set after shard allocation
	ShardPressureMetrics  ShardPressureProvider      // set after shard allocation
	ConnBufBytesFunc      func() int64            // RDMA connection buffer bytes (from A2)
	MaxKeysPerShard    uint64                   // configured max_keys per shard (for capacity metrics)
	NumShards          int                      // number of shards (for capacity metrics)
	ReplicationQueue   func() (depth, cap int)  // cluster replication queue metrics (nil = no cluster)

	// Slab-aware readiness: returns per-shard detection status strings.
	// Set after shard allocation. Returns nil if not wired up.
	ShardDetectionFunc func() []string

	// Preflight metrics (point-in-time values captured once at startup)
	PreflightMemAvailGB    float64
	PreflightSwapUsedPct   float64
	PreflightStaleProcs    int

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
	gets                     int64
	sets                     int64
	deletes                  int64
	exists                   int64
	existsHits               int64
	existsMisses             int64
	hits                     int64
	misses                   int64
	evictions                int64
	ttlExpirations           int64
	evictionsKeyPressure     int64
	evictionsValuePressure   int64
	evictionsFailed          int64
	evictionsLeaseSkip       int64
	evictionsRebalance       int64
	promotions               int64
	oomRejections            int64
	opsDropped               int64
	bytesIn                  int64
	bytesOut                 int64
	rdmaReadBytesOut         int64
	keyBytesIn               int64
	valueBytesIn             int64
	migrationsInProgress     int32
	panics                   int64
	circuitTrips             int64
	shardHalted              int32
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

// aggregateLifetimeShards computes lifetime (monotonic) totals across all shards.
// Unlike aggregateShards, these values survive FLUSH and are suitable for Prometheus counters.
func aggregateLifetimeShards(shards []*ShardMetrics) *shardTotals {
	if shards == nil {
		return nil
	}
	t := &shardTotals{}
	for _, m := range shards {
		t.gets += m.LifetimeGets()
		t.sets += m.LifetimeSets()
		t.deletes += m.LifetimeDeletes()
		t.exists += m.LifetimeExists()
		t.existsHits += m.LifetimeExistsHits()
		t.existsMisses += m.LifetimeExistsMisses()
		t.hits += m.LifetimeHits()
		t.misses += m.LifetimeMisses()
		t.evictions += m.LifetimeEvictions()
		t.ttlExpirations += m.LifetimeTTLExpirations()
		t.bytesIn += m.LifetimeBytesIn()
		t.bytesOut += m.LifetimeBytesOut()
		t.rdmaReadBytesOut += m.LifetimeRDMAReadBytesOut()
		t.keyBytesIn += m.LifetimeKeyBytesIn()
		t.valueBytesIn += m.LifetimeValueBytesIn()
		t.evictionsKeyPressure += m.LifetimeEvictionsKeyPressure()
		t.evictionsValuePressure += m.LifetimeEvictionsValuePressure()
		t.evictionsFailed += m.LifetimeEvictionsFailed()
		t.evictionsLeaseSkip += m.LifetimeEvictionsLeaseSkip()
		t.evictionsRebalance += m.LifetimeEvictionsRebalance()
		t.promotions += m.LifetimePromotions()
		t.oomRejections += m.LifetimeOOMRejections()
		t.opsDropped += m.OpsDropped()             // already lifetime â€” never reset
		t.migrationsInProgress += m.MigrationActive() // gauge, not a counter
		t.panics += m.Panics()             // logically lifetime â€” never reset
		t.circuitTrips += m.CircuitTrips() // logically lifetime â€” never reset
		t.shardHalted += m.ShardHalted()   // gauge, not a counter
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

func (c *Collector) writeServerState(b *strings.Builder, ltTotals *shardTotals, epochTotals *shardTotals, shards []*ShardMetrics) {
	b.WriteString("\n# ---- server state ------------------------------------------------\n\n")

	ready := int32(0)
	if c.startup.Ready.Load() {
		ready = 1
	}
	b.WriteString("# HELP l3_server_ready Whether the server has completed startup (0/1).\n")
	b.WriteString("# TYPE l3_server_ready gauge\n")
	fmt.Fprintf(b, "l3_server_ready %d\n", ready)

	b.WriteString("\n# HELP l3_server_uptime_seconds Seconds since server start.\n")
	b.WriteString("# TYPE l3_server_uptime_seconds gauge\n")
	fmt.Fprintf(b, "l3_server_uptime_seconds %.0f\n", time.Since(c.startedAt).Seconds())

	if c.CacheStateProvider != nil {
		allocBytes, entries := c.CacheStateProvider()
		activeGB := float64(allocBytes) / (1 << 30)

		b.WriteString("\n# HELP l3_server_active_gb Currently allocated memory in GiB.\n")
		b.WriteString("# TYPE l3_server_active_gb gauge\n")
		fmt.Fprintf(b, "l3_server_active_gb %.6f\n", activeGB)

		b.WriteString("\n# HELP l3_server_entries Total live key count across all shards.\n")
		b.WriteString("# TYPE l3_server_entries gauge\n")
		fmt.Fprintf(b, "l3_server_entries %d\n", entries)

		if c.MaxKeysPerShard > 0 && c.NumShards > 0 {
			maxKeys := int64(c.MaxKeysPerShard) * int64(c.NumShards)
			evictionCap := maxKeys * index.MaxLoadNumerator / index.MaxLoadDenominator
			utilPct := float64(0)
			if evictionCap > 0 {
				utilPct = float64(entries) / float64(evictionCap) * 100
			}
			b.WriteString("\n# HELP l3_server_max_keys Configured maximum keys (max_keys * num_shards).\n")
			b.WriteString("# TYPE l3_server_max_keys gauge\n")
			fmt.Fprintf(b, "l3_server_max_keys %d\n", maxKeys)

			b.WriteString("\n# HELP l3_server_eviction_cap Effective key capacity after 7/8 Swiss table load factor.\n")
			b.WriteString("# TYPE l3_server_eviction_cap gauge\n")
			fmt.Fprintf(b, "l3_server_eviction_cap %d\n", evictionCap)

			b.WriteString("\n# HELP l3_server_key_utilization_percent Current entries as percentage of eviction cap.\n")
			b.WriteString("# TYPE l3_server_key_utilization_percent gauge\n")
			fmt.Fprintf(b, "l3_server_key_utilization_percent %.2f\n", utilPct)
		}

		// Churn ratio (active vs ingested) â€” uses epoch totals for accurate churn detection.
		var epochBytesIn int64
		if epochTotals != nil {
			epochBytesIn = epochTotals.bytesIn
		}
		const (
			churnThreshold  = 0.20
			minIngestedGB   = 1.0
			churnWarnPeriod = 5 * time.Minute
		)
		ingestedGB := float64(epochBytesIn) / (1 << 30)
		if ingestedGB >= minIngestedGB && activeGB < ingestedGB*churnThreshold {
			ratio := activeGB / ingestedGB * 100
			b.WriteString("\n# HELP l3_eviction_churn_ratio Active GB as percentage of ingested GB (low = heavy churn).\n")
			b.WriteString("# TYPE l3_eviction_churn_ratio gauge\n")
			fmt.Fprintf(b, "l3_eviction_churn_ratio %.2f\n", ratio)
			now := time.Now()
			if now.Sub(c.lastChurnWarn) >= churnWarnPeriod {
				c.lastChurnWarn = now
				log.Printf("[l3server] WARN: active %.2f GB is only %.1f%% of %.2f GB ingested â€” cache may be undersized for this workload",
					activeGB, ratio, ingestedGB)
			}
		} else if ingestedGB > 0 {
			b.WriteString("\n# HELP l3_eviction_churn_ratio Active GB as percentage of ingested GB (low = heavy churn).\n")
			b.WriteString("# TYPE l3_eviction_churn_ratio gauge\n")
			fmt.Fprintf(b, "l3_eviction_churn_ratio %.2f\n", activeGB/ingestedGB*100)
		}
	}

	b.WriteString("\n# HELP l3_server_active_connections Number of active client connections.\n")
	b.WriteString("# TYPE l3_server_active_connections gauge\n")
	fmt.Fprintf(b, "l3_server_active_connections %d\n", c.connReg.ActiveCount())

	if c.InflightOpsFunc != nil {
		b.WriteString("\n# HELP l3_server_inflight_ops Number of operations currently being processed.\n")
		b.WriteString("# TYPE l3_server_inflight_ops gauge\n")
		fmt.Fprintf(b, "l3_server_inflight_ops %d\n", c.InflightOpsFunc())
	}

	if ltTotals != nil {
		b.WriteString("\n# HELP l3_server_payload_gb_in Total payload bytes ingested (key+value on SET) in GiB.\n")
		b.WriteString("# TYPE l3_server_payload_gb_in counter\n")
		fmt.Fprintf(b, "l3_server_payload_gb_in %.6f\n", float64(ltTotals.bytesIn)/(1<<30))

		b.WriteString("\n# HELP l3_server_payload_gb_out Total payload bytes served (value on GET hit) in GiB.\n")
		b.WriteString("# TYPE l3_server_payload_gb_out counter\n")
		fmt.Fprintf(b, "l3_server_payload_gb_out %.6f\n", float64(ltTotals.bytesOut)/(1<<30))

		b.WriteString("\n# HELP l3_server_rdma_read_gb_out Subset of payload_gb_out served via RDMA Read in GiB.\n")
		b.WriteString("# TYPE l3_server_rdma_read_gb_out counter\n")
		fmt.Fprintf(b, "l3_server_rdma_read_gb_out %.6f\n", float64(ltTotals.rdmaReadBytesOut)/(1<<30))

		if ltTotals.sets > 0 {
			b.WriteString("\n# HELP l3_server_avg_key_bytes Average key size in bytes.\n")
			b.WriteString("# TYPE l3_server_avg_key_bytes gauge\n")
			fmt.Fprintf(b, "l3_server_avg_key_bytes %.2f\n", float64(ltTotals.keyBytesIn)/float64(ltTotals.sets))

			b.WriteString("\n# HELP l3_server_avg_value_bytes Average value size in bytes.\n")
			b.WriteString("# TYPE l3_server_avg_value_bytes gauge\n")
			fmt.Fprintf(b, "l3_server_avg_value_bytes %.2f\n", float64(ltTotals.valueBytesIn)/float64(ltTotals.sets))
		}
	}
}

func (c *Collector) writeAggregateOps(b *strings.Builder, totals *shardTotals) {
	if totals == nil {
		return
	}

	b.WriteString("\n\n# ---- aggregate operations ----------------------------------------\n\n")

	b.WriteString("# HELP l3_ops_gets_total Total GET operations across all shards.\n")
	b.WriteString("# TYPE l3_ops_gets_total counter\n")
	fmt.Fprintf(b, "l3_ops_gets_total %d\n", totals.gets)

	b.WriteString("\n# HELP l3_ops_sets_total Total SET operations across all shards.\n")
	b.WriteString("# TYPE l3_ops_sets_total counter\n")
	fmt.Fprintf(b, "l3_ops_sets_total %d\n", totals.sets)

	b.WriteString("\n# HELP l3_ops_deletes_total Total DELETE operations across all shards.\n")
	b.WriteString("# TYPE l3_ops_deletes_total counter\n")
	fmt.Fprintf(b, "l3_ops_deletes_total %d\n", totals.deletes)

	b.WriteString("\n# HELP l3_ops_exists_total Total EXISTS operations across all shards.\n")
	b.WriteString("# TYPE l3_ops_exists_total counter\n")
	fmt.Fprintf(b, "l3_ops_exists_total %d\n", totals.exists)

	b.WriteString("\n# HELP l3_ops_hits_total Total cache hits on GET.\n")
	b.WriteString("# TYPE l3_ops_hits_total counter\n")
	fmt.Fprintf(b, "l3_ops_hits_total %d\n", totals.hits)

	b.WriteString("\n# HELP l3_ops_misses_total Total cache misses on GET.\n")
	b.WriteString("# TYPE l3_ops_misses_total counter\n")
	fmt.Fprintf(b, "l3_ops_misses_total %d\n", totals.misses)

	b.WriteString("\n# HELP l3_ops_ttl_expirations_total Total entries removed due to TTL expiry.\n")
	b.WriteString("# TYPE l3_ops_ttl_expirations_total counter\n")
	fmt.Fprintf(b, "l3_ops_ttl_expirations_total %d\n", totals.ttlExpirations)

	hitRate := float64(0)
	if totals.gets > 0 {
		hitRate = float64(totals.hits) / float64(totals.gets) * 100
	}
	b.WriteString("\n# HELP l3_ops_hit_rate_percent GET hit rate (0-100).\n")
	b.WriteString("# TYPE l3_ops_hit_rate_percent gauge\n")
	fmt.Fprintf(b, "l3_ops_hit_rate_percent %.2f\n", hitRate)

	existsHitRate := float64(0)
	if totals.exists > 0 {
		existsHitRate = float64(totals.existsHits) / float64(totals.exists) * 100
	}
	b.WriteString("\n# HELP l3_ops_exists_hit_rate_percent EXISTS hit rate (0-100).\n")
	b.WriteString("# TYPE l3_ops_exists_hit_rate_percent gauge\n")
	fmt.Fprintf(b, "l3_ops_exists_hit_rate_percent %.2f\n", existsHitRate)

	b.WriteString("\n# HELP l3_ops_dropped_total Total ops dropped due to full shard channels.\n")
	b.WriteString("# TYPE l3_ops_dropped_total counter\n")
	fmt.Fprintf(b, "l3_ops_dropped_total %d\n", totals.opsDropped)

	b.WriteString("\n# HELP l3_ops_batch_timeouts_total Total batch dispatch timeouts.\n")
	b.WriteString("# TYPE l3_ops_batch_timeouts_total counter\n")
	fmt.Fprintf(b, "l3_ops_batch_timeouts_total %d\n", BatchTimeouts())

	b.WriteString("\n# HELP l3_rdma_connection_rejections_total Total RDMA connections rejected at limit.\n")
	b.WriteString("# TYPE l3_rdma_connection_rejections_total counter\n")
	fmt.Fprintf(b, "l3_rdma_connection_rejections_total %d\n", RDMAConnectionRejections())

	b.WriteString("\n# HELP l3_wire_tcp_connection_rejections_total Total TCP connections rejected at limit.\n")
	b.WriteString("# TYPE l3_wire_tcp_connection_rejections_total counter\n")
	fmt.Fprintf(b, "l3_wire_tcp_connection_rejections_total %d\n", TCPConnectionRejections())

	// --- Panic recovery metrics ---

	b.WriteString("\n# HELP l3_shard_panics_total Total panics recovered in shard run loops.\n")
	b.WriteString("# TYPE l3_shard_panics_total counter\n")
	fmt.Fprintf(b, "l3_shard_panics_total %d\n", totals.panics)

	b.WriteString("\n# HELP l3_shard_circuit_trips_total Total shard circuit breaker activations.\n")
	b.WriteString("# TYPE l3_shard_circuit_trips_total counter\n")
	fmt.Fprintf(b, "l3_shard_circuit_trips_total %d\n", totals.circuitTrips)


	b.WriteString("\n# HELP l3_server_tcp_handler_panics_total Total panics in TCP connection handlers.\n")
	b.WriteString("# TYPE l3_server_tcp_handler_panics_total counter\n")
	fmt.Fprintf(b, "l3_server_tcp_handler_panics_total %d\n", TCPHandlerPanics())

	b.WriteString("\n# HELP l3_rdma_dispatch_panics_total Total panics in RDMA dispatch workers.\n")
	b.WriteString("# TYPE l3_rdma_dispatch_panics_total counter\n")
	fmt.Fprintf(b, "l3_rdma_dispatch_panics_total %d\n", RDMADispatchPanics())

	b.WriteString("\n# HELP l3_rdma_poller_panics_total Total panics in RDMA CQ poller.\n")
	b.WriteString("# TYPE l3_rdma_poller_panics_total counter\n")
	fmt.Fprintf(b, "l3_rdma_poller_panics_total %d\n", RDMAPollerPanics())
}

func (c *Collector) writeEviction(b *strings.Builder, totals *shardTotals) {
	if totals == nil {
		return
	}

	b.WriteString("\n\n# ---- eviction ----------------------------------------------------\n\n")

	b.WriteString("# HELP l3_eviction_total Total evictions across all shards.\n")
	b.WriteString("# TYPE l3_eviction_total counter\n")
	fmt.Fprintf(b, "l3_eviction_total %d\n", totals.evictions)

	b.WriteString("\n# HELP l3_eviction_key_pressure_total Evictions triggered during key buffer allocation.\n")
	b.WriteString("# TYPE l3_eviction_key_pressure_total counter\n")
	fmt.Fprintf(b, "l3_eviction_key_pressure_total %d\n", totals.evictionsKeyPressure)

	b.WriteString("\n# HELP l3_eviction_value_pressure_total Evictions triggered during value buffer allocation.\n")
	b.WriteString("# TYPE l3_eviction_value_pressure_total counter\n")
	fmt.Fprintf(b, "l3_eviction_value_pressure_total %d\n", totals.evictionsValuePressure)

	b.WriteString("\n# HELP l3_eviction_failed_total Eviction attempts that failed.\n")
	b.WriteString("# TYPE l3_eviction_failed_total counter\n")
	fmt.Fprintf(b, "l3_eviction_failed_total %d\n", totals.evictionsFailed)

	b.WriteString("\n# HELP l3_eviction_lease_skip_total Evictions skipped due to active leases.\n")
	b.WriteString("# TYPE l3_eviction_lease_skip_total counter\n")
	fmt.Fprintf(b, "l3_eviction_lease_skip_total %d\n", totals.evictionsLeaseSkip)

	b.WriteString("\n# HELP l3_eviction_rebalance_total Evictions from ZeroLatencyBalance abort (non-migrated entries).\n")
	b.WriteString("# TYPE l3_eviction_rebalance_total counter\n")
	fmt.Fprintf(b, "l3_eviction_rebalance_total %d\n", totals.evictionsRebalance)

	totalEvictionsOrganic := totals.evictions - totals.evictionsRebalance
	if totalEvictionsOrganic < 0 {
		totalEvictionsOrganic = 0
	}
	b.WriteString("\n# HELP l3_eviction_organic_total Organic evictions (total minus rebalance abort).\n")
	b.WriteString("# TYPE l3_eviction_organic_total counter\n")
	fmt.Fprintf(b, "l3_eviction_organic_total %d\n", totalEvictionsOrganic)

	b.WriteString("\n# HELP l3_ops_promotions_total Values promoted to larger slab class to avoid eviction.\n")
	b.WriteString("# TYPE l3_ops_promotions_total counter\n")
	fmt.Fprintf(b, "l3_ops_promotions_total %d\n", totals.promotions)

	b.WriteString("\n# HELP l3_ops_oom_rejections_total SETs rejected due to memory pressure (no space after eviction retries).\n")
	b.WriteString("# TYPE l3_ops_oom_rejections_total counter\n")
	fmt.Fprintf(b, "l3_ops_oom_rejections_total %d\n", totals.oomRejections)

	evictionRate := float64(0)
	if totals.sets > 0 {
		evictionRate = float64(totals.evictions) / float64(totals.sets) * 100
	}
	b.WriteString("\n# HELP l3_eviction_rate_percent Evictions as percentage of SETs.\n")
	b.WriteString("# TYPE l3_eviction_rate_percent gauge\n")
	fmt.Fprintf(b, "l3_eviction_rate_percent %.2f\n", evictionRate)

	evictionFailRate := float64(0)
	if totals.evictionsKeyPressure+totals.evictionsValuePressure+totals.evictionsFailed > 0 {
		evictionFailRate = float64(totals.evictionsFailed) / float64(totals.evictionsKeyPressure+totals.evictionsValuePressure+totals.evictionsFailed) * 100
	}
	b.WriteString("\n# HELP l3_eviction_fail_rate_percent Failed evictions as percentage of all pressure evictions.\n")
	b.WriteString("# TYPE l3_eviction_fail_rate_percent gauge\n")
	fmt.Fprintf(b, "l3_eviction_fail_rate_percent %.2f\n", evictionFailRate)
}

func (c *Collector) writePerShardOps(b *strings.Builder, shards []*ShardMetrics, totals *shardTotals) {
	if totals == nil {
		return
	}

	var shardBuf strings.Builder
	for _, m := range shards {
		id := m.shardID
		fmt.Fprintf(&shardBuf, "l3_shard_gets{shard=\"%d\"} %d\n", id, m.LifetimeGets())
		fmt.Fprintf(&shardBuf, "l3_shard_sets{shard=\"%d\"} %d\n", id, m.LifetimeSets())
		fmt.Fprintf(&shardBuf, "l3_shard_evictions{shard=\"%d\"} %d\n", id, m.LifetimeEvictions())
		fmt.Fprintf(&shardBuf, "l3_shard_bytes_in{shard=\"%d\"} %d\n", id, m.LifetimeBytesIn())
		fmt.Fprintf(&shardBuf, "l3_shard_evictions_key_pressure{shard=\"%d\"} %d\n", id, m.LifetimeEvictionsKeyPressure())
		fmt.Fprintf(&shardBuf, "l3_shard_evictions_value_pressure{shard=\"%d\"} %d\n", id, m.LifetimeEvictionsValuePressure())
		fmt.Fprintf(&shardBuf, "l3_shard_evictions_failed{shard=\"%d\"} %d\n", id, m.LifetimeEvictionsFailed())
		fmt.Fprintf(&shardBuf, "l3_shard_evictions_lease_skip{shard=\"%d\"} %d\n", id, m.LifetimeEvictionsLeaseSkip())
		fmt.Fprintf(&shardBuf, "l3_shard_evictions_rebalance{shard=\"%d\"} %d\n", id, m.LifetimeEvictionsRebalance())
		fmt.Fprintf(&shardBuf, "l3_shard_ops_dropped{shard=\"%d\"} %d\n", id, m.OpsDropped())
		fmt.Fprintf(&shardBuf, "l3_shard_oom_rejections{shard=\"%d\"} %d\n", id, m.LifetimeOOMRejections())
		fmt.Fprintf(&shardBuf, "l3_shard_memory_pressure{shard=\"%d\"} %d\n", id, m.MemoryPressure())
		fmt.Fprintf(&shardBuf, "l3_shard_halted{shard=\"%d\"} %d\n", id, m.ShardHalted())
	}

	b.WriteString("\n\n# ---- per-shard operations ----------------------------------------\n\n")

	b.WriteString("# HELP l3_shard_gets Per-shard GET operations.\n")
	b.WriteString("# TYPE l3_shard_gets counter\n")
	b.WriteString("# HELP l3_shard_sets Per-shard SET operations.\n")
	b.WriteString("# TYPE l3_shard_sets counter\n")
	b.WriteString("# HELP l3_shard_evictions Per-shard evictions.\n")
	b.WriteString("# TYPE l3_shard_evictions counter\n")
	b.WriteString("# HELP l3_shard_bytes_in Per-shard payload bytes ingested.\n")
	b.WriteString("# TYPE l3_shard_bytes_in counter\n")
	b.WriteString("# HELP l3_shard_evictions_key_pressure Per-shard evictions from key pressure.\n")
	b.WriteString("# TYPE l3_shard_evictions_key_pressure counter\n")
	b.WriteString("# HELP l3_shard_evictions_value_pressure Per-shard evictions from value pressure.\n")
	b.WriteString("# TYPE l3_shard_evictions_value_pressure counter\n")
	b.WriteString("# HELP l3_shard_evictions_failed Per-shard failed eviction attempts.\n")
	b.WriteString("# TYPE l3_shard_evictions_failed counter\n")
	b.WriteString("# HELP l3_shard_evictions_lease_skip Per-shard evictions skipped due to leases.\n")
	b.WriteString("# TYPE l3_shard_evictions_lease_skip counter\n")
	b.WriteString("# HELP l3_shard_evictions_rebalance Per-shard evictions from rebalance abort.\n")
	b.WriteString("# TYPE l3_shard_evictions_rebalance counter\n")
	b.WriteString("# HELP l3_shard_ops_dropped Per-shard ops dropped due to full channel.\n")
	b.WriteString("# TYPE l3_shard_ops_dropped counter\n")
	b.WriteString("# HELP l3_shard_oom_rejections Per-shard SETs rejected due to memory pressure.\n")
	b.WriteString("# TYPE l3_shard_oom_rejections counter\n")
	b.WriteString("# HELP l3_shard_memory_pressure Per-shard memory pressure state (1=OOM rejection active, 0=normal).\n")
	b.WriteString("# TYPE l3_shard_memory_pressure gauge\n")
	b.WriteString("# HELP l3_shard_halted Per-shard panic cooldown state (1=in cooldown, 0=normal).\n")
	b.WriteString("# TYPE l3_shard_halted gauge\n")
	b.WriteString(shardBuf.String())
}

func (c *Collector) writeOpLatency(b *strings.Builder) {
	if c.OpLatencyMetrics == nil {
		return
	}
	snaps := c.OpLatencyMetrics()
	if len(snaps) == 0 {
		return
	}

	b.WriteString("\n\n# ---- op latency --------------------------------------------------\n\n")

	// Per-shard per-op latency
	b.WriteString("# HELP l3_ops_latency_us Per-shard per-op-type latency in microseconds.\n")
	b.WriteString("# TYPE l3_ops_latency_us gauge\n")

	type opCat struct {
		name string
		p50  func(OpLatencySnapshot) int64
		p99  func(OpLatencySnapshot) int64
	}
	cats := []opCat{
		{"all", func(s OpLatencySnapshot) int64 { return s.AllP50Us }, func(s OpLatencySnapshot) int64 { return s.AllP99Us }},
		{"get", func(s OpLatencySnapshot) int64 { return s.GetP50Us }, func(s OpLatencySnapshot) int64 { return s.GetP99Us }},
		{"set", func(s OpLatencySnapshot) int64 { return s.SetP50Us }, func(s OpLatencySnapshot) int64 { return s.SetP99Us }},
		{"exists", func(s OpLatencySnapshot) int64 { return s.ExistsP50Us }, func(s OpLatencySnapshot) int64 { return s.ExistsP99Us }},
		{"queue_wait", func(s OpLatencySnapshot) int64 { return s.QueueWaitP50Us }, func(s OpLatencySnapshot) int64 { return s.QueueWaitP99Us }},
		{"alloc_dur", func(s OpLatencySnapshot) int64 { return s.AllocDurP50Us }, func(s OpLatencySnapshot) int64 { return s.AllocDurP99Us }},
	}

	// Aggregate: avg p50 and max p99 across shards
	type aggLatency struct {
		sumP50 int64
		maxP99 int64
		count  int
	}
	agg := make([]aggLatency, len(cats))

	for _, snap := range snaps {
		for ci, cat := range cats {
			p50 := cat.p50(snap)
			p99 := cat.p99(snap)
			fmt.Fprintf(b, "l3_ops_latency_us{shard=\"%d\",op=\"%s\",quantile=\"0.5\"} %d\n", snap.ShardID, cat.name, p50)
			fmt.Fprintf(b, "l3_ops_latency_us{shard=\"%d\",op=\"%s\",quantile=\"0.99\"} %d\n", snap.ShardID, cat.name, p99)
			agg[ci].sumP50 += p50
			agg[ci].count++
			if p99 > agg[ci].maxP99 {
				agg[ci].maxP99 = p99
			}
		}
	}

	// Aggregate metrics
	b.WriteString("\n# HELP l3_ops_latency_avg_p50_us Average p50 latency across all shards (us).\n")
	b.WriteString("# TYPE l3_ops_latency_avg_p50_us gauge\n")
	for ci, cat := range cats {
		avgP50 := int64(0)
		if agg[ci].count > 0 {
			avgP50 = agg[ci].sumP50 / int64(agg[ci].count)
		}
		fmt.Fprintf(b, "l3_ops_latency_avg_p50_us{op=\"%s\"} %d\n", cat.name, avgP50)
	}

	b.WriteString("\n# HELP l3_ops_latency_max_p99_us Max p99 latency across all shards (us, worst-case).\n")
	b.WriteString("# TYPE l3_ops_latency_max_p99_us gauge\n")
	for ci, cat := range cats {
		fmt.Fprintf(b, "l3_ops_latency_max_p99_us{op=\"%s\"} %d\n", cat.name, agg[ci].maxP99)
	}

	// Queue depth and capacity
	b.WriteString("\n# HELP l3_shard_queue_depth Pending ops in shard channel.\n")
	b.WriteString("# TYPE l3_shard_queue_depth gauge\n")
	for _, snap := range snaps {
		fmt.Fprintf(b, "l3_shard_queue_depth{shard=\"%d\"} %d\n", snap.ShardID, snap.QueueDepth)
	}

	b.WriteString("\n# HELP l3_shard_queue_capacity Shard op channel capacity.\n")
	b.WriteString("# TYPE l3_shard_queue_capacity gauge\n")
	for _, snap := range snaps {
		fmt.Fprintf(b, "l3_shard_queue_capacity{shard=\"%d\"} %d\n", snap.ShardID, snap.QueueCap)
	}
}

func (c *Collector) writeShardPressure(b *strings.Builder) {
	if c.ShardPressureMetrics == nil {
		return
	}
	snaps := c.ShardPressureMetrics()
	if len(snaps) == 0 {
		return
	}

	b.WriteString("\n\n# ---- shard class pressure -----------------------------------------\n\n")

	b.WriteString("# HELP l3_shard_class_alloc_ops Per-shard per-class allocation attempts.\n")
	b.WriteString("# TYPE l3_shard_class_alloc_ops gauge\n")
	for _, snap := range snaps {
		labels := fmt.Sprintf("shard=\"%d\",class_size=\"%d\"", snap.ShardID, snap.ClassSize)
		fmt.Fprintf(b, "l3_shard_class_alloc_ops{%s} %d\n", labels, snap.AllocOps)
	}

	b.WriteString("# HELP l3_shard_class_alloc_fails Per-shard per-class allocation failures.\n")
	b.WriteString("# TYPE l3_shard_class_alloc_fails gauge\n")
	for _, snap := range snaps {
		labels := fmt.Sprintf("shard=\"%d\",class_size=\"%d\"", snap.ShardID, snap.ClassSize)
		fmt.Fprintf(b, "l3_shard_class_alloc_fails{%s} %d\n", labels, snap.AllocFails)
	}

	b.WriteString("# HELP l3_shard_class_evictions Per-shard per-class evictions.\n")
	b.WriteString("# TYPE l3_shard_class_evictions gauge\n")
	for _, snap := range snaps {
		labels := fmt.Sprintf("shard=\"%d\",class_size=\"%d\"", snap.ShardID, snap.ClassSize)
		fmt.Fprintf(b, "l3_shard_class_evictions{%s} %d\n", labels, snap.Evictions)
	}
}

func (c *Collector) writeBottleneck(b *strings.Builder) {
	if c.OpLatencyMetrics == nil || c.ClientStatsReg == nil {
		return
	}

	// Get server-side latency (avg p50 across shards for "all" ops)
	snaps := c.OpLatencyMetrics()
	if len(snaps) == 0 {
		return
	}
	var sumP50 int64
	for _, snap := range snaps {
		sumP50 += snap.AllP50Us
	}
	serverP50Us := float64(sumP50) / float64(len(snaps))

	// Get client-side phase timing from reported stats
	clientSnap := c.ClientStatsReg.Snapshot()
	if len(clientSnap) == 0 {
		return
	}

	// Use the first client that has reported phase timing
	var preprocessMs, existsMs, transferMs, postprocessMs, roundtripUs float64
	found := false
	for _, cs := range clientSnap {
		if cs.AvgTransferMs > 0 {
			preprocessMs = cs.AvgPreprocessMs
			existsMs = cs.AvgExistsMs
			transferMs = cs.AvgTransferMs
			postprocessMs = cs.AvgPostprocessMs
			roundtripUs = cs.AvgRoundtripUs
			found = true
			break
		}
	}
	if !found {
		return
	}

	b.WriteString("\n\n# ---- bottleneck summary ------------------------------------------\n\n")

	// Convert everything to ms for the breakdown
	serverMs := serverP50Us / 1000.0
	roundtripMs := roundtripUs / 1000.0
	connectorOverheadMs := preprocessMs + postprocessMs + existsMs
	rdmaDataMs := transferMs - serverMs - roundtripMs
	if rdmaDataMs < 0 {
		rdmaDataMs = 0
	}
	networkMs := transferMs - connectorOverheadMs - serverMs - rdmaDataMs
	if networkMs < 0 {
		networkMs = 0
	}

	// Total = io_latency (transfer_ms is the main IO phase)
	totalMs := transferMs
	if totalMs <= 0 {
		return
	}

	connectorPct := connectorOverheadMs / totalMs * 100
	serverPct := serverMs / totalMs * 100
	rdmaPct := rdmaDataMs / totalMs * 100
	networkPct := 100.0 - connectorPct - serverPct - rdmaPct
	if networkPct < 0 {
		networkPct = 0
	}

	b.WriteString("# HELP l3_bottleneck_pct Time breakdown percentage by layer (0-100).\n")
	b.WriteString("# TYPE l3_bottleneck_pct gauge\n")
	fmt.Fprintf(b, "l3_bottleneck_pct{layer=\"connector_overhead\"} %.2f\n", connectorPct)
	fmt.Fprintf(b, "l3_bottleneck_pct{layer=\"server_processing\"} %.2f\n", serverPct)
	fmt.Fprintf(b, "l3_bottleneck_pct{layer=\"rdma_data_transfer\"} %.2f\n", rdmaPct)
	fmt.Fprintf(b, "l3_bottleneck_pct{layer=\"network_queueing\"} %.2f\n", networkPct)

	b.WriteString("\n# HELP l3_bottleneck_ms Absolute time per layer in milliseconds.\n")
	b.WriteString("# TYPE l3_bottleneck_ms gauge\n")
	fmt.Fprintf(b, "l3_bottleneck_ms{layer=\"connector_overhead\"} %.3f\n", connectorOverheadMs)
	fmt.Fprintf(b, "l3_bottleneck_ms{layer=\"server_processing\"} %.3f\n", serverMs)
	fmt.Fprintf(b, "l3_bottleneck_ms{layer=\"rdma_data_transfer\"} %.3f\n", rdmaDataMs)
	fmt.Fprintf(b, "l3_bottleneck_ms{layer=\"network_queueing\"} %.3f\n", networkMs)
}

func (c *Collector) writeMigration(b *strings.Builder, shards []*ShardMetrics, totals *shardTotals) {
	if totals == nil {
		return
	}

	var migrationBuf strings.Builder
	for _, m := range shards {
		id := m.shardID
		migActive := m.MigrationActive()
		migDurMs := m.MigrationDurationMs()
		migEntries := m.MigrationEntries()
		migTotal := m.MigrationsTotal()
		migPreRegMs := m.MigrationPreRegWaitMs()
		fmt.Fprintf(&migrationBuf, "l3_migration_active{shard=\"%d\"} %d\n", id, migActive)
		fmt.Fprintf(&migrationBuf, "l3_migration_last_duration_seconds{shard=\"%d\"} %.3f\n", id, float64(migDurMs)/1000)
		fmt.Fprintf(&migrationBuf, "l3_migration_completed_total{shard=\"%d\"} %d\n", id, migTotal)
		fmt.Fprintf(&migrationBuf, "l3_migration_last_entries{shard=\"%d\"} %d\n", id, migEntries)
		fmt.Fprintf(&migrationBuf, "l3_migration_prereg_wait_seconds{shard=\"%d\"} %.3f\n", id, float64(migPreRegMs)/1000)
		migP99 := m.MigrationP99Us()
		migBatch := m.MigrationBatchSize()
		if migActive != 0 {
			fmt.Fprintf(&migrationBuf, "l3_migration_p99_us{shard=\"%d\"} %d\n", id, migP99)
			fmt.Fprintf(&migrationBuf, "l3_migration_batch_size{shard=\"%d\"} %d\n", id, migBatch)
		}
	}

	b.WriteString("\n\n# ---- migration (ZeroLatencyBalance) ------------------------------\n\n")

	b.WriteString("# HELP l3_migration_active Whether a migration is active on this shard (0/1).\n")
	b.WriteString("# TYPE l3_migration_active gauge\n")
	b.WriteString("# HELP l3_migration_last_duration_seconds Duration of the last completed migration.\n")
	b.WriteString("# TYPE l3_migration_last_duration_seconds gauge\n")
	b.WriteString("# HELP l3_migration_completed_total Total completed migrations per shard.\n")
	b.WriteString("# TYPE l3_migration_completed_total counter\n")
	b.WriteString("# HELP l3_migration_last_entries Entries migrated in the last migration.\n")
	b.WriteString("# TYPE l3_migration_last_entries gauge\n")
	b.WriteString("# HELP l3_migration_prereg_wait_seconds Time spent waiting for MR pre-registration.\n")
	b.WriteString("# TYPE l3_migration_prereg_wait_seconds gauge\n")
	b.WriteString("# HELP l3_migration_p99_us Adaptive p99 latency during active migration.\n")
	b.WriteString("# TYPE l3_migration_p99_us gauge\n")
	b.WriteString("# HELP l3_migration_batch_size Adaptive batch size during active migration.\n")
	b.WriteString("# TYPE l3_migration_batch_size gauge\n")
	b.WriteString(migrationBuf.String())

	b.WriteString("\n# HELP l3_migration_in_progress Total shards with an active migration.\n")
	b.WriteString("# TYPE l3_migration_in_progress gauge\n")
	fmt.Fprintf(b, "l3_migration_in_progress %d\n", totals.migrationsInProgress)
}

func (c *Collector) writeWireProtocol(b *strings.Builder, totals *shardTotals) {
	wireBytesRecv := c.connReg.TotalBytesRecv()
	wireBytesSent := c.connReg.TotalBytesSent()

	b.WriteString("\n\n# ---- wire protocol -----------------------------------------------\n\n")

	b.WriteString("# HELP l3_wire_gb_recv_total Total bytes received on the wire in GiB.\n")
	b.WriteString("# TYPE l3_wire_gb_recv_total counter\n")
	fmt.Fprintf(b, "l3_wire_gb_recv_total %.6f\n", float64(wireBytesRecv)/(1<<30))

	b.WriteString("\n# HELP l3_wire_gb_sent_total Total bytes sent on the wire in GiB.\n")
	b.WriteString("# TYPE l3_wire_gb_sent_total counter\n")
	fmt.Fprintf(b, "l3_wire_gb_sent_total %.6f\n", float64(wireBytesSent)/(1<<30))

	if totals != nil {
		const gb = 1 << 30
		inlinePayloadSent := totals.bytesOut - totals.rdmaReadBytesOut
		effectiveSent := wireBytesSent + totals.rdmaReadBytesOut
		opsRecv := wireBytesRecv - totals.bytesIn
		opsSent := wireBytesSent - inlinePayloadSent

		b.WriteString("\n# HELP l3_wire_effective_gb_sent_total Wire sent + RDMA Read payload in GiB.\n")
		b.WriteString("# TYPE l3_wire_effective_gb_sent_total counter\n")
		fmt.Fprintf(b, "l3_wire_effective_gb_sent_total %.6f\n", float64(effectiveSent)/gb)

		// Backward compatibility: deprecated in v0.22.0, kept for existing dashboards
		b.WriteString("\n# HELP l3_wire_payload_gb_recv Payload bytes received (excluding framing) in GiB (DEPRECATED: use l3_server_payload_gb_in).\n")
		b.WriteString("# TYPE l3_wire_payload_gb_recv counter\n")
		fmt.Fprintf(b, "l3_wire_payload_gb_recv %.6f\n", float64(totals.bytesIn)/gb)

		// Backward compatibility: deprecated in v0.22.0, kept for existing dashboards
		b.WriteString("\n# HELP l3_wire_payload_gb_sent Inline payload bytes sent (excluding RDMA Read) in GiB (DEPRECATED: use l3_server_payload_gb_out).\n")
		b.WriteString("# TYPE l3_wire_payload_gb_sent counter\n")
		fmt.Fprintf(b, "l3_wire_payload_gb_sent %.6f\n", float64(inlinePayloadSent)/gb)

		b.WriteString("\n# HELP l3_wire_ops_gb_recv Framing/opcode overhead bytes received in GiB.\n")
		b.WriteString("# TYPE l3_wire_ops_gb_recv counter\n")
		fmt.Fprintf(b, "l3_wire_ops_gb_recv %.6f\n", float64(opsRecv)/gb)

		b.WriteString("\n# HELP l3_wire_ops_gb_sent Framing/opcode overhead bytes sent in GiB.\n")
		b.WriteString("# TYPE l3_wire_ops_gb_sent counter\n")
		fmt.Fprintf(b, "l3_wire_ops_gb_sent %.6f\n", float64(opsSent)/gb)

		epochSec := time.Since(c.connReg.StatsEpoch()).Seconds()
		if epochSec > 0 {
			gbIn := float64(totals.bytesIn) / gb
			gbOut := float64(totals.bytesOut) / gb

			b.WriteString("\n# HELP l3_wire_throughput_gbps_in Payload ingest throughput in GiB/s.\n")
			b.WriteString("# TYPE l3_wire_throughput_gbps_in gauge\n")
			fmt.Fprintf(b, "l3_wire_throughput_gbps_in %.6f\n", gbIn/epochSec)

			b.WriteString("\n# HELP l3_wire_throughput_gbps_out Payload serve throughput in GiB/s.\n")
			b.WriteString("# TYPE l3_wire_throughput_gbps_out gauge\n")
			fmt.Fprintf(b, "l3_wire_throughput_gbps_out %.6f\n", gbOut/epochSec)

			b.WriteString("\n# HELP l3_wire_throughput_gbps_total Combined payload throughput in GiB/s.\n")
			b.WriteString("# TYPE l3_wire_throughput_gbps_total gauge\n")
			fmt.Fprintf(b, "l3_wire_throughput_gbps_total %.6f\n", (gbIn+gbOut)/epochSec)
		}
	}
}

func (c *Collector) writeRDMA(b *strings.Builder) {
	b.WriteString("\n\n# ---- RDMA --------------------------------------------------------\n\n")

	nicIPs := c.connReg.NICIPs()
	nicBytes := c.connReg.NICWireBytes()
	if len(nicIPs) > 0 {
		b.WriteString("# HELP l3_rdma_nic_wire_gb_total Per-NIC RDMA wire bytes in GiB.\n")
		b.WriteString("# TYPE l3_rdma_nic_wire_gb_total counter\n")
		for dev, ip := range nicIPs {
			total := nicBytes[dev]
			wireGB := float64(total[0]+total[1]) / (1 << 30)
			fmt.Fprintf(b, "l3_rdma_nic_wire_gb_total{device=\"%s\",ip=\"%s\"} %.6f\n", dev, ip, wireGB)
		}

		b.WriteString("\n# HELP l3_rdma_nic_wire_gb_recv Per-NIC RDMA bytes received in GiB.\n")
		b.WriteString("# TYPE l3_rdma_nic_wire_gb_recv counter\n")
		b.WriteString("# HELP l3_rdma_nic_wire_gb_sent Per-NIC RDMA bytes sent in GiB.\n")
		b.WriteString("# TYPE l3_rdma_nic_wire_gb_sent counter\n")
		for dev, ip := range nicIPs {
			total := nicBytes[dev]
			recvGB := float64(total[0]) / (1 << 30)
			sentGB := float64(total[1]) / (1 << 30)
			fmt.Fprintf(b, "l3_rdma_nic_wire_gb_recv{device=\"%s\",ip=\"%s\"} %.6f\n", dev, ip, recvGB)
			fmt.Fprintf(b, "l3_rdma_nic_wire_gb_sent{device=\"%s\",ip=\"%s\"} %.6f\n", dev, ip, sentGB)
		}

		// --- Aggregate RDMA wire totals across all NICs ---
		var aggRecvBytes, aggSentBytes int64
		for _, total := range nicBytes {
			aggRecvBytes += total[0]
			aggSentBytes += total[1]
		}
		b.WriteString("\n# HELP l3_rdma_wire_gb_recv_total Aggregate RDMA bytes received across all NICs in GiB.\n")
		b.WriteString("# TYPE l3_rdma_wire_gb_recv_total counter\n")
		fmt.Fprintf(b, "l3_rdma_wire_gb_recv_total %.6f\n", float64(aggRecvBytes)/(1<<30))

		b.WriteString("\n# HELP l3_rdma_wire_gb_sent_total Aggregate RDMA bytes sent across all NICs in GiB.\n")
		b.WriteString("# TYPE l3_rdma_wire_gb_sent_total counter\n")
		fmt.Fprintf(b, "l3_rdma_wire_gb_sent_total %.6f\n", float64(aggSentBytes)/(1<<30))

		b.WriteString("\n# HELP l3_rdma_wire_gb_total Aggregate RDMA wire bytes (recv+sent) across all NICs in GiB.\n")
		b.WriteString("# TYPE l3_rdma_wire_gb_total counter\n")
		fmt.Fprintf(b, "l3_rdma_wire_gb_total %.6f\n", float64(aggRecvBytes+aggSentBytes)/(1<<30))

		// --- Per-NIC throughput and saturation ---
		nicLinkRates := c.connReg.NICLinkRates()
		nicThroughput := c.connReg.NICThroughputGbps()

		b.WriteString("\n# HELP l3_rdma_nic_throughput_gbps Per-NIC bidirectional throughput in Gbps (averaged since epoch).\n")
		b.WriteString("# TYPE l3_rdma_nic_throughput_gbps gauge\n")
		for dev, ip := range nicIPs {
			fmt.Fprintf(b, "l3_rdma_nic_throughput_gbps{device=\"%s\",ip=\"%s\"} %.6f\n",
				dev, ip, nicThroughput[dev])
		}

		// Aggregate RDMA throughput across all NICs
		var totalTput float64
		for dev := range nicIPs {
			totalTput += nicThroughput[dev]
		}
		b.WriteString("\n# HELP l3_rdma_throughput_gbps Aggregate RDMA throughput across all NICs in Gbps.\n")
		b.WriteString("# TYPE l3_rdma_throughput_gbps gauge\n")
		fmt.Fprintf(b, "l3_rdma_throughput_gbps %.6f\n", totalTput)

		if len(nicLinkRates) > 0 {
			b.WriteString("\n# HELP l3_rdma_nic_link_rate_gbps Detected RDMA link rate per NIC in Gbps.\n")
			b.WriteString("# TYPE l3_rdma_nic_link_rate_gbps gauge\n")
			for dev, ip := range nicIPs {
				fmt.Fprintf(b, "l3_rdma_nic_link_rate_gbps{device=\"%s\",ip=\"%s\"} %.0f\n",
					dev, ip, nicLinkRates[dev])
			}

			// Aggregate link capacity
			var totalCap float64
			for dev := range nicIPs {
				totalCap += nicLinkRates[dev]
			}
			b.WriteString("\n# HELP l3_rdma_link_rate_gbps_total Aggregate RDMA link capacity across all NICs in Gbps.\n")
			b.WriteString("# TYPE l3_rdma_link_rate_gbps_total gauge\n")
			fmt.Fprintf(b, "l3_rdma_link_rate_gbps_total %.0f\n", totalCap)

			b.WriteString("\n# HELP l3_rdma_nic_saturation_pct Per-NIC wire saturation as a percentage of link rate (0-100).\n")
			b.WriteString("# TYPE l3_rdma_nic_saturation_pct gauge\n")
			var minSat, maxSat float64
			first := true
			for dev, ip := range nicIPs {
				rate := nicLinkRates[dev]
				tput := nicThroughput[dev]
				pct := 0.0
				if rate > 0 {
					pct = tput / rate * 100
					if pct > 100 {
						pct = 100
					}
				}
				fmt.Fprintf(b, "l3_rdma_nic_saturation_pct{device=\"%s\",ip=\"%s\"} %.2f\n",
					dev, ip, pct)
				if first || pct < minSat {
					minSat = pct
				}
				if first || pct > maxSat {
					maxSat = pct
				}
				first = false
			}

			balance := 100.0
			if maxSat > 0 {
				balance = minSat / maxSat * 100
			}
			b.WriteString("\n# HELP l3_rdma_nic_balance_pct Cross-NIC traffic balance (100=perfectly equal, low=skewed).\n")
			b.WriteString("# TYPE l3_rdma_nic_balance_pct gauge\n")
			fmt.Fprintf(b, "l3_rdma_nic_balance_pct %.2f\n", balance)

			wireSat := 0.0
			if totalCap > 0 {
				wireSat = totalTput / totalCap * 100
			}
			b.WriteString("\n# HELP l3_wire_saturation_pct Aggregate wire saturation across all NICs as a percentage of total link capacity (0-100).\n")
			b.WriteString("# TYPE l3_wire_saturation_pct gauge\n")
			fmt.Fprintf(b, "l3_wire_saturation_pct %.2f\n", wireSat)
		}
	}

	if c.PollerMetrics != nil {
		psnaps := c.PollerMetrics()
		if len(psnaps) > 0 {
			b.WriteString("\n# HELP l3_rdma_poller_active_conns Active connections on this CQ poller.\n")
			b.WriteString("# TYPE l3_rdma_poller_active_conns gauge\n")
			b.WriteString("# HELP l3_rdma_poller_completions_total Total CQ completions polled.\n")
			b.WriteString("# TYPE l3_rdma_poller_completions_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_enqueued_total Items enqueued to dispatch workers.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_enqueued_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_dropped_total Items dropped (dispatch queue full).\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_dropped_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_send_ch_dropped_total Responses dropped (send channel full).\n")
			b.WriteString("# TYPE l3_rdma_poller_send_ch_dropped_total counter\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_workers Number of dispatch workers.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_workers gauge\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_queue_depth Pending items in RDMA dispatch queue.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_queue_depth gauge\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_queue_capacity RDMA dispatch queue capacity.\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_queue_capacity gauge\n")
			b.WriteString("# HELP l3_rdma_cleanup_queue_depth Pending items in RDMA cleanup queue.\n")
			b.WriteString("# TYPE l3_rdma_cleanup_queue_depth gauge\n")
			b.WriteString("# HELP l3_rdma_cleanup_queue_capacity RDMA cleanup queue capacity.\n")
			b.WriteString("# TYPE l3_rdma_cleanup_queue_capacity gauge\n")
			b.WriteString("# HELP l3_rdma_poller_dispatch_saturation_pct Dispatch queue saturation (0-100).\n")
			b.WriteString("# TYPE l3_rdma_poller_dispatch_saturation_pct gauge\n")
			for _, ps := range psnaps {
				fmt.Fprintf(b, "l3_rdma_poller_active_conns{device=\"%s\"} %d\n", ps.Device, ps.ActiveConns)
				fmt.Fprintf(b, "l3_rdma_poller_completions_total{device=\"%s\"} %d\n", ps.Device, ps.Completions)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_enqueued_total{device=\"%s\"} %d\n", ps.Device, ps.DispatchEnqueued)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_dropped_total{device=\"%s\"} %d\n", ps.Device, ps.DispatchDropped)
				fmt.Fprintf(b, "l3_rdma_poller_send_ch_dropped_total{device=\"%s\"} %d\n", ps.Device, ps.SendChDropped)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_workers{device=\"%s\"} %d\n", ps.Device, ps.DispatchWorkers)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_queue_depth{device=\"%s\"} %d\n", ps.Device, ps.DispatchQueueDepth)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_queue_capacity{device=\"%s\"} %d\n", ps.Device, ps.DispatchQueueCap)
				fmt.Fprintf(b, "l3_rdma_cleanup_queue_depth{device=\"%s\"} %d\n", ps.Device, ps.CleanupQueueDepth)
				fmt.Fprintf(b, "l3_rdma_cleanup_queue_capacity{device=\"%s\"} %d\n", ps.Device, ps.CleanupQueueCap)
				fmt.Fprintf(b, "l3_rdma_poller_dispatch_saturation_pct{device=\"%s\"} %.2f\n", ps.Device, ps.DispatchSaturationPct)
			}
		}
	}

	if c.ReplicationQueue != nil {
		rqDepth, rqCap := c.ReplicationQueue()
		b.WriteString("\n# HELP l3_cluster_replication_queue_depth Pending items in cluster replication queue.\n")
		b.WriteString("# TYPE l3_cluster_replication_queue_depth gauge\n")
		fmt.Fprintf(b, "l3_cluster_replication_queue_depth %d\n", rqDepth)
		b.WriteString("# HELP l3_cluster_replication_queue_capacity Cluster replication queue capacity.\n")
		b.WriteString("# TYPE l3_cluster_replication_queue_capacity gauge\n")
		fmt.Fprintf(b, "l3_cluster_replication_queue_capacity %d\n", rqCap)
	}

	if c.RDMAReadMetrics != nil {
		rs := c.RDMAReadMetrics()
		b.WriteString("\n# HELP l3_rdma_reads_issued_total RDMA Read operations issued.\n")
		b.WriteString("# TYPE l3_rdma_reads_issued_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_issued_total %d\n", rs.Issued)

		b.WriteString("\n# HELP l3_rdma_reads_confirmed_total RDMA Read operations confirmed by client.\n")
		b.WriteString("# TYPE l3_rdma_reads_confirmed_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_confirmed_total %d\n", rs.Confirmed)

		b.WriteString("\n# HELP l3_rdma_reads_failed_total RDMA Read operations that failed.\n")
		b.WriteString("# TYPE l3_rdma_reads_failed_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_failed_total %d\n", rs.Failed)

		b.WriteString("\n# HELP l3_rdma_reads_forced_inline_total GETs forced inline during migration.\n")
		b.WriteString("# TYPE l3_rdma_reads_forced_inline_total counter\n")
		fmt.Fprintf(b, "l3_rdma_reads_forced_inline_total %d\n", rs.ForcedInline)

		b.WriteString("\n# HELP l3_rdma_lease_drops_total RDMA leases dropped (shard channel full), forced inline fallback.\n")
		b.WriteString("# TYPE l3_rdma_lease_drops_total counter\n")
		fmt.Fprintf(b, "l3_rdma_lease_drops_total %d\n", rs.LeaseDrops)

		b.WriteString("\n# HELP l3_rdma_mget_migration_skips_total MGET RDMA keys skipped because their shard was migrating.\n")
		b.WriteString("# TYPE l3_rdma_mget_migration_skips_total counter\n")
		fmt.Fprintf(b, "l3_rdma_mget_migration_skips_total %d\n", rs.MgetMigrationSkips)
	}
}

func (c *Collector) writeSystemHealth(b *strings.Builder) {
	if c.SystemHealth == nil {
		return
	}
	h := c.SystemHealth()

	b.WriteString("\n\n# ---- system health -----------------------------------------------\n\n")

	b.WriteString("# HELP l3_system_memory_available_bytes Available system memory in bytes.\n")
	b.WriteString("# TYPE l3_system_memory_available_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_memory_available_bytes %d\n", h.MemAvailableBytes)

	memPressureVal := 0
	if h.MemPressureActive {
		memPressureVal = 1
	}
	b.WriteString("\n# HELP l3_system_memory_pressure_active Whether the memory pressure circuit breaker is engaged (1=active, 0=normal).\n")
	b.WriteString("# TYPE l3_system_memory_pressure_active gauge\n")
	fmt.Fprintf(b, "l3_system_memory_pressure_active %d\n", memPressureVal)

	b.WriteString("\n# HELP l3_system_memory_pressure_level Tiered memory pressure level (0=normal, 1=elevated, 2=high, 3=critical, 4=emergency).\n")
	b.WriteString("# TYPE l3_system_memory_pressure_level gauge\n")
	fmt.Fprintf(b, "l3_system_memory_pressure_level %d\n", h.MemPressureLevel)

	b.WriteString("\n# HELP l3_system_memory_psi_some_bp Memory PSI 'some' pressure in basis points.\n")
	b.WriteString("# TYPE l3_system_memory_psi_some_bp gauge\n")
	fmt.Fprintf(b, "l3_system_memory_psi_some_bp %d\n", h.MemPSISomeBP)

	b.WriteString("\n# HELP l3_system_memory_psi_full_bp Memory PSI 'full' pressure in basis points.\n")
	b.WriteString("# TYPE l3_system_memory_psi_full_bp gauge\n")
	fmt.Fprintf(b, "l3_system_memory_psi_full_bp %d\n", h.MemPSIFullBP)

	// Memory accounting: Linux splits process memory into two pools.
	// VmRSS (/proc/self/status) counts regular 4KB pages ONLY â€” it excludes
	// hugepage-backed memory. Slab memory on hugepages shows up exclusively in
	// smaps_rollup (Private_Hugetlb + Shared_Hugetlb). htop's RES column reads
	// /proc/[pid]/statm which includes BOTH, so htop shows a much larger number
	// than VmRSS alone. Use l3_system_process_total_rss_bytes to match htop.
	b.WriteString("\n# HELP l3_system_process_total_rss_bytes Total process resident memory in bytes (VmRSS + HugeTLB). Matches htop RES column. This is the metric you want for dashboards and alerts.\n")
	b.WriteString("# TYPE l3_system_process_total_rss_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_process_total_rss_bytes %d\n", h.ProcessTotalRSSBytes)

	b.WriteString("\n# HELP l3_system_process_rss_bytes VmRSS in bytes (regular 4KB pages ONLY). Does NOT include hugeTLB-backed slab memory â€” on a healthy system with hugepages this will be much smaller than htop shows. See l3_system_process_total_rss_bytes for the full picture.\n")
	b.WriteString("# TYPE l3_system_process_rss_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_process_rss_bytes %d\n", h.ProcessRSSBytes)

	b.WriteString("\n# HELP l3_system_process_hugetlb_bytes Hugepage-backed memory in bytes (Private_Hugetlb + Shared_Hugetlb from smaps_rollup). This is where slab memory lives when hugepages are working. 0 means hugepages are not in use or kernel < 4.14.\n")
	b.WriteString("# TYPE l3_system_process_hugetlb_bytes gauge\n")
	fmt.Fprintf(b, "l3_system_process_hugetlb_bytes %d\n", h.ProcessHugetlbBytes)

	if h.FDLimit > 0 {
		fdRatio := float64(h.FDCount) / float64(h.FDLimit)
		b.WriteString("\n# HELP l3_system_fd_ratio File descriptor usage ratio (count/limit).\n")
		b.WriteString("# TYPE l3_system_fd_ratio gauge\n")
		fmt.Fprintf(b, "l3_system_fd_ratio %.4f\n", fdRatio)
	}

	if len(h.NICPortActive) > 0 {
		b.WriteString("\n# HELP l3_rdma_nic_port_active Whether the RDMA NIC port is active (1=active, 0=down).\n")
		b.WriteString("# TYPE l3_rdma_nic_port_active gauge\n")
		for dev, active := range h.NICPortActive {
			val := 0
			if active {
				val = 1
			}
			fmt.Fprintf(b, "l3_rdma_nic_port_active{device=\"%s\"} %d\n", dev, val)
		}
	}

	if len(h.NICHWErrors) > 0 {
		b.WriteString("# HELP l3_rdma_nic_hw_errors_total Accumulated RDMA NIC hardware errors.\n")
		b.WriteString("# TYPE l3_rdma_nic_hw_errors_total counter\n")
		for dev, errs := range h.NICHWErrors {
			for errType, count := range errs {
				if count > 0 {
					fmt.Fprintf(b, "l3_rdma_nic_hw_errors_total{device=\"%s\",type=\"%s\"} %d\n", dev, errType, count)
				}
			}
		}
	}

	// CPU metrics
	b.WriteString("\n# HELP l3_system_cpu_seconds_total Cumulative process CPU time in seconds.\n")
	b.WriteString("# TYPE l3_system_cpu_seconds_total counter\n")
	fmt.Fprintf(b, "l3_system_cpu_seconds_total{mode=\"user\"} %.2f\n", h.CPUUserSeconds)
	fmt.Fprintf(b, "l3_system_cpu_seconds_total{mode=\"system\"} %.2f\n", h.CPUSystemSeconds)

	b.WriteString("\n# HELP l3_system_goroutines Current number of goroutines.\n")
	b.WriteString("# TYPE l3_system_goroutines gauge\n")
	fmt.Fprintf(b, "l3_system_goroutines %d\n", h.Goroutines)

	b.WriteString("\n# HELP l3_system_threads Current number of OS threads.\n")
	b.WriteString("# TYPE l3_system_threads gauge\n")
	fmt.Fprintf(b, "l3_system_threads %d\n", h.Threads)

	b.WriteString("\n# HELP l3_system_context_switches_total Cumulative context switches.\n")
	b.WriteString("# TYPE l3_system_context_switches_total counter\n")
	fmt.Fprintf(b, "l3_system_context_switches_total{type=\"voluntary\"} %d\n", h.VoluntaryCtxSwitches)
	fmt.Fprintf(b, "l3_system_context_switches_total{type=\"involuntary\"} %d\n", h.InvoluntaryCtxSwitches)
}

func (c *Collector) writeSlabAllocator(b *strings.Builder) {
	if c.SlabMetrics == nil {
		return
	}
	classes, detection := c.SlabMetrics()

	b.WriteString("\n\n# ---- slab allocator ----------------------------------------------\n\n")

	if len(classes) > 0 {
		b.WriteString("# HELP l3_slab_class_used_slots Slots currently in use per slab class.\n")
		b.WriteString("# TYPE l3_slab_class_used_slots gauge\n")
		b.WriteString("# HELP l3_slab_class_total_slots Total slots available per slab class.\n")
		b.WriteString("# TYPE l3_slab_class_total_slots gauge\n")
		b.WriteString("# HELP l3_slab_class_alloc_count Cumulative allocations per slab class.\n")
		b.WriteString("# TYPE l3_slab_class_alloc_count counter\n")
		b.WriteString("# HELP l3_slab_class_slot_utilization Slot utilization percentage per slab class.\n")
		b.WriteString("# TYPE l3_slab_class_slot_utilization gauge\n")
		for _, cls := range classes {
			sz := fmt.Sprintf("%d", cls.Size)
			fmt.Fprintf(b, "l3_slab_class_used_slots{size=\"%s\"} %d\n", sz, cls.UsedSlots)
			fmt.Fprintf(b, "l3_slab_class_total_slots{size=\"%s\"} %d\n", sz, cls.TotalSlots)
			fmt.Fprintf(b, "l3_slab_class_alloc_count{size=\"%s\"} %d\n", sz, cls.AllocCount)
			fmt.Fprintf(b, "l3_slab_class_slot_utilization{size=\"%s\"} %.2f\n", sz, cls.SlotUtilization*100)
		}
	}

	if detection.Detected {
		b.WriteString("\n# HELP l3_slab_detected_page_bytes Auto-detected page size in bytes.\n")
		b.WriteString("# TYPE l3_slab_detected_page_bytes gauge\n")
		fmt.Fprintf(b, "l3_slab_detected_page_bytes %d\n", detection.DetectedPageBytes)
	}

	b.WriteString("\n# HELP l3_slab_model_page_bytes Configured model page size in bytes.\n")
	b.WriteString("# TYPE l3_slab_model_page_bytes gauge\n")
	fmt.Fprintf(b, "l3_slab_model_page_bytes %d\n", detection.ConfiguredPageBytes)

	effectiveGB := float64(detection.ModelTotalSlots) * float64(detection.ModelClassSize) / (1 << 30)
	b.WriteString("\n# HELP l3_slab_model_effective_gb Effective memory for the model page class in GiB.\n")
	b.WriteString("# TYPE l3_slab_model_effective_gb gauge\n")
	fmt.Fprintf(b, "l3_slab_model_effective_gb %.6f\n", effectiveGB)

	// Per-class pressure metrics (part of slab section)
	if c.PressureMetrics != nil {
		psnaps := c.PressureMetrics()
		if len(psnaps) > 0 {
			b.WriteString("\n# HELP l3_slab_class_pressure Allocation pressure score per slab class.\n")
			b.WriteString("# TYPE l3_slab_class_pressure gauge\n")
			b.WriteString("# HELP l3_slab_class_eviction_rate Eviction rate per slab class.\n")
			b.WriteString("# TYPE l3_slab_class_eviction_rate gauge\n")
			b.WriteString("# HELP l3_slab_class_alloc_fail_rate Allocation failure rate per slab class.\n")
			b.WriteString("# TYPE l3_slab_class_alloc_fail_rate gauge\n")
			b.WriteString("# HELP l3_slab_class_current_weight Current weight assigned to this slab class.\n")
			b.WriteString("# TYPE l3_slab_class_current_weight gauge\n")
			for _, p := range psnaps {
				sz := fmt.Sprintf("%d", p.Size)
				fmt.Fprintf(b, "l3_slab_class_pressure{size=\"%s\"} %.4f\n", sz, p.Pressure)
				fmt.Fprintf(b, "l3_slab_class_eviction_rate{size=\"%s\"} %.4f\n", sz, p.EvictionRate)
				fmt.Fprintf(b, "l3_slab_class_alloc_fail_rate{size=\"%s\"} %.4f\n", sz, p.AllocFailRate)
				fmt.Fprintf(b, "l3_slab_class_current_weight{size=\"%s\"} %.4f\n", sz, p.CurrentWeight)
			}
		}
	}
}

func (c *Collector) writePerConnection(b *strings.Builder) {
	conns := c.connReg.Snapshot()
	if len(conns) == 0 {
		return
	}

	b.WriteString("\n\n# ---- per-connection ----------------------------------------------\n\n")

	b.WriteString("# HELP l3_conn_bytes_recv Bytes received per connection (label 'remote' contains client IP:port).\n")
	b.WriteString("# TYPE l3_conn_bytes_recv counter\n")
	b.WriteString("# HELP l3_conn_bytes_sent Bytes sent per connection.\n")
	b.WriteString("# TYPE l3_conn_bytes_sent counter\n")
	b.WriteString("# HELP l3_conn_requests Requests handled per connection.\n")
	b.WriteString("# TYPE l3_conn_requests counter\n")
	for _, cm := range conns {
		fmt.Fprintf(b, "l3_conn_bytes_recv{transport=\"%s\",remote=\"%s\",device=\"%s\"} %d\n",
			cm.Transport, cm.RemoteAddr, cm.Device, cm.BytesRecv())
		fmt.Fprintf(b, "l3_conn_bytes_sent{transport=\"%s\",remote=\"%s\",device=\"%s\"} %d\n",
			cm.Transport, cm.RemoteAddr, cm.Device, cm.BytesSent())
		fmt.Fprintf(b, "l3_conn_requests{transport=\"%s\",remote=\"%s\",device=\"%s\"} %d\n",
			cm.Transport, cm.RemoteAddr, cm.Device, cm.Requests())
	}
}

func (c *Collector) writeVacuum(b *strings.Builder) {
	if c.VacuumMetrics == nil {
		return
	}
	vs := c.VacuumMetrics()

	b.WriteString("\n\n# ---- vacuum coordinator ------------------------------------------\n\n")

	b.WriteString("# HELP l3_vacuum_rebalances_total Total vacuum-triggered rebalances.\n")
	b.WriteString("# TYPE l3_vacuum_rebalances_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_rebalances_total %d\n", vs.RebalancesTotal)

	if vs.LastRebalanceEpoch > 0 {
		b.WriteString("\n# HELP l3_vacuum_last_rebalance_seconds Seconds since the last rebalance.\n")
		b.WriteString("# TYPE l3_vacuum_last_rebalance_seconds gauge\n")
		fmt.Fprintf(b, "l3_vacuum_last_rebalance_seconds %.0f\n",
			time.Since(time.Unix(vs.LastRebalanceEpoch, 0)).Seconds())
	}

	b.WriteString("\n# HELP l3_vacuum_pending_shards Shards waiting for rebalance.\n")
	b.WriteString("# TYPE l3_vacuum_pending_shards gauge\n")
	fmt.Fprintf(b, "l3_vacuum_pending_shards %d\n", vs.PendingShards)

	b.WriteString("\n# HELP l3_vacuum_pressure_evals_total Total pressure evaluation cycles.\n")
	b.WriteString("# TYPE l3_vacuum_pressure_evals_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_pressure_evals_total %d\n", vs.PressureEvals)

	b.WriteString("\n# HELP l3_vacuum_pressure_rebuilds_total Rebuilds triggered by pressure.\n")
	b.WriteString("# TYPE l3_vacuum_pressure_rebuilds_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_pressure_rebuilds_total %d\n", vs.PressureRebuilds)

	b.WriteString("\n# HELP l3_vacuum_max_drift Maximum weight drift across slab classes.\n")
	b.WriteString("# TYPE l3_vacuum_max_drift gauge\n")
	fmt.Fprintf(b, "l3_vacuum_max_drift %.4f\n", vs.MaxDrift)

	b.WriteString("\n# HELP l3_vacuum_rebalance_failures_total Failed rebalance attempts.\n")
	b.WriteString("# TYPE l3_vacuum_rebalance_failures_total counter\n")
	fmt.Fprintf(b, "l3_vacuum_rebalance_failures_total %d\n", vs.RebalanceFailures)
}

func (c *Collector) writeStartup(b *strings.Builder) {
	b.WriteString("\n\n# ---- startup progress --------------------------------------------\n\n")

	b.WriteString("# HELP l3_startup_shards_ready Shards that have completed initialization.\n")
	b.WriteString("# TYPE l3_startup_shards_ready gauge\n")
	fmt.Fprintf(b, "l3_startup_shards_ready %d\n", c.startup.ShardsReady.Load())

	b.WriteString("\n# HELP l3_startup_shards_total Total shards to initialize.\n")
	b.WriteString("# TYPE l3_startup_shards_total gauge\n")
	fmt.Fprintf(b, "l3_startup_shards_total %d\n", c.startup.ShardsTotal.Load())

	b.WriteString("\n# HELP l3_startup_mem_reserved_bytes Memory reserved during startup in bytes.\n")
	b.WriteString("# TYPE l3_startup_mem_reserved_bytes gauge\n")
	fmt.Fprintf(b, "l3_startup_mem_reserved_bytes %d\n", c.startup.MemReservedBytes.Load())

	b.WriteString("\n# HELP l3_startup_mem_total_bytes Total memory target in bytes.\n")
	b.WriteString("# TYPE l3_startup_mem_total_bytes gauge\n")
	fmt.Fprintf(b, "l3_startup_mem_total_bytes %d\n", c.startup.MemTotalBytes.Load())
}

func (c *Collector) writeClientStats(b *strings.Builder) {
	if c.ClientStatsReg == nil {
		return
	}
	clientSnap := c.ClientStatsReg.Snapshot()
	if len(clientSnap) == 0 {
		return
	}
	now := time.Now()

	b.WriteString("\n\n# ---- client-reported stats ---------------------------------------\n\n")

	b.WriteString("# HELP l3_client_get_errors Client-reported GET errors (label 'remote' contains client IP:port).\n")
	b.WriteString("# TYPE l3_client_get_errors counter\n")
	b.WriteString("# HELP l3_client_get_successes Client-reported GET successes.\n")
	b.WriteString("# TYPE l3_client_get_successes counter\n")
	b.WriteString("# HELP l3_client_set_errors Client-reported SET errors.\n")
	b.WriteString("# TYPE l3_client_set_errors counter\n")
	b.WriteString("# HELP l3_client_exists_errors Client-reported EXISTS errors.\n")
	b.WriteString("# TYPE l3_client_exists_errors counter\n")
	b.WriteString("# HELP l3_client_exists_timeouts Client-reported EXISTS timeouts.\n")
	b.WriteString("# TYPE l3_client_exists_timeouts counter\n")
	b.WriteString("# HELP l3_client_io_workers Active IO workers on this client.\n")
	b.WriteString("# TYPE l3_client_io_workers gauge\n")
	b.WriteString("# HELP l3_client_avg_io_batch_size Average IO batch size.\n")
	b.WriteString("# TYPE l3_client_avg_io_batch_size gauge\n")
	b.WriteString("# HELP l3_client_avg_io_latency_ms Average IO latency in milliseconds.\n")
	b.WriteString("# TYPE l3_client_avg_io_latency_ms gauge\n")
	b.WriteString("# HELP l3_client_io_calls Total IO calls made by this client.\n")
	b.WriteString("# TYPE l3_client_io_calls counter\n")
	b.WriteString("# HELP l3_client_last_report_seconds Seconds since the last stats report.\n")
	b.WriteString("# TYPE l3_client_last_report_seconds gauge\n")
	b.WriteString("# HELP l3_client_cache_hit_rate SGLang cache hit rate (0-1).\n")
	b.WriteString("# TYPE l3_client_cache_hit_rate gauge\n")
	b.WriteString("# HELP l3_client_token_usage SGLang token usage ratio.\n")
	b.WriteString("# TYPE l3_client_token_usage gauge\n")
	b.WriteString("# HELP l3_client_num_running_reqs SGLang running requests.\n")
	b.WriteString("# TYPE l3_client_num_running_reqs gauge\n")
	b.WriteString("# HELP l3_client_evictable_ratio SGLang evictable cache ratio.\n")
	b.WriteString("# TYPE l3_client_evictable_ratio gauge\n")
	b.WriteString("# HELP l3_client_gen_throughput SGLang generation throughput (tokens/s).\n")
	b.WriteString("# TYPE l3_client_gen_throughput gauge\n")
	b.WriteString("# HELP l3_client_prefetch_bandwidth_gbps Prefetch bandwidth in GiB/s.\n")
	b.WriteString("# TYPE l3_client_prefetch_bandwidth_gbps gauge\n")
	b.WriteString("# HELP l3_client_rdma_read_retries RDMA Read retries on this client.\n")
	b.WriteString("# TYPE l3_client_rdma_read_retries counter\n")
	b.WriteString("# HELP l3_client_rdma_read_failures RDMA Read failures on this client.\n")
	b.WriteString("# TYPE l3_client_rdma_read_failures counter\n")
	b.WriteString("# HELP l3_client_avg_roundtrip_us Avg RDMA control roundtrip latency (us).\n")
	b.WriteString("# TYPE l3_client_avg_roundtrip_us gauge\n")
	b.WriteString("# HELP l3_client_avg_rdma_read_us Avg RDMA Read data transfer latency (us).\n")
	b.WriteString("# TYPE l3_client_avg_rdma_read_us gauge\n")
	b.WriteString("# HELP l3_client_roundtrip_count Total sampled roundtrip operations.\n")
	b.WriteString("# TYPE l3_client_roundtrip_count counter\n")
	b.WriteString("# HELP l3_client_rdma_read_count Total sampled RDMA Read operations.\n")
	b.WriteString("# TYPE l3_client_rdma_read_count counter\n")
	b.WriteString("# HELP l3_client_inflight_gets Currently in-flight GET operations.\n")
	b.WriteString("# TYPE l3_client_inflight_gets gauge\n")
	b.WriteString("# HELP l3_client_inflight_sets Currently in-flight SET operations.\n")
	b.WriteString("# TYPE l3_client_inflight_sets gauge\n")
	b.WriteString("# HELP l3_client_avg_preprocess_ms Avg connector preprocessing latency (ms).\n")
	b.WriteString("# TYPE l3_client_avg_preprocess_ms gauge\n")
	b.WriteString("# HELP l3_client_avg_exists_ms Avg connector dedup existence check latency (ms).\n")
	b.WriteString("# TYPE l3_client_avg_exists_ms gauge\n")
	b.WriteString("# HELP l3_client_avg_transfer_ms Avg connector data transfer latency (ms).\n")
	b.WriteString("# TYPE l3_client_avg_transfer_ms gauge\n")
	b.WriteString("# HELP l3_client_avg_postprocess_ms Avg connector postprocessing latency (ms).\n")
	b.WriteString("# TYPE l3_client_avg_postprocess_ms gauge\n")
	b.WriteString("# HELP l3_client_get_avg_ctrl_ms Avg GET control roundtrip sub-phase (ms).\n")
	b.WriteString("# TYPE l3_client_get_avg_ctrl_ms gauge\n")
	b.WriteString("# HELP l3_client_get_avg_meta_ms Avg GET metadata parse sub-phase (ms).\n")
	b.WriteString("# TYPE l3_client_get_avg_meta_ms gauge\n")
	b.WriteString("# HELP l3_client_get_avg_read_ms Avg GET RDMA Read sub-phase (ms).\n")
	b.WriteString("# TYPE l3_client_get_avg_read_ms gauge\n")
	b.WriteString("# HELP l3_client_get_avg_ack_ms Avg GET ReadAck sub-phase (ms).\n")
	b.WriteString("# TYPE l3_client_get_avg_ack_ms gauge\n")
	b.WriteString("# HELP l3_client_get_batch_count GET batch operations in interval.\n")
	b.WriteString("# TYPE l3_client_get_batch_count gauge\n")
	b.WriteString("# HELP l3_client_get_avg_bytes Avg bytes per GET batch.\n")
	b.WriteString("# TYPE l3_client_get_avg_bytes gauge\n")
	b.WriteString("# HELP l3_client_set_avg_serialize_ms Avg SET serialization sub-phase (ms).\n")
	b.WriteString("# TYPE l3_client_set_avg_serialize_ms gauge\n")
	b.WriteString("# HELP l3_client_set_avg_send_ms Avg SET send sub-phase (ms).\n")
	b.WriteString("# TYPE l3_client_set_avg_send_ms gauge\n")
	b.WriteString("# HELP l3_client_set_batch_count SET batch operations in interval.\n")
	b.WriteString("# TYPE l3_client_set_batch_count gauge\n")
	b.WriteString("# HELP l3_client_set_avg_bytes Avg bytes per SET batch.\n")
	b.WriteString("# TYPE l3_client_set_avg_bytes gauge\n")
	b.WriteString("# HELP l3_client_set_avg_sub_batches Avg MSET sub-batches per call (1.0 = no chunking).\n")
	b.WriteString("# TYPE l3_client_set_avg_sub_batches gauge\n")
	b.WriteString("# HELP l3_client_avg_batch_read_us Avg C++ batch RDMA Read latency (us).\n")
	b.WriteString("# TYPE l3_client_avg_batch_read_us gauge\n")
	b.WriteString("# HELP l3_client_batch_read_count C++ batch RDMA Read operations.\n")
	b.WriteString("# TYPE l3_client_batch_read_count counter\n")
	b.WriteString("# HELP l3_client_io_latency_ms Batch I/O latency histogram (ms).\n")
	b.WriteString("# TYPE l3_client_io_latency_ms histogram\n")
	b.WriteString("# HELP l3_client_host_alloc_drops Cumulative host pool allocation failures.\n")
	b.WriteString("# TYPE l3_client_host_alloc_drops counter\n")
	b.WriteString("# HELP l3_client_backup_queue_depth Pending backup operations.\n")
	b.WriteString("# TYPE l3_client_backup_queue_depth gauge\n")
	b.WriteString("# HELP l3_client_backup_ops_completed Cumulative successful backup ops.\n")
	b.WriteString("# TYPE l3_client_backup_ops_completed counter\n")
	b.WriteString("# HELP l3_client_backup_ops_failed Cumulative failed backup ops.\n")
	b.WriteString("# TYPE l3_client_backup_ops_failed counter\n")
	b.WriteString("# HELP l3_client_backup_in_flight Currently executing backup tasks.\n")
	b.WriteString("# TYPE l3_client_backup_in_flight gauge\n")
	b.WriteString("# HELP l3_client_backup_avg_latency_ms Average backup op latency.\n")
	b.WriteString("# TYPE l3_client_backup_avg_latency_ms gauge\n")
	b.WriteString("# HELP l3_client_backup_jitter_total_ms Total jitter sleep over interval.\n")
	b.WriteString("# TYPE l3_client_backup_jitter_total_ms gauge\n")
	b.WriteString("# HELP l3_client_backup_avg_gap_ms Average time between sub-batches.\n")
	b.WriteString("# TYPE l3_client_backup_avg_gap_ms gauge\n")
	b.WriteString("# HELP l3_client_backup_jitter_cfg_ms Configured jitter value.\n")
	b.WriteString("# TYPE l3_client_backup_jitter_cfg_ms gauge\n")
	b.WriteString("# HELP l3_client_backup_batch_size Current effective backup batch size.\n")
	b.WriteString("# TYPE l3_client_backup_batch_size gauge\n")
	b.WriteString("# HELP l3_client_prefetch_received Total prefetch requests received.\n")
	b.WriteString("# TYPE l3_client_prefetch_received counter\n")
	b.WriteString("# HELP l3_client_prefetch_dispatched Prefetch requests passed threshold.\n")
	b.WriteString("# TYPE l3_client_prefetch_dispatched counter\n")
	b.WriteString("# HELP l3_client_prefetch_revoked Prefetch requests revoked.\n")
	b.WriteString("# TYPE l3_client_prefetch_revoked counter\n")
	b.WriteString("# HELP l3_client_prefetch_io_completed Successful prefetch IO ops.\n")
	b.WriteString("# TYPE l3_client_prefetch_io_completed counter\n")
	b.WriteString("# HELP l3_client_prefetch_io_failed Failed prefetch IO ops.\n")
	b.WriteString("# TYPE l3_client_prefetch_io_failed counter\n")
	b.WriteString("# HELP l3_client_prefetch_tokens Cumulative tokens prefetched.\n")
	b.WriteString("# TYPE l3_client_prefetch_tokens counter\n")
	b.WriteString("# HELP l3_client_prefetch_pages Cumulative pages prefetched.\n")
	b.WriteString("# TYPE l3_client_prefetch_pages counter\n")
	b.WriteString("# HELP l3_client_prefetch_avg_io_ms Average prefetch IO latency.\n")
	b.WriteString("# TYPE l3_client_prefetch_avg_io_ms gauge\n")
	b.WriteString("# HELP l3_client_prefetch_queue_depth Pending prefetch requests.\n")
	b.WriteString("# TYPE l3_client_prefetch_queue_depth gauge\n")
	b.WriteString("# HELP l3_client_prefetch_in_flight Currently executing prefetch IO tasks.\n")
	b.WriteString("# TYPE l3_client_prefetch_in_flight gauge\n")
	b.WriteString("# HELP l3_client_reconnect_count Cumulative successful reconnections.\n")
	b.WriteString("# TYPE l3_client_reconnect_count counter\n")
	b.WriteString("# HELP l3_client_pool_size Configured connection pool size.\n")
	b.WriteString("# TYPE l3_client_pool_size gauge\n")
	b.WriteString("# HELP l3_client_dedup_low_hit_streak Consecutive low-hit batches.\n")
	b.WriteString("# TYPE l3_client_dedup_low_hit_streak gauge\n")
	b.WriteString("# HELP l3_client_total_pages_set Cumulative pages stored by this client.\n")
	b.WriteString("# TYPE l3_client_total_pages_set counter\n")
	b.WriteString("# HELP l3_client_total_pages_get Cumulative pages retrieved by this client.\n")
	b.WriteString("# TYPE l3_client_total_pages_get counter\n")
	b.WriteString("# HELP l3_client_backup_coalesce_avg_ops Average operations per coalesced backup batch.\n")
	b.WriteString("# TYPE l3_client_backup_coalesce_avg_ops gauge\n")
	b.WriteString("# HELP l3_client_dedup_mode Deduplication mode active on this client.\n")
	b.WriteString("# TYPE l3_client_dedup_mode gauge\n")
	b.WriteString("# HELP l3_client_dedup_auto_disabled Whether adaptive dedup has been auto-disabled.\n")
	b.WriteString("# TYPE l3_client_dedup_auto_disabled gauge\n")
	b.WriteString("# HELP l3_client_codec Compression codec info (always 1; codec name in 'codec' label).\n")
	b.WriteString("# TYPE l3_client_codec gauge\n")
	b.WriteString("# HELP l3_client_codec_lossy Whether active codec is lossy (1=yes, 0=no).\n")
	b.WriteString("# TYPE l3_client_codec_lossy gauge\n")
	b.WriteString("# HELP l3_client_backup_bandwidth_gbps Average backup write bandwidth (GiB/s).\n")
	b.WriteString("# TYPE l3_client_backup_bandwidth_gbps gauge\n")
	b.WriteString("# HELP l3_client_io_max_latency_ms Max batch I/O latency in interval (ms).\n")
	b.WriteString("# TYPE l3_client_io_max_latency_ms gauge\n")
	b.WriteString("# HELP l3_client_warmup_phase Connector warmup state machine phase.\n")
	b.WriteString("# TYPE l3_client_warmup_phase gauge\n")
	b.WriteString("# HELP l3_client_warmup_effective_batch_size Effective batch size during warmup.\n")
	b.WriteString("# TYPE l3_client_warmup_effective_batch_size gauge\n")
	b.WriteString("# HELP l3_client_pool_rebuilds Cumulative RDMA pool full rebuilds.\n")
	b.WriteString("# TYPE l3_client_pool_rebuilds counter\n")
	b.WriteString("# HELP l3_client_dedup_cost_streak Consecutive cost-exceeding batches.\n")
	b.WriteString("# TYPE l3_client_dedup_cost_streak gauge\n")
	b.WriteString("# HELP l3_client_transport Client transport type info (always 1; transport in 'transport' label).\n")
	b.WriteString("# TYPE l3_client_transport gauge\n")
	b.WriteString("# HELP l3_client_model_page_bytes Model page size in bytes (0=auto-detect).\n")
	b.WriteString("# TYPE l3_client_model_page_bytes gauge\n")
	b.WriteString("# HELP l3_client_nic_reads RDMA Read ops per NIC.\n")
	b.WriteString("# TYPE l3_client_nic_reads counter\n")
	b.WriteString("# HELP l3_client_nic_bytes_gb RDMA Read bytes (GB) per NIC.\n")
	b.WriteString("# TYPE l3_client_nic_bytes_gb counter\n")
	b.WriteString("# HELP l3_client_nic_writes RDMA Write (SET) ops per NIC.\n")
	b.WriteString("# TYPE l3_client_nic_writes counter\n")
	b.WriteString("# HELP l3_client_nic_write_bytes_gb RDMA Write bytes (GB) per NIC.\n")
	b.WriteString("# TYPE l3_client_nic_write_bytes_gb counter\n")
	b.WriteString("# HELP l3_client_stripe_calls Total striped mget_rdma calls.\n")
	b.WriteString("# TYPE l3_client_stripe_calls counter\n")
	b.WriteString("# HELP l3_client_stripe_errors Striped mget_rdma errors.\n")
	b.WriteString("# TYPE l3_client_stripe_errors counter\n")
	b.WriteString("# HELP l3_client_stripe_write_calls Total striped mset calls.\n")
	b.WriteString("# TYPE l3_client_stripe_write_calls counter\n")
	b.WriteString("# HELP l3_client_stripe_write_errors Striped mset errors.\n")
	b.WriteString("# TYPE l3_client_stripe_write_errors counter\n")
	b.WriteString("# HELP l3_client_stripe_avg_nics Average NICs used per stripe call.\n")
	b.WriteString("# TYPE l3_client_stripe_avg_nics gauge\n")

	for _, cs := range clientSnap {
		labels := fmt.Sprintf("remote=\"%s\"", cs.RemoteAddr)
		fmt.Fprintf(b, "l3_client_get_errors{%s} %d\n", labels, cs.GetErrors)
		fmt.Fprintf(b, "l3_client_get_successes{%s} %d\n", labels, cs.GetSuccesses)
		fmt.Fprintf(b, "l3_client_set_errors{%s} %d\n", labels, cs.SetErrors)
		fmt.Fprintf(b, "l3_client_exists_errors{%s} %d\n", labels, cs.ExistsErrors)
		fmt.Fprintf(b, "l3_client_exists_timeouts{%s} %d\n", labels, cs.ExistsTimeouts)
		fmt.Fprintf(b, "l3_client_io_workers{%s} %d\n", labels, cs.IoWorkers)
		fmt.Fprintf(b, "l3_client_avg_io_batch_size{%s} %.1f\n", labels, cs.AvgIoBatchSize)
		fmt.Fprintf(b, "l3_client_avg_io_latency_ms{%s} %.1f\n", labels, cs.AvgIoLatencyMs)
		fmt.Fprintf(b, "l3_client_io_calls{%s} %d\n", labels, cs.IoCalls)
		fmt.Fprintf(b, "l3_client_last_report_seconds{%s} %.1f\n", labels, now.Sub(cs.LastUpdated).Seconds())
		fmt.Fprintf(b, "l3_client_cache_hit_rate{%s} %.4f\n", labels, cs.CacheHitRate)
		fmt.Fprintf(b, "l3_client_token_usage{%s} %.4f\n", labels, cs.TokenUsage)
		fmt.Fprintf(b, "l3_client_num_running_reqs{%s} %d\n", labels, cs.NumRunningReqs)
		fmt.Fprintf(b, "l3_client_evictable_ratio{%s} %.4f\n", labels, cs.EvictableRatio)
		fmt.Fprintf(b, "l3_client_gen_throughput{%s} %.2f\n", labels, cs.GenThroughput)
		fmt.Fprintf(b, "l3_client_prefetch_bandwidth_gbps{%s} %.4f\n", labels, cs.PrefetchBandwidthGbps)
		fmt.Fprintf(b, "l3_client_rdma_read_retries{%s} %d\n", labels, cs.RdmaReadRetries)
		fmt.Fprintf(b, "l3_client_rdma_read_failures{%s} %d\n", labels, cs.RdmaReadFailures)
		fmt.Fprintf(b, "l3_client_avg_roundtrip_us{%s} %.2f\n", labels, cs.AvgRoundtripUs)
		fmt.Fprintf(b, "l3_client_avg_rdma_read_us{%s} %.2f\n", labels, cs.AvgRdmaReadUs)
		fmt.Fprintf(b, "l3_client_roundtrip_count{%s} %d\n", labels, cs.RoundtripCount)
		fmt.Fprintf(b, "l3_client_rdma_read_count{%s} %d\n", labels, cs.RdmaReadCount)
		fmt.Fprintf(b, "l3_client_inflight_gets{%s} %d\n", labels, cs.InFlightGets)
		fmt.Fprintf(b, "l3_client_inflight_sets{%s} %d\n", labels, cs.InFlightSets)
		fmt.Fprintf(b, "l3_client_avg_preprocess_ms{%s} %.2f\n", labels, cs.AvgPreprocessMs)
		fmt.Fprintf(b, "l3_client_avg_exists_ms{%s} %.2f\n", labels, cs.AvgExistsMs)
		fmt.Fprintf(b, "l3_client_avg_transfer_ms{%s} %.2f\n", labels, cs.AvgTransferMs)
		fmt.Fprintf(b, "l3_client_avg_postprocess_ms{%s} %.2f\n", labels, cs.AvgPostprocessMs)
		// Transfer sub-phase timing
		fmt.Fprintf(b, "l3_client_get_avg_ctrl_ms{%s} %.2f\n", labels, cs.GetAvgCtrlMs)
		fmt.Fprintf(b, "l3_client_get_avg_meta_ms{%s} %.2f\n", labels, cs.GetAvgMetaMs)
		fmt.Fprintf(b, "l3_client_get_avg_read_ms{%s} %.2f\n", labels, cs.GetAvgReadMs)
		fmt.Fprintf(b, "l3_client_get_avg_ack_ms{%s} %.2f\n", labels, cs.GetAvgAckMs)
		fmt.Fprintf(b, "l3_client_get_batch_count{%s} %d\n", labels, cs.GetBatchCount)
		fmt.Fprintf(b, "l3_client_get_avg_bytes{%s} %.0f\n", labels, cs.GetAvgBytes)
		fmt.Fprintf(b, "l3_client_set_avg_serialize_ms{%s} %.2f\n", labels, cs.SetAvgSerializeMs)
		fmt.Fprintf(b, "l3_client_set_avg_send_ms{%s} %.2f\n", labels, cs.SetAvgSendMs)
		fmt.Fprintf(b, "l3_client_set_batch_count{%s} %d\n", labels, cs.SetBatchCount)
		fmt.Fprintf(b, "l3_client_set_avg_bytes{%s} %.0f\n", labels, cs.SetAvgBytes)
		fmt.Fprintf(b, "l3_client_set_avg_sub_batches{%s} %.2f\n", labels, cs.SetAvgSubBatches)
		fmt.Fprintf(b, "l3_client_avg_batch_read_us{%s} %.2f\n", labels, cs.AvgBatchReadUs)
		fmt.Fprintf(b, "l3_client_batch_read_count{%s} %d\n", labels, cs.BatchReadCount)
		// Backup thread metrics
		fmt.Fprintf(b, "l3_client_host_alloc_drops{%s} %d\n", labels, cs.HostAllocDrops)
		fmt.Fprintf(b, "l3_client_backup_queue_depth{%s} %d\n", labels, cs.BackupQueueDepth)
		fmt.Fprintf(b, "l3_client_backup_ops_completed{%s} %d\n", labels, cs.BackupOpsCompleted)
		fmt.Fprintf(b, "l3_client_backup_ops_failed{%s} %d\n", labels, cs.BackupOpsFailed)
		fmt.Fprintf(b, "l3_client_backup_in_flight{%s} %d\n", labels, cs.BackupInFlight)
		fmt.Fprintf(b, "l3_client_backup_avg_latency_ms{%s} %.2f\n", labels, cs.BackupAvgLatencyMs)
		fmt.Fprintf(b, "l3_client_backup_jitter_total_ms{%s} %.2f\n", labels, cs.BackupJitterTotalMs)
		fmt.Fprintf(b, "l3_client_backup_avg_gap_ms{%s} %.2f\n", labels, cs.BackupAvgGapMs)
		fmt.Fprintf(b, "l3_client_backup_jitter_cfg_ms{%s} %.2f\n", labels, cs.BackupJitterCfgMs)
		fmt.Fprintf(b, "l3_client_backup_batch_size{%s} %d\n", labels, cs.BackupBatchSize)
		// Prefetch thread metrics
		fmt.Fprintf(b, "l3_client_prefetch_received{%s} %d\n", labels, cs.PrefetchReceived)
		fmt.Fprintf(b, "l3_client_prefetch_dispatched{%s} %d\n", labels, cs.PrefetchDispatched)
		fmt.Fprintf(b, "l3_client_prefetch_revoked{%s} %d\n", labels, cs.PrefetchRevoked)
		fmt.Fprintf(b, "l3_client_prefetch_io_completed{%s} %d\n", labels, cs.PrefetchIoCompleted)
		fmt.Fprintf(b, "l3_client_prefetch_io_failed{%s} %d\n", labels, cs.PrefetchIoFailed)
		fmt.Fprintf(b, "l3_client_prefetch_tokens{%s} %d\n", labels, cs.PrefetchTokens)
		fmt.Fprintf(b, "l3_client_prefetch_pages{%s} %d\n", labels, cs.PrefetchPages)
		fmt.Fprintf(b, "l3_client_prefetch_avg_io_ms{%s} %.2f\n", labels, cs.PrefetchAvgIoMs)
		fmt.Fprintf(b, "l3_client_prefetch_queue_depth{%s} %d\n", labels, cs.PrefetchQueueDepth)
		fmt.Fprintf(b, "l3_client_prefetch_in_flight{%s} %d\n", labels, cs.PrefetchInFlight)
		// Connector config/state
		fmt.Fprintf(b, "l3_client_reconnect_count{%s} %d\n", labels, cs.ReconnectCount)
		fmt.Fprintf(b, "l3_client_pool_size{%s} %d\n", labels, cs.PoolSize)
		fmt.Fprintf(b, "l3_client_dedup_low_hit_streak{%s} %d\n", labels, cs.DedupLowHitStreak)
		fmt.Fprintf(b, "l3_client_total_pages_set{%s} %d\n", labels, cs.TotalPagesSet)
		fmt.Fprintf(b, "l3_client_total_pages_get{%s} %d\n", labels, cs.TotalPagesGet)
		fmt.Fprintf(b, "l3_client_backup_coalesce_avg_ops{%s} %.1f\n", labels, cs.BackupCoalesceAvgOps)
		if cs.DedupMode != "" {
			fmt.Fprintf(b, "l3_client_dedup_mode{%s,mode=\"%s\"} 1\n", labels, cs.DedupMode)
		}
		dedupDisabled := 0
		if cs.DedupAutoDisabled {
			dedupDisabled = 1
		}
		fmt.Fprintf(b, "l3_client_dedup_auto_disabled{%s} %d\n", labels, dedupDisabled)
		if cs.Codec != "" {
			fmt.Fprintf(b, "l3_client_codec{%s,codec=\"%s\"} 1\n", labels, cs.Codec)
		}
		codecLossy := 0
		if cs.CodecLossy {
			codecLossy = 1
		}
		fmt.Fprintf(b, "l3_client_codec_lossy{%s} %d\n", labels, codecLossy)
		fmt.Fprintf(b, "l3_client_backup_bandwidth_gbps{%s} %.4f\n", labels, cs.BackupBandwidthGbps)
		fmt.Fprintf(b, "l3_client_io_max_latency_ms{%s} %.2f\n", labels, cs.IoMaxLatencyMs)
		if cs.WarmupPhase != "" {
			fmt.Fprintf(b, "l3_client_warmup_phase{%s,phase=\"%s\"} 1\n", labels, cs.WarmupPhase)
		}
		fmt.Fprintf(b, "l3_client_warmup_effective_batch_size{%s} %d\n", labels, cs.WarmupEffectiveBatchSize)
		fmt.Fprintf(b, "l3_client_pool_rebuilds{%s} %d\n", labels, cs.PoolRebuilds)
		fmt.Fprintf(b, "l3_client_dedup_cost_streak{%s} %d\n", labels, cs.DedupCostStreak)
		if cs.Transport != "" {
			fmt.Fprintf(b, "l3_client_transport{%s,transport=\"%s\"} 1\n", labels, cs.Transport)
		}
		// Model page bytes
		fmt.Fprintf(b, "l3_client_model_page_bytes{%s} %d\n", labels, cs.ModelPageBytes)
		// Per-NIC RDMA stripe metrics
		for i := 0; i < cs.NICCount; i++ {
			nicLabel := fmt.Sprintf("%s,nic=\"%d\"", labels, i)
			fmt.Fprintf(b, "l3_client_nic_reads{%s} %d\n", nicLabel, cs.NICReads[i])
			fmt.Fprintf(b, "l3_client_nic_bytes_gb{%s} %.6f\n", nicLabel, cs.NICBytesGB[i])
			fmt.Fprintf(b, "l3_client_nic_writes{%s} %d\n", nicLabel, cs.NICWrites[i])
			fmt.Fprintf(b, "l3_client_nic_write_bytes_gb{%s} %.6f\n", nicLabel, cs.NICWriteBytesGB[i])
		}
		// Stripe aggregates
		fmt.Fprintf(b, "l3_client_stripe_calls{%s} %d\n", labels, cs.StripeCalls)
		fmt.Fprintf(b, "l3_client_stripe_errors{%s} %d\n", labels, cs.StripeErrors)
		fmt.Fprintf(b, "l3_client_stripe_write_calls{%s} %d\n", labels, cs.StripeWriteCalls)
		fmt.Fprintf(b, "l3_client_stripe_write_errors{%s} %d\n", labels, cs.StripeWriteErrors)
		fmt.Fprintf(b, "l3_client_stripe_avg_nics{%s} %.2f\n", labels, cs.StripeAvgNICs)
		// Latency histogram
		if len(cs.IoLatencyHistBuckets) > 0 && len(cs.IoLatencyHistCounts) > 0 {
			for i, le := range cs.IoLatencyHistBuckets {
				if i < len(cs.IoLatencyHistCounts) {
					fmt.Fprintf(b, "l3_client_io_latency_ms_bucket{%s,le=\"%.1f\"} %d\n",
						labels, le, cs.IoLatencyHistCounts[i])
				}
			}
			// +Inf bucket: use last counts element if present, else total
			if len(cs.IoLatencyHistCounts) > len(cs.IoLatencyHistBuckets) {
				fmt.Fprintf(b, "l3_client_io_latency_ms_bucket{%s,le=\"+Inf\"} %d\n",
					labels, cs.IoLatencyHistCounts[len(cs.IoLatencyHistCounts)-1])
			} else {
				fmt.Fprintf(b, "l3_client_io_latency_ms_bucket{%s,le=\"+Inf\"} %d\n",
					labels, cs.IoLatencyHistTotal)
			}
			fmt.Fprintf(b, "l3_client_io_latency_ms_sum{%s} %.2f\n", labels, cs.IoLatencyHistSum)
			fmt.Fprintf(b, "l3_client_io_latency_ms_count{%s} %d\n", labels, cs.IoLatencyHistTotal)
		}
	}
}

// slabReady returns true when all shards have finished slab detection (no shard
// is in "warming_up" status). Returns true if ShardDetectionFunc is not wired up.
func (c *Collector) slabReady() (ready bool, pending int, total int) {
	if c.ShardDetectionFunc == nil {
		return true, 0, 0
	}
	statuses := c.ShardDetectionFunc()
	total = len(statuses)
	for _, s := range statuses {
		if s == "warming_up" {
			pending++
		}
	}
	return pending == 0, pending, total
}

// ReadyHandler returns 200 with {"status":"ready"} when the server is ready,
// or 503 with JSON progress details otherwise. HTTP status codes are unchanged
// for backward compatibility with k8s probes.
//
// When ?slab=true is present, also checks that all shards have completed slab
// detection (no shard in "warming_up" status). Without ?slab=true, behavior is
// identical to previous versions.
func (c *Collector) ReadyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if c.startup.Ready.Load() {
		// Base server is ready; optionally check slab detection status.
		if r.URL.Query().Get("slab") == "true" {
			if ok, pending, total := c.slabReady(); !ok {
				w.WriteHeader(http.StatusServiceUnavailable)
				json.NewEncoder(w).Encode(map[string]interface{}{
					"status":         "slab_not_ready",
					"shards_pending": pending,
					"shards_total":   total,
				})
				return
			}
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ready"}` + "\n"))
		return
	}

	shardsReady := c.startup.ShardsReady.Load()
	shardsTotal := c.startup.ShardsTotal.Load()
	memReserved := c.startup.MemReservedBytes.Load()
	memTotal := c.startup.MemTotalBytes.Load()

	phaseLabel := "init"
	if v := c.startup.PhaseLabel.Load(); v != nil {
		phaseLabel = v.(string)
	}

	status := "not_ready"
	if c.startup.AcceptingConnections.Load() {
		status = "accepting"
	}
	resp := map[string]interface{}{
		"status":          status,
		"phase":           phaseLabel,
		"shards_ready":    shardsReady,
		"shards_total":    shardsTotal,
		"mem_reserved_gb": float64(memReserved) / (1 << 30),
		"mem_total_gb":    float64(memTotal) / (1 << 30),
	}

	if shardsTotal > 0 {
		pct := float64(shardsReady) / float64(shardsTotal) * 100
		resp["percent"] = pct

		avgNs := c.startup.AvgShardNanos.Load()
		if avgNs > 0 && shardsReady < shardsTotal {
			remaining := shardsTotal - shardsReady
			etaSeconds := float64(remaining) * float64(avgNs) / 1e9
			resp["eta_seconds"] = etaSeconds
		}
	}

	w.WriteHeader(http.StatusServiceUnavailable)
	json.NewEncoder(w).Encode(resp)
}

// MetricsJSONHandler returns all metrics as a single JSON document.
// Mirrors the full Prometheus /metrics endpoint for programmatic export.
// Accessible at /debug/metrics.json.
func (c *Collector) MetricsJSONHandler(w http.ResponseWriter, r *http.Request) {
	result := make(map[string]interface{})

	c.shardsMu.RLock()
	shards := c.shards
	c.shardsMu.RUnlock()

	totals := aggregateShards(shards)

	// ================================================================
	// Section 1: Server State
	// ================================================================

	serverState := map[string]interface{}{
		"ready":              c.startup.Ready.Load(),
		"uptime_seconds":     time.Since(c.startedAt).Seconds(),
		"active_connections": c.connReg.ActiveCount(),
	}

	if c.InflightOpsFunc != nil {
		serverState["inflight_ops"] = c.InflightOpsFunc()
	}

	if c.CacheStateProvider != nil {
		allocBytes, entries := c.CacheStateProvider()
		activeGB := float64(allocBytes) / (1 << 30)
		serverState["active_gb"] = activeGB
		serverState["entries"] = entries

		if c.MaxKeysPerShard > 0 && c.NumShards > 0 {
			maxKeys := int64(c.MaxKeysPerShard) * int64(c.NumShards)
			evictionCap := maxKeys * index.MaxLoadNumerator / index.MaxLoadDenominator
			utilPct := float64(0)
			if evictionCap > 0 {
				utilPct = float64(entries) / float64(evictionCap) * 100
			}
			serverState["max_keys"] = maxKeys
			serverState["eviction_cap"] = evictionCap
			serverState["key_utilization_percent"] = utilPct
		}

		var totalBytesIn int64
		if totals != nil {
			totalBytesIn = totals.bytesIn
		}
		ingestedGB := float64(totalBytesIn) / (1 << 30)
		if ingestedGB > 0 {
			serverState["churn_ratio"] = activeGB / ingestedGB * 100
		}
	}

	result["server_state"] = serverState

	// ================================================================
	// Section 2: Aggregate Operations
	// ================================================================

	if totals != nil {
		hitRate := float64(0)
		if totals.gets > 0 {
			hitRate = float64(totals.hits) / float64(totals.gets) * 100
		}
		existsHitRate := float64(0)
		if totals.exists > 0 {
			existsHitRate = float64(totals.existsHits) / float64(totals.exists) * 100
		}
		result["ops"] = map[string]interface{}{
			"gets":                  totals.gets,
			"sets":                  totals.sets,
			"deletes":               totals.deletes,
			"exists":                totals.exists,
			"exists_hits":           totals.existsHits,
			"exists_misses":         totals.existsMisses,
			"hits":                  totals.hits,
			"misses":                totals.misses,
			"ttl_expirations":       totals.ttlExpirations,
			"hit_rate_percent":      hitRate,
			"exists_hit_rate_percent": existsHitRate,
		}
	}

	// ================================================================
	// Section 3: Eviction
	// ================================================================

	if totals != nil {
		evictionRate := float64(0)
		if totals.sets > 0 {
			evictionRate = float64(totals.evictions) / float64(totals.sets) * 100
		}
		evictionFailRate := float64(0)
		if totals.evictionsKeyPressure+totals.evictionsValuePressure+totals.evictionsFailed > 0 {
			evictionFailRate = float64(totals.evictionsFailed) / float64(totals.evictionsKeyPressure+totals.evictionsValuePressure+totals.evictionsFailed) * 100
		}
		totalEvictionsOrganic := totals.evictions - totals.evictionsRebalance
		if totalEvictionsOrganic < 0 {
			totalEvictionsOrganic = 0
		}
		result["eviction"] = map[string]interface{}{
			"total":          totals.evictions,
			"key_pressure":   totals.evictionsKeyPressure,
			"value_pressure": totals.evictionsValuePressure,
			"failed":         totals.evictionsFailed,
			"lease_skip":     totals.evictionsLeaseSkip,
			"rebalance":      totals.evictionsRebalance,
			"organic":        totalEvictionsOrganic,
			"promotions":     totals.promotions,
			"rate_percent":   evictionRate,
			"fail_rate_percent": evictionFailRate,
		}
	}

	// ================================================================
	// Section 4: Payload
	// ================================================================

	if totals != nil {
		payload := map[string]interface{}{
			"gb_in":             float64(totals.bytesIn) / (1 << 30),
			"gb_out":            float64(totals.bytesOut) / (1 << 30),
			"rdma_read_gb_out":  float64(totals.rdmaReadBytesOut) / (1 << 30),
		}
		if totals.sets > 0 {
			payload["avg_key_bytes"] = float64(totals.keyBytesIn) / float64(totals.sets)
			payload["avg_value_bytes"] = float64(totals.valueBytesIn) / float64(totals.sets)
		}
		result["payload"] = payload
	}

	// ================================================================
	// Section 5: Per-Shard Operations + Migration
	// ================================================================

	if shards != nil {
		type shardEntry struct {
			ID                       int   `json:"id"`
			Gets                     int64 `json:"gets"`
			Sets                     int64 `json:"sets"`
			Evictions                int64 `json:"evictions"`
			BytesIn                  int64 `json:"bytes_in"`
			EvictionsKeyPressure     int64 `json:"evictions_key_pressure"`
			EvictionsValuePressure   int64 `json:"evictions_value_pressure"`
			EvictionsFailed          int64 `json:"evictions_failed"`
			EvictionsLeaseSkip       int64 `json:"evictions_lease_skip"`
			EvictionsRebalance       int64 `json:"evictions_rebalance"`
			MigrationActive          int32 `json:"migration_active"`
			MigrationDurationSeconds float64 `json:"migration_last_duration_seconds"`
			MigrationCompletedTotal  int64 `json:"migration_completed_total"`
			MigrationLastEntries     int64 `json:"migration_last_entries"`
			MigrationPreRegSeconds   float64 `json:"migration_prereg_wait_seconds"`
			MigrationP99Us           int64 `json:"migration_p99_us,omitempty"`
			MigrationBatchSize       int32 `json:"migration_batch_size,omitempty"`
		}

		shardEntries := make([]shardEntry, len(shards))
		for i, m := range shards {
			migActive := m.MigrationActive()
			se := shardEntry{
				ID:                       m.shardID,
				Gets:                     m.Gets(),
				Sets:                     m.Sets(),
				Evictions:                m.Evictions(),
				BytesIn:                  m.BytesIn(),
				EvictionsKeyPressure:     m.EvictionsKeyPressure(),
				EvictionsValuePressure:   m.EvictionsValuePressure(),
				EvictionsFailed:          m.EvictionsFailed(),
				EvictionsLeaseSkip:       m.EvictionsLeaseSkip(),
				EvictionsRebalance:       m.EvictionsRebalance(),
				MigrationActive:          migActive,
				MigrationDurationSeconds: float64(m.MigrationDurationMs()) / 1000,
				MigrationCompletedTotal:  m.MigrationsTotal(),
				MigrationLastEntries:     m.MigrationEntries(),
				MigrationPreRegSeconds:   float64(m.MigrationPreRegWaitMs()) / 1000,
			}
			if migActive != 0 {
				se.MigrationP99Us = m.MigrationP99Us()
				se.MigrationBatchSize = m.MigrationBatchSize()
			}
			shardEntries[i] = se
		}
		result["shards"] = shardEntries
		result["migrations_in_progress"] = totals.migrationsInProgress
	}

	// ================================================================
	// Section 6: Wire Protocol
	// ================================================================

	if c.connReg != nil {
		wireBytesRecv := c.connReg.TotalBytesRecv()
		wireBytesSent := c.connReg.TotalBytesSent()
		const gb = 1 << 30

		wire := map[string]interface{}{
			"gb_recv_total": float64(wireBytesRecv) / gb,
			"gb_sent_total": float64(wireBytesSent) / gb,
		}

		if totals != nil {
			inlinePayloadSent := totals.bytesOut - totals.rdmaReadBytesOut
			effectiveSent := wireBytesSent + totals.rdmaReadBytesOut
			opsRecv := wireBytesRecv - totals.bytesIn
			opsSent := wireBytesSent - inlinePayloadSent

			wire["effective_gb_sent_total"] = float64(effectiveSent) / gb
			wire["payload_gb_recv"] = float64(totals.bytesIn) / gb
			wire["payload_gb_sent"] = float64(inlinePayloadSent) / gb
			wire["ops_gb_recv"] = float64(opsRecv) / gb
			wire["ops_gb_sent"] = float64(opsSent) / gb

			epochSec := time.Since(c.connReg.StatsEpoch()).Seconds()
			if epochSec > 0 {
				gbIn := float64(totals.bytesIn) / gb
				gbOut := float64(totals.bytesOut) / gb
				wire["throughput_gbps_in"] = gbIn / epochSec
				wire["throughput_gbps_out"] = gbOut / epochSec
				wire["throughput_gbps_total"] = (gbIn + gbOut) / epochSec
			}
		}

		result["wire"] = wire
	}

	// ================================================================
	// Section 7: RDMA
	// ================================================================

	rdma := make(map[string]interface{})

	// Per-NIC wire bytes
	if c.connReg != nil {
		nicIPs := c.connReg.NICIPs()
		nicBytes := c.connReg.NICWireBytes()
		if len(nicIPs) > 0 {
			nics := make([]map[string]interface{}, 0, len(nicIPs))
			for dev, ip := range nicIPs {
				entry := map[string]interface{}{
					"device": dev,
					"ip":     ip,
				}
				if wb, ok := nicBytes[dev]; ok {
					entry["wire_bytes_recv"] = wb[0]
					entry["wire_bytes_sent"] = wb[1]
					entry["wire_gb_total"] = float64(wb[0]+wb[1]) / (1 << 30)
				}
				nics = append(nics, entry)
			}
			rdma["nics"] = nics
		}
	}

	// CQ poller stats
	if c.PollerMetrics != nil {
		snaps := c.PollerMetrics()
		if len(snaps) > 0 {
			pollers := make([]map[string]interface{}, len(snaps))
			for i, ps := range snaps {
				pollers[i] = map[string]interface{}{
					"device":       ps.Device,
					"active_conns": ps.ActiveConns,
					"completions":  ps.Completions,
				}
			}
			rdma["pollers"] = pollers
		}
	}

	// RDMA Read tracking
	if c.RDMAReadMetrics != nil {
		snap := c.RDMAReadMetrics()
		rdma["reads"] = map[string]int64{
			"issued":                snap.Issued,
			"confirmed":             snap.Confirmed,
			"failed":                snap.Failed,
			"forced_inline":         snap.ForcedInline,
			"lease_drops":           snap.LeaseDrops,
			"mget_migration_skips":  snap.MgetMigrationSkips,
		}
	}

	if len(rdma) > 0 {
		result["rdma"] = rdma
	}

	// ================================================================
	// Section 8: Slab Allocator
	// ================================================================

	if c.SlabMetrics != nil {
		classes, detection := c.SlabMetrics()
		slab := make(map[string]interface{})

		if len(classes) > 0 {
			slabClasses := make([]map[string]interface{}, len(classes))
			for i, cls := range classes {
				slabClasses[i] = map[string]interface{}{
					"size":             cls.Size,
					"total_slots":      cls.TotalSlots,
					"used_slots":       cls.UsedSlots,
					"alloc_count":      cls.AllocCount,
					"slot_utilization": cls.SlotUtilization,
				}
			}
			slab["classes"] = slabClasses
		}

		if detection.Detected {
			slab["detected_page_bytes"] = detection.DetectedPageBytes
		}
		if detection.ConfiguredPageBytes > 0 {
			slab["model_page_bytes"] = detection.ConfiguredPageBytes
			slab["model_effective_gb"] = float64(detection.ModelTotalSlots) * float64(detection.ModelClassSize) / (1 << 30)
		}

		result["slab"] = slab
	}

	// Pressure metrics (part of slab section)
	if c.PressureMetrics != nil {
		psnaps := c.PressureMetrics()
		if len(psnaps) > 0 {
			pressure := make([]map[string]interface{}, len(psnaps))
			for i, p := range psnaps {
				pressure[i] = map[string]interface{}{
					"size":            p.Size,
					"pressure":        p.Pressure,
					"eviction_rate":   p.EvictionRate,
					"alloc_fail_rate": p.AllocFailRate,
					"current_weight":  p.CurrentWeight,
				}
			}
			result["pressure"] = pressure
		}
	}

	// Shard pressure metrics (per-shard per-class)
	if c.ShardPressureMetrics != nil {
		spSnaps := c.ShardPressureMetrics()
		if len(spSnaps) > 0 {
			shardPressure := make([]map[string]interface{}, len(spSnaps))
			for i, sp := range spSnaps {
				shardPressure[i] = map[string]interface{}{
					"shard_id":    sp.ShardID,
					"class_size":  sp.ClassSize,
					"alloc_ops":   sp.AllocOps,
					"alloc_fails": sp.AllocFails,
					"evictions":   sp.Evictions,
				}
			}
			result["shard_pressure"] = shardPressure
		}
	}

	// ================================================================
	// Section 9: Per-Connection
	// ================================================================

	conns := c.connReg.Snapshot()
	if len(conns) > 0 {
		connEntries := make([]map[string]interface{}, len(conns))
		for i, cm := range conns {
			connEntries[i] = map[string]interface{}{
				"transport":  cm.Transport,
				"remote":     cm.RemoteAddr,
				"device":     cm.Device,
				"bytes_recv": cm.BytesRecv(),
				"bytes_sent": cm.BytesSent(),
				"requests":   cm.Requests(),
			}
		}
		result["connections"] = connEntries
	}

	// ================================================================
	// Section 10: Vacuum Coordinator
	// ================================================================

	if c.VacuumMetrics != nil {
		vs := c.VacuumMetrics()
		vacuum := map[string]interface{}{
			"rebalances_total":    vs.RebalancesTotal,
			"pending_shards":     vs.PendingShards,
			"pressure_evals":     vs.PressureEvals,
			"pressure_rebuilds":  vs.PressureRebuilds,
			"max_drift":          vs.MaxDrift,
			"rebalance_failures": vs.RebalanceFailures,
		}
		if vs.LastRebalanceEpoch > 0 {
			vacuum["last_rebalance_seconds_ago"] = time.Since(time.Unix(vs.LastRebalanceEpoch, 0)).Seconds()
		}
		result["vacuum"] = vacuum
	}

	// ================================================================
	// Section 11: System Health
	// ================================================================

	if c.SystemHealth != nil {
		h := c.SystemHealth()
		health := map[string]interface{}{
			"mem_available_bytes": h.MemAvailableBytes,
			"mem_psi_some_bp":    h.MemPSISomeBP,
			"mem_psi_full_bp":    h.MemPSIFullBP,
			// Memory accounting: process_rss_bytes is VmRSS (regular 4KB pages only),
			// process_hugetlb_bytes is hugepage-backed slab memory, and
			// process_total_rss_bytes = VmRSS + HugeTLB (matches htop RES column).
			// On a healthy system with hugepages, rss_bytes is small and hugetlb_bytes
			// is large â€” this is expected, not a bug.
			"process_total_rss_bytes": h.ProcessTotalRSSBytes,
			"process_rss_bytes":       h.ProcessRSSBytes,
			"process_hugetlb_bytes":   h.ProcessHugetlbBytes,
			"cpu_user_seconds":           h.CPUUserSeconds,
			"cpu_system_seconds":         h.CPUSystemSeconds,
			"goroutines":                 h.Goroutines,
			"threads":                    h.Threads,
			"voluntary_ctx_switches":     h.VoluntaryCtxSwitches,
			"involuntary_ctx_switches":   h.InvoluntaryCtxSwitches,
		}
		if h.FDLimit > 0 {
			health["fd_ratio"] = float64(h.FDCount) / float64(h.FDLimit)
		}
		if len(h.NICPortActive) > 0 {
			health["nic_port_active"] = h.NICPortActive
		}
		if len(h.NICHWErrors) > 0 {
			health["nic_hw_errors"] = h.NICHWErrors
		}
		result["system_health"] = health
	}

	// ================================================================
	// Section 12: Startup Progress
	// ================================================================

	result["startup"] = map[string]interface{}{
		"shards_ready":     c.startup.ShardsReady.Load(),
		"shards_total":     c.startup.ShardsTotal.Load(),
		"mem_reserved_bytes": c.startup.MemReservedBytes.Load(),
		"mem_total_bytes":  c.startup.MemTotalBytes.Load(),
	}

	// ================================================================
	// Section 13: Client-Reported Stats
	// ================================================================

	if c.ClientStatsReg != nil {
		clientSnap := c.ClientStatsReg.Snapshot()
		if len(clientSnap) > 0 {
			now := time.Now()
			clients := make([]map[string]interface{}, len(clientSnap))
			for i, cs := range clientSnap {
				entry := map[string]interface{}{
					"remote":              cs.RemoteAddr,
					"last_report_seconds": now.Sub(cs.LastUpdated).Seconds(),
					// Core errors/ops
					"get_errors":      cs.GetErrors,
					"get_successes":   cs.GetSuccesses,
					"set_errors":      cs.SetErrors,
					"exists_errors":   cs.ExistsErrors,
					"exists_timeouts": cs.ExistsTimeouts,
					"io_workers":      cs.IoWorkers,
					"avg_io_batch_size": cs.AvgIoBatchSize,
					"avg_io_latency_ms": cs.AvgIoLatencyMs,
					"io_calls":        cs.IoCalls,
					// SGLang metrics
					"cache_hit_rate":          cs.CacheHitRate,
					"token_usage":             cs.TokenUsage,
					"num_running_reqs":        cs.NumRunningReqs,
					"evictable_ratio":         cs.EvictableRatio,
					"gen_throughput":          cs.GenThroughput,
					"prefetch_bandwidth_gbps": cs.PrefetchBandwidthGbps,
					// RDMA Read reliability
					"rdma_read_retries":  cs.RdmaReadRetries,
					"rdma_read_failures": cs.RdmaReadFailures,
					// C++ timing
					"avg_roundtrip_us": cs.AvgRoundtripUs,
					"avg_rdma_read_us": cs.AvgRdmaReadUs,
					"roundtrip_count":  cs.RoundtripCount,
					"rdma_read_count":  cs.RdmaReadCount,
					// In-flight
					"inflight_gets": cs.InFlightGets,
					"inflight_sets": cs.InFlightSets,
					// Connector phase timing
					"avg_preprocess_ms":  cs.AvgPreprocessMs,
					"avg_exists_ms":      cs.AvgExistsMs,
					"avg_transfer_ms":    cs.AvgTransferMs,
					"avg_postprocess_ms": cs.AvgPostprocessMs,
					// Backup thread
					"host_alloc_drops":       cs.HostAllocDrops,
					"backup_queue_depth":     cs.BackupQueueDepth,
					"backup_ops_completed":   cs.BackupOpsCompleted,
					"backup_ops_failed":      cs.BackupOpsFailed,
					"backup_in_flight":       cs.BackupInFlight,
					"backup_avg_latency_ms":  cs.BackupAvgLatencyMs,
					"backup_jitter_total_ms": cs.BackupJitterTotalMs,
					"backup_avg_gap_ms":      cs.BackupAvgGapMs,
					"backup_jitter_cfg_ms":   cs.BackupJitterCfgMs,
					"backup_batch_size":      cs.BackupBatchSize,
					// Prefetch thread
					"prefetch_received":     cs.PrefetchReceived,
					"prefetch_dispatched":   cs.PrefetchDispatched,
					"prefetch_revoked":      cs.PrefetchRevoked,
					"prefetch_io_completed": cs.PrefetchIoCompleted,
					"prefetch_io_failed":    cs.PrefetchIoFailed,
					"prefetch_tokens":       cs.PrefetchTokens,
					"prefetch_pages":        cs.PrefetchPages,
					"prefetch_avg_io_ms":    cs.PrefetchAvgIoMs,
					"prefetch_queue_depth":  cs.PrefetchQueueDepth,
					"prefetch_in_flight":    cs.PrefetchInFlight,
					// Connector config/state
					"reconnect_count":          cs.ReconnectCount,
					"pool_size":                cs.PoolSize,
					"dedup_low_hit_streak":     cs.DedupLowHitStreak,
					"total_pages_set":          cs.TotalPagesSet,
					"total_pages_get":          cs.TotalPagesGet,
					"backup_coalesce_avg_ops":  cs.BackupCoalesceAvgOps,
					"dedup_mode":               cs.DedupMode,
					"dedup_auto_disabled":      cs.DedupAutoDisabled,
				}
				// Latency histogram
				if len(cs.IoLatencyHistBuckets) > 0 && len(cs.IoLatencyHistCounts) > 0 {
					entry["io_latency_hist_buckets"] = cs.IoLatencyHistBuckets
					entry["io_latency_hist_counts"] = cs.IoLatencyHistCounts
					entry["io_latency_hist_sum"] = cs.IoLatencyHistSum
					entry["io_latency_hist_total"] = cs.IoLatencyHistTotal
				}
				clients[i] = entry
			}
			result["clients"] = clients
		}
	}

	// ================================================================
	// Connection buffer bytes (RDMA)
	// ================================================================

	if c.ConnBufBytesFunc != nil {
		result["conn_buf_bytes"] = c.ConnBufBytesFunc()
	}

	// ================================================================
	// Section 14: Op Latency
	// ================================================================

	if c.OpLatencyMetrics != nil {
		snaps := c.OpLatencyMetrics()
		if len(snaps) > 0 {
			latencyEntries := make([]map[string]interface{}, len(snaps))
			for i, snap := range snaps {
				latencyEntries[i] = map[string]interface{}{
					"shard_id":      snap.ShardID,
					"all_p50_us":    snap.AllP50Us,
					"all_p99_us":    snap.AllP99Us,
					"get_p50_us":    snap.GetP50Us,
					"get_p99_us":    snap.GetP99Us,
					"set_p50_us":    snap.SetP50Us,
					"set_p99_us":    snap.SetP99Us,
					"exists_p50_us":      snap.ExistsP50Us,
					"exists_p99_us":      snap.ExistsP99Us,
					"queue_wait_p50_us":  snap.QueueWaitP50Us,
					"queue_wait_p99_us":  snap.QueueWaitP99Us,
					"alloc_dur_p50_us":   snap.AllocDurP50Us,
					"alloc_dur_p99_us":   snap.AllocDurP99Us,
					"queue_depth":        snap.QueueDepth,
					"queue_cap":          snap.QueueCap,
				}
			}
			result["op_latency"] = latencyEntries
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}
