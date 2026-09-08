package metrics

import (
	"fmt"
	"strings"
)

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
		t.opsDropped += m.OpsDropped()                // already lifetime â€” never reset
		t.migrationsInProgress += m.MigrationActive() // gauge, not a counter
		t.panics += m.Panics()                        // logically lifetime â€” never reset
		t.circuitTrips += m.CircuitTrips()            // logically lifetime â€” never reset
		t.shardHalted += m.ShardHalted()              // gauge, not a counter
	}
	return t
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
