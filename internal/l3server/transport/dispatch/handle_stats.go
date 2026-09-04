package dispatch

import (
	"encoding/json"
	"runtime"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
	"github.com/anthony-chaudhary/fak/internal/l3server/version"
)

// statsShardEntry holds per-shard statistics for JSON serialization.
type statsShardEntry struct {
	ShardID                int   `json:"shard_id"`
	Gets                   int64 `json:"gets"`
	Sets                   int64 `json:"sets"`
	Exists                 int64 `json:"exists"`
	ExistsHits             int64 `json:"exists_hits"`
	ExistsMisses           int64 `json:"exists_misses"`
	Hits                   int64 `json:"hits"`
	Misses                 int64 `json:"misses"`
	Evictions              int64 `json:"evictions"`
	EvictionsKeyPressure   int64 `json:"evictions_key_pressure"`
	EvictionsValuePressure int64 `json:"evictions_value_pressure"`
	EvictionsFailed        int64 `json:"evictions_failed"`
	EvictionsLeaseSkip     int64 `json:"evictions_lease_skip"`
	EvictionsRebalance     int64 `json:"evictions_rebalance"`
	TTLExpirations         int64 `json:"ttl_expirations"`
	BytesIn                int64 `json:"bytes_in"`
	BytesOut               int64 `json:"bytes_out"`
	RDMAReadBytesOut       int64 `json:"rdma_read_bytes_out"`
}

// statsConnEntry holds per-connection statistics for JSON serialization.
type statsConnEntry struct {
	ID          uint64 `json:"id"`
	Transport   string `json:"transport"`
	RemoteAddr  string `json:"remote_addr"`
	BytesRecv   int64  `json:"bytes_recv"`
	BytesSent   int64  `json:"bytes_sent"`
	Requests    int64  `json:"requests"`
	ConnectedAt string `json:"connected_at"`
}

// statsNICEntry holds per-NIC wire and CQ poller statistics for JSON serialization.
type statsNICEntry struct {
	Device           string  `json:"device"`
	IP               string  `json:"ip"`
	WireGBTotal      float64 `json:"wire_gb_total"`
	WireGBRecv       float64 `json:"wire_gb_recv"`
	WireGBSent       float64 `json:"wire_gb_sent"`
	ThroughputGbps   float64 `json:"throughput_gbps"`
	ActiveConns      int32   `json:"active_conns"`
	Completions      int64   `json:"completions"`
	DispatchEnqueued int64   `json:"dispatch_enqueued"`
	DispatchDropped  int64   `json:"dispatch_dropped"`
	SendChDropped    int64   `json:"send_ch_dropped"`
	DispatchWorkers       int     `json:"dispatch_workers"`
	LinkRateGbps          float64 `json:"link_rate_gbps"`
	SaturationPct         float64 `json:"saturation_pct"`
	DispatchSaturationPct float64 `json:"dispatch_saturation_pct,omitempty"`
}

// statsTotals holds aggregate statistics for JSON serialization.
// Byte totals are converted to gigabytes (GB, base-2 / GiB) for display.
type statsTotals struct {
	GBIn                float64 `json:"gb_in"`
	GBOut               float64 `json:"gb_out"`
	RDMAReadGBOut       float64 `json:"rdma_read_gb_out"`
	WireGBRecv          float64 `json:"wire_gb_recv"`
	WireGBSent          float64 `json:"wire_gb_sent"`
	EffectiveGBSent     float64 `json:"effective_gb_sent"`
	InlinePayloadGBSent float64 `json:"inline_payload_gb_sent"`
	OpsOverheadGBRecv   float64 `json:"ops_overhead_gb_recv"`
	OpsOverheadGBSent   float64 `json:"ops_overhead_gb_sent"`
	GBpsIn              float64 `json:"gbps_in"`
	GBpsOut             float64 `json:"gbps_out"`
	GBpsTotal           float64 `json:"gbps_total"`
	Gets                int64   `json:"gets"`
	Sets                int64   `json:"sets"`
	Evictions              int64   `json:"evictions"`
	EvictionsKeyPressure   int64   `json:"evictions_key_pressure"`
	EvictionsValuePressure int64   `json:"evictions_value_pressure"`
	EvictionsFailed        int64   `json:"evictions_failed"`
	EvictionsLeaseSkip     int64   `json:"evictions_lease_skip"`
	EvictionsRebalance     int64   `json:"evictions_rebalance"`
	EvictionsOrganic       int64   `json:"evictions_organic"`
	TTLExpirations         int64   `json:"ttl_expirations"`
	Deletes                int64   `json:"deletes"`
	Exists              int64   `json:"exists"`
	ExistsHits          int64   `json:"exists_hits"`
	ExistsMisses        int64   `json:"exists_misses"`
	Hits                int64   `json:"hits"`
	Misses              int64   `json:"misses"`
	ActiveGB            float64 `json:"active_gb"`
	Entries             int64   `json:"entries"`
	EvictionRatePercent     float64 `json:"eviction_rate_percent"`
	EvictionFailRatePercent float64 `json:"eviction_fail_rate_percent"`
	ExistsHitRatePercent float64 `json:"exists_hit_rate_percent"`
	GetHitRatePercent    float64 `json:"get_hit_rate_percent"`
	ActiveConnections   int     `json:"active_connections"`
	InflightOps             int64   `json:"inflight_ops"`
	MaxKeys                 int64   `json:"max_keys"`
	EvictionCap             int64   `json:"eviction_cap"`
	KeyUtilizationPercent   float64 `json:"key_utilization_percent"`
	AvgKeyBytes             float64 `json:"avg_key_bytes"`
	AvgValueBytes           float64 `json:"avg_value_bytes"`
	MaxMemoryGB             int     `json:"max_memory_gb"`
	NICBalancePct           float64 `json:"nic_balance_pct"`
	WireSaturationPct       float64 `json:"wire_saturation_pct"`
	RDMAWireGBRecv          float64 `json:"rdma_wire_gb_recv"`
	RDMAWireGBSent          float64 `json:"rdma_wire_gb_sent"`
	RDMAWireGBTotal         float64 `json:"rdma_wire_gb_total"`
	RDMAThroughputGbps      float64 `json:"rdma_throughput_gbps"`
	RDMALinkRateGbpsTotal   float64 `json:"rdma_link_rate_gbps_total"`
}

// statsSlabClassEntry holds per-class slab utilization for JSON serialization.
type statsSlabClassEntry struct {
	Size            uint64  `json:"size"`
	TotalSlots      uint64  `json:"total_slots"`
	UsedSlots       uint64  `json:"used_slots"`
	AllocCount      int64   `json:"alloc_count"`
	AvgRequestBytes float64 `json:"avg_request_bytes"`
	SlotUtilization float64 `json:"slot_utilization"`
}

// sglangClientEntry holds per-client SGLang metrics for JSON serialization.
type sglangClientEntry struct {
	ConnID                uint64  `json:"conn_id"`
	RemoteAddr            string  `json:"remote_addr"`
	CacheHitRate          float64 `json:"cache_hit_rate"`
	TokenUsage            float64 `json:"token_usage"`
	NumRunningReqs        int64   `json:"num_running_reqs"`
	EvictableRatio        float64 `json:"evictable_ratio"`
	GenThroughput         float64 `json:"gen_throughput"`
	PrefetchBandwidthGbps float64 `json:"prefetch_bandwidth_gbps"`
}

// collectShardStats gathers per-shard metrics and computes aggregate totals.
func (d *Dispatcher) collectShardStats() ([]statsShardEntry, statsTotals) {
	n := d.Manager.NumShards()
	shards := make([]statsShardEntry, n)
	var totals statsTotals
	var totalBytesIn, totalBytesOut, totalRDMAReadBytesOut int64
	var totalKeyBytesIn, totalValueBytesIn int64
	var totalAllocBytes int64
	var totalEntries int64

	for i := 0; i < n; i++ {
		sh := d.Manager.Shard(i)
		m := sh.Metrics()
		s := statsShardEntry{
			ShardID:                i,
			Gets:                   m.Gets(),
			Sets:                   m.Sets(),
			Exists:                 m.Exists(),
			ExistsHits:             m.ExistsHits(),
			ExistsMisses:           m.ExistsMisses(),
			Hits:                   m.Hits(),
			Misses:                 m.Misses(),
			Evictions:              m.Evictions(),
			EvictionsKeyPressure:   m.EvictionsKeyPressure(),
			EvictionsValuePressure: m.EvictionsValuePressure(),
			EvictionsFailed:        m.EvictionsFailed(),
			EvictionsLeaseSkip:     m.EvictionsLeaseSkip(),
			EvictionsRebalance:     m.EvictionsRebalance(),
			TTLExpirations:         m.TTLExpirations(),
			BytesIn:                m.BytesIn(),
			BytesOut:               m.BytesOut(),
			RDMAReadBytesOut:       m.RDMAReadBytesOut(),
		}
		shards[i] = s
		totals.Gets += s.Gets
		totals.Sets += s.Sets
		totals.Evictions += s.Evictions
		totals.EvictionsKeyPressure += s.EvictionsKeyPressure
		totals.EvictionsValuePressure += s.EvictionsValuePressure
		totals.EvictionsFailed += s.EvictionsFailed
		totals.EvictionsLeaseSkip += s.EvictionsLeaseSkip
		totals.EvictionsRebalance += s.EvictionsRebalance
		totals.TTLExpirations += s.TTLExpirations
		totals.Deletes += m.Deletes()
		totals.Exists += s.Exists
		totals.ExistsHits += s.ExistsHits
		totals.ExistsMisses += s.ExistsMisses
		totals.Hits += s.Hits
		totals.Misses += s.Misses
		totalBytesIn += s.BytesIn
		totalBytesOut += s.BytesOut
		totalRDMAReadBytesOut += s.RDMAReadBytesOut
		totalKeyBytesIn += m.KeyBytesIn()
		totalValueBytesIn += m.ValueBytesIn()
		totalAllocBytes += sh.Allocator().AllocatedBytes()
		totalEntries += int64(sh.IndexCount())
	}

	totals.GBIn = float64(totalBytesIn) / bytesPerGB
	totals.GBOut = float64(totalBytesOut) / bytesPerGB
	totals.RDMAReadGBOut = float64(totalRDMAReadBytesOut) / bytesPerGB
	totals.ActiveGB = float64(totalAllocBytes) / bytesPerGB
	totals.Entries = totalEntries

	// Eviction breakdown: organic = total - rebalance
	totals.EvictionsOrganic = totals.Evictions - totals.EvictionsRebalance
	if totals.EvictionsOrganic < 0 {
		totals.EvictionsOrganic = 0
	}

	// Key capacity (IndexCapacity = max_keys per shard)
	if d.Manager.NumShards() > 0 {
		cfg := d.Manager.Shard(0).Config()
		maxKeys := int64(cfg.IndexCapacity) * int64(d.Manager.NumShards())
		evictionCap := maxKeys * index.MaxLoadNumerator / index.MaxLoadDenominator
		totals.MaxKeys = maxKeys
		totals.EvictionCap = evictionCap
		if evictionCap > 0 {
			totals.KeyUtilizationPercent = float64(totalEntries) / float64(evictionCap) * 100
		}
		totals.MaxMemoryGB = int(cfg.MaxMemoryBytes * uint64(d.Manager.NumShards()) / (1 << 30))
	}

	if totals.Sets > 0 {
		totals.AvgKeyBytes = float64(totalKeyBytesIn) / float64(totals.Sets)
		totals.AvgValueBytes = float64(totalValueBytesIn) / float64(totals.Sets)
	}

	return shards, totals
}

// collectConnStats gathers per-connection metrics, updates wire-level totals,
// and returns per-NIC stats.
func (d *Dispatcher) collectConnStats(totals *statsTotals) ([]statsConnEntry, []statsNICEntry) {
	if d.ConnReg == nil {
		return nil, nil
	}
	snap := d.ConnReg.Snapshot()
	conns := make([]statsConnEntry, len(snap))
	for i, cm := range snap {
		conns[i] = statsConnEntry{
			ID:          cm.ID,
			Transport:   cm.Transport,
			RemoteAddr:  cm.RemoteAddr,
			BytesRecv:   cm.BytesRecv(),
			BytesSent:   cm.BytesSent(),
			Requests:    cm.Requests(),
			ConnectedAt: cm.ConnectedAt.Format(time.RFC3339),
		}
	}
	totals.WireGBRecv = float64(d.ConnReg.TotalBytesRecv()) / bytesPerGB
	totals.WireGBSent = float64(d.ConnReg.TotalBytesSent()) / bytesPerGB
	totals.ActiveConnections = d.ConnReg.ActiveCount()
	totals.InflightOps = GlobalInflightOps()

	// Derived: ops/payload framing
	totals.InlinePayloadGBSent = totals.GBOut - totals.RDMAReadGBOut
	totals.EffectiveGBSent = totals.WireGBSent + totals.RDMAReadGBOut
	totals.OpsOverheadGBRecv = totals.WireGBRecv - totals.GBIn
	totals.OpsOverheadGBSent = totals.WireGBSent - totals.InlinePayloadGBSent

	// Per-NIC stats
	nicIPs := d.ConnReg.NICIPs()
	nicBytes := d.ConnReg.NICWireBytes()

	// Build poller stats lookup (device â†’ snapshot)
	pollerByDevice := make(map[string]metrics.PollerSnapshot)
	if d.PollerMetrics != nil {
		for _, ps := range d.PollerMetrics() {
			pollerByDevice[ps.Device] = ps
		}
	}

	nicLinkRates := d.ConnReg.NICLinkRates()
	nicThroughput := d.ConnReg.NICThroughputGbps()

	var nics []statsNICEntry
	for dev, ip := range nicIPs {
		total := nicBytes[dev]
		entry := statsNICEntry{
			Device:      dev,
			IP:          ip,
			WireGBTotal: float64(total[0]+total[1]) / bytesPerGB,
			WireGBRecv:  float64(total[0]) / bytesPerGB,
			WireGBSent:  float64(total[1]) / bytesPerGB,
		}
		entry.ThroughputGbps = nicThroughput[dev]
		entry.LinkRateGbps = nicLinkRates[dev]
		if nicLinkRates[dev] > 0 {
			entry.SaturationPct = nicThroughput[dev] / nicLinkRates[dev] * 100
			if entry.SaturationPct > 100 {
				entry.SaturationPct = 100
			}
		}
		if ps, ok := pollerByDevice[dev]; ok {
			entry.ActiveConns = ps.ActiveConns
			entry.Completions = ps.Completions
			entry.DispatchEnqueued = ps.DispatchEnqueued
			entry.DispatchDropped = ps.DispatchDropped
			entry.SendChDropped = ps.SendChDropped
			entry.DispatchWorkers = ps.DispatchWorkers
			if ps.DispatchQueueCap > 0 {
				entry.DispatchSaturationPct = float64(ps.DispatchQueueDepth) / float64(ps.DispatchQueueCap) * 100
			}
		}
		nics = append(nics, entry)
	}

	// Compute aggregate RDMA wire totals and saturation
	var minSat, maxSat, totalTput, totalCap float64
	first := true
	for _, n := range nics {
		totals.RDMAWireGBRecv += n.WireGBRecv
		totals.RDMAWireGBSent += n.WireGBSent
		totals.RDMAWireGBTotal += n.WireGBTotal
		if first || n.SaturationPct < minSat {
			minSat = n.SaturationPct
		}
		if first || n.SaturationPct > maxSat {
			maxSat = n.SaturationPct
		}
		totalTput += nicThroughput[n.Device]
		totalCap += nicLinkRates[n.Device]
		first = false
	}
	totals.RDMAThroughputGbps = totalTput
	totals.RDMALinkRateGbpsTotal = totalCap
	if maxSat > 0 {
		totals.NICBalancePct = minSat / maxSat * 100
	} else {
		totals.NICBalancePct = 100
	}
	if totalCap > 0 {
		totals.WireSaturationPct = totalTput / totalCap * 100
	}

	return conns, nics
}

// collectSlabStats gathers per-class slab utilization and value-size detection state.
func (d *Dispatcher) collectSlabStats() ([]statsSlabClassEntry, interface{}) {
	n := d.Manager.NumShards()
	if n == 0 {
		return nil, nil
	}

	// Use shard 0 as the structural reference and aggregate across all shards
	utils0 := d.Manager.Shard(0).Allocator().ClassUtilizations()
	classes := make([]statsSlabClassEntry, len(utils0))
	for i, u := range utils0 {
		classes[i] = statsSlabClassEntry{
			Size:            u.Size,
			TotalSlots:      u.TotalSlots,
			UsedSlots:       u.UsedSlots,
			AllocCount:      u.AllocCount,
			AvgRequestBytes: u.AvgRequestBytes,
			SlotUtilization: u.SlotUtilization * 100,
		}
	}
	for si := 1; si < n; si++ {
		su := d.Manager.Shard(si).Allocator().ClassUtilizations()
		for i := range su {
			if i < len(classes) {
				classes[i].TotalSlots += su[i].TotalSlots
				classes[i].UsedSlots += su[i].UsedSlots
				classes[i].AllocCount += su[i].AllocCount
			}
		}
	}

	// Value-size detection: use shard 0 as representative
	detection := d.Manager.Shard(0).SizeDetectionSnapshot()
	return classes, detection
}

// collectSGLangMetrics builds a per-client snapshot of SGLang metrics for the stats JSON.
func collectSGLangMetrics(snap []*metrics.ClientStats) []sglangClientEntry {
	entries := make([]sglangClientEntry, 0, len(snap))
	for _, cs := range snap {
		entries = append(entries, sglangClientEntry{
			ConnID:                cs.ConnID,
			RemoteAddr:            cs.RemoteAddr,
			CacheHitRate:          cs.CacheHitRate,
			TokenUsage:            cs.TokenUsage,
			NumRunningReqs:        cs.NumRunningReqs,
			EvictableRatio:        cs.EvictableRatio,
			GenThroughput:         cs.GenThroughput,
			PrefetchBandwidthGbps: cs.PrefetchBandwidthGbps,
		})
	}
	return entries
}

// HandleReportStats processes a client-reported stats message.
// connID and remoteAddr are provided by the transport layer (TCP/RDMA).
func (d *Dispatcher) HandleReportStats(msg protocol.Message, connID uint64, remoteAddr string) protocol.Message {
	if d.ClientStatsReg == nil {
		return OKResponse(msg.Header.RequestID)
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(msg.Body, &payload); err != nil {
		return ErrResponse(msg.Header.RequestID, "invalid stats JSON: "+err.Error())
	}
	d.ClientStatsReg.Update(connID, remoteAddr, payload)

	// Extract model_page_bytes hint and broadcast to all shards
	if mpb, ok := payload["model_page_bytes"].(float64); ok && mpb > 0 && d.Manager != nil {
		d.Manager.SetModelPageHint(uint64(mpb))
	}

	return OKResponse(msg.Header.RequestID)
}

// HandleStats returns detailed bandwidth and connection statistics as JSON.
func (d *Dispatcher) HandleStats(msg protocol.Message) protocol.Message {
	// Early-connect path: Manager not yet wired â€” return minimal stats
	if d.Manager == nil {
		var totals statsTotals
		connections, rdmaNICs := d.collectConnStats(&totals)
		result := map[string]interface{}{
			"shards":         []statsShardEntry{},
			"connections":    connections,
			"totals":         totals,
			"uptime_seconds": int64(time.Since(d.StartedAt).Seconds()),
			"startup_state":  "allocating",
		}
		if len(rdmaNICs) > 0 {
			result["rdma_nics"] = rdmaNICs
		}
		body, err := json.Marshal(result)
		if err != nil {
			return ErrResponse(msg.Header.RequestID, "failed to marshal stats: "+err.Error())
		}
		return protocol.Message{
			Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
			Body:   body,
		}
	}

	shards, totals := d.collectShardStats()
	connections, rdmaNICs := d.collectConnStats(&totals)

	// Throughput: GB/s since stats epoch (start or last flush)
	if d.ConnReg != nil {
		epochSec := time.Since(d.ConnReg.StatsEpoch()).Seconds()
		if epochSec > 0 {
			totals.GBpsIn = totals.GBIn / epochSec
			totals.GBpsOut = totals.GBOut / epochSec
			totals.GBpsTotal = (totals.GBIn + totals.GBOut) / epochSec
		}
	}

	// Derived rates
	if totals.Sets > 0 {
		totals.EvictionRatePercent = float64(totals.Evictions) / float64(totals.Sets) * 100
	}
	evictionAttempts := totals.EvictionsKeyPressure + totals.EvictionsValuePressure + totals.EvictionsFailed
	if evictionAttempts > 0 {
		totals.EvictionFailRatePercent = float64(totals.EvictionsFailed) / float64(evictionAttempts) * 100
	}
	if totals.Exists > 0 {
		totals.ExistsHitRatePercent = float64(totals.ExistsHits) / float64(totals.Exists) * 100
	}
	if totals.Gets > 0 {
		totals.GetHitRatePercent = float64(totals.Hits) / float64(totals.Gets) * 100
	}

	// Collect slab class utilization + value-size detection
	slabClasses, valueSizeDetection := d.collectSlabStats()

	result := map[string]interface{}{
		"shards":               shards,
		"connections":          connections,
		"totals":               totals,
		"uptime_seconds":       int64(time.Since(d.StartedAt).Seconds()),
		"slab_classes":         slabClasses,
		"value_size_detection": valueSizeDetection,
	}
	if len(rdmaNICs) > 0 {
		result["rdma_nics"] = rdmaNICs
	}

	// Model capacity: report if model_page_bytes is configured
	if d.Manager.NumShards() > 0 {
		mpb := d.Manager.Shard(0).Config().ModelPageBytes
		if mpb > 0 {
			slots, classSize := d.Manager.Shard(0).Allocator().ModelClassCapacity(mpb)
			totalSlots := slots * uint64(d.Manager.NumShards())
			result["model_capacity"] = map[string]interface{}{
				"model_page_bytes":    mpb,
				"slots_per_shard":     slots,
				"total_slots":         totalSlots,
				"class_size":          classSize,
				"effective_memory_gb": float64(totalSlots) * float64(classSize) / (1 << 30),
			}
		}
	}

	// Op latency per shard
	if d.OpLatencyMetrics != nil {
		snaps := d.OpLatencyMetrics()
		if len(snaps) > 0 {
			latencyEntries := make([]map[string]interface{}, len(snaps))
			for i, snap := range snaps {
				latencyEntries[i] = map[string]interface{}{
					"shard_id":     snap.ShardID,
					"all_p50_us":   snap.AllP50Us,
					"all_p99_us":   snap.AllP99Us,
					"get_p50_us":   snap.GetP50Us,
					"get_p99_us":   snap.GetP99Us,
					"set_p50_us":   snap.SetP50Us,
					"set_p99_us":   snap.SetP99Us,
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

	// Aggregate SGLang metrics from connected clients
	if d.ClientStatsReg != nil {
		snap := d.ClientStatsReg.Snapshot()
		if len(snap) > 0 {
			result["client_sglang_metrics"] = collectSGLangMetrics(snap)
		}
	}

	body, err := json.Marshal(result)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, "failed to marshal stats: "+err.Error())
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
		Body:   body,
	}
}

// HandleInfo processes an INFO request.
// Returns JSON metadata about the server. Safe to call with nil Manager.
func (d *Dispatcher) HandleInfo(msg protocol.Message) protocol.Message {
	info := map[string]interface{}{
		"server":             "l3server",
		"server_version":     version.ServerVersion,
		"server_commit":      version.Commit,
		"protocol_version":   version.ProtocolVersion,
		"api_version":        version.APIVersion,
		"min_client_version": version.MinClientVersion,
		"min_sglang_version": version.MinSGLangVersion,
		"go_version":         runtime.Version(),
		"uptime_seconds":     int64(time.Since(d.StartedAt).Seconds()),
		"started_at":         d.StartedAt.Format(time.RFC3339),
		"rdma_endpoints":     d.RDMAEndpoints,
		"ready":              d.managerReady.Load(),
	}
	// Add RDMA buffer sizes (so clients can see server-side config)
	if d.RDMASendBufSize > 0 {
		info["rdma_send_buf_mb"] = d.RDMASendBufSize / (1024 * 1024)
	}
	if d.RDMARecvBufSize > 0 {
		info["rdma_recv_buf_mb"] = d.RDMARecvBufSize / (1024 * 1024)
	}
	// Add cluster info
	if d.Ring != nil {
		nodes := d.Ring.Nodes()
		info["cluster_enabled"] = true
		info["cluster_node_id"] = d.LocalID
		info["cluster_node_count"] = len(nodes)
	} else {
		info["cluster_enabled"] = false
	}
	body, err := json.Marshal(info)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, "failed to marshal info: "+err.Error())
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK},
		Body:   body,
	}
}

// HandleCluster returns cluster membership information.
func (d *Dispatcher) HandleCluster(msg protocol.Message) protocol.Message {
	if d.Ring == nil {
		body, _ := json.Marshal(map[string]interface{}{
			"cluster_enabled": false,
			"mode":            "standalone",
		})
		return protocol.Message{
			Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
			Body:   body,
		}
	}

	nodes := d.Ring.Nodes()
	type nodeEntry struct {
		ID      string  `json:"id"`
		Addr    string  `json:"addr"`
		IsLocal bool    `json:"is_local"`
		Alive   bool    `json:"alive"`
		Weight  float64 `json:"weight"`
	}
	entries := make([]nodeEntry, len(nodes))
	for i, n := range nodes {
		entries[i] = nodeEntry{
			ID:      n.ID,
			Addr:    n.Addr,
			IsLocal: n.IsLocal,
			Alive:   n.Alive,
			Weight:  n.Weight,
		}
	}
	result := map[string]interface{}{
		"cluster_enabled": true,
		"local_node_id":   d.LocalID,
		"nodes":           entries,
		"node_count":      len(entries),
	}
	body, err := json.Marshal(result)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, "failed to marshal cluster info: "+err.Error())
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
		Body:   body,
	}
}

// HandleHandshake processes a connection handshake request.
// The client sends JSON with its version info; the server responds with its own.
func (d *Dispatcher) HandleHandshake(msg protocol.Message) protocol.Message {
	// Parse client info (best-effort; malformed JSON is not fatal)
	var clientInfo map[string]interface{}
	if len(msg.Body) > 0 {
		_ = json.Unmarshal(msg.Body, &clientInfo)
	}

	// Check client version compatibility
	clientVersion, _ := clientInfo["client_version"].(string)
	if clientVersion != "" && version.CompareVersions(clientVersion, version.MinClientVersion) < 0 {
		resp := map[string]interface{}{
			"status":             "incompatible",
			"reason":             "client version " + clientVersion + " < minimum " + version.MinClientVersion,
			"api_version":        version.APIVersion,
			"server_version":     version.ServerVersion,
			"min_client_version": version.MinClientVersion,
		}
		body, _ := json.Marshal(resp)
		return protocol.Message{
			Header: protocol.Header{OpCode: protocol.RespError, RequestID: msg.Header.RequestID},
			Body:   body,
		}
	}

	// Build server response
	capabilities := []string{"batch_ops", "rdma_read", "prefix_cache", "cluster", "snapshot", "mget_rdma", "early_connect"}
	resp := map[string]interface{}{
		"status":             "ok",
		"api_version":        version.APIVersion,
		"server_version":     version.ServerVersion,
		"server_commit":      version.Commit,
		"protocol_version":   version.ProtocolVersion,
		"min_client_version": version.MinClientVersion,
		"min_sglang_version": version.MinSGLangVersion,
		"capabilities":       capabilities,
		"rdma_endpoints":     d.RDMAEndpoints,
		"metrics_addr":       d.MetricsAddr,
		"cluster_enabled":    d.Ring != nil,
	}
	// Include startup readiness state
	if !d.managerReady.Load() {
		var shardsReady, shardsTotal int
		if d.Manager != nil {
			shardsReady = d.Manager.ReadyCount()
			shardsTotal = d.Manager.NumShards()
		} else {
			shardsTotal = d.numShards
		}
		resp["ready"] = false
		resp["shards_ready"] = shardsReady
		resp["shards_total"] = shardsTotal
		resp["phase"] = "shard_alloc"
	} else {
		resp["ready"] = true
	}
	if d.Ring != nil {
		resp["cluster_node_id"] = d.LocalID
	}
	body, err := json.Marshal(resp)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, "failed to marshal handshake response: "+err.Error())
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
		Body:   body,
	}
}
