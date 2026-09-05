package metrics

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// MaxNICs is the maximum number of per-NIC stripe metrics tracked.
const MaxNICs = 8

// ClientStats holds per-connection client-reported metrics (from the connector).
type ClientStats struct {
	ConnID      uint64
	RemoteAddr  string
	LastUpdated time.Time

	GetErrors      int64
	GetSuccesses   int64
	SetErrors      int64
	ExistsErrors   int64
	ExistsTimeouts int64
	IoWorkers      int
	AvgIoBatchSize float64
	AvgIoLatencyMs float64
	IoCalls        int64

	CacheHitRate          float64
	TokenUsage            float64
	NumRunningReqs        int64
	EvictableRatio        float64
	GenThroughput         float64
	PrefetchBandwidthGbps float64

	RdmaReadRetries  int64
	RdmaReadFailures int64

	AvgRoundtripUs float64
	AvgRdmaReadUs  float64
	RoundtripCount int64
	RdmaReadCount  int64

	InFlightGets int64
	InFlightSets int64

	AvgPreprocessMs  float64
	AvgExistsMs      float64
	AvgTransferMs    float64
	AvgPostprocessMs float64

	GetAvgCtrlMs      float64
	GetAvgMetaMs      float64
	GetAvgReadMs      float64
	GetAvgAckMs       float64
	GetBatchCount     int64
	GetAvgBytes       float64
	SetAvgSerializeMs float64
	SetAvgSendMs      float64
	SetBatchCount     int64
	SetAvgBytes       float64
	SetAvgSubBatches  float64
	AvgBatchReadUs    float64
	BatchReadCount    int64

	IoLatencyHistBuckets []float64
	IoLatencyHistCounts  []int64
	IoLatencyHistSum     float64
	IoLatencyHistTotal   int64

	HostAllocDrops      int64
	BackupQueueDepth    int64
	BackupOpsCompleted  int64
	BackupOpsFailed     int64
	BackupInFlight      int64
	BackupAvgLatencyMs  float64
	BackupJitterTotalMs float64
	BackupAvgGapMs      float64
	BackupJitterCfgMs   float64
	BackupBatchSize     int64

	PrefetchReceived    int64
	PrefetchDispatched  int64
	PrefetchRevoked     int64
	PrefetchIoCompleted int64
	PrefetchIoFailed    int64
	PrefetchTokens      int64
	PrefetchPages       int64
	PrefetchAvgIoMs     float64
	PrefetchQueueDepth  int64
	PrefetchInFlight    int64

	ReconnectCount    int64
	PoolSize          int64
	DedupMode         string
	DedupAutoDisabled bool
	DedupLowHitStreak int64
	DedupCostStreak   int64

	TotalPagesSet        int64
	TotalPagesGet        int64
	BackupCoalesceAvgOps float64
	Codec                string
	CodecLossy           bool

	PoolRebuilds int64
	Transport    string

	BackupBandwidthGbps      float64
	IoMaxLatencyMs           float64
	WarmupPhase              string
	WarmupEffectiveBatchSize int64

	ModelPageBytes int64

	NICReads        [MaxNICs]int64
	NICBytesGB      [MaxNICs]float64
	NICWrites       [MaxNICs]int64
	NICWriteBytesGB [MaxNICs]float64
	NICCount        int

	StripeCalls       int64
	StripeErrors      int64
	StripeWriteCalls  int64
	StripeWriteErrors int64
	StripeAvgNICs     float64
}

type ClientStatsRegistry struct {
	mu         sync.Mutex
	stats      map[uint64]*ClientStats
	statsPeak  int
	warnedKeys map[string]bool
	Advisor    *TuningAdvisor
}

func NewClientStatsRegistry() *ClientStatsRegistry {
	return &ClientStatsRegistry{
		stats:      make(map[uint64]*ClientStats),
		warnedKeys: make(map[string]bool),
		Advisor:    NewTuningAdvisor(),
	}
}

func (r *ClientStatsRegistry) Update(connID uint64, remoteAddr string, payload map[string]interface{}) {
	r.mu.Lock()
	defer r.mu.Unlock()

	cs, ok := r.stats[connID]
	if !ok {
		cs = &ClientStats{ConnID: connID, RemoteAddr: remoteAddr}
		r.stats[connID] = cs
		if n := len(r.stats); n > r.statsPeak {
			r.statsPeak = n
		}
	}
	cs.LastUpdated = time.Now()
	cs.RemoteAddr = remoteAddr

	if v, ok := payload["get_errors"].(float64); ok {
		cs.GetErrors = int64(v)
	}
	if v, ok := payload["get_successes"].(float64); ok {
		cs.GetSuccesses = int64(v)
	}
	if v, ok := payload["set_errors"].(float64); ok {
		cs.SetErrors = int64(v)
	}
	if v, ok := payload["exists_errors"].(float64); ok {
		cs.ExistsErrors = int64(v)
	}
	if v, ok := payload["exists_timeouts"].(float64); ok {
		cs.ExistsTimeouts = int64(v)
	}
	if v, ok := payload["io_workers"].(float64); ok {
		cs.IoWorkers = int(v)
	}
	if v, ok := payload["avg_io_batch_size"].(float64); ok {
		cs.AvgIoBatchSize = v
	}
	if v, ok := payload["avg_io_latency_ms"].(float64); ok {
		cs.AvgIoLatencyMs = v
	}
	if v, ok := payload["io_calls"].(float64); ok {
		cs.IoCalls = int64(v)
	}

	if v, ok := payload["cache_hit_rate"].(float64); ok {
		cs.CacheHitRate = v
	}
	if v, ok := payload["token_usage"].(float64); ok {
		cs.TokenUsage = v
	}
	if v, ok := payload["num_running_reqs"].(float64); ok {
		cs.NumRunningReqs = int64(v)
	}
	if v, ok := payload["evictable_ratio"].(float64); ok {
		cs.EvictableRatio = v
	}
	if v, ok := payload["gen_throughput"].(float64); ok {
		cs.GenThroughput = v
	}
	if v, ok := payload["prefetch_bandwidth_gbps"].(float64); ok {
		cs.PrefetchBandwidthGbps = v
	}

	if v, ok := payload["rdma_read_retries"].(float64); ok {
		cs.RdmaReadRetries = int64(v)
	}
	if v, ok := payload["rdma_read_failures"].(float64); ok {
		cs.RdmaReadFailures = int64(v)
	}

	if v, ok := payload["avg_roundtrip_us"].(float64); ok {
		cs.AvgRoundtripUs = v
	}
	if v, ok := payload["avg_rdma_read_us"].(float64); ok {
		cs.AvgRdmaReadUs = v
	}
	if v, ok := payload["roundtrip_count"].(float64); ok {
		cs.RoundtripCount = int64(v)
	}
	if v, ok := payload["rdma_read_count"].(float64); ok {
		cs.RdmaReadCount = int64(v)
	}

	if v, ok := payload["inflight_gets"].(float64); ok {
		cs.InFlightGets = int64(v)
	}
	if v, ok := payload["inflight_sets"].(float64); ok {
		cs.InFlightSets = int64(v)
	}

	if v, ok := payload["avg_preprocess_ms"].(float64); ok {
		cs.AvgPreprocessMs = v
	}
	if v, ok := payload["avg_exists_ms"].(float64); ok {
		cs.AvgExistsMs = v
	}
	if v, ok := payload["avg_transfer_ms"].(float64); ok {
		cs.AvgTransferMs = v
	}
	if v, ok := payload["avg_postprocess_ms"].(float64); ok {
		cs.AvgPostprocessMs = v
	}

	if v, ok := payload["get_avg_ctrl_ms"].(float64); ok {
		cs.GetAvgCtrlMs = v
	}
	if v, ok := payload["get_avg_meta_ms"].(float64); ok {
		cs.GetAvgMetaMs = v
	}
	if v, ok := payload["get_avg_read_ms"].(float64); ok {
		cs.GetAvgReadMs = v
	}
	if v, ok := payload["get_avg_ack_ms"].(float64); ok {
		cs.GetAvgAckMs = v
	}
	if v, ok := payload["get_batch_count"].(float64); ok {
		cs.GetBatchCount = int64(v)
	}
	if v, ok := payload["get_avg_bytes"].(float64); ok {
		cs.GetAvgBytes = v
	}
	if v, ok := payload["set_avg_serialize_ms"].(float64); ok {
		cs.SetAvgSerializeMs = v
	}
	if v, ok := payload["set_avg_send_ms"].(float64); ok {
		cs.SetAvgSendMs = v
	}
	if v, ok := payload["set_batch_count"].(float64); ok {
		cs.SetBatchCount = int64(v)
	}
	if v, ok := payload["set_avg_bytes"].(float64); ok {
		cs.SetAvgBytes = v
	}
	if v, ok := payload["set_avg_sub_batches"].(float64); ok {
		cs.SetAvgSubBatches = v
	}
	if v, ok := payload["avg_batch_read_us"].(float64); ok {
		cs.AvgBatchReadUs = v
	}
	if v, ok := payload["batch_read_count"].(float64); ok {
		cs.BatchReadCount = int64(v)
	}

	cs.updateHistogram(payload)

	if v, ok := payload["host_alloc_drops"].(float64); ok {
		cs.HostAllocDrops = int64(v)
	}
	if v, ok := payload["backup_queue_depth"].(float64); ok {
		cs.BackupQueueDepth = int64(v)
	}
	if v, ok := payload["backup_ops_completed"].(float64); ok {
		cs.BackupOpsCompleted = int64(v)
	}
	if v, ok := payload["backup_ops_failed"].(float64); ok {
		cs.BackupOpsFailed = int64(v)
	}
	if v, ok := payload["backup_in_flight"].(float64); ok {
		cs.BackupInFlight = int64(v)
	}
	if v, ok := payload["backup_avg_latency_ms"].(float64); ok {
		cs.BackupAvgLatencyMs = v
	}
	if v, ok := payload["backup_jitter_total_ms"].(float64); ok {
		cs.BackupJitterTotalMs = v
	}
	if v, ok := payload["backup_avg_gap_ms"].(float64); ok {
		cs.BackupAvgGapMs = v
	}
	if v, ok := payload["backup_jitter_cfg_ms"].(float64); ok {
		cs.BackupJitterCfgMs = v
	}
	if v, ok := payload["backup_batch_size"].(float64); ok {
		cs.BackupBatchSize = int64(v)
	}

	if v, ok := payload["prefetch_received"].(float64); ok {
		cs.PrefetchReceived = int64(v)
	}
	if v, ok := payload["prefetch_dispatched"].(float64); ok {
		cs.PrefetchDispatched = int64(v)
	}
	if v, ok := payload["prefetch_revoked"].(float64); ok {
		cs.PrefetchRevoked = int64(v)
	}
	if v, ok := payload["prefetch_io_completed"].(float64); ok {
		cs.PrefetchIoCompleted = int64(v)
	}
	if v, ok := payload["prefetch_io_failed"].(float64); ok {
		cs.PrefetchIoFailed = int64(v)
	}
	if v, ok := payload["prefetch_tokens"].(float64); ok {
		cs.PrefetchTokens = int64(v)
	}
	if v, ok := payload["prefetch_pages"].(float64); ok {
		cs.PrefetchPages = int64(v)
	}
	if v, ok := payload["prefetch_avg_io_ms"].(float64); ok {
		cs.PrefetchAvgIoMs = v
	}
	if v, ok := payload["prefetch_queue_depth"].(float64); ok {
		cs.PrefetchQueueDepth = int64(v)
	}
	if v, ok := payload["prefetch_in_flight"].(float64); ok {
		cs.PrefetchInFlight = int64(v)
	}

	if v, ok := payload["reconnect_count"].(float64); ok {
		cs.ReconnectCount = int64(v)
	}
	if v, ok := payload["pool_size"].(float64); ok {
		cs.PoolSize = int64(v)
	}
	if v, ok := payload["dedup_mode"].(string); ok {
		cs.DedupMode = v
	}
	if v, ok := payload["dedup_auto_disabled"].(bool); ok {
		cs.DedupAutoDisabled = v
	}
	if v, ok := payload["dedup_low_hit_streak"].(float64); ok {
		cs.DedupLowHitStreak = int64(v)
	}
	if v, ok := payload["dedup_cost_streak"].(float64); ok {
		cs.DedupCostStreak = int64(v)
	}
	if v, ok := payload["total_pages_set"].(float64); ok {
		cs.TotalPagesSet = int64(v)
	}
	if v, ok := payload["total_pages_get"].(float64); ok {
		cs.TotalPagesGet = int64(v)
	}
	if v, ok := payload["backup_coalesce_avg_ops"].(float64); ok {
		cs.BackupCoalesceAvgOps = v
	}
	if v, ok := payload["codec"].(string); ok {
		cs.Codec = v
	}
	if v, ok := payload["codec_lossy"].(bool); ok {
		cs.CodecLossy = v
	}
	if v, ok := payload["pool_rebuilds"].(float64); ok {
		cs.PoolRebuilds = int64(v)
	}
	if v, ok := payload["transport"].(string); ok {
		cs.Transport = v
	}
	if v, ok := payload["backup_bandwidth_gbps"].(float64); ok {
		cs.BackupBandwidthGbps = v
	}
	if v, ok := payload["io_max_latency_ms"].(float64); ok {
		cs.IoMaxLatencyMs = v
	}
	if v, ok := payload["warmup_phase"].(string); ok {
		cs.WarmupPhase = v
	}
	if v, ok := payload["warmup_effective_batch_size"].(float64); ok {
		cs.WarmupEffectiveBatchSize = int64(v)
	}

	if v, ok := payload["model_page_bytes"].(float64); ok {
		cs.ModelPageBytes = int64(v)
	}

	for i := 0; i < MaxNICs; i++ {
		if v, ok := payload[fmt.Sprintf("nic_%d_reads", i)].(float64); ok {
			cs.NICReads[i] = int64(v)
			if i+1 > cs.NICCount {
				cs.NICCount = i + 1
			}
		}
		if v, ok := payload[fmt.Sprintf("nic_%d_bytes_gb", i)].(float64); ok {
			cs.NICBytesGB[i] = v
			if i+1 > cs.NICCount {
				cs.NICCount = i + 1
			}
		}
		if v, ok := payload[fmt.Sprintf("nic_%d_writes", i)].(float64); ok {
			cs.NICWrites[i] = int64(v)
			if i+1 > cs.NICCount {
				cs.NICCount = i + 1
			}
		}
		if v, ok := payload[fmt.Sprintf("nic_%d_write_bytes_gb", i)].(float64); ok {
			cs.NICWriteBytesGB[i] = v
			if i+1 > cs.NICCount {
				cs.NICCount = i + 1
			}
		}
	}

	if v, ok := payload["stripe_calls"].(float64); ok {
		cs.StripeCalls = int64(v)
	}
	if v, ok := payload["stripe_errors"].(float64); ok {
		cs.StripeErrors = int64(v)
	}
	if v, ok := payload["stripe_write_calls"].(float64); ok {
		cs.StripeWriteCalls = int64(v)
	}
	if v, ok := payload["stripe_write_errors"].(float64); ok {
		cs.StripeWriteErrors = int64(v)
	}
	if v, ok := payload["stripe_avg_nics"].(float64); ok {
		cs.StripeAvgNICs = v
	}

	var newUnhandled []string
	for k := range payload {
		if !knownClientStatsFields[k] && !r.warnedKeys[k] {
			r.warnedKeys[k] = true
			newUnhandled = append(newUnhandled, k)
		}
	}
	if len(newUnhandled) > 0 {
		log.Printf("[client_stats] conn=%d: %d new unhandled fields (logged once): %v", connID, len(newUnhandled), newUnhandled)
	}

	if r.Advisor != nil {
		r.Advisor.Evaluate(cs)
	}
}

var knownClientStatsFields = map[string]bool{
	"get_errors": true, "get_successes": true, "set_errors": true,
	"exists_errors": true, "exists_timeouts": true, "io_workers": true,
	"avg_io_batch_size": true, "avg_io_latency_ms": true, "io_calls": true,
	"cache_hit_rate": true, "token_usage": true, "num_running_reqs": true,
	"evictable_ratio": true, "gen_throughput": true, "prefetch_bandwidth_gbps": true,
	"rdma_read_retries": true, "rdma_read_failures": true,
	"avg_roundtrip_us": true, "avg_rdma_read_us": true,
	"roundtrip_count": true, "rdma_read_count": true,
	"inflight_gets": true, "inflight_sets": true,
	"avg_preprocess_ms": true, "avg_exists_ms": true,
	"avg_transfer_ms": true, "avg_postprocess_ms": true,
	"get_avg_ctrl_ms": true, "get_avg_meta_ms": true,
	"get_avg_read_ms": true, "get_avg_ack_ms": true,
	"get_batch_count": true, "get_avg_bytes": true,
	"set_avg_serialize_ms": true, "set_avg_send_ms": true,
	"set_batch_count": true, "set_avg_bytes": true, "set_avg_sub_batches": true,
	"avg_batch_read_us": true, "batch_read_count": true,
	"io_latency_hist_buckets": true, "io_latency_hist_counts": true,
	"io_latency_hist_sum": true, "io_latency_hist_total": true,
	"host_alloc_drops": true, "backup_queue_depth": true,
	"backup_ops_completed": true, "backup_ops_failed": true,
	"backup_in_flight": true, "backup_avg_latency_ms": true,
	"backup_jitter_total_ms": true, "backup_avg_gap_ms": true,
	"backup_jitter_cfg_ms": true, "backup_batch_size": true,
	"prefetch_received": true, "prefetch_dispatched": true,
	"prefetch_revoked": true, "prefetch_io_completed": true,
	"prefetch_io_failed": true, "prefetch_tokens": true,
	"prefetch_pages": true, "prefetch_avg_io_ms": true,
	"prefetch_queue_depth": true, "prefetch_in_flight": true,
	"reconnect_count": true, "pool_size": true,
	"dedup_mode": true, "dedup_auto_disabled": true,
	"dedup_low_hit_streak": true, "dedup_cost_streak": true,
	"total_pages_set": true, "total_pages_get": true,
	"backup_coalesce_avg_ops": true,
	"codec":                   true, "codec_lossy": true,
	"pool_rebuilds": true, "transport": true,
	"backup_bandwidth_gbps": true, "io_max_latency_ms": true,
	"warmup_phase": true, "warmup_effective_batch_size": true,
	"model_page_bytes": true,
	"nic_0_reads":      true, "nic_0_bytes_gb": true, "nic_0_writes": true, "nic_0_write_bytes_gb": true,
	"nic_1_reads": true, "nic_1_bytes_gb": true, "nic_1_writes": true, "nic_1_write_bytes_gb": true,
	"nic_2_reads": true, "nic_2_bytes_gb": true, "nic_2_writes": true, "nic_2_write_bytes_gb": true,
	"nic_3_reads": true, "nic_3_bytes_gb": true, "nic_3_writes": true, "nic_3_write_bytes_gb": true,
	"nic_4_reads": true, "nic_4_bytes_gb": true, "nic_4_writes": true, "nic_4_write_bytes_gb": true,
	"nic_5_reads": true, "nic_5_bytes_gb": true, "nic_5_writes": true, "nic_5_write_bytes_gb": true,
	"nic_6_reads": true, "nic_6_bytes_gb": true, "nic_6_writes": true, "nic_6_write_bytes_gb": true,
	"nic_7_reads": true, "nic_7_bytes_gb": true, "nic_7_writes": true, "nic_7_write_bytes_gb": true,
	"stripe_calls": true, "stripe_errors": true,
	"stripe_write_calls": true, "stripe_write_errors": true,
	"stripe_avg_nics": true,
}

func (cs *ClientStats) updateHistogram(payload map[string]interface{}) {
	if v, ok := payload["io_latency_hist_buckets"].([]interface{}); ok {
		newBuckets := make([]float64, len(v))
		for i, x := range v {
			if f, ok := x.(float64); ok {
				newBuckets[i] = f
			}
		}
		if len(cs.IoLatencyHistBuckets) != len(newBuckets) {
			cs.IoLatencyHistCounts = nil
			cs.IoLatencyHistSum = 0
			cs.IoLatencyHistTotal = 0
		}
		cs.IoLatencyHistBuckets = newBuckets
	}
	if v, ok := payload["io_latency_hist_counts"].([]interface{}); ok {
		incoming := make([]int64, len(v))
		for i, x := range v {
			if f, ok := x.(float64); ok {
				incoming[i] = int64(f)
			}
		}
		if len(cs.IoLatencyHistCounts) == len(incoming) {
			for i := range incoming {
				cs.IoLatencyHistCounts[i] += incoming[i]
			}
		} else {
			cs.IoLatencyHistCounts = incoming
		}
	}
	if v, ok := payload["io_latency_hist_sum"].(float64); ok {
		cs.IoLatencyHistSum += v
	}
	if v, ok := payload["io_latency_hist_total"].(float64); ok {
		cs.IoLatencyHistTotal += int64(v)
	}
}

func (r *ClientStatsRegistry) Snapshot() []*ClientStats {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]*ClientStats, 0, len(r.stats))
	for _, cs := range r.stats {
		cp := *cs
		if cs.IoLatencyHistBuckets != nil {
			cp.IoLatencyHistBuckets = make([]float64, len(cs.IoLatencyHistBuckets))
			copy(cp.IoLatencyHistBuckets, cs.IoLatencyHistBuckets)
		}
		if cs.IoLatencyHistCounts != nil {
			cp.IoLatencyHistCounts = make([]int64, len(cs.IoLatencyHistCounts))
			copy(cp.IoLatencyHistCounts, cs.IoLatencyHistCounts)
		}
		out = append(out, &cp)
	}
	return out
}

func (r *ClientStatsRegistry) Remove(connID uint64) {
	r.mu.Lock()
	delete(r.stats, connID)
	r.stats, r.statsPeak = compactMap(r.stats, r.statsPeak)
	r.mu.Unlock()
	if r.Advisor != nil {
		r.Advisor.Remove(connID)
	}
}
