package config

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// Config is the top-level L3 configuration.
type Config struct {
	ListenAddrs       []string `json:"listen_addrs"`
	RDMAAddrs         []string `json:"rdma_addrs"`
	RDMAListenPort    int      `json:"rdma_listen_port"`
	RDMAReadThreshold int      `json:"rdma_read_threshold"`
	RDMARecvBufSize   int      `json:"rdma_recv_buf_size"`
	RDMASendBufSize   int      `json:"rdma_send_buf_size"`
	RDMARecvBufCount  int      `json:"rdma_recv_buf_count"`
	RDMACQDepth       int      `json:"rdma_cq_depth"`
	RDMAODP           string   `json:"rdma_odp"`            // "auto" (default), "enable", "disable"
	RDMALinkRateGbps  float64  `json:"rdma_link_rate_gbps"` // 0 = auto-detect from sysfs
	NUMANodes         string   `json:"numa_nodes"`          // "auto" (default), "none", or comma-separated node IDs (e.g. "0,1")
	NumShards         int      `json:"num_shards"`
	MaxMemoryGB       int      `json:"max_memory_gb"`
	UseHugePages       bool     `json:"use_huge_pages"`
	AutoAllocHugePages bool     `json:"auto_alloc_huge_pages"`
	HugepageSize       string   `json:"hugepage_size"` // "auto" (default), "2mb", "1gb"
	MaxKeys           uint64   `json:"max_keys"`
	EvictionPolicy    string   `json:"eviction_policy"`
	ModelPageBytes    uint64   `json:"model_page_bytes"`
	MaxLeaseDurationMs int64   `json:"max_lease_duration_ms"`
	DispatchTimeoutMs int64    `json:"dispatch_timeout_ms"`
	AllocatorMode     string   `json:"allocator_mode"` // "slab" (default) or "offset"
	SlabPreset        string   `json:"slab_preset"`    // named preset: "static", "auto", "benchmark"
	SlabDistribution  string   `json:"slab_distribution"`
	WarmupOps         int      `json:"warmup_ops"`      // SET ops before auto-detect fires; 0 = disabled
	AutoTuneSlabs     bool     `json:"auto_tune_slabs"` // rebuild slab allocator on FLUSH with detected sizes
	SlabClassWeights  map[string]float64 `json:"slab_class_weights"`
	MetricsAddr       string   `json:"metrics_addr"`

	// Slab vacuum
	VacuumEnabled              bool    `json:"vacuum_enabled"`
	VacuumIntervalSeconds      int     `json:"vacuum_interval_seconds"`
	VacuumCooldownSeconds      int     `json:"vacuum_cooldown_seconds"`
	VacuumUtilizationThreshold float64 `json:"vacuum_utilization_threshold"`
	VacuumMinAgeSeconds        int     `json:"vacuum_min_age_seconds"`

	// Pressure-driven rebalancing
	VacuumPressureRebalancing bool    `json:"vacuum_pressure_rebalancing"`
	VacuumDampingFactor       float64 `json:"vacuum_damping_factor"`
	VacuumDriftThreshold      float64 `json:"vacuum_drift_threshold"`
	VacuumMinClassWeight      float64 `json:"vacuum_min_class_weight"`
	VacuumEvictionRateNorm    float64 `json:"vacuum_eviction_rate_norm"`
	VacuumWatermarkThreshold  float64 `json:"vacuum_watermark_threshold"`

	// ZeroLatencyBalance
	MigrateBatchSize        int `json:"migrate_batch_size"`
	MaxConcurrentMigrations int `json:"max_concurrent_migrations"`
	MigrateDrainBudget      int `json:"migrate_drain_budget"`

	// Adaptive migration rate
	MigrationP99TargetUs  int64 `json:"migration_p99_target_us"`
	MigrationMinBatchSize int   `json:"migration_min_batch_size"`
	MigrationMaxBatchSize int   `json:"migration_max_batch_size"`

	// Observability
	VerboseShardLogging bool   `json:"verbose_shard_logging"`
	PprofAddr           string `json:"pprof_addr"`

	// Cluster
	ClusterEnabled  bool     `json:"cluster_enabled"`
	ClusterPeers    []string `json:"cluster_peers"`
	ClusterNodeID   string   `json:"cluster_node_id"`
	ReplicaCount    int      `json:"replica_count"`
	GossipPort      int      `json:"gossip_port"`
	GossipPingMs    int      `json:"gossip_ping_ms"`
	GossipSuspectMs int      `json:"gossip_suspect_ms"`

	// TCP connection limit
	MaxTCPConnections int `json:"max_tcp_connections"`

	// RDMA connection limit
	MaxRDMAConnections     int `json:"max_rdma_connections"`
	RDMADispatchWorkers    int `json:"rdma_dispatch_workers"`
	RDMADispatchQueueDepth int `json:"rdma_dispatch_queue_depth"`

	// Per-connection rate limiting
	MaxOpsPerConnSec int64 `json:"max_ops_per_conn_sec"`

	// OOM graceful handling
	OOMRejectAfterFails int `json:"oom_reject_after_fails"`

	// CXL transport
	CXLEnabled       bool   `json:"cxl_enabled"`
	CXLDevdaxPath    string `json:"cxl_devdax_path"`
	CXLPoolSizeGB    int    `json:"cxl_pool_size_gb"`
	CXLListenPort    int    `json:"cxl_listen_port"`
	CXLReadThreshold int    `json:"cxl_read_threshold"`

	// Snapshot/Restore
	SnapshotDir  string `json:"snapshot_dir"`
	AutoSnapshot bool   `json:"auto_snapshot"`
	AutoRestore  bool   `json:"auto_restore"`

	// Preflight system checks
	PreflightEnabled       bool    `json:"preflight_enabled"`
	PreflightAbortOnLowMem bool    `json:"preflight_abort_on_low_mem"`
	PreflightWarnSwapPct   float64 `json:"preflight_warn_swap_pct"`
	PreflightWarnMemPct    float64 `json:"preflight_warn_mem_pct"`
	PreflightKillStale     bool    `json:"preflight_kill_stale"`
	PreflightClearSwap     bool    `json:"preflight_clear_swap"`
	AutoDisableSwap        bool    `json:"auto_disable_swap"`
	RestoreSwapOnStop      bool    `json:"restore_swap_on_stop"`
	PIDFilePath            string  `json:"pid_file"`
	Mlockall               bool    `json:"mlockall"`
	ReleaseHugepagesOnStop bool    `json:"release_hugepages_on_stop"`
	HugepageStatePath      string  `json:"hugepage_state_path"`

	// OOM score adjustment
	OOMScoreAdj int `json:"oom_score_adj"`

	// Runtime memory pressure circuit breaker
	MemPressureThresholdPct float64 `json:"mem_pressure_threshold_pct"`

	// Budget headroom
	BudgetHeadroomGB float64 `json:"budget_headroom_gb"`

	// Crash auto-restart
	AutoPanicReboot       bool `json:"auto_panic_reboot"`
	PanicRebootTimeoutSec int  `json:"panic_reboot_timeout_sec"`
	WatchdogEnabled       bool `json:"watchdog_enabled"`
	WatchdogIntervalSec   int  `json:"watchdog_interval_sec"`

	SkipStartupProbes bool `json:"skip_startup_probes"`

	// Internal: tracks which keys were explicitly set in JSON
	jsonKeys map[string]bool `json:"-"`
}

const DefaultRDMARecvBufCount = 4

// DefaultConfig returns configuration with sensible defaults.
func DefaultConfig() Config {
	return Config{
		ListenAddrs:                []string{"0.0.0.0:18000"},
		RDMAAddrs:                  []string{"auto"},
		RDMAListenPort:             18001,
		RDMAReadThreshold:          4096,
		RDMARecvBufSize:            16 * 1024 * 1024,
		RDMASendBufSize:            16 * 1024 * 1024,
		RDMARecvBufCount:           DefaultRDMARecvBufCount,
		RDMACQDepth:                32768,
		RDMAODP:                    "auto",
		NumShards:                  0,
		MaxMemoryGB:                512,
		UseHugePages:               true,
		AutoAllocHugePages:         true,
		MaxKeys:                    10000000,
		EvictionPolicy:             "wtinylfu",
		ModelPageBytes:             0,
		MaxLeaseDurationMs:         30000,
		DispatchTimeoutMs:          30000,
		SlabDistribution:           "auto",
		WarmupOps:                  100,
		AutoTuneSlabs:              true,
		MetricsAddr:                ":9090",
		PprofAddr:                  ":6060",
		VacuumEnabled:              true,
		VacuumIntervalSeconds:      30,
		VacuumCooldownSeconds:      10,
		VacuumUtilizationThreshold: 0.50,
		VacuumMinAgeSeconds:        30,
		VacuumPressureRebalancing:  true,
		VacuumDampingFactor:        0.3,
		VacuumDriftThreshold:       0.15,
		VacuumMinClassWeight:       0.5,
		VacuumEvictionRateNorm:     10.0,
		VacuumWatermarkThreshold:   0.50,
		MigrateBatchSize:           512,
		MaxConcurrentMigrations:    2,
		MigrateDrainBudget:         64,
		MigrationMinBatchSize:      32,
		MigrationMaxBatchSize:      2048,
		ReplicaCount:               1,
		GossipPort:                 19000,
		GossipPingMs:               500,
		GossipSuspectMs:            3000,
		PreflightEnabled:           true,
		PreflightAbortOnLowMem:     true,
		PreflightWarnSwapPct:       0.80,
		PreflightWarnMemPct:        0.90,
		CXLListenPort:              18002,
		Mlockall:                   true,
		AutoDisableSwap:            false,
		ReleaseHugepagesOnStop:     true,
		AutoPanicReboot:            true,
		PanicRebootTimeoutSec:      10,
		WatchdogIntervalSec:        15,
		MaxOpsPerConnSec:           500000,
		OOMRejectAfterFails:        10,
		OOMScoreAdj:                0,
		MemPressureThresholdPct:    10.0,
		BudgetHeadroomGB:           0,
	}
}

// Load reads a JSON config file and merges with defaults.
func Load(path string) (Config, error) {
	cfg := DefaultConfig()
	if path == "" {
		return cfg, nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, fmt.Errorf("reading config %s: %w", path, err)
	}

	if err := json.Unmarshal(data, &cfg); err != nil {
		return cfg, fmt.Errorf("parsing config %s: %w", path, err)
	}

	// Track which keys were explicitly set in the JSON file.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err == nil {
		cfg.jsonKeys = make(map[string]bool, len(raw))
		for k := range raw {
			cfg.jsonKeys[k] = true
		}
	}

	return cfg, nil
}

// JSONKeys returns the set of keys explicitly set in the config file.
func (c *Config) JSONKeys() map[string]bool {
	if c.jsonKeys == nil {
		return map[string]bool{}
	}
	return c.jsonKeys
}

// TomlKeys is an alias for JSONKeys for compatibility with preset code.
func (c *Config) TomlKeys() map[string]bool {
	return c.JSONKeys()
}

// Validate checks the configuration for errors.
func (c *Config) Validate() error {
	if c.SlabPreset != "" {
		if err := ApplyPreset(c, c.JSONKeys()); err != nil {
			return err
		}
		if err := ValidatePreset(c); err != nil {
			return err
		}
	}

	if c.MaxMemoryGB <= 0 {
		return fmt.Errorf("max_memory_gb must be > 0")
	}

	if c.BudgetHeadroomGB <= 0 {
		if sysRAM := systemMemoryGB(); sysRAM > 0 {
			c.BudgetHeadroomGB = sysRAM * 0.10
			if c.BudgetHeadroomGB < 8.0 {
				c.BudgetHeadroomGB = 8.0
			}
			log.Printf("[l3server] budget_headroom_gb auto-scaled to %.1f GB (10%% of %.0f GB system RAM)",
				c.BudgetHeadroomGB, sysRAM)
		} else {
			c.BudgetHeadroomGB = 8.0
		}
	}

	if sysRAM := systemMemoryGB(); sysRAM > 0 && float64(c.MaxMemoryGB) > 0.90*sysRAM {
		return fmt.Errorf("max_memory_gb (%d) exceeds 90%% of system RAM (%.0f GB) — reduce max_memory_gb or add RAM", c.MaxMemoryGB, sysRAM)
	} else if sysRAM > 0 && float64(c.MaxMemoryGB) > 0.85*sysRAM {
		log.Printf("WARNING: max_memory_gb (%d) exceeds 85%% of system RAM (%.0f GB) — risk of OOM under load", c.MaxMemoryGB, sysRAM)
	}

	recvBufCount := c.RDMARecvBufCount
	if recvBufCount <= 0 {
		recvBufCount = DefaultRDMARecvBufCount
	}
	perConnMB := (c.RDMARecvBufSize*recvBufCount + c.RDMASendBufSize) / (1024 * 1024)
	if perConnMB > 256 {
		log.Printf("WARNING: RDMA per-connection buffer cost is %d MB "+
			"(%d recv × %d MB + %d MB send) — this is high and will slow "+
			"connection setup (ibv_reg_mr page pinning). Consider reducing "+
			"rdma_recv_buf_size or rdma_recv_buf_count.",
			perConnMB, recvBufCount,
			c.RDMARecvBufSize/(1024*1024), c.RDMASendBufSize/(1024*1024))
	}

	if c.MaxRDMAConnections <= 0 {
		if sysRAM := systemMemoryGB(); sysRAM > 0 {
			c.MaxRDMAConnections = int((sysRAM * 0.1 * 1024) / float64(perConnMB))
			if c.MaxRDMAConnections < 4 {
				c.MaxRDMAConnections = 4
			}
			log.Printf("[l3server] RDMA max_connections auto-computed: %d (%.1f GB RAM × 10%% / %d MB per-conn)", c.MaxRDMAConnections, sysRAM, perConnMB)
			if c.MaxRDMAConnections < 16 {
				log.Printf("[l3server] WARNING: auto-computed max_rdma_connections=%d is very low — supports at most %d concurrent GPU ranks (pool_size=4). Set max_rdma_connections in JSON to override.", c.MaxRDMAConnections, c.MaxRDMAConnections/4)
			}
		}
	}

	if c.MaxRDMAConnections > 0 && c.RDMACQDepth > 0 {
		const maxSendWR = 256
		cqSafeConns := c.RDMACQDepth / (maxSendWR + recvBufCount)
		if cqSafeConns < 4 {
			cqSafeConns = 4
		}
		if c.MaxRDMAConnections > cqSafeConns {
			log.Printf("[l3server] max_rdma_connections capped %d → %d (CQ depth %d / %d CQEs per conn — increase rdma_cq_depth to raise limit)",
				c.MaxRDMAConnections, cqSafeConns, c.RDMACQDepth, maxSendWR+recvBufCount)
			c.MaxRDMAConnections = cqSafeConns
		}
	}
	if c.RDMADispatchWorkers < 0 {
		return fmt.Errorf("rdma_dispatch_workers must be >= 0, got %d", c.RDMADispatchWorkers)
	}
	if c.RDMADispatchQueueDepth < 0 {
		return fmt.Errorf("rdma_dispatch_queue_depth must be >= 0, got %d", c.RDMADispatchQueueDepth)
	}
	if len(c.ListenAddrs) == 0 {
		return fmt.Errorf("at least one listen_addr required")
	}
	switch c.EvictionPolicy {
	case "wtinylfu", "sieve", "lru":
		// ok
	default:
		return fmt.Errorf("unknown eviction_policy: %s (valid: wtinylfu, sieve, lru)", c.EvictionPolicy)
	}
	switch c.AllocatorMode {
	case "", "slab", "offset":
		// ok
	default:
		return fmt.Errorf("unknown allocator_mode: %s (valid: slab, offset)", c.AllocatorMode)
	}

	if c.SlabDistribution == "" {
		switch {
		case c.ModelPageBytes > 0 && !c.AutoTuneSlabs:
			c.SlabDistribution = "model"
		case c.AutoTuneSlabs:
			c.SlabDistribution = "auto"
		default:
			c.SlabDistribution = "uniform"
		}
		log.Printf("[l3server] slab_distribution not set, inferred %q from legacy fields", c.SlabDistribution)
	}
	switch c.SlabDistribution {
	case "auto", "model", "uniform", "dedicated":
		// ok
	default:
		return fmt.Errorf("unknown slab_distribution: %s (valid: auto, model, uniform, dedicated)", c.SlabDistribution)
	}

	if c.SlabDistribution == "model" && c.ModelPageBytes == 0 {
		return fmt.Errorf("slab_distribution=%q requires model_page_bytes > 0", c.SlabDistribution)
	}
	if c.SlabDistribution == "dedicated" && c.ModelPageBytes == 0 {
		return fmt.Errorf("slab_distribution=%q requires model_page_bytes > 0 (or use \"auto\" to auto-promote on page-size hint)", c.SlabDistribution)
	}
	if c.SlabDistribution == "auto" {
		if c.WarmupOps <= 0 {
			c.WarmupOps = 100
		}
		c.AutoTuneSlabs = true
	}
	if c.SlabDistribution == "dedicated" {
		c.AutoTuneSlabs = true
		if c.WarmupOps <= 0 {
			c.WarmupOps = 100
		}
	}
	if c.SlabDistribution == "uniform" && c.ModelPageBytes > 0 {
		log.Printf("[l3server] slab_distribution=%q: zeroing model_page_bytes=%d (uniform ignores model weighting)",
			c.SlabDistribution, c.ModelPageBytes)
		c.ModelPageBytes = 0
	}

	switch c.HugepageSize {
	case "", "auto", "2mb", "1gb":
		// ok
	default:
		return fmt.Errorf("unknown hugepage_size: %s (valid: auto, 2mb, 1gb)", c.HugepageSize)
	}

	switch c.RDMAODP {
	case "", "auto", "enable", "disable":
		// ok
	default:
		return fmt.Errorf("unknown rdma_odp: %s (valid: auto, enable, disable)", c.RDMAODP)
	}

	for _, addr := range c.RDMAAddrs {
		if addr == "" {
			return fmt.Errorf("rdma_addrs must not contain empty strings")
		}
	}
	const minBufSize = 1048576
	if c.RDMARecvBufSize > 0 && c.RDMARecvBufSize < minBufSize {
		return fmt.Errorf("rdma_recv_buf_size must be >= %d (1MB), got %d", minBufSize, c.RDMARecvBufSize)
	}
	if c.RDMASendBufSize > 0 && c.RDMASendBufSize < minBufSize {
		return fmt.Errorf("rdma_send_buf_size must be >= %d (1MB), got %d", minBufSize, c.RDMASendBufSize)
	}
	if c.RDMARecvBufCount < 1 {
		return fmt.Errorf("rdma_recv_buf_count must be >= 1, got %d", c.RDMARecvBufCount)
	}
	if c.RDMARecvBufCount > 64 {
		log.Printf("WARNING: rdma_recv_buf_count=%d is unusually high (>64) — this will increase per-connection memory significantly", c.RDMARecvBufCount)
	}
	if c.RDMACQDepth > 0 && c.RDMACQDepth < 512 {
		return fmt.Errorf("rdma_cq_depth must be >= 512, got %d", c.RDMACQDepth)
	}

	if c.VacuumEnabled {
		if c.VacuumIntervalSeconds < 10 {
			return fmt.Errorf("vacuum_interval_seconds must be >= 10, got %d", c.VacuumIntervalSeconds)
		}
		if c.VacuumCooldownSeconds < 5 {
			return fmt.Errorf("vacuum_cooldown_seconds must be >= 5, got %d", c.VacuumCooldownSeconds)
		}
		if c.VacuumUtilizationThreshold <= 0 || c.VacuumUtilizationThreshold > 1 {
			return fmt.Errorf("vacuum_utilization_threshold must be in (0, 1], got %.2f", c.VacuumUtilizationThreshold)
		}
		if !c.AutoTuneSlabs || c.WarmupOps <= 0 {
			log.Printf("WARN: vacuum_enabled=true requires auto_tune_slabs=true and warmup_ops>0 — vacuum will have no effect")
		}
		if c.VacuumPressureRebalancing {
			if c.VacuumEvictionRateNorm <= 0 {
				return fmt.Errorf("vacuum_eviction_rate_norm must be > 0, got %.2f", c.VacuumEvictionRateNorm)
			}
		}
		if c.VacuumWatermarkThreshold <= 0 || c.VacuumWatermarkThreshold > 1 {
			return fmt.Errorf("vacuum_watermark_threshold must be in (0, 1], got %.2f", c.VacuumWatermarkThreshold)
		}
	}

	if c.PanicRebootTimeoutSec < 0 {
		return fmt.Errorf("panic_reboot_timeout_sec must be >= 0, got %d", c.PanicRebootTimeoutSec)
	}
	if c.WatchdogIntervalSec < 1 {
		return fmt.Errorf("watchdog_interval_sec must be >= 1, got %d", c.WatchdogIntervalSec)
	}

	if c.PprofAddr != "" {
		if _, _, err := net.SplitHostPort(c.PprofAddr); err != nil {
			return fmt.Errorf("pprof_addr: invalid address %q: %w", c.PprofAddr, err)
		}
	}

	if c.CXLEnabled {
		if c.CXLDevdaxPath == "" {
			return fmt.Errorf("cxl_enabled=true requires cxl_devdax_path (e.g. /dev/dax0.0)")
		}
		if c.CXLPoolSizeGB <= 0 {
			return fmt.Errorf("cxl_enabled=true requires cxl_pool_size_gb > 0")
		}
		if c.CXLListenPort <= 0 {
			return fmt.Errorf("cxl_listen_port must be > 0, got %d", c.CXLListenPort)
		}
		for _, addr := range c.ListenAddrs {
			if _, p, err := net.SplitHostPort(addr); err == nil {
				if pn, _ := strconv.Atoi(p); pn == c.CXLListenPort {
					return fmt.Errorf("cxl_listen_port %d conflicts with TCP listen port", c.CXLListenPort)
				}
			}
		}
		if c.CXLListenPort == c.RDMAListenPort {
			return fmt.Errorf("cxl_listen_port %d conflicts with RDMA listen port", c.CXLListenPort)
		}
	}

	if c.OOMScoreAdj < -1000 || c.OOMScoreAdj > 1000 {
		return fmt.Errorf("oom_score_adj must be in [-1000, 1000], got %d", c.OOMScoreAdj)
	}
	if c.OOMScoreAdj == -1000 {
		log.Printf("WARNING: oom_score_adj=-1000 makes this process fully OOM-immune — if memory is exhausted, " +
			"the kernel will kill all OTHER processes (systemd, monitoring, networking). " +
			"Consider a less aggressive value (e.g. -500 or 0) on shared systems.")
	}

	if c.MemPressureThresholdPct < 0 || c.MemPressureThresholdPct > 50 {
		return fmt.Errorf("mem_pressure_threshold_pct must be in [0, 50], got %.1f", c.MemPressureThresholdPct)
	}

	if c.MigrationP99TargetUs > 0 {
		log.Printf("WARNING: migration_p99_target_us is deprecated and ignored (adaptive migration rate removed)")
	}
	if c.jsonKeys != nil && (c.jsonKeys["migration_min_batch_size"] || c.jsonKeys["migration_max_batch_size"]) {
		log.Printf("WARNING: migration_min_batch_size / migration_max_batch_size are deprecated and ignored")
	}

	if c.jsonKeys != nil {
		if c.jsonKeys["vacuum_damping_factor"] {
			log.Printf("WARNING: vacuum_damping_factor is deprecated and ignored (pressure rebalancing simplified)")
		}
		if c.jsonKeys["vacuum_drift_threshold"] {
			log.Printf("WARNING: vacuum_drift_threshold is deprecated and ignored (pressure rebalancing simplified)")
		}
		if c.jsonKeys["vacuum_min_class_weight"] {
			log.Printf("WARNING: vacuum_min_class_weight is deprecated and ignored (pressure rebalancing simplified)")
		}
	}

	if c.BudgetHeadroomGB < 1.0 {
		return fmt.Errorf("budget_headroom_gb must be >= 1.0 (or 0 for auto-scale), got %.1f", c.BudgetHeadroomGB)
	}

	if c.PreflightWarnSwapPct <= 0 || c.PreflightWarnSwapPct > 1 {
		return fmt.Errorf("preflight_warn_swap_pct must be in (0, 1], got %.2f", c.PreflightWarnSwapPct)
	}
	if c.PreflightWarnMemPct <= 0 || c.PreflightWarnMemPct > 1 {
		return fmt.Errorf("preflight_warn_mem_pct must be in (0, 1], got %.2f", c.PreflightWarnMemPct)
	}

	if c.MaxTCPConnections <= 0 {
		c.MaxTCPConnections = 1000
		if nofile := nofileLimit(); nofile > 0 {
			computed := int(float64(nofile) * 0.8)
			if computed > c.MaxTCPConnections {
				c.MaxTCPConnections = computed
			}
		}
		if c.MaxTCPConnections < 256 {
			c.MaxTCPConnections = 256
		}
		log.Printf("[l3server] max_tcp_connections auto-computed: %d", c.MaxTCPConnections)
	} else if c.MaxTCPConnections < 256 {
		return fmt.Errorf("max_tcp_connections must be >= 256, got %d", c.MaxTCPConnections)
	}

	if sysRAM := systemMemoryGB(); sysRAM > 0 {
		numShards := c.NumShards
		if numShards <= 0 {
			numShards = 16
		}
		idxCapPow2 := nextPow2Cfg(c.MaxKeys)
		indexGB := float64(idxCapPow2*index.BytesPerEntry*uint64(numShards)) / float64(1<<30)

		rdmaConns := c.MaxRDMAConnections
		if rdmaConns <= 0 {
			rdmaConns = 32
			log.Printf("[l3server] WARNING: max_rdma_connections=0 (unlimited) — using %d for budget estimate. "+
				"Set max_rdma_connections in JSON to control RDMA memory usage.", rdmaConns)
		}
		rdmaGB := float64(rdmaConns) * float64(perConnMB) / 1024.0

		headroomGB := c.BudgetHeadroomGB
		totalBudgetGB := float64(c.MaxMemoryGB) + rdmaGB + indexGB + headroomGB

		if totalBudgetGB > sysRAM*0.90 {
			return fmt.Errorf("aggregate memory budget %.1f GB (slab %d + RDMA %.1f [%d conns × %d MB] + index %.1f + headroom %.0f) "+
				"exceeds 90%% of system RAM (%.0f GB) — reduce max_memory_gb, max_rdma_connections, max_keys, or budget_headroom_gb",
				totalBudgetGB, c.MaxMemoryGB, rdmaGB, rdmaConns, perConnMB, indexGB, headroomGB, sysRAM)
		}
	}

	if c.ClusterEnabled {
		if c.ClusterNodeID == "" {
			hostname, _ := os.Hostname()
			if hostname == "" {
				hostname = "localhost"
			}
			c.ClusterNodeID = fmt.Sprintf("%s:%d", hostname, c.GossipPort)
			log.Printf("[l3server] cluster_node_id auto-generated: %s", c.ClusterNodeID)
		}
		for _, peer := range c.ClusterPeers {
			if _, _, err := net.SplitHostPort(peer); err != nil {
				return fmt.Errorf("cluster_peers: invalid address %q (expected host:port): %w", peer, err)
			}
		}
		if c.ReplicaCount > 1 && len(c.ClusterPeers) == 0 {
			log.Printf("WARN: replica_count=%d but no cluster_peers configured — replication will have no effect", c.ReplicaCount)
		}
	}
	if c.ReplicaCount < 1 {
		c.ReplicaCount = 1
	}

	if c.ModelPageBytes > 0 && c.MaxKeys > 0 {
		numShards := uint64(c.NumShards)
		if numShards == 0 {
			numShards = 16
		}
		perEntryBytes := c.ModelPageBytes + 128
		maxStorable := (uint64(c.MaxMemoryGB) * (1 << 30)) / perEntryBytes
		maxPerShard := maxStorable / numShards

		if c.MaxKeys < maxPerShard {
			log.Printf("WARN: max_keys (%d/shard) < estimated capacity (%d/shard for %d GB / %d shards / %d value_bytes) — "+
				"memory will be underutilized, consider max_keys = %d",
				c.MaxKeys, maxPerShard, c.MaxMemoryGB, numShards, c.ModelPageBytes, maxPerShard*2)
		}

		if c.MaxKeys > maxPerShard*4 {
			idxCapPow2 := nextPow2Cfg(c.MaxKeys)
			idxBytesPerShard := idxCapPow2 * index.BytesPerEntry
			idxBytesTotal := idxBytesPerShard * numShards
			rightSizedIdx := nextPow2Cfg(maxPerShard * 2)
			wastedGB := float64(idxBytesTotal-(rightSizedIdx*index.BytesPerEntry*numShards)) / float64(1<<30)
			log.Printf("WARN: max_keys (%d/shard) is %.0fx the estimated capacity (%d/shard for %d GB / %d shards / %d value_bytes) — "+
				"index overhead %.1f GB, consider max_keys = %d to save %.1f GB",
				c.MaxKeys, float64(c.MaxKeys)/float64(maxPerShard), maxPerShard,
				c.MaxMemoryGB, numShards, c.ModelPageBytes,
				float64(idxBytesTotal)/float64(1<<30),
				maxPerShard*2,
				wastedGB)
		}
	}

	if c.MaxKeys > 0 {
		ns := uint64(c.NumShards)
		if ns == 0 {
			ns = 8
		}
		totalKeys := c.MaxKeys * ns
		idxCapPow2 := nextPow2Cfg(c.MaxKeys)
		indexBytesTotal := idxCapPow2 * index.BytesPerEntry * ns
		indexGB := float64(indexBytesTotal) / float64(1<<30)
		if totalKeys > 50_000_000 && indexGB > 4.0 {
			log.Printf("[l3server] WARNING: max_keys=%d × %d shards = %dM total key capacity, "+
				"index overhead %.1f GB — consider reducing max_keys if your workload "+
				"uses fewer keys to reclaim memory for slab data",
				c.MaxKeys, ns, totalKeys/1_000_000, indexGB)
		}
	}

	return nil
}

func nextPow2Cfg(v uint64) uint64 {
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
