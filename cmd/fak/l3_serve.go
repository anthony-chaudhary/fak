package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/http/pprof"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/cluster"
	"github.com/anthony-chaudhary/fak/internal/l3server/config"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/dispatch"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/tcp"
	"github.com/anthony-chaudhary/fak/internal/l3server/version"
)

func cmdL3Serve(argv []string) {
	fs := flag.NewFlagSet("l3-serve", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), `Usage: fak l3-serve [flags]

Run the pure-Go L3 KV-cache tier daemon.

Flags:
`)
		fs.PrintDefaults()
	}

	configPath := fs.String("config", "", "Path to JSON config file")
	listenAddr := fs.String("listen", "0.0.0.0:18000", "TCP listen address")
	metricsAddr := fs.String("metrics", ":9090", "Prometheus metrics HTTP address")
	pprofAddr := fs.String("pprof", ":6060", "pprof HTTP address (empty to disable)")
	numShards := fs.Int("shards", 0, "Number of shards (default: 0 = auto)")
	maxMemGB := fs.Int("max-memory-gb", 512, "Maximum memory in GB")
	eviction := fs.String("eviction", "wtinylfu", "Eviction policy: wtinylfu, sieve, lru")
	allocMode := fs.String("allocator-mode", "slab", "Allocator mode: slab, offset")
	slabPreset := fs.String("slab-preset", "", "Slab preset: static, auto, benchmark, sglang")
	slabDist := fs.String("slab-distribution", "", "Slab distribution: auto, model, uniform, dedicated")
	modelPageBytes := fs.Uint64("model-page-bytes", 0, "Dominant value size for slab weighting (bytes)")
	maxKeys := fs.Uint64("max-keys", 10000000, "Maximum keys per shard")
	warmupOps := fs.Int("warmup-ops", 100, "SET ops before auto-detect fires (0=disabled)")
	autoTune := fs.Bool("auto-tune-slabs", true, "Rebuild slab allocator with detected sizes")
	vacuumEnabled := fs.Bool("vacuum", true, "Enable slab vacuum")
	noVacuum := fs.Bool("no-vacuum", false, "Disable slab vacuum")
	vacuumInterval := fs.Int("vacuum-interval", 30, "Vacuum check interval (seconds)")
	vacuumCooldown := fs.Int("vacuum-cooldown", 10, "Min seconds between rebalances")
	vacuumThreshold := fs.Float64("vacuum-threshold", 0.50, "Vacuum utilization threshold (0-1)")
	maxTCPConns := fs.Int("max-tcp-connections", 0, "Max concurrent TCP connections (0=auto)")
	maxOpsPerConnSec := fs.Int64("max-ops-per-conn-sec", 500000, "Max operations per connection per second (0=unlimited)")
	dispatchTimeoutMs := fs.Int64("dispatch-timeout-ms", 30000, "Operation dispatch timeout (ms)")
	snapshotDir := fs.String("snapshot-dir", "", "Snapshot directory path")
	autoSnapshot := fs.Bool("auto-snapshot", false, "Save snapshot on shutdown")
	autoRestore := fs.Bool("auto-restore", false, "Restore snapshot on startup")
	restoreFrom := fs.String("restore-from", "", "Snapshot directory to restore from immediately")
	verboseShardLog := fs.Bool("verbose-shard-logging", false, "Emit per-shard log lines")
	dryRun := fs.Bool("dry-run", false, "Validate configuration and print plan without starting")
	printVersion := fs.Bool("version", false, "Print server version and exit")

	if err := fs.Parse(argv); err != nil {
		fmt.Fprintf(os.Stderr, "fak l3-serve: flag parse error: %v\n", err)
		os.Exit(2)
	}

	if *printVersion {
		fmt.Printf("fak l3-serve %s (commit: %s, built: %s)\n",
			version.ServerVersion, version.Commit, version.BuildDate)
		return
	}

	if *noVacuum {
		*vacuumEnabled = false
	}

	flagsSet := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		flagsSet[f.Name] = true
	})
	if flagsSet["no-vacuum"] {
		flagsSet["vacuum"] = true
	}

	cfg := config.DefaultConfig()
	if *configPath != "" {
		loaded, err := config.Load(*configPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak l3-serve: load config %s: %v\n", *configPath, err)
			os.Exit(1)
		}
		cfg = loaded
	}

	if flagsSet["listen"] {
		cfg.ListenAddrs = []string{*listenAddr}
	}
	if flagsSet["metrics"] {
		cfg.MetricsAddr = *metricsAddr
	}
	if flagsSet["pprof"] {
		cfg.PprofAddr = *pprofAddr
	}
	if flagsSet["shards"] {
		cfg.NumShards = *numShards
	}
	if flagsSet["max-memory-gb"] {
		cfg.MaxMemoryGB = *maxMemGB
	}
	if flagsSet["eviction"] {
		cfg.EvictionPolicy = *eviction
	}
	if flagsSet["allocator-mode"] {
		cfg.AllocatorMode = *allocMode
	}
	if flagsSet["slab-preset"] {
		cfg.SlabPreset = *slabPreset
	}
	if flagsSet["slab-distribution"] {
		cfg.SlabDistribution = *slabDist
	}
	if flagsSet["model-page-bytes"] {
		cfg.ModelPageBytes = *modelPageBytes
	}
	if flagsSet["max-keys"] {
		cfg.MaxKeys = *maxKeys
	}
	if flagsSet["warmup-ops"] {
		cfg.WarmupOps = *warmupOps
	}
	if flagsSet["auto-tune-slabs"] {
		cfg.AutoTuneSlabs = *autoTune
	}
	if flagsSet["vacuum"] || flagsSet["no-vacuum"] {
		cfg.VacuumEnabled = *vacuumEnabled
	}
	if flagsSet["vacuum-interval"] {
		cfg.VacuumIntervalSeconds = *vacuumInterval
	}
	if flagsSet["vacuum-cooldown"] {
		cfg.VacuumCooldownSeconds = *vacuumCooldown
	}
	if flagsSet["vacuum-threshold"] {
		cfg.VacuumUtilizationThreshold = *vacuumThreshold
	}
	if flagsSet["max-tcp-connections"] {
		cfg.MaxTCPConnections = *maxTCPConns
	}
	if flagsSet["max-ops-per-conn-sec"] {
		cfg.MaxOpsPerConnSec = *maxOpsPerConnSec
	}
	if flagsSet["dispatch-timeout-ms"] {
		cfg.DispatchTimeoutMs = *dispatchTimeoutMs
	}
	if flagsSet["snapshot-dir"] {
		cfg.SnapshotDir = *snapshotDir
	}
	if flagsSet["auto-snapshot"] {
		cfg.AutoSnapshot = *autoSnapshot
	}
	if flagsSet["auto-restore"] {
		cfg.AutoRestore = *autoRestore
	}
	if flagsSet["verbose-shard-logging"] {
		cfg.VerboseShardLogging = *verboseShardLog
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "fak l3-serve: configuration error: %v\n", err)
		os.Exit(1)
	}

	if *dryRun {
		fmt.Println("fak l3-serve: configuration valid")
		fmt.Printf("  Listen: %v\n", cfg.ListenAddrs)
		fmt.Printf("  Metrics: %s\n", cfg.MetricsAddr)
		fmt.Printf("  Shards: %d\n", cfg.NumShards)
		fmt.Printf("  MaxMemoryGB: %d\n", cfg.MaxMemoryGB)
		fmt.Printf("  Eviction: %s\n", cfg.EvictionPolicy)
		fmt.Printf("  Allocator: %s\n", cfg.AllocatorMode)
		return
	}

	startedAt := time.Now()
	startup := &metrics.StartupState{}
	startup.Ready.Store(true)
	connReg := metrics.NewConnRegistry()
	clientStatsReg := metrics.NewClientStatsRegistry()
	collector := metrics.NewCollector(startup, connReg, clientStatsReg, startedAt)
	collector.MaxKeysPerShard = cfg.MaxKeys

	var metricsSrv *http.Server
	if cfg.MetricsAddr != "" {
		mux := http.NewServeMux()
		mux.Handle("/metrics", collector)
		mux.HandleFunc("/ready", collector.ReadyHandler)
		mux.HandleFunc("/debug/metrics.json", collector.MetricsJSONHandler)
		metricsSrv = &http.Server{Addr: cfg.MetricsAddr, Handler: mux}
		go func() {
			if err := metricsSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Printf("[l3-serve] metrics server error: %v", err)
			}
		}()
		log.Printf("[l3-serve] metrics listening on %s", cfg.MetricsAddr)
	}

	var pprofSrv *http.Server
	if cfg.PprofAddr != "" {
		mux := http.NewServeMux()
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		pprofSrv = &http.Server{Addr: cfg.PprofAddr, Handler: mux}
		go func() {
			if err := pprofSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Printf("[l3-serve] pprof server error: %v", err)
			}
		}()
		log.Printf("[l3-serve] pprof listening on %s", cfg.PprofAddr)
	}

	var initialWeights map[uint64]float64
	if len(cfg.SlabClassWeights) > 0 {
		initialWeights = make(map[uint64]float64, len(cfg.SlabClassWeights))
		for k, v := range cfg.SlabClassWeights {
			if sz, err := strconv.ParseUint(k, 10, 64); err == nil {
				initialWeights[sz] = v
			}
		}
	}

	mgrCfg := shard.ManagerConfig{
		NumShards:           cfg.NumShards,
		MaxMemoryGB:         cfg.MaxMemoryGB,
		EvictionPolicy:      cfg.EvictionPolicy,
		AllocatorMode:       cfg.AllocatorMode,
		ModelPageBytes:      cfg.ModelPageBytes,
		MaxLeaseDurMs:       cfg.MaxLeaseDurationMs,
		WarmupOps:           cfg.WarmupOps,
		AutoTuneSlabs:       cfg.AutoTuneSlabs,
		SlabDistribution:    cfg.SlabDistribution,
		InitialClassWeights: initialWeights,
		VerboseShardLogging: cfg.VerboseShardLogging,
		Vacuum: shard.VacuumConfig{
			Enabled:              cfg.VacuumEnabled,
			IntervalSeconds:      cfg.VacuumIntervalSeconds,
			CooldownSeconds:      cfg.VacuumCooldownSeconds,
			UtilizationThreshold: cfg.VacuumUtilizationThreshold,
			MinAgeSeconds:        cfg.VacuumMinAgeSeconds,
			PressureRebalancing:  cfg.VacuumPressureRebalancing,
			EvictionRateNorm:     cfg.VacuumEvictionRateNorm,
			WatermarkThreshold:   cfg.VacuumWatermarkThreshold,
		},
		MigrateBatchSize:        cfg.MigrateBatchSize,
		MaxConcurrentMigrations: cfg.MaxConcurrentMigrations,
		MigrateDrainBudget:      cfg.MigrateDrainBudget,
	}

	mgr, err := shard.NewManager(mgrCfg)
	if err != nil {
		log.Fatalf("[l3-serve] failed to create shard manager: %v", err)
	}
	mgr.Start()
	numShardsVal := mgr.NumShards()
	collector.NumShards = numShardsVal

	shardMetrics := make([]*metrics.ShardMetrics, numShardsVal)
	for i := 0; i < numShardsVal; i++ {
		shardMetrics[i] = mgr.Shard(i).Metrics()
	}
	collector.SetShards(shardMetrics)
	collector.InflightOpsFunc = dispatch.GlobalInflightOps
	collector.ShardDetectionFunc = func() []string {
		n := mgr.NumShards()
		statuses := make([]string, n)
		for i := 0; i < n; i++ {
			statuses[i] = mgr.Shard(i).SizeDetectionSnapshot().Status
		}
		return statuses
	}
	collector.SlabMetrics = func() ([]metrics.SlabClassSnapshot, metrics.SlabDetectionSnapshot) {
		if mgr.NumShards() == 0 {
			return nil, metrics.SlabDetectionSnapshot{}
		}
		utils := mgr.Shard(0).Allocator().ClassUtilizations()
		classes := make([]metrics.SlabClassSnapshot, len(utils))
		for i, u := range utils {
			classes[i] = metrics.SlabClassSnapshot{
				Size:            u.Size,
				TotalSlots:      u.TotalSlots,
				UsedSlots:       u.UsedSlots,
				AllocCount:      u.AllocCount,
				SlotUtilization: u.SlotUtilization,
			}
		}
		for si := 1; si < mgr.NumShards(); si++ {
			su := mgr.Shard(si).Allocator().ClassUtilizations()
			for i := range su {
				if i < len(classes) {
					classes[i].TotalSlots += su[i].TotalSlots
					classes[i].UsedSlots += su[i].UsedSlots
					classes[i].AllocCount += su[i].AllocCount
				}
			}
		}
		for i := range classes {
			if classes[i].TotalSlots > 0 {
				classes[i].SlotUtilization = float64(classes[i].UsedSlots) / float64(classes[i].TotalSlots)
			} else {
				classes[i].SlotUtilization = 0
			}
		}
		det := mgr.Shard(0).SizeDetectionSnapshot()
		detection := metrics.SlabDetectionSnapshot{
			Detected:          det.Status == "detected",
			DetectedPageBytes: det.DominantValueSize,
			SlotUtilization:   det.CurrentSlotUtilization / 100,
		}
		if cfg.ModelPageBytes > 0 {
			querySize := cfg.ModelPageBytes
			if det.Status == "detected" && det.DominantValueSize > 0 {
				querySize = det.DominantValueSize
			}
			slots, classSize := mgr.Shard(0).Allocator().ModelClassCapacity(querySize)
			detection.ConfiguredPageBytes = cfg.ModelPageBytes
			detection.ModelTotalSlots = slots * uint64(mgr.NumShards())
			detection.ModelClassSize = classSize
		}
		return classes, detection
	}
	collector.CacheStateProvider = func() (int64, int64) {
		var allocBytes int64
		var entries int64
		for i := 0; i < mgr.NumShards(); i++ {
			allocBytes += mgr.Shard(i).Allocator().AllocatedBytes()
			entries += int64(mgr.Shard(i).IndexCount())
		}
		return allocBytes, entries
	}
	collector.OpLatencyMetrics = func() []metrics.OpLatencySnapshot {
		n := mgr.NumShards()
		snaps := make([]metrics.OpLatencySnapshot, n)
		for i := 0; i < n; i++ {
			allP50, allP99, getP50, getP99, setP50, setP99, existsP50, existsP99, qwP50, qwP99, adP50, adP99, qDepth, qCap := mgr.Shard(i).OpLatencySnapshot()
			snaps[i] = metrics.OpLatencySnapshot{
				ShardID:        i,
				AllP50Us:       allP50,
				AllP99Us:       allP99,
				GetP50Us:       getP50,
				GetP99Us:       getP99,
				SetP50Us:       setP50,
				SetP99Us:       setP99,
				ExistsP50Us:    existsP50,
				ExistsP99Us:    existsP99,
				QueueWaitP50Us: qwP50,
				QueueWaitP99Us: qwP99,
				AllocDurP50Us:  adP50,
				AllocDurP99Us:  adP99,
				QueueDepth:     qDepth,
				QueueCap:       qCap,
			}
		}
		return snaps
	}
	if cfg.VacuumEnabled {
		collector.VacuumMetrics = func() metrics.VacuumSnapshot {
			vs := mgr.VacuumStats()
			return metrics.VacuumSnapshot{
				RebalancesTotal:    vs.RebalancesTotal,
				LastRebalanceEpoch: vs.LastRebalanceEpoch,
				PendingShards:      vs.PendingShards,
				PressureEvals:      vs.PressureEvals,
				PressureRebuilds:   vs.PressureRebuilds,
				MaxDrift:           vs.MaxDrift,
				RebalanceFailures:  vs.RebalanceFailures,
			}
		}
	}

	resDir := cfg.SnapshotDir
	if *restoreFrom != "" {
		resDir = *restoreFrom
	}
	if resDir != "" && (cfg.AutoRestore || *restoreFrom != "") {
		manifest, entries, rerr := snapshot.Load(resDir, mgr.NumShards())
		if rerr == nil {
			shardEntries := make([][]snapshot.KVEntry, mgr.NumShards())
			for i := range shardEntries {
				shardEntries[i] = []snapshot.KVEntry{}
			}
			for _, e := range entries {
				sh, _ := mgr.RouteKey(e.Key)
				shardEntries[sh.ID()] = append(shardEntries[sh.ID()], e)
			}
			var totalLoaded int
			for i, se := range shardEntries {
				if len(se) == 0 {
					continue
				}
				res := mgr.Shard(i).Submit(shard.ShardOp{
					Type:           shard.OpRestore,
					RestoreEntries: se,
					Result:         make(chan shard.OpResult, 1),
				})
				totalLoaded += res.Loaded
			}
			log.Printf("[l3-serve] restored %d keys from %s (v%s, %d shards)",
				totalLoaded, resDir, manifest.ServerVersion, manifest.ShardCount)
		} else if !os.IsNotExist(rerr) {
			log.Printf("[l3-serve] snapshot restore error: %v", rerr)
		}
	}

	var ring *cluster.Ring
	var replicator *cluster.Replicator
	var gossip *cluster.Gossip
	if cfg.ClusterEnabled {
		ring = cluster.NewRing()
		replicator = cluster.NewReplicator(ring, cfg.ClusterNodeID, cfg.ReplicaCount)
		replicator.Start()

		gossipCfg := cluster.GossipConfig{
			BindAddr:       fmt.Sprintf("0.0.0.0:%d", cfg.GossipPort),
			PingInterval:   time.Duration(cfg.GossipPingMs) * time.Millisecond,
			SuspectTimeout: time.Duration(cfg.GossipSuspectMs) * time.Millisecond,
			Seeds:          cfg.ClusterPeers,
			NodeID:         cfg.ClusterNodeID,
			NodeAddr:       cfg.ListenAddrs[0],
		}
		gossip = cluster.NewGossip(gossipCfg, ring)
		if err := gossip.Start(); err != nil {
			log.Printf("[cluster] gossip start failed: %v", err)
		} else {
			log.Printf("[cluster] started: node=%s, gossip port=%d, peers=%v",
				cfg.ClusterNodeID, cfg.GossipPort, cfg.ClusterPeers)
		}
	}

	dispatchTimeout := time.Duration(cfg.DispatchTimeoutMs) * time.Millisecond
	var tcpServers []*tcp.Server
	for _, addr := range cfg.ListenAddrs {
		srv := tcp.NewServer(addr, mgr, connReg, startedAt, numShardsVal)
		srv.SetClientStatsReg(clientStatsReg)
		srv.SetMetricsAddr(cfg.MetricsAddr)
		srv.SetDispatchTimeout(dispatchTimeout)
		srv.SetOpLatencyMetrics(collector.OpLatencyMetrics)
		if cfg.MaxTCPConnections > 0 {
			srv.SetMaxConns(cfg.MaxTCPConnections)
		}
		if cfg.MaxOpsPerConnSec > 0 {
			srv.SetMaxOpsPerConnSec(cfg.MaxOpsPerConnSec)
		}
		if cfg.ClusterEnabled {
			srv.SetCluster(ring, replicator, cfg.ClusterNodeID, cfg.SnapshotDir)
		}
		if err := srv.Start(); err != nil {
			log.Fatalf("[l3-serve] failed to start TCP server on %s: %v", addr, err)
		}
		tcpServers = append(tcpServers, srv)
		log.Printf("[l3-serve] TCP server listening on %s", addr)
	}

	log.Printf("[l3-serve] L3 server ready: %d shards, max_memory=%d GB, eviction=%s",
		numShardsVal, cfg.MaxMemoryGB, cfg.EvictionPolicy)

	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	log.Printf("[l3-serve] received signal %v, shutting down gracefully...", sig)

	collector.SetShuttingDown()
	for _, srv := range tcpServers {
		srv.Stop()
	}
	if gossip != nil {
		gossip.Stop()
	}
	if replicator != nil {
		replicator.Stop()
	}

	if cfg.AutoSnapshot && cfg.SnapshotDir != "" {
		log.Printf("[snapshot] saving snapshot to %s...", cfg.SnapshotDir)
		w := snapshot.NewWriter(cfg.SnapshotDir)
		var totalKeys int64
		var totalBytes int64
		for i := 0; i < mgr.NumShards(); i++ {
			result := mgr.Shard(i).Submit(shard.ShardOp{
				Type:   shard.OpSnapshot,
				Result: make(chan shard.OpResult, 1),
			})
			if err := w.WriteShard(i, result.SnapshotEntries); err != nil {
				log.Printf("[snapshot] shard %d write error: %v", i, err)
			}
			totalKeys += int64(len(result.SnapshotEntries))
			for _, e := range result.SnapshotEntries {
				totalBytes += int64(len(e.Key) + len(e.Value))
			}
		}
		m := snapshot.Manifest{
			Version:       1,
			ServerVersion: version.ServerVersion,
			CreatedAt:     time.Now().Format(time.RFC3339),
			ShardCount:    mgr.NumShards(),
			AllocatorMode: cfg.AllocatorMode,
			TotalKeys:     totalKeys,
			TotalBytes:    totalBytes,
		}
		for i := 0; i < mgr.NumShards(); i++ {
			m.Files = append(m.Files, fmt.Sprintf("shard-%d.dat", i))
		}
		if err := w.WriteManifest(m); err != nil {
			log.Printf("[snapshot] manifest write error: %v", err)
		} else {
			log.Printf("[snapshot] saved %d keys (%.1f MB) to %s",
				totalKeys, float64(totalBytes)/(1024*1024), cfg.SnapshotDir)
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if metricsSrv != nil {
		_ = metricsSrv.Shutdown(ctx)
	}
	if pprofSrv != nil {
		_ = pprofSrv.Shutdown(ctx)
	}
	mgr.Stop()
	log.Printf("[l3-serve] shutdown complete")
}
