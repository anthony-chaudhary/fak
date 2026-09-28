// Package l3server provides the top-level orchestration and server runtime for
// the L3 disaggregated KV-cache tier.
//
// Invariant: L3 server provides disaggregated, multi-tenant KV-cache storage with bounded slab allocation and deterministic eviction.
// Guard: fail-closed admission and shard isolation prevent memory corruption and cross-tenant leakage.
package l3server

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/config"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/version"
)

// ServerStatus describes the operational state of the L3 server instance.
type ServerStatus string

const (
	// StatusStarting indicates the server is initializing shards and allocations.
	StatusStarting ServerStatus = "starting"
	// StatusRunning indicates the server is actively serving requests.
	StatusRunning ServerStatus = "running"
	// StatusStopping indicates graceful shutdown is underway.
	StatusStopping ServerStatus = "stopping"
	// StatusStopped indicates the server has fully released all resources.
	StatusStopped ServerStatus = "stopped"
)

// Server encapsulates the multi-shard disaggregated cache daemon.
type Server struct {
	mu           sync.RWMutex
	cfg          *config.Config
	mgrCfg       shard.ManagerConfig
	shardManager *shard.Manager
	metrics      *metrics.Collector
	status       atomic.Value
	startedAt    time.Time
}

// defaultConfig supplies the configuration NewServer uses when given nil. It is
// a package seam so tests can exercise the nil-config path without provisioning
// the appliance-sized default (512 GB on Linux, fully committed up front by the
// MAP_POPULATE slab regions).
var defaultConfig = config.DefaultConfig

// NewServer constructs an initialized L3 cache server from configuration.
//
// NewServer provisions (and on Linux commits) every shard's slab memory; call
// Stop to release it even if the server is never started.
func NewServer(cfg *config.Config) (*Server, error) {
	if cfg == nil {
		def := defaultConfig()
		cfg = &def
	}

	startedAt := time.Now()
	startup := &metrics.StartupState{}
	startup.Ready.Store(true)
	connReg := metrics.NewConnRegistry()
	clientStatsReg := metrics.NewClientStatsRegistry()
	collector := metrics.NewCollector(startup, connReg, clientStatsReg, startedAt)
	collector.MaxKeysPerShard = cfg.MaxKeys
	mgrCfg := shard.ManagerConfig{
		NumShards:           cfg.NumShards,
		MaxMemoryGB:         cfg.MaxMemoryGB,
		EvictionPolicy:      cfg.EvictionPolicy,
		AllocatorMode:       cfg.AllocatorMode,
		ModelPageBytes:      cfg.ModelPageBytes,
		WarmupOps:           cfg.WarmupOps,
		AutoTuneSlabs:       cfg.AutoTuneSlabs,
		SlabDistribution:    cfg.SlabDistribution,
		VerboseShardLogging: cfg.VerboseShardLogging,
	}

	manager, err := shard.NewManager(mgrCfg)
	if err != nil {
		return nil, fmt.Errorf("l3server: initialize shard manager: %w", err)
	}

	s := &Server{
		cfg:          cfg,
		mgrCfg:       mgrCfg,
		shardManager: manager,
		metrics:      collector,
	}
	s.status.Store(StatusStopped)
	return s, nil
}

// Start begins serving cache traffic across all configured shards.
//
// Start is idempotent in the sense that a Stop/Start cycle is supported: the
// shard manager owns single-shot lifecycle channels (quit/done) and its shards'
// allocators are released on Stop, so a restart provisions a fresh manager
// rather than reusing a torn-down one. A Start while already running is
// refused, matching the existing "server is already running" contract.
func (s *Server) Start(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	curr := s.status.Load().(ServerStatus)
	if curr == StatusRunning {
		return fmt.Errorf("l3server: server is already running")
	}

	// A previous Stop released the manager (and its shard allocators); build a
	// fresh one so the single-shot channels are never closed twice.
	if s.shardManager == nil {
		manager, err := shard.NewManager(s.mgrCfg)
		if err != nil {
			return fmt.Errorf("l3server: reinitialize shard manager: %w", err)
		}
		s.shardManager = manager
	}

	s.status.Store(StatusStarting)
	s.startedAt = time.Now().UTC()
	s.shardManager.Start()
	s.status.Store(StatusRunning)
	return nil
}

// Stop initiates graceful termination of all shards and frees memory mappings.
// Stop is idempotent: a Stop on an already-stopped server is a no-op, and the
// released manager is dropped so the next Start provisions a fresh one. A
// server that was constructed but never started still holds the shard
// allocators NewServer provisioned, so Stop releases those too.
func (s *Server) Stop(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	curr := s.status.Load().(ServerStatus)
	if curr == StatusStopped && s.shardManager == nil {
		return nil
	}

	s.status.Store(StatusStopping)
	if s.shardManager != nil {
		s.shardManager.Stop()
		// Drop the stopped manager so a subsequent Start rebuilds one with
		// fresh lifecycle channels and allocators.
		s.shardManager = nil
	}
	s.status.Store(StatusStopped)
	return nil
}

// Status returns the current operational lifecycle state of the server.
func (s *Server) Status() ServerStatus {
	val := s.status.Load()
	if val == nil {
		return StatusStopped
	}
	return val.(ServerStatus)
}

// Uptime calculates the duration since the server transitioned to running.
func (s *Server) Uptime() time.Duration {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.Status() != StatusRunning {
		return 0
	}
	return time.Since(s.startedAt)
}

// Version returns canonical build and semantic version identity metadata.
func (s *Server) Version() string {
	return fmt.Sprintf("l3server %s (commit %s)", version.ServerVersion, version.Commit)
}

// ShardManager exposes the internal shard coordinator for diagnostic inspections.
func (s *Server) ShardManager() *shard.Manager {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.shardManager
}

// MetricsCollector returns the telemetry collector recording operational counters.
func (s *Server) MetricsCollector() *metrics.Collector {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.metrics
}
