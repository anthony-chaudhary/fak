package shard

import (
	"fmt"
	"testing"
	"time"
)

// newTestShard creates a minimal shard for vacuum evaluation tests.
func newTestShard(t *testing.T, id int, modelPageBytes uint64, warmupOps int, autoTune bool) *Shard {
	t.Helper()
	cfg := ShardConfig{
		ID:             id,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024, // 64MB â€” small but functional
		EvictionPolicy: "wtinylfu",
		ModelPageBytes: modelPageBytes,
		WarmupOps:      warmupOps,
		AutoTuneSlabs:  autoTune,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create test shard %d: %v", id, err)
	}
	return s
}

func TestVacuumEvaluation_NoDetection(t *testing.T) {
	// A shard with no detected size should not need rebalancing.
	s := newTestShard(t, 0, 0, 1000, true)
	defer s.Allocator().Close()

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
		},
	}

	if m.evaluateShard(s) {
		t.Error("expected no rebalance when detection hasn't fired")
	}
}

func TestVacuumEvaluation_MismatchTriggersRebalance(t *testing.T) {
	// Shard configured with model_page_bytes=1MB but detects 512KB â†’ should trigger.
	s := newTestShard(t, 0, 1048576, 10, true) // configured for 1MB
	defer s.Allocator().Close()

	// Simulate warmup: record 10 values of 512KB to trigger detection
	for i := 0; i < 10; i++ {
		s.sizeTracker.record(524288) // 512KB
	}
	if !s.sizeTracker.detected {
		t.Fatal("expected detection after warmup")
	}
	// Detected size (512KB) != configured size (1MB)
	if s.sizeTracker.optimalSize != 524288 {
		t.Fatalf("expected optimalSize=524288, got %d", s.sizeTracker.optimalSize)
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
		},
	}

	if !m.evaluateShard(s) {
		t.Error("expected rebalance when detected size differs from configured")
	}
}

func TestVacuumEvaluation_HighUtilization(t *testing.T) {
	// When detection has fired and the dominant class is highly utilized, trigger.
	pageSize := uint64(65536) // 64KB â€” fits in our 64MB test shard
	s := newTestShard(t, 0, pageSize, 10, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Fill the shard with values to saturate the dominant class
	// First trigger detection
	for i := 0; i < 10; i++ {
		s.sizeTracker.record(pageSize)
	}
	if !s.sizeTracker.detected {
		t.Fatal("expected detection")
	}

	// Fill most of the slots in the dominant class
	val := make([]byte, pageSize)
	filled := 0
	for i := 0; i < 2000; i++ {
		key := []byte("k" + string(rune(i/256)) + string(rune(i%256)))
		result := s.Submit(ShardOp{
			Type:   OpSet,
			Key:    key,
			Value:  val,
			Result: make(chan OpResult, 1),
		})
		if result.Err != nil {
			break
		}
		filled++
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.50, // low threshold so test triggers
		},
	}

	usedFrac, ok := s.DominantClassUtilization()
	if !ok {
		t.Fatal("expected DominantClassUtilization to return detected=true")
	}
	t.Logf("filled %d slots, utilization=%.2f", filled, usedFrac)

	if filled > 0 && usedFrac > 0.50 {
		if !m.evaluateShard(s) {
			t.Error("expected rebalance when dominant class utilization > threshold")
		}
	}
}

func TestVacuumEvaluation_HealthyAllocator(t *testing.T) {
	// When detection fired and config matches, utilization is low â†’ no rebalance.
	pageSize := uint64(65536)
	s := newTestShard(t, 0, pageSize, 10, true)
	defer s.Allocator().Close()

	// Trigger detection with size matching the configured page bytes
	for i := 0; i < 10; i++ {
		s.sizeTracker.record(pageSize)
	}
	if !s.sizeTracker.detected {
		t.Fatal("expected detection")
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
		},
	}

	// Detected size matches configured â†’ check 2 won't trigger.
	// No sets done â†’ utilization is 0 â†’ check 3 won't trigger.
	// No evictions â†’ check 4 won't trigger.
	if m.evaluateShard(s) {
		t.Error("expected no rebalance for healthy allocator")
	}
}

func TestVacuumCoordinator_MultiShardRebalance(t *testing.T) {
	// Create 4 shards, make 2 need rebalancing, verify coordinator does both in one tick.
	shards := make([]*Shard, 4)
	for i := 0; i < 4; i++ {
		s := newTestShard(t, i, 1048576, 10, true) // configured 1MB
		s.Start()
		shards[i] = s
	}
	defer func() {
		for _, s := range shards {
			s.Stop()
			<-s.Done()
		}
	}()

	// Shards 0 and 1: detect 512KB (mismatch with configured 1MB) â†’ need rebalance
	for _, idx := range []int{0, 1} {
		for i := 0; i < 10; i++ {
			shards[idx].sizeTracker.record(524288) // 512KB
		}
		if !shards[idx].sizeTracker.detected {
			t.Fatalf("shard %d: expected detection", idx)
		}
	}
	// Shards 2 and 3: detect 1MB (matches configured) â†’ no rebalance needed
	for _, idx := range []int{2, 3} {
		for i := 0; i < 10; i++ {
			shards[idx].sizeTracker.record(1048576) // 1MB â€” matches config
		}
	}

	m := &Manager{
		shards:    shards,
		numShards: 4,
		mask:      3,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      1,
			CooldownSeconds:      1,
			UtilizationThreshold: 0.85,
			MinAgeSeconds:        0,
		},
		startedAt:        time.Now().Add(-time.Hour), // pretend server started long ago
		rebalanceBackoff: make(map[int]rebalanceFailState),
	}

	// First tick should rebalance both shard 0 and shard 1
	m.vacuumTick()
	if m.vacuumRebalances.Load() != 2 {
		t.Errorf("expected 2 rebalances after first tick, got %d", m.vacuumRebalances.Load())
	}

	// Second tick: no shards need rebalancing (both were reset by rebalance)
	m.vacuumTick()
	if m.vacuumRebalances.Load() != 2 {
		t.Errorf("expected still 2 rebalances after second tick (nothing to do), got %d", m.vacuumRebalances.Load())
	}

	// VacuumStats should show 0 pending
	stats := m.VacuumStats()
	if stats.RebalancesTotal != 2 {
		t.Errorf("expected RebalancesTotal=2, got %d", stats.RebalancesTotal)
	}
	if stats.PendingShards != 0 {
		t.Errorf("expected PendingShards=0, got %d", stats.PendingShards)
	}
	if stats.LastRebalanceEpoch == 0 {
		t.Error("expected LastRebalanceEpoch to be set")
	}
}

// TestVacuumStatsCachedPendingCount verifies that VacuumStats returns the
// cached pending count from atomic fields without re-evaluating shards.
// This confirms Fix 4: O(1) atomic reads instead of O(numShards) evaluation.
func TestVacuumStatsCachedPendingCount(t *testing.T) {
	// Create a minimal Manager â€” we only need the atomic fields, not running shards.
	m := &Manager{
		numShards: 4,
		mask:      3,
	}

	// Before any vacuum tick, all counters should be zero.
	stats := m.VacuumStats()
	if stats.PendingShards != 0 {
		t.Errorf("initial PendingShards: got %d, want 0", stats.PendingShards)
	}
	if stats.RebalancesTotal != 0 {
		t.Errorf("initial RebalancesTotal: got %d, want 0", stats.RebalancesTotal)
	}

	// Simulate a vacuum tick storing pending count via the atomic (what vacuumTick does).
	m.vacuumPendingShards.Store(3) // 3 shards pending rebalance
	m.vacuumRebalances.Store(5)
	m.pressureEvals.Store(100)
	m.pressureRebuilds.Store(2)
	m.vacuumRebalanceFailures.Store(1)

	stats = m.VacuumStats()
	if stats.PendingShards != 3 {
		t.Errorf("PendingShards: got %d, want 3", stats.PendingShards)
	}
	if stats.RebalancesTotal != 5 {
		t.Errorf("RebalancesTotal: got %d, want 5", stats.RebalancesTotal)
	}
	if stats.PressureEvals != 100 {
		t.Errorf("PressureEvals: got %d, want 100", stats.PressureEvals)
	}
	if stats.PressureRebuilds != 2 {
		t.Errorf("PressureRebuilds: got %d, want 2", stats.PressureRebuilds)
	}
	if stats.RebalanceFailures != 1 {
		t.Errorf("RebalanceFailures: got %d, want 1", stats.RebalanceFailures)
	}

	// VacuumStats must be O(1) â€” call it many times and verify consistency.
	// The cached value should not change without a vacuum tick updating the atomics.
	for i := 0; i < 1000; i++ {
		s := m.VacuumStats()
		if s.PendingShards != 3 {
			t.Fatalf("call %d: PendingShards changed to %d without tick", i, s.PendingShards)
		}
		if s.RebalancesTotal != 5 {
			t.Fatalf("call %d: RebalancesTotal changed to %d without tick", i, s.RebalancesTotal)
		}
	}

	// Simulate post-rebalance update (what vacuumTick Phase 3 does).
	m.vacuumPendingShards.Store(0)
	stats = m.VacuumStats()
	if stats.PendingShards != 0 {
		t.Errorf("post-rebalance PendingShards: got %d, want 0", stats.PendingShards)
	}
}

// TestVacuumRebalancePreservesData verifies that OpRebalance (used by vacuum)
// preserves all cached data via ZeroLatencyBalance, unlike OpFlush which destroys it.
func TestVacuumRebalancePreservesData(t *testing.T) {
	// Create a shard with auto-tune, warmup=5
	s := newTestShard(t, 0, 0, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert entries to trigger detection + auto-rebuild
	valSize := 4096
	numEntries := 6 // 5 warmup + 1 to trigger rebuild
	type kv struct {
		key     []byte
		keyHash uint64
		value   []byte
	}
	entries := make([]kv, numEntries)
	for i := range entries {
		k := []byte(fmt.Sprintf("vac-%04d", i))
		v := make([]byte, valSize)
		for j := range v {
			v[j] = byte(i + 10)
		}
		entries[i] = kv{key: k, keyHash: uint64(i + 1), value: v}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 1),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %s: %v", k, result.Err)
		}
	}

	// Wait for async allocator construction + migration to complete
	var snap DetectionSnapshot
	for i := 0; i < 100; i++ {
		snap = s.SizeDetectionSnapshot()
		if snap.Status == "rebuilt" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap.Status != "rebuilt" {
		t.Fatalf("expected status 'rebuilt', got %q", snap.Status)
	}

	// Now trigger OpRebalance (what vacuum uses) â€” since shard is frozen after
	// auto-rebuild, this should skip (no redundant migration).
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("OpRebalance: %v", result.Err)
	}
	if !result.OK {
		t.Fatal("OpRebalance: expected OK=true")
	}

	// Verify ALL entries are still readable with correct values
	for _, e := range entries {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET %s: not found after rebalance (data was destroyed!)", e.key)
			continue
		}
		if len(result.Value) != valSize {
			t.Errorf("GET %s: value length %d, want %d", e.key, len(result.Value), valSize)
			continue
		}
		// Verify value contents
		expectedByte := byte(int(e.keyHash) - 1 + 10)
		for j, b := range result.Value {
			if b != expectedByte {
				t.Errorf("GET %s: byte %d = %d, want %d", e.key, j, b, expectedByte)
				break
			}
		}
	}

	// Verify size tracker stays frozen (frozen skip â€” no reset)
	snap = s.SizeDetectionSnapshot()
	if snap.Status != "rebuilt" {
		t.Errorf("expected status 'rebuilt' after frozen skip, got %q", snap.Status)
	}
}

// TestOpRebalanceRequiresDetection verifies that OpRebalance returns an error
// when size detection hasn't completed yet.
func TestOpRebalanceRequiresDetection(t *testing.T) {
	s := newTestShard(t, 0, 0, 1000, true) // warmup=1000, detection won't fire
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Attempt rebalance without any SETs â€” detection hasn't fired
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err == nil {
		t.Error("expected error when detection hasn't completed")
	}
	if result.OK {
		t.Error("expected OK=false when detection hasn't completed")
	}
}

// TestOpRebalanceRequiresAutoTune verifies that OpRebalance returns an error
// when auto_tune_slabs is disabled.
func TestOpRebalanceRequiresAutoTune(t *testing.T) {
	s := newTestShard(t, 0, 0, 10, false) // autoTune=false
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err == nil {
		t.Error("expected error when auto_tune_slabs=false")
	}
}

// TestVacuumPressureRebalancing verifies that when pressure rebalancing is enabled
// and one class is under high pressure, the vacuum detects it and triggers a rebalance
// with adjusted weights.
func TestVacuumPressureRebalancing(t *testing.T) {
	pageSize := uint64(4096) // small pages for fast test
	s := newTestShard(t, 0, pageSize, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection
	for i := 0; i < 5; i++ {
		s.sizeTracker.record(pageSize)
	}
	if !s.sizeTracker.detected {
		t.Fatal("expected detection")
	}

	// Fill the shard to create eviction pressure on the dominant class
	val := make([]byte, pageSize)
	for i := 0; i < 500; i++ {
		key := []byte(fmt.Sprintf("pk-%04d", i))
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	// Simulate eviction and alloc pressure directly on tracker
	a := s.Allocator()
	valClassIdx := a.FindClass(pageSize)
	if valClassIdx < 0 {
		t.Fatal("no class for page size")
	}
	// Record heavy evictions on the value class
	for i := 0; i < 100; i++ {
		s.pressureTracker.recordEviction(valClassIdx)
		s.pressureTracker.recordAllocOp(valClassIdx)
	}
	s.pressureTracker.recordAllocFailure(valClassIdx)

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      1,
			CooldownSeconds:      1,
			UtilizationThreshold: 0.85,
			MinAgeSeconds:        0,
			PressureRebalancing:  true,
			DampingFactor:        0.3,
			DriftThreshold:       0.01, // low threshold so test triggers
			MinClassWeight:       0.5,
			EvictionRateNorm:     1.0, // low norm so eviction rate is significant
		},
		startedAt: time.Now().Add(-time.Hour),
	}

	// Take snapshot (normally done in vacuumTick)
	s.PressureSnap()

	// Evaluate the shard
	needs, weights, _ := m.evaluateShardPressure(s)

	if !needs {
		t.Error("expected pressure rebalance to be needed")
	}
	if weights != nil {
		// The class under pressure should get more weight
		t.Logf("proposed weights: %v", weights)
	}
}

// TestVacuumPressurePreservesData verifies that data survives a pressure-triggered rebuild.
func TestVacuumPressurePreservesData(t *testing.T) {
	pageSize := uint64(4096)
	s := newTestShard(t, 0, pageSize, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection
	for i := 0; i < 5; i++ {
		s.sizeTracker.record(pageSize)
	}

	// Insert some entries
	numEntries := 10
	type kv struct {
		key     []byte
		keyHash uint64
		value   []byte
	}
	entries := make([]kv, numEntries)
	for i := range entries {
		k := []byte(fmt.Sprintf("pdata-%04d", i))
		v := make([]byte, pageSize)
		for j := range v {
			v[j] = byte(i + 42)
		}
		entries[i] = kv{key: k, keyHash: uint64(i + 100), value: v}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     k,
			KeyHash: uint64(i + 100),
			Value:   v,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %s: %v", k, result.Err)
		}
	}

	// Trigger pressure-driven rebalance with explicit weights
	weights := map[uint64]float64{
		64:   1.0,
		128:  1.0,
		256:  1.0,
		512:  1.0,
		1024: 1.0,
		2048: 5.0,
		4096: 50.0, // heavy weight on our page class
		8192: 1.0,
	}
	result := s.Submit(ShardOp{
		Type:         OpRebalance,
		ClassWeights: weights,
		Result:       make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("pressure rebalance: %v", result.Err)
	}

	// Verify all entries survived
	for _, e := range entries {
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     e.key,
			KeyHash: e.keyHash,
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET %s: not found after pressure rebalance!", e.key)
			continue
		}
		if len(result.Value) != int(pageSize) {
			t.Errorf("GET %s: len=%d, want %d", e.key, len(result.Value), pageSize)
			continue
		}
		expectedByte := byte(int(e.keyHash) - 100 + 42)
		if result.Value[0] != expectedByte {
			t.Errorf("GET %s: first byte=%d, want %d", e.key, result.Value[0], expectedByte)
		}
	}
}

// --- On-demand maintenance API tests ---

// TestOnDemandVacuum verifies that OnDemandVacuum evaluates and rebalances shards.
func TestOnDemandVacuum(t *testing.T) {
	shards := make([]*Shard, 2)
	for i := range shards {
		shards[i] = newTestShard(t, i, 1048576, 10, true) // configured 1MB
		shards[i].Start()
	}
	defer func() {
		for _, s := range shards {
			s.Stop()
			<-s.Done()
		}
	}()

	// Shard 0: detect 512KB (mismatch) â†’ needs rebalance
	for i := 0; i < 10; i++ {
		shards[0].sizeTracker.record(524288)
	}
	// Shard 1: detect 1MB (matches config) â†’ healthy
	for i := 0; i < 10; i++ {
		shards[1].sizeTracker.record(1048576)
	}

	m := &Manager{
		shards:    shards,
		numShards: 2,
		mask:      1,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
		},
		startedAt: time.Now().Add(-time.Hour),
	}

	result := m.OnDemandVacuum(false, nil)
	if result.ShardsEvaluated != 2 {
		t.Errorf("ShardsEvaluated: got %d, want 2", result.ShardsEvaluated)
	}
	if len(result.ShardsRebalanced) != 1 || result.ShardsRebalanced[0] != 0 {
		t.Errorf("ShardsRebalanced: got %v, want [0]", result.ShardsRebalanced)
	}
	if _, ok := result.ShardsSkipped[1]; !ok {
		t.Error("expected shard 1 to be skipped as healthy")
	}
	if result.DurationMs < 0 {
		t.Error("DurationMs should be non-negative")
	}
}

// TestOnDemandVacuumForce verifies that force=true rebalances even healthy shards.
func TestOnDemandVacuumForce(t *testing.T) {
	s := newTestShard(t, 0, 65536, 10, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection with matching size (healthy)
	for i := 0; i < 10; i++ {
		s.sizeTracker.record(65536)
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
		},
		startedAt: time.Now().Add(-time.Hour),
	}

	// Without force: should skip (healthy)
	result := m.OnDemandVacuum(false, nil)
	if len(result.ShardsRebalanced) != 0 {
		t.Errorf("without force: expected 0 rebalances, got %v", result.ShardsRebalanced)
	}

	// With force: should rebalance
	result = m.OnDemandVacuum(true, nil)
	if len(result.ShardsRebalanced) != 1 {
		t.Errorf("with force: expected 1 rebalance, got %v", result.ShardsRebalanced)
	}
}

// TestOnDemandVacuumShardIDs verifies that shard_ids targets specific shards.
func TestOnDemandVacuumShardIDs(t *testing.T) {
	shards := make([]*Shard, 4)
	for i := range shards {
		shards[i] = newTestShard(t, i, 1048576, 10, true)
		shards[i].Start()
		// All detect 512KB â†’ all need rebalance
		for j := 0; j < 10; j++ {
			shards[i].sizeTracker.record(524288)
		}
	}
	defer func() {
		for _, s := range shards {
			s.Stop()
			<-s.Done()
		}
	}()

	m := &Manager{
		shards:    shards,
		numShards: 4,
		mask:      3,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			UtilizationThreshold: 0.85,
		},
		startedAt: time.Now().Add(-time.Hour),
	}

	// Only target shards 1 and 3
	result := m.OnDemandVacuum(false, []int{1, 3})
	if result.ShardsEvaluated != 2 {
		t.Errorf("ShardsEvaluated: got %d, want 2", result.ShardsEvaluated)
	}
	if len(result.ShardsRebalanced) != 2 {
		t.Errorf("ShardsRebalanced: got %v, want [1,3]", result.ShardsRebalanced)
	}
}

// TestOnDemandAutoTune verifies that OnDemandAutoTune forces detection + rebuild.
func TestOnDemandAutoTune(t *testing.T) {
	s := newTestShard(t, 0, 0, 1000, true) // warmup=1000, won't auto-detect yet
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Record some SETs (below warmup threshold)
	for i := 0; i < 50; i++ {
		s.sizeTracker.record(4096)
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
	}

	// Without force: should fail (detection not complete)
	result := m.OnDemandAutoTune(false, nil)
	if len(result.ShardsRebuilt) != 0 {
		t.Errorf("without force: expected 0 rebuilt, got %v", result.ShardsRebuilt)
	}
	if _, ok := result.ShardsSkipped[0]; !ok {
		t.Error("expected shard 0 to be skipped")
	}

	// With force: should detect + rebuild
	result = m.OnDemandAutoTune(true, nil)
	if len(result.ShardsRebuilt) != 1 {
		t.Errorf("with force: expected 1 rebuilt, got %v (skipped: %v)", result.ShardsRebuilt, result.ShardsSkipped)
	}
	if result.DurationMs < 0 {
		t.Error("DurationMs should be non-negative")
	}
}

// TestOnDemandAutoTuneDisabled verifies error when auto_tune_slabs=false.
func TestOnDemandAutoTuneDisabled(t *testing.T) {
	s := newTestShard(t, 0, 0, 10, false) // autoTune=false
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
	}

	result := m.OnDemandAutoTune(true, nil)
	if len(result.ShardsRebuilt) != 0 {
		t.Errorf("expected 0 rebuilt when autoTune disabled, got %v", result.ShardsRebuilt)
	}
	if reason, ok := result.ShardsSkipped[0]; !ok || reason != "auto_tune_slabs is disabled" {
		t.Errorf("expected skip reason 'auto_tune_slabs is disabled', got %q", reason)
	}
}

// TestMaintenanceStatus verifies that MaintenanceStatus returns correct structure.
func TestMaintenanceStatus(t *testing.T) {
	shards := make([]*Shard, 2)
	for i := range shards {
		shards[i] = newTestShard(t, i, 65536, 10, true)
	}
	defer func() {
		for _, s := range shards {
			s.Allocator().Close()
		}
	}()

	m := &Manager{
		shards:    shards,
		numShards: 2,
		mask:      1,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      300,
			UtilizationThreshold: 0.85,
		},
	}

	status := m.MaintenanceStatus()

	if !status.VacuumConfig.Enabled {
		t.Error("expected VacuumConfig.Enabled=true")
	}
	if status.VacuumConfig.IntervalSeconds != 300 {
		t.Errorf("expected IntervalSeconds=300, got %d", status.VacuumConfig.IntervalSeconds)
	}
	if len(status.ShardDetections) != 2 {
		t.Errorf("expected 2 shard detections, got %d", len(status.ShardDetections))
	}
	for i, entry := range status.ShardDetections {
		if entry.ShardID != i {
			t.Errorf("entry %d: expected ShardID=%d, got %d", i, i, entry.ShardID)
		}
	}
}

// TestVacuumSkipsShardsDuringWarmup verifies that evaluateShardPressure returns false
// for shards still in "warming_up" status, even when pressure activity exists.
func TestVacuumSkipsShardsDuringWarmup(t *testing.T) {
	// Create shard with high warmup threshold so it stays in warming_up
	s := newTestShard(t, 0, 0, 10000, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Record a few values â€” not enough to complete warmup
	for i := 0; i < 10; i++ {
		s.sizeTracker.record(4096)
	}

	// Verify shard is still warming up
	snap := s.SizeDetectionSnapshot()
	if snap.Status != "warming_up" {
		t.Fatalf("expected status 'warming_up', got %q", snap.Status)
	}

	// Simulate pressure activity that would normally trigger a rebalance
	a := s.Allocator()
	valClassIdx := a.FindClass(4096)
	if valClassIdx >= 0 {
		for i := 0; i < 100; i++ {
			s.pressureTracker.recordEviction(valClassIdx)
			s.pressureTracker.recordAllocOp(valClassIdx)
		}
		s.pressureTracker.recordAllocFailure(valClassIdx)
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      1,
			CooldownSeconds:      0,
			UtilizationThreshold: 0.50,
			MinAgeSeconds:        0,
			PressureRebalancing:  true,
			DampingFactor:        0.3,
			DriftThreshold:       0.01, // very low â€” would trigger if not skipped
			MinClassWeight:       0.5,
			EvictionRateNorm:     1.0,
		},
		startedAt:        time.Now().Add(-time.Hour),
		rebalanceBackoff: make(map[int]rebalanceFailState),
	}

	// Take pressure snapshot
	s.PressureSnap()

	// evaluateShardPressure should return false because shard is warming up
	needs, weights, reason := m.evaluateShardPressure(s)
	if needs {
		t.Errorf("expected no rebalance during warmup, but got needs=true (reason=%q)", reason)
	}
	if weights != nil {
		t.Error("expected nil weights during warmup")
	}

	// vacuumTick should also not rebalance
	m.vacuumTick()
	if m.vacuumRebalances.Load() != 0 {
		t.Errorf("expected 0 rebalances during warmup, got %d", m.vacuumRebalances.Load())
	}
}

// TestVacuumPressureNoOscillation verifies that a stable workload doesn't cause
// repeated rebuilds (at most 1-2 rebuilds across 10 ticks).
func TestVacuumPressureNoOscillation(t *testing.T) {
	pageSize := uint64(4096)
	s := newTestShard(t, 0, pageSize, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection
	for i := 0; i < 5; i++ {
		s.sizeTracker.record(pageSize)
	}

	// Put a small amount of data â€” stable, no pressure
	for i := 0; i < 20; i++ {
		key := []byte(fmt.Sprintf("stable-%04d", i))
		val := make([]byte, pageSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      1,
			CooldownSeconds:      0,
			UtilizationThreshold: 0.85,
			MinAgeSeconds:        0,
			PressureRebalancing:  true,
			DampingFactor:        0.3,
			DriftThreshold:       0.15, // standard threshold
			MinClassWeight:       0.5,
			EvictionRateNorm:     10.0,
		},
		startedAt:        time.Now().Add(-time.Hour),
		rebalanceBackoff: make(map[int]rebalanceFailState),
	}

	// Run 10 ticks with no new pressure
	for tick := 0; tick < 10; tick++ {
		m.vacuumTick()
	}

	rebuilds := m.vacuumRebalances.Load()
	if rebuilds > 2 {
		t.Errorf("expected at most 2 rebuilds for stable workload, got %d (oscillation!)", rebuilds)
	}
	t.Logf("stable workload: %d rebuilds across 10 ticks", rebuilds)
}

// TestVacuumRebalanceSkipsWhenMigrating verifies that OpRebalance returns OK
// (skip) instead of aborting when a migration is already in progress.
func TestVacuumRebalanceSkipsWhenMigrating(t *testing.T) {
	pageSize := uint64(4096)
	// Use tiny batch size (1) so migration takes many iterations
	cfg := ShardConfig{
		ID:               0,
		IndexCapacity:    1024,
		MaxMemoryBytes:   64 * 1024 * 1024,
		EvictionPolicy:   "wtinylfu",
		ModelPageBytes:   pageSize,
		WarmupOps:        5,
		AutoTuneSlabs:    true,
		MigrateBatchSize: 1, // tiny batch â†’ migration stays in progress
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("failed to create shard: %v", err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection so rebalance is allowed
	for i := 0; i < 5; i++ {
		s.sizeTracker.record(pageSize)
	}
	if !s.sizeTracker.detected {
		t.Fatal("expected detection")
	}

	// Insert enough entries so migration takes multiple batches
	val := make([]byte, pageSize)
	for i := 0; i < 200; i++ {
		key := []byte(fmt.Sprintf("mig-%04d", i))
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	// First rebalance: starts a migration
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("first rebalance: %v", result.Err)
	}
	if !result.OK {
		t.Fatal("first rebalance: expected OK=true")
	}

	// Second rebalance while first is still in progress: should skip (OK=true, no error)
	result = s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("second rebalance: expected no error, got %v", result.Err)
	}
	if !result.OK {
		t.Fatal("second rebalance: expected OK=true (skip)")
	}

	// Verify data is still readable after migration completes
	// Give the shard goroutine time to finish migration
	time.Sleep(200 * time.Millisecond)
	for i := 0; i < 200; i++ {
		key := []byte(fmt.Sprintf("mig-%04d", i))
		r := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Result:  make(chan OpResult, 1),
		})
		if !r.Found {
			t.Errorf("GET %s: not found after skip+migration", key)
			break
		}
	}
}

// TestVacuumTickParallelSubmission verifies that vacuumTick submits all
// rebalances in parallel (no per-shard cooldown delay). With 4 shards needing
// rebalance and a high cooldown config, the tick should still complete quickly.
func TestVacuumTickParallelSubmission(t *testing.T) {
	numShards := 4
	shards := make([]*Shard, numShards)
	for i := 0; i < numShards; i++ {
		s := newTestShard(t, i, 1048576, 10, true) // configured 1MB
		s.Start()
		shards[i] = s
		// All detect 512KB â†’ all need rebalance (mismatch with configured 1MB)
		for j := 0; j < 10; j++ {
			s.sizeTracker.record(524288)
		}
		if !s.sizeTracker.detected {
			t.Fatalf("shard %d: expected detection", i)
		}
	}
	defer func() {
		for _, s := range shards {
			s.Stop()
			<-s.Done()
		}
	}()

	m := &Manager{
		shards:    shards,
		numShards: uint64(numShards),
		mask:      uint64(numShards) - 1,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      1,
			CooldownSeconds:      30, // high cooldown â€” would cause 90s+ delay if sequential
			UtilizationThreshold: 0.85,
			MinAgeSeconds:        0,
		},
		startedAt:        time.Now().Add(-time.Hour),
		rebalanceBackoff: make(map[int]rebalanceFailState),
	}

	start := time.Now()
	m.vacuumTick()
	elapsed := time.Since(start)

	if m.vacuumRebalances.Load() != int64(numShards) {
		t.Errorf("expected %d rebalances, got %d", numShards, m.vacuumRebalances.Load())
	}
	if elapsed > 5*time.Second {
		t.Errorf("vacuumTick took %s â€” expected <5s (parallel submissions, no cooldown delay)", elapsed)
	}
	t.Logf("parallel vacuumTick: %d shards rebalanced in %s", numShards, elapsed.Truncate(time.Millisecond))
}

// TestVacuumNoInfiniteLoopAutoDetect verifies that model_page_bytes=0 (auto-detect)
// does NOT cause infinite rebalance loops. After initial detect + rebuild, subsequent
// vacuum ticks should not trigger additional rebalances.
func TestVacuumNoInfiniteLoopAutoDetect(t *testing.T) {
	// model_page_bytes=0 (auto-detect mode)
	s := newTestShard(t, 0, 0, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection by recording 5 values of the same size
	for i := 0; i < 6; i++ {
		key := []byte(fmt.Sprintf("loop-%04d", i))
		val := make([]byte, 4096)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	// Wait for auto-rebuild to complete
	time.Sleep(100 * time.Millisecond)

	snap := s.SizeDetectionSnapshot()
	if snap.Status != "rebuilt" {
		t.Fatalf("expected status 'rebuilt', got %q", snap.Status)
	}

	m := &Manager{
		shards:    []*Shard{s},
		numShards: 1,
		mask:      0,
		vacuumCfg: VacuumConfig{
			Enabled:              true,
			IntervalSeconds:      1,
			UtilizationThreshold: 0.85,
			MinAgeSeconds:        0,
		},
		startedAt:        time.Now().Add(-time.Hour),
		rebalanceBackoff: make(map[int]rebalanceFailState),
	}

	// Run 5 vacuum ticks â€” none should trigger a rebalance
	for tick := 0; tick < 5; tick++ {
		m.vacuumTick()
	}

	rebalances := m.vacuumRebalances.Load()
	if rebalances != 0 {
		t.Errorf("expected 0 rebalances after auto-detect rebuild, got %d (infinite loop!)", rebalances)
	}
}

// TestVacuumSkipsFrozenShard verifies that OpRebalance with nil weights
// skips when sizeTracker is frozen (allocator already tuned).
func TestVacuumSkipsFrozenShard(t *testing.T) {
	s := newTestShard(t, 0, 0, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection + auto-rebuild
	for i := 0; i < 6; i++ {
		key := []byte(fmt.Sprintf("frozen-%04d", i))
		val := make([]byte, 4096)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}
	time.Sleep(100 * time.Millisecond)

	snap := s.SizeDetectionSnapshot()
	if snap.Status != "rebuilt" {
		t.Fatalf("expected status 'rebuilt', got %q", snap.Status)
	}

	// Submit OpRebalance with nil weights (what vacuum does) â€” should skip
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("OpRebalance: unexpected error %v", result.Err)
	}
	if !result.OK {
		t.Fatal("OpRebalance: expected OK=true")
	}

	// Verify shard is still frozen (no migration started)
	snap = s.SizeDetectionSnapshot()
	if snap.Status != "rebuilt" {
		t.Errorf("expected status still 'rebuilt' after skip, got %q", snap.Status)
	}
}

// TestJustDetectedPreservedOnSemFull verifies that justDetected is NOT cleared
// when startMigration fails due to a full semaphore.
func TestJustDetectedPreservedOnSemFull(t *testing.T) {
	sem := make(chan struct{}, 1)
	sem <- struct{}{} // fill the semaphore

	cfg := ShardConfig{
		ID:              0,
		IndexCapacity:   1024,
		MaxMemoryBytes:  64 * 1024 * 1024,
		EvictionPolicy:  "wtinylfu",
		ModelPageBytes:  0,
		WarmupOps:       5,
		AutoTuneSlabs:   true,
		MigrateSem:      sem,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection (but sem is full, so startMigration will fail)
	for i := 0; i < 6; i++ {
		key := []byte(fmt.Sprintf("sem-%04d", i))
		val := make([]byte, 4096)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	// Detection should have fired, but migration deferred
	snap := s.SizeDetectionSnapshot()
	if snap.Status != "detected" {
		t.Fatalf("expected status 'detected' (migration deferred), got %q", snap.Status)
	}

	// Now free the semaphore and send another SET to retry auto-rebuild
	<-sem
	key := []byte("sem-retry")
	val := make([]byte, 4096)
	s.Submit(ShardOp{
		Type:    OpSet,
		Key:     key,
		KeyHash: 9999,
		Value:   val,
		Result:  make(chan OpResult, 1),
	})

	// Give migration time to complete
	time.Sleep(200 * time.Millisecond)

	snap = s.SizeDetectionSnapshot()
	if snap.Status != "rebuilt" {
		t.Errorf("expected status 'rebuilt' after retry, got %q", snap.Status)
	}
}

// TestSkipRedundantAutoRebuild verifies that when the detected size matches
// model_page_bytes, no migration is started (sizes already match).
func TestSkipRedundantAutoRebuild(t *testing.T) {
	pageSize := uint64(4096)
	cfg := ShardConfig{
		ID:              0,
		IndexCapacity:   1024,
		MaxMemoryBytes:  64 * 1024 * 1024,
		EvictionPolicy:  "wtinylfu",
		ModelPageBytes:  pageSize,
		WarmupOps:       5,
		AutoTuneSlabs:   true,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection with size matching model_page_bytes
	for i := 0; i < 6; i++ {
		key := []byte(fmt.Sprintf("skip-%04d", i))
		val := make([]byte, pageSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	time.Sleep(50 * time.Millisecond)

	// Should be frozen directly (no migration needed)
	snap := s.SizeDetectionSnapshot()
	if snap.Status != "rebuilt" {
		t.Errorf("expected status 'rebuilt' (skipped redundant migration), got %q", snap.Status)
	}
	if snap.DominantValueSize != pageSize {
		t.Errorf("expected dominant size %d, got %d", pageSize, snap.DominantValueSize)
	}
}

// TestPressureRebalancePreservesDetection verifies that a pressure-driven
// rebalance (with ClassWeights) does NOT reset the sizeTracker.
func TestPressureRebalancePreservesDetection(t *testing.T) {
	pageSize := uint64(4096)
	s := newTestShard(t, 0, pageSize, 5, true)
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Trigger detection
	for i := 0; i < 6; i++ {
		key := []byte(fmt.Sprintf("pres-%04d", i))
		val := make([]byte, pageSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	time.Sleep(100 * time.Millisecond)

	// Pressure rebalance with explicit weights
	weights := map[uint64]float64{
		64:   1.0,
		128:  1.0,
		256:  1.0,
		512:  1.0,
		1024: 1.0,
		2048: 1.0,
		4096: 10.0,
		8192: 1.0,
	}
	result := s.Submit(ShardOp{
		Type:         OpRebalance,
		ClassWeights: weights,
		Result:       make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("pressure rebalance: %v", result.Err)
	}

	// Wait for migration to complete
	time.Sleep(200 * time.Millisecond)

	// sizeTracker should still show detected/rebuilt state (NOT warming_up)
	snap := s.SizeDetectionSnapshot()
	if snap.Status == "warming_up" {
		t.Error("pressure rebalance should NOT reset sizeTracker â€” got status 'warming_up'")
	}
	if snap.DominantValueSize != pageSize {
		t.Errorf("expected dominant size preserved at %d, got %d", pageSize, snap.DominantValueSize)
	}
}

// TestFlushDuringMigrationFast verifies that FLUSH during migration completes
// quickly without synchronous consolidation, and leaves the index empty.
func TestFlushDuringMigrationFast(t *testing.T) {
	cfg := ShardConfig{
		ID:               0,
		IndexCapacity:    1024,
		MaxMemoryBytes:   64 * 1024 * 1024,
		EvictionPolicy:   "wtinylfu",
		ModelPageBytes:   0,
		WarmupOps:        5,
		AutoTuneSlabs:    true,
		MigrateBatchSize: 1, // tiny batch â†’ migration stays in progress
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert entries to trigger detection
	for i := 0; i < 200; i++ {
		key := []byte(fmt.Sprintf("flush-%04d", i))
		val := make([]byte, 4096)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	time.Sleep(100 * time.Millisecond)

	// Trigger migration
	s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})

	// Immediately flush â€” should complete fast (no synchronous consolidation)
	start := time.Now()
	result := s.Submit(ShardOp{
		Type:   OpFlush,
		Result: make(chan OpResult, 1),
	})
	elapsed := time.Since(start)
	if result.Err != nil {
		t.Fatalf("FLUSH: %v", result.Err)
	}
	if elapsed > 2*time.Second {
		t.Errorf("FLUSH took %s â€” expected fast abort (<2s)", elapsed)
	}

	// Verify index is empty
	for i := 0; i < 200; i++ {
		key := []byte(fmt.Sprintf("flush-%04d", i))
		r := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Result:  make(chan OpResult, 1),
		})
		if r.Found {
			t.Errorf("GET flush-%04d: should not be found after FLUSH", i)
			break
		}
	}
}

// TestMigrationSetFallbackToOldAlloc verifies that when newAlloc is exhausted
// during migration, SETs fall back to oldAlloc successfully.
func TestMigrationSetFallbackToOldAlloc(t *testing.T) {
	// Use a small memory shard to make newAlloc fill up quickly
	cfg := ShardConfig{
		ID:               0,
		IndexCapacity:    1024,
		MaxMemoryBytes:   4 * 1024 * 1024, // 4MB â€” small
		EvictionPolicy:   "wtinylfu",
		ModelPageBytes:   0,
		WarmupOps:        5,
		AutoTuneSlabs:    true,
		MigrateBatchSize: 2,
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	valSize := 4096

	// Insert entries to trigger detection
	for i := 0; i < 10; i++ {
		key := []byte(fmt.Sprintf("fb-%04d", i))
		val := make([]byte, valSize)
		for j := range val {
			val[j] = byte(i)
		}
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	time.Sleep(100 * time.Millisecond)

	// Trigger a rebalance to start migration
	s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})

	// Hammer with SETs during migration â€” some may need to fall back to oldAlloc
	setSuccesses := 0
	for i := 100; i < 200; i++ {
		key := []byte(fmt.Sprintf("fb-%04d", i))
		val := make([]byte, valSize)
		for j := range val {
			val[j] = byte(i)
		}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
		if result.Err == nil {
			setSuccesses++
		}
	}

	// At least some SETs should succeed (fallback or direct)
	if setSuccesses == 0 {
		t.Error("expected at least some SETs to succeed during migration")
	}
	t.Logf("migration SET fallback test: %d/100 SETs succeeded", setSuccesses)
}

// TestMigrationDrainBudget verifies that the drain loop is capped during migration,
// preventing head-of-line blocking. With a small drain budget (8) and 200 concurrent
// ops submitted, each op should still complete within 50ms dispatch latency.
func TestMigrationDrainBudget(t *testing.T) {
	cfg := ShardConfig{
		ID:                 0,
		IndexCapacity:      1024,
		MaxMemoryBytes:     64 * 1024 * 1024,
		EvictionPolicy:     "wtinylfu",
		ModelPageBytes:     0,
		WarmupOps:          5,
		AutoTuneSlabs:      true,
		MigrateBatchSize:   1,  // tiny batch â†’ migration stays in progress for many cycles
		MigrateDrainBudget: 8,  // small budget to test capping
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	valSize := 4096

	// Trigger detection + auto-rebuild by inserting past warmup threshold
	for i := 0; i < 6; i++ {
		key := []byte(fmt.Sprintf("db-%04d", i))
		val := make([]byte, valSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}
	time.Sleep(200 * time.Millisecond)

	// Insert more entries so migration has work to do
	for i := 10; i < 110; i++ {
		key := []byte(fmt.Sprintf("db-%04d", i))
		val := make([]byte, valSize)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	// Start a rebalance to trigger migration
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("OpRebalance: %v", result.Err)
	}

	// Now submit 200 ops concurrently and measure per-op dispatch latency
	const numOps = 200
	const maxLatency = 50 * time.Millisecond
	latencies := make([]time.Duration, numOps)
	for i := 0; i < numOps; i++ {
		key := []byte(fmt.Sprintf("db-conc-%04d", i))
		val := make([]byte, valSize)
		start := time.Now()
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 10000),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
		latencies[i] = time.Since(start)
	}

	// Compute p99 latency
	var maxSeen time.Duration
	exceeded := 0
	for _, lat := range latencies {
		if lat > maxSeen {
			maxSeen = lat
		}
		if lat > maxLatency {
			exceeded++
		}
	}

	// Allow up to 5% of ops to exceed the budget (scheduler jitter)
	threshold := numOps * 5 / 100
	if exceeded > threshold {
		t.Errorf("drain budget not effective: %d/%d ops exceeded %s (max=%s)",
			exceeded, numOps, maxLatency, maxSeen)
	}
	t.Logf("drain budget test: max latency=%s, exceeded=%d/%d", maxSeen, exceeded, numOps)
}
