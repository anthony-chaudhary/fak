package shard

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/alloc"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
)

// mockPreRegisterer implements both AllocChangeListener and AllocPreRegisterer
// to verify the pre-registration flow without real RDMA hardware.
type mockPreRegisterer struct {
	mu sync.Mutex

	// PreRegisterAllocator tracking
	preRegCalled  atomic.Int32
	preRegDelay   time.Duration // artificial delay to simulate ibv_reg_mr
	preRegShardID int
	preRegAlloc   alloc.Allocator

	// OnAllocatorChanged tracking
	onChangedCalled atomic.Int32
	usedPreReg      atomic.Bool // true if staged data was available

	// DiscardPreRegistered tracking
	discardCalled atomic.Int32

	// Staged state (simulates stagedMRs map)
	staged sync.Map // shardID -> bool
}

func (m *mockPreRegisterer) PreRegisterAllocator(shardID int, newAlloc alloc.Allocator) <-chan struct{} {
	m.preRegCalled.Add(1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		if m.preRegDelay > 0 {
			time.Sleep(m.preRegDelay)
		}
		m.mu.Lock()
		m.preRegShardID = shardID
		m.preRegAlloc = newAlloc
		m.mu.Unlock()
		m.staged.Store(shardID, true)
	}()
	return done
}

func (m *mockPreRegisterer) DiscardPreRegistered(shardID int) {
	m.discardCalled.Add(1)
	m.staged.Delete(shardID)
}

func (m *mockPreRegisterer) OnAllocatorChanged(change AllocatorChange) {
	m.onChangedCalled.Add(1)
	// Check if pre-registered data was staged
	if _, ok := m.staged.LoadAndDelete(change.ShardID); ok {
		m.usedPreReg.Store(true)
	}
	// Simulate deferred cleanup in background (like RDMA server does)
	go func() {
		time.Sleep(10 * time.Millisecond)
		change.OldAllocator.Close()
	}()
}

// TestPreRegistrationDuringMigration verifies the full pre-registration lifecycle:
// 1. PreRegisterAllocator is called at startMigration
// 2. Pre-registration completes before finalizeMigration (overlaps with batch migration)
// 3. OnAllocatorChanged uses the pre-registered path
// 4. Shard remains responsive during finalize (ops are drained)
// 5. Migration metrics are recorded correctly
func TestPreRegistrationDuringMigration(t *testing.T) {
	s := newTestShard(t, 0, 0, 50, true)
	s.config.MigrateBatchSize = 10

	// Register mock listener that implements both interfaces
	mock := &mockPreRegisterer{
		preRegDelay: 20 * time.Millisecond, // simulate ibv_reg_mr time
	}
	s.RegisterAllocListener(mock)

	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	const numEntries = 80
	valSize := 512

	// Insert entries (enough to trigger warmup detection)
	for i := 0; i < numEntries; i++ {
		key := []byte(fmt.Sprintf("prereg-key-%04d", i))
		value := make([]byte, valSize)
		for j := range value {
			value[j] = byte(i % 256)
		}
		result := s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
		if result.Err != nil {
			t.Fatalf("SET %d: %v", i, result.Err)
		}
	}

	// Trigger rebalance (starts ZeroLatencyBalance + pre-registration)
	result := s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})
	if result.Err != nil {
		t.Fatalf("OpRebalance: %v", result.Err)
	}

	// Verify PreRegisterAllocator was called
	// (async allocator construction + commitMigration may take a few ms)
	for i := 0; i < 100; i++ {
		if mock.preRegCalled.Load() >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if got := mock.preRegCalled.Load(); got != 1 {
		t.Errorf("PreRegisterAllocator called %d times, want 1", got)
	}

	// Submit GETs during migration â€” these should succeed (shard is responsive)
	hitCount := 0
	for i := 0; i < numEntries; i++ {
		key := []byte(fmt.Sprintf("prereg-key-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if result.Found {
			hitCount++
		}
	}
	if hitCount == 0 {
		t.Error("no GETs succeeded during migration â€” shard appears blocked")
	}

	// Wait for migration to fully complete
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if mock.onChangedCalled.Load() > 0 {
			break
		}
		// Keep submitting ops to drive the shard event loop
		key := []byte("poll-key")
		s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		time.Sleep(5 * time.Millisecond)
	}

	// Verify OnAllocatorChanged was called
	if got := mock.onChangedCalled.Load(); got < 1 {
		t.Fatalf("OnAllocatorChanged called %d times, want >= 1", got)
	}

	// Verify pre-registered path was used (not sync fallback)
	if !mock.usedPreReg.Load() {
		t.Error("OnAllocatorChanged did NOT use pre-registered data â€” sync fallback was used instead")
	}

	// Verify DiscardPreRegistered was NOT called (successful migration)
	if got := mock.discardCalled.Load(); got != 0 {
		t.Errorf("DiscardPreRegistered called %d times, want 0 (migration succeeded)", got)
	}

	// Verify migration metrics
	m := s.Metrics()
	if m.MigrationActive() != 0 {
		t.Error("migrationActive should be 0 after completion")
	}
	if m.MigrationsTotal() < 1 {
		t.Errorf("migrationsTotal = %d, want >= 1", m.MigrationsTotal())
	}
	if m.MigrationDurationMs() <= 0 {
		t.Errorf("migrationDurationMs = %d, want > 0", m.MigrationDurationMs())
	}
	if m.MigrationEntries() <= 0 {
		t.Errorf("migrationEntries = %d, want > 0", m.MigrationEntries())
	}
	// preRegWaitMs should be near 0 since pre-registration (20ms) finishes well
	// before batch migration of 80 entries at batch=10 completes
	preRegWait := m.MigrationPreRegWaitMs()
	t.Logf("migration metrics: entries=%d, duration=%dms, prereg_wait=%dms, total=%d",
		m.MigrationEntries(), m.MigrationDurationMs(), preRegWait, m.MigrationsTotal())

	// Verify all data is still correct after migration
	for i := 0; i < numEntries; i++ {
		key := []byte(fmt.Sprintf("prereg-key-%04d", i))
		result := s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		if !result.Found {
			t.Errorf("GET prereg-key-%04d: not found after migration", i)
			continue
		}
		if result.Value[0] != byte(i%256) {
			t.Errorf("GET prereg-key-%04d: value[0] = %d, want %d", i, result.Value[0], byte(i%256))
		}
	}
}

// TestPreRegistrationSlowOverlap verifies that when pre-registration takes longer
// than batch migration, finalizeMigration drains ops while waiting (no blocking).
func TestPreRegistrationSlowOverlap(t *testing.T) {
	s := newTestShard(t, 0, 0, 20, true)
	s.config.MigrateBatchSize = 100 // large batch = fast migration

	// Slow pre-registration to force the wait path in finalizeMigration
	mock := &mockPreRegisterer{
		preRegDelay: 200 * time.Millisecond,
	}
	s.RegisterAllocListener(mock)

	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert entries
	for i := 0; i < 30; i++ {
		key := []byte(fmt.Sprintf("slow-key-%04d", i))
		value := make([]byte, 256)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}

	// Trigger migration
	s.Submit(ShardOp{
		Type:   OpRebalance,
		Result: make(chan OpResult, 1),
	})

	// Immediately flood ops â€” these should be drained during the pre-reg wait
	var opsSucceeded atomic.Int32
	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			key := []byte(fmt.Sprintf("slow-key-%04d", idx%30))
			result := s.Submit(ShardOp{
				Type:    OpGet,
				Key:     key,
				KeyHash: index.KeyHash(key),
				Result:  make(chan OpResult, 1),
			})
			if result.Err == nil {
				opsSucceeded.Add(1)
			}
		}(i)
	}
	wg.Wait()

	if got := opsSucceeded.Load(); got == 0 {
		t.Error("zero ops succeeded during slow pre-registration â€” shard was blocked")
	}
	t.Logf("ops succeeded during slow pre-reg: %d/20", opsSucceeded.Load())

	// Wait for everything to settle
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if mock.onChangedCalled.Load() > 0 {
			break
		}
		key := []byte("poll")
		s.Submit(ShardOp{
			Type:    OpGet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Result:  make(chan OpResult, 1),
		})
		time.Sleep(10 * time.Millisecond)
	}

	// prereg_wait should be > 0 since pre-reg was slow
	preRegWait := s.Metrics().MigrationPreRegWaitMs()
	t.Logf("prereg_wait=%dms (expected > 0 due to 200ms delay)", preRegWait)
	if preRegWait == 0 && mock.onChangedCalled.Load() > 0 {
		t.Log("NOTE: prereg_wait=0 means pre-reg finished before batch migration (unexpected with 200ms delay)")
	}
}

// TestPreRegistrationAbortDiscard verifies that aborting a migration via FLUSH
// calls DiscardPreRegistered (not OnAllocatorChanged) when pre-registration is
// still in progress.
func TestPreRegistrationAbortDiscard(t *testing.T) {
	// warmupOps=1000 prevents auto-detection (only 30 entries inserted).
	// autoTune=true required for OpRebalance to work.
	s := newTestShard(t, 0, 0, 1000, true)
	s.config.MigrateBatchSize = 1 // small batches so migration takes multiple iterations

	mock := &mockPreRegisterer{
		preRegDelay: 500 * time.Millisecond, // long delay so FLUSH interrupts during pre-reg wait
	}
	s.RegisterAllocListener(mock)

	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert entries (no auto-detection with warmup=1000)
	for i := 0; i < 30; i++ {
		key := []byte(fmt.Sprintf("abort-key-%04d", i))
		value := make([]byte, 256)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}

	// Trigger migration via pressure-driven rebalance (doesn't need detection)
	s.Submit(ShardOp{
		Type:         OpRebalance,
		ClassWeights: map[uint64]float64{256: 1.0},
		Result:       make(chan OpResult, 1),
	})

	// Wait for async allocator construction to complete and migration to start
	for i := 0; i < 100; i++ {
		if s.Metrics().MigrationActive() != 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// FLUSH aborts the in-progress migration (pre-reg is still running due to 500ms delay)
	s.Submit(ShardOp{
		Type:   OpFlush,
		Result: make(chan OpResult, 1),
	})

	// Wait for cleanup (must exceed preRegDelay so FLUSH's pre-reg wait completes)
	time.Sleep(700 * time.Millisecond)

	// Verify migration is not active
	if s.Metrics().MigrationActive() != 0 {
		t.Error("migrationActive should be 0 after FLUSH abort")
	}

	// DiscardPreRegistered should have been called
	if got := mock.discardCalled.Load(); got < 1 {
		t.Errorf("DiscardPreRegistered called %d times, want >= 1", got)
	}

	t.Logf("preRegCalled=%d, onChangedCalled=%d, discardCalled=%d",
		mock.preRegCalled.Load(), mock.onChangedCalled.Load(), mock.discardCalled.Load())
}

// TestMigrationMetricsReset verifies that migration metrics are reset on FLUSH.
func TestMigrationMetricsReset(t *testing.T) {
	s := newTestShard(t, 0, 0, 20, true)
	s.config.MigrateBatchSize = 100
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Insert, trigger migration, let it complete
	for i := 0; i < 30; i++ {
		key := []byte(fmt.Sprintf("reset-key-%04d", i))
		value := make([]byte, 256)
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: index.KeyHash(key),
			Value:   value,
			Result:  make(chan OpResult, 1),
		})
	}
	s.Submit(ShardOp{Type: OpRebalance})

	// Wait for async migration to complete (allocator construction + migration).
	// The rebalance op returns immediately; migration runs in the background.
	m := s.Metrics()
	deadline := time.Now().Add(5 * time.Second)
	for m.MigrationsTotal() < 1 && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if m.MigrationsTotal() < 1 {
		t.Fatalf("expected migrationsTotal >= 1 before flush, got %d", m.MigrationsTotal())
	}

	// FLUSH should reset migration metrics.
	// Submit is synchronous â€” blocks until the shard goroutine completes the op.
	s.Submit(ShardOp{Type: OpFlush})

	// migrationsTotal is a lifetime counter â€” NOT reset by FLUSH (by design).
	// Verify the epoch-scoped metrics are reset instead.
	if m.MigrationDurationMs() != 0 {
		t.Errorf("migrationDurationMs after FLUSH = %d, want 0", m.MigrationDurationMs())
	}
	if m.MigrationEntries() != 0 {
		t.Errorf("migrationEntries after FLUSH = %d, want 0", m.MigrationEntries())
	}
}
