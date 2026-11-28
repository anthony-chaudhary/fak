package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

func spillDigest(n uint64) string { return fmt.Sprintf("%064x", n) }

// spillBlockedStore delays Put behind a gate channel and delegates reads/writes to inner
// so copy-before-release and full-staging-permit skip tests remain deterministic without sleeps.
type spillBlockedStore struct {
	inner   *l3kv.DiskStore
	gate    chan struct{}
	entered chan struct{}
	once    sync.Once
	failErr error
}

func (s *spillBlockedStore) Put(ctx context.Context, key string, payload []byte) error {
	s.once.Do(func() {
		if s.entered != nil {
			close(s.entered)
		}
	})
	if s.gate != nil {
		select {
		case <-s.gate:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	if s.failErr != nil {
		return s.failErr
	}
	return s.inner.Put(ctx, key, payload)
}

func (s *spillBlockedStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return s.inner.Get(ctx, key)
}

// spillCutoffStore simulates an interrupted mid-write cut-off while touching a real temp
// file in dir and removing it on failure, proving no truncated target or temp survives.
type spillCutoffStore struct {
	dir   string
	inner *l3kv.DiskStore
}

func (s *spillCutoffStore) Put(ctx context.Context, key string, payload []byte) error {
	tmp, err := os.CreateTemp(s.dir, ".l3kv-cutoff-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	half := len(payload) / 2
	if half == 0 && len(payload) > 0 {
		half = len(payload)
	}
	_, _ = tmp.Write(payload[:half])
	_ = tmp.Close()
	_ = os.Remove(name)
	return io.ErrUnexpectedEOF
}

func (s *spillCutoffStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return s.inner.Get(ctx, key)
}

// fak-test:runtime fast est=1s lane=default
func TestSpillCopyBeforeReleaseAndSlotGate(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("enable-spill-live-wiring-and-flag-gating", func(t *testing.T) {
		const gib = int64(1) << 30
		tempRoot := t.TempDir()

		// Checkpoint slots without an active spill budget (explicit 0 or unknown disk) must refuse.
		refusedZero := resolveInKernelRadixLocalSSD(
			ssdEnv(map[string]string{
				l3kv.EnvSpec:                            tempRoot,
				inKernelRadixLocalSSDBytesEnv:           "0",
				inKernelRadixLocalSSDCheckpointSlotsEnv: "2",
			}),
			ssdDisk(100*gib, true),
		)
		if !errors.Is(refusedZero.Err, ErrInKernelRadixLocalSSDConfig) || refusedZero.SpillEnabled || refusedZero.Bytes != 0 {
			t.Fatalf("slots with 0 spill budget = %+v, want refused ErrInKernelRadixLocalSSDConfig", refusedZero)
		}
		refusedUnknown := resolveInKernelRadixLocalSSD(
			ssdEnv(map[string]string{
				l3kv.EnvSpec:                            tempRoot,
				inKernelRadixLocalSSDCheckpointSlotsEnv: "2",
			}),
			ssdDisk(0, false),
		)
		if !errors.Is(refusedUnknown.Err, ErrInKernelRadixLocalSSDConfig) || refusedUnknown.SpillEnabled {
			t.Fatalf("slots with unknown disk = %+v, want refused ErrInKernelRadixLocalSSDConfig", refusedUnknown)
		}

		// Disabled constructor path leaves spill disabled and refuses positive checkpoint slots.
		disk, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		offStore := newLocalSSDSnapshotStore(disk, 0)
		if offStore.SpillEnabled() {
			t.Fatal("budget=0 enabled spill, want disabled")
		}
		if err := offStore.Spill(ctx, spillDigest(1), []byte("ignored")); !errors.Is(err, l3kv.ErrSpillDisabled) {
			t.Fatalf("disabled Spill = %v, want ErrSpillDisabled", err)
		}
		slotNoDisk := newLocalSSDSnapshotStoreWithConfig(nil, 0, localSSDSpillConfig{CheckpointSlots: 2})
		if slotNoDisk.err == nil || !strings.Contains(slotNoDisk.err.Error(), "checkpoint slots") {
			t.Fatalf("nil-store slots error = %v, want refusal naming checkpoint slots", slotNoDisk.err)
		}

		m := model.NewSynthetic(tinyHybridCfg())
		// CPU-only planners (backend == nil) via NewInKernelPlannerWithConfig keep SSD spill disabled.
		pCPU := NewInKernelPlannerWithConfig(m, nil, "spill-cpu-only", false, nil, false, InKernelPlannerConfig{})
		if got := pCPU.RadixLocalSSD(); got.SpillEnabled || got.Bytes != 0 || got.store != nil {
			t.Fatalf("CPU-only RadixLocalSSD = %+v, want disabled", got)
		}

		// Live planner wiring via armRadixLocalSSDWith arms EnableSpill on the planner's tree.
		be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
		newWiredPlanner := func(modelID string) *InKernelPlanner {
			p := &InKernelPlanner{
				m:                 m,
				modelID:           modelID,
				backend:           be,
				tree:              radixkv.NewWithTierBudgetsAndEvictionPolicy(0, 0, 1<<20, radixkv.EvictionLRU),
				radixOffLockAdmit: true,
			}
			p.scopedTree = radixkv.WrapScopedWithLocker(p.tree, &p.mu)
			return p
		}

		liveRoot := t.TempDir()
		p := newWiredPlanner("spill-wiring-live")
		p.armRadixLocalSSDWith(m, be, "spill-wiring-live",
			ssdEnv(map[string]string{
				l3kv.EnvSpec:                  liveRoot,
				inKernelRadixLocalSSDBytesEnv: "1048576",
			}),
			ssdDisk(100*gib, true),
		)
		t.Cleanup(func() { _, _ = p.FlushRadixLocalSSD(time.Second) })

		resolved := p.RadixLocalSSD()
		if !resolved.SpillEnabled || resolved.Bytes != 1048576 || resolved.store == nil {
			t.Fatalf("planner RadixLocalSSD = %+v, want SpillEnabled=true Bytes=1048576", resolved)
		}
		if err := resolved.store.Spill(ctx, spillDigest(2), []byte("live-wired-state")); err != nil {
			t.Fatalf("live wired Spill = %v, want nil", err)
		}
		if st := p.RadixLocalSSDSpillStats(); st.Spilled != 1 || st.DiskBytes != int64(len("live-wired-state")) {
			t.Fatalf("RadixLocalSSDSpillStats = %+v, want Spilled=1", st)
		}

		// armRadixLocalSSDWith wires checkpoint slots cleanly when spill is active.
		slotRoot := t.TempDir()
		p2 := newWiredPlanner("spill-slots-live")
		p2.armRadixLocalSSDWith(m, be, "spill-slots-live",
			ssdEnv(map[string]string{
				l3kv.EnvSpec:                            slotRoot,
				inKernelRadixLocalSSDBytesEnv:           "1048576",
				inKernelRadixLocalSSDCheckpointSlotsEnv: "2",
			}),
			ssdDisk(100*gib, true),
		)
		t.Cleanup(func() { _, _ = p2.FlushRadixLocalSSD(time.Second) })
		if got := p2.RadixLocalSSD(); !got.SpillEnabled || got.CheckpointSlots != 2 || got.store.CheckpointSlots() != 2 {
			t.Fatalf("armed slot config = %+v, want SpillEnabled=true CheckpointSlots=2", got)
		}

		// Explicit 0 disables armRadixLocalSSD and leaves spill off.
		pOff := newWiredPlanner("spill-off")
		pOff.armRadixLocalSSDWith(m, be, "spill-off",
			ssdEnv(map[string]string{
				l3kv.EnvSpec:                  t.TempDir(),
				inKernelRadixLocalSSDBytesEnv: "0",
			}),
			ssdDisk(100*gib, true),
		)
		if got := pOff.RadixLocalSSD(); got.SpillEnabled || got.Bytes != 0 || got.store != nil {
			t.Fatalf("off-arm RadixLocalSSD = %+v, want disabled", got)
		}
		if pub, skip := pOff.FlushRadixLocalSSD(time.Second); pub != 0 || skip != 0 {
			t.Fatalf("off-arm Flush = (%d,%d), want (0,0)", pub, skip)
		}
	})

	t.Run("copy-before-release-semantics", func(t *testing.T) {
		disk, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		gate := make(chan struct{})
		entered := make(chan struct{})
		blocked := &spillBlockedStore{inner: disk, gate: gate, entered: entered}
		cfg := defaultLocalSSDSpillConfig(1 << 20)
		cfg.CheckpointSlots = 1
		store := newLocalSSDSnapshotStoreWithStore(blocked, disk, 1<<20, cfg)

		want := []byte("prompt-state-copy-before-release-payload")
		ramBuf := append([]byte(nil), want...)
		var released atomic.Bool
		k1 := spillDigest(10)
		if _, err := store.RetainRAMState(ctx, k1, ramBuf, func() {
			released.Store(true)
			for i := range ramBuf {
				ramBuf[i] = 0xFF
			}
		}); err != nil {
			t.Fatalf("RetainRAMState: %v", err)
		}

		spilled, skipped, err := store.EvictRAMState(ctx, k1)
		if err != nil || !spilled || skipped {
			t.Fatalf("EvictRAMState = (spilled=%v skipped=%v err=%v), want (true,false,nil)", spilled, skipped, err)
		}
		if !released.Load() || store.HasInRAM(k1) {
			t.Fatalf("RAM not released before disk completion: released=%v inRAM=%v", released.Load(), store.HasInRAM(k1))
		}
		// Original caller buffer is already poisoned while disk write has not completed.
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("background writer did not pick up staged spill")
		}
		if _, found, err := disk.Get(ctx, k1); err != nil || found {
			t.Fatalf("disk already holds k1 before gate opened: found=%v err=%v", found, err)
		}

		close(gate)
		pub, skip := store.Flush(5 * time.Second)
		if pub != 1 || skip != 0 {
			t.Fatalf("Flush = (%d,%d), want (1,0)", pub, skip)
		}
		got, found, err := store.Get(ctx, k1)
		if err != nil || !found || !bytes.Equal(got, want) {
			t.Fatalf("restored = (%q, %v, %v), want exact original %q", got, found, err, want)
		}
	})

	t.Run("checkpoint-slots-bound-and-lru-eviction", func(t *testing.T) {
		disk, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		cfg := defaultLocalSSDSpillConfig(1 << 20)
		cfg.CheckpointSlots = 2
		store := newLocalSSDSnapshotStoreWithConfig(disk, 1<<20, cfg)
		t.Cleanup(func() { _ = store.Close() })

		k1, k2, k3 := spillDigest(21), spillDigest(22), spillDigest(23)
		p1, p2, p3 := []byte("state-1"), []byte("state-2"), []byte("state-3")
		var r1, r2, r3 atomic.Bool

		if _, err := store.RetainRAMState(ctx, k1, p1, func() { r1.Store(true) }); err != nil {
			t.Fatal(err)
		}
		if _, err := store.RetainRAMState(ctx, k2, p2, func() { r2.Store(true) }); err != nil {
			t.Fatal(err)
		}
		if store.RAMCount() != 2 || r1.Load() || r2.Load() {
			t.Fatalf("2-slot RAM count=%d r1=%v r2=%v, want 2/false/false", store.RAMCount(), r1.Load(), r2.Load())
		}
		// Refresh k1 so k2 becomes oldest and gets evicted when k3 arrives.
		if _, err := store.RetainRAMState(ctx, k1, p1, func() { r1.Store(true) }); err != nil {
			t.Fatal(err)
		}
		evicted, err := store.RetainRAMState(ctx, k3, p3, func() { r3.Store(true) })
		if err != nil {
			t.Fatal(err)
		}
		if len(evicted) != 1 || evicted[0] != k2 {
			t.Fatalf("evicted = %v, want [%s]", evicted, k2)
		}
		if store.RAMCount() != 2 || !r2.Load() || store.HasInRAM(k2) || !store.HasInRAM(k1) || !store.HasInRAM(k3) {
			t.Fatalf("post-evict slots count=%d r2=%v has(k1,k2,k3)=(%v,%v,%v)",
				store.RAMCount(), r2.Load(), store.HasInRAM(k1), store.HasInRAM(k2), store.HasInRAM(k3))
		}
		// Drain so k2 is published to disk, then verify k2 restores.
		pub, _ := store.Flush(5 * time.Second)
		if pub < 1 {
			t.Fatalf("Flush published=%d, want >=1 (evicted k2 + remaining)", pub)
		}
		got2, found, err := disk.Get(ctx, k2)
		if err != nil || !found || !bytes.Equal(got2, p2) {
			t.Fatalf("evicted k2 on disk = (%q, %v, %v), want %q", got2, found, err, p2)
		}
	})

	t.Run("skip-on-full-staging-permit-and-disk-budget-exhaustion", func(t *testing.T) {
		// 1. Full staging permit skips non-blockingly.
		disk, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		gate := make(chan struct{})
		entered := make(chan struct{})
		blocked := &spillBlockedStore{inner: disk, gate: gate, entered: entered}
		cfg := localSSDSpillConfig{
			MaxDiskBytes:   1 << 20,
			MaxEntryBytes:  1 << 20,
			PutTimeout:     5 * time.Second,
			StopBudget:     5 * time.Second,
			StagingPermit:  64,
			MaxStagingJobs: 2,
			PruneOnPut:     true,
		}
		store := newLocalSSDSnapshotStoreWithStore(blocked, disk, 1<<20, cfg)

		k1, k2, k3 := spillDigest(31), spillDigest(32), spillDigest(33)
		skipped, err := store.StageSpill(ctx, k1, bytes.Repeat([]byte("a"), 50))
		if err != nil || skipped {
			t.Fatalf("StageSpill(k1) = (skipped=%v err=%v), want (false,nil)", skipped, err)
		}
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("writer did not pick up k1")
		}
		// Staging permit now holds 50/64 bytes; 30-byte k2 exceeds remaining permit -> skip immediately.
		skipped2, err := store.StageSpill(ctx, k2, bytes.Repeat([]byte("b"), 30))
		if err != nil || !skipped2 {
			t.Fatalf("StageSpill(k2 full permit) = (skipped=%v err=%v), want (true,nil)", skipped2, err)
		}
		// Direct Put also skips non-blockingly while staging permit is full.
		if err := store.Put(ctx, k3, bytes.Repeat([]byte("c"), 30)); err != nil {
			t.Fatalf("Put while staging full = %v, want nil (skip)", err)
		}
		if st := store.StagingStats(); st.SkippedPermit < 2 {
			t.Fatalf("SkippedPermit = %d, want >= 2", st.SkippedPermit)
		}
		close(gate)
		store.Flush(5 * time.Second)
		if _, found, _ := disk.Get(ctx, k2); found {
			t.Fatal("skipped k2 appeared on disk")
		}
		if _, found, _ := disk.Get(ctx, k3); found {
			t.Fatal("skipped k3 appeared on disk")
		}

		// 2. Disk budget exhaustion skips past MaxDiskBytes.
		diskCap, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		capCfg := localSSDSpillConfig{
			MaxDiskBytes:  150,
			MaxEntryBytes: 150,
			PutTimeout:    5 * time.Second,
			StopBudget:    5 * time.Second,
		}
		capStore := newLocalSSDSnapshotStoreWithConfig(diskCap, 150, capCfg)
		defer capStore.Close()

		b70 := bytes.Repeat([]byte("x"), 70)
		if err := capStore.Spill(ctx, spillDigest(41), b70); err != nil {
			t.Fatalf("first Spill: %v", err)
		}
		if err := capStore.Spill(ctx, spillDigest(42), b70); err != nil {
			t.Fatalf("second Spill: %v", err)
		}
		if err := capStore.Spill(ctx, spillDigest(43), b70); err != nil {
			t.Fatalf("third Spill over cap: %v, want nil (skip)", err)
		}
		if st := capStore.SpillStats(); st.Spilled != 2 || st.SkippedBudget != 1 || st.DiskBytes != 140 {
			t.Fatalf("cap SpillStats = %+v, want Spilled=2 SkippedBudget=1 DiskBytes=140", st)
		}
		if _, found, _ := diskCap.Get(ctx, spillDigest(43)); found {
			t.Fatal("over-budget spill wrote to disk")
		}
	})

	t.Run("planned-stop-flush-within-disk-cap-and-20s-budget", func(t *testing.T) {
		if got := clampLocalSSDStopBudget(60 * time.Second); got != 20*time.Second {
			t.Fatalf("clampLocalSSDStopBudget(60s) = %v, want 20s cap", got)
		}
		if got := clampLocalSSDStopBudget(0); got != 20*time.Second {
			t.Fatalf("clampLocalSSDStopBudget(0) = %v, want 20s default", got)
		}

		dir := t.TempDir()
		disk, err := l3kv.NewDiskStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		cfg := localSSDSpillConfig{
			MaxDiskBytes:    130,
			MaxEntryBytes:   130,
			PutTimeout:      5 * time.Second,
			StopBudget:      20 * time.Second,
			CheckpointSlots: 4,
		}
		store := newLocalSSDSnapshotStoreWithConfig(disk, 130, cfg)

		k1, k2, k3 := spillDigest(51), spillDigest(52), spillDigest(53)
		p1, p2, p3 := bytes.Repeat([]byte("1"), 60), bytes.Repeat([]byte("2"), 60), bytes.Repeat([]byte("3"), 60)
		var relCount atomic.Int32
		for _, tc := range []struct {
			k string
			p []byte
		}{{k1, p1}, {k2, p2}, {k3, p3}} {
			if _, err := store.RetainRAMState(ctx, tc.k, tc.p, func() { relCount.Add(1) }); err != nil {
				t.Fatal(err)
			}
		}

		published, skipped := store.Flush(30 * time.Second)
		if published != 2 || skipped != 1 || relCount.Load() != 3 || store.RAMCount() != 0 {
			t.Fatalf("Flush = (pub=%d skip=%d rel=%d ram=%d), want (2,1,3,0)",
				published, skipped, relCount.Load(), store.RAMCount())
		}
		if st := store.SpillStats(); st.DiskBytes != 120 || st.SkippedBudget != 1 {
			t.Fatalf("SpillStats = %+v, want DiskBytes=120 SkippedBudget=1", st)
		}
		// Reopen fresh DiskStore to confirm restart survival under the disk cap.
		reopened, err := l3kv.NewDiskStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		if g1, ok, err := reopened.Get(ctx, k1); err != nil || !ok || !bytes.Equal(g1, p1) {
			t.Fatalf("reopened k1 = (%v, %v), want match", ok, err)
		}
		if g2, ok, err := reopened.Get(ctx, k2); err != nil || !ok || !bytes.Equal(g2, p2) {
			t.Fatalf("reopened k2 = (%v, %v), want match", ok, err)
		}
		if _, ok, _ := reopened.Get(ctx, k3); ok {
			t.Fatal("over-cap k3 survived planned stop, want skipped")
		}

		// Tight stop deadline aborts queued items and returns promptly without hanging.
		disk2, err := l3kv.NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		gate := make(chan struct{})
		entered := make(chan struct{})
		blocked := &spillBlockedStore{inner: disk2, gate: gate, entered: entered}
		bCfg := localSSDSpillConfig{MaxDiskBytes: 1 << 20, MaxEntryBytes: 1 << 20, PutTimeout: 5 * time.Second, StopBudget: 5 * time.Second}
		bStore := newLocalSSDSnapshotStoreWithStore(blocked, disk2, 1<<20, bCfg)
		defer close(gate)

		item1, _, err := bStore.stageSpillItem(ctx, spillDigest(54), []byte("inflight"))
		if err != nil {
			t.Fatal(err)
		}
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("writer never picked up inflight item")
		}
		item2, _, err := bStore.stageSpillItem(ctx, spillDigest(55), []byte("queued"))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := bStore.RetainRAMState(ctx, spillDigest(56), []byte("in-ram"), nil); err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		pub2, skip2 := bStore.Flush(40 * time.Millisecond)
		elapsed := time.Since(start)
		if elapsed > 2*time.Second || pub2 != 0 || skip2 < 2 {
			t.Fatalf("bounded Flush elapsed=%v pub=%d skip=%d, want prompt abort with >=2 skipped", elapsed, pub2, skip2)
		}
		select {
		case err := <-item2.done:
			if !errors.Is(err, errLocalSSDSpillCancelled) {
				t.Fatalf("queued item2 done = %v, want errLocalSSDSpillCancelled", err)
			}
		case <-time.After(time.Second):
			t.Fatal("queued item2 never cancelled on stop deadline")
		}
		_ = item1
	})

	t.Run("damaged-file-removal-and-token-prefill-fallback", func(t *testing.T) {
		dir := t.TempDir()
		disk, err := l3kv.NewDiskStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		store := newLocalSSDSnapshotStore(disk, 1<<20)
		t.Cleanup(func() { _ = store.Close() })

		kHash := spillDigest(61)
		kTrunc := spillDigest(62)
		kMagic := spillDigest(63)
		kAsync := spillDigest(64)
		kHealthy := spillDigest(65)
		for _, k := range []string{kHash, kTrunc, kMagic, kAsync, kHealthy} {
			if err := store.Put(ctx, k, []byte("valid-snapshot-envelope-bytes")); err != nil {
				t.Fatalf("Put(%s): %v", k, err)
			}
		}

		// Corrupt kHash (flip last byte), kTrunc (truncate below header), kMagic (bad magic), kAsync (bad magic).
		rawHash, err := os.ReadFile(filepath.Join(dir, kHash))
		if err != nil {
			t.Fatal(err)
		}
		rawHash[len(rawHash)-1] ^= 0x5A
		if err := os.WriteFile(filepath.Join(dir, kHash), rawHash, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, kTrunc), rawHash[:10], 0o600); err != nil {
			t.Fatal(err)
		}
		badMagic := append([]byte(nil), rawHash...)
		copy(badMagic[:4], "NOPE")
		if err := os.WriteFile(filepath.Join(dir, kMagic), badMagic, 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, kAsync), badMagic, 0o600); err != nil {
			t.Fatal(err)
		}

		for _, k := range []string{kHash, kTrunc, kMagic} {
			got, found, err := store.Get(ctx, k)
			if err != nil || found || got != nil {
				t.Fatalf("Get damaged %s = (%v, %v, %v), want clean miss (nil,false,nil)", k, got, found, err)
			}
			if _, statErr := os.Stat(filepath.Join(dir, k)); !os.IsNotExist(statErr) {
				t.Fatalf("damaged file %s still on disk (stat err=%v), want deleted", k, statErr)
			}
		}

		asyncDone := make(chan error, 1)
		var asyncFound bool
		store.GetAsync(ctx, kAsync, func(b []byte, found bool, err error) {
			asyncFound = found
			asyncDone <- err
		})
		if err := <-asyncDone; err != nil || asyncFound {
			t.Fatalf("GetAsync damaged = (found=%v err=%v), want (false,nil)", asyncFound, err)
		}
		if _, statErr := os.Stat(filepath.Join(dir, kAsync)); !os.IsNotExist(statErr) {
			t.Fatalf("damaged async file %s still on disk (stat err=%v), want deleted", kAsync, statErr)
		}

		// Cancelled Get must NOT delete healthy records.
		cancelCtx, cancel := context.WithCancel(ctx)
		cancel()
		if _, _, err := store.Get(cancelCtx, kHealthy); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Get = %v, want context.Canceled", err)
		}
		if _, found, err := store.Get(ctx, kHealthy); err != nil || !found {
			t.Fatalf("healthy record missing after cancelled Get: found=%v err=%v", found, err)
		}

		// End-to-end InKernelPlanner + radixkv fallback: damaged SSD record is deleted and request prefills tokens.
		const gib = int64(1) << 30
		cfg := tinyHybridCfg()
		be := &countingBackend{Backend: compute.Default(), deviceMemory: true}
		newTestPlanner := func(modelID string, l2Budget int64, ssdRoot string) *InKernelPlanner {
			m := model.NewSynthetic(cfg)
			pl := &InKernelPlanner{
				m:                 m,
				modelID:           modelID,
				backend:           be,
				tree:              radixkv.NewWithTierBudgetsAndEvictionPolicy(0, 0, l2Budget, radixkv.EvictionLRU),
				radixOffLockAdmit: true,
			}
			pl.scopedTree = radixkv.WrapScopedWithLocker(pl.tree, &pl.mu)
			pl.pagedPrefix = inKernelPagedPrefixLedgerFor(m, be, pl.tree)
			pl.armRadixLocalSSDWith(m, be, modelID, ssdEnv(map[string]string{
				l3kv.EnvSpec:                  ssdRoot,
				inKernelRadixLocalSSDBytesEnv: "1048576",
			}), ssdDisk(100*gib, true))
			return pl
		}
		p := newTestPlanner("damaged-prefill-e2e", 1<<20, t.TempDir())
		t.Cleanup(func() { _, _ = p.FlushRadixLocalSSD(time.Second) })

		idsA := []int{3, 7, 11, 13}
		idsB := []int{5, 9, 15, 17}
		runReq := func(planner *InKernelPlanner, ids []int) ([]int, int, radixkv.SnapshotTier) {
			t.Helper()
			var out []int
			_, _, _, matched, tier, _, _, _, err := planner.generateReusedContextWithBias(
				ctx, ids, 1, 0, 0, 0, nil, 0, 0, map[int]bool{},
				func(id int) bool { out = append(out, id); return false },
			)
			if err != nil {
				t.Fatalf("generateReusedContextWithBias(%v): %v", ids, err)
			}
			return out, matched, tier
		}
		coldTokens, _, _ := runReq(p, idsA)

		// Stage A to L2, drop L1 hot copy, measure exact A size, then stage B under exact L2 budget to evict A -> SSD.
		_, cands := p.KVPrefixPressuredCandidates()
		if len(cands) != 1 {
			t.Fatalf("candidates = %+v, want 1", cands)
		}
		if got := p.StageKVPrefixToHost(ctx, cands[0].SpanDigest); got.Outcome != radixkv.SnapshotTransferOK {
			t.Fatalf("StageKVPrefixToHost(A): %+v", got)
		}
		if p.EvictHotKVPrefix(cands[0].SpanDigest) != len(idsA) {
			t.Fatal("failed to evict hot copy of A")
		}
		p.mu.Lock()
		l2Stats := p.tree.Stats()
		p.mu.Unlock()
		if l2Stats.HostSnapshotBytes <= 0 {
			t.Fatalf("HostSnapshotBytes = %d, want >0", l2Stats.HostSnapshotBytes)
		}
		p2 := newTestPlanner("damaged-prefill-e2e-2", l2Stats.HostSnapshotBytes, t.TempDir())
		t.Cleanup(func() { _, _ = p2.FlushRadixLocalSSD(time.Second) })

		runReq(p2, idsA)
		_, candsA := p2.KVPrefixPressuredCandidates()
		if len(candsA) != 1 {
			t.Fatalf("p2 candidatesA = %+v, want 1", candsA)
		}
		if got := p2.StageKVPrefixToHost(ctx, candsA[0].SpanDigest); got.Outcome != radixkv.SnapshotTransferOK {
			t.Fatalf("p2 stage A: %+v", got)
		}
		p2.EvictHotKVPrefix(candsA[0].SpanDigest)

		runReq(p2, idsB)
		_, candsB := p2.KVPrefixPressuredCandidates()
		var digB string
		for _, c := range candsB {
			if c.Tokens == len(idsB) {
				digB = c.SpanDigest
				break
			}
		}
		if digB == "" {
			t.Fatalf("missing candidate B in %+v", candsB)
		}
		if got := p2.StageKVPrefixToHost(ctx, digB); got.Outcome != radixkv.SnapshotTransferOK {
			t.Fatalf("p2 stage B (should evict A to SSD): %+v", got)
		}
		p2.EvictHotKVPrefix(digB)

		// Corrupt the spilled SSD record on disk.
		ssdDir := p2.RadixLocalSSD().Dir
		entries, err := os.ReadDir(ssdDir)
		if err != nil {
			t.Fatal(err)
		}
		var spilledFiles []string
		for _, e := range entries {
			if !e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
				spilledFiles = append(spilledFiles, e.Name())
			}
		}
		if len(spilledFiles) != 1 {
			t.Fatalf("spilled files in %s = %v, want 1", ssdDir, spilledFiles)
		}
		p2.mu.Lock()
		preCorruptStats := p2.tree.Stats()
		p2.mu.Unlock()
		if preCorruptStats.LocalL3StageBytes <= 0 || preCorruptStats.LocalL3ReferencedBytes <= 0 {
			t.Fatalf("pre-corrupt LocalL3 stats = %+v, want positive stage/referenced bytes", preCorruptStats)
		}
		corruptPath := filepath.Join(ssdDir, spilledFiles[0])
		if err := os.WriteFile(corruptPath, []byte("corrupt-ssd-bytes"), 0o600); err != nil {
			t.Fatal(err)
		}

		// Demand request A: damaged file is deleted and request cleanly cold-prefills identical tokens.
		recoveredTokens, matched, tier := runReq(p2, idsA)
		if matched != 0 || tier != radixkv.SnapshotTierMiss {
			t.Fatalf("damaged SSD demand matched=%d tier=%q, want 0/miss fallback to prefill", matched, tier)
		}
		if !eqInts(recoveredTokens, coldTokens) {
			t.Fatalf("prefilled tokens after damaged SSD = %v, want cold %v", recoveredTokens, coldTokens)
		}
		if _, statErr := os.Stat(corruptPath); !os.IsNotExist(statErr) {
			t.Fatalf("damaged SSD file %s survived lookup (stat err=%v), want deleted", corruptPath, statErr)
		}
		p2.mu.Lock()
		postCorruptStats := p2.tree.Stats()
		p2.mu.Unlock()
		if postCorruptStats.LocalL3ReferencedBytes != 0 {
			t.Fatalf("post-corrupt LocalL3ReferencedBytes = %d, want 0", postCorruptStats.LocalL3ReferencedBytes)
		}
	})

	t.Run("atomic-cutoff-writes-leave-no-artifacts", func(t *testing.T) {
		dir := t.TempDir()
		disk, err := l3kv.NewDiskStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		cutoff := &spillCutoffStore{dir: dir, inner: disk}
		store := newLocalSSDSnapshotStoreWithStore(cutoff, disk, 1<<20, defaultLocalSSDSpillConfig(1<<20))
		defer store.Close()

		kCut := spillDigest(71)
		if err := store.Spill(ctx, kCut, []byte("will-be-cut-off-mid-write")); !errors.Is(err, io.ErrUnexpectedEOF) {
			t.Fatalf("cutoff Spill = %v, want io.ErrUnexpectedEOF", err)
		}
		if _, found, err := store.Get(ctx, kCut); err != nil || found {
			t.Fatalf("cutoff Get = (found=%v err=%v), want clean miss", found, err)
		}
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 0 {
			t.Fatalf("cutoff left %d artifact(s) in dir, want 0", len(entries))
		}

		// Cancelled Put/Spill leaves existing intact records untouched and writes no truncated artifact.
		realStore := newLocalSSDSnapshotStore(disk, 1<<20)
		defer realStore.Close()
		kExist := spillDigest(72)
		orig := []byte("intact-existing-record")
		if err := realStore.Put(ctx, kExist, orig); err != nil {
			t.Fatal(err)
		}
		canceledCtx, cancel := context.WithCancel(ctx)
		cancel()
		kNever := spillDigest(73)
		if err := realStore.Put(canceledCtx, kNever, []byte("never-written")); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Put = %v, want context.Canceled", err)
		}
		if err := realStore.Put(canceledCtx, kExist, []byte("truncated-overwrite")); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled overwrite = %v, want context.Canceled", err)
		}
		if _, found, _ := realStore.Get(ctx, kNever); found {
			t.Fatal("cancelled Put left record on disk")
		}
		if got, found, err := realStore.Get(ctx, kExist); err != nil || !found || !bytes.Equal(got, orig) {
			t.Fatalf("cancelled overwrite corrupted existing record: (%q, %v, %v)", got, found, err)
		}

		// Stale temp cleanup removes orphan temp files while keeping valid records intact.
		for _, tmpName := range []string{".l3kv-orphan-1", ".fakssd-orphan-2"} {
			p := filepath.Join(dir, tmpName)
			if err := os.WriteFile(p, []byte("partial"), 0o600); err != nil {
				t.Fatal(err)
			}
			old := time.Now().Add(-48 * time.Hour)
			if err := os.Chtimes(p, old, old); err != nil {
				t.Fatal(err)
			}
		}
		if removed := cleanupTempFiles(dir, inKernelRadixLocalSSDStaleTempAge); removed != 2 {
			t.Fatalf("cleanupTempFiles removed=%d, want 2", removed)
		}
		if got, found, err := realStore.Get(ctx, kExist); err != nil || !found || !bytes.Equal(got, orig) {
			t.Fatalf("cleanup touched valid record: (%q, %v, %v)", got, found, err)
		}
	})
}
