package agent

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Local-SSD KV overflow tier defaults (perf-parity T-06 / umem TICKET-04): with no operator
// byte count, evicted host-L2 prefix snapshots demote to local disk up to the smaller of
// inKernelRadixLocalSSDCap and this fraction of the free bytes on the store's volume, so a
// high-RAM node keeps its cold prefixes restorable instead of paying a full cold prefill.
// The reserve is a ceiling on REFERENCED bytes, not an allocation: the tier holds nothing
// until L2 pressure actually evicts a staged image into it.
//
// Eviction-spill gating, checkpoint-slot RAM bounding, copy-before-release staging, and
// 20 s planned-stop drain are adapted from TensorFold (ashhart/TensorFold,
const (
	inKernelRadixLocalSSDBytesEnv                 = "FAK_INKERNEL_RADIX_SSD_BYTES"
	inKernelRadixLocalSSDCheckpointSlotsEnv       = "FAK_INKERNEL_RADIX_SSD_CHECKPOINT_SLOTS"
	inKernelRadixLocalSSDFractionDef              = 0.25
	inKernelRadixLocalSSDCap                int64 = 64 << 30 // 64 GiB
	// inKernelRadixLocalSSDMaxEntries bounds the record COUNT the store owner prunes to after
	// each spill, independent of the byte cap, so a pathological run of tiny snapshots cannot
	// fragment the volume past a bounded directory scan.
	inKernelRadixLocalSSDMaxEntries = 1024

	inKernelRadixLocalSSDSpillTimeout         = 20 * time.Second
	inKernelRadixLocalSSDStopBudget           = 20 * time.Second
	inKernelRadixLocalSSDDefaultStagingPermit = int64(256 << 20)
	inKernelRadixLocalSSDDefaultStagingJobs   = 8
	inKernelRadixLocalSSDStaleTempAge         = 24 * time.Hour
)

// Sources of a resolved local SSD overflow budget, printed at planner construction.
const (
	InKernelRadixLocalSSDSourceEnv     = "env"
	InKernelRadixLocalSSDSourceDefault = "default min(64 GiB, 25% free on volume)"
	InKernelRadixLocalSSDSourceUnknown = "unknown disk capacity"
	InKernelRadixLocalSSDSourceRefused = "refused"
)

// ErrInKernelRadixLocalSSDConfig marks a local SSD env value the resolver refused; the tier
// then fails closed to off (0 bytes), today's behaviour.
var ErrInKernelRadixLocalSSDConfig = errors.New("fak inkernel: invalid radix local SSD KV overflow")

var (
	errLocalSSDSpillClosed    = errors.New("fak inkernel: local SSD spill stager closed")
	errLocalSSDSpillCancelled = errors.New("fak inkernel: local SSD spill cancelled before publish")
)

// InKernelRadixLocalSSD is the resolved local-SSD KV overflow budget, its store directory,
// and where the budget came from.
type InKernelRadixLocalSSD struct {
	Bytes           int64
	Dir             string
	Source          string
	CheckpointSlots int
	SpillEnabled    bool
	Err             error
	store           *localSSDSnapshotStore
}

// localSSDDir picks the overflow store's directory: the FAK_L3_KVBACKEND root when set to a
// usable directory (sibling of the warm-disk native-prefix slot, never inside it, so the two
// tiers' prune policies cannot delete each other's records), otherwise the per-user cache
// default. It mirrors resolveWarmDiskConfig's root handling.
func localSSDDir() string {
	root := strings.TrimSpace(os.Getenv(l3kv.EnvSpec))
	if root != "" && !strings.EqualFold(root, "off") {
		return filepath.Join(root, "ssd-l3")
	}
	cacheRoot, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(cacheRoot, "fak", "ssd-l3")
}

func localSSDDirWith(lookupEnv func(string) (string, bool)) string {
	if lookupEnv != nil {
		if v, ok := lookupEnv(l3kv.EnvSpec); ok {
			root := strings.TrimSpace(v)
			if root != "" && !strings.EqualFold(root, "off") {
				return filepath.Join(root, "ssd-l3")
			}
		}
	}
	return localSSDDir()
}

// resolveInKernelRadixLocalSSD sizes the radix tree's local-SSD overflow tier:
//
//  1. FAK_INKERNEL_RADIX_SSD_BYTES set (any value, including 0 = the OFF arm) is honoured
//     exactly; an unparseable or negative value refuses and fails closed to off.
//  2. Otherwise, with the store volume's free bytes known, min(cap, fraction x free) rounded
//     down to MiB.
//  3. Unknown disk capacity: off, today's default.
//  4. FAK_INKERNEL_RADIX_SSD_CHECKPOINT_SLOTS > 0 requires an active spill budget (> 0);
//     setting it while spill is disabled/off refuses and leaves the tier off.
//
// lookupEnv and diskInfo are injected so the derivation is unit-testable with simulated
// volumes; production passes os.LookupEnv and compute.DiskInfo.
func resolveInKernelRadixLocalSSD(lookupEnv func(string) (string, bool), diskInfo func(string) (int64, int64, bool)) InKernelRadixLocalSSD {
	dir := localSSDDirWith(lookupEnv)
	var checkpointSlots int
	if lookupEnv != nil {
		if v, ok := lookupEnv(inKernelRadixLocalSSDCheckpointSlotsEnv); ok && strings.TrimSpace(v) != "" {
			slots, err := strconv.Atoi(strings.TrimSpace(v))
			if err != nil || slots < 0 {
				return InKernelRadixLocalSSD{
					Source: InKernelRadixLocalSSDSourceRefused,
					Err:    fmt.Errorf("%w: %s=%q: want a non-negative integer", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDCheckpointSlotsEnv, v),
				}
			}
			checkpointSlots = slots
		}
		if v, ok := lookupEnv(inKernelRadixLocalSSDBytesEnv); ok {
			n, err := strconv.ParseInt(strings.TrimSpace(v), 10, 64)
			if err != nil || n < 0 {
				return InKernelRadixLocalSSD{
					Source: InKernelRadixLocalSSDSourceRefused,
					Err:    fmt.Errorf("%w: %s=%q: want a non-negative byte count", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDBytesEnv, v),
				}
			}
			if n == 0 && checkpointSlots > 0 {
				return InKernelRadixLocalSSD{
					Source: InKernelRadixLocalSSDSourceRefused,
					Err:    fmt.Errorf("%w: %s=%d requires active spill (%s > 0)", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDCheckpointSlotsEnv, checkpointSlots, inKernelRadixLocalSSDBytesEnv),
				}
			}
			return InKernelRadixLocalSSD{
				Bytes:           n,
				Dir:             dir,
				Source:          InKernelRadixLocalSSDSourceEnv,
				CheckpointSlots: checkpointSlots,
				SpillEnabled:    n > 0,
			}
		}
	}
	if diskInfo == nil {
		if checkpointSlots > 0 {
			return InKernelRadixLocalSSD{
				Source: InKernelRadixLocalSSDSourceRefused,
				Err:    fmt.Errorf("%w: %s=%d requires active spill (%s > 0)", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDCheckpointSlotsEnv, checkpointSlots, inKernelRadixLocalSSDBytesEnv),
			}
		}
		return InKernelRadixLocalSSD{Dir: dir, Source: InKernelRadixLocalSSDSourceUnknown}
	}
	_, free, known := diskInfo(dir)
	if !known || free <= 0 {
		if checkpointSlots > 0 {
			return InKernelRadixLocalSSD{
				Source: InKernelRadixLocalSSDSourceRefused,
				Err:    fmt.Errorf("%w: %s=%d requires active spill (%s > 0)", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDCheckpointSlotsEnv, checkpointSlots, inKernelRadixLocalSSDBytesEnv),
			}
		}
		return InKernelRadixLocalSSD{Dir: dir, Source: InKernelRadixLocalSSDSourceUnknown}
	}
	budget := int64(float64(free) * inKernelRadixLocalSSDFractionDef)
	if budget > inKernelRadixLocalSSDCap {
		budget = inKernelRadixLocalSSDCap
	}
	budget &^= 1<<20 - 1
	if budget <= 0 && checkpointSlots > 0 {
		return InKernelRadixLocalSSD{
			Source: InKernelRadixLocalSSDSourceRefused,
			Err:    fmt.Errorf("%w: %s=%d requires active spill (%s > 0)", ErrInKernelRadixLocalSSDConfig, inKernelRadixLocalSSDCheckpointSlotsEnv, checkpointSlots, inKernelRadixLocalSSDBytesEnv),
		}
	}
	return InKernelRadixLocalSSD{
		Bytes:           budget,
		Dir:             dir,
		Source:          InKernelRadixLocalSSDSourceDefault,
		CheckpointSlots: checkpointSlots,
		SpillEnabled:    budget > 0,
	}
}

// armRadixLocalSSD installs the local-SSD KV overflow tier on the planner's radix tree at
// construction, so both `fak serve` and turnkey `fak up` demote evicted host-L2 prefixes to
// local disk instead of dropping them. CPU-only planners (backend == nil) stay off: their
// snapshots have no device install target to restore into, exactly like the remote L3 tier.
// A resolution refusal or an unopenable store prints and leaves the tier off; the host L2
// tier is unaffected either way.
func (p *InKernelPlanner) armRadixLocalSSD(m *model.Model, backend compute.Backend, modelID string) {
	p.armRadixLocalSSDWith(m, backend, modelID, os.LookupEnv, compute.DiskInfo)
}

func (p *InKernelPlanner) armRadixLocalSSDWith(
	m *model.Model,
	backend compute.Backend,
	modelID string,
	lookupEnv func(string) (string, bool),
	diskInfo func(string) (int64, int64, bool),
) {
	if p == nil || p.tree == nil || m == nil || backend == nil {
		return
	}
	ssd := resolveInKernelRadixLocalSSD(lookupEnv, diskInfo)
	if ssd.Err != nil {
		fmt.Fprintln(os.Stderr, ssd.Err)
	}
	p.radixLocalSSD = ssd
	if ssd.Bytes <= 0 || ssd.Dir == "" {
		return
	}
	store, err := l3kv.NewDiskStore(ssd.Dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak inkernel: local SSD L3 KV overflow unavailable: %v\n", err)
		p.radixLocalSSD.Bytes = 0
		p.radixLocalSSD.SpillEnabled = false
		return
	}
	cleanupTempFiles(ssd.Dir, inKernelRadixLocalSSDStaleTempAge)
	cfg := defaultLocalSSDSpillConfig(ssd.Bytes)
	cfg.CheckpointSlots = ssd.CheckpointSlots
	ssdStore := newLocalSSDSnapshotStoreWithConfig(store, ssd.Bytes, cfg)
	if ssdStore.err != nil {
		fmt.Fprintln(os.Stderr, ssdStore.err)
		p.radixLocalSSD.Bytes = 0
		p.radixLocalSSD.SpillEnabled = false
		p.radixLocalSSD.Err = ssdStore.err
		return
	}
	if err := p.tree.ConfigureLocalSSDSnapshotStore(ssdStore, modelID, backend, m.Cfg, ssd.Bytes); err != nil {
		fmt.Fprintln(os.Stderr, err)
		_ = ssdStore.Close()
		p.radixLocalSSD.Bytes = 0
		p.radixLocalSSD.SpillEnabled = false
		return
	}
	p.radixLocalSSD.store = ssdStore
	p.radixLocalSSD.SpillEnabled = ssdStore.SpillEnabled()
	fmt.Fprintf(os.Stderr, "fak inkernel: radix local-SSD L3 KV overflow=%d bytes dir=%s source=%s checkpoint_slots=%d\n",
		ssd.Bytes, ssd.Dir, ssd.Source, ssd.CheckpointSlots)
}

// RadixLocalSSD reports the local-SSD KV overflow budget this planner resolved at
// construction; the zero value when the tier is off.
func (p *InKernelPlanner) RadixLocalSSD() InKernelRadixLocalSSD {
	if p == nil {
		return InKernelRadixLocalSSD{}
	}
	return p.radixLocalSSD
}

// RadixLocalSSDSpillStats snapshots eviction-spill pressure on the armed local-SSD
// store (zero value when the tier is disabled).
func (p *InKernelPlanner) RadixLocalSSDSpillStats() l3kv.SpillStats {
	if p == nil || p.radixLocalSSD.store == nil {
		return l3kv.SpillStats{}
	}
	return p.radixLocalSSD.store.SpillStats()
}

// FlushRadixLocalSSD runs the bounded planned-stop flush (<= 20 s budget) for the
// local-SSD eviction-spill tier, publishing queued and remaining retained RAM states
// under the disk cap and cancelling anything past deadline.
func (p *InKernelPlanner) FlushRadixLocalSSD(deadline time.Duration) (published, skipped int) {
	if p == nil || p.radixLocalSSD.store == nil {
		return 0, 0
	}
	return p.radixLocalSSD.store.Flush(deadline)
}

type localSSDSpillConfig struct {
	MaxDiskBytes    int64
	MaxEntryBytes   int64
	PutTimeout      time.Duration
	StopBudget      time.Duration
	StagingPermit   int64
	MaxStagingJobs  int
	CheckpointSlots int
	PruneOnPut      bool
}

func defaultLocalSSDSpillConfig(budget int64) localSSDSpillConfig {
	return localSSDSpillConfig{
		MaxDiskBytes:   budget,
		MaxEntryBytes:  budget,
		PutTimeout:     inKernelRadixLocalSSDSpillTimeout,
		StopBudget:     inKernelRadixLocalSSDStopBudget,
		StagingPermit:  inKernelRadixLocalSSDDefaultStagingPermit,
		MaxStagingJobs: inKernelRadixLocalSSDDefaultStagingJobs,
		PruneOnPut:     true,
	}
}

func clampLocalSSDStopBudget(d time.Duration) time.Duration {
	if d <= 0 || d > inKernelRadixLocalSSDStopBudget {
		return inKernelRadixLocalSSDStopBudget
	}
	return d
}

// localSSDSnapshotStore is the SnapshotStore adapter that owns the overflow tier's disk
// hygiene and eviction-spill lifecycle:
//   - Live l3kv.AsyncStore.EnableSpill wiring when budget > 0 (disabled when <= 0)
//   - Checkpoint-slot RAM bounding with copy-before-release on eviction
//   - Bounded async staging queue that skips saves when staging permit is full
//   - Planned-stop flush bounded to <= 20 s and the configured disk cap
//   - Damaged-file deletion on Get/GetAsync so callers fall back cleanly to prefill
//   - Atomic writes (no truncated artifacts left on cut-off writes) and stale temp cleanup
type localSSDSnapshotStore struct {
	disk           *l3kv.DiskStore
	base           l3kv.Store
	async          *l3kv.AsyncStore
	spiller        *l3kv.Spiller
	slots          *checkpointSlotGate
	stager         *localSSDSpillStager
	maxB           int64
	maxEnt         int
	putTimeout     time.Duration
	stopBudget     time.Duration
	pruneOnPut     bool
	skippedStaging atomic.Int64
	err            error
}

func newLocalSSDSnapshotStore(disk *l3kv.DiskStore, budget int64) *localSSDSnapshotStore {
	return newLocalSSDSnapshotStoreWithConfig(disk, budget, defaultLocalSSDSpillConfig(budget))
}

func newLocalSSDSnapshotStoreWithConfig(disk *l3kv.DiskStore, budget int64, cfg localSSDSpillConfig) *localSSDSnapshotStore {
	var base l3kv.Store
	if disk != nil {
		base = disk
	}
	return newLocalSSDSnapshotStoreWithStore(base, disk, budget, cfg)
}

func newLocalSSDSnapshotStoreWithStore(store l3kv.Store, disk *l3kv.DiskStore, budget int64, cfg localSSDSpillConfig) *localSSDSnapshotStore {
	if store == nil || budget <= 0 {
		var cfgErr error
		if cfg.CheckpointSlots > 0 {
			_, cfgErr = l3kv.NewSpiller(nil, l3kv.SpillConfig{CheckpointSlots: cfg.CheckpointSlots})
		}
		return &localSSDSnapshotStore{
			disk:   disk,
			base:   store,
			maxEnt: inKernelRadixLocalSSDMaxEntries,
			slots:  newCheckpointSlotGate(0),
			err:    cfgErr,
		}
	}
	if cfg.MaxDiskBytes <= 0 {
		cfg.MaxDiskBytes = budget
	}
	if cfg.MaxEntryBytes <= 0 {
		cfg.MaxEntryBytes = budget
	}
	if cfg.PutTimeout <= 0 || cfg.PutTimeout > inKernelRadixLocalSSDSpillTimeout {
		cfg.PutTimeout = inKernelRadixLocalSSDSpillTimeout
	}
	cfg.StopBudget = clampLocalSSDStopBudget(cfg.StopBudget)
	if cfg.StagingPermit <= 0 {
		cfg.StagingPermit = inKernelRadixLocalSSDDefaultStagingPermit
	}
	if cfg.MaxStagingJobs <= 0 {
		cfg.MaxStagingJobs = inKernelRadixLocalSSDDefaultStagingJobs
	}
	async := l3kv.NewAsyncStore(store)
	sp, err := async.EnableSpill(l3kv.SpillConfig{
		MaxDiskBytes:    cfg.MaxDiskBytes,
		MaxEntryBytes:   cfg.MaxEntryBytes,
		PutTimeout:      cfg.PutTimeout,
		CheckpointSlots: cfg.CheckpointSlots,
	})
	if err != nil {
		return &localSSDSnapshotStore{
			disk:   disk,
			base:   store,
			async:  async,
			maxEnt: inKernelRadixLocalSSDMaxEntries,
			slots:  newCheckpointSlotGate(0),
			err:    err,
		}
	}
	return &localSSDSnapshotStore{
		disk:       disk,
		base:       store,
		async:      async,
		spiller:    sp,
		slots:      newCheckpointSlotGate(cfg.CheckpointSlots),
		stager:     newLocalSSDSpillStager(async, cfg.StagingPermit, cfg.MaxStagingJobs),
		maxB:       budget,
		maxEnt:     inKernelRadixLocalSSDMaxEntries,
		putTimeout: cfg.PutTimeout,
		stopBudget: cfg.StopBudget,
		pruneOnPut: cfg.PruneOnPut,
	}
}

func (s *localSSDSnapshotStore) SpillEnabled() bool {
	return s != nil && s.spiller != nil && s.maxB > 0
}

func (s *localSSDSnapshotStore) CheckpointSlots() int {
	if s == nil || s.spiller == nil {
		return 0
	}
	return s.spiller.CheckpointSlots()
}

func (s *localSSDSnapshotStore) SpillStats() l3kv.SpillStats {
	if s == nil || s.async == nil {
		return l3kv.SpillStats{}
	}
	return s.async.SpillStats()
}

func (s *localSSDSnapshotStore) StagingStats() localSSDStagingStats {
	var st localSSDStagingStats
	if s == nil {
		return st
	}
	if s.stager != nil {
		st = s.stager.stats()
	}
	st.SkippedPermit += int(s.skippedStaging.Load())
	return st
}

func (s *localSSDSnapshotStore) RAMCount() int {
	if s == nil || s.slots == nil {
		return 0
	}
	return s.slots.count()
}

func (s *localSSDSnapshotStore) HasInRAM(key string) bool {
	if s == nil || s.slots == nil {
		return false
	}
	return s.slots.has(key)
}

// Spill stages an owned copy of payload to the durable backing store under the configured
// SpillConfig caps (MaxDiskBytes, MaxEntryBytes, PutTimeout). Returns ErrSpillDisabled
// when spill is off.
func (s *localSSDSnapshotStore) Spill(ctx context.Context, key string, payload []byte) error {
	if !s.SpillEnabled() {
		return l3kv.ErrSpillDisabled
	}
	if ctx == nil {
		ctx = context.Background()
	}
	return s.async.Spill(ctx, key, payload)
}

// RetainRAMState records a prompt state in the RAM checkpoint-slot table beside the disk
// store. When CheckpointSlots > 0 and retained states exceed the bound, the oldest RAM
// state is copied BEFORE its RAM release callback runs and its owned copy is staged for
// spill under the disk/staging caps.
func (s *localSSDSnapshotStore) RetainRAMState(ctx context.Context, key string, payload []byte, onRelease func()) ([]string, error) {
	if s == nil || s.slots == nil {
		return nil, l3kv.ErrSpillDisabled
	}
	if s.err != nil {
		return nil, s.err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	evicted := s.slots.retain(key, payload, onRelease)
	var keys []string
	for _, victim := range evicted {
		keys = append(keys, victim.key)
		if victim.onRelease != nil {
			victim.onRelease()
		}
		if s.SpillEnabled() {
			_, _, _ = s.stageSpillItem(ctx, victim.key, victim.copied)
		}
	}
	return keys, nil
}

// EvictRAMState evicts key from the RAM checkpoint-slot table, copying its payload BEFORE
// releasing RAM and staging the copy to disk when spill is active.
func (s *localSSDSnapshotStore) EvictRAMState(ctx context.Context, key string) (spilled, skipped bool, err error) {
	if s == nil || s.slots == nil {
		return false, true, l3kv.ErrSpillDisabled
	}
	if ctx == nil {
		ctx = context.Background()
	}
	victim, found := s.slots.evict(key)
	if !found {
		return false, false, nil
	}
	if victim.onRelease != nil {
		victim.onRelease()
	}
	if !s.SpillEnabled() {
		return false, true, l3kv.ErrSpillDisabled
	}
	_, skipped, err = s.stageSpillItem(ctx, victim.key, victim.copied)
	return !skipped && err == nil, skipped, err
}

func (s *localSSDSnapshotStore) StageSpill(ctx context.Context, key string, payload []byte) (skipped bool, err error) {
	_, skipped, err = s.stageSpillItem(ctx, key, payload)
	return skipped, err
}

func (s *localSSDSnapshotStore) stageSpillItem(ctx context.Context, key string, payload []byte) (*localSSDStagedItem, bool, error) {
	if !s.SpillEnabled() {
		return nil, true, l3kv.ErrSpillDisabled
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, true, err
	}
	if len(payload) == 0 {
		return nil, false, nil
	}
	owned := append([]byte(nil), payload...)
	return s.stager.stage(key, owned)
}

// Flush runs the bounded planned-stop protocol (<= 20 s budget): drains queued spills
// until empty or deadline, then copies, releases, and saves any remaining retained RAM
// states that fit under the disk cap and remaining stop budget.
func (s *localSSDSnapshotStore) Flush(deadline time.Duration) (published, skipped int) {
	if s == nil {
		return 0, 0
	}
	budget := deadline
	if budget <= 0 {
		budget = s.stopBudget
	}
	budget = clampLocalSSDStopBudget(budget)
	deadlineAt := time.Now().Add(budget)

	if s.stager != nil {
		p1, s1 := s.stager.drain(budget)
		published += p1
		skipped += s1
	}
	if s.slots == nil {
		return published, skipped
	}
	remaining := s.slots.drainAll()
	for _, entry := range remaining {
		if entry.onRelease != nil {
			entry.onRelease()
		}
		if !s.SpillEnabled() || len(entry.copied) == 0 {
			skipped++
			continue
		}
		rem := time.Until(deadlineAt)
		if rem <= 0 {
			skipped++
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), rem)
		before := s.async.SpillStats()
		err := s.async.Spill(ctx, entry.key, entry.copied)
		after := s.async.SpillStats()
		cancel()
		if err == nil && after.Spilled > before.Spilled {
			published++
		} else {
			skipped++
		}
	}
	return published, skipped
}

// Close executes the bounded planned-stop flush and closes the stager.
func (s *localSSDSnapshotStore) Close() error {
	if s == nil {
		return nil
	}
	s.Flush(s.stopBudget)
	return nil
}

func (s *localSSDSnapshotStore) Put(ctx context.Context, key string, value []byte) error {
	if s == nil || s.base == nil || s.maxB <= 0 {
		return l3kv.ErrSpillDisabled
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if s.stager != nil && s.stager.isFull(int64(len(value))) {
		s.skippedStaging.Add(1)
		return nil
	}
	// Copy before writing so callers may reuse or release value immediately.
	owned := append([]byte(nil), value...)
	putCtx := ctx
	var cancel context.CancelFunc
	if s.putTimeout > 0 {
		putCtx, cancel = context.WithTimeout(ctx, s.putTimeout)
		defer cancel()
	}
	done := make(chan error, 1)
	go func() {
		done <- s.base.Put(putCtx, key, owned)
	}()
	select {
	case err := <-done:
		if err != nil {
			return err
		}
	case <-putCtx.Done():
		return putCtx.Err()
	}
	if s.disk != nil && s.pruneOnPut {
		// Best-effort hygiene: a prune failure leaves the record in place (the tree-side
		// budget already refused the spill if the referenced total was over the cap).
		_ = s.disk.Prune(s.maxEnt, s.maxB, key)
	}
	return nil
}

func (s *localSSDSnapshotStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	if s == nil || s.base == nil {
		return nil, false, l3kv.ErrSpillDisabled
	}
	if ctx == nil {
		ctx = context.Background()
	}
	b, found, err := s.base.Get(ctx, key)
	if err != nil && isDamagedDiskRecordErr(err) {
		_ = s.Delete(context.Background(), []string{key})
		return nil, false, nil
	}
	return b, found, err
}

func (s *localSSDSnapshotStore) Touch(ctx context.Context, keys []string) error {
	if s == nil || s.disk == nil {
		return nil
	}
	return s.disk.Touch(ctx, keys)
}

func (s *localSSDSnapshotStore) Delete(ctx context.Context, keys []string) error {
	if s == nil {
		return nil
	}
	if s.disk != nil {
		return s.disk.Delete(ctx, keys)
	}
	if deleter, ok := s.base.(radixkv.SnapshotDeleter); ok {
		return deleter.Delete(ctx, keys)
	}
	return nil
}

func (s *localSSDSnapshotStore) GetAsync(ctx context.Context, key string, onComplete func([]byte, bool, error)) {
	wrapped := func(b []byte, found bool, err error) {
		if err != nil && !errors.Is(err, l3kv.ErrAsyncJobStale) && isDamagedDiskRecordErr(err) {
			_ = s.Delete(context.Background(), []string{key})
			b, found, err = nil, false, nil
		}
		if onComplete != nil {
			onComplete(b, found, err)
		}
	}
	if s != nil && s.async != nil {
		s.async.GetAsync(ctx, key, wrapped)
		return
	}
	go func() {
		b, ok, err := s.Get(ctx, key)
		wrapped(b, ok, err)
	}()
}

// isDamagedDiskRecordErr matches the deterministic on-disk corruption errors emitted by
// l3kv.DiskStore.Get so damaged records are removed and requests cleanly fall back to prefill,
// without deleting records on context cancellation or stale async jobs.
func isDamagedDiskRecordErr(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	msg := err.Error()
	return strings.Contains(msg, "truncated record") ||
		strings.Contains(msg, "unrecognized record magic") ||
		strings.Contains(msg, "unsupported record version") ||
		strings.Contains(msg, "integrity check failed")
}

// cleanupTempFiles removes orphan atomic-write temporary files (.l3kv-* and .fakssd-*)
// in dir whose age meets or exceeds maxAge (<= 0 cleans all matching temp files).
func cleanupTempFiles(dir string, maxAge time.Duration) int {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0
	}
	removed := 0
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasPrefix(name, ".l3kv-") && !strings.HasPrefix(name, ".fakssd-") {
			continue
		}
		if maxAge > 0 {
			info, infoErr := entry.Info()
			if infoErr != nil || time.Since(info.ModTime()) < maxAge {
				continue
			}
		}
		if err := os.Remove(filepath.Join(dir, name)); err == nil {
			removed++
		}
	}
	return removed
}

type ramCheckpointEntry struct {
	key       string
	payload   []byte
	onRelease func()
	seq       uint64
}

type evictedRAMCheckpoint struct {
	key       string
	copied    []byte
	onRelease func()
	seq       uint64
}

type checkpointSlotGate struct {
	mu       sync.Mutex
	maxSlots int
	seq      uint64
	entries  map[string]*ramCheckpointEntry
}

func newCheckpointSlotGate(maxSlots int) *checkpointSlotGate {
	if maxSlots < 0 {
		maxSlots = 0
	}
	return &checkpointSlotGate{
		maxSlots: maxSlots,
		entries:  map[string]*ramCheckpointEntry{},
	}
}

func (g *checkpointSlotGate) count() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return len(g.entries)
}

func (g *checkpointSlotGate) has(key string) bool {
	g.mu.Lock()
	defer g.mu.Unlock()
	_, ok := g.entries[key]
	return ok
}

func (g *checkpointSlotGate) retain(key string, payload []byte, onRelease func()) []evictedRAMCheckpoint {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.seq++
	if existing, ok := g.entries[key]; ok {
		existing.payload = payload
		existing.onRelease = onRelease
		existing.seq = g.seq
	} else {
		g.entries[key] = &ramCheckpointEntry{
			key:       key,
			payload:   payload,
			onRelease: onRelease,
			seq:       g.seq,
		}
	}
	if g.maxSlots <= 0 || len(g.entries) <= g.maxSlots {
		return nil
	}
	var evicted []evictedRAMCheckpoint
	for len(g.entries) > g.maxSlots {
		var victim *ramCheckpointEntry
		for k, e := range g.entries {
			if k == key {
				continue
			}
			if victim == nil || e.seq < victim.seq {
				victim = e
			}
		}
		if victim == nil {
			break
		}
		evicted = append(evicted, evictedRAMCheckpoint{
			key:       victim.key,
			copied:    append([]byte(nil), victim.payload...),
			onRelease: victim.onRelease,
			seq:       victim.seq,
		})
		delete(g.entries, victim.key)
	}
	return evicted
}

func (g *checkpointSlotGate) evict(key string) (evictedRAMCheckpoint, bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	entry, ok := g.entries[key]
	if !ok {
		return evictedRAMCheckpoint{}, false
	}
	out := evictedRAMCheckpoint{
		key:       entry.key,
		copied:    append([]byte(nil), entry.payload...),
		onRelease: entry.onRelease,
		seq:       entry.seq,
	}
	delete(g.entries, key)
	return out, true
}

func (g *checkpointSlotGate) drainAll() []evictedRAMCheckpoint {
	g.mu.Lock()
	defer g.mu.Unlock()
	out := make([]evictedRAMCheckpoint, 0, len(g.entries))
	for _, entry := range g.entries {
		out = append(out, evictedRAMCheckpoint{
			key:       entry.key,
			copied:    append([]byte(nil), entry.payload...),
			onRelease: entry.onRelease,
			seq:       entry.seq,
		})
	}
	g.entries = map[string]*ramCheckpointEntry{}
	sort.Slice(out, func(i, j int) bool {
		if out[i].seq == out[j].seq {
			return out[i].key < out[j].key
		}
		return out[i].seq < out[j].seq
	})
	return out
}

type localSSDStagingStats struct {
	Queued        int
	Inflight      int
	ChargedBytes  int64
	Written       int
	Failed        int
	SkippedPermit int
	SkippedStop   int
}

type localSSDStagedItem struct {
	key     string
	payload []byte
	done    chan error
}

type localSSDSpillStager struct {
	async       *l3kv.AsyncStore
	mu          sync.Mutex
	cond        *sync.Cond
	queue       []localSSDStagedItem
	charged     int64
	inflight    int
	permit      int64
	maxJobs     int
	started     bool
	closed      bool
	stopped     chan struct{}
	written     int
	failed      int
	skippedPerm int
	skippedStop int
}

func newLocalSSDSpillStager(async *l3kv.AsyncStore, permit int64, maxJobs int) *localSSDSpillStager {
	s := &localSSDSpillStager{
		async:   async,
		permit:  permit,
		maxJobs: maxJobs,
	}
	s.cond = sync.NewCond(&s.mu)
	return s
}

func (s *localSSDSpillStager) fitsLocked(size int64) bool {
	return size >= 0 && len(s.queue)+s.inflight < s.maxJobs && s.charged+size <= s.permit
}

func (s *localSSDSpillStager) isFull(size int64) bool {
	if s == nil {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return true
	}
	return !s.fitsLocked(size)
}

func (s *localSSDSpillStager) stage(key string, copied []byte) (*localSSDStagedItem, bool, error) {
	size := int64(len(copied))
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		s.skippedStop++
		return nil, true, errLocalSSDSpillClosed
	}
	if !s.fitsLocked(size) {
		s.skippedPerm++
		return nil, true, nil
	}
	if !s.started {
		s.started = true
		s.stopped = make(chan struct{})
		go s.run()
	}
	item := &localSSDStagedItem{
		key:     key,
		payload: copied,
		done:    make(chan error, 1),
	}
	s.queue = append(s.queue, *item)
	s.charged += size
	s.cond.Signal()
	return item, false, nil
}

func (s *localSSDSpillStager) run() {
	defer close(s.stopped)
	for {
		s.mu.Lock()
		for len(s.queue) == 0 && !s.closed {
			s.cond.Wait()
		}
		if len(s.queue) == 0 {
			s.mu.Unlock()
			return
		}
		save := s.queue[0]
		s.queue = s.queue[1:]
		s.inflight++
		s.mu.Unlock()

		before := s.async.SpillStats()
		err := s.async.Spill(context.Background(), save.key, save.payload)
		after := s.async.SpillStats()

		s.mu.Lock()
		s.inflight--
		s.charged -= int64(len(save.payload))
		if s.charged < 0 {
			s.charged = 0
		}
		if err != nil {
			s.failed++
		} else if after.Spilled > before.Spilled {
			s.written++
		}
		s.cond.Broadcast()
		s.mu.Unlock()
		save.done <- err
	}
}

func (s *localSSDSpillStager) drain(deadline time.Duration) (published, skipped int) {
	if s == nil {
		return 0, 0
	}
	s.mu.Lock()
	startWritten := s.written
	alreadyClosed := s.closed
	started := s.started
	s.mu.Unlock()
	if !alreadyClosed && started {
		deadlineAt := time.Now().Add(deadline)
		for {
			s.mu.Lock()
			empty := len(s.queue) == 0 && s.inflight == 0
			s.mu.Unlock()
			if empty || !time.Now().Before(deadlineAt) {
				break
			}
			time.Sleep(2 * time.Millisecond)
		}
	}
	s.mu.Lock()
	for i := range s.queue {
		s.charged -= int64(len(s.queue[i].payload))
		s.queue[i].done <- errLocalSSDSpillCancelled
	}
	if s.charged < 0 {
		s.charged = 0
	}
	cancelled := len(s.queue)
	s.queue = nil
	s.closed = true
	s.skippedStop += cancelled
	s.cond.Broadcast()
	waitStopped := s.started && s.inflight == 0
	stoppedCh := s.stopped
	s.mu.Unlock()
	if waitStopped && stoppedCh != nil {
		<-stoppedCh
	}
	s.mu.Lock()
	published = s.written - startWritten
	s.mu.Unlock()
	return published, cancelled
}

func (s *localSSDSpillStager) stats() localSSDStagingStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	return localSSDStagingStats{
		Queued:        len(s.queue),
		Inflight:      s.inflight,
		ChargedBytes:  s.charged,
		Written:       s.written,
		Failed:        s.failed,
		SkippedPermit: s.skippedPerm,
		SkippedStop:   s.skippedStop,
	}
}

var _ radixkv.SnapshotStore = (*localSSDSnapshotStore)(nil)
var _ radixkv.SnapshotToucher = (*localSSDSnapshotStore)(nil)
var _ radixkv.SnapshotDeleter = (*localSSDSnapshotStore)(nil)
var _ radixkv.SnapshotAsyncGetter = (*localSSDSnapshotStore)(nil)
