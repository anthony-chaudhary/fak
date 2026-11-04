package l3kv

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// DefaultSpillPutTimeout bounds how long one staged spill copy may take before
// the bounded window closes and the eviction proceeds without it.
const DefaultSpillPutTimeout = 30 * time.Second

// ErrSpillDisabled reports that the spill write side is off (no store attached
// or not enabled); the eviction proceeds without a durable copy.
var ErrSpillDisabled = errors.New("l3kv: spill disabled")

// SpillConfig tunes a Spiller. MaxDiskBytes <= 0 means no total disk budget;
// MaxEntryBytes <= 0 means no per-entry cap; PutTimeout <= 0 takes
// DefaultSpillPutTimeout; CheckpointSlots < 0 normalizes to 0. CheckpointSlots
// bounds how many span states stay retained in RAM beside the disk store; the
// Spiller carries and validates the knob — enforcement lives on the retaining
// side (the wrapped backend's eviction policy).
type SpillConfig struct {
	MaxDiskBytes    int64
	MaxEntryBytes   int64
	PutTimeout      time.Duration
	CheckpointSlots int
}

// DefaultSpillConfig is the default setting: the 30 s staging bound, uncapped
// disk and entry sizes, no explicit checkpoint-slot bound.
var DefaultSpillConfig = SpillConfig{PutTimeout: DefaultSpillPutTimeout}

func (c SpillConfig) normalized() SpillConfig {
	if c.PutTimeout <= 0 {
		c.PutTimeout = DefaultSpillPutTimeout
	}
	if c.CheckpointSlots < 0 {
		c.CheckpointSlots = 0
	}
	return c
}

// SpillStats is a point-in-time snapshot of spill pressure.
type SpillStats struct {
	Spilled          int   // durable copies confirmed
	SkippedOversized int   // entries larger than MaxEntryBytes
	SkippedBudget    int   // running disk total would exceed MaxDiskBytes
	SkippedTimeout   int   // bounded staging window closed (deadline or caller cancel)
	Faults           int   // store Put failures
	DiskBytes        int64 // running total of confirmed spilled payload bytes
}

// Spiller is the write side of eviction-spill: when a span's RAM residency is
// about to be released, Spill stages an owned durable copy under the span's
// residency digest through the SAME Store.Put path StageSpan uses, so the
// existing restore path (Store.Get -> RestoreSpan) reads it back. Adapted from
// TensorFold's disk_spill contract (zig/src/native/disk_spill.zig,
// zig/src/server/cli.zig:518-523, zig/src/core/prompt_cache.zig:191
// @9e8c368b0b4c2c7892a0c9f0f302d7c164016209, Apache-2.0): optional spill copies
// before releasing RAM, reusing the durable restore path, capped on disk,
// entry size, and time; checkpoint slots bound retained RAM states.
// It is safe for concurrent use.
type Spiller struct {
	store Store
	cfg   SpillConfig

	mu    sync.Mutex
	stats SpillStats
}

// NewSpiller builds a Spiller over store. A nil store disables every Spill
// call (ErrSpillDisabled). CheckpointSlots > 0 without a store is refused:
// the slots bound RAM states BESIDE the disk store, so they need one.
func NewSpiller(store Store, cfg SpillConfig) (*Spiller, error) {
	cfg = cfg.normalized()
	if store == nil && cfg.CheckpointSlots > 0 {
		return nil, fmt.Errorf("l3kv: checkpoint slots (%d) set how many span states stay in RAM beside the disk store: pass a store (enable spill), or clear CheckpointSlots", cfg.CheckpointSlots)
	}
	return &Spiller{store: store, cfg: cfg}, nil
}

// CheckpointSlots reports the normalized retained-RAM-state bound carried by
// the Spiller (0: no explicit bound — the retaining side's plan applies).
func (sp *Spiller) CheckpointSlots() int {
	if sp == nil {
		return 0
	}
	return sp.cfg.CheckpointSlots
}

// Spill stages an owned durable copy of payload under digest before the
// caller releases the span's RAM residency. It is best-effort by contract:
// cap skips and bounded-window aborts return nil and are reported through
// Stats only; only a store fault returns an error, and then the caller
// retains the live span (fail-safe, the StageSpan FAULT posture). The payload
// is copied: the caller may free or reuse the slice immediately. Caps are
// soft under concurrency — each Spill checks the running totals before its
// write, so simultaneous spills may overshoot MaxDiskBytes by the in-flight
// entries. A store that ignores its context may finish a timed-out write
// late; the copy is an idempotent overwrite and simply goes uncredited.
func (sp *Spiller) Spill(ctx context.Context, digest string, payload []byte) error {
	sp.mu.Lock()
	if sp.store == nil {
		sp.mu.Unlock()
		return ErrSpillDisabled
	}
	if len(payload) == 0 {
		sp.mu.Unlock()
		return nil // nothing to preserve; eviction proceeds
	}
	size := int64(len(payload))
	if sp.cfg.MaxEntryBytes > 0 && size > sp.cfg.MaxEntryBytes {
		sp.stats.SkippedOversized++
		sp.mu.Unlock()
		return nil
	}
	if sp.cfg.MaxDiskBytes > 0 && sp.stats.DiskBytes+size > sp.cfg.MaxDiskBytes {
		sp.stats.SkippedBudget++
		sp.mu.Unlock()
		return nil
	}
	owned := make([]byte, size)
	copy(owned, payload)
	putCtx, cancel := context.WithTimeout(ctx, sp.cfg.PutTimeout)
	sp.mu.Unlock()

	done := make(chan error, 1)
	go func() { done <- sp.store.Put(putCtx, digest, owned) }()
	var err error
	select {
	case err = <-done:
	case <-putCtx.Done():
		err = putCtx.Err()
	}
	cancel()

	sp.mu.Lock()
	defer sp.mu.Unlock()
	switch {
	case err == nil:
		sp.stats.Spilled++
		sp.stats.DiskBytes += size
	case errors.Is(err, context.DeadlineExceeded), errors.Is(err, context.Canceled):
		sp.stats.SkippedTimeout++
	default:
		sp.stats.Faults++
		return err
	}
	return nil
}

// Stats snapshots current spill pressure.
func (sp *Spiller) Stats() SpillStats {
	if sp == nil {
		return SpillStats{}
	}
	sp.mu.Lock()
	defer sp.mu.Unlock()
	return sp.stats
}
