// Physical-pool charge/release ledger (SEAM-01) with a pinned-host row class.
//
// Provenance (ADAPT; no upstream code copied): the host tier adapts the
// DeepSeek-V4.1-Flash Engram host-offload accounting from the H200 OSS serving
// recipes — vLLM --engram-config cpu_offload + use_thp at TP4 moves 23.6 GiB/rank
// of Engram tables into pinned host DRAM read through UVA (bitwise-unchanged
// output), and SGLang placement cpu / lookup peer_read is the equivalent
// placement. Source: vllm-project/vllm,
// recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml @ e740badc2e3df13a5c2abfca04ed0ac917985c5f
// (Apache-2.0); study note fak-private docs/research/STUDY-ngram-engram-serving-
// H200-recipe-seams-2026-11-05.md (row 1). Failure classes closed here: MBL-084
// (host+device counted as disjoint windows), MBL-085 (unbounded eager dequant),
// MBL-087 (double-charging produced spurious refusals).
package kvmmu

import (
	"errors"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// PoolRowClass names one physical memory tier the ledger accounts. Every resident
// byte is charged to exactly one row; on a single physical pool the rows are
// apertures of the same RAM, so the joint view is TotalCharged, never a sum of
// "free windows" per aperture (MBL-084).
type PoolRowClass string

const (
	// RowDevice is the device-local aperture (HBM) row.
	RowDevice PoolRowClass = "device"
	// RowHost is the pinned host DRAM row. Offloaded shards (Engram-style
	// cpu_offload tables, SGLang placement cpu) count as pool capacity under
	// their own row here instead of being folded into the device row or treated
	// as an unbounded free window.
	RowHost PoolRowClass = "host"
)

// PoolCapacity is the measured per-row ceiling of one physical pool, in bytes.
// Negative values clamp to zero: a negative measurement is no measurement.
type PoolCapacity struct {
	DeviceBytes int64
	HostBytes   int64
}

func (c PoolCapacity) bytes(class PoolRowClass) int64 {
	switch class {
	case RowDevice:
		if c.DeviceBytes < 0 {
			return 0
		}
		return c.DeviceBytes
	case RowHost:
		if c.HostBytes < 0 {
			return 0
		}
		return c.HostBytes
	default:
		return 0
	}
}

// Structural sentinels for ledger misuse. Capacity overruns are NOT structural:
// they refuse with the typed compute.FitError (Verdict FitTooBig) so machine-facing
// callers assert verdict/scope/want/avail, never prose.
var (
	// ErrEmptyOwner refuses a charge without a stable owner identity.
	ErrEmptyOwner = errors.New("kvmmu: pool charge requires a non-empty owner")
	// ErrUnknownRowClass refuses a charge against a row the ledger does not account.
	ErrUnknownRowClass = errors.New("kvmmu: unknown pool row class")
	// ErrNonPositiveBytes refuses a charge of zero or negative bytes.
	ErrNonPositiveBytes = errors.New("kvmmu: pool charge bytes must be positive")
	// ErrDuplicateOwner refuses charging an owner that already holds a charge:
	// one byte, one row, one charge (MBL-087 double-charge guard).
	ErrDuplicateOwner = errors.New("kvmmu: owner already charged to the pool ledger")
)

// PhysicalPoolLedger is the SEAM-01 physical-pool charge/release ledger shared by
// estimator, loader, and serve planner: every resident byte is charged exactly
// once, to exactly one row class, and released when the shard leaves residency.
// Safe for concurrent use.
type PhysicalPoolLedger struct {
	mu       sync.Mutex
	capacity PoolCapacity
	rows     map[PoolRowClass]int64 // used bytes per row
	charges  map[string]poolCharge  // owner -> its single charge
}

type poolCharge struct {
	class PoolRowClass
	bytes int64
}

// NewPhysicalPoolLedger builds a ledger over the measured per-row ceilings.
func NewPhysicalPoolLedger(capacity PoolCapacity) *PhysicalPoolLedger {
	return &PhysicalPoolLedger{
		capacity: capacity,
		// Seed both accounted rows so membership doubles as the closed-row guard;
		// any other class is refused structurally, not silently zero-capped.
		rows:    map[PoolRowClass]int64{RowDevice: 0, RowHost: 0},
		charges: map[string]poolCharge{},
	}
}

// Capacity reports the measured per-row ceilings the ledger was built with.
func (l *PhysicalPoolLedger) Capacity() PoolCapacity {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.capacity
}

// Charge lands one shard's resident bytes on its row. It refuses with a typed
// *compute.FitError (Verdict FitTooBig, Want = post-charge row total, Avail = the
// measured row peak) when the charge would push the row past its ceiling, so a
// load that would OOM the box turns into an answerable refusal before a byte is
// allocated. Structural misuse returns the package sentinels.
func (l *PhysicalPoolLedger) Charge(owner string, class PoolRowClass, bytes int64) error {
	if owner == "" {
		return ErrEmptyOwner
	}
	if bytes <= 0 {
		return ErrNonPositiveBytes
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if _, known := l.rows[class]; !known {
		return ErrUnknownRowClass
	}
	if _, dup := l.charges[owner]; dup {
		return ErrDuplicateOwner
	}
	ceiling := l.capacity.bytes(class)
	used := l.rows[class]
	if bytes > ceiling-used {
		return &compute.FitError{
			Verdict: compute.FitTooBig,
			Want:    used + bytes,
			Avail:   ceiling,
			Scope:   scopeOfRow(class),
		}
	}
	l.rows[class] = used + bytes
	l.charges[owner] = poolCharge{class: class, bytes: bytes}
	return nil
}

// Release retires one owner's charge and returns the freed bytes. Unknown or
// already-released owners return (0, false); releasing never errors.
func (l *PhysicalPoolLedger) Release(owner string) (int64, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	ch, ok := l.charges[owner]
	if !ok {
		return 0, false
	}
	delete(l.charges, owner)
	l.rows[ch.class] -= ch.bytes
	return ch.bytes, true
}

// RowUsed reports the bytes currently charged to one row.
func (l *PhysicalPoolLedger) RowUsed(class PoolRowClass) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.rows[class]
}

// RowFree reports the headroom left on one row (capacity minus used, never negative).
func (l *PhysicalPoolLedger) RowFree(class PoolRowClass) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	free := l.capacity.bytes(class) - l.rows[class]
	if free < 0 {
		return 0
	}
	return free
}

// TotalCharged reports the joint single-pool view: the sum charged across every
// row. On a shared physical pool this, not any per-row "free window", is what the
// kernel actually has to back (MBL-084).
func (l *PhysicalPoolLedger) TotalCharged() int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	var total int64
	for _, used := range l.rows {
		total += used
	}
	return total
}

// Admit is the admission math: it checks a prospective classed demand against the
// measured row ceilings BEFORE any charge lands. Host-scoped demands are judged
// against the host row — host RAM is a requirement, not a free window, so a plan
// that leans on offload is refused exactly like one that leans on HBM. Device-
// scoped (including empty-scope) demands are judged against the device row. Rows
// are checked in fixed order (device, then host); the first overrun refuses with
// the typed *compute.FitError naming that row's post-demand total and measured
// peak. A plan that fits both rows admits with nil.
func (l *PhysicalPoolLedger) Admit(plan compute.MemoryPlan) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, check := range []struct {
		class PoolRowClass
		want  int64
	}{
		{RowDevice, plan.DeviceTotal()},
		{RowHost, plan.HostTotal()},
	} {
		if check.want <= 0 {
			continue
		}
		ceiling := l.capacity.bytes(check.class)
		used := l.rows[check.class]
		if check.want > ceiling-used {
			return &compute.FitError{
				Verdict: compute.FitTooBig,
				Want:    used + check.want,
				Avail:   ceiling,
				Demands: plan,
				Scope:   scopeOfRow(check.class),
			}
		}
	}
	return nil
}

func scopeOfRow(class PoolRowClass) compute.MemoryScope {
	if class == RowHost {
		return compute.MemoryScopeHost
	}
	return compute.MemoryScopeDevice
}
