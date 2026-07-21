package radixkv

import (
	"errors"
	"sort"
)

// retention_request.go — CLIENT-DECLARED per-request KV retention as (priority, TTL-window),
// the pure ordering core for #5259 (INSPIRE borrow, clean-room, from NVIDIA/TensorRT-LLM's
// per-request KV retention config and its token-range retention band at
// cpp/include/tensorrt_llm/executor/executor.h:589-617@f4c5c935, Apache-2.0). Unlike the
// telemetry-derived victim rule in eviction_strategy.go (seg/cost/age scored from what the
// tree OBSERVES), this is a channel for the CALLER to declare "retain this at priority P for
// a bounded window T" up front. The window is measured in LOGICAL time (ticks / token index),
// never wall-clock — the clock is INJECTED as a parameter to every decision, so the reclaim
// verdict is deterministic and replayable. This file is the pure descriptor + ordering only;
// wiring the resulting order into the tree's budget-pressure loop is a follow-on (it maps to a
// new top-precedence victimKey.seg tier, leaving telemetry keys as the within-band tie-break).
//
// Scale mirrors the borrow: priority runs MinRetentionPriority(0)..MaxRetentionPriority(100),
// with DefaultRetentionPriority(35) as the "age normally" baseline. A HIGHER priority is
// retained more strongly; among live entries the LOWEST priority is reclaimed first.

const (
	// MinRetentionPriority is the weakest retention: an entry declared at this priority is
	// reclaimed before any higher-priority live entry.
	MinRetentionPriority = 0
	// MaxRetentionPriority is the strongest retention: a live entry at this priority is
	// reclaimed only after every lower-priority live entry is gone.
	MaxRetentionPriority = 100
	// DefaultRetentionPriority is the neutral "age normally" baseline a caller gets when it
	// declares no explicit priority — the point a TTL demotion returns an entry to.
	DefaultRetentionPriority = 35
)

// retainForever is the TTL sentinel meaning "never expire on a window basis": an entry with
// this TTL is past its window at no finite clock value, so only priority governs its reclaim.
const retainForever int64 = 0

// RetentionRequest is a client-declared per-request KV retention descriptor: a (priority,
// TTL-window) pair over LOGICAL time. Priority ranks the entry against other live entries for
// reclaim order (lower reclaims first). TTL is the window length in ticks measured from
// Admitted; a TTL of 0 means never expire on the window (retain until only priority decides).
// Admitted is the logical tick the entry entered retention — it both anchors the TTL window
// and serves as the deterministic older-first tie-break among equal-priority live entries.
type RetentionRequest struct {
	// Priority is the retention strength in MinRetentionPriority..MaxRetentionPriority.
	Priority int
	// TTL is the window length in logical ticks; 0 (retainForever) means never expire.
	TTL int64
	// Admitted is the logical tick the entry entered retention (window anchor + tie-break).
	Admitted int64
}

var (
	// ErrRetentionPriorityRange is returned when a declared priority is outside the
	// MinRetentionPriority..MaxRetentionPriority scale — validation fails CLOSED rather than
	// clamping a caller's out-of-range intent to a silently different retention.
	ErrRetentionPriorityRange = errors.New("radixkv: retention priority out of range [0,100]")
	// ErrRetentionTTLNegative is returned when a declared TTL window is negative — a window
	// cannot run backwards, so validation fails CLOSED.
	ErrRetentionTTLNegative = errors.New("radixkv: retention TTL window is negative")
	// ErrRetentionAdmittedNegative is returned when the admission tick is negative — logical
	// time is non-negative, so validation fails CLOSED.
	ErrRetentionAdmittedNegative = errors.New("radixkv: retention admission tick is negative")
)

// Validate reports whether the descriptor is well-formed, failing CLOSED on any out-of-range
// field (negative TTL/admission, priority outside the scale). A caller must validate before
// admitting a request; the ordering functions below assume validated inputs.
func (r RetentionRequest) Validate() error {
	if r.Priority < MinRetentionPriority || r.Priority > MaxRetentionPriority {
		return ErrRetentionPriorityRange
	}
	if r.TTL < 0 {
		return ErrRetentionTTLNegative
	}
	if r.Admitted < 0 {
		return ErrRetentionAdmittedNegative
	}
	return nil
}

// windowEnd returns the logical tick strictly after which the entry is past its TTL window,
// plus whether the window is finite. A retainForever TTL has no finite end (ever == false).
func (r RetentionRequest) windowEnd() (end int64, ever bool) {
	if r.TTL == retainForever {
		return 0, false
	}
	return r.Admitted + r.TTL, true
}

// Expired reports whether the entry is past its TTL window at the INJECTED logical clock now.
// An entry is expired once the clock advances strictly beyond Admitted+TTL; a retainForever
// entry is never expired on the window (only priority can reclaim it). Wall-clock-free: the
// caller passes the logical time, so the verdict is deterministic and replayable.
func (r RetentionRequest) Expired(now int64) bool {
	end, ever := r.windowEnd()
	if !ever {
		return false
	}
	return now > end
}

// RetentionEntry binds a stable identity to a RetentionRequest so a set of retained entries
// can be ordered deterministically. ID is the caller's block/prefix handle; it is the final
// total-order tie-break, so two entries with identical priority and admission still reclaim in
// a fixed, replayable order.
type RetentionEntry struct {
	ID string
	RetentionRequest
}

// reclaimKey is the total-order reclaim priority among LIVE entries: the entry with the LEAST
// key is reclaimed first. Fields compare lexicographically, the same tuple-ordering shape as
// victimKey in eviction_strategy.go:
//
//   - priority ascending: the weakest-retained live entry is reclaimed before any stronger
//     one, so a MaxRetentionPriority entry survives while lower-priority entries are reclaimed.
//   - admitted ascending: on an equal priority the OLDER (smaller admission tick) entry is
//     reclaimed first — the deterministic logical-time tie-break.
//   - id ascending: the final lexicographic tie-break, so the order is total and replayable
//     even when priority and admission are identical.
type reclaimKey struct {
	priority int
	admitted int64
	id       string
}

// less reports whether k is reclaimed before o (k is the weaker-retained, reclaim-first key).
func (k reclaimKey) less(o reclaimKey) bool {
	if k.priority != o.priority {
		return k.priority < o.priority
	}
	if k.admitted != o.admitted {
		return k.admitted < o.admitted
	}
	return k.id < o.id
}

func (e RetentionEntry) reclaimKey() reclaimKey {
	return reclaimKey{priority: e.Priority, admitted: e.Admitted, id: e.ID}
}

// ReclaimVerdict is the deterministic partition of a set of retained entries against an
// injected logical clock: Expired holds every entry past its TTL window (immediately
// reclaim-eligible), and Live holds the remainder in RECLAIM ORDER — Live[0] is reclaimed
// first (weakest retention), Live[len-1] last (strongest). Both slices are stably ordered.
type ReclaimVerdict struct {
	// Expired lists the entries past their TTL window, ordered by reclaim order for a stable
	// batch reclaim (weakest-retained expired entry first).
	Expired []RetentionEntry
	// Live lists the still-in-window entries in reclaim order (index 0 reclaims first).
	Live []RetentionEntry
}

// ReclaimOrder partitions entries into TTL-expired vs live at the injected logical clock now,
// and orders each partition by ascending reclaim key (weakest retention first: lower priority,
// then older admission, then ID). It is pure and deterministic — no wall-clock, no GPU, no
// network — so the same inputs always yield the same verdict. Inputs are assumed validated
// (call Validate first); ReclaimOrder does not itself re-check ranges.
func ReclaimOrder(entries []RetentionEntry, now int64) ReclaimVerdict {
	var v ReclaimVerdict
	for _, e := range entries {
		if e.Expired(now) {
			v.Expired = append(v.Expired, e)
		} else {
			v.Live = append(v.Live, e)
		}
	}
	sortByReclaimKey(v.Expired)
	sortByReclaimKey(v.Live)
	return v
}

// sortByReclaimKey stably orders entries so the first is reclaimed first (least reclaimKey).
func sortByReclaimKey(entries []RetentionEntry) {
	sort.SliceStable(entries, func(i, j int) bool {
		return entries[i].reclaimKey().less(entries[j].reclaimKey())
	})
}
