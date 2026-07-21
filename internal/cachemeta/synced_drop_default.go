package cachemeta

// synced_drop_default.go — fail-open-until-synced drop default for a managed KV
// store control plane (lmcache-study #5265, LMCache @e38ee415).
//
// A managed KV store that has NOT yet re-synced its authoritative present-set
// from the external control source has INCOMPLETE knowledge: it does not yet
// know which blocks the fleet still holds. If such a not-yet-synced plane
// treated "unknown block" as "safe to drop," a control-plane restart would
// mass-drop every block whose presence had not yet re-synced — the exact reboot
// hazard LMCache's quota resolver guards against (a freshly booted coordinator
// resolves unknown tenants to exempt, not to zero-budget, until the controller
// arms deny post-sync).
//
// This file is the deterministic, wall-clock-free core of that invariant. It
// keys the drop judgement on a single sync flag:
//
//   - UNSYNCED (default): fail OPEN. Every block is kept — a block that is not
//     known-present is still treated as must-keep, because pre-sync state is
//     incomplete and declaring a block droppable could be wrong. Degenerate /
//     empty pre-sync state keeps everything.
//   - SYNCED (armed): normal judgement. The authoritative present-set is now
//     trusted, so a block absent from it is droppable and a block present in it
//     is kept.
//
// It reads no clock and moves no bytes: the verdict is a pure function of the
// injected store shape and the block key, so a replay reproduces it exactly.

// ManagedStore is the field-only shape of a managed KV store's drop control
// plane at one instant: whether it has re-synced its authoritative present-set,
// and (once synced) which block keys that set holds. No transport is implied;
// the caller fills Present from whatever authoritative source it re-synced.
type ManagedStore struct {
	// Synced reports whether the authoritative Present set has been re-synced
	// from the external control source. It is the arm flag: while false the
	// plane fails OPEN (keeps everything); once true the plane trusts Present.
	Synced bool
	// Present is the authoritative set of block keys the plane knows are held,
	// meaningful only once Synced is true. A nil or empty set is legal — once
	// synced it simply means nothing is held (everything absent is droppable).
	Present map[string]bool
}

// DropVerdict is the resolved judgement for one block key: whether the plane
// may drop it, and a stable metric-readable reason. Droppable is false whenever
// the plane fails open, so an unsynced plane never yields a droppable verdict.
type DropVerdict struct {
	Droppable bool
	Reason    string
}

const (
	// reasonUnsyncedKeep — the plane has not re-synced, so it fails open and
	// keeps the block regardless of the present-set.
	reasonUnsyncedKeep = "unsynced_fail_open_keep"
	// reasonPresentKeep — synced and the block is in the authoritative set.
	reasonPresentKeep = "synced_present_keep"
	// reasonAbsentDroppable — synced and the block is absent from the
	// authoritative set, so it is safe to drop.
	reasonAbsentDroppable = "synced_absent_droppable"
)

// ResolveDrop yields the fail-open-until-synced verdict for one block key. While
// the store is UNSYNCED it fails open: the verdict is always keep (never
// droppable), so a not-yet-synced plane cannot drop anything — including blocks
// it has never heard of and including the degenerate empty-state case. Once the
// store is SYNCED the authoritative Present set governs: a block in the set is
// kept, a block absent from it is droppable.
func ResolveDrop(store ManagedStore, block string) DropVerdict {
	if !store.Synced {
		return DropVerdict{Droppable: false, Reason: reasonUnsyncedKeep}
	}
	if store.Present[block] {
		return DropVerdict{Droppable: false, Reason: reasonPresentKeep}
	}
	return DropVerdict{Droppable: true, Reason: reasonAbsentDroppable}
}

// PartitionDroppable splits a set of block keys into the keys to keep and the
// keys safe to drop under the store's current sync state. An unsynced store
// fails open — every key lands in keep and droppable is empty. A synced store
// keeps the keys its authoritative set holds and marks the rest droppable. Both
// results are order-preserving over the input; nil input yields two nil slices.
func PartitionDroppable(store ManagedStore, blocks []string) (keep, droppable []string) {
	for _, b := range blocks {
		if ResolveDrop(store, b).Droppable {
			droppable = append(droppable, b)
		} else {
			keep = append(keep, b)
		}
	}
	return keep, droppable
}

// Unsynced returns the default ManagedStore: not yet re-synced, so it fails open
// and keeps everything. This is the boot-time posture — the plane starts here
// and only leaves it by an explicit Arm once the authoritative set has synced.
func Unsynced() ManagedStore {
	return ManagedStore{Synced: false}
}

// Arm returns a SYNCED store over the given authoritative present-set: the
// explicit transition an unsynced plane makes once its present-set has re-synced
// from the control source. After Arm, blocks absent from present become
// droppable — the deny arm that a not-yet-synced plane deliberately withheld.
func (s ManagedStore) Arm(present map[string]bool) ManagedStore {
	return ManagedStore{Synced: true, Present: present}
}
