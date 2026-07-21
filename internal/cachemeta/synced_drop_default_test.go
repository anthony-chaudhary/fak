package cachemeta

import (
	"reflect"
	"testing"
)

// TestResolveDropUnsyncedFailsOpen — pre-sync, a block that is NOT known-present
// is still kept (fail open, never droppable), even against a nil present-set.
func TestResolveDropUnsyncedFailsOpen(t *testing.T) {
	store := Unsynced()
	for _, block := range []string{"blk-known", "blk-unknown", ""} {
		v := ResolveDrop(store, block)
		if v.Droppable {
			t.Fatalf("unsynced store must fail open (keep) for %q, got droppable", block)
		}
		if v.Reason != reasonUnsyncedKeep {
			t.Fatalf("block %q: reason = %q, want %q", block, v.Reason, reasonUnsyncedKeep)
		}
	}
}

// TestResolveDropSyncedAbsentIsDroppable — post-sync, a block absent from the
// authoritative present-set is droppable.
func TestResolveDropSyncedAbsentIsDroppable(t *testing.T) {
	store := Unsynced().Arm(map[string]bool{"blk-a": true})
	v := ResolveDrop(store, "blk-gone")
	if !v.Droppable {
		t.Fatalf("synced+absent must be droppable, got kept (%q)", v.Reason)
	}
	if v.Reason != reasonAbsentDroppable {
		t.Fatalf("reason = %q, want %q", v.Reason, reasonAbsentDroppable)
	}
}

// TestResolveDropSyncedPresentIsKept — post-sync, a block present in the
// authoritative set is kept.
func TestResolveDropSyncedPresentIsKept(t *testing.T) {
	store := Unsynced().Arm(map[string]bool{"blk-a": true})
	v := ResolveDrop(store, "blk-a")
	if v.Droppable {
		t.Fatalf("synced+present must be kept, got droppable")
	}
	if v.Reason != reasonPresentKeep {
		t.Fatalf("reason = %q, want %q", v.Reason, reasonPresentKeep)
	}
}

// TestSyncFlagFlipsBehavior — the same block against the same present-set flips
// from kept (unsynced) to droppable (synced) purely on the sync flag.
func TestSyncFlagFlipsBehavior(t *testing.T) {
	present := map[string]bool{"blk-held": true}
	block := "blk-absent"

	pre := ManagedStore{Synced: false, Present: present}
	if ResolveDrop(pre, block).Droppable {
		t.Fatal("pre-sync: absent block must be kept (fail open)")
	}

	post := ManagedStore{Synced: true, Present: present}
	if !ResolveDrop(post, block).Droppable {
		t.Fatal("post-sync: absent block must be droppable")
	}
}

// TestDegenerateEmptyPreSyncFailsOpen — an empty/nil present-set pre-sync keeps
// EVERYTHING (the reboot-must-not-mass-drop invariant); the same empty set
// post-sync makes every block droppable (deny armed).
func TestDegenerateEmptyPreSyncFailsOpen(t *testing.T) {
	blocks := []string{"b1", "b2", "b3"}

	preKeep, preDrop := PartitionDroppable(Unsynced(), blocks)
	if len(preDrop) != 0 {
		t.Fatalf("empty pre-sync must drop nothing, dropped %v", preDrop)
	}
	if !reflect.DeepEqual(preKeep, blocks) {
		t.Fatalf("empty pre-sync must keep all, kept %v", preKeep)
	}

	postKeep, postDrop := PartitionDroppable(Unsynced().Arm(nil), blocks)
	if len(postKeep) != 0 {
		t.Fatalf("empty synced set must keep nothing, kept %v", postKeep)
	}
	if !reflect.DeepEqual(postDrop, blocks) {
		t.Fatalf("empty synced set must drop all absent, dropped %v", postDrop)
	}
}

// TestPartitionDroppableSynced — a synced store splits a mixed set into the held
// keep side and the absent droppable side, order-preserving.
func TestPartitionDroppableSynced(t *testing.T) {
	store := Unsynced().Arm(map[string]bool{"keep-1": true, "keep-2": true})
	keep, drop := PartitionDroppable(store, []string{"keep-1", "drop-1", "keep-2", "drop-2"})
	if !reflect.DeepEqual(keep, []string{"keep-1", "keep-2"}) {
		t.Fatalf("keep = %v", keep)
	}
	if !reflect.DeepEqual(drop, []string{"drop-1", "drop-2"}) {
		t.Fatalf("drop = %v", drop)
	}
}
