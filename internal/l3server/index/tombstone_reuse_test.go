package index

import "testing"

// fak-test:runtime fast est=1ms lane=default
// Estimate only: bounded in-memory tables, with no allocator or server startup.
func TestSwissTableTombstoneReuse(t *testing.T) {
	t.Parallel()
	t.Run("reuse and empty insertion", func(t *testing.T) {
		tbl := NewTable(64)
		const first, replacement, separate = uint64(0x11), uint64(0x22), uint64(32<<7 | 0x33)
		idx, _ := tbl.Insert(first, Entry{KeyLen: 1, ValueOffset: 7})
		if !tbl.Delete(first, 1) || tbl.Tombstones() != 1 {
			t.Fatal("delete did not create one tombstone")
		}
		other, _ := tbl.Insert(separate, Entry{KeyLen: 1, ValueOffset: 8})
		if other == idx || tbl.Tombstones() != 1 {
			t.Fatal("empty insertion consumed an unrelated tombstone")
		}
		got, grew := tbl.Insert(replacement, Entry{KeyLen: 1, ValueOffset: 9})
		if got != idx || grew || tbl.Tombstones() != 0 || tbl.Count() != 2 {
			t.Fatalf("reuse: slot=%d want=%d grew=%t tombstones=%d count=%d", got, idx, grew, tbl.Tombstones(), tbl.Count())
		}
		for hash, want := range map[uint64]uint64{replacement: 9, separate: 8} {
			e, _, ok := tbl.Lookup(hash, 1)
			if !ok || e.KeyHash != hash || e.ValueOffset != want {
				t.Fatalf("entry %x changed: %+v found=%t", hash, e, ok)
			}
		}
		if _, _, ok := tbl.Lookup(first, 1); ok {
			t.Fatal("deleted key reappeared")
		}
		got, grew = tbl.Insert(replacement, Entry{KeyHash: replacement, KeyLen: 1, ValueOffset: 10})
		if got != idx || grew || tbl.Tombstones() != 0 || tbl.Count() != 2 {
			t.Fatal("existing-key update changed accounting")
		}
	})
	t.Run("churn does not compact phantom tombstones", func(t *testing.T) {
		tbl := NewTable(64)
		const hash = uint64(0x11)
		tbl.Insert(hash, Entry{KeyLen: 1})
		backing := &tbl.ctrl[0]
		for i := 0; i < 64; i++ {
			if !tbl.Delete(hash, 1) {
				t.Fatalf("delete failed at cycle %d", i)
			}
			_, grew := tbl.Insert(hash, Entry{KeyLen: 1})
			if grew || tbl.Count() != 1 || tbl.Tombstones() != 0 || tbl.Capacity() != 64 || &tbl.ctrl[0] != backing {
				t.Fatalf("churn changed table at cycle %d: grew=%t count=%d tombstones=%d capacity=%d", i, grew, tbl.Count(), tbl.Tombstones(), tbl.Capacity())
			}
		}
	})
	t.Run("real tombstones still compact", func(t *testing.T) {
		tbl := NewTable(64)
		for i := uint64(0); i < 18; i++ {
			tbl.Insert(i<<7|1, Entry{KeyLen: 1, ValueOffset: i + 1})
		}
		for i := uint64(0); i < 17; i++ {
			if !tbl.Delete(i<<7|1, 1) {
				t.Fatalf("delete failed for %d", i)
			}
		}
		if tbl.Tombstones() != 17 {
			t.Fatalf("real tombstone count=%d, want 17", tbl.Tombstones())
		}
		backing := &tbl.ctrl[0]
		_, grew := tbl.Insert(0x7fff, Entry{KeyLen: 1, ValueOffset: 99})
		if grew || tbl.Tombstones() != 0 || tbl.Count() != 2 || tbl.Capacity() != 64 || &tbl.ctrl[0] == backing {
			t.Fatal("genuine tombstones no longer trigger same-capacity compaction")
		}
		for hash, want := range map[uint64]uint64{17<<7 | 1: 18, 0x7fff: 99} {
			e, _, ok := tbl.Lookup(hash, 1)
			if !ok || e.ValueOffset != want {
				t.Fatalf("compaction lost %x: %+v found=%t", hash, e, ok)
			}
		}
	})
}
