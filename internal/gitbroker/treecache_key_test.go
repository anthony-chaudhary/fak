package gitbroker

import "testing"

// THE UNIT-LEVEL PIN ON THE CLASS B CACHE-VALIDITY TEST (#5623).
//
// treeCache holds exactly one entry, so the ONLY thing standing between a reader
// and a tree that no longer exists is the key comparison inside lookup. The
// integration tests drive that through a real repo and a real peer commit, which
// is the right proof that the KEY MOVES when the tree changes. This file proves
// the other half, directly and without git: that lookup actually COMPARES the
// key it is handed against the key the entry was stored under.
//
// The distinction matters because the two halves fail independently. Drop the
// comparison (`if !t.full` with no `|| t.key != key`) and the cache still
// compiles, still passes every positive caching test, and silently answers every
// question with the first tree it ever saw — a wrong answer, not a crash. The
// integration tests do catch that, but only via a multi-second real-git fixture;
// this test names the defect in microseconds and localizes it to one line.

// aTreeKey is a fully-populated, usable StateKey to mutate one axis at a time.
func aTreeKey() StateKey {
	return StateKey{
		IndexMod:  1_700_000_000_000_000_000,
		IndexSize: 4096,
		HeadOID:   "1f75c56d0a2b3c4d5e6f708192a3b4c5d6e7f809",
		RefsMod:   1_700_000_000_000_000_001,
	}
}

// TestTreeCacheLookupMissesOnADifferentKey pins the cache-validity contract on
// every axis of the key.
//
// Each subtest stores an entry under one key and looks it up under a key that
// differs in EXACTLY ONE field. Every one of those must MISS: the key is the
// entire validity test, so an axis that lookup ignores is an axis along which a
// peer's write is invisible. Testing the fields one at a time rather than with a
// single wholly-different key is deliberate — it distinguishes "the comparison
// is missing" from "the comparison is there but reads the wrong field".
func TestTreeCacheLookupMissesOnADifferentKey(t *testing.T) {
	stored := aTreeKey()
	entry := TreeState{Dirty: true, Status: " M seed.txt\n", Key: stored}

	cases := []struct {
		axis string
		why  string
		key  StateKey
	}{
		{
			axis: "IndexMod",
			why:  "a peer rewrote .git/index (any `git add`, any commit)",
			key:  StateKey{IndexMod: stored.IndexMod + 1, IndexSize: stored.IndexSize, HeadOID: stored.HeadOID, RefsMod: stored.RefsMod},
		},
		{
			axis: "IndexSize",
			why:  "the index changed size within one mtime tick",
			key:  StateKey{IndexMod: stored.IndexMod, IndexSize: stored.IndexSize + 1, HeadOID: stored.HeadOID, RefsMod: stored.RefsMod},
		},
		{
			axis: "HeadOID",
			why:  "HEAD moved — a peer committed, or the checkout switched",
			key:  StateKey{IndexMod: stored.IndexMod, IndexSize: stored.IndexSize, HeadOID: "0000000000000000000000000000000000000001", RefsMod: stored.RefsMod},
		},
		{
			axis: "RefsMod",
			why:  "a ref moved under refs/ or packed-refs",
			key:  StateKey{IndexMod: stored.IndexMod, IndexSize: stored.IndexSize, HeadOID: stored.HeadOID, RefsMod: stored.RefsMod + 1},
		},
	}

	for _, tc := range cases {
		t.Run(tc.axis, func(t *testing.T) {
			c := &treeCache{}
			c.store(stored, entry)

			// The entry really is resident, so a miss below is the key
			// comparison talking and not an empty cache.
			if got, ok := c.lookup(stored); !ok || got != entry {
				t.Fatalf("lookup under the STORED key = (%+v, %v), want (%+v, true); this subtest says nothing unless the entry is resident", got, ok, entry)
			}

			got, ok := c.lookup(tc.key)
			if ok {
				t.Fatalf("lookup under a key differing only in %s was a HIT (%+v). %s, so the stored answer describes a tree that no longer exists — the Class B cache is serving a stale working-tree state to every reader", tc.axis, got, tc.why)
			}
			if got != (TreeState{}) {
				t.Fatalf("a miss returned a non-zero TreeState %+v; a miss must hand back nothing at all", got)
			}
		})
	}
}

// TestTreeCacheLookupHitsOnlyTheExactKey is the positive companion: the cache is
// still a cache. Without it, `return TreeState{}, false` unconditionally would
// pass the test above and break every caller instead.
func TestTreeCacheLookupHitsOnlyTheExactKey(t *testing.T) {
	key := aTreeKey()
	entry := TreeState{Dirty: false, Status: "", Key: key}

	c := &treeCache{}
	if _, ok := c.lookup(key); ok {
		t.Fatal("an empty treeCache reported a hit")
	}
	if c.held() {
		t.Fatal("an empty treeCache reports an entry resident")
	}

	c.store(key, entry)
	if !c.held() {
		t.Fatal("treeCache reports no entry resident after store")
	}
	got, ok := c.lookup(key)
	if !ok {
		t.Fatal("lookup under the exact stored key MISSED; the Class B cache would never serve anything")
	}
	if got != entry {
		t.Fatalf("lookup returned %+v, want the stored entry %+v", got, entry)
	}

	// store replaces the single entry, and the replacement rebinds the key: the
	// old key must stop hitting the moment a newer tree is recorded.
	newer := aTreeKey()
	newer.IndexMod++
	newerEntry := TreeState{Dirty: true, Status: "?? peer.txt\n", Key: newer}
	c.store(newer, newerEntry)

	if got, ok := c.lookup(key); ok {
		t.Fatalf("the SUPERSEDED key still hits, returning %+v; treeCache holds one entry, so a store must retire the key it replaced", got)
	}
	if got, ok := c.lookup(newer); !ok || got != newerEntry {
		t.Fatalf("lookup under the newly stored key = (%+v, %v), want (%+v, true)", got, ok, newerEntry)
	}
}
