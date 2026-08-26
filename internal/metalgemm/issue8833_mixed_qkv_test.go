package metalgemm

import (
	"sync"
	"testing"
)

func TestIssue8833PortableTypedUnavailable(t *testing.T) {
	err := MixedQKVUnavailable("native owner unavailable")
	if !IsMixedQKVDecline(err) {
		t.Fatalf("not typed decline: %v", err)
	}
	if e := err.(*MixedQKVError); e.CallID == 0 || e.Stage != MixedQKVDeclined {
		t.Fatalf("bad decline: %+v", e)
	}
}

func TestIssue8833ConcurrentCallIDsUnique(t *testing.T) {
	const n = 256
	ids := make(chan MixedQKVCallID, n)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() { defer wg.Done(); ids <- MixedQKVUnavailable("test").(*MixedQKVError).CallID }()
	}
	wg.Wait()
	close(ids)
	seen := make(map[MixedQKVCallID]bool, n)
	for id := range ids {
		if id == 0 || seen[id] {
			t.Fatalf("non-unique call id %d", id)
		}
		seen[id] = true
	}
	if len(seen) != n {
		t.Fatalf("got %d IDs", len(seen))
	}
}
