package eviction

import (
	"fmt"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestWTinyLFUVictimHashIsNotSuccessSentinel(t *testing.T) {
	t.Parallel()
	type victim struct {
		hash   uint64
		keyLen uint16
	}
	for _, hash := range []uint64{0, 17} {
		for _, segment := range []string{"window", "probation", "protected"} {
			t.Run(fmt.Sprintf("forced/%s/%d", segment, hash), func(t *testing.T) {
				var callbacks []victim
				w := NewWTinyLFU(10, func(h uint64, n uint16) {
					callbacks = append(callbacks, victim{h, n})
				})
				w.Admit(hash, 23)
				if segment != "window" {
					w.Admit(99, 2)
					if segment == "protected" && !w.Access(hash) {
						t.Fatal("failed to promote tracked key")
					}
					w.Remove(99)
				}
				if !w.EvictOne() {
					t.Fatal("eviction reported empty for a tracked victim")
				}
				if len(callbacks) != 1 || callbacks[0] != (victim{hash, 23}) {
					t.Fatalf("callbacks = %v, want one exact victim", callbacks)
				}
				if w.Size() != 0 || len(w.keyToNode) != 0 {
					t.Fatalf("victim remains tracked: size=%d map=%v", w.Size(), w.keyToNode)
				}
				if w.EvictOne() || len(callbacks) != 1 {
					t.Fatal("empty eviction reported success or invoked callback")
				}
			})
		}
		for _, mode := range []string{"candidate-rejected", "victim-replaced", "pool-full"} {
			t.Run(fmt.Sprintf("admission/%s/%d", mode, hash), func(t *testing.T) {
				var callbacks []victim
				w := NewWTinyLFU(4, func(h uint64, n uint16) {
					callbacks = append(callbacks, victim{h, n})
				})
				if mode == "pool-full" {
					// A consistent tight pool exercises the defensive allocation fallback
					// without relying on the constructor's temporary overflow slot.
					w.pool = w.pool[:4]
					w.pool[3].freeNext = -1
				}
				order := []uint64{hash, 1, 2, 3}
				if mode == "candidate-rejected" {
					order = []uint64{1, 2, 3, hash}
				}
				for _, key := range order {
					if got := w.Admit(key, uint16(20+key)); len(got) != 0 {
						t.Fatalf("setup admitted %d but reported victims %v", key, got)
					}
				}
				if mode == "victim-replaced" && !w.Access(3) {
					t.Fatal("failed to increase candidate frequency")
				}
				got := w.Admit(4, 24)
				if len(got) != 1 || got[0] != hash {
					t.Fatalf("victims = %v, want [%d]", got, hash)
				}
				if len(callbacks) != 1 || callbacks[0] != (victim{hash, uint16(20 + hash)}) {
					t.Fatalf("callbacks = %v, want exact reported victim", callbacks)
				}
				if _, exists := w.keyToNode[hash]; exists {
					t.Fatal("reported victim remains tracked")
				}
				if w.Size() != 4 || len(w.keyToNode) != 4 {
					t.Fatalf("size=%d map entries=%d, want four", w.Size(), len(w.keyToNode))
				}
				if got := w.Admit(4, 24); len(got) != 0 || len(callbacks) != 1 {
					t.Fatalf("duplicate admission evicted again: %v", got)
				}
			})
		}
	}
	t.Run("nil-callback", func(t *testing.T) {
		w := NewWTinyLFU(4, nil)
		w.Admit(0, 1)
		if !w.EvictOne() || w.EvictOne() || w.Size() != 0 {
			t.Fatal("zero-hash result depends on callback presence")
		}
	})
}
