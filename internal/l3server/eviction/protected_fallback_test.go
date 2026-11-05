package eviction

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestWTinyLFUForcedEvictionReachesProtected(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name  string
		setup func(*testing.T, *WTinyLFU)
		want  []uint64
	}{
		{
			name: "protected only",
			setup: func(_ *testing.T, w *WTinyLFU) {
				w.Remove(2)
			},
			want: []uint64{1},
		},
		{
			name: "window before protected",
			want: []uint64{2, 1},
		},
		{
			name: "probation before window before protected",
			setup: func(_ *testing.T, w *WTinyLFU) {
				w.Admit(3, 13)
			},
			want: []uint64{2, 3, 1},
		},
		{
			name: "protected tail follows recent access",
			setup: func(t *testing.T, w *WTinyLFU) {
				w.Admit(3, 13)
				if !w.Access(2) {
					t.Fatal("second probation entry was not promoted")
				}
				w.Remove(3)
				if !w.Access(1) {
					t.Fatal("protected entry was not refreshed")
				}
			},
			want: []uint64{2, 1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			type event struct {
				hash   uint64
				keyLen uint16
			}
			var events []event
			w := NewWTinyLFU(10, func(hash uint64, keyLen uint16) {
				events = append(events, event{hash: hash, keyLen: keyLen})
			})
			w.Admit(1, 11)
			w.Admit(2, 12)
			if !w.Access(1) || w.protected.size != 1 {
				t.Fatal("first probation entry was not promoted")
			}
			if tc.setup != nil {
				tc.setup(t, w)
			}
			if w.Size() != uint64(len(tc.want)) || len(events) != 0 {
				t.Fatalf("setup size = %d, callbacks = %v", w.Size(), events)
			}
			for i, hash := range tc.want {
				if !w.EvictOne() {
					t.Fatalf("eviction %d stopped with %d entries remaining", i, w.Size())
				}
				wantEvent := event{hash: hash, keyLen: uint16(hash + 10)}
				if len(events) != i+1 || events[i] != wantEvent {
					t.Fatalf("eviction %d callbacks = %v, want next %+v", i, events, wantEvent)
				}
				if _, found := w.keyToNode[hash]; found || w.Size() != uint64(len(tc.want)-i-1) {
					t.Fatalf("eviction %d left key %d tracked or wrong size %d", i, hash, w.Size())
				}
			}
			if w.EvictOne() || w.Size() != 0 || len(events) != len(tc.want) {
				t.Fatalf("empty eviction changed state: size = %d, callbacks = %v", w.Size(), events)
			}
			w.Admit(9, 19)
			if !w.Access(9) || !w.EvictOne() || w.Size() != 0 || len(events) != len(tc.want)+1 || events[len(tc.want)] != (event{hash: 9, keyLen: 19}) {
				t.Fatal("evicted pool entry was not reusable")
			}
		})
	}
}
