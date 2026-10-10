package model

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestPagedKVRestoreFromHostOwnership(t *testing.T) {
	t.Parallel()
	cfg := Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1}
	sourcePool := NewPagedKVPoolWithRaw(cfg, 1)
	source := sourcePool.NewSequence()
	for _, value := range []float32{11, 22} {
		source.AppendRaw([][]float32{{value}}, [][]float32{{value + 1}}, [][]float32{{-value}})
	}
	blob, err := source.SwapToHost()
	if err != nil {
		t.Fatal(err)
	}
	source.Free()
	const headerBytes = len(pagedKVSwapMagic) + 6*4
	blockBytes := sourcePool.blockFloats() * 4
	for _, tc := range []struct {
		name       string
		data       []byte
		wantBlocks int
		wantError  bool
	}{
		{"partial_first_block", blob[:headerBytes+4], 2, true},
		{"partial_second_block", blob[:headerBytes+blockBytes+4], 3, true},
		{"trailing_bytes", append(bytes.Clone(blob), 0), 3, true},
		{"success", blob, 3, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := NewPagedKVPoolWithRaw(cfg, 1)
			victim := pool.NewSequence()
			victim.AppendRaw([][]float32{{71}}, [][]float32{{72}}, [][]float32{{-71}})
			defer victim.Free()
			if victim.table[0] != 0 {
				t.Fatal("fixture must keep block zero owned by the victim")
			}
			before, err := victim.SwapToHost()
			if err != nil {
				t.Fatal(err)
			}
			// Restore reuses block one while block zero remains live.
			pool.Release(pool.Alloc())
			restored, err := pool.RestoreFromHost(tc.data)
			if tc.wantError {
				if err == nil || restored != nil {
					t.Fatalf("failed restore returned (%v, %v), want (nil, error)", restored, err)
				}
			} else {
				if err != nil || restored == nil {
					t.Fatalf("successful restore returned (%v, %v)", restored, err)
				}
				after, err := restored.SwapToHost()
				if err != nil || !bytes.Equal(after, blob) {
					t.Fatalf("successful restore changed the serialized payload: %v", err)
				}
				for _, id := range restored.table {
					if id == 0 || pool.ref[id] != 1 {
						t.Fatalf("restored block %d has invalid ownership: refs=%v", id, pool.ref)
					}
				}
				restored.Free()
			}
			if pool.TotalBlocks() != tc.wantBlocks {
				t.Errorf("allocated %d blocks, want %d", pool.TotalBlocks(), tc.wantBlocks)
			}
			for id, ref := range pool.ref {
				want := 0
				if id == 0 {
					want = 1
				}
				if ref != want {
					t.Errorf("block %d refs=%d, want %d", id, ref, want)
				}
			}
			if pool.PhysicalBlocks() != 1 || pool.FreeBlocks() != tc.wantBlocks-1 {
				t.Errorf("unwind left live/free=%d/%d, want 1/%d", pool.PhysicalBlocks(), pool.FreeBlocks(), tc.wantBlocks-1)
			}
			// Reallocation must not clear or overwrite the victim's live payload.
			next := pool.NewSequence()
			next.AppendRaw([][]float32{{91}}, [][]float32{{92}}, [][]float32{{-91}})
			defer next.Free()
			if next.table[0] == 0 {
				t.Error("reallocation took the victim's live block")
			}
			after, err := victim.SwapToHost()
			if err != nil || !bytes.Equal(after, before) {
				t.Errorf("victim payload changed after restore and reallocation: %v", err)
			}
		})
	}
}
