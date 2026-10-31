//go:build cuda && cgo

package compute

import (
	"reflect"
	"testing"
)

// TestCUDAPagedKVWritePagesMatchesContiguous exercises the actual prompt-panel
// caller: ragged shared-tail COW, >64 pages in one append, both layers and all
// K/Kraw/V planes. A later one-row append retains the original single-page path.
// fak-test:runtime integration est=1s lane=optin
func TestCUDAPagedKVWritePagesMatchesContiguous(t *testing.T) {
	cb := cudaOrSkip(t)
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 120, RopeTheta: 10000}
	a := newPagedForTest(t, cb, cfg)
	defer a.Free()
	refA, refB := cb.newContiguousKV(cfg), cb.newContiguousKV(cfg)
	defer refA.Free()
	defer refB.Free()
	appendRows := func(stores []KVStore, start, rows, seed int) {
		t.Helper()
		w := cfg.NumKVHeads * cfg.HeadDim
		for l := 0; l < cfg.NumLayers; l++ {
			var data [3][]float32
			for plane := range data {
				data[plane] = make([]float32, rows*w)
				for i := range data[plane] {
					data[plane][i] = float32(seed*10000 + l*2000 + plane*1000 + i%997)
				}
			}
			for _, kv := range stores {
				cb.PromptPanelAppendKV(kv, l, mkResident(cb, []int{rows, w}, data[0]),
					mkResident(cb, []int{rows, w}, data[1]), mkResident(cb, []int{rows, w}, data[2]), start, rows)
			}
		}
	}
	appendRows([]KVStore{a, refA, refB}, 0, 14, 1)
	b := a.Clone().(*cudaPagedKV)
	defer b.Free()
	sharedTail := a.BlockTable()[0]
	appendRows([]KVStore{b, refB}, 14, 1050, 2) // one call/layer, not pagedKVFill's 512-row chunks
	if b.BlockTable()[0] == sharedTail || a.BlockTable()[0] != sharedTail || a.Len() != 14 || b.Len() != 1064 {
		t.Fatalf("COW/count contract: a=%v len=%d b=%v len=%d", a.BlockTable(), a.Len(), b.BlockTable(), b.Len())
	}
	appendRows([]KVStore{a, refA}, 14, 1, 3)
	for _, pair := range []struct{ paged, contiguous KVStore }{{a, refA}, {b, refB}} {
		got, err := pair.paged.(KVHostSnapshotter).SnapshotToHost()
		if err != nil {
			t.Fatal(err)
		}
		want, err := pair.contiguous.(KVHostSnapshotter).SnapshotToHost()
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Pos, want.Pos) || !reflect.DeepEqual(got.K, want.K) ||
			!reflect.DeepEqual(got.KRaw, want.KRaw) || !reflect.DeepEqual(got.V, want.V) {
			t.Fatal("actual multi-page append/COW differs from contiguous K/Kraw/V or positions")
		}
	}
}
