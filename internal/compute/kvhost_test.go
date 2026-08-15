package compute

import (
	"reflect"
	"testing"
)

func TestKVHostSnapshotRoundTripPreservesKRawAndPositions(t *testing.T) {
	be := Default()
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, RopeTheta: 10000}
	kv := be.NewKV(cfg)
	for pos := 0; pos < 3; pos++ {
		for layer := 0; layer < cfg.NumLayers; layer++ {
			base := float32(100*layer + 10*pos)
			raw := NewF32(be, []int{2}, []float32{base + 1, base + 2})
			rope := NewF32(be, []int{2}, []float32{base + 3, base + 4})
			value := NewF32(be, []int{2}, []float32{base + 5, base + 6})
			kv.AppendKV(layer, raw, rope, value, pos+7)
		}
	}

	host, err := SnapshotKVToHost(kv)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := host.ResidentBytes(), int64(2*3*2*3*4+3*8); got != want {
		t.Fatalf("host resident bytes = %d, want %d", got, want)
	}
	if got, want := host.TransferBytes(), int64(2*3*2*3*4); got != want {
		t.Fatalf("host transfer bytes = %d, want %d", got, want)
	}

	restored, err := RestoreKVFromHost(be, host)
	if err != nil {
		t.Fatal(err)
	}
	defer restored.Free()
	roundTrip, err := SnapshotKVToHost(restored)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(roundTrip, host) {
		t.Fatalf("KV host round trip drifted:\n got=%+v\nwant=%+v", roundTrip, host)
	}
}

type opaqueKV struct{ KVStore }

func TestSnapshotKVToHostRefusesStoreWithoutCompleteKRawSeam(t *testing.T) {
	kv := opaqueKV{KVStore: Default().NewKV(KVConfig{})}
	if _, err := SnapshotKVToHost(kv); err != ErrKVHostSnapshotUnsupported {
		t.Fatalf("error = %v, want %v", err, ErrKVHostSnapshotUnsupported)
	}
}

type noBulkKVHostBackend struct{ Backend }

func TestRestoreKVFromHostRequiresBulkSeamForSparseLayerZero(t *testing.T) {
	state := KVHostSnapshot{
		Config: KVConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2},
		Pos:    []int{4},
		K:      [][]float32{nil, {1, 2}},
		KRaw:   [][]float32{nil, {3, 4}},
		V:      [][]float32{nil, {5, 6}},
	}
	_, err := RestoreKVFromHost(noBulkKVHostBackend{Backend: Default()}, state)
	if err == nil {
		t.Fatal("sparse layer-0 restore used AppendKV and silently lost positions")
	}
}
