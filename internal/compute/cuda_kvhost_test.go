//go:build cuda && cgo

package compute

import (
	"reflect"
	"testing"
)

// FAK_CUDA_GDN_REQUIRED=1 requires exact CUDA registration; no host fallback.
// fak-test:runtime integration est=1s lane=optin
func TestCUDAKVHostRoundTripOwnsCompletePayload(t *testing.T) {
	be := cudaGDNBackend(t)
	cfg := KVConfig{
		NumLayers:  2,
		NumKVHeads: 1,
		HeadDim:    4,
		RopeTheta:  10000,
		Precision:  KVPrecisionF32,
	}
	kv := be.NewKV(cfg)
	width := cfg.NumKVHeads * cfg.HeadDim
	want := KVHostSnapshot{
		Config: cloneKVConfig(cfg), Pos: []int{7, 8, 9},
		K: make([][]float32, cfg.NumLayers), KRaw: make([][]float32, cfg.NumLayers), V: make([][]float32, cfg.NumLayers),
	}
	for pos := 0; pos < 3; pos++ {
		for layer := 0; layer < cfg.NumLayers; layer++ {
			row := func(base float32) []float32 {
				out := make([]float32, width)
				for i := range out {
					out[i] = base + float32(i)/16
				}
				return out
			}
			rawRow, ropeRow, valueRow := row(float32(100*layer+10*pos+1)), row(float32(100*layer+10*pos+2)), row(float32(100*layer+10*pos+3))
			want.KRaw[layer] = append(want.KRaw[layer], rawRow...)
			want.K[layer] = append(want.K[layer], ropeRow...)
			want.V[layer] = append(want.V[layer], valueRow...)
			kRaw := mkResident(be, []int{width}, rawRow)
			kRoPE := mkResident(be, []int{width}, ropeRow)
			value := mkResident(be, []int{width}, valueRow)
			kv.AppendKV(layer, kRaw, kRoPE, value, pos+7)
			be.Free(kRaw)
			be.Free(kRoPE)
			be.Free(value)
		}
	}

	host, err := SnapshotKVToHost(kv)
	if err != nil {
		kv.Free()
		t.Fatalf("snapshot CUDA KV to host: %v", err)
	}
	if !reflect.DeepEqual(host, want) {
		kv.Free()
		t.Fatalf("snapshot changed uploaded K/Kraw/V or positions: got %#v want %#v", host, want)
	}
	if host.TransferBytes() <= 0 || host.ResidentBytes() <= host.TransferBytes() {
		kv.Free()
		t.Fatalf("host accounting transfer=%d resident=%d", host.TransferBytes(), host.ResidentBytes())
	}
	hostMetadataBytes := host.ResidentBytes() - host.TransferBytes()
	if got, want := int64(kv.ResidentBytes())-hostMetadataBytes, host.TransferBytes(); got != want {
		kv.Free()
		t.Fatalf("device payload bytes=%d, want host-transfer bytes=%d", got, want)
	}

	kv.Free()
	restored, err := RestoreKVFromHost(be, host)
	if err != nil {
		t.Fatalf("restore CUDA KV from host: %v", err)
	}
	defer restored.Free()
	host.Pos[0] = -1
	host.K[0][0], host.KRaw[0][0], host.V[0][0] = -1, -2, -3
	roundTrip, err := SnapshotKVToHost(restored)
	if err != nil {
		t.Fatalf("snapshot restored CUDA KV: %v", err)
	}
	if !reflect.DeepEqual(roundTrip, want) {
		t.Fatalf("D2H/H2D round trip changed complete KV payload:\n got %#v\nwant %#v", roundTrip, want)
	}
}
