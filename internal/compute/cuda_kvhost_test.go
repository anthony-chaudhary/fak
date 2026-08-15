//go:build cuda

package compute

import (
	"reflect"
	"testing"
)

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
	for pos := 0; pos < 3; pos++ {
		for layer := 0; layer < cfg.NumLayers; layer++ {
			row := func(base float32) []float32 {
				out := make([]float32, width)
				for i := range out {
					out[i] = base + float32(i)/16
				}
				return out
			}
			kRaw := mkResident(be, []int{width}, row(float32(100*layer+10*pos+1)))
			kRoPE := mkResident(be, []int{width}, row(float32(100*layer+10*pos+2)))
			value := mkResident(be, []int{width}, row(float32(100*layer+10*pos+3)))
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
	roundTrip, err := SnapshotKVToHost(restored)
	if err != nil {
		t.Fatalf("snapshot restored CUDA KV: %v", err)
	}
	if !reflect.DeepEqual(roundTrip, host) {
		t.Fatalf("D2H/H2D round trip changed complete KV payload:\n got %#v\nwant %#v", roundTrip, host)
	}
}
