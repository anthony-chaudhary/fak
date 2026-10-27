//go:build cuda && cgo

package compute

import (
	"math"
	"reflect"
	"testing"
)

// Physical NVIDIA witness only. FAK_CUDA_GDN_REQUIRED=1 forbids a skipped backend.
// fak-test:runtime integration est=1s lane=optin
func TestCUDAKV16Lifecycle(t *testing.T) {
	be := cudaGDNBackend(t)
	for _, precision := range []KVPrecision{KVPrecisionF16, KVPrecisionBF16} {
		t.Run(precision.String(), func(t *testing.T) {
			cfg := KVConfig{NumLayers: 1, NumKVHeads: 2, HeadDim: 256, RopeTheta: 10000, Precision: precision}
			kv := be.NewKV(cfg)
			defer kv.Free()
			width := cfg.NumKVHeads * cfg.HeadDim
			var keep [][]float32
			appendRow := func(store KVStore, source, pos int) {
				raw, values := make([]float32, width), make([]float32, width)
				for i := range raw {
					raw[i] = float32(math.Sin(float64(source*width+i))) * 0.25
					values[i] = float32((source+i)%17-8) / 32
				}
				keep = append(keep, raw, values)
				r := mkResident(be, []int{width}, raw)
				v := mkResident(be, []int{width}, values)
				rope := be.RoPE(r, pos, cfg.NumKVHeads, cfg.HeadDim, cfg.RopeTheta)
				store.AppendKV(0, r, rope, v, pos)
				be.Free(r)
				be.Free(v)
				be.Free(rope)
			}
			for i := 0; i < 5; i++ {
				appendRow(kv, i, i)
			}
			if got, want := kv.(interface{ ResidentBytes() int64 }).ResidentBytes(), int64(5*width*8+5*8); got != want {
				t.Fatalf("resident bytes=%d want %d", got, want)
			}
			host, err := SnapshotKVToHost(kv)
			if err != nil {
				t.Fatal(err)
			}
			clone := kv.Clone()
			defer clone.Free()
			restored, err := RestoreKVFromHost(be, host)
			if err != nil {
				t.Fatal(err)
			}
			defer restored.Free()
			for _, other := range []KVStore{clone, restored} {
				image, err := SnapshotKVToHost(other)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(host, image) {
					t.Fatal("clone/restore changed tier or payload")
				}
			}
			// Same rounded attended rows through the old F32 kernel form the attention oracle.
			f32 := host.Clone()
			f32.Config.Precision = KVPrecisionF32
			ref, err := RestoreKVFromHost(be, f32)
			if err != nil {
				t.Fatal(err)
			}
			defer ref.Free()
			qh := make([]float32, 4*256)
			for i := range qh {
				qh[i] = float32(i%11-5) / 32
			}
			keep = append(keep, qh)
			q := mkResident(be, []int{len(qh)}, qh)
			defer be.Free(q)
			gotT := be.Attention(q, kv, 0, true, 2, 1.0/16)
			wantT := be.Attention(q, ref, 0, true, 2, 1.0/16)
			got, want := be.Read(gotT), be.Read(wantT)
			be.Free(gotT)
			be.Free(wantT)
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-5 {
					t.Fatalf("attention[%d]=%g want %g", i, got[i], want[i])
				}
			}
			// Middle eviction must reproduce a never-saw cache using the original F32 shadow.
			never := be.NewKV(cfg)
			defer never.Free()
			for pos, source := range []int{0, 3, 4} {
				appendRow(never, source, pos)
			}
			if kv.Evict(1, 2) != 2 {
				t.Fatal("eviction count")
			}
			a, err := SnapshotKVToHost(kv)
			if err != nil {
				t.Fatal(err)
			}
			b, err := SnapshotKVToHost(never)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(a, b) {
				t.Fatal("eviction differs from never-saw cache")
			}
			// Cloned/restored rows remain independent after source mutation.
			image, err := SnapshotKVToHost(clone)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(host, image) {
				t.Fatal("source eviction mutated clone")
			}
			_ = keep
		})
	}
}

// RNE tie cases distinguish actual half storage from truncation or an F32 facade.
func TestCUDAKV16RoundsTiesEven(t *testing.T) {
	be := cudaGDNBackend(t)
	for _, tc := range []struct {
		tier  KVPrecision
		delta float32
	}{{KVPrecisionF16, 1.0 / 2048}, {KVPrecisionBF16, 1.0 / 256}} {
		cfg := KVConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 4, Precision: tc.tier}
		kv := be.NewKV(cfg)
		row := []float32{1 + tc.delta, 1 + 3*tc.delta, -1 - tc.delta, -1 - 3*tc.delta}
		x := mkResident(be, []int{4}, row)
		kv.AppendKV(0, x, x, x, 0)
		state, err := SnapshotKVToHost(kv)
		if err != nil {
			t.Fatal(err)
		}
		want := []float32{1, 1 + 4*tc.delta, -1, -1 - 4*tc.delta}
		if !reflect.DeepEqual(state.K[0], want) || !reflect.DeepEqual(state.V[0], want) || !reflect.DeepEqual(state.KRaw[0], row) {
			t.Fatalf("%s narrowing/shadow mismatch: %#v", tc.tier, state)
		}
		be.Free(x)
		kv.Free()
	}
}
