package ggufload

import (
	"bytes"
	"math"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/ggufload/**
// fak-test:runtime medium est=1s lane=default
func TestGGUFIntegerPayloadSizing(t *testing.T) {
	for _, tc := range []struct {
		typ   TensorType
		width uint64
	}{{TensorType(26), 4}, {TensorType(27), 8}} {
		t.Run(tc.typ.String(), func(t *testing.T) {
			info := TensorInfo{Name: "integer", Type: tc.typ, Dims: []uint64{3, 2}, FileOffset: 3}
			want := tc.width * 6
			if got, err := tensorPayloadBytes(info); err != nil || got != want {
				t.Fatalf("payload bytes=%d, want %d: %v", got, want, err)
			}
			// Sign bits are payload, not dimensions or floating-point values.
			raw := bytes.Repeat([]byte{0xff}, int(want))
			if HostDequantSupported(tc.typ) {
				t.Fatal("integer byte sizing granted host float-decode capability")
			}
			if _, err := dequantF32(info, raw); err == nil {
				t.Fatal("integer payload silently converted to float")
			}
			data := append([]byte{1, 2, 3}, raw...)
			ws, err := NewWeightSource(&File{Tensors: []TensorInfo{info}}, bytes.NewReader(data), int64(len(data)))
			if err != nil {
				t.Fatal(err)
			}
			got, _, err := ws.TensorBytes(info.Name)
			if err != nil || !bytes.Equal(got, raw) {
				t.Fatalf("integer raw read differs: %v", err)
			}
			ws.size--
			if _, _, err := ws.TensorBytes(info.Name); err == nil {
				t.Fatal("truncated integer extent accepted")
			}
			// GGUF dimensions are unsigned. A negative signed dimension encoded
			// as bits is huge and must not narrow or bypass overflow admission.
			for _, dims := range [][]uint64{nil, {0}, {math.MaxUint64, 2}, {math.MaxUint64/tc.width + 1}, {1 << 63}} {
				bad := info
				bad.Dims = dims
				if n, err := tensorPayloadBytes(bad); err == nil || n != 0 {
					t.Errorf("invalid integer dimensions %v returned bytes=%d err=%v", dims, n, err)
				}
			}
		})
	}
	t.Run("validated Engram auxiliaries compose with load estimates", func(t *testing.T) {
		ws, _, raw := tensorEngramAuxFixture(t)
		if _, err := ws.File.Config(); err != nil {
			t.Fatalf("valid tensor-carried auxiliary Config control: %v", err)
		}
		want := int64(len(raw))
		for _, info := range ws.File.Tensors {
			if deepSeek41EngramAuxTensor(info.Name) {
				continue
			}
			n, err := tensorPayloadBytes(info)
			if err != nil {
				t.Fatalf("noninteger control %s: %v", info.Name, err)
			}
			want += int64(n)
		}
		if got, err := ws.EstimateLoadBytes(); err != nil || got != want {
			t.Fatalf("validated auxiliary payload bytes=%d, want %d: %v", got, want, err)
		}
		if plan, err := ws.EstimateLoadMemoryPlan(); err != nil || plan.Total() != want {
			t.Fatalf("validated auxiliary memory plan=%v, want %d: %v", plan, want, err)
		}
	})
}
