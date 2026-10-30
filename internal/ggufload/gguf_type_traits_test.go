package ggufload

import (
	"bytes"
	"math"
	"testing"
)

type ggufGeometryWitness struct {
	typ         TensorType
	block, size uint64
}

var ggufGeometryWitnesses = []ggufGeometryWitness{
	{TensorF32, 1, 4}, {TensorF16, 1, 2}, {TensorBF16, 1, 2},
	{TensorQ4_0, 32, 18}, {TensorQ4_1, 32, 20}, {TensorQ5_0, 32, 22}, {TensorQ5_1, 32, 24}, {TensorQ8_0, 32, 34},
	{TensorQ2_K, 256, 84}, {TensorQ3_K, 256, 110}, {TensorQ4_K, 256, 144}, {TensorQ5_K, 256, 176}, {TensorQ6_K, 256, 210},
	{TensorIQ2_XXS, 256, 66}, {TensorIQ2_XS, 256, 74}, {TensorIQ3_XXS, 256, 98}, {TensorIQ1_S, 256, 50},
	{TensorIQ4_NL, 32, 18}, {TensorIQ3_S, 256, 110}, {TensorIQ2_S, 256, 82}, {TensorIQ4_XS, 256, 136}, {TensorIQ1_M, 256, 56},
	{TensorMXFP4, 32, 17},
	{TensorQ2_0, 128, 34}, {TensorPQ2_0, 128, 34}, {TensorQ1_0, 128, 18}, {TensorHQQ4, 64, 36},
}

// fak-test:justify why=contract when=changed:internal/ggufload/**
// fak-test:runtime fast est=1ms
func TestGGUFTypeTraitsAccessorClosedSet(t *testing.T) {
	t.Parallel()
	for _, tc := range ggufGeometryWitnesses {
		got, ok := tensorTypeTraits(tc.typ)
		if !ok || got.BlockSize != tc.block || got.TypeSize != tc.size || got.IsQuantized != (tc.block > 1) {
			t.Errorf("%s traits: ok=%t block=%d size=%d quantized=%t", tc.typ, ok, got.BlockSize, got.TypeSize, got.IsQuantized)
		}
	}
	for _, typ := range []TensorType{4, 5, 9, 15, 31, math.MaxUint32} {
		if _, ok := tensorTypeTraits(typ); ok {
			t.Errorf("unsupported type %d gained traits", typ)
		}
	}
}

// fak-test:justify why=regression when=changed:internal/ggufload/**
// fak-test:runtime fast est=1ms
func TestGGUFTypeTraitsPayloadGeometry(t *testing.T) {
	t.Parallel()
	for _, tc := range ggufGeometryWitnesses {
		for _, units := range []uint64{1, 3} {
			info := TensorInfo{Name: "geometry", Type: tc.typ, Dims: []uint64{tc.block, units}}
			got, err := tensorPayloadBytes(info)
			if err != nil || got != tc.size*units {
				t.Errorf("%s blocks=%d bytes=%d want=%d err=%v", tc.typ, units, got, tc.size*units, err)
			}
		}
	}
}

// fak-test:justify why=contract when=changed:internal/ggufload/**
// fak-test:runtime fast est=1ms
func TestGGUFTypeTraitsRejectInvalidPayloads(t *testing.T) {
	t.Parallel()
	cases := []TensorInfo{
		{Name: "missing-dimensions", Type: TensorF32},
		{Name: "zero-dimension", Type: TensorF32, Dims: []uint64{32, 0}},
		{Name: "element-overflow", Type: TensorF32, Dims: []uint64{math.MaxUint64, 2}},
		{Name: "byte-overflow", Type: TensorF16, Dims: []uint64{math.MaxUint64/2 + 1}},
	}
	for _, typ := range []TensorType{4, 5, 9, 15, 31, math.MaxUint32} {
		cases = append(cases, TensorInfo{Name: "unsupported", Type: typ, Dims: []uint64{256}})
	}
	for _, tc := range ggufGeometryWitnesses {
		if tc.block > 1 {
			cases = append(cases, TensorInfo{Name: "partial-block", Type: tc.typ, Dims: []uint64{tc.block - 1}})
		}
	}
	for _, info := range cases {
		if got, err := tensorPayloadBytes(info); err == nil || got != 0 {
			t.Errorf("invalid %s type=%s dimensions=%v returned bytes=%d err=%v", info.Name, info.Type, info.Dims, got, err)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/ggufload/**
// fak-test:runtime fast est=1ms
func TestTypeTraitsMatchTensorPayloadBytes(t *testing.T) {
	t.Parallel()
	// Every named GGUF type constant must size its payload through the shared
	// traits table as elements/BlockSize*TypeSize; any enum value without a
	// table entry must refuse instead of guessing geometry.
	witness := make(map[TensorType]ggufGeometryWitness, len(ggufGeometryWitnesses))
	for _, w := range ggufGeometryWitnesses {
		witness[w.typ] = w
	}
	all := []TensorType{
		TensorF32, TensorF16, TensorQ4_0, TensorQ4_1, TensorQ5_0, TensorQ5_1,
		TensorQ8_0, TensorQ2_K, TensorQ3_K, TensorQ4_K, TensorQ5_K, TensorQ6_K,
		TensorIQ2_XXS, TensorIQ2_XS, TensorIQ3_XXS, TensorIQ1_S, TensorIQ4_NL,
		TensorIQ3_S, TensorIQ2_S, TensorIQ4_XS, TensorIQ1_M, TensorBF16,
		TensorMXFP4, TensorQ2_0, TensorPQ2_0, TensorQ1_0, TensorHQQ4,
	}
	if len(all) != len(witness) {
		t.Fatalf("enum coverage drifted: %d named types vs %d witnesses", len(all), len(witness))
	}
	for _, typ := range all {
		tr, ok := tensorTypeTraits(typ)
		if !ok {
			if got, err := tensorPayloadBytes(TensorInfo{Name: "payload", Type: typ, Dims: []uint64{256}}); err == nil || got != 0 {
				t.Errorf("%s: unsupported enum value sized %d bytes without error", typ, got)
			}
			continue
		}
		w := witness[typ]
		if tr.BlockSize != w.block || tr.TypeSize != w.size {
			t.Errorf("%s: table block=%d size=%d want block=%d size=%d", typ, tr.BlockSize, tr.TypeSize, w.block, w.size)
			continue
		}
		info := TensorInfo{Name: "payload", Type: typ, Dims: []uint64{tr.BlockSize, 2}}
		got, err := tensorPayloadBytes(info)
		want := w.size * 2 // (BlockSize*2)/BlockSize*TypeSize
		if err != nil || got != want {
			t.Errorf("%s: payload bytes=%d err=%v want=%d", typ, got, err, want)
		}
	}
}

// fak-test:justify why=integration when=changed:internal/ggufload/**
// fak-test:runtime fast est=1ms
func TestGGUFTypeTraitsWeightSourceTensorBytes(t *testing.T) {
	t.Parallel()
	for _, tc := range []ggufGeometryWitness{{TensorIQ2_XS, 256, 74}, {TensorQ2_0, 128, 34}, {TensorHQQ4, 64, 36}} {
		payload := make([]byte, tc.size*2)
		for i := range payload {
			payload[i] = byte(i*7 + 3)
		}
		data := append([]byte{0xaa, 0xbb, 0xcc}, payload...)
		data = append(data, 0xdd)
		info := TensorInfo{Name: "ordinary.weight", Type: tc.typ, Dims: []uint64{tc.block, 2}, FileOffset: 3}
		ws := &WeightSource{File: &File{Tensors: []TensorInfo{info}}, r: bytes.NewReader(data), size: int64(len(data)), byName: map[string]int{info.Name: 0}}
		got, tensor, err := ws.TensorBytes(info.Name)
		if err != nil || tensor.Type != tc.typ || !bytes.Equal(got, payload) {
			t.Fatalf("%s ordinary reader returned type=%s len=%d err=%v", tc.typ, tensor.Type, len(got), err)
		}
		got[0] ^= 0xff
		if data[3] != payload[0] {
			t.Fatal("ordinary reader returned an alias instead of an owned copy")
		}
		ws.size--
		ws.size--
		if _, _, err := ws.TensorBytes(info.Name); err == nil {
			t.Fatalf("%s truncated payload accepted", tc.typ)
		}
	}
}
