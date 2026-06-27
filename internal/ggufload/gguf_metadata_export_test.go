package ggufload

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"testing"
)

// TestExportMetadataPreservesEveryTypeAndFullArrays witnesses the full-metadata-export
// surface (#292): every scalar GGUF value type is exported under its spec type name with
// its decoded value intact, every tensor directory entry is preserved, and — the property
// the summarizing ggufprobe dump deliberately lacks — a large array is exported in FULL,
// surviving a JSON round-trip with no truncation.
func TestExportMetadataPreservesEveryTypeAndFullArrays(t *testing.T) {
	var b bytes.Buffer
	writeStr := func(s string) {
		_ = binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	writeKVHeader := func(k string, typ ValueType) {
		writeStr(k)
		_ = binary.Write(&b, binary.LittleEndian, uint32(typ))
	}
	le := func(v any) { _ = binary.Write(&b, binary.LittleEndian, v) }

	// A deliberately large string array: proves the export is lossless where the
	// human-oriented probe summarizes a big vocab.
	const bigN = 300
	bigToks := make([]string, bigN)
	for i := range bigToks {
		bigToks[i] = fmt.Sprintf("tok%03d", i)
	}

	b.WriteString(Magic)
	le(uint32(Version))
	le(uint64(2))  // tensors
	le(uint64(16)) // metadata KVs

	writeKVHeader("general.alignment", TypeUint32)
	le(uint32(32))
	writeKVHeader("general.architecture", TypeString)
	writeStr("llama")
	writeKVHeader("m.u8", TypeUint8)
	b.WriteByte(7)
	writeKVHeader("m.i8", TypeInt8)
	b.WriteByte(0xFD) // two's-complement int8(-3)
	writeKVHeader("m.u16", TypeUint16)
	le(uint16(40000))
	writeKVHeader("m.i16", TypeInt16)
	le(int16(-1234))
	writeKVHeader("m.u32", TypeUint32)
	le(uint32(0xdeadbeef))
	writeKVHeader("m.i32", TypeInt32)
	le(int32(-77))
	writeKVHeader("m.f32", TypeFloat32)
	le(math.Float32bits(1.5))
	writeKVHeader("m.bool", TypeBool)
	b.WriteByte(1)
	writeKVHeader("m.str", TypeString)
	writeStr("hello")
	writeKVHeader("m.u64", TypeUint64)
	le(uint64(1) << 40)
	writeKVHeader("m.i64", TypeInt64)
	le(int64(-(1 << 40)))
	writeKVHeader("m.f64", TypeFloat64)
	le(math.Float64bits(3.25))
	writeKVHeader("m.i32arr", TypeArray)
	le(uint32(TypeInt32))
	le(uint64(3))
	le(int32(10))
	le(int32(20))
	le(int32(30))
	writeKVHeader("m.bigtokens", TypeArray)
	le(uint32(TypeString))
	le(uint64(bigN))
	for _, s := range bigToks {
		writeStr(s)
	}

	writeTensor := func(name string, dims []uint64, typ TensorType, off uint64) {
		writeStr(name)
		le(uint32(len(dims)))
		for _, d := range dims {
			le(d)
		}
		le(uint32(typ))
		le(off)
	}
	writeTensor("token_embd.weight", []uint64{32, 8}, TensorF32, 0)
	writeTensor("output.weight", []uint64{32, 8}, TensorQ4_K, 64)

	f, err := Read(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	exp := f.ExportMetadata()
	if exp.Version != Version || exp.Alignment != 32 {
		t.Fatalf("header scalars: version=%d alignment=%d", exp.Version, exp.Alignment)
	}
	if exp.TensorDataOffset <= 0 || exp.TensorDataOffset%32 != 0 {
		t.Fatalf("tensor data offset %d not a positive multiple of 32", exp.TensorDataOffset)
	}

	byKey := make(map[string]MetaKV, len(exp.Metadata))
	for i, kv := range exp.Metadata {
		byKey[kv.Key] = kv
		if i > 0 && exp.Metadata[i-1].Key > kv.Key {
			t.Fatalf("metadata not sorted at %d: %q after %q", i, kv.Key, exp.Metadata[i-1].Key)
		}
	}

	wantScalar := []struct {
		key, typ string
		val      any
	}{
		{"general.alignment", "uint32", uint32(32)},
		{"general.architecture", "string", "llama"},
		{"m.u8", "uint8", uint8(7)},
		{"m.i8", "int8", int8(-3)},
		{"m.u16", "uint16", uint16(40000)},
		{"m.i16", "int16", int16(-1234)},
		{"m.u32", "uint32", uint32(0xdeadbeef)},
		{"m.i32", "int32", int32(-77)},
		{"m.f32", "float32", float32(1.5)},
		{"m.bool", "bool", true},
		{"m.str", "string", "hello"},
		{"m.u64", "uint64", uint64(1) << 40},
		{"m.i64", "int64", int64(-(1 << 40))},
		{"m.f64", "float64", float64(3.25)},
	}
	for _, w := range wantScalar {
		kv, ok := byKey[w.key]
		if !ok {
			t.Fatalf("missing metadata key %q", w.key)
		}
		if kv.Type != w.typ {
			t.Fatalf("%s: type=%q want %q", w.key, kv.Type, w.typ)
		}
		if !reflect.DeepEqual(kv.Value, w.val) {
			t.Fatalf("%s: value=%#v (%T) want %#v (%T)", w.key, kv.Value, kv.Value, w.val, w.val)
		}
	}

	// Int32 array: type "array", element type "int32", full element set.
	i32arr := byKey["m.i32arr"]
	if i32arr.Type != "array" || i32arr.ElemType != "int32" {
		t.Fatalf("m.i32arr: type=%q elem=%q", i32arr.Type, i32arr.ElemType)
	}
	if got := i32arr.Value.([]any); !reflect.DeepEqual(got, []any{int32(10), int32(20), int32(30)}) {
		t.Fatalf("m.i32arr value=%#v", got)
	}

	// Large string array: every element preserved, in order — the lossless witness.
	big := byKey["m.bigtokens"]
	if big.Type != "array" || big.ElemType != "string" {
		t.Fatalf("m.bigtokens: type=%q elem=%q", big.Type, big.ElemType)
	}
	bigVals := big.Value.([]any)
	if len(bigVals) != bigN {
		t.Fatalf("m.bigtokens truncated: got %d elements want %d", len(bigVals), bigN)
	}
	for i, want := range bigToks {
		if bigVals[i] != want {
			t.Fatalf("m.bigtokens[%d]=%v want %v", i, bigVals[i], want)
		}
	}

	// Tensor directory preserved in file order with spec type names.
	if len(exp.Tensors) != 2 {
		t.Fatalf("tensor count=%d", len(exp.Tensors))
	}
	if exp.Tensors[0].Name != "token_embd.weight" || exp.Tensors[0].Type != "F32" ||
		!reflect.DeepEqual(exp.Tensors[0].Dims, []uint64{32, 8}) || exp.Tensors[0].Offset != 0 {
		t.Fatalf("tensor[0]=%#v", exp.Tensors[0])
	}
	if exp.Tensors[1].Name != "output.weight" || exp.Tensors[1].Type != "Q4_K" || exp.Tensors[1].Offset != 64 {
		t.Fatalf("tensor[1]=%#v", exp.Tensors[1])
	}

	// JSON surface: valid, and the big array survives serialization with no truncation.
	js, err := f.MetadataJSON()
	if err != nil {
		t.Fatalf("MetadataJSON: %v", err)
	}
	if !json.Valid(js) {
		t.Fatalf("MetadataJSON produced invalid JSON")
	}
	var round Metadata
	if err := json.Unmarshal(js, &round); err != nil {
		t.Fatalf("unmarshal export JSON: %v", err)
	}
	if len(round.Metadata) != len(exp.Metadata) || len(round.Tensors) != len(exp.Tensors) {
		t.Fatalf("JSON round-trip dropped entries: meta %d->%d tensors %d->%d",
			len(exp.Metadata), len(round.Metadata), len(exp.Tensors), len(round.Tensors))
	}
	for _, kv := range round.Metadata {
		if kv.Key == "m.bigtokens" {
			if arr, ok := kv.Value.([]any); !ok || len(arr) != bigN {
				t.Fatalf("m.bigtokens not full after JSON round-trip: %d", len(arr))
			}
		}
	}
}
