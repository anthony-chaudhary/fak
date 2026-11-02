package ggufload

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestConvertGGUFKQuant(t *testing.T) {
	t.Run("ordinary-reader-unchanged", func(t *testing.T) {
		// The historical reader accepts an empty nested-array descriptor. The
		// conversion-only budget must not silently alter that existing surface.
		raw := conversionFixture("nested-array")
		if _, err := ReadSize(bytes.NewReader(raw), int64(len(raw))); err != nil {
			t.Fatalf("normal reader changed: %v", err)
		}
	})
	for _, typ := range []TensorType{TensorQ4_K, TensorQ5_K, TensorQ6_K} {
		t.Run(typ.String(), func(t *testing.T) {
			dir := t.TempDir()
			input := filepath.Join(dir, "input.gguf")
			output := filepath.Join(dir, "output.gguf")
			original := conversionFixture("")
			if err := os.WriteFile(input, original, 0600); err != nil {
				t.Fatal(err)
			}
			report, err := ConvertGGUF(t.Context(), input, output, typ)
			if err != nil {
				t.Fatal(err)
			}
			if report.ConvertedTensors != 1 || report.PreservedTensors != 1 {
				t.Fatalf("report %+v", report)
			}
			after, _ := os.ReadFile(input)
			if !bytes.Equal(after, original) {
				t.Fatal("source changed")
			}
			out, err := os.ReadFile(output)
			if err != nil {
				t.Fatal(err)
			}
			var beforeLayout, afterLayout ggufHeaderLayout
			before, err := readConversionHeader(bytes.NewReader(original), int64(len(original)), &beforeLayout)
			if err != nil {
				t.Fatal(err)
			}
			converted, err := readConversionHeader(bytes.NewReader(out), int64(len(out)), &afterLayout)
			if err != nil {
				t.Fatal(err)
			}
			if converted.Alignment != 64 || converted.Tensors[0].Type != typ || converted.Tensors[1].Type != TensorF32 {
				t.Fatal("lost layout/type")
			}
			if v, _ := converted.Uint64("general.file_type"); v != map[TensorType]uint64{TensorQ4_K: 14, TensorQ5_K: 16, TensorQ6_K: 18}[typ] {
				t.Fatal("stale file type")
			}
			if v, _ := converted.Uint64("general.quantization_version"); v != 2 {
				t.Fatal("stale quantization version")
			}
			for i, m := range beforeLayout.metadata {
				if m.key == "general.file_type" || m.key == "general.quantization_version" {
					continue
				}
				n := afterLayout.metadata[i]
				if !bytes.Equal(original[m.start:m.end], out[n.start:n.end]) {
					t.Fatalf("metadata %s reserialized", m.key)
				}
			}
			source, err := OpenWeights(input)
			if err != nil {
				t.Fatal(err)
			}
			defer source.Close()
			loaded, err := OpenWeights(output)
			if err != nil {
				t.Fatal(err)
			}
			defer loaded.Close()
			preserved, _, err := loaded.TensorBytes("preserved")
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(preserved, original[before.Tensors[1].FileOffset:before.Tensors[1].FileOffset+12]) {
				t.Fatal("preserved tensor bytes changed")
			}
			row, _, err := source.TensorF32("matrix")
			if err != nil {
				t.Fatal(err)
			}
			want, err := quantizeKRow(row, typ)
			if err != nil {
				t.Fatal(err)
			}
			got, _, err := loaded.TensorBytes("matrix")
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(want, got) {
				t.Fatal("converted matrix differs from row encoder")
			}
			decoded, _, err := loaded.TensorF32("matrix")
			if err != nil || len(decoded) != 512 {
				t.Fatalf("reload %d: %v", len(decoded), err)
			}
			if _, err := ConvertGGUF(t.Context(), input, output, typ); err == nil {
				t.Fatal("existing destination overwritten")
			}
			still, _ := os.ReadFile(output)
			if !bytes.Equal(still, out) {
				t.Fatal("refusal modified destination")
			}
		})
	}
	for _, mode := range []string{"nonfinite", "split", "duplicate", "overlap", "overflow", "no-eligible", "metadata-count", "tensor-count", "huge-array", "nested-array", "stale-recipe", "recipe-type", "truncated"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			in := filepath.Join(dir, "in.gguf")
			out := filepath.Join(dir, "out.gguf")
			if err := os.WriteFile(in, conversionFixture(mode), 0600); err != nil {
				t.Fatal(err)
			}
			if _, err := ConvertGGUF(t.Context(), in, out, TensorQ4_K); err == nil {
				t.Fatal("malformed/unsupported input accepted")
			}
			if _, err := os.Lstat(out); !os.IsNotExist(err) {
				t.Fatal("failure published output")
			}
			temps, _ := filepath.Glob(filepath.Join(dir, ".fak-quantize-*"))
			if len(temps) != 0 {
				t.Fatal("failure leaked owned temp")
			}
		})
	}
	t.Run("cancel", func(t *testing.T) {
		dir := t.TempDir()
		in := filepath.Join(dir, "in.gguf")
		out := filepath.Join(dir, "out.gguf")
		os.WriteFile(in, conversionFixture(""), 0600)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := ConvertGGUF(ctx, in, out, TensorQ4_K); err == nil {
			t.Fatal("canceled operation succeeded")
		}
		if _, err := os.Lstat(out); !os.IsNotExist(err) {
			t.Fatal("cancel published")
		}
	})
}

// fak-test:runtime fast est=10ms lane=default
func TestConvertGGUFRejectsMixedRecipe(t *testing.T) {
	for _, mode := range []string{"mixed-recipe", "no-eligible"} {
		dir := t.TempDir()
		in := filepath.Join(dir, "in.gguf")
		out := filepath.Join(dir, "out.gguf")
		if err := os.WriteFile(in, conversionFixture(mode), 0600); err != nil {
			t.Fatal(err)
		}
		_, err := ConvertGGUF(t.Context(), in, out, TensorQ4_K)
		want := "truthful uniform K-quant recipe"
		if mode == "no-eligible" {
			want = "no eligible F32 matrices"
		}
		if err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("%s refusal: %v", mode, err)
		}
		if _, err := os.Lstat(out); !os.IsNotExist(err) {
			t.Fatal("refusal published output")
		}
	}
}

func conversionFixture(mode string) []byte {
	var b bytes.Buffer
	u32 := func(v uint32) { binary.Write(&b, binary.LittleEndian, v) }
	u64 := func(v uint64) { binary.Write(&b, binary.LittleEndian, v) }
	str := func(s string) { u64(uint64(len(s))); b.WriteString(s) }
	key := func(s string, typ ValueType) { str(s); u32(uint32(typ)) }
	b.WriteString(Magic)
	u32(Version)
	tc, kc := uint64(2), uint64(7)
	if mode == "tensor-count" {
		tc = math.MaxUint64
	}
	if mode == "metadata-count" {
		kc = math.MaxUint64
	}
	u64(tc)
	u64(kc)
	ft := TypeUint32
	if mode == "recipe-type" {
		ft = TypeInt32
	}
	key("general.file_type", ft)
	u32(0)
	key("general.quantization_version", TypeUint32)
	u32(0)
	key("general.alignment", TypeUint32)
	u32(64)
	name := "test.empty"
	if mode == "split" {
		name = "split.count"
	}
	if mode == "stale-recipe" {
		name = "quantize.imatrix.file"
	}
	key(name, TypeArray)
	elem := TypeFloat64
	if mode == "nested-array" {
		elem = TypeArray
	}
	u32(uint32(elem))
	n := uint64(0)
	if mode == "huge-array" {
		n = 1 << 60
	}
	u64(n)
	key("test.nan", TypeFloat32)
	u32(0x7fc12345)
	key("test.duplicate", TypeString)
	str("first")
	key("test.duplicate", TypeString)
	str("second")
	str("matrix")
	if mode == "no-eligible" {
		u32(1)
		u64(512)
	} else {
		u32(2)
		width := uint64(256)
		if mode == "overflow" {
			width = math.MaxUint64
		}
		u64(width)
		u64(2)
	}
	u32(uint32(TensorF32))
	u64(0)
	name = "preserved"
	if mode == "duplicate" {
		name = "matrix"
	}
	str(name)
	if mode == "mixed-recipe" {
		u32(2)
		u64(1)
		u64(3)
	} else {
		u32(1)
		u64(3)
	}
	u32(uint32(TensorF32))
	off := uint64(2048)
	if mode == "overlap" {
		off = 0
	}
	u64(off)
	for b.Len()%64 != 0 {
		b.WriteByte(0)
	}
	for i := 0; i < 512; i++ {
		bits := math.Float32bits(float32((i*97)%503-251) / 32)
		if mode == "nonfinite" && i == 7 {
			bits = 0x7f800000
		}
		u32(bits)
	}
	u32(0x7fc01234)
	u32(0x3f800000)
	u32(0x80000000)
	if mode == "truncated" {
		return b.Bytes()[:b.Len()-1]
	}
	return b.Bytes()
}
