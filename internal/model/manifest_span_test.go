package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=25ms
func TestLoadNativeManifestRejectsInvalidSpans(t *testing.T) {
	m := nativeManifestSpanFixture()
	span := func(shape []int, offset, size int) tensorMeta {
		return tensorMeta{Dtype: "f32", Shape: shape, Offset: offset, Nbytes: size}
	}
	wrongDtype := span([]int{1}, 0, 4)
	wrongDtype.Dtype = "F16"
	for _, tc := range []struct {
		name      string
		meta      any
		invariant string
	}{
		{"dtype", wrongDtype, "dtype"},
		{"missing_shape", map[string]any{"dtype": "f32", "offset": 0, "nbytes": 4}, "shape_missing"},
		{"null_shape", span(nil, 0, 4), "shape_missing"},
		{"negative_shape", span([]int{-1}, 0, 0), "shape_negative"},
		{"zero_before_negative", span([]int{0, -1}, 0, 0), "shape_negative"},
		{"negative_before_zero", span([]int{-1, 0}, 0, 0), "shape_negative"},
		{"byte_count_overflow", span([]int{math.MaxInt/4 + 1}, 0, 4), "shape_overflow"},
		{"element_count_overflow", span([]int{math.MaxInt / 4, 8}, 0, 4), "shape_overflow"},
		{"negative_offset", span([]int{1}, -4, 4), "offset_negative"},
		{"negative_size", span([]int{1}, 0, -4), "size_negative"},
		{"unaligned_offset", span([]int{1}, 1, 4), "offset_alignment"},
		{"unaligned_size", span([]int{1}, 0, 5), "size_alignment"},
		{"positive_at_eof", span([]int{1}, len(m.raw), 4), "bounds"},
		{"empty_past_eof", span([]int{0}, len(m.raw)+4, 0), "bounds"},
		{"overflowing_end", span([]int{2}, math.MaxInt-3, 8), "bounds"},
		{"size_exceeds_raw", span([]int{1}, 8, math.MaxInt-3), "bounds"},
		{"shape_bytes_mismatch", span([]int{2}, 0, 4), "shape_size"},
		{"proper_overlap", span([]int{1}, 4, 4), "overlap"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			man := make(map[string]any, len(m.manifest)+1)
			for name, meta := range m.manifest {
				man[name] = meta
			}
			// The unmodified loader ignores this name, so the RED witness is
			// successful admission, never a malformed unsafe view or a crash.
			man["unused.invalid"] = tc.meta
			got, err := Load(writeNativeManifestSpanFixture(t, m.Cfg, man, m.raw))
			assertNativeManifestSpanRefusal(t, got, err, "unused.invalid", tc.invariant)
		})
	}

	t.Run("deterministic_tensor", func(t *testing.T) {
		for _, names := range [][]string{{"unused.a", "unused.z"}, {"unused.z", "unused.a"}} {
			man := make(map[string]tensorMeta, len(m.manifest)+2)
			for name, meta := range m.manifest {
				man[name] = meta
			}
			for _, name := range names {
				man[name] = wrongDtype
			}
			got, err := Load(writeNativeManifestSpanFixture(t, m.Cfg, man, m.raw))
			assertNativeManifestSpanRefusal(t, got, err, "unused.a", "dtype")
		}
	})
}

// fak-test:runtime fast est=5ms
func TestLoadNativeManifestPreservesValidSpans(t *testing.T) {
	m := nativeManifestSpanFixture()
	raw := append(append([]byte(nil), m.raw...), make([]byte, 12)...)
	// Leave a four-byte gap and four trailing bytes outside all positive spans.
	scalarOffset := len(m.raw) + 4
	binary.LittleEndian.PutUint32(raw[scalarOffset:], math.Float32bits(3.25))
	man := make(map[string]tensorMeta, len(m.manifest)+4)
	for name, meta := range m.manifest {
		man[name] = meta // the native synthetic producer uses uppercase F32
	}
	man["unused.scalar"] = tensorMeta{Dtype: "f32", Shape: []int{}, Offset: scalarOffset, Nbytes: 4}
	man["unused.alias"] = tensorMeta{Dtype: "F32", Shape: []int{1}, Offset: scalarOffset, Nbytes: 4}
	man["unused.empty_eof"] = tensorMeta{Dtype: "f32", Shape: []int{math.MaxInt, 2, 0}, Offset: len(raw)}
	man["unused.empty_inside"] = tensorMeta{Dtype: "F32", Shape: []int{0}, Offset: 4}
	got, err := Load(writeNativeManifestSpanFixture(t, m.Cfg, man, raw))
	if err != nil || got == nil {
		t.Fatalf("valid native export: model=%v error=%v", got != nil, err)
	}
	if !bytes.Equal(got.raw, raw) {
		t.Fatal("native admission changed the exported weight bytes")
	}
	for name := range m.manifest {
		if !slices.Equal(got.tensor(name), m.tensor(name)) {
			t.Fatalf("native admission changed tensor %q", name)
		}
	}
	scalar, alias := got.tensor("unused.scalar"), got.tensor("unused.alias")
	if len(scalar) != 1 || scalar[0] != 3.25 || len(alias) != 1 || &scalar[0] != &alias[0] {
		t.Fatal("scalar and exact alias did not preserve the same zero-copy f32 view")
	}
	for _, name := range []string{"unused.empty_eof", "unused.empty_inside"} {
		if view := got.tensor(name); len(view) != 0 {
			t.Fatalf("empty tensor %q has %d elements", name, len(view))
		}
	}
	// Empty storage must not require a byte at offset zero, either.
	if view := manifestTensor(map[string]tensorMeta{"empty": {Nbytes: 0}}, nil, "", "empty"); len(view) != 0 {
		t.Fatalf("empty storage has %d elements", len(view))
	}
}

// fak-test:runtime fast est=5ms
func TestLoadNativeManifestValidationPrecedesCohere(t *testing.T) {
	m := nativeManifestSpanFixture()
	m.Cfg.ModelType, m.Cfg.HeadDim = "cohere", 3
	// This config has an independent rotary refusal. A malformed unused span
	// must be refused first, before the architecture pass can touch raw bytes.
	m.manifest["unused.invalid"] = tensorMeta{Dtype: "f32", Shape: []int{1}, Offset: -4, Nbytes: 4}
	got, err := Load(writeNativeManifestSpanFixture(t, m.Cfg, m.manifest, m.raw))
	assertNativeManifestSpanRefusal(t, got, err, "unused.invalid", "offset_negative")
	delete(m.manifest, "unused.invalid")
	got, err = Load(writeNativeManifestSpanFixture(t, m.Cfg, m.manifest, m.raw))
	if got != nil || err == nil || !strings.Contains(err.Error(), "cohere rotary") {
		t.Fatalf("valid descriptors should reach the existing Cohere refusal: model=%v error=%v", got != nil, err)
	}
}

func nativeManifestSpanFixture() *Model {
	return NewSynthetic(Config{
		ModelType: "llama", HiddenSize: 8, IntermediateSize: 16,
		NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
		VocabSize: 16, RMSNormEps: 1e-5, RopeTheta: 10000,
	})
}

func writeNativeManifestSpanFixture(t *testing.T, cfg Config, man any, raw []byte) string {
	t.Helper()
	dir := t.TempDir()
	for name, value := range map[string]any{"config.json": cfg, "manifest.json": man} {
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "weights.f32"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func assertNativeManifestSpanRefusal(t *testing.T, got *Model, err error, tensor, invariant string) {
	t.Helper()
	if got != nil || err == nil {
		t.Fatalf("Load admitted malformed %s descriptor: model=%v error=%v", invariant, got != nil, err)
	}
	// This structural contract compiles on the parent for the native RED witness;
	// it deliberately does not reference the new concrete error declaration.
	var detail interface {
		error
		TensorName() string
		FailedInvariant() string
	}
	if !errors.As(err, &detail) || detail.TensorName() != tensor || detail.FailedInvariant() != invariant {
		t.Fatalf("want typed tensor=%q invariant=%q refusal, got %T: %v", tensor, invariant, err, err)
	}
	if !strings.Contains(err.Error(), tensor) || !strings.Contains(err.Error(), invariant) {
		t.Fatalf("refusal does not name tensor and invariant: %v", err)
	}
}
