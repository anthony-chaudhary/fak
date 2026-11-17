package model

import (
	"os"
	"path/filepath"
	"testing"
)

// writePLEMetadata writes raw checkpoint metadata to a temp file and returns its
// path. Fixtures live under t.TempDir() so nothing lands in the repo tree.
func writePLEMetadata(t *testing.T, raw string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatalf("write metadata fixture: %v", err)
	}
	return path
}

func readPLEMetadata(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read metadata fixture: %v", err)
	}
	return raw
}

// TestPLEDtypeMetadataDetection pins that the PLE table's dtype is read from the
// checkpoint's own metadata rather than assumed to be the model default. It
// fails if the code assumes the default when the metadata declares a different
// dtype (the upstream TensorRT-LLM ple.py:69 defect).
func TestPLEDtypeMetadataDetection(t *testing.T) {
	const modelDefault = "bfloat16"

	declared := []struct {
		name  string
		value string
	}{
		{"bf16", "bf16"},
		{"f16", "f16"},
		{"f32", "f32"},
	}
	for _, tc := range declared {
		t.Run("declared_"+tc.name, func(t *testing.T) {
			path := writePLEMetadata(t, `{"dtype":"bfloat16","ple_embedding_dtype":"`+tc.value+`"}`)
			got, err := ResolvePLEDType(readPLEMetadata(t, path), modelDefault)
			if err != nil {
				t.Fatalf("ResolvePLEDType: %v", err)
			}
			if got.DType != tc.value {
				t.Fatalf("dtype = %q, want %q", got.DType, tc.value)
			}
			if !got.Declared() || got.Source != PLEDTypeDeclared {
				t.Fatalf("source = %q, want declared", got.Source)
			}
		})
	}

	t.Run("omitted_falls_back_and_is_assumed", func(t *testing.T) {
		path := writePLEMetadata(t, `{"dtype":"bfloat16"}`)
		got, err := ResolvePLEDType(readPLEMetadata(t, path), modelDefault)
		if err != nil {
			t.Fatalf("ResolvePLEDType: %v", err)
		}
		if got.DType != modelDefault {
			t.Fatalf("dtype = %q, want model default %q", got.DType, modelDefault)
		}
		if got.Declared() || got.Source != PLEDTypeAssumed {
			t.Fatalf("source = %q, want assumed (not declared)", got.Source)
		}
	})

	t.Run("declared_wins_over_model_default", func(t *testing.T) {
		// The model default is bfloat16; the metadata declares f32. Assuming the
		// default here is exactly the wrong-precision lookup upstream fixed.
		path := writePLEMetadata(t, `{"dtype":"bfloat16","ple_embedding_dtype":"f32"}`)
		got, err := ResolvePLEDType(readPLEMetadata(t, path), modelDefault)
		if err != nil {
			t.Fatalf("ResolvePLEDType: %v", err)
		}
		if got.DType == modelDefault {
			t.Fatalf("dtype = %q: assumed the model default despite a declared f32", got.DType)
		}
		if got.DType != "f32" {
			t.Fatalf("dtype = %q, want f32", got.DType)
		}
		if !got.Declared() {
			t.Fatalf("source = %q, want declared", got.Source)
		}
	})

	t.Run("empty_declared_value_is_omitted", func(t *testing.T) {
		path := writePLEMetadata(t, `{"ple_embedding_dtype":""}`)
		got, err := ResolvePLEDType(readPLEMetadata(t, path), modelDefault)
		if err != nil {
			t.Fatalf("ResolvePLEDType: %v", err)
		}
		if got.DType != modelDefault || got.Declared() {
			t.Fatalf("empty declared value: dtype=%q source=%q, want assumed default", got.DType, got.Source)
		}
	})

	t.Run("omitted_without_default_is_an_error", func(t *testing.T) {
		path := writePLEMetadata(t, `{"dtype":"bfloat16"}`)
		if _, err := ResolvePLEDType(readPLEMetadata(t, path), ""); err == nil {
			t.Fatal("omitted PLE dtype with no model default was accepted")
		}
	})
}
