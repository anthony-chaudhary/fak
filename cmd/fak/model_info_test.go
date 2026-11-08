package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
)

// TestModelInfoReportsCachedCheckpointMetadata pins the closed contract of
// `fak model info` (the TensorFold checkpoint-metadata-discovery borrow): a
// cached GGUF reports family/format/context/weights/floor with exact values and
// floor = weights + the fixed runtime margin; an uncached ref is refused with
// the pull hint (ErrModelNotCached); a ref outside the model registry is
// refused naming the registry; an alias-declared family absent from the
// capability registry is refused with ErrUnknownModelType before any cache
// probe; and a non-GGUF artifact is refused with ErrModelFormatUnreadable.
func TestModelInfoReportsCachedCheckpointMetadata(t *testing.T) {
	root := t.TempDir()
	t.Setenv("FAK_MODELS_DIR", root)

	// Fixture: the package's minimal Q8_0 GGUF — arch "llama", declared context
	// 16, two 256x256 Q8_0 tensors of 69632 bytes each (139264 weight bytes).
	fixturePath := createTestQ8GGUF(t)
	const wantWeights int64 = 2 * 69632 // 139264
	const wantFloor int64 = wantWeights + modelInfoRuntimeMargin

	t.Run("cached local path reports exact metadata", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{"--json", fixturePath})
		if code != 0 {
			t.Fatalf("runModelInfo exit = %d, stderr=%q, want 0", code, errb.String())
		}
		var rep modelInfoReport
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("decode report: %v\n%s", err, out.String())
		}
		if rep.Schema != "fak.model-info/1" {
			t.Errorf("schema = %q, want fak.model-info/1", rep.Schema)
		}
		if rep.Path != fixturePath {
			t.Errorf("path = %q, want %q", rep.Path, fixturePath)
		}
		if rep.Family != "llama" {
			t.Errorf("family = %q, want llama", rep.Family)
		}
		if rep.Format != "Q8_0" {
			t.Errorf("format = %q, want Q8_0", rep.Format)
		}
		if rep.Context != 16 {
			t.Errorf("context = %d, want 16", rep.Context)
		}
		if rep.WeightsBytes != wantWeights {
			t.Errorf("weights_bytes = %d, want %d", rep.WeightsBytes, wantWeights)
		}
		if rep.RuntimeMarginBytes != modelInfoRuntimeMargin {
			t.Errorf("runtime_margin_bytes = %d, want %d", rep.RuntimeMarginBytes, modelInfoRuntimeMargin)
		}
		if rep.FloorBytes != wantFloor {
			t.Errorf("floor_bytes = %d, want %d (weights + fixed margin)", rep.FloorBytes, wantFloor)
		}
	})

	t.Run("human render carries the upstream line shape", func(t *testing.T) {
		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{fixturePath})
		if code != 0 {
			t.Fatalf("runModelInfo exit = %d, stderr=%q, want 0", code, errb.String())
		}
		got := out.String()
		for _, want := range []string{
			"family: llama",
			"format: Q8_0",
			"context: 16",
			"weights: 0.00 GiB (139264 bytes)",
			"memory floor: 2.00 GiB (weights plus a 2.00 GiB runtime margin; not a live-serving-capacity claim)",
			"path: " + fixturePath,
		} {
			if !strings.Contains(got, want) {
				t.Errorf("render missing %q in:\n%s", want, got)
			}
		}
	})

	t.Run("cached alias resolves through the hf cache layout", func(t *testing.T) {
		target := modelreg.Catalog["smollm2"]
		ref, err := hfhub.ParseURI(target)
		if err != nil {
			t.Fatalf("ParseURI(%q): %v", target, err)
		}
		cachePath := hfhub.NewClient().CachePath(ref)
		data, err := os.ReadFile(fixturePath)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(cachePath), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(cachePath, data, 0o644); err != nil {
			t.Fatal(err)
		}

		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{"--json", "smollm2"})
		if code != 0 {
			t.Fatalf("runModelInfo exit = %d, stderr=%q, want 0", code, errb.String())
		}
		var rep modelInfoReport
		if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
			t.Fatalf("decode report: %v\n%s", err, out.String())
		}
		if rep.Ref != "smollm2" || rep.Target != target {
			t.Errorf("ref/target = %q/%q, want smollm2/%s", rep.Ref, rep.Target, target)
		}
		if rep.Path != cachePath {
			t.Errorf("path = %q, want the cache path %q", rep.Path, cachePath)
		}
		if rep.Family != "llama" || rep.WeightsBytes != wantWeights {
			t.Errorf("family/weights = %q/%d, want llama/%d", rep.Family, rep.WeightsBytes, wantWeights)
		}
	})

	t.Run("uncached ref refuses with the pull hint", func(t *testing.T) {
		// qwen2.5:0.5b gates on the listed family qwen2.5, so the refusal must
		// come from the missing cache entry, not the capability gate.
		path, _, _, err := resolveModelInfoTarget("qwen2.5:0.5b")
		if !errors.Is(err, ErrModelNotCached) {
			t.Fatalf("resolveModelInfoTarget error = %v, want ErrModelNotCached", err)
		}
		if path != "" {
			t.Errorf("path = %q, want empty for an uncached ref", path)
		}

		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{"qwen2.5:0.5b"})
		if code != 1 {
			t.Fatalf("runModelInfo exit = %d, want 1", code)
		}
		if !strings.Contains(errb.String(), "fak model pull qwen2.5:0.5b") {
			t.Errorf("stderr = %q, want the pull hint naming 'fak model pull qwen2.5:0.5b'", errb.String())
		}
		if out.Len() != 0 {
			t.Errorf("stdout = %q, want empty on refusal", out.String())
		}
	})

	t.Run("unknown ref names the model registry", func(t *testing.T) {
		_, _, _, err := resolveModelInfoTarget("definitely-not-a-model-zzz")
		var notFound *modelInfoNotFoundError
		if !errors.As(err, &notFound) {
			t.Fatalf("error = %v, want *modelInfoNotFoundError", err)
		}

		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{"definitely-not-a-model-zzz"})
		if code != 2 {
			t.Fatalf("runModelInfo exit = %d, want 2", code)
		}
		msg := errb.String()
		if !strings.Contains(msg, "model registry") || !strings.Contains(msg, "fak model ls") {
			t.Errorf("stderr = %q, want the registry named with a 'fak model ls' pointer", msg)
		}
	})

	t.Run("unregistered alias family refuses with the gate sentinel", func(t *testing.T) {
		reg := map[string]string{"exotic:7b": "hf://some-owner/some-repo/exotic-7b.gguf"}
		data, _ := json.Marshal(reg)
		if err := os.WriteFile(filepath.Join(root, "registry.json"), data, 0o644); err != nil {
			t.Fatal(err)
		}

		_, _, _, err := resolveModelInfoTarget("exotic:7b")
		if !errors.Is(err, ErrUnknownModelType) {
			t.Fatalf("error = %v, want ErrUnknownModelType chain", err)
		}

		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{"exotic:7b"})
		if code != 1 {
			t.Fatalf("runModelInfo exit = %d, want 1", code)
		}
		msg := errb.String()
		if !strings.Contains(msg, `"exotic"`) || !strings.Contains(msg, "capability registry") {
			t.Errorf("stderr = %q, want the type named against the capability registry", msg)
		}
	})

	t.Run("non-gguf artifact refuses with the preflight redirect", func(t *testing.T) {
		junk := filepath.Join(t.TempDir(), "config.json")
		if err := os.WriteFile(junk, []byte(`{"model_type":"llama"}`), 0o644); err != nil {
			t.Fatal(err)
		}

		// Resolution itself succeeds (the path exists); the typed refusal is
		// minted when the report reader sees the non-GGUF bytes.
		if _, _, _, rerr := resolveModelInfoTarget(junk); rerr != nil {
			t.Fatalf("resolveModelInfoTarget error = %v, want nil", rerr)
		}
		if _, err := readModelInfoReport(junk); !errors.Is(err, ErrModelFormatUnreadable) {
			t.Fatalf("readModelInfoReport error = %v, want ErrModelFormatUnreadable", err)
		}

		var out, errb bytes.Buffer
		code := runModelInfo(&out, &errb, []string{junk})
		if code != 1 {
			t.Fatalf("runModelInfo exit = %d, want 1", code)
		}
		if !strings.Contains(errb.String(), "fak model preflight --safetensors") {
			t.Errorf("stderr = %q, want the preflight redirect", errb.String())
		}
	})
}
