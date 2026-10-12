package main

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/newmodel"
)

// writeV41HeaderShards gunzips the committed V4.1 header-only shards into dir
// under their real shard names and returns the shard-1 path.
func writeV41HeaderShards(t *testing.T, dir string) string {
	t.Helper()
	src := filepath.Join("..", "..", "internal", "ggufload", "testdata", "deepseek41_q2k_header")
	paths, err := filepath.Glob(filepath.Join(src, "*.gguf.gz"))
	if err != nil || len(paths) != 7 {
		t.Fatalf("fixture shards: %d, err %v", len(paths), err)
	}
	for _, p := range paths {
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		zr, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		hdr, err := io.ReadAll(zr)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, strings.TrimSuffix(filepath.Base(p), ".gz")), hdr, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return filepath.Join(dir, "DeepSeek-V4.1-Flash-Q2_K-00001-of-00007.gguf")
}

func v41HFConfig(t *testing.T, dir string, mutate func(map[string]any)) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "internal", "model", "testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	if mutate != nil {
		var m map[string]any
		if err := json.Unmarshal(raw, &m); err != nil {
			t.Fatal(err)
		}
		mutate(m)
		if raw, err = json.Marshal(m); err != nil {
			t.Fatal(err)
		}
	}
	p := filepath.Join(dir, "config.json")
	if err := os.WriteFile(p, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

// fak-test:runtime fast est=500ms
func TestRunModelPreflightV41HeaderReadyAndHFMismatchRefused(t *testing.T) {
	dir := t.TempDir()
	shard := writeV41HeaderShards(t, dir)

	var out, errOut bytes.Buffer
	cfg := v41HFConfig(t, dir, nil)
	if code := runModelPreflight(&out, &errOut, []string{"--gguf", shard, "--hf-config", cfg, "--json"}); code != 0 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var report ggufload.ManifestPreflight
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != ggufload.ManifestPreflightSchema || report.Verdict != ggufload.ManifestPreflightReady || len(report.Refusals) != 0 {
		t.Fatalf("report = %+v, want READY with no refusals", report)
	}

	out.Reset()
	cfg = v41HFConfig(t, dir, func(m map[string]any) {
		m["text_config"].(map[string]any)["compress_rope_theta"] = 0
	})
	if code := runModelPreflight(&out, &errOut, []string{"--gguf", shard, "--hf-config", cfg, "--json"}); code != 3 {
		t.Fatalf("mismatch exit=%d, want 3", code)
	}
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Refusals) != 1 || report.Refusals[0].Reason != newmodel.RefusalHFMismatch ||
		report.Refusals[0].Axis != "deepseek41.attention.compress_rope_freq_base" {
		t.Fatalf("refusals = %+v, want one HF_MISMATCH on compress_rope_freq_base", report.Refusals)
	}

	if code := runModelPreflight(&out, &errOut, nil); code != 2 {
		t.Fatalf("no --gguf exit=%d, want 2", code)
	}
}

// fak-test:runtime fast est=500ms
func TestNewModelManifestRunsHeaderPreflight(t *testing.T) {
	dir := t.TempDir()
	shard := writeV41HeaderShards(t, dir)
	manifest := filepath.Join("..", "..", "internal", "newmodel", "testdata", "qwen38-valid.json")
	cfg := v41HFConfig(t, dir, func(m map[string]any) {
		m["text_config"].(map[string]any)["hidden_size"] = 4096
	})
	var out, errOut bytes.Buffer
	if code := runNewModel(&out, &errOut, []string{"--from-manifest", manifest, "--json", "--gguf", shard, "--hf-config", cfg}); code != 3 {
		t.Fatalf("exit=%d stderr=%s, want 3", code, errOut.String())
	}
	var report ggufload.ManifestPreflight
	if err := json.Unmarshal(errOut.Bytes(), &report); err != nil {
		t.Fatalf("stderr is not the preflight report: %v\n%s", err, errOut.String())
	}
	if len(report.Refusals) != 1 || report.Refusals[0].Reason != newmodel.RefusalHFMismatch {
		t.Fatalf("refusals = %+v, want one HF_MISMATCH", report.Refusals)
	}
	if out.Len() != 0 {
		t.Fatal("a refused header still emitted the onboarding packet")
	}
}
