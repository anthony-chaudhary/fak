package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// findSafetensors locates the cached SmolLM2-135M model.safetensors in the HF hub.
func findSafetensors(t *testing.T) string {
	t.Helper()
	// -short skips the safetensors-backed witnesses for the same reason loadFixtureDir
	// does: decoding 134.5M params is the heavy, OOM-prone path. -short keeps the
	// synthetic + architest invariants for a fast pre-commit/pre-push gate.
	if testing.Short() {
		t.Skip("safetensors-backed witness skipped under -short (decode is the OOM/slow path)")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	g := filepath.Join(home, ".cache", "huggingface", "hub",
		"models--HuggingFaceTB--SmolLM2-135M-Instruct", "snapshots", "*", "model.safetensors")
	matches, _ := filepath.Glob(g)
	if len(matches) == 0 {
		t.Skip("no cached model.safetensors")
	}
	return matches[0]
}

// TestGoSafetensorsDecodeMatchesTorch is the rung-1a witness that makes the WEIGHTS
// Go-authored, not just the math. A pure-Go safetensors reader + bf16->f32 decoder
// must reproduce torch's decode BITWISE across all 134.5M parameters. bf16->f32 is a
// lossless 16-bit widening, so the only acceptable result is exact bit equality —
// this is NOT a tolerance check. Closes the "Go diffs torch's decode against itself"
// tautology the adversarial review flagged.
func TestGoSafetensorsDecodeMatchesTorch(t *testing.T) {
	if _, err := os.Stat(filepath.Join(cacheDir, "weights.f32")); err != nil {
		t.Skip("no torch export; run export_oracle.py")
	}
	torch, err := Load(cacheDir)
	if err != nil {
		t.Fatalf("load torch export: %v", err)
	}
	st, err := LoadSafetensors(findSafetensors(t), torch.Cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}

	var tensors, elems, mismatch int
	for name := range torch.manifest {
		a := torch.tensor(name)
		b := st.tensor(name)
		if len(a) != len(b) {
			t.Fatalf("%s len %d != %d", name, len(a), len(b))
		}
		for i := range a {
			if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
				mismatch++
				if mismatch <= 3 {
					t.Errorf("%s[%d] torch=%v go=%v (bits differ)", name, i, a[i], b[i])
				}
			}
		}
		tensors++
		elems += len(a)
	}
	t.Logf("Go bf16->f32 decode: %d tensors, %d params, %d bit-mismatches", tensors, elems, mismatch)
	if mismatch != 0 {
		t.Errorf("Go decode differs from torch in %d/%d params (must be 0 — lossless widening)", mismatch, elems)
	}
}

// TestForwardOnGoDecodedWeights proves the verified forward pass runs end-to-end on
// weights Go decoded itself from the raw safetensors — no torch in the load path —
// and still reproduces HF's greedy continuation token-for-token.
func TestForwardOnGoDecodedWeights(t *testing.T) {
	if _, err := os.Stat(filepath.Join(cacheDir, "weights.f32")); err != nil {
		t.Skip("no export")
	}
	var doc oracleDoc
	if err := readJSON(filepath.Join(cacheDir, "oracle.json"), &doc); err != nil {
		t.Fatalf("oracle: %v", err)
	}
	// load Cfg from the cache config (cheap), then weights purely from safetensors
	var cfg Config
	if err := readJSON(filepath.Join(cacheDir, "config.json"), &cfg); err != nil {
		t.Fatalf("config: %v", err)
	}
	st, err := LoadSafetensors(findSafetensors(t), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	p := doc.Prompts[0]
	got := st.NewSession().Generate(p.Ids, len(p.GreedyIds))
	t.Logf("go-decoded greedy=%v", got)
	t.Logf("hf greedy       =%v", p.GreedyIds)
	for i := range p.GreedyIds {
		if i < len(got) && got[i] != p.GreedyIds[i] {
			t.Errorf("token %d: go=%d hf=%d", i, got[i], p.GreedyIds[i])
			break
		}
	}
}
