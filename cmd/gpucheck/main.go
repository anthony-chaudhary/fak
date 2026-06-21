// Command gpucheck is the real-model correctness witness for the GPU backend: it loads a
// HuggingFace safetensors checkpoint (e.g. Qwen2.5-1.5B-Instruct) and greedily decodes the
// SAME prompt twice — once on the legacy pure-Go f32 path (the proven bit-exact reference)
// and once through the compute HAL on a device backend (e.g. cuda, optionally with
// FAK_CUDA_F16=1) — then asserts the two greedy token streams agree. This is the Approx
// gate (argmax-exact) applied to a real multi-billion-parameter model, not a synthetic one:
// the synthetic witness lives in internal/model/hal_cuda_test.go; this proves the same
// property survives real Qwen weights at f16 on the GPU.
//
//	go run -tags cuda ./cmd/gpucheck -hf <dir> -backend cuda -n 12   (set FAK_CUDA_F16=1 for f16)
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func readCfg(dir string) (model.Config, error) {
	var cfg model.Config
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return cfg, err
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		return cfg, err
	}
	if cfg.HeadDim == 0 && cfg.NumHeads != 0 {
		cfg.HeadDim = cfg.HiddenSize / cfg.NumHeads
	}
	return cfg, nil
}

func main() {
	hf := flag.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors)")
	backendName := flag.String("backend", "cuda", "device backend name")
	n := flag.Int("n", 12, "greedy tokens to compare")
	lean := flag.Bool("lean", false, "memory-lean Q8 load (sharded dir); reference is the CPU Q8 path (use for 2.5-3B that won't fit f32)")
	flag.Parse()
	if *hf == "" {
		fmt.Fprintln(os.Stderr, "need -hf")
		os.Exit(2)
	}
	cfg, err := readCfg(*hf)
	if err != nil {
		fmt.Fprintln(os.Stderr, "config:", err)
		os.Exit(1)
	}
	be, ok := compute.Lookup(*backendName)
	if !ok {
		fmt.Fprintf(os.Stderr, "backend %q not registered (have %v)\n", *backendName, compute.Registered())
		os.Exit(1)
	}
	// A fixed, arbitrary prompt; correctness of the forward pass is token-value-independent,
	// so deterministic ids exercise the identical arithmetic a real prompt would.
	prompt := []int{9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911}
	for i := range prompt {
		prompt[i] %= cfg.VocabSize
	}
	var ref, dev []int
	var refLabel string
	if *lean {
		// Lean Q8 load: the f32 weights are dropped, so the reference is the CPU Q8 lane (the
		// model's proven quantized forward), not f32. Both sides are Q8-weight; argmax must agree.
		m, err := model.LoadSafetensorsQuantDir(*hf, cfg)
		if err != nil {
			fmt.Fprintln(os.Stderr, "lean load:", err)
			os.Exit(1)
		}
		rs := m.NewSession()
		rs.Quant = true
		ref = rs.Generate(prompt, *n)
		dev = m.NewBackendSession(be).Generate(prompt, *n)
		refLabel = "cpu-Q8"
	} else {
		m, err := model.LoadSafetensors(filepath.Join(*hf, "model.safetensors"), cfg)
		if err != nil {
			fmt.Fprintln(os.Stderr, "load:", err)
			os.Exit(1)
		}
		ref = m.NewSession().Generate(prompt, *n)          // legacy pure-Go f32 reference
		dev = m.NewBackendSession(be).Generate(prompt, *n) // HAL device path (Approx)
		refLabel = "f32 legacy"
	}
	mismatch := -1
	for i := 0; i < *n && i < len(ref) && i < len(dev); i++ {
		if ref[i] != dev[i] {
			mismatch = i
			break
		}
	}
	fmt.Printf("model=%s backend=%s f16=%v q8=%v\n", filepath.Base(*hf), be.Name(),
		os.Getenv("FAK_CUDA_F16") == "1", os.Getenv("FAK_CUDA_Q8") == "1")
	fmt.Printf("ref(%s):       %v\n", refLabel, ref)
	fmt.Printf("dev(%s approx): %v\n", be.Name(), dev)
	if mismatch >= 0 {
		fmt.Printf("MISMATCH at token %d: ref=%d dev=%d\n", mismatch, ref[mismatch], dev[mismatch])
		os.Exit(1)
	}
	fmt.Printf("OK — greedy argmax-exact over %d tokens vs %s (real-model Approx-gate witness)\n", *n, refLabel)
}
