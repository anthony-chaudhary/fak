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
//	go run ./cmd/gpucheck -backend metal                            (darwin/arm64+cgo two-state Metal admission receipt)
//
// `-backend metal` does not decode: Metal is the CPU-session seam, not a compute HAL
// backend, so this command reports the two-state admission (not-compiled vs no-device,
// exit 1) or the verified device identity (exit 2) and points at the Q4_K parity dose in
// `cmd/modelbench -q4k -metal <gguf>`. A high-fidelity Metal witness belongs to that lane.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
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

// metalGuard folds the two-state Metal admission check so both failure arms are
// unit-testable without cgo: (false,false) and the (false,true) self-report gap both name
// the compile state (stub Compiled()==false wins); (true,false) names the device state;
// (true,true) proceeds, printing the shared device's identity once.
type metalGuardResult struct {
	exitCode int
	message  string
}

func metalGuard(compiled, available bool) metalGuardResult {
	switch {
	case !compiled:
		return metalGuardResult{exitCode: 1, message: "gpucheck: metal backend not compiled in (requires darwin/arm64 with cgo — auto-compiled, no build tag needed)"}
	case !available:
		return metalGuardResult{exitCode: 1, message: "gpucheck: metal compiled but no usable Metal device is available on this host"}
	default:
		return metalGuardResult{exitCode: 0, message: fmt.Sprintf("gpucheck: metal device %q", metalDeviceName)}
	}
}

// metalDeviceName is injected at the (true,true) call site; a package-level var keeps
// metalGuard pure (no direct metalgemm dependency) so cgo-free tests exercise the matrix.
var metalDeviceName = ""

func main() {
	hf := flag.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors)")
	backendName := flag.String("backend", "cuda", "device backend name (cuda, vulkan, metal)")
	n := flag.Int("n", 12, "greedy tokens to compare")
	lean := flag.Bool("lean", false, "memory-lean Q8 load (sharded dir); reference is the CPU Q8 path (use for 2.5-3B that won't fit f32)")
	vulkanQ4KProfile := flag.Bool("vulkan-q4k-profile", false, "enable Vulkan Q4_K timing profiles (requires -backend vulkan)")
	vulkanStageQ4K := flag.Bool("vulkan-stage-q4k", false, "use Vulkan host-visible Q4_K staging (requires -backend vulkan)")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*hf = pathutil.ExpandTilde(*hf)
	metalRequested := *backendName == "metal"
	if metalRequested {
		metalDeviceName = metalgemm.DeviceName()
		g := metalGuard(metalgemm.Compiled(), metalgemm.Available())
		fmt.Fprintln(os.Stderr, g.message)
		if g.exitCode == 0 {
			// Compiled + device present. The gpucheck micro-dose drives
			// compute.Backend device lanes; Metal is the CPU-session seam, and its
			// real Q4_K parity qualification lives in `cmd/modelbench -q4k -metal
			// <gguf>`. Report the honest device receipt and exit non-zero (2 =
			// verified device, witness lane not wired here) rather than running a
			// comparison this command cannot drive correctly — a fabricated pass
			// would be worse than an explicit not-wired signal.
			fmt.Fprintln(os.Stderr, "gpucheck: metal is compiled and available; the Q4_K parity micro-dose lives in `cmd/modelbench -q4k -metal <gguf>` (Session.Metal prefill vs CPU Q8 reference)")
			os.Exit(2)
		}
		os.Exit(g.exitCode)
	}
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
	if (*vulkanQ4KProfile || *vulkanStageQ4K) && !compute.ConfigureVulkanQ4K(be, *vulkanQ4KProfile, *vulkanStageQ4K) {
		fmt.Fprintln(os.Stderr, "-vulkan-q4k-profile/-vulkan-stage-q4k require -backend vulkan")
		os.Exit(2)
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
		m, lerr := model.LoadSafetensorsQuantDir(*hf, cfg)
		if lerr != nil {
			fmt.Fprintln(os.Stderr, "lean load:", lerr)
			os.Exit(1)
		}
		rs := m.NewSession()
		rs.Quant = true
		ref = rs.Generate(prompt, *n)
		dev = m.NewBackendSession(be).Generate(prompt, *n)
		refLabel = "cpu-Q8"
	} else {
		m, lerr := model.LoadSafetensors(filepath.Join(*hf, "model.safetensors"), cfg)
		if lerr != nil {
			fmt.Fprintln(os.Stderr, "load:", lerr)
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
	fmt.Printf("model=%s backend=%s f16=%v q8=%v\n", filepath.Base(*hf), *backendName,
		os.Getenv("FAK_CUDA_F16") == "1", os.Getenv("FAK_CUDA_Q8") == "1")
	fmt.Printf("ref(%s):       %v\n", refLabel, ref)
	fmt.Printf("dev(%s approx): %v\n", *backendName, dev)
	if mismatch >= 0 {
		fmt.Printf("MISMATCH at token %d: ref=%d dev=%d\n", mismatch, ref[mismatch], dev[mismatch])
		os.Exit(1)
	}
	fmt.Printf("OK — greedy argmax-exact over %d tokens vs %s (real-model Approx-gate witness)\n", *n, refLabel)
}
