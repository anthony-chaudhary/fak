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
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

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
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

func run(args []string, stdout, stderr io.Writer) int {
	jsonMode := hasArg(args, "-json", "--json")
	fs := flag.NewFlagSet("gpucheck", flag.ContinueOnError)
	if jsonMode {
		fs.SetOutput(io.Discard)
	} else {
		fs.SetOutput(stderr)
	}
	hf := fs.String("hf", "", "HuggingFace snapshot dir (config.json + model.safetensors)")
	backendName := fs.String("backend", "cuda", "device backend name (cuda, vulkan, metal)")
	n := fs.Int("n", 12, "greedy tokens to compare")
	lean := fs.Bool("lean", false, "memory-lean Q8 load (sharded dir); reference is the CPU Q8 path (use for 2.5-3B that won't fit f32)")
	vulkanQ4KProfile := fs.Bool("vulkan-q4k-profile", false, "enable Vulkan Q4_K timing profiles (requires -backend vulkan)")
	vulkanStageQ4K := fs.Bool("vulkan-stage-q4k", false, "use Vulkan host-visible Q4_K staging (requires -backend vulkan)")
	jsonOutput := fs.Bool("json", false, "emit one machine-readable gpucheck receipt on stdout")
	requireNative := fs.Bool("require-native", false, "require the observed session backend to match the requested device-memory backend")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		if jsonMode {
			if emitJSON(stdout, failureReceipt("", "cuda", 0, reasonInvalidArguments, err.Error())) != nil {
				return 1
			}
		}
		return 2
	}
	jsonMode = *jsonOutput
	fail := func(code int, reasonCode parityReasonCode, reason string) int {
		if jsonMode {
			if emitJSON(stdout, failureReceipt(filepath.Base(*hf), *backendName, *n, reasonCode, reason)) != nil {
				return 1
			}
		} else {
			fmt.Fprintln(stderr, reason)
		}
		return code
	}
	if *n <= 0 {
		return fail(2, reasonInvalidTokenCount, "-n must be positive")
	}
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*hf = pathutil.ExpandTilde(*hf)
	metalRequested := *backendName == "metal"
	if metalRequested {
		metalDeviceName = metalgemm.DeviceName()
		g := metalGuard(metalgemm.Compiled(), metalgemm.Available())
		if jsonMode && g.exitCode != 0 {
			if emitJSON(stdout, failureReceipt(filepath.Base(*hf), *backendName, *n, reasonBackendUnavailable, g.message)) != nil {
				return 1
			}
		} else if !jsonMode {
			fmt.Fprintln(stderr, g.message)
		}
		if g.exitCode == 0 {
			// Compiled + device present. The gpucheck micro-dose drives
			// compute.Backend device lanes; Metal is the CPU-session seam, and its
			// real Q4_K parity qualification lives in `cmd/modelbench -q4k -metal
			// <gguf>`. Report the honest device receipt and exit non-zero (2 =
			// verified device, witness lane not wired here) rather than running a
			// comparison this command cannot drive correctly — a fabricated pass
			// would be worse than an explicit not-wired signal.
			message := "gpucheck: metal is compiled and available; the Q4_K parity micro-dose lives in `cmd/modelbench -q4k -metal <gguf>` (Session.Metal prefill vs CPU Q8 reference)"
			if jsonMode {
				if emitJSON(stdout, failureReceipt(filepath.Base(*hf), *backendName, *n, reasonMetalWitnessNotHere, message)) != nil {
					return 1
				}
			} else {
				fmt.Fprintln(stderr, message)
			}
			return 2
		}
		return g.exitCode
	}
	if *hf == "" {
		return fail(2, reasonInvalidArguments, "need -hf")
	}
	cfg, err := readCfg(*hf)
	if err != nil {
		return fail(1, reasonConfiguration, "config: "+err.Error())
	}
	if cfg.VocabSize <= 0 {
		return fail(1, reasonInvalidVocabulary, "config: vocabulary size must be positive")
	}
	be, ok := compute.Lookup(*backendName)
	if !ok {
		return fail(1, reasonBackendUnavailable, fmt.Sprintf("backend %q not registered (have %v)", *backendName, compute.Registered()))
	}
	if (*vulkanQ4KProfile || *vulkanStageQ4K) && !compute.ConfigureVulkanQ4K(be, *vulkanQ4KProfile, *vulkanStageQ4K) {
		return fail(2, reasonUnsupportedOptions, "-vulkan-q4k-profile/-vulkan-stage-q4k require -backend vulkan")
	}
	// A fixed, arbitrary prompt; correctness of the forward pass is token-value-independent,
	// so deterministic ids exercise the identical arithmetic a real prompt would.
	prompt := []int{9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911}
	for i := range prompt {
		prompt[i] %= cfg.VocabSize
	}
	var ref, dev []int
	var refLabel string
	var observed backendIdentity
	if *lean {
		// Lean Q8 load: the f32 weights are dropped, so the reference is the CPU Q8 lane (the
		// model's proven quantized forward), not f32. Both sides are Q8-weight; argmax must agree.
		// A backend that cannot stage quantized uploads would fall back to the dropped f32
		// weights and panic, so it is refused before the load.
		if !be.Caps().UploadDtype {
			return fail(2, reasonUnsupportedOptions, fmt.Sprintf("-lean needs a backend that stages quantized uploads; %q does not", be.Name()))
		}
		m, lerr := model.LoadSafetensorsQuantDir(*hf, cfg)
		if lerr != nil {
			return fail(1, reasonModelLoad, "lean load: "+lerr.Error())
		}
		defer m.CloseWeights()
		rs := m.NewSession()
		defer rs.Close()
		rs.Quant = true
		ds, sessionErr := m.NewBackendSessionChecked(be)
		if sessionErr != nil {
			return fail(1, reasonBackendUnavailable, "backend session: "+sessionErr.Error())
		}
		defer ds.Close()
		// The lean load kept only Q8 matmul weights, so the device session must stage them.
		ds.Quant = true
		observed = sessionBackendIdentity(ds)
		ref = rs.Generate(prompt, *n)
		dev = ds.Generate(prompt, *n)
		refLabel = "cpu-Q8"
	} else {
		m, lerr := model.LoadSafetensors(filepath.Join(*hf, "model.safetensors"), cfg)
		if lerr != nil {
			return fail(1, reasonModelLoad, "load: "+lerr.Error())
		}
		defer m.CloseWeights()
		rs := m.NewSession()
		defer rs.Close()
		ds, sessionErr := m.NewBackendSessionChecked(be)
		if sessionErr != nil {
			return fail(1, reasonBackendUnavailable, "backend session: "+sessionErr.Error())
		}
		defer ds.Close()
		observed = sessionBackendIdentity(ds)
		ref = rs.Generate(prompt, *n) // legacy pure-Go f32 reference
		dev = ds.Generate(prompt, *n) // HAL device path (Approx)
		refLabel = "f32 legacy"
	}
	receipt := evaluateParity(parityInput{
		Requested:        *n,
		VocabSize:        cfg.VocabSize,
		Reference:        ref,
		Device:           dev,
		RequestedBackend: *backendName,
		Observed:         observed,
		ReferenceLabel:   refLabel,
		RequireNative:    *requireNative,
		IsEOS:            cfg.IsEOS,
	})
	receipt.Model = filepath.Base(*hf)
	if jsonMode {
		if emitJSON(stdout, receipt) != nil {
			return 1
		}
		if receipt.Pass {
			return 0
		}
		return 1
	}
	fmt.Fprintf(stdout, "model=%s backend=%s f16=%v q8=%v\n", receipt.Model, *backendName,
		os.Getenv("FAK_CUDA_F16") == "1", os.Getenv("FAK_CUDA_Q8") == "1")
	fmt.Fprintf(stdout, "ref(%s):       %v\n", refLabel, ref)
	fmt.Fprintf(stdout, "dev(%s approx): %v\n", *backendName, dev)
	if !receipt.Pass {
		if mismatch := firstMismatch(ref, dev); mismatch >= 0 {
			fmt.Fprintf(stdout, "MISMATCH at token %d: ref=%d dev=%d\n", mismatch, ref[mismatch], dev[mismatch])
		} else {
			fmt.Fprintf(stdout, "FAIL — %s\n", receipt.Reason)
		}
		return 1
	}
	fmt.Fprintf(stdout, "OK — greedy argmax-exact over %d tokens vs %s (real-model Approx-gate witness)\n", receipt.ComparedTokens, refLabel)
	return 0
}

func hasArg(args []string, names ...string) bool {
	for _, arg := range args {
		for _, name := range names {
			if arg == name || strings.HasPrefix(arg, name+"=") {
				return true
			}
		}
	}
	return false
}

func failureReceipt(modelName, backend string, requested int, reasonCode parityReasonCode, reason string) parityReceipt {
	r := newParityReceipt(parityInput{Requested: requested, RequestedBackend: backend})
	r.Model = modelName
	r.ReasonCode = reasonCode
	r.Reason = reason
	return r
}

func emitJSON(w io.Writer, receipt parityReceipt) error {
	return json.NewEncoder(w).Encode(receipt)
}

func sessionBackendIdentity(s *model.Session) backendIdentity {
	if s == nil || s.Backend == nil {
		return backendIdentity{}
	}
	return backendIdentity{Name: s.Backend.Name(), Tier: s.Backend.Tier(), DeviceMemory: s.Backend.Caps().DeviceMemory}
}

func firstMismatch(ref, dev []int) int {
	n := len(ref)
	if len(dev) < n {
		n = len(dev)
	}
	for i := 0; i < n; i++ {
		if ref[i] != dev[i] {
			return i
		}
	}
	return -1
}
