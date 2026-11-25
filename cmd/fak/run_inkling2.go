package main

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/inklingmetal"
	"github.com/anthony-chaudhary/fak/internal/localadmission"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// The looped forward owns independent KV/conv slots for each layer visit.
func openInkling2Run(path, name, backend string, metalFlag bool, cfg nativeControlConfig) (agent.Planner, func(), error) {
	arch, err := serveInkling2Arch(path)
	if err != nil || arch == "" {
		return nil, nil, nil // Other loaders also admit directories and other architectures.
	}
	shards := serveInkling2ShardSet(path)
	f, err := ggufload.Open(shards[0]) // only the first split shard carries the tokenizer
	if err != nil {
		return nil, nil, err
	}
	be, metal, err := inkling2RunBackend(backend, metalFlag)
	if err != nil {
		return nil, nil, err
	}
	if err = applyNativeControls(be, cfg); err != nil {
		return nil, nil, err
	}
	tok, eos, err := runInkling2Tokenizer(f)
	if err != nil {
		return nil, nil, err
	}
	if metal && len(shards) > 1 {
		return nil, nil, fmt.Errorf("Inkling 2: the Metal path binds a single-file GGUF; %s is split into %d shards", path, len(shards))
	}
	var m *ggufload.Inkling2LoopModel
	var target *inklingmetal.Model
	var release func()
	backendName := "cpu"
	if metal {
		backendName = "metal"
		useTarget, e := inkling2MetalUseVendored(os.Getenv(inkling2MetalTargetEnv), arch == serveInkling2LoopArch, inklingmetal.Available())
		if e != nil {
			return nil, nil, e
		}
		var weights localadmission.MemoryPlan
		if useTarget {
			// Match the launcher's default single-session context budget, including
			// the direct target's packed images, verification scratch and KV extents.
			planned, e := inklingmetal.Estimate(path, gateway.DefaultAdmissionPolicy().TokenBudget)
			if e != nil {
				return nil, nil, e
			}
			if planned.Bytes() > math.MaxInt64 {
				return nil, nil, fmt.Errorf("Inkling 2: Metal plan exceeds address space")
			}
			weights.SteadyBytes = int64(planned.Bytes())
			weights.StartupPeakBytes, _ = metalServeStartupPeakBytes(serveLoadArmResidentQ4K, weights.SteadyBytes, 0, false)
		} else {
			weights, err = inkling2RunMetalMemory(path, f)
			if err != nil {
				return nil, nil, err
			}
		}
		var loadErr error
		release, err = admitLocalMetalModel(true, path, gpulease.Options{}, metalAdmissionSpec{
			Weights: &weights,
			Load: func() bool {
				if useTarget {
					target, loadErr = inklingmetal.Open(path)
				} else {
					m, loadErr = ggufload.OpenInkling2LoopMetal(path)
				}
				return loadErr == nil
			},
			Teardown: func() error {
				if target != nil {
					return target.Close()
				}
				return m.Close()
			},
		})
		if err == nil {
			err = loadErr
		}
	} else {
		// CUDA binds under the free-VRAM budget and spills cold experts to host;
		// with no device backend the host forward runs over the mapped shards.
		var l *serveInkling2Loaded
		if l, err = loadServeInkling2(path, be, os.Stderr); err == nil {
			m = l.Model
			release = func() { _ = l.Close() }
			if be != nil {
				backendName = be.Name()
			}
		}
	}
	if err != nil {
		return nil, nil, err
	}
	if target != nil {
		fmt.Fprintf(os.Stderr, "fak run: inkling2_loop backend=metal target=%s layers=24 passes=3\n", inkling2MetalTargetVendoredLabel)
		return agent.NewInkling2MetalRunPlanner(name, target, tok, eos), release, nil
	}
	if metal {
		backendName += " target=" + inkling2MetalTargetNativeLabel
	}
	fmt.Fprintf(os.Stderr, "fak run: %s backend=%s layers=%d passes=%d\n", arch, backendName, m.HP.Layers, m.HP.Passes)
	if be != nil {
		return agent.NewInkling2DeviceRunPlanner(name, m, be, tok, eos), release, nil
	}
	return agent.NewInkling2RunPlanner(name, m, tok, eos), release, nil
}

const (
	inkling2MetalTargetEnv           = "FAK_INKLING_METAL_TARGET"
	inkling2MetalTargetVendoredLabel = "vendored-splash"
	inkling2MetalTargetNativeLabel   = "fak-metalgemm"
)

// inkling2MetalUseVendored picks between the vendored direct target and fak's
// own packed-kernel forward. Only an unset selector may fall back: an explicit
// request that cannot be honoured is refused so a measurement never reports
// one path under the other's name.
func inkling2MetalUseVendored(selector string, loopArch, vendoredAvailable bool) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(selector)) {
	case "":
		return loopArch && vendoredAvailable, nil
	case "vendored":
		if !loopArch {
			return false, fmt.Errorf("Inkling 2: %s=vendored needs the %s architecture", inkling2MetalTargetEnv, serveInkling2LoopArch)
		}
		if !vendoredAvailable {
			return false, fmt.Errorf("Inkling 2: %s=vendored but the vendored Metal target is not available in this build or on this device", inkling2MetalTargetEnv)
		}
		return true, nil
	case "native":
		return false, nil
	}
	return false, fmt.Errorf("Inkling 2: unknown %s=%q (want vendored or native)", inkling2MetalTargetEnv, selector)
}

// runInkling2Tokenizer is a seam so tests can bind a toy vocab to a synthetic GGUF.
var runInkling2Tokenizer = func(f *ggufload.File) (*tokenizer.Tokenizer, int, error) {
	table, ok := f.GGMLTokenizer()
	if !ok || table.EOS < 0 {
		return nil, 0, fmt.Errorf("Inkling 2: checkpoint needs an embedded BPE tokenizer and EOS")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		return nil, 0, err
	}
	for _, marker := range []string{"<|message_system|>", "<|message_user|>", "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>"} {
		if _, ok := tok.SpecialID(marker); !ok {
			return nil, 0, fmt.Errorf("Inkling 2: checkpoint tokenizer lacks %s", marker)
		}
	}
	return tok, table.EOS, nil
}

// Inkling uses packed MSL kernels directly, including on devices where the
// general MPS compute backend does not register. CUDA takes the budgeted
// device bind; every other device backend is refused.
func inkling2RunBackend(name string, metalFlag bool) (compute.Backend, bool, error) {
	selector := strings.TrimSpace(name)
	if selector == "" {
		selector = strings.TrimSpace(os.Getenv("FAK_BACKEND"))
	}
	if strings.EqualFold(selector, "metal") {
		if !metalgemm.Available() {
			return nil, false, metalgemm.ErrMetalUnavailable
		}
		return nil, true, nil
	}
	be, err := resolveServeChatBackend(name)
	if err != nil {
		return nil, false, err
	}
	if be != nil {
		if be.Name() != "cuda" {
			return nil, false, fmt.Errorf("Inkling 2 run: unsupported backend %q", be.Name())
		}
		return be, false, nil
	}
	metal, err := resolveRunMetal(metalFlag, os.Getenv("FAK_METAL") != "", name)
	return be, metal, err
}

// Weight views map packed projections directly. Count an additional F32 copy
// for decoded vectors, convolution taps, controller queries and unknown tensors.
func inkling2RunMetalMemory(path string, f *ggufload.File) (localadmission.MemoryPlan, error) {
	st, err := os.Stat(path)
	if err != nil {
		return localadmission.MemoryPlan{}, err
	}
	steady := st.Size()
	for _, t := range f.Tensors {
		if inkling2RunMappedWeight(t.Name) {
			continue
		}
		var bytes int64 = 4
		for _, n := range t.Dims {
			if n == 0 || n > uint64(math.MaxInt64/bytes) {
				return localadmission.MemoryPlan{}, fmt.Errorf("Inkling 2: invalid dimensions for %s", t.Name)
			}
			bytes *= int64(n)
		}
		if bytes > math.MaxInt64-steady {
			return localadmission.MemoryPlan{}, fmt.Errorf("Inkling 2: weight plan overflow at %s", t.Name)
		}
		steady += bytes
	}
	peak, _ := metalServeStartupPeakBytes(serveLoadArmResidentQ4K, steady, 0, false)
	return localadmission.MemoryPlan{SteadyBytes: steady, StartupPeakBytes: peak}, nil
}

func inkling2RunMappedWeight(name string) bool {
	if name == "token_embd.weight" || name == "output.weight" {
		return true
	}
	block, ok := strings.CutPrefix(name, "blk.")
	if !ok {
		return false
	}
	_, field, ok := strings.Cut(block, ".")
	if !ok {
		return false
	}
	switch field {
	case "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_gate.weight", "attn_rel.weight", "attn_sink_proj.weight",
		"attn_k_conv_proj.weight", "attn_v_conv_proj.weight", "ffn_gate_inp.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight":
		return true
	}
	controller, ok := strings.CutPrefix(field, "loop_")
	if !ok {
		return false
	}
	channel, projection, ok := strings.Cut(controller, "_ctrl_")
	if !ok {
		return false
	}
	switch channel {
	case "q", "k", "v", "ffn":
		return projection == "down.weight" || projection == "gain.weight" || projection == "forget.weight"
	}
	return false
}
