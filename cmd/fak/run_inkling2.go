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
	f, err := ggufload.Open(path)
	if err != nil {
		return nil, nil, nil // Other loaders also admit directories and sharded GGUFs.
	}
	if arch, _ := f.String("general.architecture"); arch != "inkling2_loop" {
		return nil, nil, nil
	}
	be, metal, err := inkling2RunBackend(backend, metalFlag)
	if err != nil {
		return nil, nil, err
	}
	if err = applyNativeControls(be, cfg); err != nil {
		return nil, nil, err
	}
	table, ok := f.GGMLTokenizer()
	if !ok || table.EOS < 0 {
		return nil, nil, fmt.Errorf("Inkling 2: checkpoint needs an embedded BPE tokenizer and EOS")
	}
	tok, err := tokenizer.FromGGML(table.Tokens, table.Merges, table.TokenTypes, table.Pre)
	if err != nil {
		return nil, nil, err
	}
	for _, marker := range []string{"<|message_system|>", "<|message_user|>", "<|message_model|>", "<|content_text|>", "<|content_thinking|>", "<|end_message|>", "<|content_model_end_sampling|>"} {
		if _, ok := tok.SpecialID(marker); !ok {
			return nil, nil, fmt.Errorf("Inkling 2: checkpoint tokenizer lacks %s", marker)
		}
	}
	var m *ggufload.Inkling2LoopModel
	var target *inklingmetal.Model
	var release func()
	backendName := "cpu"
	if metal {
		backendName = "metal"
		useTarget := inklingmetal.Available()
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
		m, err = ggufload.OpenInkling2Loop(path)
		release = func() { _ = m.Close() }
	}
	if err != nil {
		return nil, nil, err
	}
	if target != nil {
		fmt.Fprintln(os.Stderr, "fak run: inkling2_loop backend=metal native target layers=24 passes=3")
		return agent.NewInkling2MetalRunPlanner(name, target, tok, table.EOS), release, nil
	}
	fmt.Fprintf(os.Stderr, "fak run: inkling2_loop backend=%s layers=%d passes=%d\n", backendName, m.HP.Layers, m.HP.Passes)
	return agent.NewInkling2RunPlanner(name, m, tok, table.EOS), release, nil
}

// Inkling uses packed MSL kernels directly, including on devices where the
// general MPS compute backend does not register.
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
		return nil, false, fmt.Errorf("Inkling 2 run: unsupported backend %q", be.Name())
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
