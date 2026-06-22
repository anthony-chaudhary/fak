// Command qwen35check loads a Qwen3.5 / Qwen3-Next hybrid HF snapshot with fak's own
// in-kernel forward pass and greedy-decodes a few tokens from a given prompt (token ids),
// printing the generated ids. It is the correctness witness for the Gated-DeltaNet
// linear-attention + gated full-attention support (qwen35.go / forward.go): the generated
// ids are detokenized and compared to llama.cpp's greedy continuation of the same prompt.
// The cached path uses Session.Prefill/Step: full-attention KV and Gated-DeltaNet recurrent
// state are both kernel-owned.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

type topLogit struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
}

type stepResult struct {
	Index   int        `json:"index"`
	TokenID int        `json:"token_id"`
	EOS     bool       `json:"eos"`
	Top     []topLogit `json:"top,omitempty"`
}

type modelSummary struct {
	Source             string  `json:"source"`
	Type               string  `json:"type"`
	Hidden             int     `json:"hidden"`
	Layers             int     `json:"layers"`
	Heads              int     `json:"heads"`
	KVHeads            int     `json:"kv_heads"`
	HeadDim            int     `json:"head_dim"`
	Intermediate       int     `json:"intermediate"`
	Vocab              int     `json:"vocab"`
	QuantLoaded        bool    `json:"quant_loaded"`
	Qwen35Hybrid       bool    `json:"qwen35_hybrid"`
	NormGain1p         bool    `json:"norm_gain_1p"`
	PartialRotary      float64 `json:"partial_rotary"`
	RopeTheta          float64 `json:"rope_theta"`
	AttnOutputGate     bool    `json:"attn_output_gate"`
	LinearConvKernel   int     `json:"linear_conv_kernel"`
	LinearKeyHeads     int     `json:"linear_key_heads"`
	LinearKeyHeadDim   int     `json:"linear_key_head_dim"`
	LinearValueHeads   int     `json:"linear_value_heads"`
	LinearValueHeadDim int     `json:"linear_value_head_dim"`
}

type checkResult struct {
	Model        modelSummary `json:"model"`
	PromptIDs    []int        `json:"prompt_ids"`
	GeneratedIDs []int        `json:"generated_ids"`
	ExpectedIDs  []int        `json:"expected_ids,omitempty"`
	ExpectMatch  *bool        `json:"expect_match,omitempty"`
	Steps        []stepResult `json:"steps"`
}

func argmax(v []float32) int {
	bi, best := 0, float32(-math.MaxFloat32)
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

func topKLogits(logits []float32, k int) []topLogit {
	if k <= 0 {
		return nil
	}
	top := make([]topLogit, len(logits))
	for i, v := range logits {
		top[i] = topLogit{ID: i, Logit: v}
	}
	sort.Slice(top, func(i, j int) bool {
		if top[i].Logit == top[j].Logit {
			return top[i].ID < top[j].ID
		}
		return top[i].Logit > top[j].Logit
	})
	if k > len(top) {
		k = len(top)
	}
	return top[:k]
}

func printTopK(top []topLogit) {
	if len(top) == 0 {
		return
	}
	fmt.Printf("TOP%d", len(top))
	for _, item := range top {
		fmt.Printf(" %d:%.6g", item.ID, item.Logit)
	}
	fmt.Println()
}

func parseIDList(s string) ([]int, error) {
	var ids []int
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		v, err := strconv.Atoi(part)
		if err != nil {
			return nil, fmt.Errorf("parse token id %q: %w", part, err)
		}
		ids = append(ids, v)
	}
	return ids, nil
}

func compareIDs(got, want []int) error {
	if len(got) != len(want) {
		return fmt.Errorf("generated %d ids, want %d: got=%v want=%v", len(got), len(want), got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("generated id mismatch at step %d: got=%d want=%d (got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}
	return nil
}

func writeResult(path string, res checkResult) error {
	b, err := json.MarshalIndent(res, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	if path == "" || path == "-" {
		_, err = os.Stdout.Write(b)
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "ERR:", err)
		os.Exit(1)
	}
}

func main() {
	dir := flag.String("dir", "", "HF snapshot dir (config.json + safetensors)")
	gguf := flag.String("gguf", "", "GGUF checkpoint path; loads through the memory-lean quant path")
	idsStr := flag.String("ids", "", "comma-separated prompt token ids")
	n := flag.Int("n", 16, "greedy tokens to generate")
	topK := flag.Int("topk", 0, "print the top-k logits before each greedy step")
	expectStr := flag.String("expect", "", "comma-separated expected generated ids; exits non-zero on mismatch")
	jsonOnly := flag.Bool("json", false, "write only the structured JSON result to stdout")
	out := flag.String("out", "", "write the structured JSON result to this path")
	flag.Parse()
	// Expand a leading ~ in path flags (Go/PowerShell don't), so ~/... opens as intended.
	*dir = pathutil.ExpandTilde(*dir)
	*gguf = pathutil.ExpandTilde(*gguf)

	var cfg model.Config
	var m *model.Model
	quantLoaded := false
	source := ""
	if (*dir == "") == (*gguf == "") {
		fmt.Fprintln(os.Stderr, "usage: qwen35check (-dir <hf-snapshot> | -gguf <model.gguf>) -ids <id,id,...> [-n N] [-topk K] [-expect id,id,...] [-json | -out path]")
		os.Exit(2)
	}
	q4kResident := false
	if *gguf != "" {
		source = *gguf
		f, err := ggufload.Open(*gguf)
		must(err)
		cfg, err = f.Config()
		must(err)
		if os.Getenv("FAK_Q4K") != "" {
			// Faithful resident-Q4_K path: stream the raw q4_k_m blocks llama.cpp
			// streams (no Q4→f32→Q8 round-trip). Run the session with Q4K=true.
			m, err = ggufload.LoadModelQ4K(*gguf)
			must(err)
			q4kResident = true
		} else {
			m, err = ggufload.LoadModelQuant(*gguf)
			must(err)
		}
		quantLoaded = true
	} else {
		source = *dir
		cb, err := os.ReadFile(filepath.Join(*dir, "config.json"))
		must(err)
		must(json.Unmarshal(cb, &cfg))
		m, err = model.LoadSafetensorsDir(*dir, cfg)
		must(err)
	}
	if !*jsonOnly {
		fmt.Printf("MODEL type=%s hidden=%d layers=%d heads=%d/%d headDim=%d inter=%d vocab=%d\n",
			cfg.ModelType, cfg.HiddenSize, cfg.NumLayers, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim, cfg.IntermediateSize, cfg.VocabSize)
		fmt.Printf("QWEN35 hybrid=%v normGain1p=%v partialRotary=%.4g ropeTheta=%.0f attnGate=%v conv=%d lin(k=%dx%d v=%dx%d)\n",
			cfg.IsQwen35Hybrid(), cfg.NormGain1p, cfg.PartialRotaryFactor, cfg.RopeTheta, cfg.AttnOutputGate,
			cfg.LinearConvKernelDim, cfg.LinearNumKeyHeads, cfg.LinearKeyHeadDim, cfg.LinearNumValueHeads, cfg.LinearValueHeadDim)

		fmt.Println("LOADED")
	}

	ids, err := parseIDList(*idsStr)
	must(err)
	if len(ids) == 0 {
		must(fmt.Errorf("-ids must contain at least one token id"))
	}
	expect, err := parseIDList(*expectStr)
	must(err)
	if len(expect) > 0 && *n < len(expect) {
		must(fmt.Errorf("-n=%d cannot satisfy %d expected ids", *n, len(expect)))
	}
	if !*jsonOnly {
		fmt.Printf("PROMPT_IDS %v\n", ids)
	}

	s := m.NewSession()
	s.Quant = quantLoaded
	s.Q4K = q4kResident
	logits := s.Prefill(ids)
	res := checkResult{
		Model: modelSummary{
			Source:             source,
			Type:               cfg.ModelType,
			Hidden:             cfg.HiddenSize,
			Layers:             cfg.NumLayers,
			Heads:              cfg.NumHeads,
			KVHeads:            cfg.NumKVHeads,
			HeadDim:            cfg.HeadDim,
			Intermediate:       cfg.IntermediateSize,
			Vocab:              cfg.VocabSize,
			QuantLoaded:        quantLoaded,
			Qwen35Hybrid:       cfg.IsQwen35Hybrid(),
			NormGain1p:         cfg.NormGain1p,
			PartialRotary:      cfg.PartialRotaryFactor,
			RopeTheta:          cfg.RopeTheta,
			AttnOutputGate:     cfg.AttnOutputGate,
			LinearConvKernel:   cfg.LinearConvKernelDim,
			LinearKeyHeads:     cfg.LinearNumKeyHeads,
			LinearKeyHeadDim:   cfg.LinearKeyHeadDim,
			LinearValueHeads:   cfg.LinearNumValueHeads,
			LinearValueHeadDim: cfg.LinearValueHeadDim,
		},
		PromptIDs: append([]int(nil), ids...),
	}
	if len(expect) > 0 {
		res.ExpectedIDs = append([]int(nil), expect...)
	}
	for i := 0; i < *n; i++ {
		top := topKLogits(logits, *topK)
		if !*jsonOnly {
			printTopK(top)
		}
		next := argmax(logits)
		res.GeneratedIDs = append(res.GeneratedIDs, next)
		ids = append(ids, next)
		eos := cfg.IsEOS(next)
		res.Steps = append(res.Steps, stepResult{
			Index:   i,
			TokenID: next,
			EOS:     eos,
			Top:     top,
		})
		if !*jsonOnly {
			fmt.Printf("STEP %2d -> %d\n", i, next)
		}
		if eos {
			if !*jsonOnly {
				fmt.Println("EOS")
			}
			break
		}
		logits = s.Step(next)
	}
	if !*jsonOnly {
		fmt.Printf("GEN_IDS %v\n", res.GeneratedIDs)
	}
	if len(expect) > 0 {
		if err := compareIDs(res.GeneratedIDs, expect); err != nil {
			match := false
			res.ExpectMatch = &match
			if *jsonOnly {
				must(writeResult("-", res))
			} else if *out != "" {
				must(writeResult(*out, res))
			}
			must(err)
		}
		match := true
		res.ExpectMatch = &match
	}
	if *jsonOnly {
		must(writeResult("-", res))
	} else if *out != "" {
		must(writeResult(*out, res))
	}
}
