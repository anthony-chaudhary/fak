package main

import (
	"context"
	"fmt"
	"math"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/benchids"
	"github.com/anthony-chaudhary/fak/internal/cmdutil"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// loadHF loads a HuggingFace snapshot directory (config.json + single-file or sharded
// safetensors) entirely in Go — the pure-Go safetensors reader + bf16->f32 decode in
// internal/model, no torch in the loop. It is what lets fak run any Llama/Qwen2-family
// checkpoint on this box without the export_oracle.py (torch) step: the generic config-driven
// forward pass already handles GQA, RoPE theta, SwiGLU, tied embeddings, and Qwen2 qkv-bias.
// Returns the model and a display name derived from model_type + parameter scale.
func loadHF(dir string) (*model.Model, string, error) {
	return loadHFWith(dir, "safetensors", "", model.LoadSafetensorsDir)
}

// loadHFLean loads via the memory-lean quantize-at-load path (f32 of the big weights dropped),
// the loader that lets a 7B-class model fit on this box. Quant-only: the bench forces -quant.
func loadHFLean(dir string) (*model.Model, string, error) {
	return loadHFWith(dir, "safetensors(lean)", " [lean]", func(d string, c model.Config) (*model.Model, error) {
		return model.LoadSafetensorsQuantDir(d, c)
	})
}

// loadHFWith reads the HF config from dir and loads the model via load, wrapping a load
// failure as "<label>: <err>" and returning the hfName display string with nameSuffix
// appended. It is the shared body of loadHF (full) and loadHFLean (memory-lean) which
// differ only by loader, error label, and name suffix.
func loadHFWith(dir, label, nameSuffix string, load func(string, model.Config) (*model.Model, error)) (*model.Model, string, error) {
	cfg, err := benchcli.ReadHFConfig(dir)
	if err != nil {
		return nil, "", err
	}
	m, err := load(dir, cfg)
	if err != nil {
		return nil, "", fmt.Errorf("%s: %w", label, err)
	}
	return m, hfName(cfg, dir) + nameSuffix, nil
}

func loadGGUF(path string) (*model.Model, string, error) {
	m, err := ggufload.LoadModel(path)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + " [gguf]", nil
}

func loadGGUFLean(path string, lp *ggufload.LoadProfiler) (*model.Model, string, error) {
	m, err := ggufload.LoadModelQuantProfile(path, lp)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + " [gguf-lean]", nil
}

type q4kModelLoader func(context.Context, string, *ggufload.LoadProfiler) (*model.Model, error)

var (
	loadResidentQ4K q4kModelLoader = func(ctx context.Context, path string, _ *ggufload.LoadProfiler) (*model.Model, error) {
		return ggufload.LoadModelQ4KContext(ctx, path)
	}
	loadStreamedDenseQ4K q4kModelLoader = func(ctx context.Context, path string, p *ggufload.LoadProfiler) (*model.Model, error) {
		return ggufload.LoadModelQ4KStreamedDenseContext(ctx, path, p)
	}
)

func loadGGUFQ4K(path string, lp *ggufload.LoadProfiler, streamed bool) (*model.Model, string, error) {
	return loadGGUFQ4KContext(context.Background(), path, lp, streamed)
}

func loadGGUFQ4KContext(ctx context.Context, path string, lp *ggufload.LoadProfiler, streamed bool) (*model.Model, string, error) {
	loader := loadResidentQ4K
	label := " [gguf-q4k]"
	if streamed {
		loader = loadStreamedDenseQ4K
		label = " [gguf-q4k-streamed-dense]"
	}
	m, err := loader(ctx, path, lp)
	if err != nil {
		return nil, "", err
	}
	return m, filepath.Base(path) + label, nil
}

// hfName builds a report label like "qwen2-1.5B" from the config (param count is approximated
// from the dominant weight shapes), falling back to the directory basename.
func hfName(cfg model.Config, dir string) string {
	base := filepath.Base(strings.TrimRight(dir, "/"))
	if cfg.ModelType == "" {
		return base
	}
	return fmt.Sprintf("%s (%s)", base, cfg.ModelType)
}

func logitTop2(v []float32) (top1Idx int, top1, top2 float32) {
	top1, top2 = float32(-math.MaxFloat32), float32(-math.MaxFloat32)
	for i, x := range v {
		if x > top1 {
			top2 = top1
			top1, top1Idx = x, i
		} else if x > top2 {
			top2 = x
		}
	}
	return top1Idx, top1, top2
}

func cosineF32(a, b []float32) float64 {
	var dot, aa, bb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		aa += x * x
		bb += y * y
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / math.Sqrt(aa*bb)
}

func tryF32Prefill(m *model.Model, ids []int) (logits []float32, ok bool) {
	defer func() {
		if recover() != nil {
			logits, ok = nil, false
		}
	}()
	s := m.NewSession()
	return s.Prefill(ids), true
}

// lcgIDs builds n deterministic token ids in [0,vocab) via a glibc LCG. The exact
// same recurrence is reproduced in bench_hf.py so both engines see identical input.
func lcgIDs(n, vocab int) []int {
	return lcgIDsSeed(n, vocab, 2463534242)
}

func lcgIDsSeed(n, vocab int, seed uint64) []int {
	return benchids.LCGFromState(n, vocab, seed)
}

func medianMS(ds []time.Duration) float64 {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[len(cp)/2].Nanoseconds()) / 1e6
}

type prefillResult struct {
	Name           string  `json:"name,omitempty"`
	Source         string  `json:"source,omitempty"`
	Tokens         int     `json:"tokens"`
	RecordedTokens int     `json:"recorded_tokens,omitempty"`
	Reps           int     `json:"reps"`
	MedianMS       float64 `json:"median_ms"`
	TokPerSec      float64 `json:"tok_per_sec"`
}

type decodeResult struct {
	PromptTokens  int     `json:"prompt_tokens"`
	DecodeSteps   int     `json:"decode_steps"`
	Reps          int     `json:"reps"`
	PerTokenMedMS float64 `json:"per_token_median_ms"`
	TokPerSec     float64 `json:"tok_per_sec"`
}

type workloadDecodeResult struct {
	Name                 string  `json:"name"`
	Source               string  `json:"source,omitempty"`
	PromptTokens         int     `json:"prompt_tokens"`
	RecordedPromptTokens int     `json:"recorded_prompt_tokens,omitempty"`
	DecodeSteps          int     `json:"decode_steps"`
	RecordedDecodeTokens int     `json:"recorded_decode_tokens"`
	Reps                 int     `json:"reps"`
	PerTokenMedMS        float64 `json:"per_token_median_ms"`
	TokPerSec            float64 `json:"tok_per_sec"`
}

func capPositive(n, cap int) int {
	return cmdutil.CapPositive(n, cap)
}
