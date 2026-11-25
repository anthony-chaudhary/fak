// Package inklingmetal binds FAK's Inkling sessions to in-process Apple GPU operators.
package inklingmetal

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
)

const Vocabulary = 200058
const MaxBlock = 32

// MaxLanes bounds the sessions one batched decode step advances.
const MaxLanes = 8

var ErrUnavailable = errors.New("Inkling Metal requires an Apple GPU and macOS 26.4 or later")

// Open uses the adjacent DFlash2 checkpoint when the model directory includes it.
func Open(path string) (*Model, error) {
	draft, err := adjacentDraft(path)
	if err != nil {
		return nil, err
	}
	return OpenWithDraft(path, draft)
}

// An empty draft path requests target-only inference.
func OpenWithDraft(path, draft string) (*Model, error) {
	if draft != "" {
		data, err := os.ReadFile(filepath.Join(draft, "config.json"))
		if err != nil {
			return nil, err
		}
		var c struct {
			Architectures []string `json:"architectures"`
			Layers        int      `json:"num_hidden_layers"`
			Intermediate  int      `json:"intermediate_size"`
			Hidden        int      `json:"hidden_size"`
			Vocab         int      `json:"vocab_size"`
			Head          int      `json:"head_dim"`
			Heads         int      `json:"num_attention_heads"`
			KV            int      `json:"num_key_value_heads"`
			Window        int      `json:"sliding_window"`
			Causal        bool     `json:"is_causal"`
			Epsilon       float64  `json:"rms_norm_eps"`
			Rope          struct {
				Theta float64 `json:"rope_theta"`
				Type  string  `json:"rope_type"`
			} `json:"rope_parameters"`
			Draft struct {
				Taps    []int   `json:"target_layer_ids"`
				Mask    int     `json:"mask_token_id"`
				Block   int     `json:"block_size"`
				Rank    int     `json:"selector_rank"`
				Top     int     `json:"selector_top_k"`
				Group   int     `json:"conv_group_size"`
				Kernel  int     `json:"conv_kernel_size"`
				Softcap float64 `json:"final_logit_softcapping"`
				Mode    string  `json:"selector_softcap_mode"`
			} `json:"dflash_config"`
		}
		if err = json.Unmarshal(data, &c); err != nil {
			return nil, fmt.Errorf("Inkling draft config: %w", err)
		}
		d := c.Draft
		if !slices.Contains(c.Architectures, "DFlash2DraftModel") || c.Layers != 5 || c.Intermediate != 6912 || c.Hidden != 2304 || c.Vocab != 201024 || c.Head != 128 || c.Heads != 32 || c.KV != 8 || c.Window != 2048 || c.Causal || c.Epsilon != 1e-6 || c.Rope.Theta != 10000 || c.Rope.Type != "default" || d.Block != 8 || d.Mask != 200056 || d.Group != 16 || d.Kernel != 2 || d.Rank != 256 || d.Top != 16 || d.Softcap != 30 || d.Mode != "after_correction" || !slices.Equal(d.Taps, []int{5, 23, 41, 48}) {
			return nil, fmt.Errorf("incompatible Inkling DFlash2 configuration in %s", draft)
		}
	}
	return openModel(path, draft)
}

// MemoryPlan sizes the native model images and one request to its context limit.
type MemoryPlan struct {
	TargetImages, DraftImages     uint64
	PrefillScratch, DecodeScratch uint64
	State, KV                     uint64
}

func (p MemoryPlan) Bytes() uint64 {
	return p.TargetImages + p.DraftImages + p.PrefillScratch + p.DecodeScratch + p.State + p.KV
}

// Estimate plans packed checkpoint images and Metal arenas without loading model tensors.
func Estimate(path string, contextTokens int) (MemoryPlan, error) {
	draft, err := adjacentDraft(path)
	if err != nil {
		return MemoryPlan{}, err
	}
	return estimateMemory(path, draft, contextTokens)
}

// EstimateWithDraft plans like Estimate for an explicit draft; an empty draft
// plans target-only inference.
func EstimateWithDraft(path, draft string, contextTokens int) (MemoryPlan, error) {
	return estimateMemory(path, draft, contextTokens)
}
