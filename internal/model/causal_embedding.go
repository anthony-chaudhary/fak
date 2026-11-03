// SPDX-License-Identifier: MIT
// Pooling/hidden-output contract adapted from llama.cpp cd8427d018c07eefab6c110f7f115c31ec4c6157,
// src/llama-graph.cpp (build_pooling/build_logits). See THIRD_PARTY_NOTICES.md.
package model

import (
	"context"
	"fmt"
	"math"
	"reflect"
)

// CausalEmbeddingMaxTokens is an admission bound, not a model context extension.
const CausalEmbeddingMaxTokens = 256

// ValidateCausalEmbedding admits only the ordinary dense, resident-F32/Q8_0 Llama
// forward. It never promotes a causal checkpoint to a trained embedding model.
// Config/weights must remain immutable after normal model publication.
func (m *Model) ValidateCausalEmbedding() error {
	if m == nil || !m.holdModelWeights() {
		return fmt.Errorf("embedding: model weights closing or unavailable")
	}
	defer m.releaseWeightSession()
	return m.validateCausalEmbedding()
}

func (m *Model) validateCausalEmbedding() error {
	if m == nil {
		return fmt.Errorf("embedding: no model")
	}
	c := m.Cfg
	if c.ModelType != "" && c.ModelType != "llama" {
		return fmt.Errorf("embedding: unsupported architecture %q", c.ModelType)
	}
	if c.HiddenSize <= 0 || c.HiddenSize > 16384 || c.NumLayers <= 0 || c.NumLayers > 128 ||
		c.NumHeads <= 0 || c.NumHeads > 256 || c.NumKVHeads <= 0 || c.NumKVHeads > c.NumHeads ||
		c.NumHeads%c.NumKVHeads != 0 || c.HeadDim <= 0 || c.HeadDim > 256 || c.HeadDim%2 != 0 ||
		c.NumHeads*c.HeadDim != c.HiddenSize || c.IntermediateSize <= 0 || c.IntermediateSize > 65536 ||
		c.VocabSize <= 0 || c.VocabSize > 262144 || c.MaxPositionEmbeddings < 0 ||
		math.IsNaN(c.RMSNormEps) || math.IsInf(c.RMSNormEps, 0) || c.RMSNormEps <= 0 ||
		math.IsNaN(c.RopeTheta) || math.IsInf(c.RopeTheta, 0) || c.RopeTheta <= 0 {
		return fmt.Errorf("embedding: invalid or unbounded dense geometry")
	}
	// Fail closed on additional architecture axes, including future Config fields.
	// These are the exact standard-Llama mechanics admitted by this provider.
	clean := c
	clean.HiddenSize = 0
	clean.NumLayers = 0
	clean.NumHeads = 0
	clean.NumKVHeads = 0
	clean.HeadDim = 0
	clean.IntermediateSize = 0
	clean.VocabSize = 0
	clean.RMSNormEps = 0
	clean.RopeTheta = 0
	clean.MaxPositionEmbeddings = 0
	clean.ModelType = ""
	clean.Name = ""
	clean.TieWordEmbeddings = false
	clean.EOSTokenID = 0
	clean.EOSTokenIDs = nil
	clean.BOSTokenID = nil
	clean.eosJSONDecoded = false
	for _, a := range clean.Architectures {
		if a != "LlamaForCausalLM" {
			return fmt.Errorf("embedding: unsupported architecture declaration")
		}
	}
	clean.Architectures = nil
	if clean.HiddenAct != "" && clean.HiddenAct != "silu" || clean.HiddenActivation != "" && clean.HiddenActivation != "silu" {
		return fmt.Errorf("embedding: unsupported activation")
	}
	clean.HiddenAct = ""
	clean.HiddenActivation = ""
	if clean.NumLoops == 1 {
		clean.NumLoops = 0
	}
	if clean.EmbedScale == 1 {
		clean.EmbedScale = 0
	}
	if clean.LogitScale == 1 {
		clean.LogitScale = 0
	}
	for _, w := range clean.Window {
		if w != -1 {
			return fmt.Errorf("embedding: nonstandard attention window")
		}
	}
	clean.Window = nil
	if !reflect.DeepEqual(clean, Config{}) {
		return fmt.Errorf("embedding: unsupported nonstandard dense architecture axes")
	}
	if len(m.q4w) != 0 || len(m.q4kw) != 0 || len(m.kqw) != 0 ||
		len(m.q2w) != 0 || len(m.awqw) != 0 || len(m.awqg) != 0 || len(m.gptqw) != 0 ||
		m.MLA != nil || m.Vision != nil || m.Q2KEmbedding != nil || m.expertCheckpoint != nil ||
		m.lora != nil || m.attnObs != nil || m.routeObs != nil || m.expertRouteObs != nil || m.expertRouteEventObs != nil ||
		m.epRanks > 1 || m.epRankSet || m.epCoord != nil || m.prism != nil {
		return fmt.Errorf("embedding: requires immutable dense F32/Q8_0 CPU weights without adapters, observers or collectives")
	}
	check := func(name string, dims ...int) error {
		t, ok := m.manifest[name]
		if !ok && len(dims) == 2 && name != "model.embed_tokens.weight" {
			q := m.q8w[name]
			if q == nil || q.out != dims[0] || q.in != dims[1] || q.in%qBlk != 0 || q.nblk != q.in/qBlk ||
				len(q.q) != q.out*q.in || len(q.d) != q.out*q.nblk {
				return fmt.Errorf("embedding: invalid resident Q8_0 tensor %s", name)
			}
			for _, v := range q.d {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return fmt.Errorf("embedding: nonfinite Q8_0 scale %s", name)
				}
			}
			return nil
		}
		if !ok || t.Dtype != "F32" || !reflect.DeepEqual(t.Shape, dims) {
			return fmt.Errorf("embedding: missing or non-F32 tensor %s", name)
		}
		n := int64(4)
		for _, d := range dims {
			n *= int64(d)
		}
		if n > int64(len(m.raw)) || t.Offset < 0 || t.Offset%4 != 0 || t.Nbytes != int(n) || t.Offset > len(m.raw)-int(n) {
			return fmt.Errorf("embedding: invalid tensor extent %s", name)
		}
		return nil
	}
	H, I, K := c.HiddenSize, c.IntermediateSize, c.NumKVHeads*c.HeadDim
	if err := check("model.embed_tokens.weight", c.VocabSize, H); err != nil {
		return err
	}
	if err := check("model.norm.weight", H); err != nil {
		return err
	}
	if m.has("model.norm.bias") {
		return fmt.Errorf("embedding: biased final norm unsupported")
	}
	for l := 0; l < c.NumLayers; l++ {
		for _, v := range []struct {
			suffix string
			dims   []int
		}{
			{"input_layernorm.weight", []int{H}}, {"post_attention_layernorm.weight", []int{H}},
			{"self_attn.q_proj.weight", []int{H, H}}, {"self_attn.k_proj.weight", []int{K, H}},
			{"self_attn.v_proj.weight", []int{K, H}}, {"self_attn.o_proj.weight", []int{H, H}},
			{"mlp.gate_proj.weight", []int{I, H}}, {"mlp.up_proj.weight", []int{I, H}}, {"mlp.down_proj.weight", []int{H, I}},
		} {
			if err := check(layerName(l, v.suffix), v.dims...); err != nil {
				return err
			}
		}
		for _, s := range []string{"input_layernorm.bias", "post_attention_layernorm.bias", "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias", "self_attn.o_proj.bias", "mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias", "mlp.gate.weight", "self_attn.sinks", "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm.bias", "post_feedforward_layernorm.weight", "post_feedforward_layernorm.bias"} {
			if m.has(layerName(l, s)) {
				return fmt.Errorf("embedding: unsupported tensor %s", layerName(l, s))
			}
		}
	}
	return nil
}

// CausalEmbedding returns an owned L2-normalized final-norm mean/last vector.
// It intentionally never constructs Activations history or invokes the LM head.
// Cancellation is checked at layer boundaries, not inside a running CPU kernel.
// The fixed numerical workspace admission is not a Go allocator/RSS guarantee.
func (m *Model) CausalEmbedding(ctx context.Context, ids []int, pooling string) ([]float32, error) {
	if ctx == nil {
		return nil, fmt.Errorf("embedding: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if pooling != "mean" && pooling != "last" {
		return nil, fmt.Errorf("embedding: pooling must be mean or last")
	}
	if m == nil || !m.holdModelWeights() {
		return nil, fmt.Errorf("embedding: model weights closing or unavailable")
	}
	defer m.releaseWeightSession()
	if err := m.validateCausalEmbedding(); err != nil {
		return nil, err
	}
	c := m.Cfg
	if len(ids) == 0 || len(ids) > CausalEmbeddingMaxTokens || c.MaxPositionEmbeddings > 0 && len(ids) > c.MaxPositionEmbeddings {
		return nil, fmt.Errorf("embedding: token length outside admitted context")
	}
	for _, id := range ids {
		if id < 0 || id >= c.VocabSize {
			return nil, fmt.Errorf("embedding: token id outside model vocabulary")
		}
	}
	S, H, I := int64(len(ids)), int64(c.HiddenSize), int64(c.IntermediateSize)
	// Conservative numerical admission includes hidden/norm/projection/MLP panels,
	// attention score work and RoPE. It does not count immutable model weights.
	if 64*S*(H+I)+8*S*S*int64(c.NumHeads) > 256<<20 {
		return nil, fmt.Errorf("embedding: numerical workspace exceeds admission limit")
	}
	if int64(c.NumLayers)*S*(8*H*H+6*H*I+4*S*H) > 1_000_000_000_000 {
		return nil, fmt.Errorf("embedding: dense work exceeds admission limit")
	}
	x := m.embedBand(ids)
	for l := 0; l < c.NumLayers; l++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		m.layer(l, x, newRopeForLayer(c, l, len(ids)))
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	sums := make([]float64, c.HiddenSize)
	begin := 0
	if pooling == "last" {
		begin = len(x) - 1
	}
	for t := begin; t < len(x); t++ {
		row := m.finalNorm(x[t])
		for j, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("embedding: nonfinite final hidden value")
			}
			sums[j] += float64(v)
		}
	}
	if pooling == "mean" {
		for j := range sums {
			sums[j] /= float64(len(x))
		}
	}
	norm := 0.0
	for _, v := range sums {
		norm = math.Hypot(norm, v)
	}
	if norm == 0 || math.IsNaN(norm) || math.IsInf(norm, 0) {
		return nil, fmt.Errorf("embedding: nonfinite or zero pooled norm")
	}
	out := make([]float32, len(sums))
	for j, v := range sums {
		out[j] = float32(v / norm)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// CausalEmbeddingPrecision describes the projections actually selected by the
// existing resident kernel. Norms, hidden state and embedding rows remain F32.
// It does not reinterpret the stored/source GGUF precision or claim quality.
func (m *Model) CausalEmbeddingPrecision() string {
	q8, f32 := 0, 0
	if m == nil {
		return "unknown"
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for _, s := range []string{"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"} {
			name := layerName(l, s)
			if _, ok := m.manifest[name]; ok {
				f32++
			} else if m.q8w[name] != nil {
				q8++
			}
		}
	}
	if q8 == 0 {
		return "f32"
	}
	if f32 == 0 {
		return "q8_0-projections-f32-state"
	}
	return "mixed-f32-q8_0-projections-f32-state"
}
