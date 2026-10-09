package model

// Explicit head targeting and weight_scale layout are adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/5375babbac9977cdb8f061cec77b6efd0987a1fd/python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py
// and schemes/compressed_tensors_w8a16_fp8.py at the same revision.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Copyright 2023-2024 SGLang Team
//
// Modified for Fak: admit only an explicitly named untied, tensor-scaled
// E4M3FN head with floating-point body weights, then convert bounded blocks
// into the existing Q8 runtime. This is partial fak#12417, not native FP8
// execution, channel/INT4 support, or KV-cache scheme support. The pinned
// upstream's channel-only W8A16 descriptor is rejected by its earlier scheme
// branch; do not infer channel admission from the shared scale parameter code.

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
)

type compressedHeadGroup struct {
	Targets []string `json:"targets"`
	Format  string   `json:"format"`
	Weights struct {
		Bits           int             `json:"num_bits"`
		Type           string          `json:"type"`
		Strategy       string          `json:"strategy"`
		Symmetric      *bool           `json:"symmetric"`
		Dynamic        *bool           `json:"dynamic"`
		GroupSize      json.RawMessage `json:"group_size"`
		BlockStructure json.RawMessage `json:"block_structure"`
		ActOrder       json.RawMessage `json:"actorder"`
	} `json:"weights"`
	Input  json.RawMessage `json:"input_activations"`
	Output json.RawMessage `json:"output_activations"`
}

func compressedHeadMetadataPresent(b json.RawMessage) bool {
	return len(b) != 0 && !bytes.Equal(bytes.TrimSpace(b), []byte("null"))
}

// Configuration objects must not hide conflicting fields behind JSON's
// last-key-wins map decoding. The shared safetensors header/index parser still
// collapses duplicate JSON keys; rejection there is outside this loader slice.
func compressedHeadUniqueKeys(raw []byte) error {
	d := json.NewDecoder(bytes.NewReader(raw))
	var value func() error
	value = func() error {
		token, err := d.Token()
		if err != nil {
			return err
		}
		switch token {
		case json.Delim('{'):
			seen := map[string]bool{}
			for d.More() {
				key, err := d.Token()
				if err != nil {
					return err
				}
				name, ok := key.(string)
				if !ok || seen[name] {
					return fmt.Errorf("duplicate or invalid field %v", key)
				}
				seen[name] = true
				if err := value(); err != nil {
					return err
				}
			}
			_, err = d.Token()
			return err
		case json.Delim('['):
			for d.More() {
				if err := value(); err != nil {
					return err
				}
			}
			_, err = d.Token()
			return err
		}
		return nil
	}
	return value()
}

// Use the upstream's name-only exact/dotted-suffix/re.match contract. In
// particular the module-type target "Linear" cannot activate an LM head.
func compressedHeadTargetMatches(target string) (bool, error) {
	const layer = "lm_head"
	if strings.HasPrefix(target, "re:") {
		r, err := regexp.Compile("^(?:" + strings.TrimPrefix(target, "re:") + ")")
		if err != nil {
			return false, fmt.Errorf("safetensors: invalid compressed-tensors target %q: %w", target, err)
		}
		return r.MatchString(layer), nil
	}
	return target == layer || strings.HasSuffix(layer, "."+target), nil
}

func compressedTensorsLMHeadEnabled(cfg Config) (bool, error) {
	if !compressedHeadMetadataPresent(cfg.QuantizationConfig) {
		return false, nil
	}
	if err := compressedHeadUniqueKeys(cfg.QuantizationConfig); err != nil {
		return false, fmt.Errorf("safetensors: quantization_config: %w", err)
	}
	var method struct {
		Method string `json:"quant_method"`
	}
	if err := json.Unmarshal(cfg.QuantizationConfig, &method); err != nil {
		return false, fmt.Errorf("safetensors: quantization_config: %w", err)
	}
	if method.Method != "compressed-tensors" {
		return false, nil
	}
	var q struct {
		Method string                         `json:"quant_method"`
		Format string                         `json:"format"`
		Status string                         `json:"quantization_status"`
		Ignore []string                       `json:"ignore"`
		Groups map[string]compressedHeadGroup `json:"config_groups"`
		KV     json.RawMessage                `json:"kv_cache_scheme"`
	}
	if err := json.Unmarshal(cfg.QuantizationConfig, &q); err != nil {
		return false, fmt.Errorf("safetensors: quantization_config: %w", err)
	}
	if q.Method != "compressed-tensors" {
		return false, nil
	}
	for _, target := range q.Ignore {
		match, err := compressedHeadTargetMatches(target)
		if err != nil {
			return false, err
		}
		if match {
			return false, nil
		}
	}
	var selected *compressedHeadGroup
	for _, group := range q.Groups {
		matches := false
		for _, target := range group.Targets {
			match, err := compressedHeadTargetMatches(target)
			if err != nil {
				return false, err
			}
			matches = matches || match
		}
		if !matches {
			continue
		}
		if selected != nil {
			return false, fmt.Errorf("safetensors: ambiguous compressed-tensors lm_head groups")
		}
		selected = &group
	}
	if selected == nil {
		return false, nil
	}
	w := selected.Weights
	if cfg.TieWordEmbeddings || q.Format != "float-quantized" ||
		(selected.Format != "" && selected.Format != q.Format) ||
		(q.Status != "" && q.Status != "compressed") ||
		w.Bits != 8 || w.Type != "float" || w.Strategy != "tensor" ||
		w.Symmetric == nil || !*w.Symmetric || w.Dynamic == nil || *w.Dynamic ||
		compressedHeadMetadataPresent(w.GroupSize) || compressedHeadMetadataPresent(w.BlockStructure) ||
		compressedHeadMetadataPresent(w.ActOrder) || compressedHeadMetadataPresent(selected.Input) ||
		compressedHeadMetadataPresent(selected.Output) || compressedHeadMetadataPresent(q.KV) {
		return false, fmt.Errorf("safetensors: unsupported compressed-tensors lm_head; want untied, static symmetric tensor-scaled FP8 weights only")
	}
	return true, nil
}

func quantizeCompressedTensorsLMHead(hdr map[string]json.RawMessage, tensorBytes func(stEntry) ([]byte, error), m *Model) error {
	const name, scaleName = "lm_head.weight", "lm_head.weight_scale"
	var weightEntry, scaleEntry stEntry
	if err := json.Unmarshal(hdr[name], &weightEntry); err != nil {
		return fmt.Errorf("safetensors: entry %s: %w", name, err)
	}
	if _, ok := hdr[scaleName]; !ok {
		return fmt.Errorf("safetensors: compressed-tensors %s requires same-shard %s", name, scaleName)
	}
	if err := json.Unmarshal(hdr[scaleName], &scaleEntry); err != nil {
		return fmt.Errorf("safetensors: entry %s: %w", scaleName, err)
	}
	out, in := m.Cfg.VocabSize, m.Cfg.HiddenSize
	elems, ok := checkedShapeProduct(out, in)
	if !ok || out <= 0 || in <= 0 || in%qBlk != 0 ||
		weightEntry.Dtype != "F8_E4M3" || len(weightEntry.Shape) != 2 ||
		weightEntry.Shape[0] != out || weightEntry.Shape[1] != in {
		return fmt.Errorf("safetensors: compressed-tensors %s must be F8_E4M3 [%d,%d] with Q8-aligned hidden size", name, out, in)
	}
	if scaleEntry.Dtype != "F32" || len(scaleEntry.Shape) != 1 || scaleEntry.Shape[0] != 1 {
		return fmt.Errorf("safetensors: compressed-tensors %s must be F32 [1]", scaleName)
	}
	weight, err := tensorBytes(weightEntry)
	if err != nil {
		return fmt.Errorf("safetensors: tensor %s: %w", name, err)
	}
	scales, err := tensorBytes(scaleEntry)
	if err != nil {
		return fmt.Errorf("safetensors: tensor %s: %w", scaleName, err)
	}
	if len(weight) != elems || len(scales) != 4 {
		return fmt.Errorf("safetensors: compressed-tensors head payload does not match declared shape")
	}
	scale := math.Float32frombits(binary.LittleEndian.Uint32(scales))
	if scale <= 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return fmt.Errorf("safetensors: compressed-tensors head scale must be finite and positive")
	}
	// Validate all products before allocating/publishing resident state. This
	// also catches E4M3FN NaN codes and finite scales that overflow multiplication.
	for _, code := range weight {
		v := fp8E4M3Lookup(code) * scale
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("safetensors: compressed-tensors head has nonfinite scaled weight")
		}
	}
	qt := newQ8Tensor(out, in, in/qBlk)
	parFor(out, currentWorkerCount(), func(lo, hi int) {
		var block [qBlk]float32
		for o := lo; o < hi; o++ {
			for b := 0; b < qt.nblk; b++ {
				base := o*in + b*qBlk
				for i := range block {
					block[i] = fp8E4M3Lookup(weight[base+i]) * scale
				}
				quantizeRowQ8scalar(block[:], qt.q[base:base+qBlk], qt.d[o*qt.nblk+b:o*qt.nblk+b+1], 1)
			}
		}
	})
	q8PrepareAccelWeight(qt)
	m.q8w[name] = qt
	return nil
}
