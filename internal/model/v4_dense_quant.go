package model

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// quantizeV4DenseFP8TensorInto consumes the dense FP8 pair published by
// DeepSeek-V4-Flash-0731 at revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062:
// an E4M3 weight and its E8M0 sibling scale. The scale sorts before the weight,
// so its first visit validates and defers the pair; the weight visit performs one
// decode and installs only the Q8 resident tensor.
func quantizeV4DenseFP8TensorInto(
	name string,
	hdr map[string]json.RawMessage,
	tensorBytes func(stEntry) ([]byte, error),
	m *Model,
	consumed map[string]bool,
) (bool, error) {
	if m == nil || !isDeepSeekV4FlashProfile(m.Cfg) {
		return false, nil
	}
	role, ok, err := v4DenseFP8RoleForName(name, m.Cfg)
	if err != nil {
		return true, err
	}
	if !ok {
		return false, nil
	}
	weightName := role.weightName
	scaleName := strings.TrimSuffix(weightName, ".weight") + ".scale"
	weightEntry, scaleEntry, err := v4DenseFP8PairEntries(weightName, scaleName, role.shape, hdr)
	if err != nil {
		return true, err
	}
	if name == scaleName {
		return true, nil
	}

	weights, err := tensorBytes(weightEntry)
	if err != nil {
		return true, fmt.Errorf("safetensors: tensor %s: %w", weightName, err)
	}
	scales, err := tensorBytes(scaleEntry)
	if err != nil {
		return true, fmt.Errorf("safetensors: tensor %s: %w", scaleName, err)
	}
	decoded, err := decodeV4DenseFP8(weightName, weightEntry.Shape, weights, scales)
	if err != nil {
		return true, err
	}
	m.q8w[weightName] = quantizeQ8(decoded, weightEntry.Shape[0], weightEntry.Shape[1])
	consumed[weightName] = true
	consumed[scaleName] = true
	return true, nil
}

func v4DenseFP8PairEntries(weightName, scaleName string, wantWeight [2]int, hdr map[string]json.RawMessage) (stEntry, stEntry, error) {
	weightRaw, ok := hdr[weightName]
	if !ok {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense FP8 pair is missing %s", weightName)
	}
	scaleRaw, ok := hdr[scaleName]
	if !ok {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense FP8 pair %s is missing %s", weightName, scaleName)
	}
	var weightEntry, scaleEntry stEntry
	if err := json.Unmarshal(weightRaw, &weightEntry); err != nil {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: entry %s: %w", weightName, err)
	}
	if err := json.Unmarshal(scaleRaw, &scaleEntry); err != nil {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: entry %s: %w", scaleName, err)
	}
	if weightEntry.Dtype != "F8_E4M3" {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense weight %s dtype %q, want F8_E4M3", weightName, weightEntry.Dtype)
	}
	if scaleEntry.Dtype != "F8_E8M0" {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense scale %s dtype %q, want F8_E8M0", scaleName, scaleEntry.Dtype)
	}
	if len(weightEntry.Shape) != 2 {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense weight %s shape %v, want rank-2", weightName, weightEntry.Shape)
	}
	if !sameShape(weightEntry.Shape, wantWeight[:]) {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense weight %s shape %v, want published shape %v", weightName, weightEntry.Shape, wantWeight)
	}
	rows, cols := weightEntry.Shape[0], weightEntry.Shape[1]
	if rows <= 0 || cols <= 0 || cols%qBlk != 0 {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense weight %s shape %v, want positive dimensions and input divisible by %d", weightName, weightEntry.Shape, qBlk)
	}
	wantScale := []int{(rows-1)/fp8BlockDim + 1, (cols-1)/fp8BlockDim + 1}
	if !sameShape(scaleEntry.Shape, wantScale) {
		return stEntry{}, stEntry{}, fmt.Errorf("safetensors: V4 dense scale %s shape %v, want %v for weight shape %v", scaleName, scaleEntry.Shape, wantScale, weightEntry.Shape)
	}
	return weightEntry, scaleEntry, nil
}

func decodeV4DenseFP8(name string, shape []int, weights, scales []byte) ([]float32, error) {
	rows, cols := shape[0], shape[1]
	wantWeights, ok := checkedShapeProduct(rows, cols)
	if !ok || len(weights) != wantWeights {
		return nil, fmt.Errorf("safetensors: V4 dense weight %s has %d bytes, shape %v implies %d", name, len(weights), shape, wantWeights)
	}
	wantScales, ok := checkedShapeProduct((rows-1)/fp8BlockDim+1, (cols-1)/fp8BlockDim+1)
	if !ok || len(scales) != wantScales {
		return nil, fmt.Errorf("safetensors: V4 dense scale for %s has %d bytes, want %d", name, len(scales), wantScales)
	}
	for i, value := range weights {
		if decoded := fp8E4M3ToF32(value); math.IsNaN(float64(decoded)) || math.IsInf(float64(decoded), 0) {
			return nil, fmt.Errorf("safetensors: V4 dense weight %s byte %d is E4M3 NaN", name, i)
		}
	}
	scaleValues := make([]float32, len(scales))
	for i, value := range scales {
		if value == 0xff {
			return nil, fmt.Errorf("safetensors: V4 dense scale for %s byte %d is E8M0 NaN", name, i)
		}
		scaleValues[i] = float32(math.Ldexp(1, int(value)-127))
		if math.IsInf(float64(scaleValues[i]), 0) {
			return nil, fmt.Errorf("safetensors: V4 dense scale for %s byte %d is non-finite", name, i)
		}
	}
	decoded, err := decodeFP8BlockScale(name, rows, cols, weights, scaleValues)
	if err != nil {
		return nil, err
	}
	for i, value := range decoded {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("safetensors: V4 dense weight %s decoded value %d is non-finite", name, i)
		}
	}
	return decoded, nil
}

type v4DenseFP8Role struct {
	weightName string
	shape      [2]int
}

func v4DenseFP8RoleForName(name string, cfg Config) (v4DenseFP8Role, bool, error) {
	weightName := name
	if strings.HasSuffix(name, ".scale") {
		weightName = strings.TrimSuffix(name, ".scale") + ".weight"
	}
	rest, ok := strings.CutPrefix(weightName, "layers.")
	if !ok {
		return v4DenseFP8Role{}, false, nil
	}
	layerText, suffix, ok := strings.Cut(rest, ".")
	if !ok || !decimalOnly(layerText) {
		return v4DenseFP8Role{}, false, nil
	}
	layer, err := strconv.Atoi(layerText)
	if err != nil || layer < 0 || layer >= cfg.NumLayers {
		return v4DenseFP8Role{}, true, fmt.Errorf("safetensors: V4 dense FP8 tensor %s has layer %q outside [0,%d)", name, layerText, cfg.NumLayers)
	}
	var shape [2]int
	switch suffix {
	case "attn.wq_a.weight":
		shape = [2]int{1024, 4096}
	case "attn.wq_b.weight":
		shape = [2]int{32768, 1024}
	case "attn.wkv.weight":
		shape = [2]int{512, 4096}
	case "attn.wo_a.weight":
		shape = [2]int{8192, 4096}
	case "attn.wo_b.weight":
		shape = [2]int{4096, 8192}
	case "ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight":
		shape = [2]int{2048, 4096}
	case "ffn.shared_experts.w2.weight":
		shape = [2]int{4096, 2048}
	case "attn.indexer.wq_b.weight":
		if v4CompressionRatio(cfg, layer) != 4 {
			return v4DenseFP8Role{}, true, fmt.Errorf("safetensors: V4 indexer tensor %s belongs to layer %d with compression ratio %d, want 4", name, layer, v4CompressionRatio(cfg, layer))
		}
		shape = [2]int{8192, 1024}
	default:
		return v4DenseFP8Role{}, false, nil
	}
	return v4DenseFP8Role{weightName: weightName, shape: shape}, true, nil
}

func v4CompressionRatio(cfg Config, layer int) int {
	if layer < 0 || layer >= len(cfg.CompressRatios) {
		return -1
	}
	return cfg.CompressRatios[layer]
}
