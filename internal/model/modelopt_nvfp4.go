package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrModelOptNVFP4SourceClosed is returned when Weight is called after the
// source has been closed.
var ErrModelOptNVFP4SourceClosed = errors.New("model: ModelOpt NVFP4 source is closed")

// ModelOptNVFP4Source exposes packed NVFP4 weights from one ModelOpt
// safetensors file through compute.WeightSource. It deliberately does not parse
// a model config or translate architecture-specific tensor names: callers use
// the checkpoint's exact base name.
//
// The two admitted sibling-name families are pinned to NVIDIA ModelOpt commit
// 0638a274cf195870564d8e3c164e4f859e5be2a2 (Apache-2.0):
//
//   - Hugging Face export: unified_export_hf.py registers weight_scale and
//     weight_scale_2, and normalizes any TensorRT-produced scale swizzle back to
//     the linear ModelOpt layout before export.
//   - TensorRT-LLM export: model_config.py and model_config_utils.py emit
//     weights_scaling_factor and weights_scaling_factor_2.
//
// https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/export/unified_export_hf.py
// https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/export/trtllm/model_config.py
// https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/export/trtllm/model_config_utils.py
type ModelOptNVFP4Source struct {
	mu       sync.RWMutex
	file     *safetensorsFile
	backend  compute.Backend
	closed   bool
	closeErr error
}

// OpenModelOptNVFP4Source opens a single ModelOpt safetensors file. Sharded
// checkpoint indices and full-model config/name wiring are intentionally outside
// this source primitive.
func OpenModelOptNVFP4Source(path string, be compute.Backend) (*ModelOptNVFP4Source, error) {
	if be == nil {
		return nil, fmt.Errorf("model: nil backend for ModelOpt NVFP4 source")
	}
	file, err := openSafetensorsFile(path)
	if err != nil {
		return nil, err
	}
	return &ModelOptNVFP4Source{file: file, backend: be}, nil
}

// Close releases the safetensors file or mapping. Tensors returned earlier stay
// valid because compute.NewNVFP4 copies both packed values and block scales.
func (s *ModelOptNVFP4Source) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.closeErr
	}
	s.closed = true
	if s.file != nil {
		s.closeErr = s.file.Close()
		s.file = nil
	}
	return s.closeErr
}

// Weight returns one packed NVFP4 tensor. name may be either the exact base name
// or that name with the final ".weight" suffix. Only compute.NVFP4 is admitted.
func (s *ModelOptNVFP4Source) Weight(name string, want compute.Dtype) (compute.Tensor, error) {
	if s == nil {
		return compute.Tensor{}, ErrModelOptNVFP4SourceClosed
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed || s.file == nil {
		return compute.Tensor{}, ErrModelOptNVFP4SourceClosed
	}
	if want != compute.NVFP4 {
		return compute.Tensor{}, fmt.Errorf("model: ModelOpt NVFP4 source cannot serve %s as %s", name, want)
	}

	base := strings.TrimSuffix(name, ".weight")
	if base == "" || strings.HasSuffix(base, ".weight_scale") ||
		strings.HasSuffix(base, ".weight_scale_2") ||
		strings.HasSuffix(base, ".weights_scaling_factor") ||
		strings.HasSuffix(base, ".weights_scaling_factor_2") {
		return compute.Tensor{}, fmt.Errorf("model: invalid ModelOpt NVFP4 weight name %q", name)
	}
	weightName := base + ".weight"
	scaleName, scale2Name, err := modelOptNVFP4ScaleNames(s.file.hdr, base)
	if err != nil {
		return compute.Tensor{}, err
	}

	weightEntry, err := modelOptNVFP4Entry(s.file.hdr, weightName)
	if err != nil {
		return compute.Tensor{}, err
	}
	scaleEntry, err := modelOptNVFP4Entry(s.file.hdr, scaleName)
	if err != nil {
		return compute.Tensor{}, err
	}
	scale2Entry, err := modelOptNVFP4Entry(s.file.hdr, scale2Name)
	if err != nil {
		return compute.Tensor{}, err
	}

	rows, cols, err := modelOptNVFP4Shapes(weightName, scaleName, weightEntry, scaleEntry)
	if err != nil {
		return compute.Tensor{}, err
	}
	if scale2Entry.Dtype != "F32" ||
		!(len(scale2Entry.Shape) == 0 || (len(scale2Entry.Shape) == 1 && scale2Entry.Shape[0] == 1)) {
		return compute.Tensor{}, fmt.Errorf("model: ModelOpt NVFP4 tensor %s must be an F32 scalar, got dtype %q shape %v", scale2Name, scale2Entry.Dtype, scale2Entry.Shape)
	}

	packed, err := s.file.tensorBytes(weightEntry)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("model: read %s: %w", weightName, err)
	}
	scales, err := s.file.tensorBytes(scaleEntry)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("model: read %s: %w", scaleName, err)
	}
	scale2Bytes, err := s.file.tensorBytes(scale2Entry)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("model: read %s: %w", scale2Name, err)
	}
	wantPacked, okPacked := checkedShapeProduct(rows, cols/2)
	wantScales, okScales := checkedShapeProduct(rows, cols/compute.NVFP4BlockSize)
	if !okPacked || !okScales || len(packed) != wantPacked || len(scales) != wantScales {
		return compute.Tensor{}, fmt.Errorf("model: malformed ModelOpt NVFP4 payload for %s: packed=%d/%d scales=%d/%d", weightName, len(packed), wantPacked, len(scales), wantScales)
	}
	if len(scale2Bytes) != 4 {
		return compute.Tensor{}, fmt.Errorf("model: ModelOpt NVFP4 tensor %s has %d bytes, want 4", scale2Name, len(scale2Bytes))
	}
	globalScale := math.Float32frombits(binary.LittleEndian.Uint32(scale2Bytes))
	if math.IsNaN(float64(globalScale)) || math.IsInf(float64(globalScale), 0) || globalScale < 0 {
		return compute.Tensor{}, fmt.Errorf("model: ModelOpt NVFP4 tensor %s must be finite and non-negative", scale2Name)
	}
	return compute.NewNVFP4(s.backend, []int{rows, cols}, packed, scales, globalScale)
}

func modelOptNVFP4ScaleNames(hdr map[string]json.RawMessage, base string) (string, string, error) {
	hfScale := base + ".weight_scale"
	hfScale2 := base + ".weight_scale_2"
	trtScale := base + ".weights_scaling_factor"
	trtScale2 := base + ".weights_scaling_factor_2"
	_, hf1 := hdr[hfScale]
	_, hf2 := hdr[hfScale2]
	_, trt1 := hdr[trtScale]
	_, trt2 := hdr[trtScale2]
	if (hf1 || hf2) && (trt1 || trt2) {
		return "", "", fmt.Errorf("model: ambiguous mixed ModelOpt NVFP4 scale naming for %s", base)
	}
	if hf1 || hf2 {
		if !hf1 || !hf2 {
			return "", "", fmt.Errorf("model: incomplete Hugging Face ModelOpt NVFP4 scale pair for %s", base)
		}
		return hfScale, hfScale2, nil
	}
	if trt1 || trt2 {
		if !trt1 || !trt2 {
			return "", "", fmt.Errorf("model: incomplete TensorRT-LLM ModelOpt NVFP4 scale pair for %s", base)
		}
		return trtScale, trtScale2, nil
	}
	return "", "", fmt.Errorf("model: missing ModelOpt NVFP4 scale pair for %s", base)
}

func modelOptNVFP4Entry(hdr map[string]json.RawMessage, name string) (stEntry, error) {
	raw, ok := hdr[name]
	if !ok {
		return stEntry{}, fmt.Errorf("model: missing ModelOpt NVFP4 tensor %s", name)
	}
	var entry stEntry
	if err := unmarshalUniqueJSON(raw, &entry); err != nil {
		return stEntry{}, fmt.Errorf("model: malformed ModelOpt NVFP4 tensor %s: %w", name, err)
	}
	return entry, nil
}

func modelOptNVFP4Shapes(weightName, scaleName string, weight, scale stEntry) (rows, cols int, err error) {
	if weight.Dtype != "U8" || len(weight.Shape) != 2 || weight.Shape[0] <= 0 || weight.Shape[1] <= 0 {
		return 0, 0, fmt.Errorf("model: ModelOpt NVFP4 tensor %s must be U8 [O,K/2], got dtype %q shape %v", weightName, weight.Dtype, weight.Shape)
	}
	cols, ok := checkedShapeProduct(weight.Shape[1], 2)
	if !ok || cols%compute.NVFP4BlockSize != 0 {
		return 0, 0, fmt.Errorf("model: ModelOpt NVFP4 tensor %s has invalid packed K dimension %d", weightName, weight.Shape[1])
	}
	rows = weight.Shape[0]
	wantScaleShape := []int{rows, cols / compute.NVFP4BlockSize}
	if scale.Dtype != "F8_E4M3" || len(scale.Shape) != 2 ||
		scale.Shape[0] != wantScaleShape[0] || scale.Shape[1] != wantScaleShape[1] {
		return 0, 0, fmt.Errorf("model: ModelOpt NVFP4 tensor %s must be F8_E4M3 %v, got dtype %q shape %v", scaleName, wantScaleShape, scale.Dtype, scale.Shape)
	}
	return rows, cols, nil
}
