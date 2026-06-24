package ggufload

import (
	"fmt"
	"math"
)

// String reads a scalar GGUF string metadata value, returning ("",false) when the key is absent or not a TypeString.
func (f *File) String(key string) (string, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeString {
		return "", false
	}
	s, ok := v.Value.(string)
	return s, ok
}

// Uint64 reads any scalar GGUF integer metadata value as a uint64, returning (0,false) when the key is absent or negative.
func (f *File) Uint64(key string) (uint64, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	return valueUint64(v)
}

// Float64 reads a scalar GGUF float metadata value (TypeFloat32 widened, or TypeFloat64), returning (0,false) otherwise.
func (f *File) Float64(key string) (float64, bool) {
	v, ok := f.Metadata[key]
	if !ok {
		return 0, false
	}
	switch v.Type {
	case TypeFloat32:
		return float64(v.Value.(float32)), true
	case TypeFloat64:
		return v.Value.(float64), true
	default:
		return 0, false
	}
}

// Bool reads a scalar GGUF boolean metadata value (TypeBool, one byte). GLM-5.2
// encodes expert_weights_norm (the MoE top-k renormalization flag, HF's
// norm_topk_prob) this way. Returns (false,false) when the key is absent or not
// a scalar bool.
func (f *File) Bool(key string) (bool, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeBool {
		return false, false
	}
	b, ok := v.Value.(bool)
	return b, ok
}

// StringArray reads a GGUF metadata array of strings into []string, returning (nil,false) on a non-array or any non-string item.
func (f *File) StringArray(key string) ([]string, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]string, len(items))
	for i, item := range items {
		if item.Type != TypeString {
			return nil, false
		}
		out[i] = item.Value.(string)
	}
	return out, true
}

// IntArray reads a GGUF metadata array of integers into []int. Gemma4 encodes a
// per-layer head_count_kv as such an array (one entry per decoder layer). Returns
// (nil,false) when the key is absent, not an array, or carries a non-integer item.
func (f *File) IntArray(key string) ([]int, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]int, len(items))
	for i, item := range items {
		u, ok := valueUint64(item)
		if !ok || u > uint64(math.MaxInt) {
			return nil, false
		}
		out[i] = int(u)
	}
	return out, true
}

// BoolArray reads a GGUF metadata array of bools into []bool. Gemma4 encodes its
// per-layer local/global cadence as sliding_window_pattern (true = sliding/local,
// false = full/global). Returns (nil,false) on any non-bool item.
func (f *File) BoolArray(key string) ([]bool, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]bool, len(items))
	for i, item := range items {
		if item.Type != TypeBool {
			return nil, false
		}
		b, ok := item.Value.(bool)
		if !ok {
			return nil, false
		}
		out[i] = b
	}
	return out, true
}

func (f *File) requiredInt(key string) (int, error) {
	v, ok := f.Uint64(key)
	if !ok {
		return 0, fmt.Errorf("gguf: missing %s", key)
	}
	if v > uint64(math.MaxInt) {
		return 0, fmt.Errorf("gguf: %s overflows int", key)
	}
	return int(v), nil
}

func (f *File) requiredFloat(key string) (float64, error) {
	v, ok := f.Float64(key)
	if !ok {
		return 0, fmt.Errorf("gguf: missing %s", key)
	}
	return v, nil
}

func (f *File) hasTensor(name string) bool {
	for _, t := range f.Tensors {
		if t.Name == name {
			return true
		}
	}
	return false
}

func valueUint64(v Value) (uint64, bool) {
	switch v.Type {
	case TypeUint8:
		return uint64(v.Value.(uint8)), true
	case TypeUint16:
		return uint64(v.Value.(uint16)), true
	case TypeUint32:
		return uint64(v.Value.(uint32)), true
	case TypeUint64:
		return v.Value.(uint64), true
	case TypeInt8:
		x := v.Value.(int8)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case TypeInt16:
		x := v.Value.(int16)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case TypeInt32:
		x := v.Value.(int32)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	case TypeInt64:
		x := v.Value.(int64)
		if x < 0 {
			return 0, false
		}
		return uint64(x), true
	default:
		return 0, false
	}
}
