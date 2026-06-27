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

// metadataArray reads a GGUF metadata array under key into a []T, converting each
// item with conv. It centralizes the shared array scaffolding — key lookup, the
// TypeArray check, the []Value assertion, allocation, and the per-item iteration —
// so each typed reader (StringArray/IntArray/BoolArray/Int32Array) supplies only its
// element conversion. conv returns (value,false) to reject an item, which fails the
// whole read with (nil,false), preserving each reader's original all-or-nothing
// behavior.
func metadataArray[T any](f *File, key string, conv func(Value) (T, bool)) ([]T, bool) {
	v, ok := f.Metadata[key]
	if !ok || v.Type != TypeArray {
		return nil, false
	}
	items, ok := v.Value.([]Value)
	if !ok {
		return nil, false
	}
	out := make([]T, len(items))
	for i, item := range items {
		c, ok := conv(item)
		if !ok {
			return nil, false
		}
		out[i] = c
	}
	return out, true
}

// StringArray reads a GGUF metadata array of strings into []string, returning (nil,false) on a non-array or any non-string item.
func (f *File) StringArray(key string) ([]string, bool) {
	return metadataArray(f, key, func(item Value) (string, bool) {
		if item.Type != TypeString {
			return "", false
		}
		s, ok := item.Value.(string)
		return s, ok
	})
}

// IntArray reads a GGUF metadata array of integers into []int. Gemma4 encodes a
// per-layer head_count_kv as such an array (one entry per decoder layer). Returns
// (nil,false) when the key is absent, not an array, or carries a non-integer item.
func (f *File) IntArray(key string) ([]int, bool) {
	return metadataArray(f, key, func(item Value) (int, bool) {
		u, ok := valueUint64(item)
		if !ok || u > uint64(math.MaxInt) {
			return 0, false
		}
		return int(u), true
	})
}

// BoolArray reads a GGUF metadata array of bools into []bool. Gemma4 encodes its
// per-layer local/global cadence as sliding_window_pattern (true = sliding/local,
// false = full/global). Returns (nil,false) on any non-bool item.
func (f *File) BoolArray(key string) ([]bool, bool) {
	return metadataArray(f, key, func(item Value) (bool, bool) {
		if item.Type != TypeBool {
			return false, false
		}
		b, ok := item.Value.(bool)
		return b, ok
	})
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
