// Package ggufload parses GGUF metadata and tensor directories for off-path model loading.
package ggufload

import (
	"fmt"
	"io"
)

const (
	Magic          = "GGUF"
	Version        = 3
	defaultAlign   = 32
	maxStringBytes = 1 << 30
	qk4            = 32
	qk5            = 32
	qk8_0          = 32
	qkMXFP4        = 32
	qkK            = 256
	kScaleSize     = 12
	blockQ4_0Bytes = 2 + qk4/2
	blockQ4_1Bytes = 2 + 2 + qk4/2
	blockQ5_0Bytes = 2 + 4 + qk5/2
	blockQ5_1Bytes = 2 + 2 + 4 + qk5/2
	blockQ8_0Bytes = 2 + qk8_0
	blockQ2KBytes  = qkK/16 + qkK/4 + 2 + 2
	blockQ3KBytes  = qkK/8 + qkK/4 + kScaleSize + 2
	blockQ4KBytes  = 2 + 2 + kScaleSize + qkK/2
	blockQ5KBytes  = 2 + 2 + kScaleSize + qkK/8 + qkK/2
	blockQ6KBytes  = qkK/2 + qkK/4 + qkK/16 + 2
	// MXFP4 (gpt-oss): a 1-byte E8M0 shared scale + qkMXFP4/2 bytes of packed
	// 4-bit E2M1 codes (two per byte) = 17 bytes per 32-element block.
	blockMXFP4Bytes = 1 + qkMXFP4/2
)

type ValueType uint32

const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

type TensorType uint32

const (
	TensorF32   TensorType = 0
	TensorF16   TensorType = 1
	TensorQ4_0  TensorType = 2
	TensorQ4_1  TensorType = 3
	TensorQ5_0  TensorType = 6
	TensorQ5_1  TensorType = 7
	TensorQ8_0  TensorType = 8
	TensorQ2_K  TensorType = 10
	TensorQ3_K  TensorType = 11
	TensorQ4_K  TensorType = 12
	TensorQ5_K  TensorType = 13
	TensorQ6_K  TensorType = 14
	TensorBF16  TensorType = 30
	TensorMXFP4 TensorType = 39
)

type Value struct {
	Type  ValueType
	Value any
}

type TensorInfo struct {
	Name       string
	Dims       []uint64
	Type       TensorType
	Offset     uint64
	FileOffset int64
}

func (t TensorType) String() string {
	switch t {
	case TensorF32:
		return "F32"
	case TensorF16:
		return "F16"
	case TensorQ4_0:
		return "Q4_0"
	case TensorQ4_1:
		return "Q4_1"
	case TensorQ5_0:
		return "Q5_0"
	case TensorQ5_1:
		return "Q5_1"
	case TensorQ8_0:
		return "Q8_0"
	case TensorQ2_K:
		return "Q2_K"
	case TensorQ3_K:
		return "Q3_K"
	case TensorQ4_K:
		return "Q4_K"
	case TensorQ5_K:
		return "Q5_K"
	case TensorQ6_K:
		return "Q6_K"
	case TensorBF16:
		return "BF16"
	case TensorMXFP4:
		return "MXFP4"
	default:
		return fmt.Sprintf("TensorType(%d)", t)
	}
}

type File struct {
	Version          uint32
	Metadata         map[string]Value
	Tensors          []TensorInfo
	Alignment        uint64
	TensorDataOffset int64
}

type WeightSource struct {
	File *File
	r    io.ReaderAt
	// readerFor (parallel to File.Tensors) routes tensor i's bytes to the shard
	// file that actually holds them. A nil entry falls back to r, which preserves
	// the original single-file behaviour. sizeFor[i] is readerFor[i]'s file size,
	// used for the overrun bounds check.
	readerFor []io.ReaderAt
	sizeFor   []int64
	// closers holds every open shard file; Close shuts them all. For a single-file
	// checkpoint this is exactly one entry.
	closers []io.Closer
	size    int64
	byName  map[string]int
}
