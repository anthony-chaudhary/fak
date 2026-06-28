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
	// IQ4_NL: a non-linear-codebook 4-bit quant — a single f16 scale d shared by
	// a 32-element block, then qkIQ4NL/2 bytes of packed 4-bit codes (two per
	// byte) that index the 16-entry kvaluesIQ4NL codebook = 18 bytes per block.
	qkIQ4NL         = 32
	blockIQ4NLBytes = 2 + qkIQ4NL/2
	// IQ4_XS: the super-block sibling of IQ4_NL over a 256-element block — one
	// f16 super-scale d, a u16 high-bit scale field, qkK/64 low-bit scale bytes,
	// then qkK/2 bytes of packed 4-bit codes = 136 bytes per super-block. Each of
	// the eight 32-element sub-blocks carries a 6-bit scale ls (4 low bits from
	// scales_l, 2 high bits from scales_h) applied as d*(ls-32).
	blockIQ4XSBytes = 2 + 2 + qkK/64 + qkK/2
	// IQ3_XXS: a 256-element super-block = one f16 super-scale d, then 3*qkK/8 = 96
	// bytes split as qkK/4 = 64 grid-index bytes (8 per 32-element sub-block, indexing
	// iq3xxsGrid) followed by qkK/8 = 32 scale/sign bytes (one u32 per sub-block: top
	// 4 bits = scale, low 28 bits = four 7-bit sign selectors). 98 bytes per super-block.
	blockIQ3XXSBytes = 2 + 3*qkK/8
)

// ValueType is the GGUF metadata value type tag (uint8/int32/string/array/... per the
// GGUF spec) that prefixes each metadata value in the header.
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

// TensorType is the GGUF tensor element/quantization encoding (F32, F16, the Q*_0/Q*_1
// and K-quant blocks, the IQ4 non-linear-codebook quants, BF16, MXFP4) that fixes a
// tensor's on-disk block layout.
type TensorType uint32

const (
	TensorF32    TensorType = 0
	TensorF16    TensorType = 1
	TensorQ4_0   TensorType = 2
	TensorQ4_1   TensorType = 3
	TensorQ5_0   TensorType = 6
	TensorQ5_1   TensorType = 7
	TensorQ8_0   TensorType = 8
	TensorQ2_K   TensorType = 10
	TensorQ3_K   TensorType = 11
	TensorQ4_K   TensorType = 12
	TensorQ5_K   TensorType = 13
	TensorQ6_K   TensorType = 14
	TensorIQ3_XXS TensorType = 18
	TensorIQ4_NL  TensorType = 20
	TensorIQ4_XS  TensorType = 23
	TensorBF16    TensorType = 30
	TensorMXFP4   TensorType = 39
)

// Value is one decoded GGUF metadata value: its ValueType tag and the Go value it
// decoded to (a scalar, a string, or a slice for an array).
type Value struct {
	Type  ValueType
	Value any
}

// TensorInfo is one tensor's directory entry from the GGUF header: its name, dims,
// quant type, and offsets (the in-file data offset and the offset within the tensor
// data section).
type TensorInfo struct {
	Name       string
	Dims       []uint64
	Type       TensorType
	Offset     uint64
	FileOffset int64
}

// String renders the TensorType as its GGUF type name (e.g. "F32", "Q4_K", "IQ4_XS",
// "MXFP4"), falling back to "TensorType(n)" for an unrecognized code.
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
	case TensorIQ3_XXS:
		return "IQ3_XXS"
	case TensorIQ4_NL:
		return "IQ4_NL"
	case TensorIQ4_XS:
		return "IQ4_XS"
	case TensorBF16:
		return "BF16"
	case TensorMXFP4:
		return "MXFP4"
	default:
		return fmt.Sprintf("TensorType(%d)", t)
	}
}

// File is a parsed GGUF header: the format version, the metadata key/value map, the
// tensor directory, the data alignment, and the file offset where the tensor data
// section begins.
type File struct {
	Version          uint32
	Metadata         map[string]Value
	Tensors          []TensorInfo
	Alignment        uint64
	TensorDataOffset int64
}

// WeightSource binds a parsed File to the readers that hold its tensor bytes, routing
// each tensor to the shard file that actually contains it (single-file or split
// checkpoint) and owning the open shard files so Close releases them all.
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
