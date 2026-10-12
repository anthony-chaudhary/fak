package compute

import (
	"fmt"
	"math"
)

// V41EngramRowEncoding names the stored dialect of a V4.1 Engram table row.
type V41EngramRowEncoding int

const (
	// V41EngramRowPacked is the 264-byte row: 256 E4M3FN codes then 8 E8M0
	// scales, one per 32 elements.
	V41EngramRowPacked V41EngramRowEncoding = iota
	// V41EngramRowQ2K is one 84-byte Q2_K super-block (GGUF byte order).
	V41EngramRowQ2K
	// V41EngramRowF32 is 256 little-endian binary32 values.
	V41EngramRowF32
)

// V41EngramRowDecodeMaxDim is the widest row prefix one decode returns: every
// dialect stores 256 elements per row.
const V41EngramRowDecodeMaxDim = 256

// V41EngramRowStride returns the stored width of one row, or 0 for an unknown
// encoding.
func V41EngramRowStride(encoding V41EngramRowEncoding) int {
	switch encoding {
	case V41EngramRowPacked:
		return 264
	case V41EngramRowQ2K:
		return 84
	case V41EngramRowF32:
		return 1024
	}
	return 0
}

// V41EngramRowDecodeBackend decodes gathered V4.1 Engram table rows on the
// device (Ref #13668). The caller hands over the rows' stored bytes; no host
// arithmetic touches them. For row r and element j < dim:
//
//	packed: BF16(E4M3(code[j]) * 2^(scale[j/32]-127)), BF16 the host v41BF16 RNE
//	q2_k:   the host scalar Q2_K super-block dequantization
//	f32:    the stored binary32 word
//
// panel is a fresh F32 [rows, dim] tensor. status is a fresh [rows] tensor of
// uint32 words (F32 dtype, read as bits): 0 for a clean row, otherwise
// 1 + 2*j + kind for the lowest faulting element j, where kind 0 is a NaN code
// or scale byte (packed only) and kind 1 a non-finite decoded value. That is
// exactly the first refusal the host decode raises.
//
// V41EngramRowDecodePoison writes a quiet NaN into target[index] when any of
// the first rows status words is nonzero, so a consumer that refuses on a
// non-finite input observes a decode fault without its own readback.
//
// Absence permits host selection; after dispatch is selected an error never
// permits host replay.
type V41EngramRowDecodeBackend interface {
	SupportsV41EngramRowDecode() bool
	V41EngramRowDecode(encoded []byte, encoding V41EngramRowEncoding, rows, dim int) (panel, status Tensor, err error)
	V41EngramRowDecodePoison(target, status Tensor, rows, index int) error
}

// V41EngramRowFault splits a nonzero status word into the faulting element and
// whether the fault is a NaN byte (packed) rather than a non-finite value.
func V41EngramRowFault(word uint32) (element int, nanByte, ok bool) {
	if word == 0 || word > 2*V41EngramRowDecodeMaxDim {
		return 0, false, false
	}
	key := word - 1
	return int(key >> 1), key&1 == 0, true
}

func validateV41EngramRowDecode(encoded []byte, encoding V41EngramRowEncoding, rows, dim int) error {
	stride := V41EngramRowStride(encoding)
	if stride == 0 {
		return fmt.Errorf("compute: V4.1 Engram row decode has unknown encoding %d", encoding)
	}
	if rows <= 0 || dim <= 0 || dim > V41EngramRowDecodeMaxDim {
		return fmt.Errorf("compute: V4.1 Engram row decode rows=%d dim=%d outside [1,%d]", rows, dim, V41EngramRowDecodeMaxDim)
	}
	if uint64(rows)*uint64(stride) > math.MaxInt32 || uint64(rows)*uint64(dim) > math.MaxInt32 {
		return fmt.Errorf("compute: V4.1 Engram row decode rows=%d exceeds the shader index range", rows)
	}
	if len(encoded) != rows*stride {
		return fmt.Errorf("compute: V4.1 Engram row decode has %d encoded bytes, want %d rows x %d", len(encoded), rows, stride)
	}
	return nil
}

func validateV41EngramRowPoison(target, status Tensor, rows, index int) error {
	if rows <= 0 || uint64(rows) > math.MaxInt32 {
		return fmt.Errorf("compute: V4.1 Engram row poison rows=%d", rows)
	}
	if target.Dtype != F32 || target.Quant != nil || index < 0 || index >= target.Numel() {
		return fmt.Errorf("compute: V4.1 Engram row poison target must be F32 with index %d in range", index)
	}
	if status.Dtype != F32 || status.Quant != nil || status.Numel() < rows {
		return fmt.Errorf("compute: V4.1 Engram row poison status must hold %d words", rows)
	}
	return nil
}

func vulkanV41EngramRowDecodeStatusError(status int, reason string) *BackendError {
	err := vulkanV41IndexerScoreStatusError(status, reason)
	err.Site = "V41EngramRowDecode"
	err.Message = fmt.Sprintf("dedicated Engram row decode failed closed (code %d): %s", status, reason)
	return err
}
