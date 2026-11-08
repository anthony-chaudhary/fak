package compute

import (
	"errors"
	"fmt"
	"math"
	"math/bits"
)

// Packed Engram gather, adapted from SGLang PR #47935 ("DSV4.1 Engram
// packed-gather"), merged 2026-11-04 at 3f5b1bec8e00115bc8439b5b9a9da902822d2f82:
// https://github.com/sgl-project/sglang/pull/47935
//
// Upstream keeps the DeepSeek-V4.1 conditional-memory lookup tables as MXFP4
// micro-blocks instead of expanded FP8 rows: two E2M1 values per payload byte
// (even column in the low nibble) with unchanged per-32 E8M0 scales, and the
// gather kernel decodes each element in-register so no widened copy of the
// table is ever materialized. This component ports that packed-storage and
// in-gather-decode contract into idiomatic Go for in-memory use. No upstream
// code is copied (ADAPT verdict); see THIRD_PARTY_NOTICES.md.
//
// Exactness: an E2M1 magnitude has at most two significant bits and an E8M0
// scale is a pure power of two, so every decoded product lands exactly on the
// bf16 grid (or flushes/saturates deterministically there). The decode runs in
// integer arithmetic and is bit-identical to gathering over the expanded e4m3
// rows, including signed zero. Software evidence only: this component makes no
// hardware or whole-model performance claim.
//
// Deviations from the upstream kernel, both deliberate and pinned by tests:
//   - The gather output carries the upstream bf16-rounded value as float32,
//     so Go callers need no bf16 ABI.
//   - Scale byte 0xFF (the reserved E8M0 pattern) is refused at construction
//     with ErrInvalidScale instead of being bitcast to +Inf.

// engramMXBlock is the number of elements sharing one E8M0 scale (OCP MX block
// size), matching the upstream BLK constant.
const engramMXBlock = 32

// Sentinel errors returned by the packed Engram gather contract. Callers match
// them with errors.Is; the wrapped details carry the offending index.
var (
	// ErrDimNotMultipleOf32 reports a hidden dimension that is not a positive
	// multiple of the MX block size.
	ErrDimNotMultipleOf32 = errors.New("compute: engram dim must be a positive multiple of 32")

	// ErrShapeMismatch reports a payload or scale slice whose length does not
	// cover the declared rows and dimension.
	ErrShapeMismatch = errors.New("compute: engram packed table shape mismatch")

	// ErrInvalidScale reports the reserved E8M0 scale byte 0xFF.
	ErrInvalidScale = errors.New("compute: engram E8M0 scale byte 0xFF is reserved")

	// ErrIDOutOfRange reports a gather id outside the table's row range.
	ErrIDOutOfRange = errors.New("compute: engram gather id out of range")

	// ErrOutputLength reports a GatherInto destination that does not hold
	// len(ids) rows.
	ErrOutputLength = errors.New("compute: engram gather output length mismatch")

	// ErrE4M3ReservedByte reports an expanded e4m3fn byte in the reserved NaN
	// patterns (0x7F, 0xFF).
	ErrE4M3ReservedByte = errors.New("compute: engram expanded row holds reserved e4m3fn byte")
)

// PackedEngramTable stores conditional-memory lookup rows as MXFP4
// micro-blocks: dim/2 payload bytes of E2M1 nibbles per row (even column in
// the low nibble) plus dim/32 E8M0 scale bytes, one per 32-element block. It
// is the packed counterpart of an expanded e4m3 [rows, dim] table; holding the
// payload packed halves the per-row storage bytes (dim/2 vs dim), which is the
// lookup-table memory traffic the borrow removes.
type PackedEngramTable struct {
	dim    int
	rows   int
	data   []byte // rows x dim/2, owned
	scales []byte // rows x dim/32, owned
}

// NewPackedEngramTable validates and takes ownership of one packed table. data
// holds rows*dim/2 payload bytes and scales holds rows*dim/32 E8M0 bytes; both
// are copied. The receiver is immutable afterwards.
func NewPackedEngramTable(dim, rows int, data, scales []byte) (*PackedEngramTable, error) {
	if dim <= 0 || dim%engramMXBlock != 0 {
		return nil, fmt.Errorf("dim %d: %w", dim, ErrDimNotMultipleOf32)
	}
	rowBytes := dim / 2
	scaleBytes := dim / engramMXBlock
	if rows < 0 || len(data) != rows*rowBytes || len(scales) != rows*scaleBytes {
		return nil, fmt.Errorf("data %d bytes, scales %d bytes for dim %d rows %d: %w",
			len(data), len(scales), dim, rows, ErrShapeMismatch)
	}
	for i, b := range scales {
		if b == 0xff {
			return nil, fmt.Errorf("row %d block %d: %w", i/scaleBytes, i%scaleBytes, ErrInvalidScale)
		}
	}
	t := &PackedEngramTable{
		dim:    dim,
		rows:   rows,
		data:   make([]byte, len(data)),
		scales: make([]byte, len(scales)),
	}
	copy(t.data, data)
	copy(t.scales, scales)
	return t, nil
}

// Dim returns the hidden dimension (columns per row).
func (t *PackedEngramTable) Dim() int { return t.dim }

// Rows returns the number of stored rows.
func (t *PackedEngramTable) Rows() int { return t.rows }

// PackedBytesPerRow is the packed payload footprint: dim/2 E2M1-nibble bytes.
func (t *PackedEngramTable) PackedBytesPerRow() int { return t.dim / 2 }

// ExpandedBytesPerRow is the widened e4m3 footprint the packed layout avoids:
// dim bytes.
func (t *PackedEngramTable) ExpandedBytesPerRow() int { return t.dim }

// Gather decodes the selected rows in-register and returns their values,
// bf16-rounded like the upstream kernel and carried as float32. It never
// materializes an expanded copy of the table.
func (t *PackedEngramTable) Gather(ids []int32) ([]float32, error) {
	out := make([]float32, len(ids)*t.dim)
	if err := t.GatherInto(out, ids); err != nil {
		return nil, err
	}
	return out, nil
}

// GatherInto decodes the selected rows into out, which must hold
// len(ids)*Dim() floats. It performs no heap allocation.
func (t *PackedEngramTable) GatherInto(out []float32, ids []int32) error {
	if len(out) != len(ids)*t.dim {
		return fmt.Errorf("out %d floats for %d ids x dim %d: %w", len(out), len(ids), t.dim, ErrOutputLength)
	}
	rowBytes := t.dim / 2
	scaleBytes := t.dim / engramMXBlock
	o := 0
	for _, id := range ids {
		if id < 0 || id >= int32(t.rows) {
			return fmt.Errorf("id %d of %d rows: %w", id, t.rows, ErrIDOutOfRange)
		}
		base := int(id)
		row := t.data[base*rowBytes : base*rowBytes+rowBytes]
		sc := t.scales[base*scaleBytes : base*scaleBytes+scaleBytes]
		for blk := 0; blk < scaleBytes; blk++ {
			k := int(sc[blk]) - 127
			b0 := blk * 16
			for j := 0; j < 16; j++ {
				b := row[b0+j]
				out[o] = engramElement(b&0x0f, k) // even column: low nibble
				out[o+1] = engramElement(b>>4, k) // odd column: high nibble
				o += 2
			}
		}
	}
	return nil
}

// ExpandToE4M3 materializes the full e4m3 [rows, dim] copy of the packed
// table, mirroring the upstream unpack_mxfp4_rows bit-exactly. It exists for
// the oracle and diagnostics: production gather paths read the packed rows
// directly instead.
func (t *PackedEngramTable) ExpandToE4M3() []byte {
	out := make([]byte, t.rows*t.dim)
	rowBytes := t.dim / 2
	scaleBytes := t.dim / engramMXBlock
	for r := 0; r < t.rows; r++ {
		row := t.data[r*rowBytes : (r+1)*rowBytes]
		dst := out[r*t.dim : (r+1)*t.dim]
		for blk := 0; blk < scaleBytes; blk++ {
			for j := 0; j < 16; j++ {
				b := row[blk*16+j]
				dst[blk*32+2*j] = engramNibbleToE4M3(b & 0x0f)
				dst[blk*32+2*j+1] = engramNibbleToE4M3(b >> 4)
			}
		}
	}
	return out
}

// GatherE4M3Rows reads selected rows from an expanded e4m3 [rows, dim] table
// with per-32 E8M0 scales and returns their values, bf16-rounded, as float32.
// It is the widened-table reference path of the packed-gather oracle: it
// assumes the full copy is already materialized, which is precisely the memory
// traffic the packed path avoids. Reserved e4m3fn NaN patterns are refused.
func GatherE4M3Rows(expanded, scales []byte, dim int, ids []int32) ([]float32, error) {
	if dim <= 0 || dim%engramMXBlock != 0 {
		return nil, fmt.Errorf("dim %d: %w", dim, ErrDimNotMultipleOf32)
	}
	scaleBytes := dim / engramMXBlock
	if len(expanded)%dim != 0 || len(scales)%scaleBytes != 0 {
		return nil, fmt.Errorf("expanded %d bytes, scales %d bytes for dim %d: %w",
			len(expanded), len(scales), dim, ErrShapeMismatch)
	}
	rows := len(expanded) / dim
	out := make([]float32, len(ids)*dim)
	for i, id := range ids {
		if id < 0 || id >= int32(rows) {
			return nil, fmt.Errorf("id %d of %d rows: %w", id, rows, ErrIDOutOfRange)
		}
		base := int(id)
		src := expanded[base*dim : (base+1)*dim]
		sc := scales[base*scaleBytes : (base+1)*scaleBytes]
		dst := out[i*dim : (i+1)*dim]
		for blk := 0; blk < scaleBytes; blk++ {
			k := int(sc[blk]) - 127
			for j := 0; j < engramMXBlock; j++ {
				b := src[blk*engramMXBlock+j]
				sign, sInt, r, ok := e4m3Parts(b)
				if !ok {
					return nil, fmt.Errorf("row %d col %d: %w", id, blk*engramMXBlock+j, ErrE4M3ReservedByte)
				}
				dst[blk*engramMXBlock+j] = engramBF16(sign, sInt, r+k)
			}
		}
	}
	return out, nil
}

// engramElement decodes one E2M1 nibble, applies the E8M0 shift k, and returns
// the bf16-rounded value as float32. It is the in-register decode of the
// packed gather: no intermediate full-precision copy of the row exists. The
// low three bits are the magnitude code and bit 3 is the sign, mirroring the
// upstream nib & 0x7 / (nib >> 3) & 1 split.
func engramElement(code byte, k int) float32 {
	sInt, r := e2m1Parts(code & 0x07)
	return engramBF16(code&0x08 != 0, sInt, r+k)
}

// e2m1Parts splits an E2M1 magnitude code into the exact dyadic factors
// sInt*2^r. Magnitude codes follow the upstream mapping: 0 -> 0, 1 -> 0.5,
// then 2..7 -> 1, 1.5, 2, 3, 4, 6.
func e2m1Parts(m byte) (sInt uint32, r int) {
	switch {
	case m == 0:
		return 0, 0
	case m == 1:
		return 1, -1
	case m&1 == 0:
		return 1, int(m)/2 - 1 // 2 -> 1*2^0, 4 -> 1*2^1, 6 -> 1*2^2
	default:
		return 3, int(m-3)/2 - 1 // 3 -> 3*2^-1, 5 -> 3*2^0, 7 -> 3*2^1
	}
}

// engramNibbleToE4M3 renders one E2M1 nibble as its exact e4m3fn byte, using
// the upstream closed form: magnitudes 0, 0.5, 1, ..., 6 are 0x00, 0x30, then
// 4*m + 48, with the sign bit set directly so -0 survives.
func engramNibbleToE4M3(code byte) byte {
	m := code & 0x07
	var bits byte
	if m >= 2 {
		bits = 4*m + 48
	} else {
		bits = 48 * m
	}
	return bits | ((code & 0x08) << 4)
}

// e4m3Parts splits a finite e4m3fn byte into sign and the exact dyadic factors
// sInt*2^r. e4m3fn is 1 sign + 4 exponent + 3 mantissa bits, bias 7. ok is
// false for the reserved NaN patterns 0x7F and 0xFF.
func e4m3Parts(b byte) (sign bool, sInt uint32, r int, ok bool) {
	ef := int(b>>3) & 0x0f
	m := int(b & 0x07)
	if ef == 0 {
		return b&0x80 != 0, uint32(m), -9, true // subnormal m*2^-9
	}
	if ef == 15 && m == 7 {
		return false, 0, 0, false
	}
	return b&0x80 != 0, uint32(8 + m), ef - 10, true // (8+m)*2^(ef-10)
}

// engramBF16 returns the bf16 round-to-nearest-even value of +-sInt*2^r
// (sInt < 256) carried as float32. Every input is an exact dyadic rational, so
// the result is computed without a floating-point multiply: in the normal
// range the value always lands exactly on the 8-bit bf16 grid, and only the
// subnormal underflow and overflow boundaries need RNE decisions.
func engramBF16(sign bool, sInt uint32, r int) float32 {
	if sInt == 0 {
		return engramSignedZero(sign)
	}
	fl := uint(31 - bits.LeadingZeros32(sInt)) // floor(log2 sInt)
	e := r + int(fl)                           // floor(log2 |v|)
	te := e + 127                              // target bf16 exponent field
	switch {
	case te > 254:
		if sign {
			return float32(math.Inf(-1))
		}
		return float32(math.Inf(1))
	case te <= 0:
		shift := r + 133 // subnormal mantissa = sInt * 2^shift, step 2^-133
		var m uint32
		if shift >= 0 {
			m = sInt << uint(shift)
			if m > 127 {
				m = 127 // beyond the largest subnormal; in-domain unreachable
			}
		} else {
			d := uint(-shift)
			if d >= 8 {
				m = 0 // |v| < half a subnormal step: flush
			} else {
				q := sInt >> d
				rem := sInt & (1<<d - 1)
				half := uint32(1) << (d - 1)
				if rem > half || (rem == half && q&1 == 1) {
					q++
				}
				m = q
			}
		}
		if m == 0 {
			return engramSignedZero(sign)
		}
		// Exact: m <= 127 fits a float64 mantissa and the result is bf16-exact.
		return engramApplySign(float32(math.Ldexp(float64(m), -133)), sign)
	default:
		pos := sInt<<uint(7-fl) - 128 // exact grid position in [0, 128)
		v := float32(math.Ldexp(float64(128+pos), e-7))
		return engramApplySign(v, sign)
	}
}

// engramApplySign sets the sign bit directly in the float32 bits, the upstream
// trick that keeps -0 distinct from +0.
func engramApplySign(v float32, sign bool) float32 {
	bits := math.Float32bits(v)
	if sign {
		bits |= 0x80000000
	}
	return math.Float32frombits(bits)
}

// engramSignedZero returns +0 or -0 as float32.
func engramSignedZero(sign bool) float32 {
	if sign {
		return math.Float32frombits(0x80000000)
	}
	return 0
}
