package model

import (
	"errors"
	"fmt"
	"math"
)

// ErrV41DeepInferIndexMXFP4 reports a refused host-format operation. Refusing
// non-finite values, overflow and malformed ownership is Fak policy, not a
// claim about upstream handling of those inputs.
var ErrV41DeepInferIndexMXFP4 = errors.New("model: DeepInfer index MXFP4 format refused")

// V41DeepInferIndexMXFP4Row is one 128-value indexer row, not the main-KV
// format. Payload stores even elements in low nibbles; each scale describes
// 32 elements. Real zero groups have scale 1, unlike absent cache segments.
//
// Adapted from DeepInfer 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9;
// see third_party/deepinfer-v41-oracles/NOTICE. This host boundary is not wired
// into Q/K casts, scoring, GGUF dtypes or devices. It does not establish CUDA
// numeric parity or parity of upstream BF16 paired-FMA scoring with Fak F32.
type V41DeepInferIndexMXFP4Row struct {
	Payload [64]byte
	Scales  [4]byte
}

// EncodeV41DeepInferIndexMXFP4Row rounds exactly 128 immutable input values to
// BF16 before the per-32 producer cast. E2M1 conversion models RNE/saturation;
// the upstream CUDA conversion implementation is external to DeepInfer.
func EncodeV41DeepInferIndexMXFP4Row(values []float32) (V41DeepInferIndexMXFP4Row, error) {
	var row V41DeepInferIndexMXFP4Row
	if len(values) != 128 {
		return row, fmt.Errorf("%w: row has %d values, want 128", ErrV41DeepInferIndexMXFP4, len(values))
	}
	for g := range row.Scales {
		var tile [32]float32
		amax := float32(6 * 0x1p-126)
		for i := range tile {
			v := values[g*32+i]
			if !finite32(v) {
				return V41DeepInferIndexMXFP4Row{}, fmt.Errorf("%w: non-finite value %d", ErrV41DeepInferIndexMXFP4, g*32+i)
			}
			tile[i] = v41RoundBF16(v)
			if !finite32(tile[i]) {
				return V41DeepInferIndexMXFP4Row{}, fmt.Errorf("%w: BF16 overflow at %d", ErrV41DeepInferIndexMXFP4, g*32+i)
			}
			amax = max(amax, float32(math.Abs(float64(tile[i]))))
		}
		// Preserve the upstream FP32 multiply and bitwise ceil, not a
		// float64 division/log2 approximation (including the zero-group floor).
		bits := math.Float32bits(float32(amax * float32(1.0/6.0)))
		exp := int((bits-1)>>23) + 1 - 127
		row.Scales[g] = byte(exp + 127)
		inv := v41RoundBF16(float32(math.Ldexp(1, -exp)))
		for i, v := range tile {
			// The fused producer multiplies BF16 operands, unlike the
			// dedicated store's FP32 reciprocal path. Generated CUDA and
			// extreme/subnormal device behavior have not been witnessed.
			code := v41DeepInferIndexE2M1(v41RoundBF16(float32(v * inv)))
			j := g*32 + i
			row.Payload[j/2] |= code << (4 * (j % 2))
		}
	}
	if _, err := row.Decode(); err != nil {
		return V41DeepInferIndexMXFP4Row{}, err
	}
	return row, nil
}

func v41DeepInferIndexE2M1(v float32) byte {
	var sign byte
	if math.Signbit(float64(v)) {
		sign, v = 8, -v
	}
	for code := byte(0); code < 7; code++ {
		mid := (v4ExpertE2M1Values[code] + v4ExpertE2M1Values[code+1]) / 2
		if v < mid || v == mid && code%2 == 0 {
			return sign | code
		}
		if v == mid {
			return sign | (code + 1)
		}
	}
	return sign | 7
}

// Decode returns an owned row. This host producer-row policy rejects scale 0
// (which also occurs in missing-segment output); it does not claim all upstream
// consumers reject that byte. Scale 255 is non-finite UE8M0. Non-finite decoded
// value is refused rather than being published to a caller.
func (row V41DeepInferIndexMXFP4Row) Decode() ([128]float32, error) {
	var out [128]float32
	for g, sf := range row.Scales {
		if sf == 0 || sf == 255 {
			return out, fmt.Errorf("%w: scale %d at group %d", ErrV41DeepInferIndexMXFP4, sf, g)
		}
	}
	for i := range out {
		code := (row.Payload[i/2] >> (4 * (i % 2))) & 15
		v := float32(math.Ldexp(float64(v4ExpertE2M1Values[code]), int(row.Scales[i/32])-127))
		if !finite32(v) {
			return [128]float32{}, fmt.Errorf("%w: decoded overflow at %d", ErrV41DeepInferIndexMXFP4, i)
		}
		out[i] = v
	}
	return out, nil
}

// V41DeepInferIndexMXFP4Pages owns block-major bytes. A block contains all
// B*64 payload bytes, then B*4 scale bytes, then padding to a 32-byte stride.
// It is deliberately not an array of 68-byte interleaved rows. Private fields
// and copy-out accessors prevent aliases from changing validated storage.
// Positive block sizes describe the host format only, not device admission.
type V41DeepInferIndexMXFP4Pages struct {
	data                    []byte
	rows, blockSize, stride int
}

func v41DeepInferIndexGeometry(rows, blockSize int) (stride, size int, err error) {
	maxInt := int(^uint(0) >> 1)
	if rows < 0 || blockSize <= 0 || blockSize > (maxInt-31)/68 {
		return 0, 0, fmt.Errorf("%w: rows %d block size %d", ErrV41DeepInferIndexMXFP4, rows, blockSize)
	}
	stride = (68*blockSize + 31) / 32 * 32
	blocks := rows / blockSize
	if rows%blockSize != 0 {
		blocks++
	}
	if blocks > maxInt/stride {
		return 0, 0, fmt.Errorf("%w: page length overflow", ErrV41DeepInferIndexMXFP4)
	}
	return stride, blocks * stride, nil
}

// PackV41DeepInferIndexMXFP4Pages validates all real rows before allocating or
// publishing output. Unoccupied final-block slots and alignment bytes remain
// zero; those slots cannot be accessed as quantized real zero rows.
func PackV41DeepInferIndexMXFP4Pages(rows []V41DeepInferIndexMXFP4Row, blockSize int) (*V41DeepInferIndexMXFP4Pages, error) {
	stride, size, err := v41DeepInferIndexGeometry(len(rows), blockSize)
	if err != nil {
		return nil, err
	}
	for i, row := range rows {
		if _, err := row.Decode(); err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
	}
	p := &V41DeepInferIndexMXFP4Pages{data: make([]byte, size), rows: len(rows), blockSize: blockSize, stride: stride}
	for i, row := range rows {
		base, t := i/blockSize*stride, i%blockSize
		copy(p.data[base+64*t:base+64*(t+1)], row.Payload[:])
		copy(p.data[base+64*blockSize+4*t:base+64*blockSize+4*(t+1)], row.Scales[:])
	}
	return p, nil
}

// ReadV41DeepInferIndexMXFP4Pages copies an exact-length external page image
// and validates all declared real rows. Padding and undeclared slots are not
// interpreted as rows. The caller must not mutate input concurrently.
func ReadV41DeepInferIndexMXFP4Pages(data []byte, rows, blockSize int) (*V41DeepInferIndexMXFP4Pages, error) {
	stride, size, err := v41DeepInferIndexGeometry(rows, blockSize)
	if err != nil {
		return nil, err
	}
	if len(data) != size {
		return nil, fmt.Errorf("%w: page length %d, want %d", ErrV41DeepInferIndexMXFP4, len(data), size)
	}
	p := &V41DeepInferIndexMXFP4Pages{data: append([]byte(nil), data...), rows: rows, blockSize: blockSize, stride: stride}
	for i := 0; i < rows; i++ {
		row := p.row(i)
		if _, err := row.Decode(); err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
	}
	return p, nil
}

func (p *V41DeepInferIndexMXFP4Pages) row(i int) V41DeepInferIndexMXFP4Row {
	var row V41DeepInferIndexMXFP4Row
	base, t := i/p.blockSize*p.stride, i%p.blockSize
	copy(row.Payload[:], p.data[base+64*t:base+64*(t+1)])
	copy(row.Scales[:], p.data[base+64*p.blockSize+4*t:base+64*p.blockSize+4*(t+1)])
	return row
}

// Row accesses a physical block and token slot, refusing undeclared tail
// slots as well as invalid geometry. It returns arrays, never backing slices.
func (p *V41DeepInferIndexMXFP4Pages) Row(block, token int) (V41DeepInferIndexMXFP4Row, error) {
	if p == nil || p.blockSize <= 0 || block < 0 || token < 0 || token >= p.blockSize || p.rows == 0 || block > (p.rows-1)/p.blockSize {
		return V41DeepInferIndexMXFP4Row{}, fmt.Errorf("%w: invalid block/token %d/%d", ErrV41DeepInferIndexMXFP4, block, token)
	}
	// Check the remaining real rows before addition, including near MaxInt.
	base := block * p.blockSize
	if token >= p.rows-base {
		return V41DeepInferIndexMXFP4Row{}, fmt.Errorf("%w: absent row", ErrV41DeepInferIndexMXFP4)
	}
	return p.row(base + token), nil
}

// Bytes returns an independent copy of the physical page image.
func (p *V41DeepInferIndexMXFP4Pages) Bytes() []byte {
	if p == nil {
		return nil
	}
	return append([]byte(nil), p.data...)
}
