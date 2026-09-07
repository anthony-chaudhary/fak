package bench

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// QuantTraceMaxBytes bounds the offline diagnostic's input and full expansion.
const QuantTraceMaxBytes = 8 << 20

// QuantTraceSample follows a logical weight through the two packed layouts.
type QuantTraceSample struct {
	Index              int     `json:"index"`
	Row                int     `json:"row"`
	Column             int     `json:"column"`
	SourceByte         int     `json:"source_byte"`
	RepackedByte       int     `json:"repacked_byte"`
	NibbleShift        int     `json:"nibble_shift"`
	Code               byte    `json:"code"`
	ScaleMetadataStart int     `json:"scale_metadata_start"`
	Subblock           int     `json:"subblock"`
	ScaleCode          byte    `json:"scale_code"`
	MinCode            byte    `json:"min_code"`
	D                  float32 `json:"d"`
	DMin               float32 `json:"d_min"`
	Value              float32 `json:"value"`
}

// QuantTraceReceipt records software transformations, never device traffic.
type QuantTraceReceipt struct {
	Schema                     string             `json:"schema"`
	Scope                      string             `json:"scope"`
	Rows                       int                `json:"rows"`
	Columns                    int                `json:"columns"`
	Width                      int                `json:"interleave_width"`
	PackedBytes                int                `json:"packed_bytes"`
	UnpackedCodeBytes          int                `json:"unpacked_code_bytes"`
	ExpandedFloatBytes         int                `json:"expanded_float_bytes"`
	OutputBytes                int                `json:"output_bytes"`
	EffectiveBitsPerValue      float64            `json:"effective_bits_per_value"`
	SourceSHA256               string             `json:"source_sha256"`
	RepackedSHA256             string             `json:"repacked_sha256"`
	RestoredSHA256             string             `json:"restored_sha256"`
	ExpandedSHA256             string             `json:"expanded_f32le_sha256"`
	ActivationSHA256           string             `json:"activation_f32le_sha256"`
	OutputSHA256               string             `json:"output_f32le_sha256"`
	ExactRoundTrip             bool               `json:"exact_byte_round_trip"`
	ExactContraction           bool               `json:"exact_contraction_across_layouts"`
	MaxContractionError        float64            `json:"max_contraction_error"`
	OriginalFloatErrorMeasured bool               `json:"original_float_error_measured"`
	Samples                    []QuantTraceSample `json:"samples"`
	// OutputPrefix lists y[0] through y[min(rows,8)-1]. The hash covers all rows.
	OutputPrefix []float32 `json:"output_prefix"`
}

func quantTraceHash(b []byte) string { return fmt.Sprintf("%x", sha256.Sum256(b)) }
func quantTraceFloatHash(v []float32) string {
	h := sha256.New()
	var b [4]byte
	for _, x := range v {
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(x))
		h.Write(b[:])
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}
func quantTraceFinite(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }

// TraceQ4K executes existing repack, dequant and GEMV helpers on actual Q4_K
// bytes. x is the contraction vector; samples are row-major element indexes.
// Full code and float arrays are materialized deliberately in this diagnostic.
func TraceQ4K(raw []byte, rows, columns, width int, x []float32, samples []int) (QuantTraceReceipt, error) {
	var r QuantTraceReceipt
	if rows <= 0 || columns <= 0 || columns%256 != 0 || width < 1 || width > 256 || len(samples) > 32 {
		return r, fmt.Errorf("quanttrace: require positive rows, columns multiple of 256, width 1..256, at most 32 samples")
	}
	nblk := columns / 256
	// Divide before multiplying so malformed shapes never overflow.
	if nblk > QuantTraceMaxBytes/144 || rows > QuantTraceMaxBytes/144/nblk || len(raw) != rows*nblk*144 {
		return r, fmt.Errorf("quanttrace: shape/payload mismatch or input exceeds %d bytes", QuantTraceMaxBytes)
	}
	if len(x) != columns {
		return r, fmt.Errorf("quanttrace: activation length must equal columns")
	}
	for _, v := range x {
		if !quantTraceFinite(v) {
			return r, fmt.Errorf("quanttrace: nonfinite activation")
		}
	}
	for _, i := range samples {
		if i < 0 || i >= rows*columns {
			return r, fmt.Errorf("quanttrace: sample index %d out of range", i)
		}
	}
	packed, err := RepackQ4KInterleaved(raw, rows, columns, width)
	if err != nil {
		return r, err
	}
	restored, err := UnrepackQ4KInterleaved(packed, rows, columns, width)
	if err != nil {
		return r, err
	}
	codes := make([]byte, rows*columns)
	values := make([]float32, len(codes))
	for b := 0; b < len(raw)/144; b++ {
		rec := raw[b*144 : (b+1)*144]
		q4kDequantRecord(values[b*256:(b+1)*256], rec)
		for j := 0; j < 256; j++ {
			codes[b*256+j] = (rec[16+j/64*32+j%32] >> uint(j%64/32*4)) & 15
			if !quantTraceFinite(values[b*256+j]) {
				return r, fmt.Errorf("quanttrace: nonfinite decoded value in block %d", b)
			}
		}
	}
	y, yp := make([]float32, rows), make([]float32, rows)
	q4kGemvRowMajor(raw, rows, columns, x, y)
	q4kGemvInterleaved(packed, rows, columns, width, x, yp)
	r = QuantTraceReceipt{Schema: "fak.quanttrace.v1", Scope: "offline software diagnostic; existing pure-Go bench helpers; no device execution or production dispatch tracing", Rows: rows, Columns: columns, Width: width, PackedBytes: len(raw), UnpackedCodeBytes: len(codes), ExpandedFloatBytes: len(values) * 4, OutputBytes: len(y) * 4, EffectiveBitsPerValue: 4.5, SourceSHA256: quantTraceHash(raw), RepackedSHA256: quantTraceHash(packed), RestoredSHA256: quantTraceHash(restored), ExpandedSHA256: quantTraceFloatHash(values), ActivationSHA256: quantTraceFloatHash(x), OutputSHA256: quantTraceFloatHash(y), ExactRoundTrip: bytes.Equal(raw, restored), ExactContraction: true, Samples: make([]QuantTraceSample, 0, len(samples))}
	for i := range y {
		if !quantTraceFinite(y[i]) || !quantTraceFinite(yp[i]) {
			return QuantTraceReceipt{}, fmt.Errorf("quanttrace: nonfinite contraction output")
		}
		if math.Float32bits(y[i]) != math.Float32bits(yp[i]) {
			r.ExactContraction = false
		}
		r.MaxContractionError = math.Max(r.MaxContractionError, math.Abs(float64(y[i])-float64(yp[i])))
	}
	for _, i := range samples {
		row, col := i/columns, i%columns
		block, j := col/256, col%256
		sourceBase := (row*nblk + block) * 144
		rec := raw[sourceBase : sourceBase+144]
		off, shift := 16+j/64*32+j%32, j%64/32*4
		group, lane := row/width, row%width
		w := min(width, rows-group*width)
		base := group*width*nblk*144 + block*w*144
		sc, mn := model.GetScaleMinK4(j/32, rec[4:16])
		r.Samples = append(r.Samples, QuantTraceSample{Index: i, Row: row, Column: col, SourceByte: sourceBase + off, RepackedByte: base + 16*w + (off-16)*w + lane, NibbleShift: shift, Code: codes[i], ScaleMetadataStart: sourceBase + 4, Subblock: j / 32, ScaleCode: sc, MinCode: mn, D: math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(rec))), DMin: math.Float32frombits(model.F16BitsToF32Bits(binary.LittleEndian.Uint16(rec[2:]))), Value: values[i]})
	}
	r.OutputPrefix = append([]float32(nil), y[:min(rows, 8)]...)
	return r, nil
}

// QuantTraceDemo provides two deterministic packed rows, not original floats.
func QuantTraceDemo() []byte {
	raw := make([]byte, 288)
	for b := 0; b < 2; b++ {
		rec := raw[b*144 : (b+1)*144]
		binary.LittleEndian.PutUint16(rec, 0x3c00)
		binary.LittleEndian.PutUint16(rec[2:], 0x3800)
		for i := 4; i < 16; i++ {
			rec[i] = 1
		}
		for i := 16; i < 144; i++ {
			rec[i] = byte(i + b)
		}
	}
	return raw
}
