package model

// awq.go — AWQ (Activation-aware Weight Quantization) 4-bit weight path.
// AWQ achieves near-float performance at 4-bit with per-channel scaling and
// activation-aware calibration. This implements the AWQ format compatible with
// HuggingFace AutoAWQ and llama.cpp's AWQ loader.
//
// Format specification:
// - 4-bit weights packed 2 per byte (nibble-packed)
// - Per-channel scales (one float32 per output channel)
// - Symmetric quantization (zero_point = 8 for unsigned 4-bit)
// - Weights stored as [out, in/2] bytes (in must be even)
//
// Dequantization: weight = scale[o] * (code - 8)
// where code is the unpacked 4-bit value (0-15) and 8 is the zero-point for
// symmetric 4-bit quantization.
//
// Correctness discipline: AWQ is lossy 4-bit quantization, so the gate is
// greedy-continuation agreement with HuggingFace transformers AWQ reference
// and cosine similarity ≥ 0.995 vs FP32 baseline.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// awqTensor is a resident AWQ 4-bit weight matrix [out, in], where in is the

// awqTensor is a resident AWQ 4-bit weight matrix [out, in], where in is the
// original input dimension (must be even). raw holds the packed 4-bit weights
// as [out, in/2] bytes. scales[o] is the per-channel scale for output channel o.
// Dequantization: weight[o*in + i] = scales[o] * (unpack4bit(raw[o*in/2 + i/2]) - 8)
type awqTensor struct {
	out, in int
	raw     []byte    // [out, in/2] packed 4-bit weights
	scales  []float32 // [out] per-channel scales
}

// awqRowBytes is the byte length of one AWQ row (in/2 bytes for packed 4-bit).
func (qt *awqTensor) awqRowBytes() int { return qt.in / 2 }

// unpack4bit extracts two 4-bit values from one byte. The low nibble is the first
// weight, high nibble is the second weight. Returns (lo, hi) where lo, hi ∈ [0, 15].
func unpack4bit(b byte) (lo, hi uint8) {
	return b & 0x0f, b >> 4
}

// awqDequantRow writes the dequantized float32 weights of one output channel
// into dst (len >= in). For each input weight i:
//
//	dst[i] = scales[o] * (code - 8)
//
// where 8 is the symmetric zero-point for unsigned 4-bit quantization.
func awqDequantRow(dst []float32, scales []float32, o int, raw []byte, in int) {
	scale := scales[o]
	const zeroPoint = 8 // symmetric 4-bit zero-point
	for i := 0; i < in/2; i++ {
		lo, hi := unpack4bit(raw[i])
		dst[i*2] = scale * float32(int16(lo)-zeroPoint)
		dst[i*2+1] = scale * float32(int16(hi)-zeroPoint)
	}
}

// awqMatRows is the AWQ 4-bit GEMV: y[o] = dot(weight row o, x). Row-parallel
// like other matmul kernels. Each row is dequantized on-the-fly and dotted
// against the input vector x.
func awqMatRows(qt *awqTensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	awqMatRowsInto(qt, x, y)
	return y
}

// awqMatRowsInto computes y = Awx where A is the AWQ 4-bit weight matrix.
func awqMatRowsInto(qt *awqTensor, x, y []float32) {
	y = y[:qt.out]
	parForRange(qt.out, qt.out*qt.in, func(lo, hi int) { awxMatRowsRange(qt, x, y, lo, hi) })
}

// awxMatRowsRange computes y[lo:hi] by dequantizing each row and dotting with x.
func awxMatRowsRange(qt *awqTensor, x, y []float32, lo, hi int) {
	// Temporary buffer for dequantized row (reused across output rows)
	buf := make([]float32, qt.in)
	rowBytes := qt.awqRowBytes()

	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		awqDequantRow(buf, qt.scales, o, row, qt.in)

		// Dot product: y[o] = sum(buf[i] * x[i])
		var acc float32
		for i := 0; i < qt.in; i++ {
			acc += buf[i] * x[i]
		}
		y[o] = acc
	}
}

// awqGemm is the AWQ 4-bit PREFILL GEMM: Y[t*out+o] = dot(weight row o, X[t*in:(t+1)*in])
// for all t in [0,P). Each weight row is dequantized once and reused across all P
// activation rows, amortizing the dequantization cost.
func awqGemm(qt *awqTensor, X []float32, P int) []float32 {
	Y := make([]float32, P*qt.out)
	awqGemmInto(qt, X, P, Y)
	return Y
}

// awqGemmInto is awqGemm writing into a caller-provided Y buffer.
func awqGemmInto(qt *awqTensor, X []float32, P int, Y []float32) {
	Y = Y[:P*qt.out]
	parForRange(qt.out, qt.out*qt.in*P, func(lo, hi int) { awxGemmRange(qt, X, P, Y, lo, hi) })
}

// awxGemmRange computes Y[t*out+o] for o in [lo,hi), all t in [0,P).
func awxGemmRange(qt *awqTensor, X []float32, P int, Y []float32, lo, hi int) {
	buf := make([]float32, qt.in) // dequantized row buffer
	rowBytes := qt.awqRowBytes()
	acc := make([]float32, P) // per-token accumulator

	for o := lo; o < hi; o++ {
		row := qt.raw[o*rowBytes:]
		awqDequantRow(buf, qt.scales, o, row, qt.in)

		for t := 0; t < P; t++ {
			acc[t] = 0
		}

		// For each input position, accumulate dot product across all tokens
		for i := 0; i < qt.in; i++ {
			w := buf[i]
			for t := 0; t < P; t++ {
				acc[t] += w * X[t*qt.in+i]
			}
		}

		for t := 0; t < P; t++ {
			Y[t*qt.out+o] = acc[t]
		}
	}
}

// quantizeAWQFromRaw wraps a raw AWQ 4-bit payload into a resident awqTensor.
// raw must be the packed 4-bit weights [out, in/2]. scales is the per-channel
// scale vector [out]. in must be even.
func quantizeAWQFromRaw(raw []byte, scales []float32, out, in int) *awqTensor {
	if in%2 != 0 {
		panic("model: AWQ input dimension must be even")
	}
	rowBytes := in / 2
	want := out * rowBytes
	if len(raw) != want {
		panic("model: AWQ payload size mismatch")
	}
	if len(scales) != out {
		panic("model: AWQ scales length mismatch")
	}
	return &awqTensor{out: out, in: in, raw: raw, scales: scales}
}

// awq returns the resident AWQ tensor for a name.
func (m *Model) awq(name string) *awqTensor {
	if m.awqw == nil {
		panic("model: no AWQ tensors loaded (call Model.LoadAWQ)")
	}
	qt, ok := m.awqw[name]
	if !ok {
		panic("model: AWQ tensor not found: " + name)
	}
	return qt
}

// hasAWQ reports whether an AWQ copy exists for a name.
func (m *Model) hasAWQ(name string) bool {
	if m.awqw == nil {
		return false
	}
	_, ok := m.awqw[name]
	return ok
}

// AWQCount returns how many tensors hold an AWQ copy.
func (m *Model) AWQCount() int {
	return len(m.awqw)
}

// AWQShape returns the (out, in) shape of an AWQ tensor, or (0,0) if absent.
func (m *Model) AWQShape(name string) (out, in int) {
	if qt := m.awqw[name]; qt != nil {
		return qt.out, qt.in
	}
	return 0, 0
}

// ---- AWQ Loader ------------------------------------------------------------

// LoadAWQ loads an AWQ-quantized model from a directory containing AWQ safetensors
// files. The directory should contain:
// - config.json: the model config
// - model.safetensors or pytorch_model.bin: AWQ quantized weights
//
// AWQ tensors are identified by their _scale suffix. For each weight tensor
// "name.weight" with a corresponding "name.weight_scale" tensor, we create an
// AWQ 4-bit tensor. The packed weights are expected as unsigned 4-bit values
// stored in the weight tensor (2 weights per byte).
func LoadAWQ(dir string) (*Model, error) {
	// Load config
	cfgPath := filepath.Join(dir, "config.json")
	var cfg Config
	if err := readJSON(cfgPath, &cfg); err != nil {
		return nil, fmt.Errorf("awq load config: %w", err)
	}

	// Try safetensors first, then pytorch bin
	stPath := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(stPath); err == nil {
		return loadAWQSafetensors(stPath, cfg)
	}

	// Try pytorch_model.bin (common in AutoAWQ exports)
	ptPath := filepath.Join(dir, "pytorch_model.bin")
	if _, err := os.Stat(ptPath); err == nil {
		return loadAWQPytorchBin(ptPath, cfg)
	}

	return nil, fmt.Errorf("awq load: no model.safetensors or pytorch_model.bin found in %s", dir)
}

// loadAWQSafetensors loads AWQ weights from a safetensors file. A genuine AutoAWQ
// export (qweight/qzeros/scales triples — the format real Llama-2/3 & Qwen2 AWQ
// checkpoints ship) is routed to the group-wise asymmetric loader; the legacy
// "<name>_scale" symmetric stub is handled below for back-compat.
func loadAWQSafetensors(path string, cfg Config) (*Model, error) {
	sf, err := openSafetensorsFile(path)
	if err != nil {
		return nil, err
	}
	defer sf.Close()

	for name := range sf.hdr {
		if strings.HasSuffix(name, ".qweight") {
			return loadAWQGroupSafetensors(sf, cfg)
		}
	}

	awqw := make(map[string]*awqTensor)

	// First pass: collect all AWQ tensor names (those with _scale suffix)
	scaleNames := make([]string, 0)
	for name := range sf.hdr {
		if name == "__metadata__" {
			continue
		}
		if strings.HasSuffix(name, "_scale") {
			scaleNames = append(scaleNames, name)
		}
	}

	// Second pass: load each AWQ tensor
	for _, scaleName := range scaleNames {
		weightName := strings.TrimSuffix(scaleName, "_scale")
		if weightName == scaleName {
			continue // Skip if _scale was the entire name
		}

		// Parse scale tensor entry
		var scaleEntry stEntry
		if err := json.Unmarshal(sf.hdr[scaleName], &scaleEntry); err != nil {
			return nil, fmt.Errorf("awq load scale entry %s: %w", scaleName, err)
		}

		// Parse weight tensor entry
		var weightEntry stEntry
		if err := json.Unmarshal(sf.hdr[weightName], &weightEntry); err != nil {
			// Weight tensor might not exist (e.g. for embeddings)
			continue
		}

		// Load scale data
		scaleData, err := sf.tensorBytes(scaleEntry)
		if err != nil {
			return nil, fmt.Errorf("awq load scale data %s: %w", scaleName, err)
		}

		// Load weight data
		weightData, err := sf.tensorBytes(weightEntry)
		if err != nil {
			return nil, fmt.Errorf("awq load weight data %s: %w", weightName, err)
		}

		// Parse scales (f32 LE)
		scales := make([]float32, len(scaleData)/4)
		for i := 0; i < len(scales); i++ {
			scales[i] = math.Float32frombits(binary.LittleEndian.Uint32(scaleData[i*4 : i*4+4]))
		}

		// Shape: AWQ weight tensors are [out, in/2] (packed 4-bit)
		if len(weightEntry.Shape) != 2 {
			continue // Skip non-2D tensors
		}

		out := weightEntry.Shape[0]
		inPacked := weightEntry.Shape[1] // in/2
		in := inPacked * 2

		// Validate dimensions
		if len(scales) != out {
			return nil, fmt.Errorf("awq tensor %s: scales len %d != out dim %d", weightName, len(scales), out)
		}

		if len(weightData) != out*inPacked {
			return nil, fmt.Errorf("awq tensor %s: weight data len %d != out*inPacked %d",
				weightName, len(weightData), out*inPacked)
		}

		// Copy weight data (ensure we own the bytes)
		weightCopy := make([]byte, len(weightData))
		copy(weightCopy, weightData)

		awqw[weightName] = quantizeAWQFromRaw(weightCopy, scales, out, in)
	}

	if len(awqw) == 0 {
		return nil, fmt.Errorf("awq load: no AWQ tensors found in %s", path)
	}

	return &Model{Cfg: cfg, awqw: awqw}, nil
}

// loadAWQGroupSafetensors loads a genuine AutoAWQ export: for every "<name>.qweight"
// it pairs the sibling "<name>.qzeros" and "<name>.scales" and builds a resident
// group-wise asymmetric tensor (awq_group.go). On disk AutoAWQ stores:
//   - qweight  int32 [in,         out/8]   8 codes per int32, awqPackOrder reorder
//   - qzeros   int32 [in/group,   out/8]   per-group 4-bit zero-points, same packing
//   - scales   f16   [in/group,   out]     per-group float scales
//
// We transpose into the output-major resident layout and dequant as
// (code - zero) * scale. Byte-equality vs a stored HF fixture is left to an oracle
// fixture (absent here); the on-host witness is the pack/unpack round-trip plus the
// FP32 cos>=0.995 oracle.
func loadAWQGroupSafetensors(sf *safetensorsFile, cfg Config) (*Model, error) {
	awqg := make(map[string]*awqGroupTensor)

	for name := range sf.hdr {
		if name == "__metadata__" || !strings.HasSuffix(name, ".qweight") {
			continue
		}
		base := strings.TrimSuffix(name, ".qweight")

		var wEntry, zEntry, sEntry stEntry
		if err := json.Unmarshal(sf.hdr[name], &wEntry); err != nil {
			return nil, fmt.Errorf("awq group: qweight entry %s: %w", name, err)
		}
		zRaw, hasZ := sf.hdr[base+".qzeros"]
		sRaw, hasS := sf.hdr[base+".scales"]
		if !hasZ || !hasS {
			return nil, fmt.Errorf("awq group: %s missing .qzeros or .scales sibling", base)
		}
		if err := json.Unmarshal(zRaw, &zEntry); err != nil {
			return nil, fmt.Errorf("awq group: qzeros entry %s: %w", base, err)
		}
		if err := json.Unmarshal(sRaw, &sEntry); err != nil {
			return nil, fmt.Errorf("awq group: scales entry %s: %w", base, err)
		}
		if len(wEntry.Shape) != 2 || len(zEntry.Shape) != 2 || len(sEntry.Shape) != 2 {
			return nil, fmt.Errorf("awq group: %s tensors must be 2-D", base)
		}

		in := wEntry.Shape[0]
		out := wEntry.Shape[1] * 8
		nGroups := sEntry.Shape[0]
		if nGroups <= 0 || in%nGroups != 0 {
			return nil, fmt.Errorf("awq group: %s in=%d not divisible by nGroups=%d", base, in, nGroups)
		}
		groupSize := in / nGroups
		if sEntry.Shape[1] != out {
			return nil, fmt.Errorf("awq group: %s scales out=%d != qweight out=%d", base, sEntry.Shape[1], out)
		}
		if in%2 != 0 || out%8 != 0 {
			return nil, fmt.Errorf("awq group: %s needs even in (%d) and out%%8==0 (%d)", base, in, out)
		}

		qwBytes, err := sf.tensorBytes(wEntry)
		if err != nil {
			return nil, fmt.Errorf("awq group: read qweight %s: %w", base, err)
		}
		qzBytes, err := sf.tensorBytes(zEntry)
		if err != nil {
			return nil, fmt.Errorf("awq group: read qzeros %s: %w", base, err)
		}
		scBytes, err := sf.tensorBytes(sEntry)
		if err != nil {
			return nil, fmt.Errorf("awq group: read scales %s: %w", base, err)
		}
		scF32Bytes, err := decodeSafetensorF32(base+".scales", sEntry, scBytes)
		if err != nil {
			return nil, fmt.Errorf("awq group: decode scales %s: %w", base, err)
		}

		qw := bytesToU32LE(qwBytes)
		qz := bytesToU32LE(qzBytes)
		colsW := out / 8
		if len(qw) != in*colsW {
			return nil, fmt.Errorf("awq group: %s qweight len %d != in*out/8 %d", base, len(qw), in*colsW)
		}
		if len(qz) != nGroups*colsW {
			return nil, fmt.Errorf("awq group: %s qzeros len %d != nGroups*out/8 %d", base, len(qz), nGroups*colsW)
		}
		scF32 := make([]float32, len(scF32Bytes)/4)
		for i := range scF32 {
			scF32[i] = math.Float32frombits(binary.LittleEndian.Uint32(scF32Bytes[i*4 : i*4+4]))
		}
		if len(scF32) != nGroups*out {
			return nil, fmt.Errorf("awq group: %s scales len %d != nGroups*out %d", base, len(scF32), nGroups*out)
		}

		qt := &awqGroupTensor{
			out:       out,
			in:        in,
			groupSize: groupSize,
			nGroups:   nGroups,
			codes:     make([]byte, out*(in/2)),
			scales:    make([]float32, out*nGroups),
			zeros:     make([]uint8, out*nGroups),
		}
		rowBytes := in / 2

		// Transpose qweight [in][out/8] -> output-major nibble-packed codes [out][in/2].
		rowCodes := make([]uint8, out)
		for i := 0; i < in; i++ {
			awqUnpackI32Row(rowCodes, qw[i*colsW:(i+1)*colsW], out)
			bi := i >> 1
			if i&1 == 0 {
				for o := 0; o < out; o++ {
					p := o*rowBytes + bi
					qt.codes[p] = (qt.codes[p] &^ 0x0f) | byte(rowCodes[o]&0x0f)
				}
			} else {
				for o := 0; o < out; o++ {
					p := o*rowBytes + bi
					qt.codes[p] = (qt.codes[p] & 0x0f) | byte((rowCodes[o]&0x0f)<<4)
				}
			}
		}

		// Transpose qzeros [nGroups][out/8] -> zeros[o*nGroups+g]; scales likewise.
		zRow := make([]uint8, out)
		for g := 0; g < nGroups; g++ {
			awqUnpackI32Row(zRow, qz[g*colsW:(g+1)*colsW], out)
			for o := 0; o < out; o++ {
				qt.zeros[o*nGroups+g] = zRow[o]
				qt.scales[o*nGroups+g] = scF32[g*out+o]
			}
		}

		awqg[base] = qt
	}

	if len(awqg) == 0 {
		return nil, fmt.Errorf("awq group: no qweight/qzeros/scales triples found")
	}
	return &Model{Cfg: cfg, awqg: awqg}, nil
}

// bytesToU32LE reinterprets a little-endian int32/uint32 byte buffer as []uint32.
func bytesToU32LE(b []byte) []uint32 {
	n := len(b) / 4
	out := make([]uint32, n)
	for i := 0; i < n; i++ {
		out[i] = binary.LittleEndian.Uint32(b[i*4 : i*4+4])
	}
	return out
}

// loadAWQPytorchBin loads AWQ weights from a pytorch_model.bin file.
// This is a simplified loader for AutoAWK exports that use pytorch bin format.
func loadAWQPytorchBin(path string, cfg Config) (*Model, error) {
	// Pytorch bin format loader would go here
	// For now, redirect to safetensors with a helpful error
	return nil, fmt.Errorf("awq load: pytorch_model.bin format not yet supported, please use safetensors export")
}
