package model

// exl2.go — EXL2 (ExLlamaV2) mixed-precision weight format [issue #299, A-003].
//
// EXL2 is ExLlamaV2's quantization format (a safetensors container, NOT a
// llama.cpp/GGUF format — the migrated tracker mislabels it). Its defining
// feature over a fixed-bit format like AWQ (awq_group.go) is MIXED PRECISION:
// the input axis of each linear is partitioned into groups, and each group is
// quantized at its OWN bit width (2/3/4/5/6/8 bpw) chosen to hit a target
// average bits-per-weight while protecting salient channels. On disk a real
// ExLlamaV2 export stores, per linear "<name>":
//
//	<name>.q_weight     int32   bit-stream of the per-group variable-width codes
//	<name>.q_scale      f16     per-(output, group) scale multiplier
//	<name>.q_scale_max  f16     per-group scale maximum (the two-level scale)
//	<name>.q_groups     int16   [2*G]   (bits, size) pair per input-axis group
//	<name>.q_invperm    int32   [in]    inverse act-order permutation over inputs
//	<name>.bias         f16     optional
//
// This file implements that storage+dequant contract end to end, so EXL2
// weights stream off a memory-mapped checkpoint (openSafetensorsFile, the same
// zero-copy mmap path AWQ/safetensors use — "mmap-based weight streaming") and
// dequantize into the row-parallel GEMV/GEMM the rest of the kernel consumes
// ("integration with existing compute backends"):
//
//	w[o, invperm[i]] = scale[o,g] * (code[o,i] - 2^(bits_g - 1)),   g = group(i)
//	scale[o,g]       = q_scale[o,g] * q_scale_max[g]
//
//   - code[o,i] is the unsigned bits_g-bit code for output channel o, storage
//     input column i, packed LSB-first into the q_weight uint32 stream
//   - 2^(bits_g - 1) is the per-group symmetric center (mid-tread zero-point)
//   - invperm maps a storage column back to its logical input channel, undoing
//     the act-order reordering EXL2 applies so high-salience channels group well
//
// HONESTY BOUNDARY (mirrors the AWQ-group "absent oracle fixture" discipline,
// awq_group.go): what is witnessed ON-HOST (exl2_test.go) is that (1) the
// variable-width bit packer round-trips bijectively across word boundaries for
// every bpw, (2) the per-group bit widths and offsets recover from q_groups,
// (3) the act-order inverse permutation round-trips, and (4) load+dequant+GEMV
// agrees with an FP32 baseline to cosine >= 0.995 with matching argmax. What is
// NOT yet validated, and needs a real ExLlamaV2 checkpoint + the upstream source
// (host-gated, no fixture or GPU offline): the exact GPU-tile shuffle ExLlamaV2
// applies inside q_weight, and its native 4-bit packing of q_scale (read here as
// f16). The three live-model acceptance bars of #299 — load a real EXL2 model,
// memory <= llama.cpp EXL2, decode parity within 10% — depend on those and a GPU
// node, and remain explicitly open. The dequant MECHANISM is what ships here.

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// exl2Group describes one input-axis group of an EXL2 tensor: its quantization
// bit width and where it begins in the (storage-order) input dimension.
type exl2Group struct {
	bits int // 2,3,4,5,6,8 — the per-group bpw, the mixed-precision knob
	off  int // first storage input column of this group
	size int // number of input columns in this group
}

// exl2Tensor is a resident EXL2 mixed-precision weight matrix [out, in]. Codes
// are packed LSB-first per output row into rowWords uint32 words; a group may use
// a different bit width than its neighbours, so the packing is a bit-stream, not
// a fixed nibble layout. scales are [out*nGroups] per (output channel, group).
// invPerm (len in) maps a storage column to its logical input channel; nil means
// the identity (no act-order reorder).
type exl2Tensor struct {
	out, in  int
	groups   []exl2Group
	colGroup []int32 // [in] storage column -> group index (precomputed dispatch)
	invPerm  []int32 // [in] storage column -> logical input channel, or nil for identity
	rowWords int     // uint32 words per output row in codes
	codes    []uint32
	scales   []float32 // [out*len(groups)]
}

// nGroups is the number of input-axis groups (the mixed-precision partition).
func (qt *exl2Tensor) nGroups() int { return len(qt.groups) }

// exl2RowBits returns the total number of code bits in one output row (sum over
// groups of size*bits), and the uint32 word count that holds them.
func exl2RowBits(groups []exl2Group) (bits, words int) {
	for _, g := range groups {
		bits += g.size * g.bits
	}
	words = (bits + 31) / 32
	return bits, words
}

// exl2WriteBits writes the low `bits` bits of code into the per-row word slice at
// bit position bitpos, LSB-first, spanning a word boundary when needed. bits<=8
// (the max EXL2 bpw) so a code straddles at most two words.
func exl2WriteBits(words []uint32, bitpos, bits, code int) {
	w := bitpos >> 5
	off := bitpos & 31
	words[w] |= uint32(code) << off
	if off+bits > 32 {
		words[w+1] |= uint32(code) >> (32 - off)
	}
}

// exl2ReadBits reads `bits` bits LSB-first at bitpos, the inverse of
// exl2WriteBits. The off>0 guard on the high-word read keeps the shift in range
// (off+bits>32 with bits<=8 implies off>=25, so 32-off is in [1,7]).
func exl2ReadBits(words []uint32, bitpos, bits int) int {
	w := bitpos >> 5
	off := bitpos & 31
	mask := uint32((1 << uint(bits)) - 1)
	v := words[w] >> off
	if off+bits > 32 {
		v |= words[w+1] << (32 - off)
	}
	return int(v & mask)
}

// exl2Center is the symmetric mid-tread zero-point for a bits-wide code: a code
// of 2^(bits-1) dequantizes to 0.
func exl2Center(bits int) int { return 1 << uint(bits-1) }

// exl2DequantRow writes the dequantized float32 weights of output channel o into
// dst (len >= in), scattering each storage column through invPerm into its
// logical input position. dst[invperm[i]] = scale[o,g] * (code - center).
func exl2DequantRow(dst []float32, qt *exl2Tensor, o int) {
	row := qt.codes[o*qt.rowWords : (o+1)*qt.rowWords]
	sc := qt.scales[o*qt.nGroups() : (o+1)*qt.nGroups()]
	bitpos := 0
	for i := 0; i < qt.in; i++ {
		g := int(qt.colGroup[i])
		bits := qt.groups[g].bits
		code := exl2ReadBits(row, bitpos, bits)
		bitpos += bits
		v := sc[g] * float32(code-exl2Center(bits))
		if qt.invPerm != nil {
			dst[qt.invPerm[i]] = v
		} else {
			dst[i] = v
		}
	}
}

// exl2MatRows is the EXL2 GEMV: y[o] = dot(weight row o, x). Each row is
// dequantized on the fly and dotted against x, row-parallel like the AWQ/Q4_K
// matmul kernels.
func exl2MatRows(qt *exl2Tensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	exl2MatRowsInto(qt, x, y)
	return y
}

// exl2MatRowsInto computes y = W·x into a caller-provided buffer.
func exl2MatRowsInto(qt *exl2Tensor, x, y []float32) {
	y = y[:qt.out]
	parForRange(qt.out, qt.out*qt.in, func(lo, hi int) { exl2MatRange(qt, x, y, lo, hi) })
}

// exl2MatRange computes y[lo:hi] by dequantizing each row and dotting with x.
func exl2MatRange(qt *exl2Tensor, x, y []float32, lo, hi int) {
	buf := make([]float32, qt.in)
	for o := lo; o < hi; o++ {
		exl2DequantRow(buf, qt, o)
		var acc float32
		for i := 0; i < qt.in; i++ {
			acc += buf[i] * x[i]
		}
		y[o] = acc
	}
}

// exl2Gemm is the EXL2 PREFILL GEMM: Y[t*out+o] = dot(row o, X[t*in:(t+1)*in])
// for all t in [0,P). Each weight row is dequantized once and reused across all
// P activation rows, amortizing the dequant cost.
func exl2Gemm(qt *exl2Tensor, X []float32, P int) []float32 {
	Y := make([]float32, P*qt.out)
	exl2GemmInto(qt, X, P, Y)
	return Y
}

// exl2GemmInto is exl2Gemm writing into a caller-provided Y buffer.
func exl2GemmInto(qt *exl2Tensor, X []float32, P int, Y []float32) {
	Y = Y[:P*qt.out]
	parForRange(qt.out, qt.out*qt.in*P, func(lo, hi int) { exl2GemmRange(qt, X, P, Y, lo, hi) })
}

// exl2GemmRange computes Y[t*out+o] for o in [lo,hi), all t in [0,P).
func exl2GemmRange(qt *exl2Tensor, X []float32, P int, Y []float32, lo, hi int) {
	buf := make([]float32, qt.in)
	for o := lo; o < hi; o++ {
		exl2DequantRow(buf, qt, o)
		for t := 0; t < P; t++ {
			xs := X[t*qt.in : t*qt.in+qt.in]
			var acc float32
			for i := 0; i < qt.in; i++ {
				acc += buf[i] * xs[i]
			}
			Y[t*qt.out+o] = acc
		}
	}
}

// exl2BuildColGroup expands the group partition into a per-column group index and
// validates that the groups tile the input axis contiguously [0,in) with no gap
// or overlap. It is the single place a malformed q_groups is rejected.
func exl2BuildColGroup(groups []exl2Group, in int) ([]int32, error) {
	colGroup := make([]int32, in)
	next := 0
	for gi, g := range groups {
		if g.off != next {
			return nil, fmt.Errorf("exl2: group %d offset %d, expected %d (non-contiguous)", gi, g.off, next)
		}
		if g.size <= 0 {
			return nil, fmt.Errorf("exl2: group %d has non-positive size %d", gi, g.size)
		}
		switch g.bits {
		case 2, 3, 4, 5, 6, 8:
		default:
			return nil, fmt.Errorf("exl2: group %d has unsupported bit width %d", gi, g.bits)
		}
		if next+g.size > in {
			return nil, fmt.Errorf("exl2: group %d overruns input axis (%d+%d > %d)", gi, next, g.size, in)
		}
		for c := g.off; c < g.off+g.size; c++ {
			colGroup[c] = int32(gi)
		}
		next += g.size
	}
	if next != in {
		return nil, fmt.Errorf("exl2: groups cover %d columns, want in=%d", next, in)
	}
	return colGroup, nil
}

// newEXL2Tensor assembles a resident tensor from its parts, validating the group
// tiling, the scale count, the row-word count, and the permutation.
func newEXL2Tensor(out, in int, groups []exl2Group, invPerm []int32, codes []uint32, scales []float32) (*exl2Tensor, error) {
	colGroup, err := exl2BuildColGroup(groups, in)
	if err != nil {
		return nil, err
	}
	_, rowWords := exl2RowBits(groups)
	if len(codes) != out*rowWords {
		return nil, fmt.Errorf("exl2: codes len %d != out*rowWords %d", len(codes), out*rowWords)
	}
	if len(scales) != out*len(groups) {
		return nil, fmt.Errorf("exl2: scales len %d != out*nGroups %d", len(scales), out*len(groups))
	}
	if invPerm != nil {
		if len(invPerm) != in {
			return nil, fmt.Errorf("exl2: invPerm len %d != in %d", len(invPerm), in)
		}
		seen := make([]bool, in)
		for _, p := range invPerm {
			if p < 0 || int(p) >= in || seen[p] {
				return nil, fmt.Errorf("exl2: invPerm is not a permutation of [0,%d)", in)
			}
			seen[p] = true
		}
	}
	return &exl2Tensor{
		out: out, in: in, groups: groups, colGroup: colGroup,
		invPerm: invPerm, rowWords: rowWords, codes: codes, scales: scales,
	}, nil
}

// exl2Quantize packs a row-major FP32 weight matrix w [out, in] into the EXL2
// mixed-precision format under groupBits[g]/groupSizes[g] (a contiguous partition
// of the input axis) and the optional act-order permutation perm (storage column
// -> logical channel, i.e. invPerm). Each group gets a per-(out,group) symmetric
// scale; codes are round-to-nearest within the group. This is the in-Go EXL2
// quantizer the round-trip witness packs with; a real ExLlamaV2 export produces
// an equivalent layout offline. perm may be nil for the identity.
func exl2Quantize(w []float32, out, in int, groupBits, groupSizes []int, perm []int32) (*exl2Tensor, error) {
	if len(groupBits) != len(groupSizes) {
		return nil, fmt.Errorf("exl2: groupBits len %d != groupSizes len %d", len(groupBits), len(groupSizes))
	}
	groups := make([]exl2Group, len(groupBits))
	off := 0
	for g := range groupBits {
		groups[g] = exl2Group{bits: groupBits[g], off: off, size: groupSizes[g]}
		off += groupSizes[g]
	}
	colGroup, err := exl2BuildColGroup(groups, in)
	if err != nil {
		return nil, err
	}
	if len(w) != out*in {
		return nil, fmt.Errorf("exl2: weight len %d != out*in %d", len(w), out*in)
	}
	// fwd[logical channel] = storage column, the inverse of perm, so we can read
	// the logical weight value for each storage column during packing.
	fwd := make([]int, in)
	if perm != nil {
		if len(perm) != in {
			return nil, fmt.Errorf("exl2: perm len %d != in %d", len(perm), in)
		}
		for storageCol, logical := range perm {
			fwd[logical] = storageCol
		}
	} else {
		for i := range fwd {
			fwd[i] = i
		}
	}
	// storageCol -> logical channel (== perm, or identity).
	storageToLogical := make([]int, in)
	for logical, storageCol := range fwd {
		storageToLogical[storageCol] = logical
	}

	_, rowWords := exl2RowBits(groups)
	codes := make([]uint32, out*rowWords)
	scales := make([]float32, out*len(groups))

	for o := 0; o < out; o++ {
		// Per-group symmetric scale from the group's max magnitude.
		for gi, g := range groups {
			maxAbs := float32(0)
			for c := g.off; c < g.off+g.size; c++ {
				v := w[o*in+storageToLogical[c]]
				if a := float32(math.Abs(float64(v))); a > maxAbs {
					maxAbs = a
				}
			}
			denom := float32(exl2Center(g.bits) - 1)
			if denom < 1 {
				denom = 1
			}
			scale := maxAbs / denom
			if scale <= 0 {
				scale = 1e-8
			}
			scales[o*len(groups)+gi] = scale
		}
		// Pack the row's codes LSB-first.
		rowWordsSlice := codes[o*rowWords : (o+1)*rowWords]
		bitpos := 0
		for c := 0; c < in; c++ {
			gi := int(colGroup[c])
			g := groups[gi]
			scale := scales[o*len(groups)+gi]
			center := exl2Center(g.bits)
			v := w[o*in+storageToLogical[c]]
			code := int(math.Round(float64(v/scale))) + center
			if code < 0 {
				code = 0
			}
			if max := (1 << uint(g.bits)) - 1; code > max {
				code = max
			}
			exl2WriteBits(rowWordsSlice, bitpos, g.bits, code)
			bitpos += g.bits
		}
	}
	return newEXL2Tensor(out, in, groups, perm, codes, scales)
}

// ---- EXL2Model: the resident EXL2 checkpoint --------------------------------

// EXL2Model is a loaded EXL2 checkpoint: the parsed config plus the per-linear
// resident mixed-precision tensors. It is intentionally a standalone type rather
// than the shared *Model — the EXL2 dequant feeds the same row-parallel GEMV/GEMM
// (exl2MatRows/exl2Gemm) the rest of the kernel uses, so "integration with the
// compute backends" needs no coupling to Model's f32/Q8/Q4/AWQ stores. A host that
// wants EXL2 tensors on a *Model can copy Tensors() into a Model.exl2w field.
type EXL2Model struct {
	Cfg Config
	w   map[string]*exl2Tensor
}

// Count returns how many linears hold an EXL2 tensor.
func (m *EXL2Model) Count() int { return len(m.w) }

// Has reports whether an EXL2 tensor exists for a name.
func (m *EXL2Model) Has(name string) bool {
	_, ok := m.w[name]
	return ok
}

// Shape returns the (out, in, nGroups) of an EXL2 tensor, or zeros if absent.
func (m *EXL2Model) Shape(name string) (out, in, nGroups int) {
	if qt := m.w[name]; qt != nil {
		return qt.out, qt.in, qt.nGroups()
	}
	return 0, 0, 0
}

// tensor returns the resident EXL2 tensor for a name (panics if absent — the
// caller is expected to gate on Has first, matching the awq accessor contract).
func (m *EXL2Model) tensor(name string) *exl2Tensor {
	qt, ok := m.w[name]
	if !ok {
		panic("model: EXL2 tensor not found: " + name)
	}
	return qt
}

// MatRows is the EXL2 GEMV for a named linear: y = W·x, dequantizing on the fly
// through the same row-parallel kernel the f32/AWQ paths use. This is the compute
// integration seam an inference loop calls per matmul.
func (m *EXL2Model) MatRows(name string, x []float32) []float32 {
	return exl2MatRows(m.tensor(name), x)
}

// Gemm is the EXL2 prefill GEMM for a named linear over P activation rows.
func (m *EXL2Model) Gemm(name string, X []float32, P int) []float32 {
	return exl2Gemm(m.tensor(name), X, P)
}

// ---- EXL2 loader (safetensors container + metadata parse) -------------------

// IsEXL2Dir reports whether dir looks like an ExLlamaV2 (EXL2) export: a
// model.safetensors whose header carries at least one ".q_weight" tensor (the
// EXL2 quantized-linear marker, distinct from AutoAWQ's ".qweight").
func IsEXL2Dir(dir string) bool {
	sf, err := openSafetensorsFile(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return false
	}
	defer sf.Close()
	for name := range sf.hdr {
		if strings.HasSuffix(name, ".q_weight") {
			return true
		}
	}
	return false
}

// LoadEXL2 loads an ExLlamaV2 (EXL2) export from a directory containing a
// config.json and a model.safetensors. Every "<name>.q_weight" is paired with its
// sibling q_groups/q_invperm/q_scale/q_scale_max tensors and built into a resident
// exl2Tensor. Weights stream off the memory-mapped checkpoint (openSafetensorsFile)
// — the whole file is never resident in the process heap.
//
// See the file header for the HONESTY BOUNDARY: this reads the EXL2 container and
// metadata and the q_weight bit-stream under fak's resident layout; matching a real
// ExLlamaV2 q_weight byte-for-byte (its GPU-tile shuffle + native 4-bit q_scale
// packing) is host-gated on a real checkpoint and remains open.
func LoadEXL2(dir string) (*EXL2Model, error) {
	cfgPath := filepath.Join(dir, "config.json")
	var cfg Config
	if err := readJSON(cfgPath, &cfg); err != nil {
		return nil, fmt.Errorf("exl2 load config: %w", err)
	}
	stPath := filepath.Join(dir, "model.safetensors")
	if _, err := os.Stat(stPath); err != nil {
		return nil, fmt.Errorf("exl2 load: no model.safetensors in %s", dir)
	}
	sf, err := openSafetensorsFile(stPath)
	if err != nil {
		return nil, err
	}
	defer sf.Close()
	return loadEXL2Safetensors(sf, cfg)
}

func loadEXL2Safetensors(sf *safetensorsFile, cfg Config) (*EXL2Model, error) {
	exl2w := make(map[string]*exl2Tensor)
	for name := range sf.hdr {
		if name == "__metadata__" || !strings.HasSuffix(name, ".q_weight") {
			continue
		}
		base := strings.TrimSuffix(name, ".q_weight")
		qt, err := loadEXL2Tensor(sf, base)
		if err != nil {
			return nil, fmt.Errorf("exl2 tensor %s: %w", base, err)
		}
		exl2w[base] = qt
	}
	if len(exl2w) == 0 {
		return nil, fmt.Errorf("exl2 load: no .q_weight tensors found")
	}
	return &EXL2Model{Cfg: cfg, w: exl2w}, nil
}

// loadEXL2Tensor reads one linear's EXL2 tensor set and builds the resident form.
// q_groups is int16 [2*G] of (bits, size) pairs; q_invperm is int32 [in]; q_scale
// is f16 [out*G]; q_scale_max is f16 [G]; q_weight is the int32 code bit-stream.
func loadEXL2Tensor(sf *safetensorsFile, base string) (*exl2Tensor, error) {
	groups, in, err := exl2ReadGroups(sf, base+".q_groups")
	if err != nil {
		return nil, err
	}
	invPerm, err := exl2ReadInvPerm(sf, base+".q_invperm", in)
	if err != nil {
		return nil, err
	}
	scaleMax, err := exl2ReadF16Vec(sf, base+".q_scale_max")
	if err != nil {
		return nil, err
	}
	if len(scaleMax) != len(groups) {
		return nil, fmt.Errorf("q_scale_max len %d != nGroups %d", len(scaleMax), len(groups))
	}
	scaleMul, err := exl2ReadF16Vec(sf, base+".q_scale")
	if err != nil {
		return nil, err
	}
	if len(scaleMul)%len(groups) != 0 {
		return nil, fmt.Errorf("q_scale len %d not a multiple of nGroups %d", len(scaleMul), len(groups))
	}
	out := len(scaleMul) / len(groups)
	if out <= 0 {
		return nil, fmt.Errorf("q_scale implies out=%d", out)
	}
	// Effective per-(out,group) scale = q_scale[o,g] * q_scale_max[g] (two-level).
	scales := make([]float32, out*len(groups))
	for o := 0; o < out; o++ {
		for g := range groups {
			scales[o*len(groups)+g] = scaleMul[o*len(groups)+g] * scaleMax[g]
		}
	}
	_, rowWords := exl2RowBits(groups)
	codes, err := exl2ReadCodes(sf, base+".q_weight", out, rowWords)
	if err != nil {
		return nil, err
	}
	return newEXL2Tensor(out, in, groups, invPerm, codes, scales)
}

// exl2ReadGroups parses q_groups into the group partition and returns the total
// input width. The resident contract is int16 [2*G] of (bits, size) pairs:
// q_groups[2g] is the group's bit width, q_groups[2g+1] is its input-column count;
// offsets are the running cumulative sum, so the groups tile [0,in) contiguously.
// (A real ExLlamaV2 q_groups stores (bits, q_weight-row-offset) against a fixed
// config group_size — the transform onto this contiguous (bits,size) form needs a
// real checkpoint to align, part of the host-gated boundary in the file header.)
func exl2ReadGroups(sf *safetensorsFile, name string) ([]exl2Group, int, error) {
	raw, e, err := exl2TensorRaw(sf, name)
	if err != nil {
		return nil, 0, err
	}
	if e.Dtype != "I16" && e.Dtype != "U16" {
		return nil, 0, fmt.Errorf("%s dtype %q, want I16/U16", name, e.Dtype)
	}
	if len(raw)%2 != 0 {
		return nil, 0, fmt.Errorf("%s odd byte length %d", name, len(raw))
	}
	n := len(raw) / 2
	if n == 0 || n%2 != 0 {
		return nil, 0, fmt.Errorf("%s holds %d int16, want 2*G", name, n)
	}
	g := n / 2
	groups := make([]exl2Group, g)
	off := 0
	for i := 0; i < g; i++ {
		bits := int(int16(binary.LittleEndian.Uint16(raw[(2*i)*2:])))
		size := int(int16(binary.LittleEndian.Uint16(raw[(2*i+1)*2:])))
		if size <= 0 {
			return nil, 0, fmt.Errorf("%s group %d non-positive size %d", name, i, size)
		}
		groups[i] = exl2Group{bits: bits, off: off, size: size}
		off += size
	}
	return groups, off, nil
}

// exl2ReadU32Words reads an I32/U32 tensor's raw bytes into little-endian uint32
// words: it fetches the bytes, checks the dtype is I32/U32 and the byte length is a
// multiple of 4, then decodes. The caller validates the word count against its own
// expectation (so it can phrase that error precisely). Shared by exl2ReadInvPerm /
// exl2ReadCodes.
func exl2ReadU32Words(sf *safetensorsFile, name string) ([]uint32, error) {
	raw, e, err := exl2TensorRaw(sf, name)
	if err != nil {
		return nil, err
	}
	if e.Dtype != "I32" && e.Dtype != "U32" {
		return nil, fmt.Errorf("%s dtype %q, want I32/U32", name, e.Dtype)
	}
	if len(raw)%4 != 0 {
		return nil, fmt.Errorf("%s byte length %d not /4", name, len(raw))
	}
	n := len(raw) / 4
	words := make([]uint32, n)
	for i := 0; i < n; i++ {
		words[i] = binary.LittleEndian.Uint32(raw[i*4:])
	}
	return words, nil
}

// exl2ReadInvPerm parses q_invperm (int32 [in]) into the inverse act-order
// permutation. A length mismatch against the group-derived `in` is an error.
func exl2ReadInvPerm(sf *safetensorsFile, name string, in int) ([]int32, error) {
	words, err := exl2ReadU32Words(sf, name)
	if err != nil {
		return nil, err
	}
	if len(words) != in {
		return nil, fmt.Errorf("%s len %d != group-derived in %d", name, len(words), in)
	}
	perm := make([]int32, len(words))
	for i, w := range words {
		perm[i] = int32(w)
	}
	return perm, nil
}

// exl2ReadF16Vec reads an f16/f32 1-D tensor into a float32 vector.
func exl2ReadF16Vec(sf *safetensorsFile, name string) ([]float32, error) {
	raw, e, err := exl2TensorRaw(sf, name)
	if err != nil {
		return nil, err
	}
	f32, err := decodeSafetensorF32(name, e, raw)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(f32)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(f32[i*4:]))
	}
	return out, nil
}

// exl2ReadCodes reads q_weight (int32 bit-stream) into out*rowWords uint32 words.
func exl2ReadCodes(sf *safetensorsFile, name string, out, rowWords int) ([]uint32, error) {
	codes, err := exl2ReadU32Words(sf, name)
	if err != nil {
		return nil, err
	}
	if len(codes) != out*rowWords {
		return nil, fmt.Errorf("%s holds %d words, want out*rowWords %d", name, len(codes), out*rowWords)
	}
	return codes, nil
}

// exl2TensorRaw fetches a tensor's raw bytes + entry from the safetensors header.
func exl2TensorRaw(sf *safetensorsFile, name string) ([]byte, stEntry, error) {
	rawHdr, ok := sf.hdr[name]
	if !ok {
		return nil, stEntry{}, fmt.Errorf("missing tensor %s", name)
	}
	var e stEntry
	if err := json.Unmarshal(rawHdr, &e); err != nil {
		return nil, stEntry{}, fmt.Errorf("entry %s: %w", name, err)
	}
	b, err := sf.tensorBytes(e)
	if err != nil {
		return nil, stEntry{}, fmt.Errorf("read %s: %w", name, err)
	}
	return b, e, nil
}
