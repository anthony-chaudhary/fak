package model

// MIT License
//
// Copyright (c) 2023 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// v41_forward_engram.go wires DeepSeek V4.1 Engram packed-row retrieval into
// the reduced text forward (issue #13007, leaf of parent #12640). The retrieval
// primitives already existed as leaves — token hashing (V41EngramHashState.Hash),
// row gather (GatherV41EngramRows), the bounded row cache (V41EngramRowCache /
// V41EngramSetCache), and the 264-byte packed-row layout (E4M3 weights + E8M0
// scales) — but nothing executed them on the forward path. This file is the
// missing injection stage.
//
// Reference schedule: antirez/ds4 @ bd66c402 (ds4.c ds41_graph_before_attention,
// ds4_engram.c ds4_engram_read, metal/dsv41.metal kernel_dsv41_engram_add). At an
// Engram layer the injection happens at the START of the layer, into the residual,
// BEFORE attention and before attn_norm:
//
//  1. Hash the token (+ up to MaxNgramSize-1 tail history) to COLS row ids,
//     cols = (MaxNgramSize-1)*HeadsPerNgram.
//  2. Read COLS rows; each packed 264-byte row is 256 E4M3 bytes followed by 8
//     E8M0 scale bytes, one scale per 32 elements. value = e4m3(code) * 2^(scale-127),
//     rounded to bf16. The E8M0 0xff (NaN) byte and the E4M3 NaN byte (code&127 == 127)
//     are rejected, and a non-finite result fails closed.
//  3. Concatenate the COLS*EngramHeadDim values and project through the F16
//     engram_kv.weight of shape [COLS*DIM, (N_HC+1)*H] into N_HC key streams plus
//     one shared value stream (stream N_HC).
//  4. Per HC stream s: dot_s = sum_i h_s[i]*coef[s,i]*bf16(key_s[i]). The per-cell
//     coefficient is admitted in exactly ONE form per layer (#14121): either the
//     runtime product qNorm[s,i]*kNorm[s,i] of the two unfolded F32 norm tensors
//     (engram_q_norm.weight / engram_k_norm.weight), or one folded F32 tensor
//     engram_folded_gate.weight carrying the pre-composed product (the upstream DS4.1
//     reference keeps engram_q and engram_k apart but only ever uses their
//     product). Both forms reach the same host equation; co-presence is refused.
//     Then dot_s *= rsqrt(mean(h_s^2)+eps)*rsqrt(mean(key_s^2)+eps)*rsqrt(W).
//  5. gate_s = sigmoid(copysign(sqrt(max(|dot_s|,1e-6)), dot_s)).
//  6. Full reference: residual_s[i] = bf16(residual_s[i] + gate_s * bf16(value[i])).
//     The reduced compatibility path retains its historical delta rounding.
//
// Persistent-stream schedule (full geometry). The forward now carries four
// DISTINCT persistent mHC streams per position (v41_forward.go v41Layer: stream 0
// is the live hidden state, streams 1..3 are the reference's persistent residual
// streams). On the full path this file computes each HC stream's gate from that
// stream's OWN vector h_s = streams[t][s], and adds the shared value stream into
// the SAME stream:
//
//	streams[t][s][i] = bf16(streams[t][s][i] + gate_s * bf16(value[i]))
//
// This full residual-sum boundary follows official DeepSeek-V4.1-Flash
// dba1be0a40aa45a94ad051997016db3960a90277 inference/model.py:350–365 (MIT),
// not the earlier Metal delta-rounding approximation. Each stream has its own
// gate; no gated values are summed across streams. x is synchronized explicitly
// from stream 0, whether or not those caller buffers alias.
//
// Reduced-model reconciliation. The reduced fixture (v41_forward.go v41Layer) is
// not a four-stream residual: it carries one `x[t]` of width H and stands in four
// IDENTICAL streams for the reference's hc persistent state. Because those
// stand-in streams are identical at injection time, the reduced path preserves
// the pre-existing single-vector reduction byte-for-byte:
//
//	x[t][i] += sum_{s=0}^{N_HC-1} bf16(gate_s * bf16(value[i]))
//
// i.e. one `h := x[t]`, the four gates still computed from x[t], and the four
// per-stream gated values summed into the single vector. The gates still differ
// per stream (each reads its own key), so the reduction is not equivalent to one
// stream times four. The per-stream key and value are bf16-rounded exactly as the
// Metal kernel rounds them; only the final accumulation into the single stand-in
// vector is left in f32 because the four reference streams cannot be represented
// by one vector. The test oracle mirrors this reduction and matches within the
// CPU-oracle tolerance (the oracle accumulates the RMS means in f64, the forward
// in f32). The witness shares the retrieval primitives under test (Hash /
// GatherV41EngramRows / fp8E4M3ToF32) and independently transcribes only the
// projection + gate + mix arithmetic, so it is a genuine cross-implementation
// check for the injection stage, not for the retrieval leaves already witnessed
// by v41_engram_test.go / v41_engram_stream_test.go. Full-checkpoint parity stays
// [SW-VERIFIED]; no hardware witness is claimed.

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
	"weak"
)

// v41BF16 rounds x to the nearest bfloat16 value (round-half-to-even), matching
// dsv41_bf16 in metal/dsv41.metal: a finite value is rounded by adding
// 0x7fff + the low sign-adjacent bit, then truncating the low 16 bits. NaN and
// Inf bit patterns are left untouched (the caller has already rejected them).
// It counts toward the host BF16 witness like v41RoundBF16.
func v41BF16(x float32) float32 {
	if v41HostBF16Witness.Load() {
		v41HostBF16Rounds.Add(1)
	}
	bits := math.Float32bits(x)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}

// v41EngramStage is the model-attached Engram retrieval stage: a hash state plus
// one bounded row cache per declared Engram layer, in the layout's layer order.
type v41EngramStage struct {
	layout        V41EngramLayout
	hashPrototype *V41EngramHashState
	caches        []*V41EngramRowCache
	columns       int
	headDim       int
	hc            int
	layerIDs      []int
	// rowBytes is the per-row width every cached row has (264 packed, 84 Q2_K
	// blocks of a verified Q2_K source, or 1024 f32). The forward dispatches row
	// decode on it, so the dialects share one architecture-neutral stage.
	rowBytes int
	// bindings records the verified artifact binding each source was admitted
	// under (zero for an unverified source), so the exported attach seam's
	// receipt can prove the verified route was taken.
	bindings []V41EngramArtifactBinding
}

// v41EngramStages attaches a stage to a *Model without widening the Model struct
// (the stage is the #13007 leaf's private wiring). Keys are weak and a runtime
// cleanup deletes the entry once the model is unreachable, so a released model
// (a reload, or every test fixture) is not kept alive by its stage.
var v41EngramStages sync.Map // map[weak.Pointer[Model]]*v41EngramStage

// v41DetachEngramStage removes the model's wired Engram stage, if any.
func (m *Model) v41DetachEngramStage() { v41EngramStages.Delete(weak.Make(m)) }

// v41EngramStageFor returns the model's wired Engram stage, or nil.
func (m *Model) v41EngramStageFor() *v41EngramStage {
	if m == nil {
		return nil
	}
	v, ok := v41EngramStages.Load(weak.Make(m))
	runtime.KeepAlive(m)
	if !ok {
		return nil
	}
	stage, _ := v.(*v41EngramStage)
	return stage
}

// wireV41Engram builds and attaches the Engram retrieval stage for a model whose
// config declares EngramLayerIDs. srcs must supply one row source per declared
// Engram layer, in declaration order, each with the layout's per-layer row count.
// It is package-private: only the reduced forward and its test build the seam.
func (m *Model) wireV41Engram(layout V41EngramLayout, srcs []V41EngramRowSource, budgetBytes int64) error {
	if m == nil || m.Cfg.DeepSeekV41 == nil {
		return fmt.Errorf("%w: Engram stage needs a V4.1 config", ErrV41NativeUnsupported)
	}
	if err := layout.validate(); err != nil {
		return fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err)
	}
	ids := m.Cfg.DeepSeekV41.EngramLayerIDs
	if len(ids) != len(layout.Rows) || len(srcs) != len(layout.Rows) {
		return fmt.Errorf("%w: Engram declares %d layers, layout has %d row counts, %d sources",
			ErrV41NativeUnsupported, len(ids), len(layout.Rows), len(srcs))
	}
	columns := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram
	// Every source must serve the same dialect; a mixed-width set would make the
	// stage's row decode ambiguous, so a disagreement is a named refusal. A
	// source's own RowBytes() is authoritative — the stage never assumes packed.
	rowBytes := srcs[0].RowBytes()
	if rowBytes != V41EngramPackedRowBytes && rowBytes != V41EngramF32RowBytes {
		return fmt.Errorf("%w: Engram source row width %d is neither packed %d nor f32 %d",
			ErrV41NativeUnsupported, rowBytes, V41EngramPackedRowBytes, V41EngramF32RowBytes)
	}
	for i, src := range srcs {
		if src == nil {
			return fmt.Errorf("%w: nil Engram row source for layer %d", ErrV41NativeUnsupported, ids[i])
		}
		if got := src.RowBytes(); got != rowBytes {
			return fmt.Errorf("%w: Engram source for layer %d has row width %d, want %d (all sources must share one dialect)",
				ErrV41NativeUnsupported, ids[i], got, rowBytes)
		}
	}
	// A verified Q2_K source is cached as its bound super-blocks: the decode
	// then runs where the row is consumed, on the device when the session binds
	// one, instead of dequantizing every cache fill on the host.
	cached := srcs
	if rowBytes == V41EngramF32RowBytes {
		encoded := make([]V41EngramRowSource, len(srcs))
		for i, src := range srcs {
			form, ok := v41EngramEncodedForm(src)
			if !ok {
				encoded = nil
				break
			}
			encoded[i] = form
		}
		if encoded != nil {
			cached, rowBytes = encoded, v41EngramQ2KEncodedRowBytes
		}
	}
	caches := make([]*V41EngramRowCache, len(cached))
	for i, src := range cached {
		budget := budgetBytes
		if budget < int64(rowBytes) {
			budget = int64(rowBytes)
		}
		cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
			TableRows: int(layout.Rows[i]), RowBytes: rowBytes, BudgetBytes: budget,
		})
		if err != nil {
			return fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err)
		}
		caches[i] = cache
	}
	hashPrototype, err := NewV41EngramHashState(layout)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrV41NativeUnsupported, err)
	}
	stage := &v41EngramStage{
		layout: hashPrototype.layout, hashPrototype: hashPrototype, caches: caches, columns: columns,
		headDim: m.Cfg.DeepSeekV41.EngramHeadDim, hc: 4,
		layerIDs: append([]int(nil), ids...), rowBytes: rowBytes,
	}
	stage.bindings = make([]V41EngramArtifactBinding, len(srcs))
	for i, src := range srcs {
		stage.bindings[i], _ = V41EngramBinding(src)
	}
	key := weak.Make(m)
	if _, loaded := v41EngramStages.Swap(key, stage); !loaded {
		runtime.AddCleanup(m, func(k weak.Pointer[Model]) { v41EngramStages.Delete(k) }, key)
	}
	return nil
}

// cacheIndex returns the cache index for decoder layer l, or -1.
func (s *v41EngramStage) cacheIndex(l int) int {
	for i, id := range s.layerIDs {
		if id == l {
			return i
		}
	}
	return -1
}

// decodeV41EngramRow decodes one cached row of the stage's dialect into dim f32
// values on the host: the packed 264-byte dialect is dequantized (E4M3 weights *
// E8M0 scales), a Q2_K block (84 bytes) is dequantized by the scalar super-block
// reference, and the f32 dialect (1024 bytes) is read little-endian. Every
// dialect fails closed on a width it does not own.
func decodeV41EngramRow(row []byte, dim, rowBytes int) ([]float32, error) {
	v41NoteHostEngramRowDecode()
	switch rowBytes {
	case V41EngramPackedRowBytes:
		return dequantV41EngramRow(row, dim)
	case v41EngramQ2KEncodedRowBytes:
		return dequantV41EngramQ2KRow(row, dim)
	case V41EngramF32RowBytes:
		if len(row) != V41EngramF32RowBytes {
			return nil, fmt.Errorf("%w: Engram f32 row has %d bytes, want %d",
				ErrV41NativeUnsupported, len(row), V41EngramF32RowBytes)
		}
		if dim <= 0 || dim > qkK {
			return nil, fmt.Errorf("%w: Engram f32 row dim %d outside [1,%d]", ErrV41NativeUnsupported, dim, qkK)
		}
		out := make([]float32, dim)
		for j := 0; j < dim; j++ {
			value := math.Float32frombits(binary.LittleEndian.Uint32(row[j*4:]))
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("%w: Engram f32 row value %d is non-finite", ErrV41NativeUnsupported, j)
			}
			out[j] = value
		}
		return out, nil
	default:
		return nil, fmt.Errorf("%w: unsupported Engram row width %d", ErrV41NativeUnsupported, rowBytes)
	}
}

// dequantV41EngramRow dequantizes one packed 264-byte row into DIM f32 values.
// It mirrors ds4_engram_read: 256 E4M3 codes followed by 8 E8M0 scales, one per
// 32 elements. The E8M0 0xff NaN byte and the E4M3 NaN byte (code&127 == 127) are
// rejected, each value is scaled by 2^(scale-127) and bf16-rounded, and a
// non-finite result fails closed.
func dequantV41EngramRow(row []byte, dim int) ([]float32, error) {
	if len(row) != V41EngramPackedRowBytes {
		return nil, fmt.Errorf("%w: Engram packed row has %d bytes, want %d",
			ErrV41NativeUnsupported, len(row), V41EngramPackedRowBytes)
	}
	if dim <= 0 || dim > 256 {
		return nil, fmt.Errorf("%w: Engram row dim %d outside [1,256]", ErrV41NativeUnsupported, dim)
	}
	out := make([]float32, dim)
	for j := 0; j < dim; j++ {
		code := row[j]
		scale := row[256+j/32]
		if code&127 == 127 || scale == 255 {
			return nil, fmt.Errorf("%w: Engram row has a NaN code/scale byte at %d", ErrV41NativeUnsupported, j)
		}
		value := v41BF16(fp8E4M3ToF32(code) * float32(math.Ldexp(1, int(scale)-127)))
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("%w: Engram row value %d is non-finite", ErrV41NativeUnsupported, j)
		}
		out[j] = value
	}
	return out, nil
}

// v41EngramInject retrieves, dequantizes, projects, gates, and mixes the Engram
// rows for one Engram layer into the layer residual. On the full path it updates
// the four DISTINCT persistent streams (streams[t][s] += bf16(gate_s*value)) and
// keeps x synchronized with stream 0; on the reduced path it preserves the
// single-vector reduction into x documented at the top of this file. It is the
// production transposition of the reference schedule and is called at the START
// of v41Layer for declared Engram layers.
func (m *Model) v41EngramInject(l int, x [][]float32, streams [][][]float32, full bool, seq []int, eps float32) error {
	return m.v41EngramInjectWithProjection(l, x, streams, full, seq, eps, nil)
}

func (m *Model) v41EngramInjectWithProjection(l int, x [][]float32, streams [][][]float32, full bool, seq []int, eps float32, project v41EngramProjectionFunc) error {
	geometry, layerRows, err := m.v41EngramLayerRows(l, seq)
	if err != nil {
		return err
	}
	geometry.project = project
	_, err = m.v41EngramInjectPrepared(l, x, streams, full, len(seq), layerRows, eps, geometry)
	return err
}

// v41EngramLayerRows hashes seq from the sequence start and returns layer l's
// row identifiers, position-major [len(seq), cols].
func (m *Model) v41EngramLayerRows(l int, seq []int) (v41EngramGeometry, []uint32, error) {
	geometry, err := m.v41EngramInjectionGeometry(l, len(seq))
	if err != nil {
		return v41EngramGeometry{}, nil, err
	}
	stage, cacheIdx := geometry.stage, geometry.cacheIdx
	layerRows, err := stage.hashPrototype.Clone().hashLayers(seq, nil, cacheIdx, cacheIdx+1)
	if err != nil {
		return v41EngramGeometry{}, nil, v41StageErr(v41StageEngram, l, fmt.Errorf("%w: %w", ErrV41NativeUnsupported, err))
	}
	return geometry, layerRows, nil
}

// v41EngramFoldedGateLeaf is the model-side leaf name of the folded Engram gate
// coefficient tensor: one F32 cell per (HC stream, hidden index), carrying the
// elementwise product q[s,i]*k[s,i] that the unfolded path recomputes at
// runtime. The upstream DS4.1 converter may publish this product directly
// instead of the two separate norm tensors; exactly one form per Engram layer
// is admitted (v41EngramInjectionGeometry). Loader aliasing of the upstream
// GGUF spelling is a follow-up leaf (ggufload); this constant pins the
// model-side contract.
const v41EngramFoldedGateLeaf = "engram_folded_gate.weight"

// v41EngramGateParamsErr rejects non-finite gate parameters before retrieval
// or publication. Finite zero/negative epsilon and gains stay admitted, as the
// host equation accepts them.
func v41EngramGateParamsErr(l int, eps float32, qNorm, kNorm []float32) error {
	if !finite32(eps) {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram gate epsilon is non-finite", ErrV41NativeUnsupported))
	}
	for _, norm := range [...]struct {
		leaf   string
		values []float32
	}{{"engram_q_norm.weight", qNorm}, {"engram_k_norm.weight", kNorm}} {
		for i, value := range norm.values {
			if !finite32(value) {
				return v41StageErr(v41StageEngram, l,
					fmt.Errorf("%w: Engram %s value[%d] is non-finite", ErrV41NativeUnsupported, norm.leaf, i))
			}
		}
	}
	return nil
}

// v41EngramFoldedCoefErr rejects non-finite folded gate coefficients before
// retrieval or publication, mirroring v41EngramGateParamsErr for the folded
// form (#14121). Geometry resolution already enforces it; this is the
// defense-in-depth recheck for geometries built outside
// v41EngramInjectionGeometry. An empty slice (the unfolded form) passes.
func v41EngramFoldedCoefErr(l int, coef []float32) error {
	for i, value := range coef {
		if !finite32(value) {
			return v41StageErr(v41StageEngram, l,
				fmt.Errorf("%w: Engram %s value[%d] is non-finite", ErrV41NativeUnsupported, v41EngramFoldedGateLeaf, i))
		}
	}
	return nil
}

// v41EngramDecodePosition decodes position t's cols gathered rows into dst
// [cols*dim], the projection input of that position.
func v41EngramDecodePosition(l int, stage *v41EngramStage, cacheIdx int, gathered [][]byte, t, cols, dim int, dst []float32) error {
	for c := 0; c < cols; c++ {
		values, err := decodeV41EngramRow(gathered[t*cols+c], dim, stage.rowBytes)
		if err != nil {
			// A successful source read can poison other cached rows in this layer.
			stage.caches[cacheIdx].invalidateRows()
			return v41StageErr(v41StageEngram, l, err)
		}
		copy(dst[c*dim:], values)
	}
	return nil
}

type v41EngramGeometry struct {
	stage            *v41EngramStage
	cacheIdx         int
	H, dim, cols, hc int
	qNorm, kNorm     []float32
	// coef holds the folded gate coefficients (one form per layer, #14121);
	// it is non-nil exactly when the layer was admitted with engram_folded_gate.weight
	// instead of the unfolded q/k norms, which are then empty.
	coef    []float32
	project v41EngramProjectionFunc
}

func (m *Model) v41EngramInjectionGeometry(l, positions int) (v41EngramGeometry, error) {
	stage := m.v41EngramStageFor()
	if stage == nil || stage.hashPrototype == nil {
		return v41EngramGeometry{}, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: layer %d declares Engram but no row source is wired", ErrV41NativeUnsupported, l))
	}
	cacheIdx := stage.cacheIndex(l)
	if cacheIdx < 0 {
		return v41EngramGeometry{}, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: layer %d is not a declared Engram layer", ErrV41NativeUnsupported, l))
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := stage.headDim
	cols := stage.columns
	hc := stage.hc

	_, _, normCells, err := m.v41EngramProjectionDimensions(l, stage)
	if err != nil {
		return v41EngramGeometry{}, err
	}
	if _, ok := checkedMulInt(positions, cols); positions < 0 || !ok {
		return v41EngramGeometry{}, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram row identifiers %d positions x %d columns overflow", ErrV41NativeUnsupported, positions, cols))
	}
	// Exactly ONE gate-coefficient form per layer (#14121): the two unfolded
	// F32 norms whose product the gate recomputes, or one folded F32 tensor
	// carrying that product pre-composed. Co-presence is ambiguous and refused;
	// neither present falls through to the historical absence refusal.
	unfolded := m.has(layerName(l, "engram_q_norm.weight")) && m.has(layerName(l, "engram_k_norm.weight"))
	folded := m.has(layerName(l, v41EngramFoldedGateLeaf))
	switch {
	case folded && unfolded:
		return v41EngramGeometry{}, v41StageErr(v41StageEngram, l, fmt.Errorf(
			"%w: Engram gate coefficients are ambiguous at layer %d: %s coexists with the unfolded q/k norms; exactly one form is admitted",
			ErrV41NativeUnsupported, l, layerName(l, v41EngramFoldedGateLeaf)))
	case folded:
		coef := m.tensor(layerName(l, v41EngramFoldedGateLeaf))
		if len(coef) != normCells {
			return v41EngramGeometry{}, v41StageErr(v41StageEngram, l,
				fmt.Errorf("%w: Engram folded gate coefficients are mis-shaped at layer %d", ErrV41NativeUnsupported, l))
		}
		if err := v41EngramFoldedCoefErr(l, coef); err != nil {
			return v41EngramGeometry{}, err
		}
		return v41EngramGeometry{stage: stage, cacheIdx: cacheIdx, H: H, dim: dim, cols: cols, hc: hc, coef: coef}, nil
	default:
		// Presence first: a declared Engram layer missing either unfolded norm is
		// refused here as a typed error instead of the load-time panic inside
		// tensor(), keeping the fail-closed contract uniform across both forms.
		if !m.has(layerName(l, "engram_q_norm.weight")) || !m.has(layerName(l, "engram_k_norm.weight")) {
			return v41EngramGeometry{}, v41StageErr(v41StageEngram, l, fmt.Errorf("%w: Engram mixing norms are absent or mis-shaped at layer %d", ErrV41NativeUnsupported, l))
		}
		qNorm := m.tensor(layerName(l, "engram_q_norm.weight"))
		kNorm := m.tensor(layerName(l, "engram_k_norm.weight"))
		if len(qNorm) != normCells || len(kNorm) != normCells {
			return v41EngramGeometry{}, v41StageErr(v41StageEngram, l, fmt.Errorf("%w: Engram mixing norms are absent or mis-shaped at layer %d", ErrV41NativeUnsupported, l))
		}
		return v41EngramGeometry{stage: stage, cacheIdx: cacheIdx, H: H, dim: dim, cols: cols, hc: hc, qNorm: qNorm, kNorm: kNorm}, nil
	}
}

func (m *Model) v41EngramInjectPrepared(l int, x [][]float32, streams [][][]float32, full bool, positions int, layerRows []uint32, eps float32, geometry v41EngramGeometry) (gatheredRows int, err error) {
	return m.v41EngramInjectOwnedPrepared(l, x, streams, full, positions, layerRows, eps, geometry, nil)
}

// v41EngramInjectOwnedPrepared consumes a current-call owned panel after a
// resident gate declines. A nil panel keeps the ordinary host preflight and
// gather order; retained panels are never stored on the geometry or session.
func (m *Model) v41EngramInjectOwnedPrepared(l int, x [][]float32, streams [][][]float32, full bool, positions int, layerRows []uint32, eps float32, geometry v41EngramGeometry, gathered *v41EngramGatheredRows) (gatheredRows int, err error) {
	stage, cacheIdx := geometry.stage, geometry.cacheIdx
	H, dim, cols, hc := geometry.H, geometry.dim, geometry.cols, geometry.hc
	qNorm, kNorm := geometry.qNorm, geometry.kNorm
	project := m.v41EngramProjector(l, (hc+1)*H, cols*dim, geometry.project)
	if positions < 0 || len(layerRows) != positions*cols {
		return gatheredRows, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram row identifiers %d, want %d positions x %d columns", ErrV41NativeUnsupported, len(layerRows), positions, cols))
	}
	// These shared gate parameters are consumed by every position.
	if err := v41EngramGateParamsErr(l, eps, qNorm, kNorm); err != nil {
		return gatheredRows, err
	}
	// Folded form (#14121): the coefficient cells replace the runtime q*k
	// product. Geometry resolution already validated shape and finiteness; this
	// recheck covers geometries built outside v41EngramInjectionGeometry.
	coef := geometry.coef
	if err := v41EngramFoldedCoefErr(l, coef); err != nil {
		return gatheredRows, err
	}
	if coef == nil {
		// Unfolded form: materialize the per-cell product once per call so the
		// hot loop stays uniform. Each cell is the same f32 product the legacy
		// inline expression computed, so publications stay bit-identical.
		coef = make([]float32, hc*H)
		for s := 0; s < hc; s++ {
			for i := 0; i < H; i++ {
				coef[s*H+i] = qNorm[s*H+i] * kNorm[s*H+i]
			}
		}
	}
	// Both paths use x[t]: reduced gates index its hidden row, while full gates
	// publish stream 0 back into it. Validate every used row before gathering or
	// mutating any position, so missing rows cannot panic and copy cannot truncate.
	if len(x) < positions {
		return gatheredRows, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram hidden geometry has %d positions for %d tokens",
				ErrV41NativeUnsupported, len(x), positions))
	}
	for t := range positions {
		if len(x[t]) != H {
			return gatheredRows, v41StageErr(v41StageEngram, l,
				fmt.Errorf("%w: Engram hidden geometry position %d has width %d, want %d",
					ErrV41NativeUnsupported, t, len(x[t]), H))
		}
	}
	if gathered == nil {
		gathered, err = m.v41EngramGatherRows(l, positions, layerRows, eps, geometry)
		if err != nil {
			return gatheredRows, err
		}
	} else if gathered.layer != l || gathered.stage != stage || gathered.cacheIdx != cacheIdx || gathered.dim != dim || len(gathered.rows) != len(layerRows) {
		return gatheredRows, v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram retained row owner does not match current injection", ErrV41NativeUnsupported))
	}

	gatheredRows = len(layerRows)
	rowVec := make([]float32, cols*dim)

	// Fail closed on malformed full geometry before any stream is mutated: the
	// full path indexes streams[t][s] directly, so a short position or stream
	// would otherwise panic instead of returning a typed error. The reduced path
	// carries no persistent streams and is unaffected.
	if full {
		if len(streams) < positions {
			return gatheredRows, v41StageErr(v41StageEngram, l,
				fmt.Errorf("%w: Engram full geometry has %d stream positions for %d tokens",
					ErrV41NativeUnsupported, len(streams), positions))
		}
		for t := range positions {
			if len(streams[t]) < hc {
				return gatheredRows, v41StageErr(v41StageEngram, l,
					fmt.Errorf("%w: Engram full geometry position %d has %d streams, want >= %d",
						ErrV41NativeUnsupported, t, len(streams[t]), hc))
			}
			for s := 0; s < hc; s++ {
				if len(streams[t][s]) != H {
					return gatheredRows, v41StageErr(v41StageEngram, l,
						fmt.Errorf("%w: Engram full geometry stream [%d][%d] has width %d, want %d",
							ErrV41NativeUnsupported, t, s, len(streams[t][s]), H))
				}
			}
		}
	}
	for t := range positions {
		if err := v41EngramDecodePosition(l, stage, cacheIdx, gathered.rows, t, cols, dim, rowVec); err != nil {
			return gatheredRows, err
		}
		// Project the COLS*DIM concatenation through engram_kv.weight
		// [COLS*DIM, (N_HC+1)*H]: output stream s is the key for HC stream s and
		// stream N_HC is the single shared value.
		projected, projectionErr := project(rowVec)
		if projectionErr != nil {
			return gatheredRows, projectionErr
		}
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41BF16(projected[hc*H+i])
		}
		h := x[t]
		// acc is consumed only by the reduced-path reduction below; the full path
		// writes each stream in place. Allocate it lazily so the full path keeps no
		// unused per-position accumulator.
		var acc []float32
		if !full {
			acc = make([]float32, H)
		}
		for s := 0; s < hc; s++ {
			// Full geometry: each HC stream gates the shared value stream with its
			// OWN persistent vector and adds the result back into that same stream.
			// Reduced geometry: the four stand-in streams are identical, so the
			// documented reduction gates every stream from the single x[t].
			hs := h
			if full {
				hs = streams[t][s]
			}
			var h2, k2, dotSum float32
			for i := 0; i < H; i++ {
				key := v41BF16(projected[s*H+i])
				hi := hs[i]
				h2 += float32(hi * hi)
				k2 += float32(key * key)
				dotSum += float32(float32(hi*coef[s*H+i]) * key)
			}
			dot := dotSum * rsqrtf(h2/float32(H)+eps) * rsqrtf(k2/float32(H)+eps) * rsqrtf(float32(H))
			gate := float32(1) / (1 + expf(-copysignf(sqrtf(maxf(absf(dot), 1e-6)), dot)))
			if full {
				// Each stream receives its own gated value; do NOT sum across streams.
				stream := streams[t][s]
				for i := 0; i < H; i++ {
					// Keep the product F32 until after residual addition, then publish BF16.
					delta := float32(gate * value[i])
					published := v41BF16(float32(stream[i] + delta))
					if !finite32(published) {
						return gatheredRows, v41StageErr(v41StageEngram, l,
							fmt.Errorf("%w: Engram full residual [%d][%d][%d] is non-finite", ErrV41NativeUnsupported, t, s, i))
					}
					stream[i] = published
				}
			} else {
				for i := 0; i < H; i++ {
					acc[i] += v41BF16(gate * value[i])
				}
			}
		}
		if full {
			// Keep x synchronized for both aliased and independently owned callers.
			copy(x[t], streams[t][0])
		} else {
			for i := 0; i < H; i++ {
				published := h[i] + acc[i]
				if !finite32(published) {
					return gatheredRows, v41StageErr(v41StageEngram, l,
						fmt.Errorf("%w: Engram reduced residual [%d][%d] is non-finite", ErrV41NativeUnsupported, t, i))
				}
				h[i] = published
			}
		}
	}
	return gatheredRows, nil
}

// V41EngramWarmDelta reports the cache-counter movement caused by one
// WarmV41EngramPrefix call. It is the machine-checkable receipt the startup
// warmer hands to its caller: a no-op warm (cold-only, unsupported model, or a
// fully resident prefix) reports zero on `Reads`/`BytesRead` rather than a
// success with no work.
type V41EngramWarmDelta struct {
	// Layers is the number of declared Engram layers touched.
	Layers int `json:"layers"`
	// Requested is the number of row addresses presented to the caches.
	Requested int64 `json:"requested"`
	// Hits is the number of requested rows already resident before this call.
	Hits int64 `json:"hits"`
	// Misses is the number of requested rows that required a backing read.
	Misses int64 `json:"misses"`
	// BytesRead is the number of backing bytes read by this call.
	BytesRead int64 `json:"bytes_read"`
}

// Resident reports whether the call left every requested row resident (no
// requested row needed a backing read) and at least one row was requested.
func (d V41EngramWarmDelta) Resident() bool { return d.Requested > 0 && d.Misses == 0 }

// WarmV41EngramPrefix prefetches the Engram rows a token prefix will consume
// into the very caches the reduced forward reads, WITHOUT dequantizing,
// projecting, or advancing any live hash history. It is the bounded public entry
// point the startup warmer (CW-11) binds to; warming a detached stream would
// fill caches the forward never touches, so this method drives the
// model-attached stage directly.
//
// The row addresses are derived from the stage's own hash layout by hashing the
// prefix with a fresh state — exactly the sequence-start schedule the reduced
// forward recomputes each call (see v41EngramInject) — so a following forward
// over the same tokens sees cache hits. Binding is verified before any read: a
// model with no V4.1 config, no declared Engram layers, no wired stage, or a
// stage whose per-layer geometry disagrees with the hash layout is refused with
// ErrV41NativeUnsupported rather than reporting a warm success.
//
// A cold fallback is always preserved: an unsupported model returns
// ErrV41NativeUnsupported and callers keep serving un-warmed.
func (m *Model) WarmV41EngramPrefix(tokens []int, mask []bool) (V41EngramWarmDelta, error) {
	var zero V41EngramWarmDelta
	if m == nil || m.Cfg.DeepSeekV41 == nil {
		return zero, fmt.Errorf("%w: Engram warm needs a V4.1 config", ErrV41NativeUnsupported)
	}
	if len(m.Cfg.DeepSeekV41.EngramLayerIDs) == 0 {
		return zero, fmt.Errorf("%w: model declares no Engram layers", ErrV41NativeUnsupported)
	}
	stage := m.v41EngramStageFor()
	if stage == nil {
		return zero, fmt.Errorf("%w: Engram stage is not wired", ErrV41NativeUnsupported)
	}
	if len(stage.caches) != len(stage.layout.Rows) || len(stage.layerIDs) != len(stage.caches) {
		return zero, fmt.Errorf("%w: Engram stage geometry is inconsistent", ErrV41NativeUnsupported)
	}

	// Hash the prefix from a fresh state. The reduced forward recomputes the
	// whole history each call, so a fresh hash (tails start DEAD) is exactly the
	// sequence-start schedule the forward will re-derive; nothing live is touched.
	hash, err := NewV41EngramHashState(stage.layout)
	if err != nil {
		return zero, v41StageErr(v41StageEngram, -1, fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
	}
	rows, err := hash.Hash(tokens, mask)
	if err != nil {
		return zero, v41StageErr(v41StageEngram, -1, fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
	}
	cols := stage.columns
	layers := len(stage.caches)
	stride := layers * cols
	if stride <= 0 || len(rows)%stride != 0 {
		return zero, v41StageErr(v41StageEngram, -1,
			fmt.Errorf("%w: hashed row geometry is not a whole number of tokens", ErrV41NativeUnsupported))
	}

	// Demand reads retain rows in the layer's bounded cache. Deduplicate
	// prefix addresses so each row is warmed once.
	before := make([]V41EngramRowStats, layers)
	for i, cache := range stage.caches {
		before[i] = cache.Stats()
	}
	for layer, cache := range stage.caches {
		seen := make(map[uint32]struct{}, len(rows)/layers)
		for token := 0; token < len(rows)/stride; token++ {
			lo := token*stride + layer*cols
			for _, row := range rows[lo : lo+cols] {
				if _, ok := seen[row]; ok {
					continue
				}
				seen[row] = struct{}{}
				if err := cache.warmRow(int(row)); err != nil {
					return zero, v41StageErr(v41StageEngram, stage.layerIDs[layer],
						fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
				}
			}
		}
	}

	// Snapshot differences include concurrent traffic through these shared caches.
	var delta V41EngramWarmDelta
	delta.Layers = layers
	for i, cache := range stage.caches {
		now := cache.Stats()
		delta.Hits += now.Hits - before[i].Hits
		delta.Misses += now.Misses - before[i].Misses
		delta.BytesRead += now.BytesRead - before[i].BytesRead
	}
	delta.Requested = delta.Hits + delta.Misses
	return delta, nil
}

// ---- scalar helpers (kept local so the Engram math is self-contained) --------

func absf(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func copysignf(mag, sign float32) float32 {
	return float32(math.Copysign(float64(mag), float64(sign)))
}

func sqrtf(x float32) float32 { return float32(math.Sqrt(float64(x))) }

func rsqrtf(x float32) float32 { return float32(1 / math.Sqrt(float64(x))) }

func expf(x float32) float32 { return float32(math.Exp(float64(x))) }

func (m *Model) v41IncrementalEngramRows(stage *v41EngramStage, history []int, id, layer int) (rows []uint32, err error) {
	opened := m.v41NowNanos()
	hashed := 0
	defer func() {
		var nanos int64
		if opened != 0 {
			nanos = m.v41NowNanos() - opened
		}
		m.v41NoteIncrementalEngram(0, 0, hashed, nanos)
	}()
	if stage == nil || stage.hashPrototype == nil {
		return nil, v41StageErr(v41StageEngram, layer,
			fmt.Errorf("%w: layer %d declares Engram but no row source is wired", ErrV41NativeUnsupported, layer))
	}
	if _, err := m.v41EngramInjectionGeometry(layer, 1); err != nil {
		return nil, err
	}
	hash := stage.hashPrototype.Clone()
	start := len(history) - len(hash.tail)
	if start < 0 {
		start = 0
	}
	if _, err := hash.Hash(history[start:], nil); err != nil {
		return nil, v41StageErr(v41StageEngram, layer, fmt.Errorf("%w: %w", ErrV41NativeUnsupported, err))
	}
	hashed = len(history) - start
	rows, err = hash.Hash([]int{id}, nil)
	if err != nil {
		return nil, v41StageErr(v41StageEngram, layer, fmt.Errorf("%w: %w", ErrV41NativeUnsupported, err))
	}
	hashed++
	return rows, nil
}
