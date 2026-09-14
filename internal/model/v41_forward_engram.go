package model

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
//  4. Per HC stream s: dot_s = sum_i h_s[i]*(qNorm[i,s]*kNorm[i,s])*bf16(key_s[i]),
//     then dot_s *= rsqrt(mean(h_s^2)+eps)*rsqrt(mean(key_s^2)+eps)*rsqrt(W).
//  5. gate_s = sigmoid(copysign(sqrt(max(|dot_s|,1e-6)), dot_s)).
//  6. residual_s[i] += bf16(gate_s * bf16(value[i])).
//
// Reduced-model reconciliation. The existing reduced forward (v41_forward.go
// v41Layer) is not a four-stream residual: it carries one `x[t]` of width H and
// stands in four IDENTICAL streams for the reference's hc persistent state (the
// comment at v41_forward.go:540-542). The reference injects the same shared value
// stream into each of the four HC streams, each scaled by that stream's own gate.
// Because the four stand-in streams are identical at injection time, this file
// collapses the write-back to the documented reduction
//
//	x[t][i] += sum_{s=0}^{N_HC-1} bf16(gate_s * bf16(value[i]))
//
// i.e. the single vector receives the sum of the four per-stream gated values.
// The gates still differ per stream (they read that stream's key), so the
// reduction is not equivalent to one stream times four. The per-stream key and
// value are bf16-rounded exactly as the Metal kernel rounds them; only the final
// accumulation into the single stand-in vector is left in f32 because the four
// reference streams cannot be represented by one vector. The test oracle mirrors
// this reduction and matches within the CPU-oracle tolerance (the oracle
// accumulates the RMS means in f64, the forward in f32). The witness shares the
// retrieval primitives under test (Hash / GatherV41EngramRows / fp8E4M3ToF32)
// and independently transcribes only the projection + gate + mix arithmetic, so
// it is a genuine cross-implementation check for the injection stage, not for the
// retrieval leaves already witnessed by v41_engram_test.go / v41_engram_stream_test.go.
// Full-checkpoint parity stays [SW-VERIFIED]; no hardware witness is claimed.

import (
	"fmt"
	"math"
	"sync"
)

// v41BF16 rounds x to the nearest bfloat16 value (round-half-to-even), matching
// dsv41_bf16 in metal/dsv41.metal: a finite value is rounded by adding
// 0x7fff + the low sign-adjacent bit, then truncating the low 16 bits. NaN and
// Inf bit patterns are left untouched (the caller has already rejected them).
func v41BF16(x float32) float32 {
	bits := math.Float32bits(x)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}

// v41EngramStage is the model-attached Engram retrieval stage: a hash state plus
// one bounded row cache per declared Engram layer, in the layout's layer order.
type v41EngramStage struct {
	layout   V41EngramLayout
	caches   []*V41EngramRowCache
	columns  int
	headDim  int
	hc       int
	layerIDs []int
}

// v41EngramStages attaches a stage to a *Model without widening the Model struct
// (the stage is the #13007 leaf's private wiring). It mirrors the existing
// metalResidentReady map[*Model] pattern; a model is loaded once and the entry is
// read for the life of that model.
var v41EngramStages sync.Map // map[*Model]*v41EngramStage

// v41EngramStageFor returns the model's wired Engram stage, or nil.
func (m *Model) v41EngramStageFor() *v41EngramStage {
	if m == nil {
		return nil
	}
	v, ok := v41EngramStages.Load(m)
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
	caches := make([]*V41EngramRowCache, len(srcs))
	for i, src := range srcs {
		if src == nil {
			return fmt.Errorf("%w: nil Engram row source for layer %d", ErrV41NativeUnsupported, ids[i])
		}
		budget := budgetBytes
		if budget < int64(V41EngramPackedRowBytes) {
			budget = int64(V41EngramPackedRowBytes)
		}
		cache, err := NewV41EngramRowCache(src, V41EngramRowCacheOptions{
			TableRows: int(layout.Rows[i]), RowBytes: V41EngramPackedRowBytes, BudgetBytes: budget,
		})
		if err != nil {
			return fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err)
		}
		caches[i] = cache
	}
	stage := &v41EngramStage{
		layout: layout, caches: caches, columns: columns,
		headDim: m.Cfg.DeepSeekV41.EngramHeadDim, hc: 4,
		layerIDs: append([]int(nil), ids...),
	}
	v41EngramStages.Store(m, stage)
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
// rows for one Engram layer into the reduced single-vector residual x. It is the
// production transposition of the reference schedule documented at the top of
// this file and is called at the START of v41Layer for declared Engram layers.
func (m *Model) v41EngramInject(l int, x [][]float32, seq []int, eps float32) error {
	stage := m.v41EngramStageFor()
	if stage == nil {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: layer %d declares Engram but no row source is wired", ErrV41NativeUnsupported, l))
	}
	cacheIdx := stage.cacheIndex(l)
	if cacheIdx < 0 {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: layer %d is not a declared Engram layer", ErrV41NativeUnsupported, l))
	}
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := stage.headDim
	cols := stage.columns
	hc := stage.hc
	if dim != H {
		// The reduced fixture uses EngramHeadDim == H so the concatenated row
		// width and the projection remain self-consistent; a mismatch is refused
		// rather than silently projected against the wrong geometry.
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram head dim %d != hidden %d", ErrV41NativeUnsupported, dim, H))
	}

	wKV := m.tensor(layerName(l, "engram_kv.weight"))
	qNorm := m.tensor(layerName(l, "engram_q_norm.weight"))
	kNorm := m.tensor(layerName(l, "engram_k_norm.weight"))
	if len(wKV) != cols*dim*(hc+1)*H || len(qNorm) != hc*H || len(kNorm) != hc*H {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram mixing tensors are absent or mis-shaped at layer %d", ErrV41NativeUnsupported, l))
	}

	// Hash the full token sequence for this layer from a fresh state. The
	// reduced forward recomputes the whole history each call, so a fresh hash
	// (tails start DEAD) is exactly the sequence-start schedule.
	hash, err := NewV41EngramHashState(stage.layout)
	if err != nil {
		return v41StageErr(v41StageEngram, l, fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
	}
	allRows, err := hash.Hash(seq, nil)
	if err != nil {
		return v41StageErr(v41StageEngram, l, fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
	}
	layers := len(stage.layout.Rows)
	// Extract this layer's [token][col] rows in the order GatherV41EngramRows
	// expects (one cache, columnsPerLayer == cols).
	layerRows := make([]uint32, 0, len(seq)*cols)
	for t := range seq {
		base := t*layers*cols + cacheIdx*cols
		layerRows = append(layerRows, allRows[base:base+cols]...)
	}
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIdx]}, layerRows, cols)
	if err != nil {
		return v41StageErr(v41StageEngram, l, fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
	}

	rowVec := make([]float32, cols*dim)
	projected := make([]float32, (hc+1)*H)
	for t := range seq {
		for c := 0; c < cols; c++ {
			values, err := dequantV41EngramRow(gathered[t*cols+c], dim)
			if err != nil {
				return v41StageErr(v41StageEngram, l, err)
			}
			copy(rowVec[c*dim:], values)
		}
		// Project the COLS*DIM concatenation through engram_kv.weight
		// [COLS*DIM, (N_HC+1)*H]: output stream s is the key for HC stream s and
		// stream N_HC is the single shared value.
		copy(projected, matRows(wKV, rowVec, (hc+1)*H, cols*dim))
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41BF16(projected[hc*H+i])
		}
		h := x[t]
		acc := make([]float32, H)
		for s := 0; s < hc; s++ {
			key := make([]float32, H)
			var h2, k2, dotSum float32
			for i := 0; i < H; i++ {
				key[i] = v41BF16(projected[s*H+i])
				hi := h[i]
				h2 += hi * hi
				k2 += key[i] * key[i]
				dotSum += hi * (qNorm[s*H+i] * kNorm[s*H+i]) * key[i]
			}
			dot := dotSum * rsqrtf(h2/float32(H)+eps) * rsqrtf(k2/float32(H)+eps) * rsqrtf(float32(H))
			gate := float32(1) / (1 + expf(-copysignf(sqrtf(maxf(absf(dot), 1e-6)), dot)))
			for i := 0; i < H; i++ {
				acc[i] += v41BF16(gate * value[i])
			}
		}
		for i := 0; i < H; i++ {
			h[i] += acc[i]
		}
	}
	return nil
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
