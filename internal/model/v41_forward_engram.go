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
// Persistent-stream schedule (full geometry). The forward now carries four
// DISTINCT persistent mHC streams per position (v41_forward.go v41Layer: stream 0
// is the live hidden state, streams 1..3 are the reference's persistent residual
// streams). On the full path this file computes each HC stream's gate from that
// stream's OWN vector h_s = streams[t][s], and adds the shared value stream into
// the SAME stream:
//
//	streams[t][s][i] += bf16(gate_s * bf16(value[i]))   for s = 0..N_HC-1
//
// so the four streams diverge exactly as the reference schedule does and no
// per-stream gated value is summed across streams. Stream 0 aliases x[t] on the
// full path (v41_forward.go:2029-2036); x is re-synchronized from streams[t][0]
// after the write so the two views can never drift.
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
	// rowBytes is the per-row width every wired source serves (264 packed or
	// 1024 dequantized-f32). The forward dispatches row decode on it, so the
	// packed and f32 dialects share one architecture-neutral stage.
	rowBytes int
	// bindings records the verified artifact binding each source was admitted
	// under (zero for an unverified source), so the exported attach seam's
	// receipt can prove the verified route was taken.
	bindings []V41EngramArtifactBinding
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
	caches := make([]*V41EngramRowCache, len(srcs))
	for i, src := range srcs {
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
	stage := &v41EngramStage{
		layout: layout, caches: caches, columns: columns,
		headDim: m.Cfg.DeepSeekV41.EngramHeadDim, hc: 4,
		layerIDs: append([]int(nil), ids...), rowBytes: rowBytes,
	}
	stage.bindings = make([]V41EngramArtifactBinding, len(srcs))
	for i, src := range srcs {
		stage.bindings[i], _ = V41EngramBinding(src)
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

// decodeV41EngramRow decodes one cached row of the stage's dialect into dim f32
// values: the packed 264-byte dialect is dequantized (E4M3 weights * E8M0 scales),
// and the dequantized-f32 dialect (1024 bytes, the published Q2_K table) is read
// little-endian. Both dialects fail closed on a width they do not own.
func decodeV41EngramRow(row []byte, dim, rowBytes int) ([]float32, error) {
	switch rowBytes {
	case V41EngramPackedRowBytes:
		return dequantV41EngramRow(row, dim)
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
	// The projection is rectangular in general: the concatenated Engram row
	// width dim is independent of the hidden width H, so a D != H declaration is
	// admitted and projected rather than refused. Validate the declared geometry
	// first, and reject any product that would overflow int before lengths,
	// allocation, indexing, or mutation are attempted.
	if dim < 1 || dim > 256 || H <= 0 {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram geometry head dim %d outside [1,256] or hidden %d not positive", ErrV41NativeUnsupported, dim, H))
	}
	if cols <= 0 {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram geometry columns %d must be positive", ErrV41NativeUnsupported, cols))
	}
	rowCells, ok := checkedMulInt(cols, dim)
	if !ok {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram row geometry %d cols x %d dim overflows", ErrV41NativeUnsupported, cols, dim))
	}
	normCells, ok := checkedMulInt(hc, H)
	if !ok {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram norm geometry %d streams x %d hidden overflows", ErrV41NativeUnsupported, hc, H))
	}
	streamWidth, ok := checkedMulInt(hc+1, H)
	if !ok {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram projection width %d streams x %d hidden overflows", ErrV41NativeUnsupported, hc+1, H))
	}
	kvCells, ok := checkedMulInt(rowCells, streamWidth)
	if !ok {
		return v41StageErr(v41StageEngram, l,
			fmt.Errorf("%w: Engram projection geometry %d x %d overflows", ErrV41NativeUnsupported, rowCells, streamWidth))
	}

	wKV := m.tensor(layerName(l, "engram_kv.weight"))
	qNorm := m.tensor(layerName(l, "engram_q_norm.weight"))
	kNorm := m.tensor(layerName(l, "engram_k_norm.weight"))
	if len(wKV) != kvCells || len(qNorm) != normCells || len(kNorm) != normCells {
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
	// Fail closed on malformed full geometry before any stream is mutated: the
	// full path indexes streams[t][s] directly, so a short position or stream
	// would otherwise panic instead of returning a typed error. The reduced path
	// carries no persistent streams and is unaffected.
	if full {
		if len(streams) < len(seq) {
			return v41StageErr(v41StageEngram, l,
				fmt.Errorf("%w: Engram full geometry has %d stream positions for %d tokens",
					ErrV41NativeUnsupported, len(streams), len(seq)))
		}
		for t := range seq {
			if len(streams[t]) < hc {
				return v41StageErr(v41StageEngram, l,
					fmt.Errorf("%w: Engram full geometry position %d has %d streams, want >= %d",
						ErrV41NativeUnsupported, t, len(streams[t]), hc))
			}
			for s := 0; s < hc; s++ {
				if len(streams[t][s]) != H {
					return v41StageErr(v41StageEngram, l,
						fmt.Errorf("%w: Engram full geometry stream [%d][%d] has width %d, want %d",
							ErrV41NativeUnsupported, t, s, len(streams[t][s]), H))
				}
			}
		}
	}
	for t := range seq {
		for c := 0; c < cols; c++ {
			values, err := decodeV41EngramRow(gathered[t*cols+c], dim, stage.rowBytes)
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
			key := make([]float32, H)
			var h2, k2, dotSum float32
			for i := 0; i < H; i++ {
				key[i] = v41BF16(projected[s*H+i])
				hi := hs[i]
				h2 += hi * hi
				k2 += key[i] * key[i]
				dotSum += hi * (qNorm[s*H+i] * kNorm[s*H+i]) * key[i]
			}
			dot := dotSum * rsqrtf(h2/float32(H)+eps) * rsqrtf(k2/float32(H)+eps) * rsqrtf(float32(H))
			gate := float32(1) / (1 + expf(-copysignf(sqrtf(maxf(absf(dot), 1e-6)), dot)))
			if full {
				// Each stream receives its own gated value; do NOT sum across streams.
				stream := streams[t][s]
				for i := 0; i < H; i++ {
					stream[i] += v41BF16(gate * value[i])
				}
			} else {
				for i := 0; i < H; i++ {
					acc[i] += v41BF16(gate * value[i])
				}
			}
		}
		if full {
			// streams[t][0] aliases x[t] on the full path (v41_forward.go:2029-2036),
			// so stream 0's write-back already advanced x; copy explicitly to keep the
			// synchronization self-evident and to cover a caller whose stream 0 is not
			// the same slice as x[t].
			copy(x[t], streams[t][0])
		} else {
			for i := 0; i < H; i++ {
				h[i] += acc[i]
			}
		}
	}
	return nil
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

	// Take the before-snapshot, then establish residency for each requested
	// address in that layer's own cache. V41EngramRowCache.Row only prefetches
	// forward from idx+1 and never retains idx itself (see v41_cache.go:223-249),
	// so warming must read the row AND store it: a following forward's
	// GatherV41EngramRows -> Row(rowID) then hits exactly these entries. A
	// repeated address within one layer is read once; the store makes the second
	// lookup a hit.
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
				payload, err := cache.Row(int(row))
				if err != nil {
					return zero, v41StageErr(v41StageEngram, stage.layerIDs[layer],
						fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
				}
				if err := cache.inner.Put(0, int(row), payload); err != nil {
					return zero, v41StageErr(v41StageEngram, stage.layerIDs[layer],
						fmt.Errorf("%w: %v", ErrV41NativeUnsupported, err))
				}
			}
		}
	}

	// Project the delta from the counter snapshots. Every field is additive and
	// monotone, so a concurrent caller can only shrink the delta, never inflate it.
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
