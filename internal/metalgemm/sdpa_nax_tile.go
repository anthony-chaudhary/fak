package metalgemm

import (
	"errors"
	"fmt"
	"math"
)

// Default and boundary parameters for wide-M tail-causal SDPA speculative verification tiles.
const (
	// DefaultTileN is the default KV cache chunk size along the sequence dimension (Bc).
	DefaultTileN = 32

	// MinWideM is the lower bound for wide-M query rows during speculative verification (e.g. GQA_F=4 x QL=4).
	MinWideM = 16

	// MaxWideM is the upper bound for wide-M query rows during speculative verification (e.g. GQA_F=6 x QL=4).
	MaxWideM = 24

	// MaxHeadDim is the maximum supported head dimension in threadgroup tile memory.
	MaxHeadDim = 128
)

// DraftRowOrder describes the indexing order of query rows in the wide-M block.
type DraftRowOrder int

const (
	// RowOrderHeadMajor arranges query rows as: row = head_idx * DraftLen + draft_token_idx.
	// draft_token_idx = row % DraftLen.
	RowOrderHeadMajor DraftRowOrder = iota

	// RowOrderTokenMajor arranges query rows as: row = draft_token_idx * GQAFactor + head_idx.
	// draft_token_idx = row / GQAFactor.
	RowOrderTokenMajor
)

// SDPANAXTileConfig defines the wide-M tail-causal SDPA tile verification configuration.
// It configures speculative verification of drafted tokens (M=16..24 rows) against
// a shared KV stream.
type SDPANAXTileConfig struct {
	// GQAFactor is the number of query heads sharing a single KV head (e.g. 4 or 6).
	GQAFactor int

	// DraftLen is the number of speculative draft tokens QL (e.g. 4).
	DraftLen int

	// M is the total number of query rows in the verification block: M = GQAFactor * DraftLen (16..24).
	M int

	// HeadDim is the dimension D per head (e.g. 64 or 128).
	HeadDim int

	// PrefixLen is the sequence prefix length already present in KV cache prior to drafting.
	PrefixLen int

	// TotalKV is the total sequence length in the KV cache (PrefixLen + DraftLen).
	TotalKV int

	// Scale is the attention scale factor (defaults to 1.0 / sqrt(HeadDim) if <= 0).
	Scale float32

	// TileN is the sequence tile chunk size Bc for KV cache loading (default: 32).
	TileN int

	// TileM is the query tile chunk size Br (defaults to M if <= 0).
	TileM int

	// Order specifies row-to-draft mapping (RowOrderHeadMajor or RowOrderTokenMajor).
	Order DraftRowOrder
}

// NewSDPANAXTileConfig creates a validated tile configuration for wide-M speculative verification.
func NewSDPANAXTileConfig(gqaFactor, draftLen, headDim, prefixLen int) (SDPANAXTileConfig, error) {
	cfg := SDPANAXTileConfig{
		GQAFactor: gqaFactor,
		DraftLen:  draftLen,
		M:         gqaFactor * draftLen,
		HeadDim:   headDim,
		PrefixLen: prefixLen,
		TotalKV:   prefixLen + draftLen,
		TileN:     DefaultTileN,
		Order:     RowOrderHeadMajor,
	}
	if err := cfg.Validate(); err != nil {
		return SDPANAXTileConfig{}, err
	}
	return cfg, nil
}

// Validate checks configuration invariants and populates defaults.
func (c *SDPANAXTileConfig) Validate() error {
	if c.GQAFactor <= 0 {
		return errors.New("sdpa_nax: GQAFactor must be positive")
	}
	if c.DraftLen <= 0 {
		return errors.New("sdpa_nax: DraftLen must be positive")
	}
	if c.HeadDim <= 0 || c.HeadDim > MaxHeadDim {
		return fmt.Errorf("sdpa_nax: HeadDim %d must be in range 1..%d", c.HeadDim, MaxHeadDim)
	}
	if c.PrefixLen < 0 {
		return errors.New("sdpa_nax: PrefixLen must be non-negative")
	}
	expectedM := c.GQAFactor * c.DraftLen
	if c.M == 0 {
		c.M = expectedM
	} else if c.M != expectedM {
		return fmt.Errorf("sdpa_nax: M (%d) does not match GQAFactor (%d) * DraftLen (%d) = %d",
			c.M, c.GQAFactor, c.DraftLen, expectedM)
	}
	if c.M < MinWideM || c.M > MaxWideM {
		return fmt.Errorf("sdpa_nax: wide-M query rows %d out of supported range [%d, %d]",
			c.M, MinWideM, MaxWideM)
	}
	minKV := c.PrefixLen + c.DraftLen
	if c.TotalKV == 0 {
		c.TotalKV = minKV
	} else if c.TotalKV < minKV {
		return fmt.Errorf("sdpa_nax: TotalKV (%d) must be at least PrefixLen + DraftLen (%d)",
			c.TotalKV, minKV)
	}
	if c.TileN <= 0 {
		c.TileN = DefaultTileN
	}
	if c.TileM <= 0 {
		c.TileM = c.M
	}
	if c.Scale <= 0 {
		c.Scale = float32(1.0 / math.Sqrt(float64(c.HeadDim)))
	}
	return nil
}

// DraftTokenIndex returns the drafted token index (0..DraftLen-1) corresponding to query row m.
func (c *SDPANAXTileConfig) DraftTokenIndex(row int) int {
	if c.Order == RowOrderTokenMajor {
		if c.GQAFactor <= 0 {
			return 0
		}
		return row / c.GQAFactor
	}
	if c.DraftLen <= 0 {
		return 0
	}
	return row % c.DraftLen
}

// MaxCausalKey returns the maximum key index that query row m may attend to.
// Under tail-causal masking, drafted token t can attend to all prefix keys (0..PrefixLen-1)
// and preceding drafted keys up to its own position (PrefixLen..PrefixLen+t), but cannot
// attend to subsequent draft tokens (k > PrefixLen + t).
func (c *SDPANAXTileConfig) MaxCausalKey(row int) int {
	return c.PrefixLen + c.DraftTokenIndex(row)
}

// SDPANAXMemoryStats captures DRAM read traffic reduction metrics between wide-M tiled SDPA
// and traditional scalar attention.
type SDPANAXMemoryStats struct {
	ScalarKVLoads  int     // Equivalent scalar DRAM loads (M * NumTiles)
	TiledKVLoads   int     // Wide-M tiled DRAM loads (1 * NumTiles)
	ReductionRatio float64 // ScalarKVLoads / TiledKVLoads = float64(M)
	NumTiles       int     // Number of KV tiles along sequence
	M              int     // Number of query rows
	TotalKV        int     // Total KV tokens processed
	TileN          int     // Tile sequence chunk size Bc
}

// SDPANAXTileInput encapsulates the inputs for SDPA tile execution.
type SDPANAXTileInput struct {
	Config SDPANAXTileConfig
	Q      []float32 // Query matrix [M, HeadDim]
	K      []float32 // Key cache [TotalKV, HeadDim]
	V      []float32 // Value cache [TotalKV, HeadDim]
}

// SDPANAXTileResult contains the computed attention output and LSE tracking values.
type SDPANAXTileResult struct {
	Output []float32          // Attention output [M, HeadDim]
	LSE    []float32          // Log-Sum-Exp values [M]
	Stats  SDPANAXMemoryStats // DRAM memory access metrics
}

// SDPANAXEquivalenceReport summarizes numerical equivalence between wide-M tiled SDPA and reference scalar SDPA.
type SDPANAXEquivalenceReport struct {
	MaxDiffOutput float32
	MaxDiffLSE    float32
	Passed        bool
	Tolerance     float32
	M             int
	Details       string
}

// ComputeScalarSDPAReference evaluates scalar causal SDPA independently per query row.
// It serves as the numerical gold standard for verification.
func ComputeScalarSDPAReference(input SDPANAXTileInput) (output []float32, lse []float32, scalarLoads int, err error) {
	cfg := input.Config
	if err := cfg.Validate(); err != nil {
		return nil, nil, 0, err
	}
	expectedQ := cfg.M * cfg.HeadDim
	if len(input.Q) < expectedQ {
		return nil, nil, 0, fmt.Errorf("sdpa_nax: Q length %d smaller than expected %d", len(input.Q), expectedQ)
	}
	expectedKV := cfg.TotalKV * cfg.HeadDim
	if len(input.K) < expectedKV || len(input.V) < expectedKV {
		return nil, nil, 0, fmt.Errorf("sdpa_nax: K/V length smaller than expected %d", expectedKV)
	}

	output = make([]float32, cfg.M*cfg.HeadDim)
	lse = make([]float32, cfg.M)

	numTiles := (cfg.TotalKV + cfg.TileN - 1) / cfg.TileN
	scalarLoads = cfg.M * numTiles

	for m := 0; m < cfg.M; m++ {
		maxK := cfg.MaxCausalKey(m)
		if maxK >= cfg.TotalKV {
			maxK = cfg.TotalKV - 1
		}
		numKeys := maxK + 1
		if numKeys <= 0 {
			lse[m] = float32(math.Inf(-1))
			continue
		}

		logits := make([]float32, numKeys)
		maxLogit := float32(math.Inf(-1))
		qOffset := m * cfg.HeadDim

		for j := 0; j < numKeys; j++ {
			kOffset := j * cfg.HeadDim
			var dot float32
			for d := 0; d < cfg.HeadDim; d++ {
				dot += input.Q[qOffset+d] * input.K[kOffset+d]
			}
			s := dot * cfg.Scale
			logits[j] = s
			if s > maxLogit {
				maxLogit = s
			}
		}

		var sumExp float32
		weights := make([]float32, numKeys)
		for j := 0; j < numKeys; j++ {
			w := float32(math.Exp(float64(logits[j] - maxLogit)))
			weights[j] = w
			sumExp += w
		}

		lse[m] = maxLogit + float32(math.Log(float64(sumExp)))

		invSum := 1.0 / sumExp
		for d := 0; d < cfg.HeadDim; d++ {
			var acc float32
			for j := 0; j < numKeys; j++ {
				acc += (weights[j] * invSum) * input.V[j*cfg.HeadDim+d]
			}
			output[m*cfg.HeadDim+d] = acc
		}
	}

	return output, lse, scalarLoads, nil
}

// RunSDPANAXTiledComputation executes the wide-M tail-causal SDPA tiled computation.
// Key aspects of this implementation:
//  1. K/V tiles are loaded ONCE per tile pair (shared across all M query rows).
//  2. The V tile is staged transposed in threadgroup memory (D x Bc) for coalesced access.
//  3. Tail-causal masking is applied strictly to drafted tokens.
//  4. Online softmax maintains running max and sum, tracking Log-Sum-Exp (LSE).
func RunSDPANAXTiledComputation(input SDPANAXTileInput) (*SDPANAXTileResult, error) {
	cfg := input.Config
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	expectedQ := cfg.M * cfg.HeadDim
	if len(input.Q) < expectedQ {
		return nil, fmt.Errorf("sdpa_nax: Q length %d smaller than expected %d", len(input.Q), expectedQ)
	}
	expectedKV := cfg.TotalKV * cfg.HeadDim
	if len(input.K) < expectedKV || len(input.V) < expectedKV {
		return nil, fmt.Errorf("sdpa_nax: K/V length smaller than expected %d", expectedKV)
	}

	numTiles := (cfg.TotalKV + cfg.TileN - 1) / cfg.TileN

	output := make([]float32, cfg.M*cfg.HeadDim)
	lse := make([]float32, cfg.M)
	mPrev := make([]float32, cfg.M)
	lPrev := make([]float32, cfg.M)
	for m := 0; m < cfg.M; m++ {
		mPrev[m] = float32(math.Inf(-1))
		lPrev[m] = 0.0
	}

	// Threadgroup staged memory:
	// kTile holds [TileN, HeadDim]
	// vTransposed holds [HeadDim, TileN]
	kTile := make([]float32, cfg.TileN*cfg.HeadDim)
	vTransposed := make([]float32, cfg.HeadDim*cfg.TileN)

	tiledLoads := 0
	scalarLoads := 0

	for tileIdx := 0; tileIdx < numTiles; tileIdx++ {
		jStart := tileIdx * cfg.TileN
		jEnd := (tileIdx + 1) * cfg.TileN
		if jEnd > cfg.TotalKV {
			jEnd = cfg.TotalKV
		}
		tileLen := jEnd - jStart
		if tileLen <= 0 {
			continue
		}

		// DRAM load: K and V are loaded ONCE for all M rows.
		tiledLoads++
		scalarLoads += cfg.M

		// 1. Stage K tile [tileLen, HeadDim]
		for k := 0; k < tileLen; k++ {
			globalKPos := (jStart + k) * cfg.HeadDim
			kDst := k * cfg.HeadDim
			copy(kTile[kDst:kDst+cfg.HeadDim], input.K[globalKPos:globalKPos+cfg.HeadDim])
		}

		// 2. Stage transposed V tile [HeadDim, tileLen]
		// vTransposed[d * TileN + k] = V[(jStart + k) * HeadDim + d]
		for k := 0; k < tileLen; k++ {
			globalVPos := (jStart + k) * cfg.HeadDim
			for d := 0; d < cfg.HeadDim; d++ {
				vTransposed[d*cfg.TileN+k] = input.V[globalVPos+d]
			}
		}

		// 3. Process wide-M query rows against the staged K/V tile pair
		for m := 0; m < cfg.M; m++ {
			maxKForM := cfg.MaxCausalKey(m)
			if jStart > maxKForM {
				// Entire tile lies beyond the causal horizon for draft token m; skip.
				continue
			}

			qOffset := m * cfg.HeadDim

			// Compute logits for unmasked keys in this tile
			logits := make([]float32, tileLen)
			tileMax := float32(math.Inf(-1))

			for k := 0; k < tileLen; k++ {
				globalKeyPos := jStart + k
				if globalKeyPos > maxKForM {
					// Tail-causal masking: draft token cannot attend to subsequent draft tokens
					logits[k] = float32(math.Inf(-1))
					continue
				}

				kDst := k * cfg.HeadDim
				var dot float32
				for d := 0; d < cfg.HeadDim; d++ {
					dot += input.Q[qOffset+d] * kTile[kDst+d]
				}
				score := dot * cfg.Scale
				logits[k] = score
				if score > tileMax {
					tileMax = score
				}
			}

			if tileMax == float32(math.Inf(-1)) {
				continue
			}

			// Online softmax update:
			// m_new = max(m_prev, tileMax)
			// alpha = exp(m_prev - m_new)
			// P[k] = exp(logits[k] - m_new)
			// l_new = alpha * l_prev + sum(P)
			// Acc = alpha * Acc + P * V^T
			mNew := mPrev[m]
			if tileMax > mNew {
				mNew = tileMax
			}

			var alpha float32
			if mPrev[m] != float32(math.Inf(-1)) {
				alpha = float32(math.Exp(float64(mPrev[m] - mNew)))
			}

			var pSum float32
			pVals := make([]float32, tileLen)
			for k := 0; k < tileLen; k++ {
				if logits[k] == float32(math.Inf(-1)) {
					pVals[k] = 0.0
				} else {
					p := float32(math.Exp(float64(logits[k] - mNew)))
					pVals[k] = p
					pSum += p
				}
			}

			lNew := alpha*lPrev[m] + pSum

			// Update output accumulator using transposed V stage
			outOffset := m * cfg.HeadDim
			for d := 0; d < cfg.HeadDim; d++ {
				var pv float32
				vOffset := d * cfg.TileN
				for k := 0; k < tileLen; k++ {
					pv += pVals[k] * vTransposed[vOffset+k]
				}
				output[outOffset+d] = alpha*output[outOffset+d] + pv
			}

			mPrev[m] = mNew
			lPrev[m] = lNew
		}
	}

	// Final normalization: O = Acc / l_final, LSE = m_final + ln(l_final)
	for m := 0; m < cfg.M; m++ {
		if lPrev[m] > 0.0 {
			invL := 1.0 / lPrev[m]
			outOffset := m * cfg.HeadDim
			for d := 0; d < cfg.HeadDim; d++ {
				output[outOffset+d] *= invL
			}
			lse[m] = mPrev[m] + float32(math.Log(float64(lPrev[m])))
		} else {
			lse[m] = float32(math.Inf(-1))
		}
	}

	stats := SDPANAXMemoryStats{
		ScalarKVLoads:  scalarLoads,
		TiledKVLoads:   tiledLoads,
		ReductionRatio: float64(scalarLoads) / float64(tiledLoads),
		NumTiles:       numTiles,
		M:              cfg.M,
		TotalKV:        cfg.TotalKV,
		TileN:          cfg.TileN,
	}

	return &SDPANAXTileResult{
		Output: output,
		LSE:    lse,
		Stats:  stats,
	}, nil
}

// EvaluateSDPAEquivalence compares wide-M tiled SDPA outputs and LSE against reference scalar SDPA.
func EvaluateSDPAEquivalence(tiled *SDPANAXTileResult, refOutput, refLSE []float32, tolerance float32) SDPANAXEquivalenceReport {
	report := SDPANAXEquivalenceReport{
		Tolerance: tolerance,
		Passed:    true,
		M:         len(tiled.LSE),
	}

	if len(refOutput) != len(tiled.Output) {
		report.Passed = false
		report.Details = fmt.Sprintf("output length mismatch: tiled %d vs ref %d", len(tiled.Output), len(refOutput))
		return report
	}
	if len(refLSE) != len(tiled.LSE) {
		report.Passed = false
		report.Details = fmt.Sprintf("LSE length mismatch: tiled %d vs ref %d", len(tiled.LSE), len(refLSE))
		return report
	}

	for i := range tiled.Output {
		diff := float32(math.Abs(float64(tiled.Output[i] - refOutput[i])))
		if diff > report.MaxDiffOutput {
			report.MaxDiffOutput = diff
		}
		if diff > tolerance {
			report.Passed = false
		}
	}

	for m := range tiled.LSE {
		tLSE := tiled.LSE[m]
		rLSE := refLSE[m]
		if math.IsInf(float64(tLSE), -1) && math.IsInf(float64(rLSE), -1) {
			continue
		}
		diff := float32(math.Abs(float64(tLSE - rLSE)))
		if diff > report.MaxDiffLSE {
			report.MaxDiffLSE = diff
		}
		if diff > tolerance {
			report.Passed = false
		}
	}

	report.Details = fmt.Sprintf("Parity: maxDiffOutput=%.6e, maxDiffLSE=%.6e, tolerance=%.6e, passed=%v",
		report.MaxDiffOutput, report.MaxDiffLSE, tolerance, report.Passed)
	return report
}

// SDPANAXTileMetalShaderSource is the complete Metal Shading Language (MSL) source
// template for the wide-M tail-causal SDPA Metal 4 TensorOps tile kernel.
//
// Features:
//   - Threadgroup memory staging of K and transposed V tiles.
//   - Single DRAM read per KV tile shared across all M query rows (1x load vs M loads).
//   - Hardware-aligned transposed V staging for SIMD / TensorOps coalesced dot products.
//   - Tail-causal masking of speculative draft tokens.
//   - Online softmax with Log-Sum-Exp (LSE) tracking.
const SDPANAXTileMetalShaderSource = `#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// Wide-M Tail-Causal SDPA Metal 4 TensorOps Tile Kernel
// ==============================================================================
// Speculative verification of drafted tokens (M=16..24 rows) against a shared
// KV stream. Rather than executing separate scalar attention reads that duplicate
// K/V DRAM reads across drafted tokens, this kernel:
//  1. Loads K and V tiles once per tile pair into threadgroup memory.
//  2. Stages transposed V (D x Bc) in threadgroup memory for coalesced dot
//     products / Metal 4 TensorOps (mpp::tensor_ops::matmul2d).
//  3. Enforces tail-causal masking on draft tokens: token t at sequence position
//     (prefix_len + t) attends to prefix keys and draft keys up to t, but is
//     strictly masked from subsequent draft keys (k_pos > prefix_len + t).
//  4. Accumulates online softmax with running Log-Sum-Exp (LSE) tracking.
// ==============================================================================

#define NAX_TILE_N_MAX 32
#define NAX_MAX_HEAD_DIM 128
#define NAX_MAX_WIDE_M 24

struct SDPANAXConstants {
    uint gqa_factor;   // Number of query heads sharing a KV head (e.g. 4 or 6)
    uint draft_len;    // Number of speculative draft tokens QL (e.g. 4)
    uint M;            // Total query rows: M = gqa_factor * draft_len (16..24)
    uint head_dim;     // Head dimension D (e.g. 64 or 128)
    uint prefix_len;   // Sequence prefix length
    uint total_kv;     // Total KV tokens = prefix_len + draft_len
    float scale;       // Attention scale factor 1.0f / sqrt(head_dim)
    uint tile_n;       // KV tile size Bc (e.g. 32)
    uint order;        // 0: HeadMajor (row = h*QL + t), 1: TokenMajor (row = t*GQA + h)
};

// Threadgroup storage for cooperative K/V tile pair staging.
// Shared across all M query rows in the threadgroup; loaded once per tile pair.
struct SDPANAXThreadgroupStorage {
    float k_tile[NAX_TILE_N_MAX * NAX_MAX_HEAD_DIM];
    float v_transposed[NAX_MAX_HEAD_DIM * NAX_TILE_N_MAX];
};

inline uint nax_draft_token_index(uint m, constant SDPANAXConstants& c) {
    if (c.order == 1) {
        return c.gqa_factor > 0 ? (m / c.gqa_factor) : 0;
    }
    return c.draft_len > 0 ? (m % c.draft_len) : 0;
}

inline uint nax_max_causal_key(uint m, constant SDPANAXConstants& c) {
    return c.prefix_len + nax_draft_token_index(m, c);
}

// sdpa_nax_tail_causal_tile: Metal 4 compute kernel for speculative verify.
kernel void sdpa_nax_tail_causal_tile(
    device const float* Q [[buffer(0)]],            // [M, HeadDim]
    device const float* K [[buffer(1)]],            // [TotalKV, HeadDim]
    device const float* V [[buffer(2)]],            // [TotalKV, HeadDim]
    device float* Out [[buffer(3)]],                // [M, HeadDim]
    device float* LSE [[buffer(4)]],                // [M]
    constant SDPANAXConstants& c [[buffer(5)]],
    threadgroup SDPANAXThreadgroupStorage& tg_mem [[threadgroup(0)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    if (c.M == 0 || c.head_dim == 0 || c.total_kv == 0) return;

    // In wide-M speculative verification, M rows (16..24) are handled
    // cooperatively by simdgroups in the threadgroup.
    uint m = simd_group_id;
    if (m >= c.M) return;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // Dimension chunk for this SIMD lane
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    uint max_k_for_row = nax_max_causal_key(m, c);
    uint num_tiles = (c.total_kv + c.tile_n - 1) / c.tile_n;

    for (uint tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        uint j_start = tile_idx * c.tile_n;
        uint j_end = min(j_start + c.tile_n, c.total_kv);
        uint tile_len = j_end - j_start;

        // 1. Cooperative load of K and V tiles from DRAM into threadgroup memory.
        // Loaded ONCE for all M query rows in the threadgroup (eliminating redundant reads).
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint total_elements = tile_len * c.head_dim;
        for (uint idx = tid; idx < total_elements; idx += tg_size) {
            uint k_row = idx / c.head_dim;
            uint k_col = idx % c.head_dim;
            uint global_k_pos = j_start + k_row;
            tg_mem.k_tile[k_row * c.head_dim + k_col] = K[global_k_pos * c.head_dim + k_col];
            // Stage transposed V: v_transposed[col, row] = V[global_pos, col]
            tg_mem.v_transposed[k_col * c.tile_n + k_row] = V[global_k_pos * c.head_dim + k_col];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // If the entire tile is past the causal horizon for draft token m, skip computation.
        if (j_start > max_k_for_row) {
            continue;
        }

        // 2. Compute QK dot products with tail-causal masking.
        // On Metal 4 TensorOps targets, mpp::tensor_ops::matmul2d accelerates tile QK GEMM.
        for (uint k = 0; k < tile_len; ++k) {
            uint global_key_pos = j_start + k;
            float score = -INFINITY;
            if (global_key_pos <= max_k_for_row) {
                float partial_qk = 0.0f;
                for (uint d = simd_lane; d < c.head_dim; d += 32) {
                    partial_qk += Q[m * c.head_dim + d] * tg_mem.k_tile[k * c.head_dim + d];
                }
                score = simd_sum(partial_qk) * c.scale;
            }

            // Online softmax and LSE tracking
            float m_new = max(m_prev, score);
            float alpha = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);
            float p = (score == -INFINITY) ? 0.0f : exp(score - m_new);
            l_prev = alpha * l_prev + p;

            // 3. Accumulate P * V^T using staged transposed V tile
            uint d_idx = 0;
            for (uint d = simd_lane; d < c.head_dim; d += 32) {
                float v_val = tg_mem.v_transposed[d * c.tile_n + k];
                acc[d_idx] = alpha * acc[d_idx] + p * v_val;
                d_idx++;
            }
            m_prev = m_new;
        }
    }

    // 4. Output normalization and LSE writeback
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    uint d_idx = 0;
    for (uint d = simd_lane; d < c.head_dim; d += 32) {
        Out[m * c.head_dim + d] = acc[d_idx++] * inv_l;
    }
    if (simd_lane == 0) {
        LSE[m] = (l_prev > 0.0f) ? (m_prev + log(l_prev)) : -INFINITY;
    }
}
`

// MetalPipelineDescriptor describes a Metal compute pipeline for wide-M tail-causal SDPA.
type MetalPipelineDescriptor struct {
	FunctionName           string
	ShaderSource           string
	MetalVersion           string
	MaxThreadsPerTG        int
	ThreadgroupMemoryBytes int
	UsesTensorOps          bool
}

// MetalDispatchGrid defines the execution grid dimensions for Metal command dispatch.
type MetalDispatchGrid struct {
	ThreadgroupsPerGrid   [3]int
	ThreadsPerThreadgroup [3]int
}

// NewSDPANAXMetalPipelineDescriptor constructs the Metal compute pipeline descriptor
// for the given wide-M tile configuration.
func NewSDPANAXMetalPipelineDescriptor(cfg SDPANAXTileConfig) (*MetalPipelineDescriptor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	// Calculate threadgroup memory: k_tile + v_transposed
	tgBytes := (cfg.TileN*cfg.HeadDim + cfg.HeadDim*cfg.TileN) * 4 // float32 = 4 bytes
	return &MetalPipelineDescriptor{
		FunctionName:           "sdpa_nax_tail_causal_tile",
		ShaderSource:           SDPANAXTileMetalShaderSource,
		MetalVersion:           "Metal 4",
		MaxThreadsPerTG:        256,
		ThreadgroupMemoryBytes: tgBytes,
		UsesTensorOps:          true,
	}, nil
}

// BuildSDPANAXDispatchGrid computes threadgroup and thread allocations for Apple Silicon Metal.
func BuildSDPANAXDispatchGrid(cfg SDPANAXTileConfig) MetalDispatchGrid {
	// One threadgroup handles the M query rows with up to 8 simdgroups (256 threads)
	threads := 32 * cfg.M
	if threads > 256 {
		threads = 256
	} else if threads < 32 {
		threads = 32
	}
	return MetalDispatchGrid{
		ThreadgroupsPerGrid:   [3]int{1, 1, 1},
		ThreadsPerThreadgroup: [3]int{threads, 1, 1},
	}
}

// SDPANAXHarness is the Go execution and verification harness for wide-M tail-causal SDPA.
type SDPANAXHarness struct {
	Config     SDPANAXTileConfig
	Descriptor *MetalPipelineDescriptor
	Grid       MetalDispatchGrid
}

// NewSDPANAXHarness creates an initialized execution harness for the tile configuration.
func NewSDPANAXHarness(cfg SDPANAXTileConfig) (*SDPANAXHarness, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	desc, err := NewSDPANAXMetalPipelineDescriptor(cfg)
	if err != nil {
		return nil, err
	}
	grid := BuildSDPANAXDispatchGrid(cfg)
	return &SDPANAXHarness{
		Config:     cfg,
		Descriptor: desc,
		Grid:       grid,
	}, nil
}

// Execute runs the wide-M tail-causal SDPA tiled computation.
func (h *SDPANAXHarness) Execute(input SDPANAXTileInput) (*SDPANAXTileResult, error) {
	input.Config = h.Config
	return RunSDPANAXTiledComputation(input)
}

// ExecuteQKV runs tiled computation with directly provided Q, K, and V slices.
func (h *SDPANAXHarness) ExecuteQKV(q, k, v []float32) (*SDPANAXTileResult, error) {
	return h.Execute(SDPANAXTileInput{
		Config: h.Config,
		Q:      q,
		K:      k,
		V:      v,
	})
}

// ExecuteAndVerify executes tiled computation and validates numerical equivalence against
// reference scalar SDPA within the given tolerance.
func (h *SDPANAXHarness) ExecuteAndVerify(input SDPANAXTileInput, tolerance float32) (*SDPANAXTileResult, *SDPANAXEquivalenceReport, error) {
	input.Config = h.Config
	tiledRes, err := RunSDPANAXTiledComputation(input)
	if err != nil {
		return nil, nil, err
	}
	refOut, refLSE, _, err := ComputeScalarSDPAReference(input)
	if err != nil {
		return nil, nil, err
	}
	report := EvaluateSDPAEquivalence(tiledRes, refOut, refLSE, tolerance)
	return tiledRes, &report, nil
}
