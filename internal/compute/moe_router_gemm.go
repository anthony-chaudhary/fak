package compute

// Batched multi-row router gate GEMM: the CPU reference of moe_router_gemm.cu
// (ADAPT of vllm-project/vllm csrc/fp32_router_gemm.cu at pin
// ae065918640a198a56f6f17eb45e251c3229fe41, Apache-2.0). This file is pure Go on
// purpose: it compiles and is tested in every build, while the device kernel
// ships behind the cuda build tag once the follow-on ticket wires
// moe_router_gemm.cu into the single libfakcuda translation unit. The cgo launch
// wrapper lands with that wiring; declaring the C ABI here would break every
// -tags cuda link against a libfakcuda.a that does not carry the symbols yet.
//
// The kernel replaces the per-row gate MatMul launches for chunked-prefill
// batches on 512-class expert models: one batched [rows,hidden] x [hidden,experts]
// GEMM instead of `rows` separate one-row gate projections. The one-row fused
// gate (V41RouterGateTopKBackend) stays canonical for decode; this is the
// prefill-batch path. The generic cpuref MatMul / BatchedMatMul are untouched.

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/mathx"
)

// MoeRouterGemmMaxExperts bounds the expert-class count the batched router gate
// GEMM admits: the 512-class expert models the chunked-prefill router targets.
const MoeRouterGemmMaxExperts = 512

// MoeRouterGemmMaxHidden bounds the hidden dimension the batched router gate
// GEMM admits.
const MoeRouterGemmMaxHidden = 8192

// MoeRouterGemmMaxRows bounds the row (token) count the batched router gate GEMM
// admits, matching the upstream M<=48 envelope.
const MoeRouterGemmMaxRows = 48

// ErrMoeRouterGemmBounds is the closed refusal for MoeRouterGemmFp32 inputs that
// violate the rows/hidden/experts bounds or overflow the int32 device ABI.
var ErrMoeRouterGemmBounds = errors.New("compute: moe router gemm geometry refused")

// MoeRouterGemmParityMaxULPs bounds the per-element ULP distance the device
// kernel (moe_router_gemm.cu) may drift from the Go oracle MoeRouterGemmFp32.
// The contract is a documented tolerance, not bit-identity: the two summation
// orders differ by construction (device: 128-thread K-slice chains of 40
// products + five-level warp butterfly + sequential four-warp fold; oracle:
// eight stride-8 chains of 640 products + fixed ((s0+s1)+(s2+s3))+((s4+s5)+
// (s6+s7)) fold), so exact bit matching would require removing the K
// parallelism the batched GEMM exists for. Sixty-four ULPs is magnitude-
// independent (<= ~7.6e-6 relative at any exponent), stays ~40x tighter than
// the Higham worst case for a 5120-term fp32 accumulation (~N*eps ~ 3e-4
// relative), and leaves >=4x headroom over the deterministic few-ULP deltas
// the raider-fak sm_120 panel observed on all five shapes under lease
// hwl-69a8382f581f24e5 (2026-11-05). A real defect (wrong index, dropped
// term, transposed operand) moves an element O(1) relative — thousands of
// ULPs — far outside this gate.
const MoeRouterGemmParityMaxULPs = 64

// moeRouterGemmULPKey maps a float32 to a monotone uint32 key: key order
// matches numeric order, +0/-0 share one key, and adjacent representable
// values differ by exactly one key step (one ULP).
func moeRouterGemmULPKey(f float32) uint32 {
	b := math.Float32bits(f)
	if b&0x80000000 != 0 { // negative half (incl. -0 and -Inf)
		return 0x80000000 - (b & 0x7fffffff)
	}
	return b + 0x80000000
}

// moeRouterGemmULPDistance returns the ULP distance between two float32 values
// as the absolute difference of their monotone keys. Cross-sign pairs are far
// apart by construction; infinities sit one key step beyond the largest
// finite magnitude of their sign.
func moeRouterGemmULPDistance(a, b float32) int {
	ka, kb := moeRouterGemmULPKey(a), moeRouterGemmULPKey(b)
	if ka > kb {
		return int(int64(ka) - int64(kb))
	}
	return int(int64(kb) - int64(ka))
}

// MoeRouterGemmWithinParity applies the documented device-vs-oracle parity
// gate element-wise: every pair must be both NaN, or finite with ULP distance
// <= MoeRouterGemmParityMaxULPs (infinities therefore must match exactly).
// It returns (true, -1) when compliant; (false, i) naming the first violating
// element i; and (false, -1) for a length mismatch. The retained panel harness
// mirrors this constant and formula on the host side.
func MoeRouterGemmWithinParity(got, want []float32) (bool, int) {
	if len(got) != len(want) {
		return false, -1
	}
	for i := range got {
		g, w := got[i], want[i]
		if math.IsNaN(float64(g)) || math.IsNaN(float64(w)) {
			if !(math.IsNaN(float64(g)) && math.IsNaN(float64(w))) {
				return false, i
			}
			continue
		}
		if moeRouterGemmULPDistance(g, w) > MoeRouterGemmParityMaxULPs {
			return false, i
		}
	}
	return true, -1
}

// MoeRouterGemmFp32 computes the batched multi-row router gate GEMM:
//
//	Y[t, o] = sum_i X[t, i] * W[o, i]   for t in [0, rows), o in [0, experts)
//
// where weight is row-major F32 [experts, hidden] and x is row-major F32
// [rows, hidden]. Each (row, expert) dot uses the same mathx.FDot eight-accumulator
// tree as the one-row gate path (MatMul), so the result is bit-identical to
// per-row MatMul for every row — the accumulation order the device kernel must
// match. It refuses out-of-bounds geometry with ErrMoeRouterGemmBounds.
func MoeRouterGemmFp32(weight, x []float32, rows, hidden, experts int) ([]float32, error) {
	if rows <= 0 || rows > MoeRouterGemmMaxRows {
		return nil, fmt.Errorf("%w: rows=%d", ErrMoeRouterGemmBounds, rows)
	}
	if experts <= 0 || experts > MoeRouterGemmMaxExperts {
		return nil, fmt.Errorf("%w: experts=%d", ErrMoeRouterGemmBounds, experts)
	}
	if hidden <= 0 || hidden > MoeRouterGemmMaxHidden {
		return nil, fmt.Errorf("%w: hidden=%d", ErrMoeRouterGemmBounds, hidden)
	}
	// Overflow guard: the int32 device ABI addresses elements, not bytes.
	if int64(rows)*int64(hidden) > 1<<31-1 || int64(experts)*int64(hidden) > 1<<31-1 {
		return nil, fmt.Errorf("%w: element count overflows the int32 ABI", ErrMoeRouterGemmBounds)
	}
	if len(weight) != experts*hidden || len(x) != rows*hidden {
		return nil, fmt.Errorf("%w: weight/x length %d/%d want %d/%d", ErrMoeRouterGemmBounds, len(weight), len(x), experts*hidden, rows*hidden)
	}

	out := make([]float32, rows*experts)
	for t := 0; t < rows; t++ {
		xrow := x[t*hidden : (t+1)*hidden]
		for o := 0; o < experts; o++ {
			wrow := weight[o*hidden : (o+1)*hidden]
			out[t*experts+o] = mathx.FDot(wrow, xrow)
		}
	}
	return out, nil
}
