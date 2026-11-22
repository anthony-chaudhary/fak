package compute

import (
	"errors"
	"fmt"
	"math"
	"os"
)

// hip_attention.go — HIP / ROCm attention execution with pre-attention f16 KV contiguization (#11746).
//
// Provides HIP/ROCm kernel adapters and dispatch wrappers that apply F16KVContiguizationPass
// on AMD APUs (gfx1151 / Strix Halo) to prevent 16-channel LPDDR5X channel camping.

// HIPF16KVContiguizationPass wraps F16KVContiguizationPass for ROCm / HIP compute graphs.
type HIPF16KVContiguizationPass struct {
	base *F16KVContiguizationPass
}

// NewHIPF16KVContiguizationPass creates a HIP-specific contiguization pass wrapper.
func NewHIPF16KVContiguizationPass(arch string, nPos, nKV, headDim int, precision KVPrecision) *HIPF16KVContiguizationPass {
	return &HIPF16KVContiguizationPass{
		base: NewF16KVContiguizationPass(arch, nPos, nKV, headDim, precision),
	}
}

// ShouldExecute delegates to ShouldContiguizeF16KV.
func (p *HIPF16KVContiguizationPass) ShouldExecute() bool {
	return p.base.ShouldExecute()
}

// ScratchBytes reports memory required in bytes.
func (p *HIPF16KVContiguizationPass) ScratchBytes() int64 {
	return p.base.ScratchBytes()
}

// Execute performs the contiguization pass on strided f16 inputs.
func (p *HIPF16KVContiguizationPass) Execute(kStrided, vStrided []uint16) ([]uint16, []uint16, error) {
	return p.base.Execute(kStrided, vStrided)
}

// ShouldHIPContiguizeF16KV is a convenience helper for ROCm/HIP pipelines.
func ShouldHIPContiguizeF16KV(arch string, nPos int, precision KVPrecision) bool {
	return ShouldContiguizeF16KV(arch, nPos, precision)
}

// NOTE: HIPContiguizeF16KVCache (a zero-caller pass-through) and
// ExecuteHIPAttentionWithContiguization (a per-call whole-KV f32 copy with no
// production caller) were retired here (fak#13875). The library math they wrapped
// is still held directly by ContiguizeF32KVCache / ComputeContiguizedAttention /
// ComputeStridedAttention and radv_attention_test.go.

// DiagnoseHIPChannelCamping simulates and diagnoses whether memory accesses will suffer
// from LPDDR5X channel camping on the target architecture.
func DiagnoseHIPChannelCamping(
	arch string,
	nPos, nKV, headDim int,
	precision KVPrecision,
) (isCampingRisk bool, diagnosis string, report ChannelEntropyReport) {
	if !isStrixHaloArch(arch) {
		report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
		return false, fmt.Sprintf("Architecture %q does not use 16-channel LPDDR5X UMA interleaving; no channel camping risk", arch), report
	}

	if precision != KVPrecisionF32 {
		report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
		return false, fmt.Sprintf("Precision %s is quantized; channel camping pass targets unquantized f16 KV caches", precision), report
	}

	if nPos < ContiguizationMinContext {
		report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
		return false, fmt.Sprintf("Context length %d < threshold %d; cache fits within L2/MALL without sustained channel camping", nPos, ContiguizationMinContext), report
	}

	// At >= 32k context on gfx1151 with unquantized f16: channel camping risk detected
	report = SimulateChannelDistribution(nPos, nKV, headDim, false, DefaultInterleaveBytes)
	diagnosis = fmt.Sprintf(
		"LPDDR5X channel camping detected on %s: strided access camps on %d channels with entropy %.4f (< 0.25); pre-attention contiguization pass required",
		arch, report.ActiveChannels, report.Entropy,
	)
	return true, diagnosis, report
}

// NOTE: HIPFlashAttnDequantScratchpad (a zero-caller pass-through wrapper) and
// ExecuteHIPAttentionWithDequantOnce (an unwired per-call scratchpad allocator,
// the same latent per-call weight path class as fak#13875's dequant-once
// entrypoints) were retired here. The underlying single-pass dequantization is
// still held directly by FlashAttnDequantScratchpad and the radv/vulkan witnesses.

// WMMA attention route for gfx1151.
//
// The device kernel in rocm_kernels.hip is a direct port of gufo-org/gufo
// src/models/qwen/hip/kernels/attention_wmma.hip at commit
// d0f7c3a023a11af81161264eeb24e9a97aa7498a (MIT License, Copyright (c) 2026
// gufo contributors; the full notice sits beside the kernel). This file holds
// the portable half: the selection contract the device guard implements and a
// CPU reference that walks the kernel's tiling, so the port can be checked on
// a host with no AMD device.
//
// The route is off unless the HIP object was built with HIPWMMABuildMacro and
// the process sets HIPWMMAOptInEnv=1. It has not run on physical gfx1151
// silicon, so it must not become a default before that parity is witnessed.

const (
	// The only model geometry the kernel is compiled for.
	HIPWMMAQueryHeads = 24
	HIPWMMAKVHeads    = 4
	HIPWMMAHeadDim    = 256

	// Tiling: each device block covers HIPWMMAQueryRows queries of
	// HIPWMMAHeadsPerBlock heads against HIPWMMAKeyTile keys per step.
	HIPWMMAQueryRows     = 32
	HIPWMMAKeyTile       = 16
	HIPWMMAHeadsPerBlock = 2

	HIPWMMAArch       = "gfx1151"
	HIPWMMAOptInEnv   = "FAK_ROCM_WMMA_ATTENTION"
	HIPWMMABuildMacro = "FAK_ROCM_WMMA_GFX1151"
)

// hipWMMAScale is 1/sqrt(HIPWMMAHeadDim); the kernel folds it into Q as a
// constant, so a call carrying any other scale cannot use the route.
const hipWMMAScale = float32(1.0 / 16.0)

// HIPAttentionRoute names the kernel an attention call is served by.
type HIPAttentionRoute string

const (
	HIPAttentionRouteGeneric HIPAttentionRoute = "generic"
	HIPAttentionRouteWMMA    HIPAttentionRoute = "wmma"
)

// HIPAttentionWMMACall is what the device entry point knows about a call when
// it picks a kernel.
type HIPAttentionWMMACall struct {
	Arch       string
	QueryHeads int
	KVHeads    int
	HeadDim    int
	GroupSize  int
	Positions  int
	Scale      float32
}

// ErrHIPWMMAShape reports reference inputs that are not the pinned geometry.
var ErrHIPWMMAShape = errors.New("hip wmma attention: inputs do not match the pinned geometry")

// HIPAttentionWMMAOptIn reports whether the process asked for the WMMA route.
func HIPAttentionWMMAOptIn() bool {
	return os.Getenv(HIPWMMAOptInEnv) == "1"
}

// SelectHIPAttentionRoute is the selection contract wmma_attention_selected in
// rocm_kernels.hip implements: WMMA only when opted in, on gfx1151 wave32, at
// exactly the pinned geometry. Anything else keeps the generic kernel.
func SelectHIPAttentionRoute(optIn bool, c HIPAttentionWMMACall) HIPAttentionRoute {
	if !optIn {
		return HIPAttentionRouteGeneric
	}
	base := c.Arch
	for i := 0; i < len(base); i++ {
		if base[i] == ':' {
			base = base[:i]
			break
		}
	}
	arch, known := LookupROCmArch(base)
	if base != HIPWMMAArch || !known || arch.Wavefront != 32 || !arch.HasNativeWave32WMMA() {
		return HIPAttentionRouteGeneric
	}
	if c.QueryHeads != HIPWMMAQueryHeads || c.KVHeads != HIPWMMAKVHeads ||
		c.HeadDim != HIPWMMAHeadDim || c.GroupSize != HIPWMMAQueryHeads/HIPWMMAKVHeads ||
		c.Positions < 1 || c.Scale != hipWMMAScale {
		return HIPAttentionRouteGeneric
	}
	return HIPAttentionRouteWMMA
}

// HIPAttentionWMMAInput is one launch of the WMMA kernel. Q and Gate are
// [BatchSize][24][256]; K and V are the resident cache, [position][4][256],
// holding at least StartPos+BatchSize positions. Row r of the chunk sits at
// absolute position StartPos+r and attends keys [KeyBegin, StartPos+r].
type HIPAttentionWMMAInput struct {
	Q, Gate   []float32
	K, V      []float32
	StartPos  int
	BatchSize int
	KeyBegin  int
	WantLSE   bool
}

// hipWMMAHalf rounds to IEEE binary16, nearest-even, and widens back: the
// value an FP16 operand of the kernel carries.
func hipWMMAHalf(f float32) float32 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>31) << 15
	exp := int(bits>>23) & 0xFF
	mant := bits & 0x7FFFFF
	var h uint16
	switch e := exp - 127; {
	case exp == 255:
		h = 0x7C00
		if mant != 0 {
			h = 0x7E00
		}
	case e > 15:
		h = 0x7C00
	case e >= -14:
		h = uint16(e+15)<<10 | uint16(mant>>13)
		if rem := mant & 0x1FFF; rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
			h++ // a mantissa carry correctly rolls into the exponent, up to Inf
		}
	case e >= -25:
		full := mant | 0x800000
		shift := uint(-1 - e)
		h = uint16(full >> shift)
		rem, half := full&(1<<shift-1), uint32(1)<<(shift-1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
	}
	h |= sign

	hexp, hmant := int(h>>10)&0x1F, float64(h&0x3FF)
	var v float64
	switch hexp {
	case 31:
		v = math.Inf(1)
		if hmant != 0 {
			v = math.NaN()
		}
	case 0:
		v = math.Ldexp(hmant, -24)
	default:
		v = math.Ldexp(1024+hmant, hexp-25)
	}
	if h&0x8000 != 0 {
		v = math.Copysign(v, -1)
	}
	return float32(v)
}

// hipWMMAStep is one 16-wide matrix-core accumulate for a single output cell:
// c plus the dot of two FP16 fragments. Every product of two FP16 values is
// exact in FP32. The silicon's summation order inside the tile is not
// specified, so a balanced tree stands in for it, and results are compared
// against FP64 by tolerance rather than bit-for-bit.
func hipWMMAStep(c float32, a, b *[16]float32) float32 {
	var t [16]float32
	for i := range t {
		t[i] = a[i] * b[i]
	}
	for n := 16; n > 1; n /= 2 {
		for i := 0; i < n/2; i++ {
			t[i] = t[2*i] + t[2*i+1]
		}
	}
	return c + t[0]
}

func hipWMMAExp(x float32) float32 { return float32(math.Exp(float64(x))) }

func hipWMMAFinite(x float32) bool {
	return !math.IsInf(float64(x), 0) && !math.IsNaN(float64(x))
}

// hipWMMAMax is fmaxf: a NaN operand loses to the other one.
func hipWMMAMax(a, b float32) float32 {
	if b > a || a != a {
		return b
	}
	return a
}

// HIPAttentionWMMAReference computes what the WmmaCausalAttention kernel in
// rocm_kernels.hip computes, block by block and key tile by key tile, with the
// same FP16 operand rounding, online-softmax rescaling, lane reduction order
// and visible-keys-only handling of a query's last partial tile. It returns
// the context [BatchSize][24][256] and, when WantLSE is set, the log-sum-exp
// statistic laid out [head][row].
func HIPAttentionWMMAReference(in HIPAttentionWMMAInput) (out, lse []float32, err error) {
	const (
		heads     = HIPWMMAQueryHeads
		kvHeads   = HIPWMMAKVHeads
		dim       = HIPWMMAHeadDim
		gqa       = heads / kvHeads
		attnWidth = heads * dim
		kvWidth   = kvHeads * dim
		keys      = HIPWMMAKeyTile
		rowBlocks = (HIPWMMAQueryRows / 16) * HIPWMMAHeadsPerBlock
		rows      = rowBlocks * 16
		lanes     = 256 / rows
		perLane   = keys / lanes
		kSteps    = dim / 16
		pairs     = gqa / HIPWMMAHeadsPerBlock
	)
	batch, start := in.BatchSize, in.StartPos
	if batch < 1 || start < 0 || in.KeyBegin < 0 {
		return nil, nil, fmt.Errorf("%w: start=%d batch=%d key_begin=%d", ErrHIPWMMAShape, start, batch, in.KeyBegin)
	}
	contextEnd := start + batch
	if len(in.Q) != batch*attnWidth || (in.Gate != nil && len(in.Gate) != batch*attnWidth) ||
		len(in.K) < contextEnd*kvWidth || len(in.V) < contextEnd*kvWidth {
		return nil, nil, fmt.Errorf("%w: q=%d gate=%d k=%d v=%d for start=%d batch=%d",
			ErrHIPWMMAShape, len(in.Q), len(in.Gate), len(in.K), len(in.V), start, batch)
	}
	out = make([]float32, batch*attnWidth)
	if in.WantLSE {
		lse = make([]float32, heads*batch)
	}
	negInf := float32(math.Inf(-1))

	var (
		qFrag  [rows][kSteps][16]float32
		oAcc   [rows][dim]float32
		runMax [rows]float32
		runSum [rows]float32
		kTile  [keys][kSteps][16]float32
		vT     [dim][16]float32 // transposed: one 16-key fragment per dim
		prob   [16]float32
	)
	for queryStart := 0; queryStart < batch; queryStart += HIPWMMAQueryRows {
		for headPair := 0; headPair < kvHeads*pairs; headPair++ {
			kvHead := headPair / pairs
			firstHead := kvHead*gqa + (headPair%pairs)*HIPWMMAHeadsPerBlock
			rowHead := func(rg int) int { return firstHead + (rg/16)%HIPWMMAHeadsPerBlock }
			rowQuery := func(rg int) int {
				return queryStart + ((rg/16)/HIPWMMAHeadsPerBlock)*16 + rg%16
			}

			for rg := 0; rg < rows; rg++ {
				local := rowQuery(rg)
				for d := 0; d < dim; d++ {
					f := float32(0)
					if local < batch {
						f = in.Q[local*attnWidth+rowHead(rg)*dim+d]
					}
					qFrag[rg][d/16][d%16] = hipWMMAHalf(f * hipWMMAScale)
					oAcc[rg][d] = 0
				}
				runMax[rg], runSum[rg] = negInf, 0
			}

			maxVisible := contextEnd
			if v := start + queryStart + HIPWMMAQueryRows; v < maxVisible {
				maxVisible = v
			}
			for keyStart := in.KeyBegin; keyStart < maxVisible; keyStart += keys {
				for key := 0; key < keys; key++ {
					pos := keyStart + key
					for d := 0; d < dim; d++ {
						var kv, vv float32
						if pos < contextEnd {
							kv = hipWMMAHalf(in.K[pos*kvWidth+kvHead*dim+d])
							vv = hipWMMAHalf(in.V[pos*kvWidth+kvHead*dim+d])
						}
						kTile[key][d/16][d%16] = kv
						vT[d][key] = vv
					}
				}

				for rg := 0; rg < rows; rg++ {
					local := rowQuery(rg)
					if local >= batch {
						continue // the kernel never stores a dead row
					}
					absQuery := start + local

					// S = Q K^T: each half of the head dimension is
					// accumulated by its own wave and the two are summed.
					var vals [keys]float32
					partMax := negInf
					for col := 0; col < keys; col++ {
						pos := keyStart + col
						vals[col] = negInf
						if pos <= absQuery && pos < contextEnd {
							var lo, hi float32
							for ks := 0; ks < kSteps/2; ks++ {
								lo = hipWMMAStep(lo, &qFrag[rg][ks], &kTile[col][ks])
								hi = hipWMMAStep(hi, &qFrag[rg][kSteps/2+ks], &kTile[col][kSteps/2+ks])
							}
							vals[col] = lo + hi
						}
						partMax = hipWMMAMax(partMax, vals[col])
					}

					prevMax := runMax[rg]
					nextMax := hipWMMAMax(prevMax, partMax)
					priorScale := float32(0)
					if hipWMMAFinite(prevMax) {
						priorScale = hipWMMAExp(prevMax - nextMax)
					}
					var laneSum [lanes]float32
					for seg := 0; seg < lanes; seg++ {
						for m := 0; m < perLane; m++ {
							col := seg*perLane + m
							w := float32(0)
							if hipWMMAFinite(vals[col]) {
								w = hipWMMAExp(vals[col] - nextMax)
							}
							laneSum[seg] += w
							prob[col] = hipWMMAHalf(w)
						}
					}
					// Butterfly reduction as lane 0 sees it; lane 0 owns the
					// row's statistics.
					for off := 1; off < lanes; off <<= 1 {
						for seg := 0; seg < lanes; seg += 2 * off {
							laneSum[seg] += laneSum[seg+off]
						}
					}
					runMax[rg] = nextMax
					runSum[rg] = float32(runSum[rg]*priorScale) + laneSum[0]

					// O += P V. A tile that runs past the query's own position
					// is summed over its visible keys only, in key order.
					partial := keyStart+keys-1 > absQuery
					for d := 0; d < dim; d++ {
						acc := oAcc[rg][d] * priorScale
						if partial {
							for key := 0; key < keys && keyStart+key <= absQuery; key++ {
								acc += prob[key] * vT[d][key]
							}
						} else {
							acc = hipWMMAStep(acc, &prob, &vT[d])
						}
						oAcc[rg][d] = acc
					}
				}
			}

			for rg := 0; rg < rows; rg++ {
				local := rowQuery(rg)
				if local >= batch {
					continue
				}
				denom := runSum[rg]
				base := local*attnWidth + rowHead(rg)*dim
				for d := 0; d < dim; d++ {
					value := float32(0)
					if denom > 0 {
						value = oAcc[rg][d] / denom
					}
					if in.Gate != nil && !in.WantLSE {
						value *= 1 / (1 + hipWMMAExp(-in.Gate[base+d]))
					}
					out[base+d] = value
				}
				if in.WantLSE {
					stat := negInf
					if denom > 0 {
						stat = runMax[rg] + float32(math.Log(float64(denom)))
					}
					lse[rowHead(rg)*batch+local] = stat
				}
			}
		}
	}
	return out, lse, nil
}

// HIPAttentionWMMADecodeReference is the launch frocm_attention_f32 issues on
// the WMMA route: one query row at the newest position, attending the whole
// resident cache.
func HIPAttentionWMMADecodeReference(q, k, v []float32, positions int) ([]float32, error) {
	if positions < 1 {
		return nil, fmt.Errorf("%w: positions=%d", ErrHIPWMMAShape, positions)
	}
	out, _, err := HIPAttentionWMMAReference(HIPAttentionWMMAInput{
		Q: q, K: k, V: v, StartPos: positions - 1, BatchSize: 1,
	})
	return out, err
}
