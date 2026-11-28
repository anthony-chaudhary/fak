package compute

// Gated Delta Net (GDN) Split-V MTP decode occupancy partitioner and host
// reduction oracle.
//
// ADAPT of vllm-project/vllm@8bb1147817f79d36669efcdf461cf63e10b3679b
// (csrc/gdn/fused_gdn_decode_kernel.cu,
// vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py, Apache-2.0).
//
// Clean-room Go contract:
//   - Supports MTP decode batches up to K=8 proposals (TokensPerRequest = 1..9).
//   - At small batch*heads relative to SMCount, splits each value head's V
//     dimension across a cluster of SplitV in {1, 2, 4} thread blocks of width
//     ChunkSize = ceil(ValueHeadDim / SplitV) so otherwise-idle SMs participate.
//   - Each split independently updates its recurrent state slice S[:, vStart:vEnd]
//     and emits a partial sum-of-squares accumulator over its V chunk.
//   - MergeGDNSplitVPartials reassembles out-of-order split partials in strict
//     ascending SplitIdx order (0..SplitV-1), computing the gated RMSNorm output
//     matching the unsplit full-head oracle.
//   - Host contract and oracle only ([SW-VERIFIED]): no hardware execution or
//     throughput claim.

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

const (
	// GDNMaxMTPProposals is the maximum MTP speculative proposal depth (K=8 -> 9 tokens/req).
	GDNMaxMTPProposals = 8
	// GDNMaxTokensPerRequest is 1 base token + GDNMaxMTPProposals proposals.
	GDNMaxTokensPerRequest = 1 + GDNMaxMTPProposals
)

var (
	// ErrGDNSplitVInvalidGeometry is the closed refusal for malformed GDN dimensions,
	// proposal counts, SM budget, or SplitV cluster sizes.
	ErrGDNSplitVInvalidGeometry = errors.New("compute: gdn split-v invalid geometry or split count")
	// ErrGDNSplitVMissingPartial is the closed refusal when split partials are missing,
	// duplicated, out-of-range, or fail to cover [0, ValueHeadDim) contiguously.
	ErrGDNSplitVMissingPartial = errors.New("compute: gdn split-v incomplete or invalid split partials")
)

// GDNSplitVConfig parameterizes GDN MTP decode geometry and Split-V occupancy.
type GDNSplitVConfig struct {
	BatchSize     int
	Proposals     int // 0..GDNMaxMTPProposals -> TokensPerRequest = 1 + Proposals
	NumKeyHeads   int
	NumValueHeads int
	KeyHeadDim    int
	ValueHeadDim  int
	SMCount       int     // optional SM count for auto SplitV selection (0 => SplitV=1)
	ForcedSplitV  int     // 0 = auto; 1, 2, or 4 = explicit cluster size
	RMSNormEps    float32 // must be finite and > 0
}

// GDNSplitVSlice describes one split's [VStart, VEnd) slice of ValueHeadDim.
type GDNSplitVSlice struct {
	SplitIdx int
	VStart   int
	VEnd     int
}

// GDNSplitVPlan is the validated Split-V tile partition over (BatchSize, NumValueHeads).
type GDNSplitVPlan struct {
	Config           GDNSplitVConfig
	TokensPerRequest int
	SplitV           int
	ChunkSize        int
	BaseBlocks       int
	TotalBlocks      int
	Partitions       []GDNSplitVSlice
}

// GDNSplitVPartial holds one split's computed readout slice and partial
// sum-of-squares accumulator for one (batch, token, valueHead).
type GDNSplitVPartial struct {
	SplitIdx     int
	VStart       int
	VEnd         int
	Readout      []float32
	PartialSumSq float64
}

// GDNSplitVInputs provides flattened row-major buffers for one GDN MTP decode batch.
//   - Q, K: [BatchSize, TokensPerRequest, NumKeyHeads, KeyHeadDim]
//   - V, Z: [BatchSize, TokensPerRequest, NumValueHeads, ValueHeadDim]
//   - Decay, Beta: [BatchSize, TokensPerRequest, NumValueHeads]
//   - Norm: [ValueHeadDim]
//   - RecurrentState: [BatchSize, NumValueHeads, KeyHeadDim, ValueHeadDim]
type GDNSplitVInputs struct {
	Q              []float32
	K              []float32
	V              []float32
	Z              []float32
	Decay          []float32
	Beta           []float32
	Norm           []float32
	RecurrentState []float32
}

// GDNSplitVResult returns the output [BatchSize, TokensPerRequest, NumValueHeads, ValueHeadDim]
// and updated recurrent state [BatchSize, NumValueHeads, KeyHeadDim, ValueHeadDim].
type GDNSplitVResult struct {
	Plan           GDNSplitVPlan
	Out            []float32
	RecurrentState []float32
}

// PlanGDNSplitV validates GDN geometry and computes the Split-V cluster size (1, 2, or 4)
// and per-split value-head chunk partitions.
func PlanGDNSplitV(cfg GDNSplitVConfig) (GDNSplitVPlan, error) {
	if cfg.BatchSize <= 0 || cfg.Proposals < 0 || cfg.Proposals > GDNMaxMTPProposals {
		return GDNSplitVPlan{}, fmt.Errorf("%w: batch=%d proposals=%d", ErrGDNSplitVInvalidGeometry, cfg.BatchSize, cfg.Proposals)
	}
	if cfg.NumKeyHeads <= 0 || cfg.NumValueHeads <= 0 || cfg.NumValueHeads%cfg.NumKeyHeads != 0 {
		return GDNSplitVPlan{}, fmt.Errorf("%w: keyHeads=%d valueHeads=%d", ErrGDNSplitVInvalidGeometry, cfg.NumKeyHeads, cfg.NumValueHeads)
	}
	if cfg.KeyHeadDim <= 0 || cfg.ValueHeadDim <= 0 {
		return GDNSplitVPlan{}, fmt.Errorf("%w: keyDim=%d valueDim=%d", ErrGDNSplitVInvalidGeometry, cfg.KeyHeadDim, cfg.ValueHeadDim)
	}
	if cfg.SMCount < 0 {
		return GDNSplitVPlan{}, fmt.Errorf("%w: smCount=%d", ErrGDNSplitVInvalidGeometry, cfg.SMCount)
	}
	if !(cfg.RMSNormEps > 0) || math.IsNaN(float64(cfg.RMSNormEps)) || math.IsInf(float64(cfg.RMSNormEps), 0) {
		return GDNSplitVPlan{}, fmt.Errorf("%w: rmsNormEps=%v", ErrGDNSplitVInvalidGeometry, cfg.RMSNormEps)
	}

	if cfg.BatchSize > math.MaxInt/cfg.NumValueHeads {
		return GDNSplitVPlan{}, fmt.Errorf("%w: baseBlocks overflow", ErrGDNSplitVInvalidGeometry)
	}
	baseBlocks := cfg.BatchSize * cfg.NumValueHeads

	var splitV int
	if cfg.ForcedSplitV != 0 {
		switch cfg.ForcedSplitV {
		case 1, 2, 4:
			if cfg.ForcedSplitV > cfg.ValueHeadDim {
				return GDNSplitVPlan{}, fmt.Errorf("%w: forcedSplitV=%d exceeds valueHeadDim=%d",
					ErrGDNSplitVInvalidGeometry, cfg.ForcedSplitV, cfg.ValueHeadDim)
			}
			splitV = cfg.ForcedSplitV
		default:
			return GDNSplitVPlan{}, fmt.Errorf("%w: forcedSplitV=%d must be 1, 2, or 4", ErrGDNSplitVInvalidGeometry, cfg.ForcedSplitV)
		}
	} else {
		splitV = 1
		if cfg.SMCount > 0 && baseBlocks < cfg.SMCount {
			ratio := cfg.SMCount / baseBlocks
			switch {
			case ratio >= 4 && cfg.ValueHeadDim >= 4:
				splitV = 4
			case ratio >= 2 && cfg.ValueHeadDim >= 2:
				splitV = 2
			}
		}
	}

	if baseBlocks > math.MaxInt/splitV {
		return GDNSplitVPlan{}, fmt.Errorf("%w: totalBlocks overflow", ErrGDNSplitVInvalidGeometry)
	}

	chunkSize := (cfg.ValueHeadDim + splitV - 1) / splitV
	partitions := make([]GDNSplitVSlice, splitV)
	for s := 0; s < splitV; s++ {
		vStart := s * chunkSize
		if vStart > cfg.ValueHeadDim {
			vStart = cfg.ValueHeadDim
		}
		vEnd := (s + 1) * chunkSize
		if vEnd > cfg.ValueHeadDim {
			vEnd = cfg.ValueHeadDim
		}
		partitions[s] = GDNSplitVSlice{
			SplitIdx: s,
			VStart:   vStart,
			VEnd:     vEnd,
		}
	}

	return GDNSplitVPlan{
		Config:           cfg,
		TokensPerRequest: 1 + cfg.Proposals,
		SplitV:           splitV,
		ChunkSize:        chunkSize,
		BaseBlocks:       baseBlocks,
		TotalBlocks:      baseBlocks * splitV,
		Partitions:       partitions,
	}, nil
}

func validateGDNSplitVInputs(plan GDNSplitVPlan, in GDNSplitVInputs) error {
	cfg := plan.Config
	bt := cfg.BatchSize * plan.TokensPerRequest
	wantQK := bt * cfg.NumKeyHeads * cfg.KeyHeadDim
	wantVZ := bt * cfg.NumValueHeads * cfg.ValueHeadDim
	wantScalars := bt * cfg.NumValueHeads
	wantState := cfg.BatchSize * cfg.NumValueHeads * cfg.KeyHeadDim * cfg.ValueHeadDim
	if len(in.Q) != wantQK || len(in.K) != wantQK {
		return fmt.Errorf("%w: Q/K length %d/%d want %d", ErrGDNSplitVInvalidGeometry, len(in.Q), len(in.K), wantQK)
	}
	if len(in.V) != wantVZ || len(in.Z) != wantVZ {
		return fmt.Errorf("%w: V/Z length %d/%d want %d", ErrGDNSplitVInvalidGeometry, len(in.V), len(in.Z), wantVZ)
	}
	if len(in.Decay) != wantScalars || len(in.Beta) != wantScalars {
		return fmt.Errorf("%w: decay/beta length %d/%d want %d", ErrGDNSplitVInvalidGeometry, len(in.Decay), len(in.Beta), wantScalars)
	}
	if len(in.Norm) != cfg.ValueHeadDim {
		return fmt.Errorf("%w: norm length %d want %d", ErrGDNSplitVInvalidGeometry, len(in.Norm), cfg.ValueHeadDim)
	}
	if len(in.RecurrentState) != wantState {
		return fmt.Errorf("%w: recurrentState length %d want %d", ErrGDNSplitVInvalidGeometry, len(in.RecurrentState), wantState)
	}
	return nil
}

func gdnSilu64(x float64) float64 {
	return x / (1.0 + math.Exp(-x))
}

// ComputeGDNSplitVPartial executes one split slice [slice.VStart, slice.VEnd)
// of a single (batch, token, valueHead) GDN state update in-place on stateHead
// (row-major [keyHeadDim, valueHeadDim]) and returns its partial readout and
// partial sum-of-squares accumulator.
func ComputeGDNSplitVPartial(
	slice GDNSplitVSlice,
	qHead, kHead, vHead, stateHead []float32,
	keyHeadDim, valueHeadDim int,
	decay, beta float32,
) (GDNSplitVPartial, error) {
	if slice.SplitIdx < 0 || slice.VStart < 0 || slice.VEnd < slice.VStart || slice.VEnd > valueHeadDim {
		return GDNSplitVPartial{}, fmt.Errorf("%w: slice %+v for valueHeadDim=%d", ErrGDNSplitVInvalidGeometry, slice, valueHeadDim)
	}
	if len(qHead) != keyHeadDim || len(kHead) != keyHeadDim || len(vHead) != valueHeadDim || len(stateHead) != keyHeadDim*valueHeadDim {
		return GDNSplitVPartial{}, fmt.Errorf("%w: head operand length mismatch", ErrGDNSplitVInvalidGeometry)
	}
	width := slice.VEnd - slice.VStart
	readout := make([]float32, width)
	var sumSq float64
	decay64 := float64(decay)
	beta64 := float64(beta)
	for offset := 0; offset < width; offset++ {
		d := slice.VStart + offset
		var kvmem float64
		for i := 0; i < keyHeadDim; i++ {
			idx := i*valueHeadDim + d
			s := float64(stateHead[idx]) * decay64
			stateHead[idx] = float32(s)
			kvmem += s * float64(kHead[i])
		}
		delta := (float64(vHead[d]) - kvmem) * beta64
		var r float64
		for i := 0; i < keyHeadDim; i++ {
			idx := i*valueHeadDim + d
			s := float64(stateHead[idx]) + float64(kHead[i])*delta
			stateHead[idx] = float32(s)
			r += s * float64(qHead[i])
		}
		rf := float32(r)
		readout[offset] = rf
		sumSq += float64(rf) * float64(rf)
	}
	return GDNSplitVPartial{
		SplitIdx:     slice.SplitIdx,
		VStart:       slice.VStart,
		VEnd:         slice.VEnd,
		Readout:      readout,
		PartialSumSq: sumSq,
	}, nil
}

// MergeGDNSplitVPartials reassembles per-split GDN partials in deterministic
// ascending SplitIdx order (0..splitV-1), verifies full contiguous coverage of
// [0, valueHeadDim), and computes the gated RMSNorm head output.
func MergeGDNSplitVPartials(
	partials []GDNSplitVPartial,
	zHead, norm []float32,
	valueHeadDim int,
	rmsNormEps float32,
) ([]float32, error) {
	if len(partials) == 0 || valueHeadDim <= 0 || len(zHead) != valueHeadDim || len(norm) != valueHeadDim {
		return nil, fmt.Errorf("%w: partials=%d valueHeadDim=%d", ErrGDNSplitVMissingPartial, len(partials), valueHeadDim)
	}
	if !(rmsNormEps > 0) || math.IsNaN(float64(rmsNormEps)) || math.IsInf(float64(rmsNormEps), 0) {
		return nil, fmt.Errorf("%w: rmsNormEps=%v", ErrGDNSplitVInvalidGeometry, rmsNormEps)
	}

	ordered := make([]GDNSplitVPartial, len(partials))
	copy(ordered, partials)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].SplitIdx < ordered[j].SplitIdx
	})

	readout := make([]float32, valueHeadDim)
	var totalSS float64
	expectedStart := 0
	for idx, p := range ordered {
		if p.SplitIdx != idx {
			return nil, fmt.Errorf("%w: split index %d want %d", ErrGDNSplitVMissingPartial, p.SplitIdx, idx)
		}
		if p.VStart != expectedStart || p.VEnd < p.VStart || p.VEnd > valueHeadDim {
			return nil, fmt.Errorf("%w: split %d range [%d,%d) expected start %d",
				ErrGDNSplitVMissingPartial, p.SplitIdx, p.VStart, p.VEnd, expectedStart)
		}
		if len(p.Readout) != p.VEnd-p.VStart {
			return nil, fmt.Errorf("%w: split %d readout length %d want %d",
				ErrGDNSplitVMissingPartial, p.SplitIdx, len(p.Readout), p.VEnd-p.VStart)
		}
		copy(readout[p.VStart:p.VEnd], p.Readout)
		totalSS += p.PartialSumSq
		expectedStart = p.VEnd
	}
	if expectedStart != valueHeadDim {
		return nil, fmt.Errorf("%w: covered [0,%d) want [0,%d)", ErrGDNSplitVMissingPartial, expectedStart, valueHeadDim)
	}

	inv := 1.0 / math.Sqrt(totalSS/float64(valueHeadDim)+float64(rmsNormEps))
	out := make([]float32, valueHeadDim)
	for d := 0; d < valueHeadDim; d++ {
		out[d] = float32(float64(norm[d]) * float64(readout[d]) * inv * gdnSilu64(float64(zHead[d])))
	}
	return out, nil
}

// RunGDNSplitVDecode runs a full MTP decode batch using the Split-V partition
// plan without mutating the caller's input state slice.
func RunGDNSplitVDecode(cfg GDNSplitVConfig, in GDNSplitVInputs) (*GDNSplitVResult, error) {
	plan, err := PlanGDNSplitV(cfg)
	if err != nil {
		return nil, err
	}
	if err := validateGDNSplitVInputs(plan, in); err != nil {
		return nil, err
	}

	state := append([]float32(nil), in.RecurrentState...)
	out := make([]float32, len(in.V))
	groupRatio := cfg.NumValueHeads / cfg.NumKeyHeads
	T := plan.TokensPerRequest

	for b := 0; b < cfg.BatchSize; b++ {
		for t := 0; t < T; t++ {
			bt := b*T + t
			for h := 0; h < cfg.NumValueHeads; h++ {
				kh := h / groupRatio
				qOff := (bt*cfg.NumKeyHeads + kh) * cfg.KeyHeadDim
				vOff := (bt*cfg.NumValueHeads + h) * cfg.ValueHeadDim
				sOff := (b*cfg.NumValueHeads + h) * cfg.KeyHeadDim * cfg.ValueHeadDim
				scOff := bt*cfg.NumValueHeads + h

				qHead := in.Q[qOff : qOff+cfg.KeyHeadDim]
				kHead := in.K[qOff : qOff+cfg.KeyHeadDim]
				vHead := in.V[vOff : vOff+cfg.ValueHeadDim]
				zHead := in.Z[vOff : vOff+cfg.ValueHeadDim]
				stateHead := state[sOff : sOff+cfg.KeyHeadDim*cfg.ValueHeadDim]

				partials := make([]GDNSplitVPartial, plan.SplitV)
				for s, slice := range plan.Partitions {
					p, err := ComputeGDNSplitVPartial(
						slice, qHead, kHead, vHead, stateHead,
						cfg.KeyHeadDim, cfg.ValueHeadDim,
						in.Decay[scOff], in.Beta[scOff],
					)
					if err != nil {
						return nil, err
					}
					partials[s] = p
				}
				headOut, err := MergeGDNSplitVPartials(partials, zHead, in.Norm, cfg.ValueHeadDim, cfg.RMSNormEps)
				if err != nil {
					return nil, err
				}
				copy(out[vOff:vOff+cfg.ValueHeadDim], headOut)
			}
		}
	}

	return &GDNSplitVResult{
		Plan:           plan,
		Out:            out,
		RecurrentState: state,
	}, nil
}

// RunGDNFullHeadOracle executes the unsplit full-head GDN decode reference
// directly (no Split-V partitioning) for closed parity verification.
func RunGDNFullHeadOracle(cfg GDNSplitVConfig, in GDNSplitVInputs) (*GDNSplitVResult, error) {
	unsplitCfg := cfg
	unsplitCfg.ForcedSplitV = 1
	return RunGDNSplitVDecode(unsplitCfg, in)
}
