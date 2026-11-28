package compute

import (
	"errors"
	"fmt"
)

// Mid-market single-node sizing bounds (1-8 GPUs, 12-141 GiB VRAM per GPU).
// Hyperscaler multi-node scale-out, cross-rack RDMA/IB, DeepEP cross-node EP,
// and NVL72 rack topologies are explicitly out of scope and refused.
const (
	MinMidMarketGPUs    = 1
	MaxMidMarketGPUs    = 8
	MinMidMarketVRAMGiB = 12
	MaxMidMarketVRAMGiB = 141
	giBytes             = int64(1024 * 1024 * 1024)
)

// ErrOutOfScopeCluster is returned when a sizing configuration falls outside the
// mid-market single-node 1-8 GPU (12-141 GiB/GPU) envelope or asks for
// multi-node scale-out primitives.
var ErrOutOfScopeCluster = errors.New("compute: configuration outside single-node 1-8 GPU (12-141 GiB) scope")

// MidMarketNodeSpec describes a single-node 1-8 GPU mid-market host for
// prefill sizing and backend selection.
type MidMarketNodeSpec struct {
	Name              string
	NumGPUs           int
	VRAMPerGPUBytes   int64
	SMMajor           int
	SMMinor           int
	PrefixTokens      int
	KVPrecision       KVPrecision
	KVDtype           Dtype // optional explicit KV storage dtype (e.g. FP8, NVFP4, F16, BF16, F32)
	PagedKV           bool
	FP8Block32Weights bool    // include ceil(O/32)*ceil(I/32) E8M0 scale tiles on FP8 projections
	PeakFLOPSPerGPU   float64 // optional device FLOP/s peak for roofline timing floor
	PeakBWBytesPerGPU float64 // optional device HBM/GDDR bytes/s peak for roofline timing floor
	CrossNodeRDMA     bool
	MultiNodeNVL72    bool
}

// Validate checks that the node specification stays within single-node 1-8 GPU
// (12-141 GiB/GPU) bounds.
func (n MidMarketNodeSpec) Validate() error {
	if n.CrossNodeRDMA || n.MultiNodeNVL72 {
		return fmt.Errorf("%w: multi-node RDMA/NVL72 is out of scope", ErrOutOfScopeCluster)
	}
	if n.NumGPUs < MinMidMarketGPUs || n.NumGPUs > MaxMidMarketGPUs {
		return fmt.Errorf("%w: numGPUs=%d must be in [%d, %d]", ErrOutOfScopeCluster, n.NumGPUs, MinMidMarketGPUs, MaxMidMarketGPUs)
	}
	minBytes := int64(MinMidMarketVRAMGiB) * giBytes
	maxBytes := int64(MaxMidMarketVRAMGiB) * giBytes
	if n.VRAMPerGPUBytes < minBytes || n.VRAMPerGPUBytes > maxBytes {
		return fmt.Errorf("%w: vramPerGPU=%d must be in [%d GiB, %d GiB]", ErrOutOfScopeCluster, n.VRAMPerGPUBytes, MinMidMarketVRAMGiB, MaxMidMarketVRAMGiB)
	}
	return nil
}

// MidMarketPrefillPlan reports memory residency, capacity fit, kernel backend
// selection, and analytic roofline sizing for a model prefill on one
// mid-market 1-8 GPU node.
type MidMarketPrefillPlan struct {
	Geometry                      PrefillGeometry
	Node                          MidMarketNodeSpec
	Regime                        ConversionRegime
	Roofline                      PrefillRoofline
	WeightsBytesTotal             int64
	WeightsBytesPerGPU            int64
	KVCacheBytesTotal             int64
	KVCacheBytesPerGPU            int64
	ActivationPanelBytesPerGPU    int64
	TPCollectiveBufferBytesPerGPU int64
	TotalRequiredBytesPerGPU      int64
	HeadroomBytesPerGPU           int64
	VRAMUtilization               float64
	Fits                          bool
	AttentionSelection            CUDAAttentionSelection
	MMQFastPathEligible           bool
	MMQRoute                      string
	RooflineFloorSecondsPerGPU    float64
}

// SizePrefillMidMarket computes prefill memory sizing, kernel backend
// selection, and analytic roofline estimates for FP8, NVFP4, Q4_K, and peer
// formats across a single-node 1-8 GPU (12-141 GiB/GPU) box.
func SizePrefillMidMarket(g PrefillGeometry, node MidMarketNodeSpec) (MidMarketPrefillPlan, error) {
	if err := node.Validate(); err != nil {
		return MidMarketPrefillPlan{}, err
	}
	if g.DModel <= 0 || g.NHeads <= 0 || g.NKVHeads <= 0 || g.HeadDim <= 0 || g.DFF <= 0 || g.NLayers <= 0 || g.P <= 0 {
		return MidMarketPrefillPlan{}, fmt.Errorf("compute: invalid prefill geometry %+v", g)
	}
	if node.PrefixTokens < 0 {
		return MidMarketPrefillPlan{}, fmt.Errorf("compute: negative prefix tokens %d", node.PrefixTokens)
	}

	// Reuses PrefillCostModel + Profile directly so weight and FLOPs accounting shares one SSOT.
	stages := PrefillCostModel(g)
	roof := Profile(g)

	var weightsTotal int64
	for _, st := range stages {
		weightsTotal += st.WeightBytes
	}
	if node.FP8Block32Weights && g.WeightDtype == FP8 {
		weightsTotal += g.fp8Block32ScaleOverhead()
	}

	tp := int64(node.NumGPUs)
	// Embeddings / LM head scale linearly with vocab shard or stay partitioned cleanly;
	// divide total weights across TP ranks with ceiling rounding.
	weightsPerGPU := (weightsTotal + tp - 1) / tp

	totalSeq := node.PrefixTokens + g.P
	kvTotal := estimatePrefillKVBytes(g, totalSeq, node.KVPrecision, node.KVDtype)
	kvPerGPU := (kvTotal + tp - 1) / tp

	// Prefill panel working memory per GPU: input/residual, Q/K/V panels, FFN gate/up,
	// and attention context output sized at panel height P (flash/tiled attention avoids O(P^2) materialization).
	qOut := int64(g.NHeads * g.HeadDim)
	kvOut := int64(g.NKVHeads * g.HeadDim)
	p := int64(g.P)
	actElemsPerGPU := p*(int64(g.DModel)*2) +
		((p*qOut + tp - 1) / tp) +
		2*((p*kvOut+tp-1)/tp) +
		2*((p*int64(g.DFF)+tp-1)/tp)
	actBytesPerGPU := actElemsPerGPU * int64(F32.Bytes())

	var tpBufBytesPerGPU int64
	if node.NumGPUs > 1 {
		// Single-node ring/all-reduce double buffer for residual panel synchronization.
		tpBufBytesPerGPU = p * int64(g.DModel) * int64(F32.Bytes()) * 2
	}

	totalReq := weightsPerGPU + kvPerGPU + actBytesPerGPU + tpBufBytesPerGPU
	headroom := node.VRAMPerGPUBytes - totalReq
	util := float64(totalReq) / float64(node.VRAMPerGPUBytes)

	major, minor := node.SMMajor, node.SMMinor
	if major == 0 {
		major, minor = 12, 0
	}
	attnSel := ResolveCUDAAttentionSelection(major, minor, CUDARegimePrefill, CUDAAttentionOptions{
		HeadDim:     g.HeadDim,
		Tokens:      g.P,
		KVPrecision: node.KVPrecision,
		KVDtype:     node.KVDtype,
		Paged:       node.PagedKV,
	})

	mmqKind, mmqOK := Ink2PDenseMMQKind(g.WeightDtype, &QuantSpec{Block: q8Block, Axis: 2, Bits: 8, Symmetric: true}, g.P, g.DModel, g.DModel, true)
	_ = mmqKind
	mmqRoute := Ink2PMatMulRoute(g.WeightDtype, &QuantSpec{Block: q8Block, Axis: 2, Bits: 8, Symmetric: true}, g.P, g.DModel, g.DModel, true)

	var floorSec float64
	if node.PeakFLOPSPerGPU > 0 && node.PeakBWBytesPerGPU > 0 {
		// Partition FLOPs and weight/KV streaming across TP ranks within the single node.
		perGPUGeom := g
		perGPUGeom.NHeads = maxInt(1, g.NHeads/node.NumGPUs)
		perGPUGeom.NKVHeads = maxInt(1, g.NKVHeads/node.NumGPUs)
		perGPUGeom.DFF = maxInt(1, g.DFF/node.NumGPUs)
		floorSec = Profile(perGPUGeom).PredictTime(node.PeakFLOPSPerGPU, node.PeakBWBytesPerGPU).TotalSeconds
	}

	return MidMarketPrefillPlan{
		Geometry:                      g,
		Node:                          node,
		Regime:                        ConversionRegimeFromDtype(g.WeightDtype),
		Roofline:                      roof,
		WeightsBytesTotal:             weightsTotal,
		WeightsBytesPerGPU:            weightsPerGPU,
		KVCacheBytesTotal:             kvTotal,
		KVCacheBytesPerGPU:            kvPerGPU,
		ActivationPanelBytesPerGPU:    actBytesPerGPU,
		TPCollectiveBufferBytesPerGPU: tpBufBytesPerGPU,
		TotalRequiredBytesPerGPU:      totalReq,
		HeadroomBytesPerGPU:           headroom,
		VRAMUtilization:               util,
		Fits:                          totalReq <= node.VRAMPerGPUBytes,
		AttentionSelection:            attnSel,
		MMQFastPathEligible:           mmqOK,
		MMQRoute:                      mmqRoute,
		RooflineFloorSecondsPerGPU:    floorSec,
	}, nil
}

func (g PrefillGeometry) fp8Block32ScaleOverhead() int64 {
	qOut := g.NHeads * g.HeadDim
	kvOut := g.NKVHeads * g.HeadDim
	tileBytes := func(o, i int) int64 {
		so, si := fp8Block32ScaleShape(o, i)
		return int64(so) * int64(si)
	}
	perLayer := tileBytes(qOut, g.DModel) +
		tileBytes(kvOut, g.DModel)*2 +
		tileBytes(g.DModel, qOut) +
		tileBytes(g.DFF, g.DModel)*2 +
		tileBytes(g.DModel, g.DFF)
	total := int64(g.NLayers) * perLayer
	if g.Vocab > 0 {
		total += tileBytes(g.Vocab, g.DModel)
	}
	return total
}

func estimatePrefillKVBytes(g PrefillGeometry, tokens int, prec KVPrecision, kvDtype Dtype) int64 {
	if tokens <= 0 || g.NLayers <= 0 || g.NKVHeads <= 0 || g.HeadDim <= 0 {
		return 0
	}
	if kvDtype == 0 || (kvDtype == F32 && prec != 0) {
		return EstimateKVStoreBytes(KVConfig{
			NumLayers:  g.NLayers,
			NumKVHeads: g.NKVHeads,
			HeadDim:    g.HeadDim,
			Precision:  prec,
		}, tokens)
	}
	elemsPerRow := int64(g.NKVHeads) * int64(g.HeadDim)
	var perTokenPerLayer int64
	switch kvDtype {
	case FP8:
		// Pre-RoPE K kept F32 for exact re-positioning + two attended E4M3 planes + per-head F32 scales.
		kRaw := elemsPerRow * int64(F32.Bytes())
		fp8Row := elemsPerRow + int64(g.NKVHeads)*int64(F32.Bytes())
		perTokenPerLayer = kRaw + 2*fp8Row
	case NVFP4:
		// Pre-RoPE K kept F32 + two packed E2M1 attended planes + per-16 E4M3FN block scales.
		kRaw := elemsPerRow * int64(F32.Bytes())
		nblk := (elemsPerRow + NVFP4BlockSize - 1) / NVFP4BlockSize
		nvfp4Row := (elemsPerRow+1)/2 + nblk
		perTokenPerLayer = kRaw + 2*nvfp4Row
	default:
		return EstimateKVStoreBytes(KVConfig{
			NumLayers:  g.NLayers,
			NumKVHeads: g.NKVHeads,
			HeadDim:    g.HeadDim,
			Precision:  prec,
		}, tokens)
	}
	return int64(g.NLayers) * int64(tokens) * perTokenPerLayer
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
