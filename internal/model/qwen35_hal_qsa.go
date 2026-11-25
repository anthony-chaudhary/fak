package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type qwen35QSAScratchPlan struct {
	rowWidth, queryWidth, kvLen, totalBlocks, gatherLen int
	scratchBytes                                        int64
}

// planQwen35QSAScratch admits the existing host-assisted gather envelope before
// any tensor readback or scratch mutation. Cache facts bound K/V allocation;
// they do not establish physical cache residency or device execution.
func planQwen35QSAScratch(be compute.Backend, totalTokens, hd, nH, nKV, grp int) (qwen35QSAScratchPlan, error) {
	var plan qwen35QSAScratchPlan
	if totalTokens < compute.QSADynamicGatingThreshold {
		return plan, fmt.Errorf("model: context length %d < dynamic gating threshold %d", totalTokens, compute.QSADynamicGatingThreshold)
	}
	profile, known := compute.ResolveDeviceMemoryProfile(be, "")
	if !known || !profile.CachePlanningAvailable() {
		return plan, errors.New("model: QSA selected-device cache planning unavailable")
	}
	if hd <= 0 || nH <= 0 || nKV <= 0 || grp <= 0 || nH%nKV != 0 || nH/nKV != grp {
		return plan, errors.New("model: invalid QSA head geometry")
	}
	maxInt := int(^uint(0) >> 1)
	maxElements := maxInt / 4 // Float32 lengths must also have representable byte sizes.
	if nKV > maxElements/hd || nH > maxElements/hd {
		return plan, errors.New("model: QSA head geometry overflows")
	}
	w := nKV * hd
	if totalTokens > maxElements/w {
		return plan, errors.New("model: QSA KV geometry overflows")
	}
	totalBlocks := 1 + (totalTokens-1)/compute.QSABlockSize
	if totalBlocks > math.MaxInt32 || totalBlocks > maxInt/compute.QSABlockSize {
		return plan, errors.New("model: QSA block geometry overflows")
	}
	gatherTokens := ((compute.QSAMaxGatherTokens + compute.QSATileSize - 1) / compute.QSATileSize) * compute.QSATileSize
	if w > maxElements/gatherTokens {
		return plan, errors.New("model: QSA gather geometry overflows")
	}
	// Include grow's per-row headroom in the aggregate score size. Each score
	// row is larger than a slice header, so this also bounds grow2D's outer slice.
	if grp > maxElements/growCap(gatherTokens) {
		return plan, errors.New("model: QSA score geometry overflows")
	}
	gatherLen := gatherTokens * w
	rawPerBuffer := int64(gatherLen) * 4
	alignment := profile.StorageAlignmentBytes
	if rawPerBuffer > math.MaxInt64-(alignment-1) {
		return plan, errors.New("model: QSA scratch alignment overflows")
	}
	alignedPerBuffer := ((rawPerBuffer + alignment - 1) / alignment) * alignment
	if alignedPerBuffer > math.MaxInt64/2 {
		return plan, errors.New("model: QSA combined scratch size overflows")
	}
	scratchBytes := alignedPerBuffer * 2
	if scratchBytes > profile.CacheCapacityBytes {
		return plan, fmt.Errorf("model: QSA scratch size %d exceeds selected-device cache cap %d", scratchBytes, profile.CacheCapacityBytes)
	}
	return qwen35QSAScratchPlan{
		rowWidth: w, queryWidth: nH * hd, kvLen: totalTokens * w,
		totalBlocks: totalBlocks, gatherLen: gatherLen, scratchBytes: scratchBytes,
	}, nil
}

func (s *Session) qwen35QSASparseAttentionHAL(layer, kvLayer int, q compute.Tensor, grp int, scale float32) (compute.Tensor, error) {
	s.refuseHostFallback("Qwen QSA sparse attention")
	be, cfg := s.Backend, s.M.Cfg
	hd, nH, nKV := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	totalTokens := s.halKV.Len()
	plan, err := planQwen35QSAScratch(be, totalTokens, hd, nH, nKV, grp)
	if err != nil {
		return compute.Tensor{}, err
	}

	blockSize := compute.QSABlockSize
	w, totalBlocks := plan.rowWidth, plan.totalBlocks

	if s.qwen35HAL == nil {
		s.qwen35HAL = &qwen35HALState{}
	}
	hal := s.qwen35HAL
	hal.qsaBlockScores = grow(hal.qsaBlockScores, totalBlocks)
	scores := hal.qsaBlockScores[:totalBlocks]

	qHost := s.readQwen35FullAttention(layer, "QSA query read", q)
	if len(qHost) < plan.queryWidth {
		return compute.Tensor{}, fmt.Errorf("model: QSA query length %d smaller than required %d", len(qHost), plan.queryWidth)
	}
	q0 := vectorHead(qHost, 0, hd)

	kTensor := s.halKV.KeysView(kvLayer)
	vTensor := s.halKV.ValuesView(kvLayer)
	Kl := be.Read(kTensor)
	Vl := be.Read(vTensor)
	if len(Kl) < plan.kvLen || len(Vl) < plan.kvLen {
		return compute.Tensor{}, fmt.Errorf("model: QSA KV cache length (%d, %d) smaller than required %d", len(Kl), len(Vl), plan.kvLen)
	}

	// Scoring: top-k block scoring using representative query head dot product against mid-block keys.
	for b := 0; b < totalBlocks; b++ {
		midToken := b*blockSize + (blockSize / 2)
		if midToken >= totalTokens {
			midToken = totalTokens - 1
		}
		kMid := packedHead(Kl, midToken, w, 0, hd)
		scores[b] = fdot(q0, kMid) * scale
	}

	topKBlocks := compute.QSABaseTopKTokens / blockSize
	tailBlocks := compute.QSALocalTailTokens / blockSize
	selectedBlocks, receipt, err := compute.RadixTopKBlockSelect(scores, totalBlocks, topKBlocks, tailBlocks)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("model: RadixTopKBlockSelect: %w", err)
	}
	if receipt.DynamicGatingBypassed {
		return compute.Tensor{}, errors.New("model: QSA dynamic gating bypassed")
	}

	// Contiguous gather: tile and align gathered rows into contiguous scratch buffers
	// (2,048 top-k + 256 local tail, aligned to 256-wide tiles -> 2,304 tokens = 9 tiles).
	if len(selectedBlocks) == 0 || len(selectedBlocks) > compute.QSAMaxGatherTokens/blockSize {
		return compute.Tensor{}, errors.New("model: QSA block selection exceeds admitted gather envelope")
	}
	paddedTokens := len(selectedBlocks) * blockSize
	tileRemainder := paddedTokens % compute.QSATileSize
	if tileRemainder != 0 {
		paddedTokens += (compute.QSATileSize - tileRemainder)
	}
	neededGatherLen := paddedTokens * w
	if neededGatherLen > plan.gatherLen {
		return compute.Tensor{}, errors.New("model: QSA scratch exceeds admitted gather envelope")
	}

	// Generic grow overallocates. Keep exact backing capacities, including when
	// a session previously retained a larger gather, within the admitted bound.
	if cap(hal.qsaGatheredK) != neededGatherLen {
		hal.qsaGatheredK = make([]float32, neededGatherLen)
	}
	if cap(hal.qsaGatheredV) != neededGatherLen {
		hal.qsaGatheredV = make([]float32, neededGatherLen)
	}
	hal.qsaGatheredK = hal.qsaGatheredK[:neededGatherLen]
	hal.qsaGatheredV = hal.qsaGatheredV[:neededGatherLen]

	// Gather using compute.SparseRowGatherKVInto.
	nGatheredElements, err := compute.SparseRowGatherKVInto(
		hal.qsaGatheredK[:neededGatherLen],
		hal.qsaGatheredV[:neededGatherLen],
		Kl, Vl, selectedBlocks, blockSize, nKV, hd, totalTokens,
	)
	if err != nil {
		return compute.Tensor{}, fmt.Errorf("model: SparseRowGatherKVInto: %w", err)
	}

	gK := hal.qsaGatheredK[:nGatheredElements]
	gV := hal.qsaGatheredV[:nGatheredElements]
	nGathered := nGatheredElements / w

	// Evaluate attention over the gathered contiguous rows without full KV cache streaming.
	attnOutHost := make([]float32, nH*hd)
	hal.qsaScores = grow2D(hal.qsaScores, grp, nGathered)
	useSaxpy3SIMD := attnSaxpy3SIMDMinBatch <= 1 && nGathered >= attnSaxpy3SIMDMinPos

	for kvh := 0; kvh < nKV; kvh++ {
		if attnGQAFuse && grp == 3 {
			h0 := kvh * grp
			q0h, q1, q2 := packedHead3(qHost, 0, len(qHost), h0, hd)
			sc0, sc1, sc2 := scoreScratchHead3(hal.qsaScores, 0, 1, nGathered)
			fillSoftmaxAttentionScores3(sc0, sc1, sc2, q0h, q1, q2, gK, 0, nGathered, w, kvh, hd, scale, fdot3scalar)
		} else {
			for g := 0; g < grp; g++ {
				h := kvh*grp + g
				qh := vectorHead(qHost, h, hd)
				sc := hal.qsaScores[g][:nGathered]
				fillSoftmaxAttentionScores(sc, qh, gK, 0, nGathered, w, kvh, hd, scale, fdot)
			}
		}
		if grp == 3 {
			h0 := kvh * grp
			sc0, sc1, sc2 := scoreScratchHead3(hal.qsaScores, 0, 1, nGathered)
			accumulatePackedAttentionValues3(attnOutHost, 0, len(attnOutHost), h0, hd, gV, sc0, sc1, sc2, 0, nGathered, w, kvh, useSaxpy3SIMD)
			continue
		}
		accumulateAttentionGroup(attnOutHost, 0, len(attnOutHost), kvh*grp, grp, hd, gV, hal.qsaScores, 0, 0, nGathered, w, kvh)
	}

	hal.qsaGatherTriggered = true
	hal.lastQSAReceipt = receipt
	hal.lastQSAScratchBytes = plan.scratchBytes

	attnOut := s.uploadHostF32([]int{nH * hd}, attnOutHost, compute.MemoryActivation, "qwen35-full-attn-qsa-output")
	return attnOut, nil
}

// LastQSABlockSelectionReceipt returns the receipt of the most recent QSA sparse gather pass.
