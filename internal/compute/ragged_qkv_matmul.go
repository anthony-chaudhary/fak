package compute

import (
	"fmt"
)

// RaggedQKVBatchRequest specifies one sequence within a continuous batch.
type RaggedQKVBatchRequest struct {
	RequestID string  `json:"request_id"`
	NumTokens int     `json:"num_tokens"`
	Cancelled bool    `json:"cancelled"`
	Scale     float32 `json:"scale"`
}

// RaggedQKVBatchReceipt records execution statistics and padding elimination metrics.
type RaggedQKVBatchReceipt struct {
	BatchSize         int     `json:"batch_size"`
	TotalRaggedTokens int     `json:"total_ragged_tokens"`
	PaddedTokens      int     `json:"padded_tokens"`
	PaddingEliminated int     `json:"padding_eliminated"`
	WasteReductionPct float64 `json:"waste_reduction_pct"`
	ProcessedTokens   int     `json:"processed_tokens"`
}

// ExecuteRaggedContinuousQKVMatMul computes QKV projection outputs over a ragged packed token sequence
// in a single pass without allocating or computing padded dummy tokens.
func ExecuteRaggedContinuousQKVMatMul(
	requests []RaggedQKVBatchRequest,
	packedTokens []float32, // [totalRaggedTokens * inDim]
	weightsQKV []float32, // [outDim * inDim]
	inDim int,
	outDim int,
	outQKV []float32, // [totalRaggedTokens * outDim]
	canaryCheck func(tokenIdx int),
) (RaggedQKVBatchReceipt, error) {
	var receipt RaggedQKVBatchReceipt
	batchSize := len(requests)
	if batchSize == 0 {
		return receipt, fmt.Errorf("requests must not be empty")
	}
	if inDim <= 0 || outDim <= 0 {
		return receipt, fmt.Errorf("dimensions must be positive: inDim=%d, outDim=%d", inDim, outDim)
	}

	totalRagged := 0
	maxTokens := 0
	for _, req := range requests {
		if req.NumTokens < 0 {
			return receipt, fmt.Errorf("negative tokens in request %q", req.RequestID)
		}
		totalRagged += req.NumTokens
		if req.NumTokens > maxTokens {
			maxTokens = req.NumTokens
		}
	}

	paddedTokens := maxTokens * batchSize
	paddingEliminated := paddedTokens - totalRagged
	wastePct := 0.0
	if paddedTokens > 0 {
		wastePct = float64(paddingEliminated) / float64(paddedTokens) * 100.0
	}

	if len(packedTokens) != totalRagged*inDim {
		return receipt, fmt.Errorf("packedTokens length %d != totalRagged*inDim %d", len(packedTokens), totalRagged*inDim)
	}
	if len(weightsQKV) != outDim*inDim {
		return receipt, fmt.Errorf("weightsQKV length %d != outDim*inDim %d", len(weightsQKV), outDim*inDim)
	}
	if len(outQKV) != totalRagged*outDim {
		return receipt, fmt.Errorf("outQKV length %d != totalRagged*outDim %d", len(outQKV), totalRagged*outDim)
	}

	processed := 0
	tokenCursor := 0

	for _, req := range requests {
		scale := req.Scale
		if scale == 0 {
			scale = 1.0
		}

		for t := 0; t < req.NumTokens; t++ {
			globalTokenIdx := tokenCursor + t

			if canaryCheck != nil {
				canaryCheck(globalTokenIdx)
			}

			// If request was cancelled, zero output and skip computation
			if req.Cancelled {
				for o := 0; o < outDim; o++ {
					outQKV[globalTokenIdx*outDim+o] = 0
				}
				continue
			}

			tokInOffset := globalTokenIdx * inDim
			tokOutOffset := globalTokenIdx * outDim

			for o := 0; o < outDim; o++ {
				wOffset := o * inDim
				var dot float64
				for i := 0; i < inDim; i++ {
					dot += float64(weightsQKV[wOffset+i]) * float64(packedTokens[tokInOffset+i])
				}
				outQKV[tokOutOffset+o] = float32(dot * float64(scale))
			}
			processed++
		}
		tokenCursor += req.NumTokens
	}

	receipt = RaggedQKVBatchReceipt{
		BatchSize:         batchSize,
		TotalRaggedTokens: totalRagged,
		PaddedTokens:      paddedTokens,
		PaddingEliminated: paddingEliminated,
		WasteReductionPct: wastePct,
		ProcessedTokens:   processed,
	}

	return receipt, nil
}
