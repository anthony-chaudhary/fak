//go:build !(darwin && arm64 && cgo)

package metalgemm

import (
	"errors"
	"fmt"
	"math"
)

// FlashAttention2Config configures the Metal 4 FlashAttention-2 execution.
type FlashAttention2Config struct {
	NumQueryHeads int     `json:"num_query_heads"`
	NumKVHeads    int     `json:"num_kv_heads"`
	HeadDim       int     `json:"head_dim"`
	Scale         float32 `json:"scale"`
	Causal        bool    `json:"causal"`
	SlidingWindow int     `json:"sliding_window,omitempty"`
	Batch         int     `json:"batch,omitempty"`
}

// FlashAttention2MemoryStats records memory metrics.
type FlashAttention2MemoryStats struct {
	QTokens                int     `json:"q_tokens"`
	KVTokens               int     `json:"kv_tokens"`
	HeadDim                int     `json:"head_dim"`
	NumHeads               int     `json:"num_heads"`
	NumKVHeads             int     `json:"num_kv_heads"`
	IntermediateBytes      int64   `json:"intermediate_bytes"`
	InputOutputBytes       int64   `json:"input_output_bytes"`
	NaiveIntermediateBytes int64   `json:"naive_intermediate_bytes"`
	MemorySavingsRatio     float64 `json:"memory_savings_ratio"`
	AllocatedBytes         int64   `json:"allocated_bytes"`
}

// FlashAttention2Result encapsulates attention output and stats.
type FlashAttention2Result struct {
	Output []float32                  `json:"output"`
	LSE    []float32                  `json:"lse"`
	Stats  FlashAttention2MemoryStats `json:"stats"`
}

// MetalFlashAttention2Available returns false on non-Metal platforms.
func MetalFlashAttention2Available() bool {
	return false
}

// ShouldDispatchFlashAttention returns whether sequence length exceeds 256 tokens.
func ShouldDispatchFlashAttention(seqLen int) bool {
	return seqLen > 256
}

// FlashAttention2MemoryProfile calculates memory scaling metrics.
func FlashAttention2MemoryProfile(cfg FlashAttention2Config, qTokens, kvTokens int) FlashAttention2MemoryStats {
	batch := cfg.Batch
	if batch <= 0 {
		batch = 1
	}
	nH := cfg.NumQueryHeads
	nKV := cfg.NumKVHeads
	if nKV <= 0 {
		nKV = nH
	}
	hd := cfg.HeadDim

	qBytes := int64(batch) * int64(qTokens) * int64(nH) * int64(hd) * 4
	kBytes := int64(batch) * int64(kvTokens) * int64(nKV) * int64(hd) * 4
	vBytes := int64(batch) * int64(kvTokens) * int64(nKV) * int64(hd) * 4
	outBytes := int64(batch) * int64(qTokens) * int64(nH) * int64(hd) * 4
	lseBytes := int64(batch) * int64(qTokens) * int64(nH) * 4

	ioBytes := qBytes + kBytes + vBytes + outBytes + lseBytes
	naiveBytes := int64(batch) * int64(nH) * int64(qTokens) * int64(kvTokens) * 4

	var ratio float64
	if ioBytes > 0 {
		ratio = float64(naiveBytes) / float64(ioBytes)
	}

	return FlashAttention2MemoryStats{
		QTokens:                qTokens,
		KVTokens:               kvTokens,
		HeadDim:                hd,
		NumHeads:               nH,
		NumKVHeads:             nKV,
		IntermediateBytes:      0,
		InputOutputBytes:       ioBytes,
		NaiveIntermediateBytes: naiveBytes,
		MemorySavingsRatio:     ratio,
		AllocatedBytes:         ioBytes,
	}
}

// ExecuteMetalFlashAttention2 returns an error on non-Metal platforms.
func ExecuteMetalFlashAttention2(cfg FlashAttention2Config, q, k, v []float32, qTokens, kvTokens int) (*FlashAttention2Result, error) {
	return nil, errors.New("metalgemm: Metal 4 FlashAttention-2 is only available on darwin arm64 with cgo")
}

// ReferenceSDPA computes reference scaled dot-product attention on CPU.
func ReferenceSDPA(cfg FlashAttention2Config, q, k, v []float32, qTokens, kvTokens int) ([]float32, []float32, error) {
	batch := cfg.Batch
	if batch <= 0 {
		batch = 1
	}
	if cfg.NumQueryHeads <= 0 {
		return nil, nil, errors.New("metalgemm: NumQueryHeads must be positive")
	}
	if cfg.NumKVHeads <= 0 {
		cfg.NumKVHeads = cfg.NumQueryHeads
	}
	if cfg.NumQueryHeads%cfg.NumKVHeads != 0 {
		return nil, nil, fmt.Errorf("metalgemm: NumQueryHeads (%d) not divisible by NumKVHeads (%d)",
			cfg.NumQueryHeads, cfg.NumKVHeads)
	}
	if cfg.HeadDim <= 0 {
		return nil, nil, errors.New("metalgemm: HeadDim must be positive")
	}

	nH := cfg.NumQueryHeads
	nKV := cfg.NumKVHeads
	hd := cfg.HeadDim
	scale := cfg.Scale
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(hd)))
	}
	grp := nH / nKV

	expectedQ := batch * qTokens * nH * hd
	expectedKV := batch * kvTokens * nKV * hd
	if len(q) < expectedQ || len(k) < expectedKV || len(v) < expectedKV {
		return nil, nil, errors.New("metalgemm: slice bounds insufficient for reference SDPA")
	}

	out := make([]float32, expectedQ)
	lse := make([]float32, batch*qTokens*nH)

	for b := 0; b < batch; b++ {
		qBatchOffset := b * (qTokens * nH * hd)
		kvBatchOffset := b * (kvTokens * nKV * hd)

		for qi := 0; qi < qTokens; qi++ {
			globalQPos := qi
			if kvTokens >= qTokens {
				globalQPos = (kvTokens - qTokens) + qi
			}

			for h := 0; h < nH; h++ {
				kvh := h / grp
				qOffset := qBatchOffset + qi*(nH*hd) + h*hd
				qRow := q[qOffset : qOffset+hd]

				maxScore := float32(math.Inf(-1))
				scores := make([]float32, kvTokens)
				for kj := 0; kj < kvTokens; kj++ {
					if cfg.Causal && kj > globalQPos {
						scores[kj] = float32(math.Inf(-1))
						continue
					}
					if cfg.SlidingWindow > 0 && kj+cfg.SlidingWindow <= globalQPos {
						scores[kj] = float32(math.Inf(-1))
						continue
					}

					kOffset := kvBatchOffset + kj*(nKV*hd) + kvh*hd
					kRow := k[kOffset : kOffset+hd]

					var dot float32
					for d := 0; d < hd; d++ {
						dot += qRow[d] * kRow[d]
					}
					s := dot * scale
					scores[kj] = s
					if s > maxScore {
						maxScore = s
					}
				}

				var sumExp float32
				weights := make([]float32, kvTokens)
				if maxScore > float32(math.Inf(-1)) {
					for kj := 0; kj < kvTokens; kj++ {
						if scores[kj] > float32(math.Inf(-1)) {
							w := float32(math.Exp(float64(scores[kj] - maxScore)))
							weights[kj] = w
							sumExp += w
						}
					}
				}

				outOffset := qBatchOffset + qi*(nH*hd) + h*hd
				lseIdx := b*(qTokens*nH) + qi*nH + h
				if sumExp > 0 {
					lse[lseIdx] = maxScore + float32(math.Log(float64(sumExp)))
					invSum := 1.0 / sumExp
					for d := 0; d < hd; d++ {
						var acc float32
						for kj := 0; kj < kvTokens; kj++ {
							if weights[kj] > 0 {
								vOffset := kvBatchOffset + kj*(nKV*hd) + kvh*hd
								acc += (weights[kj] * invSum) * v[vOffset+d]
							}
						}
						out[outOffset+d] = acc
					}
				} else {
					lse[lseIdx] = float32(math.Inf(-1))
				}
			}
		}
	}
	return out, lse, nil
}
