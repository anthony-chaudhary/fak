package kvmmu

import (
	"errors"
	"fmt"
	"math"
)

// ObservationWindowShape describes row-major [head][position][dim] query and
// key buffers for one prefill. Queries and keys cover the same positions.
type ObservationWindowShape struct {
	QueryHeads int
	// KVHeads may be smaller than QueryHeads (grouped-query attention); each
	// KV head serves QueryHeads/KVHeads consecutive query heads.
	KVHeads int
	SeqLen  int
	HeadDim int
	// Window is the number of trailing prompt queries that vote.
	Window int
}

// ObservationWindowAttentionMass scores each prefix key position (those before
// the trailing Window queries) by the attention mass the window queries place
// on it. Scores are returned per query head with no cross-head mixing:
// out[h][p] for p in [0, SeqLen-Window).
//
// Each window query row is softmaxed over all SeqLen keys after scaling by
// 1/sqrt(HeadDim); within the window block a causal mask hides keys after the
// query, while prefix keys are always visible. Pooling and top-k selection are
// left to the caller.
func ObservationWindowAttentionMass(queries, keys []float32, s ObservationWindowShape) ([][]float32, error) {
	if s.QueryHeads <= 0 || s.KVHeads <= 0 || s.SeqLen <= 0 || s.HeadDim <= 0 {
		return nil, errors.New("kvmmu: observation window shape dimensions must be positive")
	}
	if s.QueryHeads%s.KVHeads != 0 {
		return nil, fmt.Errorf("kvmmu: query heads %d not a multiple of kv heads %d", s.QueryHeads, s.KVHeads)
	}
	if s.Window <= 0 || s.Window > s.SeqLen {
		return nil, fmt.Errorf("kvmmu: observation window %d outside (0, %d]", s.Window, s.SeqLen)
	}
	if want := s.QueryHeads * s.SeqLen * s.HeadDim; len(queries) != want {
		return nil, fmt.Errorf("kvmmu: queries len %d, want %d", len(queries), want)
	}
	if want := s.KVHeads * s.SeqLen * s.HeadDim; len(keys) != want {
		return nil, fmt.Errorf("kvmmu: keys len %d, want %d", len(keys), want)
	}

	group := s.QueryHeads / s.KVHeads
	prefix := s.SeqLen - s.Window
	scale := float32(1 / math.Sqrt(float64(s.HeadDim)))
	stride := s.SeqLen * s.HeadDim
	row := make([]float32, s.SeqLen)
	out := make([][]float32, s.QueryHeads)

	for h := 0; h < s.QueryHeads; h++ {
		qh := queries[h*stride : (h+1)*stride]
		kh := keys[(h/group)*stride : (h/group+1)*stride]
		acc := make([]float32, prefix)
		for i := 0; i < s.Window; i++ {
			qPos := prefix + i
			q := qh[qPos*s.HeadDim : (qPos+1)*s.HeadDim]
			// Keys past qPos are masked; upstream adds finfo.min, which
			// softmax maps to exactly zero, so they are skipped outright.
			visible := qPos + 1
			maxScore := float32(math.Inf(-1))
			for j := 0; j < visible; j++ {
				k := kh[j*s.HeadDim : (j+1)*s.HeadDim]
				var dot float32
				for d := range q {
					dot += q[d] * k[d]
				}
				row[j] = dot * scale
				if row[j] > maxScore {
					maxScore = row[j]
				}
			}
			var denom float64
			for j := 0; j < visible; j++ {
				e := math.Exp(float64(row[j] - maxScore))
				row[j] = float32(e)
				denom += e
			}
			for j := 0; j < prefix; j++ {
				acc[j] += float32(float64(row[j]) / denom)
			}
		}
		out[h] = acc
	}
	return out, nil
}
