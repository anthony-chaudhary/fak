package compute

import (
	"math"
	"testing"
)

// v41RouterReferenceForTest is the host oracle for the device routers: the
// model package's v41RouteWithNormalization per row, or its table form when
// hash is non-nil (bias ignored). stage is the first fault stage, 0 on success.
func v41RouterReferenceForTest(logits, bias []float32, hash []int32, rows, experts, topK int, scale float32, full bool) (ids []int32, weights []float32, stage V41RouterTopKStage) {
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	for row := 0; row < rows; row++ {
		z := logits[row*experts : (row+1)*experts]
		scores := make([]float32, experts)
		var pick []int
		var choice []float32
		for e, v := range z {
			if !finite(v) {
				return nil, nil, V41RouterTopKStageLogit
			}
			zf := float64(v)
			if full {
				scores[e] = float32(math.Sqrt(float64(float32(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))))))
			} else {
				scores[e] = float32(math.Sqrt(math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))))
			}
			if !finite(scores[e]) {
				return nil, nil, V41RouterTopKStageScore
			}
			if hash != nil {
				continue
			}
			c := scores[e]
			if bias != nil {
				if !finite(bias[e]) {
					return nil, nil, V41RouterTopKStageBias
				}
				c += bias[e]
				if !finite(c) {
					return nil, nil, V41RouterTopKStageSelection
				}
			}
			at := len(pick)
			for at > 0 && c > choice[at-1] {
				at--
			}
			if at < topK {
				pick = append(pick[:at], append([]int{e}, pick[at:]...)...)
				choice = append(choice[:at], append([]float32{c}, choice[at:]...)...)
				if len(pick) > topK {
					pick, choice = pick[:topK], choice[:topK]
				}
			}
		}
		if hash != nil {
			pick = pick[:0]
			for _, id := range hash[row*topK : (row+1)*topK] {
				pick = append(pick, int(id))
			}
		}
		var sum float32
		for _, e := range pick {
			sum += scores[e]
		}
		if !finite(sum) || sum < 0 || (!full && sum == 0) {
			return nil, nil, V41RouterTopKStageNormalization
		}
		den := sum
		if full {
			den = sum + float32(1e-20)
		}
		for _, e := range pick {
			w := scores[e] / den * scale
			if !finite(w) {
				return nil, nil, V41RouterTopKStageWeight
			}
			ids = append(ids, int32(e))
			weights = append(weights, w)
		}
	}
	return ids, weights, V41RouterTopKStageSuccess
}

// fak-test:runtime fast est=5ms lane=default
func TestV41RouterHashContract(t *testing.T) {
	logits := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{2, 8}}
	good := []int32{0, 7, 3, 1, 2, 6}
	if lb, pb, sb, err := validateV41RouterHash(logits, good, 2, 8, 3, 1.5); err != nil || lb != 64 || pb != 24 || sb != 32 {
		t.Fatalf("valid hash geometry = %d %d %d %v", lb, pb, sb, err)
	}
	for name, tc := range map[string]struct {
		ids          []int32
		rows, topK   int
		scale        float32
		shape        []int
		dtype        Dtype
		wantRejected bool
	}{
		"out-of-range":  {[]int32{0, 8, 3, 1, 2, 6}, 2, 3, 1.5, []int{2, 8}, F32, true},
		"negative":      {[]int32{0, -1, 3, 1, 2, 6}, 2, 3, 1.5, []int{2, 8}, F32, true},
		"repeat-in-row": {[]int32{0, 7, 0, 1, 2, 6}, 2, 3, 1.5, []int{2, 8}, F32, true},
		"short-table":   {good[:5], 2, 3, 1.5, []int{2, 8}, F32, true},
		"topk-over-max": {make([]int32, 18), 2, 9, 1.5, []int{2, 8}, F32, true},
		"zero-scale":    {good, 2, 3, 0, []int{2, 8}, F32, true},
		"nan-scale":     {good, 2, 3, float32(math.NaN()), []int{2, 8}, F32, true},
		"wrong-shape":   {good, 2, 3, 1.5, []int{16}, F32, true},
		"wrong-dtype":   {good, 2, 3, 1.5, []int{2, 8}, F16, true},
		"repeat-across": {[]int32{0, 7, 3, 0, 7, 3}, 2, 3, 1.5, []int{2, 8}, F32, false},
	} {
		in := Tensor{Dtype: tc.dtype, Layout: RowMajor, Shape: tc.shape}
		_, _, _, err := validateV41RouterHash(in, tc.ids, tc.rows, 8, tc.topK, tc.scale)
		if (err != nil) != tc.wantRejected {
			t.Errorf("%s: err=%v, want rejected=%v", name, err, tc.wantRejected)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestV41RouterReferenceForTestOrdersAndNormalizes(t *testing.T) {
	// Equal corrected values keep the lower expert first; weights use the
	// unbiased scores; the hash form ignores bias and keeps table order.
	logits := []float32{0, 0, 2, 0}
	bias := []float32{1, 1, -5, 0}
	ids, w, stage := v41RouterReferenceForTest(logits, bias, nil, 1, 4, 2, 1, false)
	if stage != 0 || len(ids) != 2 || ids[0] != 0 || ids[1] != 1 || w[0] != w[1] || w[0] != 0.5 {
		t.Fatalf("scored reference = %v %v %v", ids, w, stage)
	}
	ids, _, stage = v41RouterReferenceForTest(logits, bias, []int32{3, 2}, 1, 4, 2, 1, false)
	if stage != 0 || ids[0] != 3 || ids[1] != 2 {
		t.Fatalf("hash reference = %v %v", ids, stage)
	}
	if _, _, stage = v41RouterReferenceForTest([]float32{-1000, -1000}, nil, nil, 1, 2, 2, 1, false); stage != V41RouterTopKStageNormalization {
		t.Fatalf("zero reduced sum stage = %v", stage)
	}
	if _, _, stage = v41RouterReferenceForTest([]float32{-1000, -1000}, nil, nil, 1, 2, 2, 1, true); stage != 0 {
		t.Fatalf("zero full sum stage = %v", stage)
	}
}
