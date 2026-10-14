package compute

import (
	"fmt"
	"math"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestV41RouterTopKScoreReuse(t *testing.T) {
	t.Parallel()
	identity := func(z float32) float32 { return z }
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	negativeZero := math.Float32frombits(1 << 31)
	cases := []struct {
		name   string
		logits []float32
		bias   []float32
		k      int
		full   bool
		scale  float32
		score  func(float32) float32
		stage  uint32
	}{
		{"corrected-ties", []float32{1, 2, 3}, []float32{3, 2, 0}, 2, false, 1, identity, 0},
		{"negative-corrected", []float32{1, 2, 3}, []float32{-9, -10, -11}, 3, true, 1.5, identity, 0},
		{"signed-zero-ties", []float32{negativeZero, 0, negativeZero, 0}, nil, 4, true, 1, identity, 0},
		{"zero-full", []float32{-1000, -1000}, nil, 2, true, 1.5, nil, 0},
		{"zero-reduced", []float32{-1000, -1000}, nil, 2, false, 1.5, nil, 5},
		{"late-invalid-unselected", []float32{60, -100, nan}, nil, 1, true, 1, nil, 1},
		{"bias-before-later-logit", []float32{0, inf}, []float32{nan, 0}, 1, true, 1, nil, 3},
		{"logit-before-bias", []float32{inf}, []float32{nan}, 1, true, 1, nil, 1},
		{"score-before-bias", []float32{0}, []float32{nan}, 1, true, 1, func(float32) float32 { return inf }, 2},
		{"corrected-overflow", []float32{math.MaxFloat32}, []float32{math.MaxFloat32}, 1, true, 1, identity, 4},
		{"sum-overflow", []float32{math.MaxFloat32, math.MaxFloat32}, nil, 2, true, 1, identity, 5},
		{"partial-weight-fault", []float32{-1, 2}, []float32{10, 0}, 2, false, math.MaxFloat32, identity, 6},
	}
	for k := 1; k <= V41RouterTopKMaxK; k++ {
		for _, experts := range []int{k, 384} {
			logits, bias := make([]float32, experts), make([]float32, experts)
			var seed uint32 = 123
			for e := range logits {
				seed = seed*1664525 + 1013904223
				logits[e] = float32(int32(seed%16001)-10000) / 100
				bias[e] = float32(int32(seed%101)-50) / 10
			}
			for _, full := range []bool{false, true} {
				for _, hasBias := range []bool{false, true} {
					b := bias
					if !hasBias {
						b = nil
					}
					cases = append(cases, struct {
						name         string
						logits, bias []float32
						k            int
						full         bool
						scale        float32
						score        func(float32) float32
						stage        uint32
					}{fmt.Sprintf("experts%d/k%d/full%t/bias%t", experts, k, full, hasBias), logits, b, k, full, 1.5, nil, 0})
				}
			}
		}
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			score := tc.score
			if score == nil {
				score = routerScoreF32ForReuseTest
			}
			want, oldCalls := routerReuseModelForTest(tc.logits, tc.bias, tc.k, tc.full, tc.scale, score, false)
			got, newCalls := routerReuseModelForTest(tc.logits, tc.bias, tc.k, tc.full, tc.scale, score, true)
			if got != want || got.status[0] != tc.stage {
				t.Fatalf("reuse=%+v repeated=%+v; want stage %d", got, want, tc.stage)
			}
			if tc.stage == 0 {
				e := len(tc.logits)
				if newCalls != e || oldCalls != e+tc.k*e-tc.k*(tc.k-1)/2+2*tc.k {
					t.Fatalf("score calls reuse=%d repeated=%d", newCalls, oldCalls)
				}
				if e == 384 && tc.k == 6 {
					t.Logf("scalar score evaluations: repeated=%d reuse=%d; no GPU timing inference", oldCalls, newCalls)
				}
			}
			if tc.name == "corrected-ties" && (got.experts[0] != 0 || got.experts[1] != 1 || got.weights[0] != math.Float32bits(float32(1)/3) || got.weights[1] != math.Float32bits(float32(2)/3)) {
				t.Fatal("corrected ties must preserve expert order while weights use distinct raw scores")
			}
			if tc.name == "partial-weight-fault" && (got.status[1] != 2 || got.weights[0] != math.Float32bits(-math.MaxFloat32) || got.weights[1] != 0) {
				t.Fatal("weight fault must preserve earlier writes and leave later output zero")
			}
		})
	}
}

type routerReuseResultForTest struct {
	status  [4]uint32
	experts [V41RouterTopKMaxK]uint32
	weights [V41RouterTopKMaxK]uint32
}

// Both models use the same explicitly rounded scorer. This is an algorithm and
// control-flow witness, not execution of GLSL or a GPU transcendental-bit proof.
// Injected scores exercise status/overflow branches outside normal router scores.
func routerReuseModelForTest(logits, bias []float32, k int, full bool, scale float32, score func(float32) float32, reuse bool) (out routerReuseResultForTest, calls int) {
	finite := func(x float32) bool { return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0) }
	fault := func(stage uint32, expert int, value float32) {
		out.status = [4]uint32{stage, uint32(expert + 1), math.Float32bits(value), 0}
	}
	eval := func(e int) float32 { calls++; return score(logits[e]) }
	var picked [V41RouterTopKMaxK]int
	var values, scores [V41RouterTopKMaxK]float32
	for e, z := range logits {
		if !finite(z) {
			fault(1, e, z)
			return
		}
		s := eval(e)
		if !finite(s) {
			fault(2, e, s)
			return
		}
		c := s
		if bias != nil {
			b := bias[e]
			if !finite(b) {
				fault(3, e, b)
				return
			}
			c = float32(s + b)
			if !finite(c) {
				fault(4, e, c)
				return
			}
		}
		if reuse {
			filled := min(e, k)
			insert := filled
			for slot := 0; slot < filled; slot++ {
				if c > values[slot] {
					insert = slot
					break
				}
			}
			if insert < k {
				for slot := min(e, k-1); slot > insert; slot-- {
					picked[slot], values[slot], scores[slot] = picked[slot-1], values[slot-1], scores[slot-1]
				}
				picked[insert], values[insert], scores[insert] = e, c, s
			}
		}
	}
	if !reuse {
		// Original shader: repeat the ascending scan for each selected expert.
		for slot := 0; slot < k; slot++ {
			best, bestValue := -1, float32(0)
			for e := range logits {
				taken := false
				for j := 0; j < slot; j++ {
					taken = taken || picked[j] == e
				}
				if taken {
					continue
				}
				c := eval(e)
				if bias != nil {
					c = float32(c + bias[e])
				}
				if best < 0 || c > bestValue {
					best, bestValue = e, c
				}
			}
			picked[slot] = best
		}
	}
	selectedScore := func(slot int) float32 {
		if reuse {
			return scores[slot]
		}
		return eval(picked[slot])
	}
	sum := float32(0)
	for slot := 0; slot < k; slot++ {
		sum = float32(sum + selectedScore(slot))
	}
	if !finite(sum) || sum < 0 || (!full && sum == 0) {
		fault(5, -1, sum)
		return
	}
	denominator := sum
	if full {
		denominator = float32(sum + float32(1e-20))
	}
	for slot := 0; slot < k; slot++ {
		q := float32(selectedScore(slot) / denominator)
		w := float32(q * scale)
		if !finite(w) {
			fault(6, picked[slot], w)
			return
		}
		out.experts[slot], out.weights[slot] = uint32(picked[slot]), math.Float32bits(w)
	}
	return
}

func routerScoreF32ForReuseTest(z float32) float32 {
	if z < -20 {
		h := float32(0.5 * z)
		return float32(math.Exp(float64(h)))
	}
	a := float32(math.Abs(float64(z)))
	e := float32(math.Exp(float64(-a)))
	u := float32(1 + e)
	r := e
	if u != 1 {
		l := float32(math.Log(float64(u)))
		ratio := float32(e / float32(u-1))
		r = float32(l * ratio)
	}
	sp := float32(max(z, 0) + r)
	return float32(math.Sqrt(float64(sp)))
}
