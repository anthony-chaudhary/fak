package mtptune

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestBlockLawTargetDistribution(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		p, q [2]float64
		cap  func(int) int
	}{
		{"joint", [2]float64{1.0 / 3, 2.0 / 3}, [2]float64{2.0 / 3, 1.0 / 3}, nil},
		{"conditional-rows", [2]float64{1.0 / 3, 2.0 / 3}, [2]float64{2.0 / 3, 1.0 / 3}, nil},
		{"target-only", [2]float64{1.0 / 3, 2.0 / 3}, [2]float64{2.0 / 3, 1.0 / 3}, func(int) int { return 1 }},
		{"one-draft", [2]float64{1.0 / 3, 2.0 / 3}, [2]float64{2.0 / 3, 1.0 / 3}, func(int) int { return 2 }},
		// The second draft is excluded based only on the first included draw.
		{"predictable-cap", [2]float64{1.0 / 3, 2.0 / 3}, [2]float64{2.0 / 3, 1.0 / 3}, func(first int) int { return first + 2 }},
		{"equal", [2]float64{0.5, 0.5}, [2]float64{0.5, 0.5}, nil},
		{"disjoint-support", [2]float64{1, 0}, [2]float64{0, 1}, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := func(prefix []int) []float64 {
				if tc.name == "conditional-rows" && len(prefix) > 0 {
					if prefix[len(prefix)-1] == 0 {
						return []float64{0.5, 0.5}
					}
					return []float64{0.25, 0.75}
				}
				return tc.p[:]
			}
			proposal := func(prefix []int) []float64 {
				if tc.name == "conditional-rows" && len(prefix) > 0 {
					if prefix[0] == 0 {
						return []float64{0.25, 0.75}
					}
					return []float64{0.5, 0.5}
				}
				return tc.q[:]
			}
			var block, token [8]float64
			var blockLengths, tokenLengths [3]float64
			// Exact quadrature for these rational fixtures: every acceptance and
			// categorical boundary is a multiple of 1/12. No statistical tolerance
			// or PRNG seed can hide a distribution error.
			const grid = 12
			for a := 0; a < 2; a++ {
				for b := 0; b < 2; b++ {
					pathMass := proposal(nil)[a] * proposal([]int{a})[b]
					if pathMass == 0 {
						continue
					}
					in := BlockLawInput{Target: [][]float64{target(nil), target([]int{a}), target([]int{a, b})}, Proposal: [][]float64{proposal(nil), proposal([]int{a})}, Draft: []int{a, b}}
					n := 2
					if tc.cap != nil {
						live := tc.cap(a)
						in.LiveLength, n = &live, live-1
					}
					for u := 0; u < grid; u++ {
						for v := 0; v < grid; v++ {
							for c := 0; c < grid; c++ {
								in.AcceptUniform = []float64{(float64(u) + 0.5) / grid, (float64(v) + 0.5) / grid}
								in.CorrectionUniform = (float64(c) + 0.5) / grid
								got, err := VerifyBlock(in)
								if err != nil {
									t.Fatal(err)
								}
								ref := tokenReference(in.Target, in.Proposal, in.Draft[:n], in.AcceptUniform, in.CorrectionUniform)
								mass := pathMass / (grid * grid * grid)
								blockLengths[got.Accepted] += mass
								tokenLengths[ref.Accepted] += mass
								completeBlockLawOutput(&block, target, append(append([]int(nil), in.Draft[:got.Accepted]...), got.Correction), mass)
								completeBlockLawOutput(&token, target, append(append([]int(nil), in.Draft[:ref.Accepted]...), ref.Correction), mass)
							}
						}
					}
				}
			}
			for seq := 0; seq < 8; seq++ {
				a, b, c := seq>>2, (seq>>1)&1, seq&1
				want := target(nil)[a] * target([]int{a})[b] * target([]int{a, b})[c]
				if math.Abs(block[seq]-want) > 1e-12 || math.Abs(token[seq]-want) > 1e-12 {
					t.Fatalf("sequence %03b: block=%g token-reference=%g target=%g", seq, block[seq], token[seq], want)
				}
			}
			if tc.name == "joint" {
				// #13664's equal-length-distribution requirement is impossible.
				// The equal invariant is the completed OUTPUT law checked above.
				for i, want := range [3]float64{3.0 / 9, 1.0 / 9, 5.0 / 9} {
					if math.Abs(blockLengths[i]-want) > 1e-12 {
						t.Fatalf("block length %d: %g, want %g", i, blockLengths[i], want)
					}
				}
				for i, want := range [3]float64{3.0 / 9, 2.0 / 9, 4.0 / 9} {
					if math.Abs(tokenLengths[i]-want) > 1e-12 {
						t.Fatalf("token length %d: %g, want %g", i, tokenLengths[i], want)
					}
				}
			}
		})
	}
}

// tokenReference independently implements sequential rejection sampling; it
// neither calls the block helper nor uses its recurrence or residual weights.
func tokenReference(p, q [][]float64, draft []int, uniforms []float64, correction float64) BlockLawResult {
	n := 0
	for n < len(draft) && uniforms[n] < math.Min(1, p[n][draft[n]]/q[n][draft[n]]) {
		n++
	}
	r := p[n]
	if n < len(draft) {
		r = []float64{math.Max(p[n][0]-q[n][0], 0), math.Max(p[n][1]-q[n][1], 0)}
	}
	x := 1
	if correction*(r[0]+r[1]) < r[0] {
		x = 0
	}
	return BlockLawResult{Accepted: n, Correction: x}
}

// Complete the variable-length output to a common three-token horizon using
// direct target sampling, as required by the losslessness criterion.
func completeBlockLawOutput(out *[8]float64, target func([]int) []float64, prefix []int, mass float64) {
	if len(prefix) == 3 {
		out[prefix[0]*4+prefix[1]*2+prefix[2]] += mass
		return
	}
	for x := 0; x < 2; x++ {
		completeBlockLawOutput(out, target, append(append([]int(nil), prefix...), x), mass*target(prefix)[x])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBlockLawLiveLengthAndSaturation(t *testing.T) {
	t.Parallel()
	p, q := []float64{1.0 / 3, 2.0 / 3}, []float64{2.0 / 3, 1.0 / 3}
	for live := 1; live <= 3; live++ {
		for _, u := range []float64{0, 0.2, 0.8} {
			in := BlockLawInput{Target: [][]float64{p, p, p}, Proposal: [][]float64{q, q}, Draft: []int{1, 0}, AcceptUniform: []float64{u, u}, CorrectionUniform: u, LiveLength: &live}
			short := in
			short.Draft, short.Proposal, short.Target, short.AcceptUniform = in.Draft[:live-1], in.Proposal[:live-1], in.Target[:live], in.AcceptUniform[:live-1]
			short.LiveLength = nil
			want, err := VerifyBlock(short)
			if err != nil {
				t.Fatal(err)
			}
			// There need be no target/proposal/uniform rows for the suffix.
			in.Target, in.Proposal, in.AcceptUniform = short.Target, short.Proposal, short.AcceptUniform
			for i := live - 1; i < len(in.Draft); i++ {
				in.Draft[i] = -99
			}
			got, err := VerifyBlock(in)
			if err != nil || got != want || got.Accepted+1 > live {
				t.Fatalf("live=%d: got %+v err=%v, shorter=%+v", live, got, err, want)
			}
		}
	}
	// Ratios 2, 1/4, 1 produce weights 1, 1/4, 1/4, not 1, 1/2, 1/2.
	in := BlockLawInput{Target: [][]float64{{0.5, 0.5}, {0.125, 0.875}, {0.5, 0.5}, {0.5, 0.5}}, Proposal: [][]float64{{0.25, 0.75}, {0.5, 0.5}, {0.5, 0.5}}, Draft: []int{0, 0, 0}, AcceptUniform: []float64{0.9, 0.9, 0.4}, CorrectionUniform: 0.5}
	got, err := VerifyBlock(in)
	if err != nil || got.Accepted != 1 || got.Correction != 1 {
		t.Fatalf("recursive saturation: %+v, %v", got, err)
	}
	// Keep S separate from 1: adding it to 1 first can lose S entirely and
	// promote an approximately one-half threshold to certain acceptance.
	eps := math.Ldexp(1, -52)
	in = BlockLawInput{Target: [][]float64{{0.75 - eps/2, 0.25 + eps/2}, {0.5 + eps, 0.5 - eps}, {0.5, 0.5}}, Proposal: [][]float64{{0.75, 0.25}, {0.5, 0.5}}, Draft: []int{0, 1}, AcceptUniform: []float64{0.75, math.Nextafter(1, 0)}, CorrectionUniform: 0.5}
	got, err = VerifyBlock(in)
	if err != nil || got.Accepted != 0 || got.Correction != 1 {
		t.Fatalf("near-one threshold cancellation: %+v, %v", got, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestBlockLawRefusesInvalidSupport(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		edit func(*BlockLawInput)
	}{
		{"q-zero-p-positive", func(in *BlockLawInput) { in.Proposal[0] = []float64{0, 1} }},
		{"q-zero-p-zero", func(in *BlockLawInput) { in.Proposal[0], in.Target[0] = []float64{0, 1}, []float64{0, 1} }},
		{"nan", func(in *BlockLawInput) { in.Target[0][0] = math.NaN() }},
		{"not-normalized", func(in *BlockLawInput) { in.Target[0][0] = 0.9 }},
		{"missing-correction-row", func(in *BlockLawInput) { in.Target = in.Target[:1] }},
		{"empty-cap", func(in *BlockLawInput) { live := 0; in.LiveLength = &live }},
		{"over-cap", func(in *BlockLawInput) { live := 3; in.LiveLength = &live }},
		{"invalid-uniform", func(in *BlockLawInput) { in.AcceptUniform[0] = 1 }},
		{"invalid-correction-uniform", func(in *BlockLawInput) { in.CorrectionUniform = math.NaN() }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := BlockLawInput{Target: [][]float64{{0.5, 0.5}, {0.5, 0.5}}, Proposal: [][]float64{{0.5, 0.5}}, Draft: []int{0}, AcceptUniform: []float64{0}, CorrectionUniform: 0}
			tc.edit(&in)
			if got, err := VerifyBlock(in); err == nil {
				t.Fatalf("invalid input accepted: %+v", got)
			}
		})
	}
}
