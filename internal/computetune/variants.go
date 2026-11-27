package computetune

import (
	"context"
	"fmt"
)

// ScaledCandidate is the optional candidate contract for hidden-input parity:
// RunScaled executes the profile with the activations multiplied by scale. Weights
// stay unscaled, so resident (possibly quantized) weights are never rewritten.
type ScaledCandidate interface {
	Candidate
	RunScaled(ctx context.Context, p Profile, scale float64) ([]float32, error)
}

// variantProfiles returns up to five profiles: p itself, M-1, N-1 and K-1 where
// the dimension exceeds one, and a clamp(1,3) shape. Duplicates are dropped.
func variantProfiles(p Profile) []Profile {
	out := []Profile{p}
	add := func(m, n, k int) {
		v := p
		v.M, v.N, v.K = m, n, k
		for _, o := range out {
			if o.M == v.M && o.N == v.N && o.K == v.K {
				return
			}
		}
		out = append(out, v)
	}
	if p.M > 1 {
		add(p.M-1, p.N, p.K)
	}
	if p.N > 1 {
		add(p.M, p.N-1, p.K)
	}
	if p.K > 1 {
		add(p.M, p.N, p.K-1)
	}
	add(clamp(p.M, 1, 3), clamp(p.N, 1, 3), clamp(p.K, 1, 3))
	return out
}

func clamp(v, lo, hi int) int { return max(lo, min(v, hi)) }

// checkVariants runs the pre-timing correctness variants for one candidate. It
// returns a refusal reason (empty when admitted) and whether scaled variants were
// requested but could not be checked because the candidate or reference lacks
// ScaledCandidate. Variants the reference cannot run are ignored.
func checkVariants(ctx context.Context, p Profile, cand, ref Candidate, correct Correct, policy Policy) (reason string, unchecked bool) {
	if policy.RaggedVariants {
		for _, v := range variantProfiles(p)[1:] {
			want, err := ref.Run(ctx, v)
			if err != nil {
				continue
			}
			got, err := cand.Run(ctx, v)
			if err != nil {
				return fmt.Sprintf("incorrect at ragged variant %dx%dx%d: execution: %v", v.M, v.N, v.K, err), unchecked
			}
			if err := correct(got, want); err != nil {
				return fmt.Sprintf("incorrect at ragged variant %dx%dx%d: %v", v.M, v.N, v.K, err), unchecked
			}
		}
	}
	if len(policy.InputScales) > 0 {
		sc, okC := cand.(ScaledCandidate)
		sr, okR := ref.(ScaledCandidate)
		if !okC || !okR {
			return "", true
		}
		for _, s := range policy.InputScales {
			want, err := sr.RunScaled(ctx, p, s)
			if err != nil {
				continue
			}
			got, err := sc.RunScaled(ctx, p, s)
			if err != nil {
				return fmt.Sprintf("incorrect at input scale %g: execution: %v", s, err), unchecked
			}
			if err := correct(got, want); err != nil {
				return fmt.Sprintf("incorrect at input scale %g: %v", s, err), unchecked
			}
		}
	}
	return "", unchecked
}
