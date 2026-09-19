package model

import (
	"fmt"
	"math"
	"sort"
)

// v4RouteError is a fail-closed admission error for the scored-layer V4
// router. Hash-routed layers are deliberately a separate seam: callers must
// supply the artifact's token-to-expert table rather than guessing from logits.
type v4RouteError struct {
	Field  string
	Reason string
}

func (e *v4RouteError) Error() string {
	return fmt.Sprintf("model: DeepSeek V4 route %s: %s", e.Field, e.Reason)
}

// v4ScoredRoute implements the score-based Gate.forward contract from the
// pinned DeepSeek-V4-Pro inference artifact. Selection uses
// sqrt(softplus(logit))+bias while returned weights use the unbiased score,
// normalized across the selected experts and multiplied by routeScale.
func v4ScoredRoute(logits, correctionBias []float32, topK int, routeScale float32) ([]routePick, error) {
	if len(logits) == 0 {
		return nil, &v4RouteError{Field: "logits", Reason: "empty"}
	}
	if len(correctionBias) != 0 && len(correctionBias) != len(logits) {
		return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("width %d does not match logits width %d", len(correctionBias), len(logits))}
	}
	if topK <= 0 || topK > len(logits) {
		return nil, &v4RouteError{Field: "top_k", Reason: fmt.Sprintf("%d outside [1,%d]", topK, len(logits))}
	}
	if !finite32(routeScale) || routeScale <= 0 {
		return nil, &v4RouteError{Field: "route_scale", Reason: fmt.Sprintf("must be finite and positive, got %g", routeScale)}
	}

	raw := make([]float32, len(logits))
	choice := make([]float32, len(logits))
	for i, z := range logits {
		if !finite32(z) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
		}
		// max(z,0)+log1p(exp(-abs(z))) is softplus without overflow.
		zf := float64(z)
		softplus := math.Max(zf, 0) + math.Log1p(math.Exp(-math.Abs(zf)))
		score := float32(math.Sqrt(softplus))
		if !finite32(score) {
			return nil, &v4RouteError{Field: "logits", Reason: fmt.Sprintf("non-finite score at expert %d", i)}
		}
		raw[i] = score
		choice[i] = score
		if len(correctionBias) != 0 {
			bias := correctionBias[i]
			if !finite32(bias) {
				return nil, &v4RouteError{Field: "correction_bias", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
			}
			choice[i] += bias
			if !finite32(choice[i]) {
				return nil, &v4RouteError{Field: "selection_score", Reason: fmt.Sprintf("non-finite value at expert %d", i)}
			}
		}
	}

	indices := v4TopKIndices(choice, topK)

	picks := make([]routePick, topK)
	var sum float32
	for i := 0; i < topK; i++ {
		expert := indices[i]
		picks[i] = routePick{expert: expert, weight: raw[expert]}
		sum += raw[expert]
	}
	if !finite32(sum) || sum <= 0 {
		return nil, &v4RouteError{Field: "normalization", Reason: fmt.Sprintf("selected score sum must be finite and positive, got %g", sum)}
	}
	for i := range picks {
		picks[i].weight = picks[i].weight / sum * routeScale
		if !finite32(picks[i].weight) {
			return nil, &v4RouteError{Field: "weight", Reason: fmt.Sprintf("non-finite result for expert %d", picks[i].expert)}
		}
	}
	return picks, nil
}

// v4BitonicTopK is the bitonic-network top-k opt-in. It is a config-surface seam
// rather than an environment read: a kernel-selection posture is behavior, not a
// credential, so it lives on the config surface (internal/envconfiglint's
// CONFIG_NOT_ENV rule; the former FAK_V4_BITONIC_TOPK env read was relocated here).
// Default (false) keeps the reference full stable sort, so the default path stays
// byte-identical to the pre-kernel router. The front door that owns this switch
// declares it with SetV4BitonicTopK.
var v4BitonicTopK bool

// SetV4BitonicTopK declares whether the bitonic-network top-k kernel is opted in.
// It is the config-surface replacement for the retired FAK_V4_BITONIC_TOPK env read.
func SetV4BitonicTopK(on bool) { v4BitonicTopK = on }

// v4BitonicTopKEnabled reports whether the bitonic-network top-k kernel is opted
// in. Default (undeclared) is OFF.
func v4BitonicTopKEnabled() bool { return v4BitonicTopK }

// v4TopKIndices returns the expert indices of the k largest selection scores,
// in the router's pinned order: descending score, lower expert index first on
// ties.
//
// With the kernel opted in it delegates to v4PartialTopKIndices, a genuine
// O(E log k) bounded-heap partial selection (fak#12975). The reference full
// stable sort remains the DEFAULT path, so the pre-kernel router stays
// byte-identical; the flag turns on the faster kernel.
//
// History: the flag was first wired to compute.PersistentBitonicTopK, which is
// NOT an O(E log k) partial select -- it is a full bitonic sorting network that
// pads E to the next power of two and sorts every slot (E=384 -> 512, ~11520
// compare-exchanges vs the reference's ~3300), so it measured ~1.7x SLOWER and
// no speedup was ever claimed for it. v4PartialTopKIndices replaces that
// equivalence-only spine with the partial selection this seam was meant to
// carry.
//
// The fallback stays fail-closed: a malformed kernel result (an out-of-range
// index or a short slice) can never select a non-existent expert or panic the
// route -- it returns the reference ordering instead.
func v4TopKIndices(choice []float32, k int) []int {
	ref := func() []int {
		indices := make([]int, len(choice))
		for i := range indices {
			indices[i] = i
		}
		sort.SliceStable(indices, func(i, j int) bool {
			return choice[indices[i]] > choice[indices[j]]
		})
		return indices
	}
	// Defense in depth: callers validate k, but a direct call with an
	// out-of-range k must not index past the reference width downstream.
	if k <= 0 || k > len(choice) {
		return ref()
	}
	if !v4BitonicTopKEnabled() {
		return ref()
	}
	indices := v4PartialTopKIndices(choice, k)
	// v4PartialTopKIndices mirrors the reference width contract, so a
	// well-formed call returns exactly k in-range indices. Re-validate anyway:
	// the seam is package-visible and a future kernel swap must not be able to
	// route a non-existent expert.
	if len(indices) != k {
		return ref()
	}
	for _, idx := range indices {
		if idx < 0 || idx >= len(choice) {
			return ref()
		}
	}
	return indices
}

func finite32(v float32) bool {
	return !float32IsNaN(v) && !float32IsInf(v)
}

// Finite32 exposes the package-internal finiteness check to the internal/model/v41
// leaf package without renaming the lowercase symbol core callers use.
func Finite32(v float32) bool { return finite32(v) }

func float32IsNaN(v float32) bool { return v != v }
func float32IsInf(v float32) bool {
	return v > math.MaxFloat32 || v < -math.MaxFloat32
}

// v4HashRouteUnsupported gives the first three V4 layers an explicit refusal
// until the pinned tid2eid table is connected to the live token path.
func v4HashRouteUnsupported(layer int) error {
	return &v4RouteError{Field: "hash_layer", Reason: fmt.Sprintf("layer %d requires token-to-expert tid2eid routing", layer)}
}
