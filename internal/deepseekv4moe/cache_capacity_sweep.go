package deepseekv4moe

import (
	"errors"
	"fmt"
)

// ExpertCacheCapacitySweepSchema identifies the JSON shape of an
// ExpertCacheCapacitySweep. Bump it when the schema changes incompatibly.
const ExpertCacheCapacitySweepSchema = "fak.moe-expert-cache-capacity-sweep/v1"

// ErrInvalidCapacitySweep means the sweep's arguments are outside the domain the
// deterministic capacity sweep can model: an empty route trace, empty or
// non-ascending/non-positive capacity steps, a non-positive expert-group byte
// size, or an invalid model geometry.
var ErrInvalidCapacitySweep = errors.New("deepseekv4moe: invalid expert-cache capacity sweep")

// ExpertCacheCapacityPoint is one row of the V4.1 routed-expert capacity sweep.
// Every field is deterministic control-plane evidence produced by the weight-free
// LRU model: it reports simulated residency behavior, never measured bandwidth,
// latency, or throughput.
type ExpertCacheCapacityPoint struct {
	CacheGiB              int     `json:"cache_gib"`
	CacheBytes            int64   `json:"cache_bytes"`
	CacheGroups           int64   `json:"cache_groups"`
	Accesses              int64   `json:"accesses"`
	Hits                  int64   `json:"hits"`
	HitRate               float64 `json:"hit_rate"`
	MarginalHitRatePerGiB float64 `json:"marginal_hit_rate_per_gib"`
}

// ExpertCacheGeometry names the model shape a sweep runs at.
type ExpertCacheGeometry struct {
	Name     string `json:"name"`
	Layers   int    `json:"layers"`
	Experts  int    `json:"experts"`
	TopK     int    `json:"top_k"`
	MoEInter int    `json:"moe_intermediate_size"`
	Hidden   int    `json:"hidden_size"`
}

// ExpertCacheCapacitySweep is the full deterministic capacity sweep result. Label
// is always "SW-VERIFIED": these rows are software-model evidence only.
type ExpertCacheCapacitySweep struct {
	Schema         string                     `json:"schema"`
	Label          string                     `json:"label"`
	Geometry       ExpertCacheGeometry        `json:"geometry"`
	GroupBytes     int64                      `json:"group_bytes"`
	Points         []ExpertCacheCapacityPoint `json:"points"`
	KneeGiB        int                        `json:"knee_gib"`
	KneeHitRate    float64                    `json:"knee_hit_rate"`
	MarginalAtKnee float64                    `json:"marginal_hit_rate_per_gib_at_knee"`
}

// V41Geometry returns the DeepSeek V4.1 Flash routed-expert geometry: 40 MoE
// layers, 384 routed experts per layer, top-6 routing, a 2304-wide MoE
// intermediate, and a 5120-wide hidden state.
func V41Geometry() ExpertCacheGeometry {
	return ExpertCacheGeometry{
		Name:     "deepseek-v4.1-flash",
		Layers:   40,
		Experts:  384,
		TopK:     6,
		MoEInter: 2304,
		Hidden:   5120,
	}
}

// V41ExpertGroupBytes returns the byte size of one (layer, expert) routed weight
// group at the given bytes-per-weight. Each group holds 3 matrices (gate, up,
// down) of Hidden x MoEInter params; at bytesPerWeight == 0.5 this is FP4
// packing (4 bits per parameter), yielding 17,694,720 bytes (16.875 MiB). A
// non-positive bytesPerWeight returns 0.
func V41ExpertGroupBytes(bytesPerWeight float64) int64 {
	if bytesPerWeight <= 0 {
		return 0
	}
	geom := V41Geometry()
	params := int64(3) * int64(geom.Hidden) * int64(geom.MoEInter)
	if params <= 0 {
		return 0
	}
	return int64(float64(params) * bytesPerWeight)
}

// GenerateV41CapacityTrace builds a deterministic route trace at the given
// geometry: tokensPerLayer synthetic decode tokens, each selecting TopK experts
// per layer by the package mix() hash. The result is a pure function of
// (geom, tokensPerLayer, skew) - there is no randomness and no clock.
//
// Access order is TOKEN-MAJOR, LAYER-MINOR: for each token the trace appends one
// route for every layer 0..Layers-1 before advancing to the next token. That
// mirrors real decode, where a single step walks every MoE layer in sequence
// before the next token arrives, so the cache's live working set is the union
// over recent TOKENS of their per-layer top-K groups. A layer-major trace (all
// tokens of layer 0, then layer 1, ...) instead keeps the working set confined
// to a single layer's expert pool at a time, which makes a global LRU saturate
// at a capacity-independent hit rate and yields a flat, uninformative sweep.
//
// score(t,l,e) is mix(mix(t*7+1, l+1), e*13+1)>>11 / 2^53, plus
// skew*(Experts-e)/Experts. The score varies with BOTH token and layer, so
// skew > 0 biases routing toward
// low-index experts, and skew == 0 is near-uniform. Ties break toward the lower
// expert index. Each route carries exactly TopK unique experts, as
// SimulateExpertCache requires. Top-K is selected by a linear scan (not a full
// sort) to keep generation cheap for the full 4096-token witness trace.
func GenerateV41CapacityTrace(geom ExpertCacheGeometry, tokensPerLayer int, skew float64) []ExpertRoute {
	if geom.Layers <= 0 || geom.Experts <= 0 || geom.TopK <= 0 || geom.TopK > geom.Experts || tokensPerLayer <= 0 {
		return nil
	}
	invExperts := 1.0 / float64(geom.Experts)
	scale := 1.0 / float64(int64(1)<<53)
	scores := make([]float64, geom.Experts)
	chosen := make([]int, geom.TopK)
	routes := make([]ExpertRoute, 0, geom.Layers*tokensPerLayer)
	for t := 0; t < tokensPerLayer; t++ {
		tokenSeed := uint64(t)*7 + 1
		for layer := 0; layer < geom.Layers; layer++ {
			seed := mix(tokenSeed, uint64(layer)+1)
			for e := 0; e < geom.Experts; e++ {
				base := float64(mix(seed, uint64(e)*13+1)>>11) * scale
				bias := skew * float64(geom.Experts-e) * invExperts
				scores[e] = base + bias
			}
			topKNoSort(scores, chosen)
			experts := make([]int, geom.TopK)
			copy(experts, chosen)
			routes = append(routes, ExpertRoute{Layer: layer, Experts: experts})
		}
	}
	return routes
}

// topKNoSort fills out[0:len(out)] with the indices of the len(out) highest
// scores from scores, in descending order, without sorting the whole slice. For
// each rank it linearly scans every expert that has not already been chosen,
// picking the maximum score and tie-breaking toward the lower expert index. This
// is O(Experts*TopK) instead of O(Experts*log(Experts)) per route and keeps the
// TopK experts unique.
func topKNoSort(scores []float64, out []int) {
	n := len(scores)
	k := len(out)
	for r := 0; r < k; r++ {
		best := -1
		var bestScore float64
		for e := 0; e < n; e++ {
			if alreadyChosen(out, r, e) {
				continue
			}
			if best < 0 || scores[e] > bestScore {
				best = e
				bestScore = scores[e]
			}
		}
		out[r] = best
	}
}

// alreadyChosen reports whether expert e already appears in out[0:filled].
func alreadyChosen(out []int, filled, e int) bool {
	for i := 0; i < filled; i++ {
		if out[i] == e {
			return true
		}
	}
	return false
}

// SweepExpertCacheCapacity runs the trace through SimulateExpertCache at each
// capacity in gibSteps and returns the sweep plus the marginal hit rate per GiB
// at each step and a heuristic knee.
//
// GiB is 1024^3 bytes. For step g the capacity is g*1024^3 bytes and
// CacheGroups = capacityBytes/groupBytes, capped at Layers*Experts (the full
// routed pool). A step whose CacheGroups would be below 1 is an error. gibSteps
// must be non-empty, strictly ascending, and all >= 1.
//
// MarginalHitRatePerGiB is 0 for the first point and
// (rate_i - rate_{i-1})/(gib_i - gib_{i-1}) thereafter, using the ACTUAL
// requested GiB values. When a step's CacheGroups caps at the model size its
// marginal naturally falls to 0, which is expected and fine.
//
// Honest finding for this synthetic trace: there is NO sharp in-band knee in the
// 48..96 GiB band. On the token-major, near-uniform (skew == 0) trace every
// added GiB keeps buying hit rate at a gently declining marginal, so the curve
// rises smoothly and the last swept step is not a sudden saturation point. The
// 72 GiB "turnover" sometimes discussed for the real model is page-cache
// behavior of the host and is OUT OF SCOPE for this weight-free LRU model; it is
// not represented here and must not be faked.
//
// Knee rule (a stated heuristic, not a measured fact): the knee is the LAST
// step index whose MarginalHitRatePerGiB is >= 0.5 * mean(marginal over all
// steps i > 0). On a smoothly diminishing curve this returns the last step
// clearing the threshold, i.e. it marks where marginal has fallen to half the
// band's average - it does NOT imply a saturation knee. When every step is above
// the threshold the knee is the final step; when just one step is supplied the
// knee is that step with a zero marginal. KneeGiB/KneeHitRate/MarginalAtKnee
// report that step. Because the mean depends only on the requested steps, the
// knee is deterministic.
func SweepExpertCacheCapacity(geom ExpertCacheGeometry, trace []ExpertRoute, gibSteps []int, groupBytes int64) (ExpertCacheCapacitySweep, error) {
	if len(trace) == 0 {
		return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: empty trace", ErrInvalidCapacitySweep)
	}
	if len(gibSteps) == 0 {
		return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: empty gib steps", ErrInvalidCapacitySweep)
	}
	if groupBytes <= 0 {
		return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: group bytes must be positive", ErrInvalidCapacitySweep)
	}
	if geom.Layers <= 0 || geom.Experts <= 0 || geom.TopK <= 0 || geom.TopK > geom.Experts {
		return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: invalid geometry", ErrInvalidCapacitySweep)
	}

	modelGroups := int64(geom.Layers) * int64(geom.Experts)
	const bytesPerGiB = int64(1024) * 1024 * 1024
	points := make([]ExpertCacheCapacityPoint, 0, len(gibSteps))
	prevGiB := 0
	for i, gib := range gibSteps {
		if gib < 1 {
			return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: gib step %d must be positive", ErrInvalidCapacitySweep, gib)
		}
		if i > 0 && gib <= prevGiB {
			return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: gib steps must be strictly ascending", ErrInvalidCapacitySweep)
		}
		cacheBytes := int64(gib) * bytesPerGiB
		cacheGroups := cacheBytes / groupBytes
		if cacheGroups < 1 {
			return ExpertCacheCapacitySweep{}, fmt.Errorf("%w: capacity %d GiB holds no expert groups", ErrInvalidCacheBudget, gib)
		}
		if cacheGroups > modelGroups {
			cacheGroups = modelGroups
		}
		result, err := SimulateExpertCache(trace, int(cacheGroups), geom.Layers, geom.Experts, geom.TopK)
		if err != nil {
			return ExpertCacheCapacitySweep{}, err
		}
		accesses := result.Hits + result.PageIns
		var hitRate float64
		if accesses > 0 {
			hitRate = float64(result.Hits) / float64(accesses)
		}
		var marginal float64
		if i > 0 {
			marginal = (hitRate - points[i-1].HitRate) / float64(gib-prevGiB)
		}
		points = append(points, ExpertCacheCapacityPoint{
			CacheGiB:              gib,
			CacheBytes:            cacheBytes,
			CacheGroups:           cacheGroups,
			Accesses:              accesses,
			Hits:                  result.Hits,
			HitRate:               hitRate,
			MarginalHitRatePerGiB: marginal,
		})
		prevGiB = gib
	}

	knee := 0
	if len(points) > 1 {
		var sum float64
		for i := 1; i < len(points); i++ {
			sum += points[i].MarginalHitRatePerGiB
		}
		threshold := 0.5 * sum / float64(len(points)-1)
		knee = len(points) - 1
		for i := 1; i < len(points); i++ {
			if points[i].MarginalHitRatePerGiB >= threshold {
				knee = i
			}
		}
	}

	return ExpertCacheCapacitySweep{
		Schema:         ExpertCacheCapacitySweepSchema,
		Label:          "SW-VERIFIED",
		Geometry:       geom,
		GroupBytes:     groupBytes,
		Points:         points,
		KneeGiB:        points[knee].CacheGiB,
		KneeHitRate:    points[knee].HitRate,
		MarginalAtKnee: points[knee].MarginalHitRatePerGiB,
	}, nil
}
