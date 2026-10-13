package deepseekv4moe

import (
	"errors"
	"math"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/replayoracle"
)

// ErrInvalidVerifyPolicy indicates SimulateVerifyUnion was asked for an offline oracle arm.
var ErrInvalidVerifyPolicy = errors.New("deepseekv4moe: verify-union replay needs an online cache policy")

// VerifyUnionLayer is the distinct-expert count one layer touches per verify window.
type VerifyUnionLayer struct {
	Layer int     `json:"layer"`
	Mean  float64 `json:"mean"`
	P50   int     `json:"p50"`
	P95   int     `json:"p95"`
}

// VerifyUnionReport compares routed-expert ring misses of one batched (K+1)-token verify
// step against K+1 sequential decode steps over the same routed tokens. It is a
// simulated control-plane count: no transfer latency, GPU time, or acceptance is modeled,
// and every window assumes all K drafts are accepted (non-overlapping windows).
type VerifyUnionReport struct {
	Window            int                `json:"window"`
	Windows           int                `json:"windows"`
	Policy            CachePolicy        `json:"policy"`
	Capacity          int                `json:"capacity"`
	UnionMean         float64            `json:"union_mean"`
	UnionP50          int                `json:"union_p50"`
	UnionP95          int                `json:"union_p95"`
	Layers            []VerifyUnionLayer `json:"layers"`
	SequentialMisses  int                `json:"sequential_misses"`
	VerifyMisses      int                `json:"verify_misses"`
	SequentialPerStep float64            `json:"sequential_misses_per_window"`
	VerifyPerStep     float64            `json:"verify_misses_per_window"`
}

// SimulateVerifyUnion slices each decode sequence (tokens in order, each token its
// per-layer routes) into non-overlapping windows of k+1 tokens; a trailing partial window
// is dropped and windows never cross sequences. The sequential stream replays every
// windowed token layer by layer; the verify stream touches, per window and layer, the
// sorted union of that layer's experts once. Both replay through the same online arm at
// the same capacity, and ring state carries across sequences in order.
func SimulateVerifyUnion(sequences [][][]ExpertRoute, k, capacity, layers, expertsPerLayer, topK int, policy CachePolicy) (VerifyUnionReport, error) {
	if capacity <= 0 {
		return VerifyUnionReport{}, ErrInvalidTraceCapacity
	}
	if k < 0 || layers <= 0 || expertsPerLayer <= 0 || topK <= 0 || topK > expertsPerLayer {
		return VerifyUnionReport{}, ErrInvalidTraceShape
	}
	if policy == CachePolicyOPT || policy.String() == "unknown" {
		return VerifyUnionReport{}, ErrInvalidVerifyPolicy
	}
	window := k + 1
	report := VerifyUnionReport{Window: window, Policy: policy, Capacity: capacity}
	var sequential, verify []ExpertGroup
	perLayer := make([][]int, layers)
	var all []int

	for _, tokens := range sequences {
		for start := 0; start+window <= len(tokens); start += window {
			union := make([]map[int]struct{}, layers)
			for _, routes := range tokens[start : start+window] {
				for _, route := range routes {
					if err := validateVerifyRoute(route, layers, expertsPerLayer, topK); err != nil {
						return VerifyUnionReport{}, err
					}
					if union[route.Layer] == nil {
						union[route.Layer] = make(map[int]struct{}, window*topK)
					}
					for _, expert := range route.Experts {
						sequential = append(sequential, ExpertGroup{Layer: route.Layer, Expert: expert})
						union[route.Layer][expert] = struct{}{}
					}
				}
			}
			for layer, set := range union {
				if set == nil {
					continue
				}
				experts := make([]int, 0, len(set))
				for expert := range set {
					experts = append(experts, expert)
				}
				sort.Ints(experts)
				for _, expert := range experts {
					verify = append(verify, ExpertGroup{Layer: layer, Expert: expert})
				}
				perLayer[layer] = append(perLayer[layer], len(experts))
				all = append(all, len(experts))
			}
			report.Windows++
		}
	}

	report.SequentialMisses = cachePolicyReplayArm(policy, sequential, capacity, nil, replayoracle.Result{}).Misses
	report.VerifyMisses = cachePolicyReplayArm(policy, verify, capacity, nil, replayoracle.Result{}).Misses
	if report.Windows > 0 {
		report.SequentialPerStep = float64(report.SequentialMisses) / float64(report.Windows)
		report.VerifyPerStep = float64(report.VerifyMisses) / float64(report.Windows)
	}
	report.UnionMean, report.UnionP50, report.UnionP95 = unionStats(all)
	for layer, sizes := range perLayer {
		if len(sizes) == 0 {
			continue
		}
		mean, p50, p95 := unionStats(sizes)
		report.Layers = append(report.Layers, VerifyUnionLayer{Layer: layer, Mean: mean, P50: p50, P95: p95})
	}
	return report, nil
}

func validateVerifyRoute(route ExpertRoute, layers, expertsPerLayer, topK int) error {
	if route.Layer < 0 || route.Layer >= layers || len(route.Experts) != topK {
		return ErrInvalidTraceRoute
	}
	seen := make(map[int]struct{}, topK)
	for _, expert := range route.Experts {
		if expert < 0 || expert >= expertsPerLayer {
			return ErrInvalidTraceRoute
		}
		if _, duplicate := seen[expert]; duplicate {
			return ErrDuplicateTraceExpert
		}
		seen[expert] = struct{}{}
	}
	return nil
}

// unionStats returns the mean and nearest-rank p50/p95 of sizes.
func unionStats(sizes []int) (float64, int, int) {
	if len(sizes) == 0 {
		return 0, 0, 0
	}
	sorted := append([]int(nil), sizes...)
	sort.Ints(sorted)
	sum := 0
	for _, s := range sorted {
		sum += s
	}
	rank := func(p float64) int {
		i := int(math.Ceil(p*float64(len(sorted)))) - 1
		if i < 0 {
			i = 0
		}
		return sorted[i]
	}
	return float64(sum) / float64(len(sorted)), rank(0.50), rank(0.95)
}
