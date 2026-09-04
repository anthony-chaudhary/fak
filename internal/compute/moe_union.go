package compute

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"time"
)

// DefaultSmallBatchExpertBudget is the upper bound on unique active experts
// for small verification and decode batches (B <= 4, K <= 10).
const DefaultSmallBatchExpertBudget = 40

// ExpertTokenGroup records the mapping for one unique active expert in the batch union,
// including all token indices routed to this expert and their routing weights.
type ExpertTokenGroup struct {
	ExpertID     int       `json:"expert_id"`
	TokenIndices []int     `json:"token_indices"` // Indices of tokens in [0, BatchSize)
	Weights      []float32 `json:"weights"`       // Corresponding routing weights
}

// TokenRouting represents top-k expert selection and routing weights for a single token.
type TokenRouting struct {
	TokenIndex int       `json:"token_index"`
	ExpertIDs  []int     `json:"expert_ids"`
	Weights    []float32 `json:"weights"`
}

// ExpertUnionPlan represents the pre-computed expert union dispatch schedule for a batch
// of tokens. It evaluates each unique active expert exactly once across all relevant tokens
// in the batch, eliminating GPU warp divergence and redundant weight memory fetches.
type ExpertUnionPlan struct {
	BatchSize           int                `json:"batch_size"`
	NumTotalExperts     int                `json:"num_total_experts"`
	TopK                int                `json:"top_k"`
	ActiveExperts       []int              `json:"active_experts"` // Sorted unique expert IDs
	NumUniqueExperts    int                `json:"num_unique_experts"`
	Groups              []ExpertTokenGroup `json:"groups"`             // One group per active expert
	Offsets             []int              `json:"offsets"`            // CSR row offsets [NumUniqueExperts + 1]
	FlatTokenIndices    []int              `json:"flat_token_indices"` // Flattened token indices
	FlatWeights         []float32          `json:"flat_weights"`       // Flattened routing weights
	PerTokenExperts     [][]int            `json:"per_token_experts"`
	PerTokenWeights     [][]float32        `json:"per_token_weights"`
	NaiveLaunches       int                `json:"naive_launches"`
	GroupedLaunches     int                `json:"grouped_launches"`
	DivergenceReduction float64            `json:"divergence_reduction"` // 1.0 - (grouped / naive)
	MaxTokensPerExpert  int                `json:"max_tokens_per_expert"`
}

// MoeExecutionStats captures execution metrics for MoE dispatch, demonstrating
// the operational divergence reduction and performance benefits of grouped dispatch.
type MoeExecutionStats struct {
	TotalLaunches       int           `json:"total_launches"`
	GroupedLaunches     int           `json:"grouped_launches"`
	NaiveLaunches       int           `json:"naive_launches"`
	TokensProcessed     int           `json:"tokens_processed"`
	UniqueExperts       int           `json:"unique_experts"`
	LaunchReduction     float64       `json:"launch_reduction"`
	GroupedDuration     time.Duration `json:"grouped_duration,omitempty"`
	NaiveDuration       time.Duration `json:"naive_duration,omitempty"`
	SpeedupRatio        float64       `json:"speedup_ratio,omitempty"`
	EquivalenceMaxDiff  float32       `json:"equivalence_max_diff"`
	CosineSimilarity    float64       `json:"cosine_similarity"`
	MathematicallyEqual bool          `json:"mathematically_equal"`
}

// ExpertBatchFn computes an expert feed-forward transformation on a batch of token inputs.
// inputs has length numTokens * hiddenDim (row-major).
// Returns output tensor of length numTokens * hiddenDim (row-major).
type ExpertBatchFn func(expertID int, inputs []float32, numTokens, hiddenDim int) []float32

// BuildExpertUnion constructs an ExpertUnionPlan from a batch of tokens and their top-k
// routed expert indices. If weights is nil, uniform weights (1.0/K) are assigned.
func BuildExpertUnion(expertIndices [][]int, weights [][]float32, numTotalExperts int) (*ExpertUnionPlan, error) {
	if numTotalExperts <= 0 {
		return nil, fmt.Errorf("compute: numTotalExperts must be positive, got %d", numTotalExperts)
	}
	batchSize := len(expertIndices)
	if batchSize == 0 {
		return nil, errors.New("compute: expertIndices batch must not be empty")
	}
	if weights != nil && len(weights) != batchSize {
		return nil, fmt.Errorf("compute: mismatched batch size: expertIndices=%d, weights=%d", batchSize, len(weights))
	}

	topK := len(expertIndices[0])
	totalNaiveLaunches := 0

	perTokenExperts := make([][]int, batchSize)
	perTokenWeights := make([][]float32, batchSize)
	seenExperts := make([]bool, numTotalExperts)
	var activeExperts []int

	for b := 0; b < batchSize; b++ {
		k := len(expertIndices[b])
		if k == 0 {
			return nil, fmt.Errorf("compute: token %d has zero routed experts", b)
		}
		if weights != nil && len(weights[b]) != k {
			return nil, fmt.Errorf("compute: token %d mismatched expert count %d and weights count %d", b, k, len(weights[b]))
		}

		totalNaiveLaunches += k
		if k > topK {
			topK = k
		}

		perTokenExperts[b] = make([]int, k)
		copy(perTokenExperts[b], expertIndices[b])

		perTokenWeights[b] = make([]float32, k)
		if weights != nil {
			copy(perTokenWeights[b], weights[b])
		} else {
			uniformWeight := float32(1.0) / float32(k)
			for i := 0; i < k; i++ {
				perTokenWeights[b][i] = uniformWeight
			}
		}

		tokenSeen := make(map[int]bool, k)
		for _, e := range expertIndices[b] {
			if e < 0 || e >= numTotalExperts {
				return nil, fmt.Errorf("compute: token %d expert ID %d out of bounds [0, %d)", b, e, numTotalExperts)
			}
			if tokenSeen[e] {
				return nil, fmt.Errorf("compute: token %d has duplicate expert ID %d", b, e)
			}
			tokenSeen[e] = true

			if !seenExperts[e] {
				seenExperts[e] = true
				activeExperts = append(activeExperts, e)
			}
		}
	}

	sort.Ints(activeExperts)
	numUnique := len(activeExperts)

	groups := make([]ExpertTokenGroup, numUnique)
	expertToGroupIdx := make(map[int]int, numUnique)
	for i, e := range activeExperts {
		groups[i] = ExpertTokenGroup{
			ExpertID:     e,
			TokenIndices: make([]int, 0, batchSize),
			Weights:      make([]float32, 0, batchSize),
		}
		expertToGroupIdx[e] = i
	}

	for b := 0; b < batchSize; b++ {
		for k, e := range perTokenExperts[b] {
			gIdx := expertToGroupIdx[e]
			groups[gIdx].TokenIndices = append(groups[gIdx].TokenIndices, b)
			groups[gIdx].Weights = append(groups[gIdx].Weights, perTokenWeights[b][k])
		}
	}

	maxTokensPerExpert := 0
	offsets := make([]int, numUnique+1)
	var flatTokens []int
	var flatWeights []float32

	for i := range groups {
		offsets[i] = len(flatTokens)
		cnt := len(groups[i].TokenIndices)
		if cnt > maxTokensPerExpert {
			maxTokensPerExpert = cnt
		}
		flatTokens = append(flatTokens, groups[i].TokenIndices...)
		flatWeights = append(flatWeights, groups[i].Weights...)
	}
	offsets[numUnique] = len(flatTokens)

	reduction := 0.0
	if totalNaiveLaunches > 0 {
		reduction = 1.0 - float64(numUnique)/float64(totalNaiveLaunches)
	}

	return &ExpertUnionPlan{
		BatchSize:           batchSize,
		NumTotalExperts:     numTotalExperts,
		TopK:                topK,
		ActiveExperts:       activeExperts,
		NumUniqueExperts:    numUnique,
		Groups:              groups,
		Offsets:             offsets,
		FlatTokenIndices:    flatTokens,
		FlatWeights:         flatWeights,
		PerTokenExperts:     perTokenExperts,
		PerTokenWeights:     perTokenWeights,
		NaiveLaunches:       totalNaiveLaunches,
		GroupedLaunches:     numUnique,
		DivergenceReduction: reduction,
		MaxTokensPerExpert:  maxTokensPerExpert,
	}, nil
}

// BuildExpertUnionFromRouting constructs an ExpertUnionPlan from a slice of TokenRouting entries.
func BuildExpertUnionFromRouting(routings []TokenRouting, numTotalExperts int) (*ExpertUnionPlan, error) {
	if len(routings) == 0 {
		return nil, errors.New("compute: routings must not be empty")
	}
	batchSize := len(routings)
	indices := make([][]int, batchSize)
	weights := make([][]float32, batchSize)
	for i, r := range routings {
		indices[i] = r.ExpertIDs
		weights[i] = r.Weights
	}
	return BuildExpertUnion(indices, weights, numTotalExperts)
}

// ExpertGroup returns the ExpertTokenGroup for a given expert ID, if active in the plan.
func (p *ExpertUnionPlan) ExpertGroup(expertID int) (ExpertTokenGroup, bool) {
	if p == nil || len(p.Groups) == 0 {
		return ExpertTokenGroup{}, false
	}
	idx := sort.SearchInts(p.ActiveExperts, expertID)
	if idx < len(p.ActiveExperts) && p.ActiveExperts[idx] == expertID {
		return p.Groups[idx], true
	}
	return ExpertTokenGroup{}, false
}

// IsWithinSmallBatchBudget reports whether the active expert count satisfies the
// <= 40 expert envelope for small batches.
func (p *ExpertUnionPlan) IsWithinSmallBatchBudget() bool {
	if p == nil {
		return false
	}
	return p.NumUniqueExperts <= DefaultSmallBatchExpertBudget
}

// EvaluateGroupedMoE executes the expert union grouped dispatch. Each unique active expert
// is evaluated exactly once for all tokens routed to it.
func EvaluateGroupedMoE(
	plan *ExpertUnionPlan,
	tokens []float32,
	hiddenDim int,
	fn ExpertBatchFn,
) ([]float32, MoeExecutionStats, error) {
	if plan == nil {
		return nil, MoeExecutionStats{}, errors.New("compute: expert union plan is nil")
	}
	if hiddenDim <= 0 {
		return nil, MoeExecutionStats{}, fmt.Errorf("compute: hiddenDim must be positive, got %d", hiddenDim)
	}
	expectedLen := plan.BatchSize * hiddenDim
	if len(tokens) != expectedLen {
		return nil, MoeExecutionStats{}, fmt.Errorf("compute: token activations length %d != expected %d", len(tokens), expectedLen)
	}
	if fn == nil {
		return nil, MoeExecutionStats{}, errors.New("compute: expert batch function is nil")
	}

	start := time.Now()
	output := make([]float32, expectedLen)
	totalTokensProcessed := 0

	for _, group := range plan.Groups {
		m := len(group.TokenIndices)
		if m == 0 {
			continue
		}
		totalTokensProcessed += m

		batchInput := make([]float32, m*hiddenDim)
		for i, tokenIdx := range group.TokenIndices {
			src := tokens[tokenIdx*hiddenDim : (tokenIdx+1)*hiddenDim]
			dst := batchInput[i*hiddenDim : (i+1)*hiddenDim]
			copy(dst, src)
		}

		batchOutput := fn(group.ExpertID, batchInput, m, hiddenDim)
		if len(batchOutput) != m*hiddenDim {
			return nil, MoeExecutionStats{}, fmt.Errorf("compute: expert %d returned length %d, want %d", group.ExpertID, len(batchOutput), m*hiddenDim)
		}

		for i, tokenIdx := range group.TokenIndices {
			w := group.Weights[i]
			outSlice := output[tokenIdx*hiddenDim : (tokenIdx+1)*hiddenDim]
			inSlice := batchOutput[i*hiddenDim : (i+1)*hiddenDim]
			for d := 0; d < hiddenDim; d++ {
				outSlice[d] += w * inSlice[d]
			}
		}
	}

	duration := time.Since(start)
	stats := MoeExecutionStats{
		TotalLaunches:   len(plan.Groups),
		GroupedLaunches: len(plan.Groups),
		NaiveLaunches:   plan.NaiveLaunches,
		TokensProcessed: totalTokensProcessed,
		UniqueExperts:   plan.NumUniqueExperts,
		LaunchReduction: plan.DivergenceReduction,
		GroupedDuration: duration,
	}

	return output, stats, nil
}

// EvaluateNaiveMoE executes the unbatched, per-token dispatch where each token invokes its
// top-k experts separately, producing B * K kernel evaluations and warp divergence.
func EvaluateNaiveMoE(
	plan *ExpertUnionPlan,
	tokens []float32,
	hiddenDim int,
	fn ExpertBatchFn,
) ([]float32, MoeExecutionStats, error) {
	if plan == nil {
		return nil, MoeExecutionStats{}, errors.New("compute: expert union plan is nil")
	}
	if hiddenDim <= 0 {
		return nil, MoeExecutionStats{}, fmt.Errorf("compute: hiddenDim must be positive, got %d", hiddenDim)
	}
	expectedLen := plan.BatchSize * hiddenDim
	if len(tokens) != expectedLen {
		return nil, MoeExecutionStats{}, fmt.Errorf("compute: token activations length %d != expected %d", len(tokens), expectedLen)
	}
	if fn == nil {
		return nil, MoeExecutionStats{}, errors.New("compute: expert batch function is nil")
	}

	start := time.Now()
	output := make([]float32, expectedLen)
	totalTokensProcessed := 0
	totalLaunches := 0

	for b := 0; b < plan.BatchSize; b++ {
		tokenInput := tokens[b*hiddenDim : (b+1)*hiddenDim]
		k := len(plan.PerTokenExperts[b])
		for j := 0; j < k; j++ {
			expertID := plan.PerTokenExperts[b][j]
			w := plan.PerTokenWeights[b][j]

			tokenOutput := fn(expertID, tokenInput, 1, hiddenDim)
			if len(tokenOutput) != hiddenDim {
				return nil, MoeExecutionStats{}, fmt.Errorf("compute: expert %d returned length %d, want %d", expertID, len(tokenOutput), hiddenDim)
			}

			outSlice := output[b*hiddenDim : (b+1)*hiddenDim]
			for d := 0; d < hiddenDim; d++ {
				outSlice[d] += w * tokenOutput[d]
			}
			totalLaunches++
			totalTokensProcessed++
		}
	}

	duration := time.Since(start)
	stats := MoeExecutionStats{
		TotalLaunches:   totalLaunches,
		GroupedLaunches: len(plan.Groups),
		NaiveLaunches:   totalLaunches,
		TokensProcessed: totalTokensProcessed,
		UniqueExperts:   plan.NumUniqueExperts,
		LaunchReduction: plan.DivergenceReduction,
		NaiveDuration:   duration,
	}

	return output, stats, nil
}

// EvaluateExpertUnion demonstrates grouped dispatch vs naive per-token dispatch by evaluating
// both, proving mathematical equivalence, and reporting comparative performance metrics.
func EvaluateExpertUnion(
	plan *ExpertUnionPlan,
	tokens []float32,
	hiddenDim int,
	fn ExpertBatchFn,
) ([]float32, MoeExecutionStats, error) {
	if plan == nil {
		return nil, MoeExecutionStats{}, errors.New("compute: expert union plan is nil")
	}

	groupedOut, groupedStats, err := EvaluateGroupedMoE(plan, tokens, hiddenDim, fn)
	if err != nil {
		return nil, MoeExecutionStats{}, fmt.Errorf("grouped dispatch failed: %w", err)
	}

	naiveOut, naiveStats, err := EvaluateNaiveMoE(plan, tokens, hiddenDim, fn)
	if err != nil {
		return nil, MoeExecutionStats{}, fmt.Errorf("naive dispatch failed: %w", err)
	}

	maxDiff := float32(0)
	for i := range groupedOut {
		d := float32(math.Abs(float64(groupedOut[i] - naiveOut[i])))
		if d > maxDiff {
			maxDiff = d
		}
	}

	cos := computeCosine(groupedOut, naiveOut)
	speedup := 0.0
	if groupedStats.GroupedDuration > 0 {
		speedup = float64(naiveStats.NaiveDuration) / float64(groupedStats.GroupedDuration)
	}

	stats := MoeExecutionStats{
		TotalLaunches:       groupedStats.TotalLaunches,
		GroupedLaunches:     groupedStats.GroupedLaunches,
		NaiveLaunches:       naiveStats.TotalLaunches,
		TokensProcessed:     groupedStats.TokensProcessed,
		UniqueExperts:       plan.NumUniqueExperts,
		LaunchReduction:     plan.DivergenceReduction,
		GroupedDuration:     groupedStats.GroupedDuration,
		NaiveDuration:       naiveStats.NaiveDuration,
		SpeedupRatio:        speedup,
		EquivalenceMaxDiff:  maxDiff,
		CosineSimilarity:    cos,
		MathematicallyEqual: maxDiff < 1e-4 && cos >= 0.99999,
	}

	return groupedOut, stats, nil
}

func computeCosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		na += va * va
		nb += vb * vb
	}
	if na == 0 && nb == 0 {
		return 1.0
	}
	if na == 0 || nb == 0 {
		return 0.0
	}
	denom := math.Sqrt(na) * math.Sqrt(nb)
	if denom == 0 {
		return 0
	}
	val := dot / denom
	if val > 1.0 {
		return 1.0
	}
	if val < -1.0 {
		return -1.0
	}
	return val
}
