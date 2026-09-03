package polymodel

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
)

// DraftMechanism identifies the speculative proposal generator strategy.
type DraftMechanism string

const (
	DraftMechanismLinearMTP    DraftMechanism = "linear_mtp"
	DraftMechanismDraftModel   DraftMechanism = "draft_model"
	DraftMechanismTreeSpec     DraftMechanism = "tree_spec"
	DraftMechanismPromptLookup DraftMechanism = "prompt_lookup"
)

// DraftSpec specifies the speculative configuration evaluated during offline profiling.
type DraftSpec struct {
	Method       DraftMechanism `json:"method"`
	DraftModelID string         `json:"draft_model_id,omitempty"`
	DraftDepth   int            `json:"draft_depth"`
	TreeTopology string         `json:"tree_topology,omitempty"`
	NGramWindow  int            `json:"ngram_window,omitempty"`
}

// OfflineTraceSample represents an offline evaluation trace instance containing prompt
// tokens, ground-truth target tokens, and speculative proposals.
type OfflineTraceSample struct {
	SampleID     string   `json:"sample_id"`
	Domain       string   `json:"domain,omitempty"`
	PromptTokens []int    `json:"prompt_tokens,omitempty"`
	TargetTokens []int    `json:"target_tokens"`
	DraftTokens  []int    `json:"draft_tokens,omitempty"`
	DraftTree    SpecTree `json:"draft_tree,omitempty"`
}

// SpecAcceptanceProfileSchema is the versioned schema string for SpecAcceptanceProfile.
const SpecAcceptanceProfileSchema = "fak-spec-acceptance/1"

// SpecAcceptanceProfile encapsulates empirical acceptance characteristics computed
// from offline trace profiling across benchmarks (Spec-Bench, HumanEval, GSM8k, ShareGPT).
type SpecAcceptanceProfile struct {
	Schema               string               `json:"schema"`
	TargetModel          string               `json:"target_model"`
	DraftSpec            DraftSpec            `json:"draft_spec"`
	Domain               string               `json:"domain"`
	Temperature          float64              `json:"temperature"`
	SampleCount          int                  `json:"sample_count"`
	TotalRounds          int                  `json:"total_rounds"`
	MeanAcceptanceLength float64              `json:"mean_acceptance_length"`
	PositionalAlpha      []float64            `json:"positional_alpha"`
	TreeDepthYield       map[int]float64      `json:"tree_depth_yield,omitempty"`
	Positions            []AcceptancePosition `json:"positions,omitempty"`
}

// JSON serializes the SpecAcceptanceProfile into indented JSON bytes.
func (p SpecAcceptanceProfile) JSON() ([]byte, error) {
	return json.MarshalIndent(p, "", "  ")
}

// OfflineProfileEvaluator executes speculative acceptance profiling over offline trace samples.
type OfflineProfileEvaluator struct {
	TargetModel string
	DraftSpec   DraftSpec
	Domain      string
	Temperature float64
}

// NewOfflineProfileEvaluator constructs an OfflineProfileEvaluator with the provided target model and draft specification.
func NewOfflineProfileEvaluator(targetModel string, spec DraftSpec) *OfflineProfileEvaluator {
	return &OfflineProfileEvaluator{
		TargetModel: targetModel,
		DraftSpec:   spec,
	}
}

// Evaluate executes evaluation over the slice of OfflineTraceSample instances.
func (e *OfflineProfileEvaluator) Evaluate(samples []OfflineTraceSample) SpecAcceptanceProfile {
	profile := acceptanceProfile{}
	totalRounds := 0
	totalAdvance := 0
	depthAccepted := make(map[int]int)
	depthProposed := make(map[int]int)
	totalTreeRounds := 0

	domain := e.Domain
	if domain == "" && len(samples) > 0 && samples[0].Domain != "" {
		domain = samples[0].Domain
	}

	for _, sample := range samples {
		isTree := len(sample.DraftTree.Nodes) > 1 || e.DraftSpec.Method == DraftMechanismTreeSpec
		if isTree {
			r, adv, dAcc, dProp := evaluateTreeSample(sample, e.DraftSpec, &profile)
			totalRounds += r
			totalAdvance += adv
			totalTreeRounds += r
			for d, count := range dAcc {
				depthAccepted[d] += count
			}
			for d, count := range dProp {
				depthProposed[d] += count
			}
		} else {
			r, adv, dAcc, dProp := evaluateLinearSample(sample, e.DraftSpec, &profile)
			totalRounds += r
			totalAdvance += adv
			for d, count := range dAcc {
				depthAccepted[d] += count
			}
			for d, count := range dProp {
				depthProposed[d] += count
			}
		}
	}

	positions := profile.snapshot()
	alphas := make([]float64, len(positions))
	for i, pos := range positions {
		if pos.Rate != nil {
			alphas[i] = *pos.Rate
		}
	}

	meanLen := 0.0
	if totalRounds > 0 {
		meanLen = float64(totalAdvance) / float64(totalRounds)
	}

	treeYield := make(map[int]float64)
	if totalTreeRounds > 0 {
		for d, accCount := range depthAccepted {
			treeYield[d] = float64(accCount) / float64(totalTreeRounds)
		}
	} else if len(depthAccepted) > 0 && totalRounds > 0 {
		for d, accCount := range depthAccepted {
			treeYield[d] = float64(accCount) / float64(totalRounds)
		}
	}

	return SpecAcceptanceProfile{
		Schema:               SpecAcceptanceProfileSchema,
		TargetModel:          e.TargetModel,
		DraftSpec:            e.DraftSpec,
		Domain:               domain,
		Temperature:          e.Temperature,
		SampleCount:          len(samples),
		TotalRounds:          totalRounds,
		MeanAcceptanceLength: meanLen,
		PositionalAlpha:      alphas,
		TreeDepthYield:       treeYield,
		Positions:            positions,
	}
}

// EvaluateTraceSamples delegates to Evaluate for API convenience.
func (e *OfflineProfileEvaluator) EvaluateTraceSamples(samples []OfflineTraceSample) SpecAcceptanceProfile {
	return e.Evaluate(samples)
}

// EvaluateLinearTrace evaluates a single linear trace sample on the evaluator.
func (e *OfflineProfileEvaluator) EvaluateLinearTrace(sample OfflineTraceSample, draftDepth ...int) SpecAcceptanceProfile {
	spec := e.DraftSpec
	if len(draftDepth) > 0 && draftDepth[0] > 0 {
		spec.DraftDepth = draftDepth[0]
	}
	sub := *e
	sub.DraftSpec = spec
	return sub.Evaluate([]OfflineTraceSample{sample})
}

// EvaluateTreeTrace evaluates a single tree trace sample on the evaluator.
func (e *OfflineProfileEvaluator) EvaluateTreeTrace(sample OfflineTraceSample) SpecAcceptanceProfile {
	sub := *e
	if sub.DraftSpec.Method == "" {
		sub.DraftSpec.Method = DraftMechanismTreeSpec
	}
	return sub.Evaluate([]OfflineTraceSample{sample})
}

// EvaluateLinearTrace evaluates speculative acceptance for a single linear trace sample.
func EvaluateLinearTrace(sample OfflineTraceSample, draftDepth ...int) SpecAcceptanceProfile {
	depth := 0
	if len(draftDepth) > 0 && draftDepth[0] > 0 {
		depth = draftDepth[0]
	}
	spec := DraftSpec{
		Method:     DraftMechanismLinearMTP,
		DraftDepth: depth,
	}
	eval := OfflineProfileEvaluator{
		DraftSpec: spec,
		Domain:    sample.Domain,
	}
	return eval.Evaluate([]OfflineTraceSample{sample})
}

// EvaluateTreeTrace evaluates speculative acceptance for a single tree trace sample.
func EvaluateTreeTrace(sample OfflineTraceSample) SpecAcceptanceProfile {
	spec := DraftSpec{
		Method: DraftMechanismTreeSpec,
	}
	if len(sample.DraftTree.Nodes) > 1 {
		spec.DraftDepth = len(sample.DraftTree.Nodes) - 1
	}
	eval := OfflineProfileEvaluator{
		DraftSpec: spec,
		Domain:    sample.Domain,
	}
	return eval.Evaluate([]OfflineTraceSample{sample})
}

// EvaluateTraceSamples evaluates speculative acceptance across multiple trace samples under the provided DraftSpec.
func EvaluateTraceSamples(samples []OfflineTraceSample, spec DraftSpec) SpecAcceptanceProfile {
	domain := ""
	if len(samples) > 0 && samples[0].Domain != "" {
		domain = samples[0].Domain
	}
	eval := OfflineProfileEvaluator{
		DraftSpec: spec,
		Domain:    domain,
	}
	return eval.Evaluate(samples)
}

func evaluateLinearSample(sample OfflineTraceSample, spec DraftSpec, profile *acceptanceProfile) (int, int, map[int]int, map[int]int) {
	if len(sample.TargetTokens) == 0 {
		return 0, 0, nil, nil
	}

	depth := spec.DraftDepth
	if depth <= 0 {
		if len(sample.DraftTokens) > 0 {
			depth = len(sample.DraftTokens)
		} else {
			depth = 4
		}
	}

	dAcc := make(map[int]int)
	dProp := make(map[int]int)
	rounds := 0
	advance := 0

	// Single-round scenario: draft is exactly one proposal matching or shorter than depth,
	// and target represents the immediate verification window.
	if len(sample.DraftTokens) > 0 && len(sample.DraftTokens) <= depth && len(sample.TargetTokens) <= depth+1 {
		draft := sample.DraftTokens
		targetArgmax := sample.TargetTokens
		res := AcceptGreedy(draft, targetArgmax)
		profile.record(len(draft), res.Accepted)
		rounds = 1
		advance = res.Advance
		for i := 0; i < len(draft); i++ {
			dProp[i+1]++
			if i < res.Accepted {
				dAcc[i+1]++
			}
		}
		return rounds, advance, dAcc, dProp
	}

	// Multi-round sequence simulation: step through TargetTokens.
	committed := 0
	for committed < len(sample.TargetTokens) {
		var draft []int
		if committed < len(sample.DraftTokens) {
			end := committed + depth
			if end > len(sample.DraftTokens) {
				end = len(sample.DraftTokens)
			}
			draft = sample.DraftTokens[committed:end]
		} else if spec.Method == DraftMechanismPromptLookup && spec.NGramWindow > 0 {
			draft = promptLookupDraft(sample.PromptTokens, sample.TargetTokens[:committed], spec.NGramWindow, depth)
		}

		if len(draft) == 0 {
			// No draft available; single autoregressive step.
			committed++
			rounds++
			advance++
			continue
		}

		targetLimit := committed + len(draft) + 1
		if targetLimit > len(sample.TargetTokens) {
			targetLimit = len(sample.TargetTokens)
		}
		targetArgmax := sample.TargetTokens[committed:targetLimit]

		res := AcceptGreedy(draft, targetArgmax)
		profile.record(len(draft), res.Accepted)
		rounds++
		advance += res.Advance

		for i := 0; i < len(draft); i++ {
			dProp[i+1]++
			if i < res.Accepted {
				dAcc[i+1]++
			}
		}

		committed += res.Advance
		if res.Advance <= 0 {
			committed++
		}
	}

	return rounds, advance, dAcc, dProp
}

func evaluateTreeSample(sample OfflineTraceSample, spec DraftSpec, profile *acceptanceProfile) (int, int, map[int]int, map[int]int) {
	if len(sample.TargetTokens) == 0 && len(sample.DraftTree.Nodes) <= 1 {
		return 0, 0, nil, nil
	}

	tree := sample.DraftTree
	if len(tree.Nodes) <= 1 {
		targetK := spec.DraftDepth
		if targetK <= 0 {
			if len(sample.DraftTokens) > 0 {
				targetK = len(sample.DraftTokens)
			} else {
				targetK = 4
			}
		}
		top := spec.TreeTopology
		if top == "" {
			top = "linear"
		}
		tree = GenerateTargetSizeTree(targetK, top)
		for i := 1; i < len(tree.Nodes) && i-1 < len(sample.DraftTokens); i++ {
			tree.Nodes[i].Token = sample.DraftTokens[i-1]
		}
	}

	if len(tree.Nodes) <= 1 {
		return 0, 0, nil, nil
	}

	dAcc := make(map[int]int)
	dProp := make(map[int]int)
	rounds := 0
	advance := 0

	// If TargetArgmax is already recorded on nodes (trace log fixture), execute one direct pass.
	if tree.Nodes[0].TargetArgmax != 0 || len(sample.TargetTokens) == 0 {
		res := AcceptTree(tree)
		posCounts := treeProposalPositions(tree)
		profile.recordCounts(posCounts, len(res.Path))
		rounds = 1
		advance = res.Advance
		for d := 1; d <= len(posCounts); d++ {
			dProp[d]++
			if len(res.Path) >= d {
				dAcc[d]++
			}
		}
		return rounds, advance, dAcc, dProp
	}

	// Multi-round sequence simulation: evaluate tree proposals against TargetTokens stream.
	committed := 0
	for committed < len(sample.TargetTokens) {
		subTree := assignTreeTargetTokens(tree, sample.TargetTokens[committed:])
		res := AcceptTree(subTree)
		if res.Advance <= 0 {
			committed++
			rounds++
			advance++
			continue
		}
		posCounts := treeProposalPositions(subTree)
		profile.recordCounts(posCounts, len(res.Path))
		rounds++
		advance += res.Advance

		for d := 1; d <= len(posCounts); d++ {
			dProp[d]++
			if len(res.Path) >= d {
				dAcc[d]++
			}
		}
		committed += res.Advance
	}

	return rounds, advance, dAcc, dProp
}

func assignTreeTargetTokens(tree SpecTree, targetTokens []int) SpecTree {
	if len(tree.Nodes) == 0 || len(targetTokens) == 0 {
		return tree
	}
	nodes := make([]TreeNode, len(tree.Nodes))
	copy(nodes, tree.Nodes)

	nodes[0].TargetArgmax = targetTokens[0]
	cur := 0
	targetIdx := 0
	for {
		want := nodes[cur].TargetArgmax
		next := -1
		for _, c := range nodes[cur].Children {
			if c >= 0 && c < len(nodes) && nodes[c].Token == want {
				next = c
				break
			}
		}
		if next < 0 {
			break
		}
		targetIdx++
		if targetIdx < len(targetTokens) {
			nodes[next].TargetArgmax = targetTokens[targetIdx]
		}
		cur = next
	}
	return SpecTree{Nodes: nodes}
}

func promptLookupDraft(prompt, committed []int, nGramWindow, draftDepth int) []int {
	history := make([]int, 0, len(prompt)+len(committed))
	history = append(history, prompt...)
	history = append(history, committed...)

	if len(history) < nGramWindow+1 || nGramWindow <= 0 || draftDepth <= 0 {
		return nil
	}

	query := history[len(history)-nGramWindow:]
	searchLimit := len(history) - nGramWindow
	for i := searchLimit - 1; i >= 0; i-- {
		match := true
		for j := 0; j < nGramWindow; j++ {
			if history[i+j] != query[j] {
				match = false
				break
			}
		}
		if match {
			start := i + nGramWindow
			if start >= len(history) {
				return nil
			}
			end := start + draftDepth
			if end > len(history) {
				end = len(history)
			}
			return append([]int(nil), history[start:end]...)
		}
	}
	return nil
}

// GenerateSyntheticEvalCorpus generates a deterministic synthetic evaluation corpus
// of trace samples for benchmark simulation across code, math, and dialog domains.
func GenerateSyntheticEvalCorpus(domain string, nSamples int) []OfflineTraceSample {
	if nSamples <= 0 {
		return nil
	}

	dom := strings.ToLower(strings.TrimSpace(domain))
	if dom == "" {
		dom = "dialog"
	}

	matchRate := 0.50
	switch dom {
	case "code":
		matchRate = 0.80
	case "math":
		matchRate = 0.65
	case "dialog":
		matchRate = 0.45
	}

	rng := rand.New(rand.NewSource(int64(len(dom)*1000 + nSamples)))
	samples := make([]OfflineTraceSample, nSamples)

	for i := 0; i < nSamples; i++ {
		sampleID := fmt.Sprintf("%s-synth-%04d", dom, i+1)
		targetLen := 24 + rng.Intn(16)
		promptLen := 12

		promptTokens := make([]int, promptLen)
		for p := 0; p < promptLen; p++ {
			promptTokens[p] = 200 + p
		}

		targetTokens := make([]int, targetLen)
		for t := 0; t < targetLen; t++ {
			targetTokens[t] = 1000 + t*3 + rng.Intn(3)
		}

		draftTokens := make([]int, targetLen)
		for d := 0; d < targetLen; d++ {
			if rng.Float64() < matchRate {
				draftTokens[d] = targetTokens[d]
			} else {
				draftTokens[d] = targetTokens[d] + 50000 + rng.Intn(100)
			}
		}

		// Construct a branching candidate tree (branchFactor 2, depth 3)
		tree := GenerateWideShallowTree(2, 3)
		// Align tree candidates with target tokens according to matchRate
		for levelIdx := 1; levelIdx < len(tree.Nodes) && levelIdx-1 < len(targetTokens); levelIdx++ {
			if rng.Float64() < matchRate {
				tree.Nodes[levelIdx].Token = targetTokens[levelIdx-1]
			} else {
				tree.Nodes[levelIdx].Token = targetTokens[levelIdx-1] + 60000 + rng.Intn(100)
			}
		}

		samples[i] = OfflineTraceSample{
			SampleID:     sampleID,
			Domain:       dom,
			PromptTokens: promptTokens,
			TargetTokens: targetTokens,
			DraftTokens:  draftTokens,
			DraftTree:    tree,
		}
	}

	return samples
}
