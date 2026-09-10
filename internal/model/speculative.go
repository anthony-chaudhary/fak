package model

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
)

// Standard error definitions for the speculative decoding engine.
var (
	ErrSpeculativeNilTarget          = errors.New("model: speculative engine requires non-nil target session or verifier")
	ErrSpeculativeNilGenerator       = errors.New("model: speculative engine requires a valid proposal generator")
	ErrSpeculativeTripwireDivergence = errors.New("model: speculative tripwire detected divergence from greedy argmax")
	ErrSpeculativeNonFiniteLogits    = errors.New("model: speculative verification encountered non-finite logits")
	ErrSpeculativeGeneratorNotFound  = errors.New("model: proposal generator not found")
)

// ProposalGenerator is the capability contract for speculative candidate proposal generators.
// It abstracts MTP heads, sidecar draft models, and prompt n-gram heuristics behind a uniform interface.
type ProposalGenerator interface {
	// Name returns the identifier of the proposal generator (e.g. "mtp", "draft_model", "ngram").
	Name() string
	// Propose produces speculative candidate tokens given committed context.
	Propose(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error)
}

// MTPProposalGenerator adapts native multi-token prediction heads to ProposalGenerator.
type MTPProposalGenerator struct {
	session   *Qwen35MTPDraftSession
	proposeFn func(ctx context.Context, committed []int, maxDraft int) ([]int, error)
	depth     int
}

// NewMTPProposalGenerator creates a ProposalGenerator backed by a native MTP draft session.
func NewMTPProposalGenerator(session *Qwen35MTPDraftSession) *MTPProposalGenerator {
	depth := Qwen35MTPMaxDraftDepth
	if session != nil && session.depth > 0 {
		depth = session.depth
	}
	return &MTPProposalGenerator{
		session: session,
		depth:   depth,
	}
}

// NewMTPProposalGeneratorWithFn creates an MTPProposalGenerator with an injected proposal function.
func NewMTPProposalGeneratorWithFn(fn func(ctx context.Context, committed []int, maxDraft int) ([]int, error)) *MTPProposalGenerator {
	return &MTPProposalGenerator{
		proposeFn: fn,
		depth:     Qwen35MTPMaxDraftDepth,
	}
}

// Name returns the identifier "mtp".
func (g *MTPProposalGenerator) Name() string {
	return "mtp"
}

// Propose produces speculative draft tokens using the native MTP head.
func (g *MTPProposalGenerator) Propose(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error) {
	if err := ctx.Err(); err != nil {
		return DraftProposal{}, err
	}
	if maxDraft <= 0 {
		maxDraft = g.depth
	}
	if g.proposeFn != nil {
		toks, err := g.proposeFn(ctx, committed, maxDraft)
		if err != nil {
			return DraftProposal{}, err
		}
		return NewLinearProposal(toks, map[string]any{"proposer": "mtp", "depth": len(toks)}), nil
	}
	if g.session == nil {
		return DraftProposal{Metadata: map[string]any{"proposer": "mtp"}}, nil
	}
	toks := g.session.Propose(committed)
	if err := g.session.Err(); err != nil {
		return DraftProposal{}, fmt.Errorf("model: mtp propose: %w", err)
	}
	if len(toks) > maxDraft {
		toks = toks[:maxDraft]
	}
	return NewLinearProposal(toks, map[string]any{"proposer": "mtp", "depth": len(toks)}), nil
}

// DraftModelProposalGenerator adapts a compact sidecar draft model to ProposalGenerator.
type DraftModelProposalGenerator struct {
	session   *Session
	proposeFn func(ctx context.Context, committed []int, maxDraft int) ([]int, error)
	maxDraft  int
}

// NewDraftModelProposalGenerator creates a ProposalGenerator backed by a co-resident draft model Session.
func NewDraftModelProposalGenerator(session *Session, maxDraft int) *DraftModelProposalGenerator {
	if maxDraft <= 0 {
		maxDraft = 4
	}
	return &DraftModelProposalGenerator{
		session:  session,
		maxDraft: maxDraft,
	}
}

// NewDraftModelProposalGeneratorWithFn creates a DraftModelProposalGenerator with an injected proposal function.
func NewDraftModelProposalGeneratorWithFn(fn func(ctx context.Context, committed []int, maxDraft int) ([]int, error)) *DraftModelProposalGenerator {
	return &DraftModelProposalGenerator{
		proposeFn: fn,
		maxDraft:  4,
	}
}

// Name returns the identifier "draft_model".
func (g *DraftModelProposalGenerator) Name() string {
	return "draft_model"
}

// Propose produces speculative candidate tokens from the sidecar draft model.
func (g *DraftModelProposalGenerator) Propose(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error) {
	if err := ctx.Err(); err != nil {
		return DraftProposal{}, err
	}
	if maxDraft <= 0 {
		maxDraft = g.maxDraft
	}
	if g.proposeFn != nil {
		toks, err := g.proposeFn(ctx, committed, maxDraft)
		if err != nil {
			return DraftProposal{}, err
		}
		return NewLinearProposal(toks, map[string]any{"proposer": "draft_model", "depth": len(toks)}), nil
	}
	if g.session == nil || len(committed) == 0 {
		return DraftProposal{Metadata: map[string]any{"proposer": "draft_model"}}, nil
	}

	// Catch up draft session with committed tokens
	cLen := g.session.Cache.Len()
	var lastLogits []float32
	if cLen == 0 {
		lastLogits = g.session.Prefill(committed)
	} else if cLen < len(committed) {
		for _, tok := range committed[cLen:] {
			lastLogits = g.session.Step(tok)
		}
	} else if cLen > len(committed) {
		g.session.evictKV(len(committed), cLen-len(committed))
		if len(committed) > 0 {
			g.session.evictKV(len(committed)-1, 1)
			lastLogits = g.session.Step(committed[len(committed)-1])
		}
	}

	if len(lastLogits) == 0 {
		return DraftProposal{Metadata: map[string]any{"proposer": "draft_model"}}, nil
	}

	draft := make([]int, 0, maxDraft)
	curLogits := lastLogits
	for i := 0; i < maxDraft; i++ {
		nextTok := argmaxF32(curLogits)
		draft = append(draft, nextTok)
		if i+1 < maxDraft {
			curLogits = g.session.Step(nextTok)
		}
	}

	return NewLinearProposal(draft, map[string]any{"proposer": "draft_model", "depth": len(draft)}), nil
}

// NGramProposalGenerator adapts prompt/history token n-gram matching to ProposalGenerator.
type NGramProposalGenerator struct {
	drafter       NgramDrafter
	treeMode      bool
	maxBranches   int
	loopSanitizer bool
}

// NewNGramProposalGenerator creates a ProposalGenerator backed by an NgramDrafter.
func NewNGramProposalGenerator(drafter NgramDrafter) *NGramProposalGenerator {
	return &NGramProposalGenerator{
		drafter:       drafter,
		loopSanitizer: true,
	}
}

// WithTree configures the generator to emit CandidateTree proposals.
func (g *NGramProposalGenerator) WithTree(maxBranches int) *NGramProposalGenerator {
	return &NGramProposalGenerator{
		drafter:       g.drafter,
		treeMode:      true,
		maxBranches:   maxBranches,
		loopSanitizer: g.loopSanitizer,
	}
}

// WithLoopSanitizer enables or disables cyclic loop detection in n-gram proposals.
func (g *NGramProposalGenerator) WithLoopSanitizer(enabled bool) *NGramProposalGenerator {
	return &NGramProposalGenerator{
		drafter:       g.drafter,
		treeMode:      g.treeMode,
		maxBranches:   g.maxBranches,
		loopSanitizer: enabled,
	}
}

// Name returns the identifier "ngram".
func (g *NGramProposalGenerator) Name() string {
	return "ngram"
}

// Propose produces speculative proposals from prompt/history n-grams.
func (g *NGramProposalGenerator) Propose(ctx context.Context, committed []int, maxDraft int) (DraftProposal, error) {
	if err := ctx.Err(); err != nil {
		return DraftProposal{}, err
	}
	proposer := NewNgramDraftProposer(g.drafter)
	if g.treeMode {
		proposer = proposer.WithTree(g.maxBranches)
	}
	prop, err := proposer.Propose(ctx, committed, maxDraft)
	if err != nil {
		return DraftProposal{}, err
	}

	// Apply loop sanitization if enabled
	if g.loopSanitizer && len(committed) >= 4 {
		sanitizer := NewRepetitionPenaltySanitizer(0, 0)
		prop = sanitizer.SanitizeProposal(prop, committed)
	}

	return prop, nil
}

// RepetitionPenaltySanitizer sanitizes repetition penalty states and guards against cyclic loops.
type RepetitionPenaltySanitizer struct {
	FrequencyPenalty float64
	PresencePenalty  float64
	MaxLoopRepeats   int
}

// NewRepetitionPenaltySanitizer creates a sanitizer configured with frequency and presence penalties.
func NewRepetitionPenaltySanitizer(frequencyPenalty, presencePenalty float64) *RepetitionPenaltySanitizer {
	return &RepetitionPenaltySanitizer{
		FrequencyPenalty: frequencyPenalty,
		PresencePenalty:  presencePenalty,
		MaxLoopRepeats:   2,
	}
}

// ApplyPenalty modifies logits according to the current generation counts.
// logit[t] -= frequencyPenalty * count[t] + presencePenalty * (1 if count[t] > 0 else 0)
func (s *RepetitionPenaltySanitizer) ApplyPenalty(logits []float32, counts []int32) []float32 {
	if s == nil || (s.FrequencyPenalty == 0 && s.PresencePenalty == 0) || len(counts) == 0 {
		return logits
	}
	eff := append([]float32(nil), logits...)
	for tok, c := range counts {
		if tok >= len(eff) || c <= 0 {
			continue
		}
		penalty := s.FrequencyPenalty*float64(c) + s.PresencePenalty
		eff[tok] -= float32(penalty)
	}
	return eff
}

// DetectDegenerateLoop checks if the suffix of history exhibits a periodic cyclic loop.
func (s *RepetitionPenaltySanitizer) DetectDegenerateLoop(history []int, minPeriod, maxPeriod, minRepeats int) (bool, int) {
	if len(history) < minPeriod*minRepeats {
		return false, 0
	}
	if minPeriod <= 0 {
		minPeriod = 1
	}
	if maxPeriod <= 0 || maxPeriod > len(history)/minRepeats {
		maxPeriod = len(history) / minRepeats
	}
	if minRepeats <= 1 {
		minRepeats = 2
	}

	n := len(history)
	for p := minPeriod; p <= maxPeriod; p++ {
		match := true
		for r := 1; r < minRepeats; r++ {
			for i := 0; i < p; i++ {
				idx1 := n - 1 - i
				idx2 := n - 1 - r*p - i
				if idx2 < 0 || history[idx1] != history[idx2] {
					match = false
					break
				}
			}
			if !match {
				break
			}
		}
		if match {
			return true, p
		}
	}
	return false, 0
}

// SanitizeProposal inspects a proposed draft and truncates or suppresses proposals that continue degenerate loops.
func (s *RepetitionPenaltySanitizer) SanitizeProposal(proposal DraftProposal, history []int) DraftProposal {
	if s == nil || len(history) == 0 {
		return proposal
	}
	repeats := s.MaxLoopRepeats
	if repeats <= 0 {
		repeats = 2
	}
	hasLoop, period := s.DetectDegenerateLoop(history, 1, 8, repeats)
	if !hasLoop {
		return proposal
	}

	// History already has a degenerate repeating loop with period.
	// Check if proposal continues this exact periodic cycle.
	if len(proposal.Tokens) > 0 {
		expectedCycleToken := history[len(history)-period]
		if proposal.Tokens[0] == expectedCycleToken {
			// Proposal continues degenerate cycle; suppress to break loop
			sanitized := proposal
			sanitized.Tokens = nil
			sanitized.Tree = nil
			sanitized.Mask = nil
			if sanitized.Metadata == nil {
				sanitized.Metadata = make(map[string]any)
			}
			sanitized.Metadata["sanitized_degenerate_loop"] = true
			sanitized.Metadata["detected_period"] = period
			return sanitized
		}
	}

	return proposal
}

// SanitizeCounts guarantees that counts only records committed and accepted tokens,
// ensuring rollback never leaks unaccepted speculative candidate counts.
func (s *RepetitionPenaltySanitizer) SanitizeCounts(counts []int32, committed []int, accepted []int) {
	if len(counts) == 0 {
		return
	}
	for i := range counts {
		counts[i] = 0
	}
	for _, tok := range committed {
		if tok >= 0 && tok < len(counts) {
			counts[tok]++
		}
	}
	for _, tok := range accepted {
		if tok >= 0 && tok < len(counts) {
			counts[tok]++
		}
	}
}

// TripwireVerify evaluates greedy temperature-zero acceptance and tripwires on non-finite logits or divergence.
func TripwireVerify(draft []int, targetArgmax []int, lastLogits []float32, targetLogits [][]float32) (accepted []int, bonus int, err error) {
	// 1. Logit validity tripwire: ensure no NaNs or Infs
	for _, l := range lastLogits {
		if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
			return nil, 0, ErrSpeculativeNonFiniteLogits
		}
	}
	for _, row := range targetLogits {
		for _, l := range row {
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				return nil, 0, ErrSpeculativeNonFiniteLogits
			}
		}
	}

	if len(targetArgmax) == 0 {
		return nil, 0, errors.New("model: empty target argmax in tripwire verification")
	}

	// 2. Step 0 verification: draft[0] must match targetArgmax[0] (argmax of lastLogits)
	if len(draft) == 0 {
		return nil, targetArgmax[0], nil
	}

	numAccepted := 0
	for i := 0; i < len(draft); i++ {
		if draft[i] != targetArgmax[i] {
			break
		}
		numAccepted++
	}

	var acceptedTokens []int
	if numAccepted > 0 {
		acceptedTokens = append([]int(nil), draft[:numAccepted]...)
	}

	bonusToken := targetArgmax[numAccepted]
	return acceptedTokens, bonusToken, nil
}

// ParallelVerifyKernel evaluates K speculative candidate tokens in a single target model forward pass.
func ParallelVerifyKernel(
	ctx context.Context,
	target *Session,
	committed []int,
	proposal DraftProposal,
	lastLogits []float32,
	sanitizer *RepetitionPenaltySanitizer,
	counts []int32,
) (VerificationResult, error) {
	if err := ctx.Err(); err != nil {
		return VerificationResult{}, err
	}
	if target == nil || target.M == nil {
		return VerificationResult{}, ErrSpeculativeNilTarget
	}
	if len(lastLogits) == 0 {
		return VerificationResult{}, errors.New("model: lastLogits required for parallel verify kernel")
	}

	// Handle candidate tree verification
	if proposal.Tree != nil && len(proposal.Tree.Nodes) > 0 {
		return parallelVerifyTree(ctx, target, committed, proposal, lastLogits, sanitizer, counts)
	}

	lastLogitsCopy := append([]float32(nil), lastLogits...)
	draft := proposal.Tokens
	if len(draft) == 0 {
		penalizedLast := lastLogitsCopy
		if sanitizer != nil && len(counts) > 0 {
			penalizedLast = sanitizer.ApplyPenalty(lastLogitsCopy, counts)
		}
		bonus := argmaxF32(penalizedLast)
		return VerificationResult{
			AcceptedTokens:  nil,
			CorrectionToken: bonus,
			NumAccepted:     0,
			RollbackKVCount: 0,
			TargetLogits:    nil,
		}, nil
	}

	tBase := target.Cache.Len()

	// Single target forward pass for all K draft tokens (cloning row buffers to avoid aliasing in quantized modes)
	var rows [][]float32
	if verifyForwardBatchedOK(target) {
		rawRows := target.VerifyForward(draft, nil, nil)
		rows = make([][]float32, len(rawRows))
		for i, r := range rawRows {
			rows[i] = append([]float32(nil), r...)
		}
	} else {
		rows = make([][]float32, len(draft))
		for i, tok := range draft {
			stepLogits := target.Step(tok)
			rows[i] = append([]float32(nil), stepLogits...)
		}
	}
	if len(rows) != len(draft) {
		return VerificationResult{}, fmt.Errorf("model: parallel verify forward returned %d rows for %d draft tokens", len(rows), len(draft))
	}

	// Compute penalized argmax array: [0..K]
	targetArgmax := make([]int, len(draft)+1)
	penalizedLast := lastLogitsCopy
	if sanitizer != nil && len(counts) > 0 {
		penalizedLast = sanitizer.ApplyPenalty(lastLogitsCopy, counts)
	}
	targetArgmax[0] = argmaxF32(penalizedLast)

	// Simulate running count for speculative steps to properly penalize each draft step
	simCounts := append([]int32(nil), counts...)
	for i, row := range rows {
		if targetArgmax[i] >= 0 && targetArgmax[i] < len(simCounts) {
			simCounts[targetArgmax[i]]++
		}
		penalizedRow := row
		if sanitizer != nil && len(simCounts) > 0 {
			penalizedRow = sanitizer.ApplyPenalty(row, simCounts)
		}
		targetArgmax[i+1] = argmaxF32(penalizedRow)
	}

	// Enforce temperature-zero exact argmax tripwire
	acceptedTokens, bonusToken, tripErr := TripwireVerify(draft, targetArgmax, lastLogitsCopy, rows)
	if tripErr != nil {
		return VerificationResult{}, tripErr
	}

	numAccepted := len(acceptedTokens)
	rollbackCount := len(draft) - numAccepted
	if rollbackCount > 0 {
		target.evictKV(tBase+numAccepted, rollbackCount)
	}

	return VerificationResult{
		AcceptedTokens:  acceptedTokens,
		CorrectionToken: bonusToken,
		NumAccepted:     numAccepted,
		RollbackKVCount: rollbackCount,
		TargetLogits:    rows,
	}, nil
}

func parallelVerifyTree(
	ctx context.Context,
	target *Session,
	committed []int,
	proposal DraftProposal,
	lastLogits []float32,
	sanitizer *RepetitionPenaltySanitizer,
	counts []int32,
) (VerificationResult, error) {
	tree := proposal.Tree
	N := len(tree.Nodes)
	ids := proposal.Tokens
	if len(ids) != N {
		ids = tree.Tokens()
	}

	tBase := target.Cache.Len()
	pos := make([]int, N)
	for i, node := range tree.Nodes {
		pos[i] = tBase + node.Depth
	}

	mask := proposal.Mask
	if mask == nil || len(mask) != N {
		var err error
		mask, err = tree.DeriveMask()
		if err != nil {
			return VerificationResult{}, fmt.Errorf("model: derive tree causal mask: %w", err)
		}
	}

	allow := func(q, k int) bool {
		if q >= 0 && q < len(mask) && k >= 0 && k < len(mask[q]) {
			return mask[q][k]
		}
		return false
	}

	lastLogitsCopy := append([]float32(nil), lastLogits...)

	rawRows := target.VerifyForward(ids, pos, allow)
	if len(rawRows) != N {
		return VerificationResult{}, fmt.Errorf("model: verify forward returned %d rows for %d tree nodes", len(rawRows), N)
	}
	rows := make([][]float32, len(rawRows))
	for i, r := range rawRows {
		rows[i] = append([]float32(nil), r...)
	}

	penalizedLast := lastLogitsCopy
	if sanitizer != nil && len(counts) > 0 {
		penalizedLast = sanitizer.ApplyPenalty(lastLogitsCopy, counts)
	}
	rootExpected := argmaxF32(penalizedLast)

	matchRoot := -1
	for i, node := range tree.Nodes {
		if node.Parent == -1 && node.Token == rootExpected {
			matchRoot = i
			break
		}
	}

	if matchRoot == -1 {
		preserveUnacceptedBranches(target, committed, tree, nil, tBase, rows)
		target.evictKV(tBase, N)
		return VerificationResult{
			AcceptedTokens:  nil,
			CorrectionToken: rootExpected,
			NumAccepted:     0,
			RollbackKVCount: N,
			TargetLogits:    rows,
		}, nil
	}

	acceptedIndices := []int{matchRoot}
	cur := matchRoot

	simCounts := append([]int32(nil), counts...)
	if rootExpected < len(simCounts) {
		simCounts[rootExpected]++
	}

	penalizedRow := rows[cur]
	if sanitizer != nil && len(simCounts) > 0 {
		penalizedRow = sanitizer.ApplyPenalty(penalizedRow, simCounts)
	}
	pred := argmaxF32(penalizedRow)

	for {
		nextChild := -1
		for _, childIdx := range tree.Nodes[cur].Children {
			if childIdx >= 0 && childIdx < N &&
				tree.Nodes[childIdx].Parent == cur &&
				tree.Nodes[childIdx].Depth == len(acceptedIndices) &&
				tree.Nodes[childIdx].Token == pred {
				nextChild = childIdx
				break
			}
		}
		if nextChild == -1 {
			break
		}
		acceptedIndices = append(acceptedIndices, nextChild)
		cur = nextChild
		if pred < len(simCounts) {
			simCounts[pred]++
		}
		penalizedRow = rows[cur]
		if sanitizer != nil && len(simCounts) > 0 {
			penalizedRow = sanitizer.ApplyPenalty(penalizedRow, simCounts)
		}
		pred = argmaxF32(penalizedRow)
	}

	acceptedTokens := make([]int, len(acceptedIndices))
	for i, idx := range acceptedIndices {
		acceptedTokens[i] = tree.Nodes[idx].Token
	}

	preserveUnacceptedBranches(target, committed, tree, acceptedIndices, tBase, rows)
	// VerifyForward appended every tree node in panel order. Keep the selected
	// root-to-leaf path resident by compacting those rows into linear decode order;
	// replay is retained only as the conservative fallback for a cache/layout that
	// cannot prove the tree-compaction contract.
	compacted := validAcceptedTreePath(tree, acceptedIndices)
	if compacted {
		compacted = PruneAndCompactTreeKV(target.Cache, tBase, acceptedIndices, N) == nil
	}
	if !compacted {
		target.evictKV(tBase, N)
		for _, tok := range acceptedTokens {
			target.Step(tok)
		}
	}

	return VerificationResult{
		AcceptedTokens:     acceptedTokens,
		CorrectionToken:    pred,
		NumAccepted:        len(acceptedTokens),
		RollbackKVCount:    N - len(acceptedTokens),
		TargetLogits:       rows,
		LastAcceptedLogits: rows[cur],
	}, nil
}

// validAcceptedTreePath proves that acceptedIndices names one contiguous
// root-to-leaf path whose logical positions are prefix+depth. The KV compactor
// deliberately accepts indices rather than the tree, so this topology check
// remains at the verifier boundary that owns both values.
func validAcceptedTreePath(tree *CandidateTree, acceptedIndices []int) bool {
	if tree == nil || len(acceptedIndices) == 0 {
		return false
	}
	parent := -1
	for depth, idx := range acceptedIndices {
		if idx < 0 || idx >= len(tree.Nodes) {
			return false
		}
		node := tree.Nodes[idx]
		if node.Parent != parent || node.Depth != depth {
			return false
		}
		parent = idx
	}
	return true
}

// SpeculativeEngineConfig configures the coordinator runtime parameters.
type SpeculativeEngineConfig struct {
	MaxDraft           int     // draft depth K (default 4)
	Temperature        float64 // 0 for greedy decoding
	TopP               float64
	TopK               int
	FrequencyPenalty   float64
	PresencePenalty    float64
	TripwireStrict     bool
	SanitizeRepetition bool
	TreeMode           bool
	MaxBranches        int
	BranchCache        *SpeculativeBranchCache
}

// DefaultSpeculativeEngineConfig returns standard greedy production defaults.
func DefaultSpeculativeEngineConfig() SpeculativeEngineConfig {
	return SpeculativeEngineConfig{
		MaxDraft:           4,
		Temperature:        0.0,
		TripwireStrict:     true,
		SanitizeRepetition: true,
	}
}

// SpeculativeEngineStats aggregates execution statistics for speculative decoding.
type SpeculativeEngineStats struct {
	DraftTokensGenerated int     `json:"draft_tokens_generated"`
	DraftTokensAccepted  int     `json:"draft_tokens_accepted"`
	BonusTokensEmitted   int     `json:"bonus_tokens_emitted"`
	VerificationRounds   int     `json:"verification_rounds"`
	RollbackTokensCount  int     `json:"rollback_tokens_count"`
	MeanAcceptanceRate   float64 `json:"mean_acceptance_rate"`
}

// SpeculativeEngine coordinates multi-architecture draft proposal generation and parallel target verification.
type SpeculativeEngine struct {
	mu               sync.Mutex
	target           *Session
	verifier         DraftVerifier
	generators       map[string]ProposalGenerator
	primaryGenerator ProposalGenerator
	cfg              SpeculativeEngineConfig
	sanitizer        *RepetitionPenaltySanitizer
	stats            SpeculativeEngineStats
	lastLogits       []float32
	branchCache      *SpeculativeBranchCache
}

// NewSpeculativeEngine creates a unified speculative decoding engine backed by a target Session.
func NewSpeculativeEngine(target *Session, primary ProposalGenerator, cfg SpeculativeEngineConfig) *SpeculativeEngine {
	if cfg.MaxDraft <= 0 {
		cfg.MaxDraft = 4
	}
	sanitizer := NewRepetitionPenaltySanitizer(cfg.FrequencyPenalty, cfg.PresencePenalty)
	eng := &SpeculativeEngine{
		target:           target,
		generators:       make(map[string]ProposalGenerator),
		primaryGenerator: primary,
		cfg:              cfg,
		sanitizer:        sanitizer,
		branchCache:      cfg.BranchCache,
	}
	if primary != nil {
		eng.generators[primary.Name()] = primary
	}
	if eng.branchCache != nil && target != nil {
		AttachBranchCache(target, eng.branchCache)
	}
	return eng
}

// BranchCache returns the engine's attached branch cache, or nil.
func (e *SpeculativeEngine) BranchCache() *SpeculativeBranchCache {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.branchCache
}

// SetBranchCache dynamically attaches a branch cache to the speculative engine.
func (e *SpeculativeEngine) SetBranchCache(c *SpeculativeBranchCache) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.branchCache = c
	if e.target != nil {
		AttachBranchCache(e.target, c)
	}
}

// NewSpeculativeEngineWithVerifier creates an engine backed by a DraftVerifier capability contract.
func NewSpeculativeEngineWithVerifier(verifier DraftVerifier, primary ProposalGenerator, cfg SpeculativeEngineConfig) *SpeculativeEngine {
	if cfg.MaxDraft <= 0 {
		cfg.MaxDraft = 4
	}
	sanitizer := NewRepetitionPenaltySanitizer(cfg.FrequencyPenalty, cfg.PresencePenalty)
	eng := &SpeculativeEngine{
		verifier:         verifier,
		generators:       make(map[string]ProposalGenerator),
		primaryGenerator: primary,
		cfg:              cfg,
		sanitizer:        sanitizer,
	}
	if primary != nil {
		eng.generators[primary.Name()] = primary
	}
	return eng
}

// RegisterGenerator registers an additional proposal generator.
func (e *SpeculativeEngine) RegisterGenerator(gen ProposalGenerator) {
	if e == nil || gen == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.generators[gen.Name()] = gen
	if e.primaryGenerator == nil {
		e.primaryGenerator = gen
	}
}

// SetPrimaryGenerator switches the active proposal generator by name.
func (e *SpeculativeEngine) SetPrimaryGenerator(name string) error {
	if e == nil {
		return errors.New("model: nil speculative engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	gen, ok := e.generators[name]
	if !ok {
		return fmt.Errorf("%w: %s", ErrSpeculativeGeneratorNotFound, name)
	}
	e.primaryGenerator = gen
	return nil
}

// PrimaryGenerator returns the currently active proposal generator.
func (e *SpeculativeEngine) PrimaryGenerator() ProposalGenerator {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.primaryGenerator
}

// Generator retrieves a registered generator by name.
func (e *SpeculativeEngine) Generator(name string) ProposalGenerator {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.generators[name]
}

// Config returns the configuration of the speculative engine.
func (e *SpeculativeEngine) Config() SpeculativeEngineConfig {
	if e == nil {
		return SpeculativeEngineConfig{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.cfg
}

// Sanitizer returns the repetition penalty sanitizer used by the engine.
func (e *SpeculativeEngine) Sanitizer() *RepetitionPenaltySanitizer {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.sanitizer
}

// SetTargetSession binds or updates the target model session.
func (e *SpeculativeEngine) SetTargetSession(s *Session) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.target = s
}

// LastLogits returns the current boundary target logits.
func (e *SpeculativeEngine) LastLogits() []float32 {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]float32(nil), e.lastLogits...)
}

// SetLastLogits updates the boundary target logits for the next speculative round.
func (e *SpeculativeEngine) SetLastLogits(logits []float32) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.lastLogits = append([]float32(nil), logits...)
}

// Step runs a single speculative generation and verification round.
func (e *SpeculativeEngine) Step(
	ctx context.Context,
	committed []int,
	lastLogits []float32,
	counts []int32,
) (acceptedTokens []int, bonusToken int, nextLogits []float32, err error) {
	if e == nil {
		return nil, 0, nil, errors.New("model: nil speculative engine")
	}
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.primaryGenerator == nil {
		return nil, 0, nil, ErrSpeculativeNilGenerator
	}
	if len(lastLogits) == 0 {
		lastLogits = e.lastLogits
	}
	if len(lastLogits) == 0 {
		return nil, 0, nil, errors.New("model: lastLogits must not be empty for speculative step")
	}

	// 1. Generate speculative draft proposal
	proposal, propErr := e.primaryGenerator.Propose(ctx, committed, e.cfg.MaxDraft)
	if propErr != nil {
		return nil, 0, nil, fmt.Errorf("model: proposal generation: %w", propErr)
	}

	// Apply repetition sanitizer to proposal if enabled
	if e.cfg.SanitizeRepetition && e.sanitizer != nil {
		proposal = e.sanitizer.SanitizeProposal(proposal, committed)
	}

	// 2. Perform parallel verification
	var res VerificationResult
	if e.target != nil {
		res, err = ParallelVerifyKernel(ctx, e.target, committed, proposal, lastLogits, e.sanitizer, counts)
	} else if e.verifier != nil {
		res, err = e.verifier.Verify(ctx, committed, proposal)
	} else {
		return nil, 0, nil, ErrSpeculativeNilTarget
	}

	if err != nil {
		return nil, 0, nil, err
	}

	// 3. Update stats
	e.stats.DraftTokensGenerated += len(proposal.Tokens)
	e.stats.DraftTokensAccepted += res.NumAccepted
	e.stats.BonusTokensEmitted++
	e.stats.VerificationRounds++
	e.stats.RollbackTokensCount += res.RollbackKVCount
	if e.stats.VerificationRounds > 0 {
		e.stats.MeanAcceptanceRate = float64(e.stats.DraftTokensAccepted) / float64(e.stats.VerificationRounds)
	}

	// 4. Update next logits: if tokens accepted, use last accepted token logits, else lastLogits
	if res.NumAccepted > 0 && len(res.LastAcceptedLogits) > 0 {
		nextLogits = res.LastAcceptedLogits
	} else if res.NumAccepted > 0 && len(res.TargetLogits) >= res.NumAccepted {
		nextLogits = res.TargetLogits[res.NumAccepted-1]
	} else {
		nextLogits = lastLogits
	}
	e.lastLogits = nextLogits

	return res.AcceptedTokens, res.CorrectionToken, nextLogits, nil
}

// Generate runs speculative decoding end-to-end for prompt up to maxNew tokens.
// It asserts exact output identity with autoregressive greedy generation under temperature zero.
func (e *SpeculativeEngine) Generate(ctx context.Context, prompt []int, maxNew int) ([]int, error) {
	if e == nil {
		return nil, errors.New("model: nil speculative engine")
	}
	if len(prompt) == 0 {
		return nil, errors.New("model: prompt cannot be empty")
	}
	if maxNew <= 0 {
		return nil, nil
	}

	e.mu.Lock()
	target := e.target
	e.mu.Unlock()

	if target == nil {
		return nil, ErrSpeculativeNilTarget
	}

	// Prefill target session
	logits := target.Prefill(prompt)
	committed := append([]int(nil), prompt...)
	e.SetLastLogits(logits)

	var counts []int32
	if e.cfg.FrequencyPenalty != 0 || e.cfg.PresencePenalty != 0 {
		counts = make([]int32, len(logits))
	}

	generated := make([]int, 0, maxNew)
	for len(generated) < maxNew {
		if err := ctx.Err(); err != nil {
			return generated, err
		}

		accepted, bonus, _, err := e.Step(ctx, committed, logits, counts)
		if err != nil {
			return generated, err
		}

		for _, tok := range accepted {
			generated = append(generated, tok)
			committed = append(committed, tok)
			if len(counts) > 0 && tok < len(counts) {
				counts[tok]++
			}
			if len(generated) == maxNew {
				break
			}
		}

		if len(generated) < maxNew {
			generated = append(generated, bonus)
			committed = append(committed, bonus)
			if len(counts) > 0 && bonus < len(counts) {
				counts[bonus]++
			}
			logits = target.Step(bonus)
			e.SetLastLogits(logits)
		}
	}

	return generated, nil
}

// Stats returns a snapshot of speculative decoding statistics.
func (e *SpeculativeEngine) Stats() SpeculativeEngineStats {
	if e == nil {
		return SpeculativeEngineStats{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.stats
}

// RecordVerification updates execution statistics for an externally driven verification round.
func (e *SpeculativeEngine) RecordVerification(generated, accepted, rollback int) {
	e.RecordVerificationOutcome(generated, accepted, rollback, true)
}

// RecordVerificationOutcome updates execution statistics for an externally
// driven verification round after its correction/bonus emission decision.
func (e *SpeculativeEngine) RecordVerificationOutcome(generated, accepted, rollback int, bonusEmitted bool) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.stats.DraftTokensGenerated += generated
	e.stats.DraftTokensAccepted += accepted
	if bonusEmitted {
		e.stats.BonusTokensEmitted++
	}
	e.stats.VerificationRounds++
	e.stats.RollbackTokensCount += rollback
	if e.stats.VerificationRounds > 0 {
		e.stats.MeanAcceptanceRate = float64(e.stats.DraftTokensAccepted) / float64(e.stats.VerificationRounds)
	}
}

// Reset clears accumulated statistics and boundary state.
func (e *SpeculativeEngine) Reset() {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.stats = SpeculativeEngineStats{}
	e.lastLogits = nil
}

// ArgmaxF32 returns the index of the maximum float32 value in v.
func ArgmaxF32(v []float32) int {
	return argmaxF32(v)
}
