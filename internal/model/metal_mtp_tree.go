package model

import (
	"context"
	"math"
	"time"
)

// StepRoundTree executes one speculative candidate tree verification cycle.
// Evaluates the full candidate tree (M=16..24 nodes) in one forward pass,
// and extracts the highest-scoring verified token branch via greedy argmax selection.
func (c *MetalMTPCoordinator) StepRoundTree(ctx context.Context, committed []int, boundaryLogits []float32, tree *CandidateTree) (accepted []int, bonus int, nextLogits []float32, err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{}
	c.hasTargetVerification = false

	if c.closed {
		return nil, -1, nil, ErrMetalMTPClosed
	}
	if c.target == nil {
		return nil, -1, nil, ErrMetalMTPNilTarget
	}
	if err := ctx.Err(); err != nil {
		return nil, -1, nil, err
	}

	// 1. Temperature-zero logit validity tripwire
	if c.cfg.EnforceGreedyTripwire {
		for _, l := range boundaryLogits {
			if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
				c.tripwireTripped = true
				c.tripwireReason = "non-finite logits in boundary"
				return nil, -1, nil, ErrMetalMTPNonFiniteLogits
			}
		}
	}

	target0 := argmaxF32(boundaryLogits)

	if tree == nil || len(tree.Nodes) == 0 {
		nextLogits = c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}

	return c.stepRoundTreeLocked(time.Now(), target0, boundaryLogits, tree, -1, -1)
}

func (c *MetalMTPCoordinator) stepRoundTreeLocked(start time.Time, target0 int, boundaryLogits []float32, tree *CandidateTree, budget, eos int) ([]int, int, []float32, error) {
	N := len(tree.Nodes)
	ids := tree.Tokens()

	// Vocabulary sanity check: reject out-of-vocab candidates
	vocabSize := c.target.M.Cfg.VocabSize
	for _, tok := range ids {
		if tok < 0 || (vocabSize > 0 && tok >= vocabSize) {
			nextLogits := c.target.Step(target0)
			c.totalGenerated++
			return nil, target0, nextLogits, nil
		}
	}

	// Capture pre-round verified snapshot for exact rollback
	snap, snapErr := c.target.PrefixSnapshot()
	if snapErr != nil {
		nextLogits := c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}
	defer snap.Close()

	// Record draft in Context-MMU with speculative page tracking
	draftTokens32 := make([]int32, N)
	for i, t := range ids {
		draftTokens32[i] = int32(t)
	}
	c.draftState.RecordDraft(draftTokens32)
	if c.checkpointMgr != nil && c.sessionID != "" {
		_ = c.checkpointMgr.RecordMTPDraft(c.sessionID, draftTokens32)
	}

	baseLen := c.target.Cache.Len()
	pos := make([]int, N)
	for i, node := range tree.Nodes {
		pos[i] = baseLen + node.Depth
	}

	mask, mErr := tree.DeriveMask()
	if mErr != nil {
		_, _ = c.draftState.RollbackDraft()
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		_ = snap.Restore(c.target)
		nextLogits := c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}

	allow := func(q, k int) bool {
		if q >= 0 && q < len(mask) && k >= 0 && k < len(mask[q]) {
			return mask[q][k]
		}
		return false
	}

	// Single-pass verification forward across all tree candidates
	var rows [][]float32
	isBatched := false
	if verifyForwardBatchedOK(c.target) {
		rawRows := c.target.VerifyForward(ids, pos, allow)
		if len(rawRows) == N {
			rows = make([][]float32, len(rawRows))
			for i, r := range rawRows {
				rows[i] = append([]float32(nil), r...)
			}
			isBatched = true
		}
	}
	if !isBatched {
		// Fallback verification: step through candidate sequence
		rows = make([][]float32, N)
		for i, tok := range ids {
			stepLogits := c.target.Step(tok)
			rows[i] = append([]float32(nil), stepLogits...)
		}
	}

	if len(rows) != N {
		_, _ = c.draftState.RollbackDraft()
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
		}
		_ = snap.Restore(c.target)
		nextLogits := c.target.Step(target0)
		c.totalGenerated++
		return []int{target0}, -1, nextLogits, nil
	}

	// Greedy temperature-zero verification tripwire
	if c.cfg.EnforceGreedyTripwire {
		for _, row := range rows {
			for _, l := range row {
				if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
					c.tripwireTripped = true
					c.tripwireReason = "non-finite logits in verification rows"
					_, _ = c.draftState.RollbackDraft()
					if c.checkpointMgr != nil && c.sessionID != "" {
						_, _ = c.checkpointMgr.RollbackMTPDraft(c.sessionID)
					}
					_ = snap.Restore(c.target)
					return nil, -1, nil, ErrMetalMTPNonFiniteLogits
				}
			}
		}
	}

	// Greedy argmax path selection over tree proposals:
	// Extracts the highest-scoring verified token branch starting from root matching target0.
	matchRoot := -1
	for i, node := range tree.Nodes {
		if node.Parent == -1 && node.Token == target0 {
			matchRoot = i
			break
		}
	}

	if matchRoot == -1 {
		// Root candidate did not match target0: reject all draft tokens
		_, _, _ = c.draftState.CommitDraft(0)
		if c.checkpointMgr != nil && c.sessionID != "" {
			_, _, _ = c.checkpointMgr.CommitMTPDraft(c.sessionID, 0)
		}
		_ = snap.Restore(c.target)
		nextLogits := c.target.Step(target0)
		elapsed := time.Since(start)
		c.totalGenerated++
		c.recordAcceptanceLocked(N, 0)
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: N,
				AcceptedTokens: 0,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(N, 0, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return nil, target0, nextLogits, nil
	}

	acceptedIndices := []int{matchRoot}
	rejected := false
	cur := matchRoot
	pred := argmaxF32(rows[cur])

	for {
		nextChild := -1
		hasCandidate := false
		for _, childIdx := range tree.Nodes[cur].Children {
			if childIdx >= 0 && childIdx < N {
				hasCandidate = true
			}
			if childIdx >= 0 && childIdx < N && tree.Nodes[childIdx].Token == pred {
				nextChild = childIdx
				break
			}
		}
		if nextChild == -1 {
			rejected = hasCandidate
			break
		}
		acceptedIndices = append(acceptedIndices, nextChild)
		cur = nextChild
		pred = argmaxF32(rows[cur])
	}

	acceptedTokens := make([]int, len(acceptedIndices))
	for i, idx := range acceptedIndices {
		acceptedTokens[i] = tree.Nodes[idx].Token
	}
	bonusTok := pred
	// Bound the committed round to the caller's admission budget so target and
	// Context-MMU state never include a token discarded by maxNew/EOS (#12344).
	acceptedTokens, bonusTok = admitRoundPrefix(acceptedTokens, bonusTok, budget, eos)
	numAccepted := len(acceptedTokens)

	// Atomic Context-MMU page commit & rollback:
	// Commits accepted token pages and immediately frees rejected pages without memory leaks
	_, _, _ = c.draftState.CommitDraft(numAccepted)
	if c.checkpointMgr != nil && c.sessionID != "" {
		_, _, _ = c.checkpointMgr.CommitMTPDraft(c.sessionID, numAccepted)
	}

	// Restore target snapshot and advance sequentially with accepted branch
	_ = snap.Restore(c.target)
	for _, tok := range acceptedTokens {
		c.target.Step(tok)
	}

	// Advance target session with the bonus token. When the bonus was not
	// admitted the target already holds exactly prompt+accepted.
	if bonusTok < 0 {
		elapsed := time.Since(start)
		c.totalGenerated += numAccepted
		c.recordObservedAcceptanceLocked(N, numAccepted, len(acceptedIndices), rejected)
		if c.governor != nil {
			obs := Qwen38AdaptiveStepObservation{
				ProposedTokens: N,
				AcceptedTokens: numAccepted,
				StepLatency:    elapsed,
				TargetLatency:  c.targetLatency,
			}
			if c.stepCostFn != nil {
				obs = c.stepCostFn(N, numAccepted, obs)
			}
			_, _, _ = c.governor.ObserveStep(obs)
		}
		return acceptedTokens, bonusTok, boundaryLogits, nil
	}
	nextLogits := c.target.Step(bonusTok)
	elapsed := time.Since(start)
	c.totalGenerated += numAccepted + 1

	c.recordObservedAcceptanceLocked(N, numAccepted, len(acceptedIndices), rejected)

	if c.governor != nil {
		obs := Qwen38AdaptiveStepObservation{
			ProposedTokens: N,
			AcceptedTokens: numAccepted,
			StepLatency:    elapsed,
			TargetLatency:  c.targetLatency,
		}
		if c.stepCostFn != nil {
			obs = c.stepCostFn(N, numAccepted, obs)
		}
		_, _, _ = c.governor.ObserveStep(obs)
	}

	treePath := targetVerificationDecodePath
	treeTargetOps := 0
	treeTargetSteps := N
	treeDowngradeReason := ""
	if isBatched {
		treePath = targetVerificationBatchedPath
		treeTargetOps = 1
		treeTargetSteps = 0
	} else {
		treeDowngradeReason = "unsupported wide-M tree target shape; fell back to serial decode"
	}
	c.lastTargetVerification = MetalMTPTargetVerificationReceipt{
		TargetVerificationReceipt: TargetVerificationReceipt{
			Schema:                       targetVerificationReceiptSchema,
			Engine:                       targetVerificationEngine,
			Path:                         treePath,
			OneOperation:                 isBatched,
			TargetVerificationOperations: treeTargetOps,
			TargetDecodeSteps:            treeTargetSteps,
			DraftTokens:                  N,
			AcceptedTokens:               numAccepted,
			RejectedTokens:               N - numAccepted,
			DowngradeReason:              treeDowngradeReason,
			Accounting: SpeculativeCostAccounting{
				Setup:              SpeculativeCostComponent{Nanoseconds: time.Since(start).Nanoseconds(), Measured: true},
				TargetVerification: SpeculativeCostComponent{Nanoseconds: time.Since(start).Nanoseconds(), Measured: true},
				KnownMemoryBytes:   snap.ResidentBytes(),
				MemoryMeasured:     true,
			},
		},
	}
	c.hasTargetVerification = true

	return acceptedTokens, bonusTok, nextLogits, nil
}
