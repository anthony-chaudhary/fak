package agent

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// greedySpeculativeRequestEligible keeps the target verifier's exact-greedy
// contract at the request boundary. Sampling and score transforms retain the
// ordinary target path; top-p and top-k are inert when temperature is zero.
func (p *InKernelPlanner) greedySpeculativeRequestEligible(temp float64, logitBias model.LogitBias, freqPenalty, presPenalty float64) bool {
	return p.speculativeEngine != nil && temp <= 0 && len(logitBias) == 0 && freqPenalty == 0 && presPenalty == 0
}

// SetSpeculativeEngine configures the speculative decoding engine for this planner.
func (p *InKernelPlanner) SetSpeculativeEngine(eng *model.SpeculativeEngine) {
	p.speculativeEngine = eng
	p.vulkanMTP = false
}

// SpeculativeEngine returns the configured speculative decoding engine, if any.
func (p *InKernelPlanner) SpeculativeEngine() *model.SpeculativeEngine {
	return p.speculativeEngine
}

// EnableSpeculativeDecoding enables speculative decoding using the given proposal generator and draft depth K.
func (p *InKernelPlanner) EnableSpeculativeDecoding(gen model.ProposalGenerator, draftDepth int) {
	if draftDepth <= 0 {
		draftDepth = 4
	}
	p.specDraftDepth = draftDepth
	cfg := model.DefaultSpeculativeEngineConfig()
	cfg.MaxDraft = draftDepth
	cfg.Temperature = p.temp
	p.speculativeEngine = model.NewSpeculativeEngine(nil, gen, cfg)
	p.vulkanMTP = false
}

// EnableVulkanMTP enables request-bound resident Vulkan MTP drafting. The
// draft session itself is constructed only after the request's target session
// exists, so mutable draft state is never shared across requests.
func (p *InKernelPlanner) EnableVulkanMTP(draftDepth int) error {
	if draftDepth <= 0 {
		draftDepth = model.Qwen35MTPMaxDraftDepth
	}
	if draftDepth > model.Qwen35MTPMaxDraftDepth {
		return fmt.Errorf("agent: Vulkan MTP draft depth %d exceeds supported maximum %d", draftDepth, model.Qwen35MTPMaxDraftDepth)
	}
	p.specDraftDepth = draftDepth
	cfg := model.DefaultSpeculativeEngineConfig()
	cfg.MaxDraft = draftDepth
	cfg.Temperature = p.temp
	p.speculativeEngine = model.NewSpeculativeEngine(nil, nil, cfg)
	p.vulkanMTP = true
	return nil
}

// VulkanMTPEnabled reports planner admission only. Request responses use the
// VulkanMTPExecution receipt to distinguish actual execution from downgrade.
func (p *InKernelPlanner) VulkanMTPEnabled() bool {
	return p != nil && p.vulkanMTP && p.speculativeEngine != nil
}

// DisableSpeculativeDecoding disables speculative decoding on this planner.
func (p *InKernelPlanner) DisableSpeculativeDecoding() {
	p.speculativeEngine = nil
	p.specDraftDepth = 0
	p.vulkanMTP = false
}

// SetMetalMTPCoordinator configures an explicit MetalMTPCoordinator on this planner.
// This is the pre-existing explicit operator API: installing a coordinator this way
// records an explicit override, so the canary evidence gate is bypassed (the caller
// has taken responsibility for execution). Use ConfigureQwen38MTPCanary for the
// evidence-bound path.
func (p *InKernelPlanner) SetMetalMTPCoordinator(c *model.MetalMTPCoordinator) {
	p.metalMTPMu.Lock()
	p.metalMTPCoordinator = c
	if c != nil {
		p.mtpExplicitOverride = true
	}
	p.metalMTPMu.Unlock()
}

// MetalMTPCoordinator returns the active MetalMTPCoordinator, if any.
func (p *InKernelPlanner) MetalMTPCoordinator() *model.MetalMTPCoordinator {
	p.metalMTPMu.Lock()
	defer p.metalMTPMu.Unlock()
	return p.metalMTPCoordinator
}

// EnableMetalMTP enables the in-kernel Metal MTP draft-verify-rollback execution loop.
// Installing a coordinator this way is NOT an explicit override: without a canary
// manager bound to witnessed qualification evidence, qwen38MTPCanaryAllowsExecution
// stays fail-closed and the planner runs ordinary target decode. Callers that have
// witnessed evidence must bind it through ConfigureQwen38MTPCanary; callers that are
// deliberately overriding the gate must use SetMetalMTPCoordinator.
func (p *InKernelPlanner) EnableMetalMTP(cfg ...model.MetalMTPConfig) error {
	c, err := model.NewMetalMTPCoordinator(nil, cfg...)
	if err != nil {
		return err
	}
	p.metalMTPMu.Lock()
	p.metalMTPCoordinator = c
	p.mtpExplicitOverride = false
	p.metalMTPMu.Unlock()
	return nil
}

// installMetalMTPCoordinator installs a coordinator without touching override or
// canary state. It backs ConfigureQwen38MTPCanary, whose decision already governs
// execution.
func (p *InKernelPlanner) installMetalMTPCoordinator(cfg ...model.MetalMTPConfig) error {
	c, err := model.NewMetalMTPCoordinator(nil, cfg...)
	if err != nil {
		return err
	}
	p.metalMTPMu.Lock()
	p.metalMTPCoordinator = c
	p.metalMTPMu.Unlock()
	return nil
}

// DisableMetalMTP disables the Metal MTP execution loop on this planner.
func (p *InKernelPlanner) DisableMetalMTP() {
	p.metalMTPMu.Lock()
	coord := p.metalMTPCoordinator
	p.metalMTPCoordinator = nil
	p.mtpExplicitOverride = false
	p.metalMTPMu.Unlock()
	if coord != nil {
		_ = coord.Close()
	}
}

// ConfigureQwen38MTPCanary connects witnessed canary admission to the planner's
// real Metal MTP decode branch. A fresh matching witness or explicit operator
// opt-in enables the coordinator; every other decision leaves ordinary
// fak-native target decode selected. The kill switch is retained and rechecked
// on every generation so an operator can revoke an already-admitted planner.
func (p *InKernelPlanner) ConfigureQwen38MTPCanary(
	manager *model.Qwen38MTPCanaryManager,
	request model.Qwen38CanaryRequest,
	killSwitch *model.Qwen38MTPKillSwitch,
	cfg ...model.MetalMTPConfig,
) (model.Qwen38CanaryDecision, error) {
	if manager == nil {
		manager = model.NewQwen38MTPCanaryManager()
	}
	decision := manager.EvaluateCanary(request)
	if killSwitch != nil {
		if allowed, reason := killSwitch.CheckEligible(); !allowed {
			decision = qwen38MTPTargetOnlyResult(request.Envelope, reason, "MTP runtime kill switch is engaged")
		}
	}

	p.mtpCanaryMu.Lock()
	p.mtpCanaryManager = manager
	p.mtpCanaryRequest = request
	p.mtpKillSwitch = killSwitch
	p.mtpCanaryResult = decision
	p.mtpCanaryMu.Unlock()

	if decision.Engine != model.Qwen38EngineMTP {
		p.DisableMetalMTP()
		return decision, nil
	}
	p.DisableMetalMTP()
	if err := p.installMetalMTPCoordinator(cfg...); err != nil {
		decision = qwen38MTPTargetOnlyResult(request.Envelope, model.Qwen38MTPAttemptFailed, err.Error())
		p.mtpCanaryMu.Lock()
		p.mtpCanaryResult = decision
		p.mtpCanaryMu.Unlock()
		return decision, err
	}
	return decision, nil
}

// Qwen38MTPCanaryResult returns the last production admission result. The
// returned value is a copy and cannot mutate planner state.
func (p *InKernelPlanner) Qwen38MTPCanaryResult() model.Qwen38CanaryDecision {
	p.mtpCanaryMu.RLock()
	defer p.mtpCanaryMu.RUnlock()
	return p.mtpCanaryResult
}

func (p *InKernelPlanner) qwen38MTPCanaryAllowsExecution() bool {
	p.mtpCanaryMu.RLock()
	manager := p.mtpCanaryManager
	request := p.mtpCanaryRequest
	killSwitch := p.mtpKillSwitch
	p.mtpCanaryMu.RUnlock()
	p.metalMTPMu.Lock()
	explicitOverride := p.mtpExplicitOverride
	p.metalMTPMu.Unlock()

	// Fail-closed when no canary manager is bound to witnessed qualification
	// evidence: an unconfigured planner must not execute speculative decode. The
	// one exception is a coordinator installed through the pre-existing explicit
	// API (SetMetalMTPCoordinator), where the caller has deliberately taken the
	// admission decision — that path stays an explicit operator override.
	if manager == nil {
		if explicitOverride {
			return true
		}
		p.mtpCanaryMu.Lock()
		p.mtpCanaryResult = qwen38MTPTargetOnlyResult(request.Envelope, model.Qwen38MTPEvidenceMissing, "no witnessed MTP canary admission is configured for this planner")
		p.mtpCanaryMu.Unlock()
		return false
	}
	decision := manager.EvaluateCanary(request)
	if killSwitch != nil {
		if allowed, reason := killSwitch.CheckEligible(); !allowed {
			decision = qwen38MTPTargetOnlyResult(request.Envelope, reason, "MTP runtime kill switch is engaged")
		}
	}
	p.mtpCanaryMu.Lock()
	p.mtpCanaryResult = decision
	p.mtpCanaryMu.Unlock()
	return decision.Engine == model.Qwen38EngineMTP
}

// MetalMTPAdmitted reports whether Metal MTP speculative decode is admitted for
// actual execution on this planner right now. It is the single truth that both
// the decode dispatch and any reported speculative status must consult: it is
// true only when a coordinator is installed AND the canary admission (or an
// explicit operator override) currently selects the MTP engine. It also refreshes
// the stored Qwen38MTPCanaryResult so reported status agrees with execution.
func (p *InKernelPlanner) MetalMTPAdmitted() bool {
	if p == nil || p.MetalMTPCoordinator() == nil {
		return false
	}
	return p.qwen38MTPCanaryAllowsExecution()
}

func qwen38MTPTargetOnlyResult(envelope model.Qwen38CanaryEnvelope, reason model.Qwen38MTPDowngradeReason, detail string) model.Qwen38CanaryDecision {
	return model.Qwen38CanaryDecision{
		Engine:          model.Qwen38EngineTargetDecode,
		DowngradeReason: reason,
		RejectionReason: detail,
		Envelope:        envelope,
	}
}

func (p *InKernelPlanner) generateReusedSpeculative(
	ctx context.Context,
	ids []int,
	maxNew int,
	temp, topP float64,
	topK int,
	logitBias model.LogitBias,
	freqPenalty, presPenalty float64,
	stops map[int]bool,
	emit func(int) bool,
	measurementOpt ...*nativeInferenceMeasurement,
) (res inKernelGenerateResult, err error) {
	// Keep request-local route evidence when a backend panic is converted to the
	// same typed device error as the outer decode boundary. Recovering only in
	// generateReusedRecovering would discard this function's named result.
	defer func() {
		if r := recover(); r != nil {
			if e, ok := recoverDevicePanic(r); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	promptTok := len(ids)
	var mtpExecution *VulkanMTPExecution
	if p.VulkanMTPEnabled() {
		backendName := ""
		if p.backend != nil {
			backendName = p.backend.Name()
		}
		mtpExecution = &VulkanMTPExecution{
			Engine:         vulkanMTPExecutionEngine,
			Backend:        backendName,
			RequestedDepth: p.specDraftDepth,
		}
		started := time.Now()
		defer func() {
			mtpExecution.Elapsed = time.Since(started)
			if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
				mtpExecution.Cancelled = true
				if mtpExecution.DowngradeReason == "" {
					mtpExecution.DowngradeReason = vulkanMTPDowngradeCancelled
				}
			}
			res.vulkanMTP = mtpExecution
		}()
	}
	var measurement *nativeInferenceMeasurement
	if len(measurementOpt) > 0 {
		measurement = measurementOpt[0]
	}
	if promptTok == 0 {
		return inKernelGenerateResult{}, nil
	}
	if err = ctx.Err(); err != nil {
		return inKernelGenerateResult{}, err
	}

	reuse := p.tree != nil && inKernelPlannerPrefixReuseSupported(p.m, p.backend)
	var s *model.Session
	var cachedLogits []float32
	var matched, cacheable int
	// structural is the longest token prefix visible in the tree (the divergence
	// point), which may exceed the restorable `matched` on a recurrent hybrid. It only
	// selects where prefill materializes a checkpoint; reporting keeps cacheable.
	var structural int
	var sourceTier radixkv.SnapshotTier
	skipExactDeviceL1Readmission := false

	if reuse {
		owner, scoped := prefixCacheIdentityFromContext(ctx)
		scopedLookup := scoped && p.scopedTree != nil
		var matchedKV *model.KVCache
		var matchedSnapshot *model.PrefixSnapshot
		var m int
		var tier radixkv.SnapshotTier
		var sourceScope radixkv.ShareScope
		if scopedLookup {
			if p.backend != nil {
				if visible, visErr := p.scopedTree.MatchLen(owner, ids); visErr == nil {
					structural = visible
				}
				matchedSnapshot, cachedLogits, m, sourceScope, tier, err = p.scopedTree.LookupSnapshotTieredContext(ctx, owner, ids)
			} else {
				matchedKV, cachedLogits, m, _, err = p.scopedTree.Lookup(owner, ids)
			}
		} else {
			p.mu.Lock()
			if p.backend != nil {
				b, snap, legacyMatched, lookupTier, lookupErr := p.tree.LookupSnapshotTieredContext(ctx, ids)
				matchedSnapshot, m, err = snap, legacyMatched, lookupErr
				if b != nil {
					structural = b.Plen()
				}
				tier = lookupTier
				if m >= len(ids) {
					cachedLogits = b.Logits()
				}
				p.tree.Done(b)
			} else {
				b, legacyMatched := p.tree.Lookup(ids)
				m = legacyMatched
				if k := b.KV(); k != nil {
					matchedKV = k.Clone()
					if m >= len(ids) {
						cachedLogits = b.Logits()
					}
				}
				p.tree.Done(b)
			}
			p.mu.Unlock()
		}
		if err != nil {
			return inKernelGenerateResult{}, err
		}
		cacheable = m
		if matchedSnapshot != nil {
			s = p.m.NewBackendSession(p.backend)
			if err = matchedSnapshot.Restore(s); err != nil {
				matchedSnapshot.Close()
				s.Close()
				return inKernelGenerateResult{}, err
			}
			matchedSnapshot.Close()
			matched, sourceTier = m, tier
		} else if matchedKV != nil {
			s = p.sessionFromPrefixClone(matchedKV)
			matched = m
			sourceTier = radixkv.SnapshotTierDeviceL1
		}

		if s != nil && matched >= len(ids) && cachedLogits == nil {
			if inKernelRefeedLastTokenForExactHit(s, len(ids)) {
				matched = len(ids) - 1
			} else {
				s.Close()
				s, matched = nil, 0
				sourceTier = radixkv.SnapshotTierMiss
			}
		}
		skipExactDeviceL1Readmission = ((!scopedLookup) || (scopedLookup && (sourceScope == radixkv.ScopeTenant || sourceScope == radixkv.ScopeAgent))) &&
			matchedSnapshot != nil && matched == len(ids) && cachedLogits != nil && sourceTier == radixkv.SnapshotTierDeviceL1
	}

	if s == nil {
		matched = 0
		s = p.newSpeculativeSession()
	}
	defer func() {
		if p.specSessionCloseHook != nil {
			p.specSessionCloseHook(s)
		}
		s.Close()
	}()
	p.configureNativeSession(s)

	eng := p.speculativeEngine
	// Each request begins with a clean rolling acceptance window so a prior
	// request's degraded draft head cannot force this one onto serial decode.
	eng.ResetAcceptanceMonitor()
	proposalGenerator := eng.PrimaryGenerator()
	if mtpExecution != nil {
		var closeDraft func()
		proposalGenerator, closeDraft, err = p.newVulkanMTPProposalGenerator(s, p.specDraftDepth)
		if err != nil && closeDraft != nil {
			closeDraft()
			closeDraft = nil
		}
		if err != nil && matched > 0 {
			// A snapshot created before raw-hidden capture was enabled cannot bind a
			// resident MTP draft. Rebuild from a fresh target before prompt prefill;
			// the failed constructor leaves the restored target unchanged.
			s.Close()
			s = p.newSpeculativeSession()
			p.configureNativeSession(s)
			matched = 0
			cachedLogits = nil
			sourceTier = radixkv.SnapshotTierMiss
			skipExactDeviceL1Readmission = false
			proposalGenerator, closeDraft, err = p.newVulkanMTPProposalGenerator(s, p.specDraftDepth)
			if err != nil && closeDraft != nil {
				closeDraft()
				closeDraft = nil
			}
		}
		if err == nil && proposalGenerator == nil {
			if closeDraft != nil {
				closeDraft()
				closeDraft = nil
			}
			err = model.ErrSpeculativeNilGenerator
		}
		if err != nil {
			mtpExecution.DowngradeReason = vulkanMTPDowngradeConstructorRefused
			proposalGenerator = nil
			err = nil
		} else if closeDraft != nil {
			mtpExecution.EffectiveDepth = p.specDraftDepth
			defer closeDraft()
		} else {
			mtpExecution.EffectiveDepth = p.specDraftDepth
		}
	}

	p.recordTurnTax(promptTok, cacheable, matched)
	enginestep.Default.ObservePrefixQueried(promptTok)
	enginestep.Default.ObservePrefixMatched(matched)

	// Prefill divergent prompt tokens
	logits := cachedLogits
	var prefillS float64
	if logits == nil {
		tp := time.Now()
		prefillAt := matched
		if reuse && p.backend != nil {
			// Same checkpoint plan as the target-only path: the shared system/tools
			// boundary plus the structural divergence point (falling back to the 64-token
			// grid), so speculative/MTP fan-out children restore the parent's prefix.
			if structural < matched {
				structural = matched
			}
			sharedBoundary := inKernelSharedPrefixBoundaryFromContext(ctx)
			required := inKernelAdaptiveSnapshotCheckpoint(prefillAt, structural, len(ids))
			for _, checkpoint := range inKernelPrefillCheckpoints(prefillAt, structural, sharedBoundary, len(ids)) {
				logits = s.Prefill(ids[prefillAt:checkpoint])
				if admitErr := p.admitPrefillCheckpoint(ctx, s, ids[:checkpoint], logits, checkpoint != required); admitErr != nil {
					return inKernelGenerateResult{}, admitErr
				}
				prefillAt = checkpoint
			}
		}
		if prefillAt < len(ids) {
			rawLogits, err := p.prefillDivergentSuffix(ctx, s, ids[prefillAt:], measurementOpt...)
			if err != nil {
				p.recordNativePhase(nativePhaseTraceID(ctx), NativePhasePrefill, tp, time.Since(tp), false)
				return inKernelGenerateResult{}, err
			}
			logits = append([]float32(nil), rawLogits...)
		}
		prefillS = time.Since(tp).Seconds()
		enginestep.Default.ObservePhase(enginestep.PhasePrefill, time.Since(tp))
		p.recordNativePhase(nativePhaseTraceID(ctx), NativePhasePrefill, tp, time.Since(tp), true)
	} else {
		logits = append([]float32(nil), cachedLogits...)
	}
	if err = ctx.Err(); err != nil {
		return inKernelGenerateResult{}, err
	}

	// Admit full prompt to prefix cache BEFORE speculative decode mutates cache
	if reuse {
		if p.backend != nil {
			if !skipExactDeviceL1Readmission {
				snapshot, snapshotErr := s.PrefixSnapshot()
				if snapshotErr != nil {
					return inKernelGenerateResult{}, snapshotErr
				}
				if admitErr := p.admitPrefixSnapshot(ctx, ids, snapshot, logits); admitErr != nil {
					snapshot.Close()
					return inKernelGenerateResult{}, admitErr
				}
			}
		} else if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped && p.scopedTree != nil {
			if admitErr := p.scopedTree.AdmitPrivate(owner, ids, s.Cache, logits); admitErr != nil {
				return inKernelGenerateResult{}, admitErr
			}
		} else {
			p.mu.Lock()
			b, m := p.tree.Lookup(ids)
			leaf := p.tree.InsertCloneWithLogits(b, ids[m:], s.Cache, logits)
			p.tree.Done(leaf)
			p.mu.Unlock()
		}
	}

	// Speculative decoding loop
	measurement.startDecodeTrace()
	td := time.Now()
	committed := append([]int(nil), ids...)
	var counts []int32
	if freqPenalty != 0 || presPenalty != 0 {
		counts = make([]int32, len(logits))
	}
	rng := rand.New(rand.NewSource(p.seed))

	gen := 0
	stopped := false
	curLogits := logits
	maxDraft := eng.Config().MaxDraft
	if maxDraft <= 0 {
		maxDraft = 4
	}

	// A verified round stays open through its bonus Step so the recorded round time is
	// one whole draft-verify-advance cycle; it closes at the next iteration or after the loop.
	var round speculativeRoundObservation

	for gen < maxNew {
		round.close(gen)
		if err = ctx.Err(); err != nil {
			break
		}

		// Native rolling acceptance fallback (Qwen3.8 MTP): once the engine's
		// trailing-window acceptance rate falls below the configured floor, stop
		// drafting and continue with unassisted serial decode. The proposal
		// generator is dropped rather than retried each token, so a degraded
		// draft head cannot stall the session.
		if proposalGenerator != nil && eng.InFallback() {
			proposalGenerator = nil
			if mtpExecution != nil {
				mtpExecution.DowngradeReason = vulkanMTPDowngradeAcceptanceFloor
			}
		}

		// Propose speculative candidate tokens
		var proposal model.DraftProposal
		if proposalGenerator != nil {
			roundDraft := maxDraft
			if remaining := maxNew - gen; roundDraft > remaining {
				roundDraft = remaining
			}
			var propErr error
			proposal, propErr = proposalGenerator.Propose(ctx, committed, roundDraft)
			if propErr != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					err = ctxErr
					break
				}
				proposal = model.DraftProposal{}
				if mtpExecution != nil {
					mtpExecution.DowngradeReason = vulkanMTPDowngradeProposalError
					proposalGenerator = nil
				}
			} else if proposal.Tree == nil && len(proposal.Tokens) > roundDraft {
				proposal.Tokens = append([]int(nil), proposal.Tokens[:roundDraft]...)
			}
		}
		if err = ctx.Err(); err != nil {
			break
		}

		// If proposal has no tokens and no tree, fall back to single-step autoregressive
		if len(proposal.Tokens) == 0 && proposal.Tree == nil {
			if mtpExecution != nil && proposalGenerator != nil {
				mtpExecution.DowngradeReason = vulkanMTPDowngradeEmptyProposal
				proposalGenerator = nil
			}
			next := sampleLogitsWithPenalty(curLogits, temp, topP, topK, logitBias, freqPenalty, presPenalty, counts, rng)
			if next < 0 || stops[next] {
				stopped = true
				break
			}
			if err = measurement.record(curLogits, next); err != nil {
				break
			}
			if counts != nil && next < len(counts) {
				counts[next]++
			}
			emitStopped := emit != nil && emit(next)
			gen++
			committed = append(committed, next)
			if err = measurement.recordDecodeTrace(gen, next); err != nil {
				break
			}
			if emitStopped || gen == maxNew {
				stopped = emitStopped
				break
			}
			if emit != nil {
				if err = ctx.Err(); err != nil {
					break
				}
			}
			stepStart := time.Now()
			curLogits = s.Step(next)
			enginestep.Default.ObserveDecodeStep(enginestep.PathSerial, 1, time.Since(stepStart))
			continue
		}

		// Evaluate candidate proposal using parallel verification kernel
		if mtpExecution != nil {
			mtpExecution.Used = true
			mtpExecution.ProposalRounds++
			mtpExecution.ProposedTokens += len(proposal.Tokens)
		}
		roundStart := time.Now()
		vRes, deviceReceipt, vErr := verifyGreedySpeculativeRoundObserved(ctx, s, committed, proposal, curLogits, eng.Sanitizer(), counts)
		if vErr != nil {
			err = vErr
			break
		}
		round.open(roundStart, speculativeProposalSize(proposal), vRes.NumAccepted, gen)
		if mtpExecution != nil {
			mtpExecution.AcceptedTokens += vRes.NumAccepted
			mtpExecution.RejectedTokens += len(proposal.Tokens) - vRes.NumAccepted
			mtpExecution.RollbackTokens += vRes.RollbackKVCount
			if deviceReceipt == nil {
				mtpExecution.DowngradeReason = vulkanMTPDowngradeTargetVerifier
				// The generic verifier completed this round exactly. Stop drafting
				// now instead of retrying an unavailable resident verifier on every
				// remaining token.
				proposalGenerator = nil
			} else {
				mtpExecution.TargetOperations += deviceReceipt.TargetVerificationOperations
				mtpExecution.TargetDecodeSteps += deviceReceipt.TargetDecodeSteps
				mtpExecution.FullTargetReplaySteps += deviceReceipt.FullTargetReplaySteps
				mtpExecution.RecurrentRepairTokens += deviceReceipt.RecurrentRepairTokens
			}
		}
		var acceptedChoiceLogits [][]float32
		var bonusChoiceLogits []float32
		if measurement != nil {
			var choiceErr error
			acceptedChoiceLogits, bonusChoiceLogits, choiceErr = speculativeVerificationChoiceLogits(proposal, vRes, curLogits)
			if choiceErr != nil {
				err = choiceErr
				eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, false)
				break
			}
		}

		// Accept verified tokens
		roundStopped := false
		// emittedThisRound counts accepted tokens actually emitted to the caller.
		// ParallelVerifyKernel appended EVERY accepted draft token to the target
		// KV during its single batched forward, so the resident cache holds
		// committed + len(vRes.AcceptedTokens) even though this loop may stop
		// early (ctx cancellation, a token-ID stop, the emit callback returning
		// true, or gen == maxNew). The un-emitted accepted suffix is then a
		// PHANTOM SUFFIX: state the caller never received. Increment only at the
		// exact commit point (after the stop/maxNew checks pass) so the rollback
		// below removes precisely the tokens that were verified but not emitted
		// (#12422 defect 3).
		emittedThisRound := 0
		for i, tok := range vRes.AcceptedTokens {
			if err = ctx.Err(); err != nil {
				roundStopped = true
				break
			}
			if tok < 0 || stops[tok] {
				// A token-ID stop is neither emitted nor stepped; leave it out of
				// emittedThisRound so the rollback also drops it from KV.
				roundStopped = true
				stopped = true
				break
			}
			if measurement != nil {
				if err = measurement.record(acceptedChoiceLogits[i], tok); err != nil {
					roundStopped = true
					break
				}
			}
			if counts != nil && tok < len(counts) {
				counts[tok]++
			}
			emitStopped := emit != nil && emit(tok)
			gen++
			committed = append(committed, tok)
			emittedThisRound++
			if err = measurement.recordDecodeTrace(gen, tok); err != nil {
				roundStopped = true
				break
			}
			if emitStopped || gen == maxNew {
				roundStopped = true
				stopped = emitStopped
				break
			}
			if emit != nil {
				if err = ctx.Err(); err != nil {
					roundStopped = true
					break
				}
			}
		}

		// Boundary-state parity: the session must return holding exactly the
		// emitted tokens. The un-emitted accepted suffix is the current KV tail
		// (ParallelVerifyKernel already evicted only the rejected draft suffix),
		// so truncating from the end by that count is exact. RollbackResidentSuffix
		// routes through the backend-aware eviction seam, so a device session
		// releases its HAL store too — RollbackSpeculative would touch only the
		// host Cache and leave the device phantom suffix resident. This is a no-op
		// on the normal path, where every accepted token was emitted.
		if unemitted := len(vRes.AcceptedTokens) - emittedThisRound; unemitted > 0 {
			s.RollbackResidentSuffix(unemitted)
		}

		if roundStopped || gen == maxNew {
			eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, false)
			break
		}

		// Bonus / correction token
		if err = ctx.Err(); err != nil {
			eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, false)
			break
		}
		bonus := vRes.CorrectionToken
		if bonus < 0 || stops[bonus] {
			stopped = true
			eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, false)
			break
		}
		if measurement != nil {
			if err = measurement.record(bonusChoiceLogits, bonus); err != nil {
				eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, false)
				break
			}
		}
		if counts != nil && bonus < len(counts) {
			counts[bonus]++
		}
		emitStopped := emit != nil && emit(bonus)
		gen++
		committed = append(committed, bonus)
		if err = measurement.recordDecodeTrace(gen, bonus); err != nil {
			eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, true)
			break
		}
		eng.RecordVerificationOutcome(len(proposal.Tokens), vRes.NumAccepted, vRes.RollbackKVCount, true)
		if emitStopped || gen == maxNew {
			stopped = emitStopped
			break
		}
		if emit != nil {
			if err = ctx.Err(); err != nil {
				break
			}
		}

		// Advance target session with bonus token
		curLogits = s.Step(bonus)
	}

	round.close(gen)
	decodeS := time.Since(td).Seconds()
	enginestep.Default.ObservePhase(enginestep.PhaseDecode, time.Since(td))
	p.recordNativePhase(nativePhaseTraceID(ctx), NativePhaseDecode, td, time.Since(td), err == nil)
	p.recordNativePhase(nativePhaseTraceID(ctx), NativePhaseTerminal, time.Now(), 0, err == nil)
	return inKernelGenerateResult{
		gen:        gen,
		promptTok:  promptTok,
		cacheable:  cacheable,
		matched:    matched,
		sourceTier: sourceTier,
		prefillS:   prefillS,
		decodeS:    decodeS,
		stopped:    stopped,
		vulkanMTP:  mtpExecution,
		spec:       round.tally,
	}, err
}

// speculativeRoundObservation carries one verified draft round to enginestep. The
// round's emitted tokens are only known once its accept/bonus loop finishes, so
// it is opened after verification and closed at the next round or decode exit.
type speculativeRoundObservation struct {
	start              time.Time
	proposed, accepted int
	genAtOpen          int
	active             bool
	// tally accumulates every closed round of the request; open never resets it.
	tally SpeculativeDecodeTally
}

func (r *speculativeRoundObservation) open(start time.Time, proposed, accepted, gen int) {
	r.start, r.proposed, r.accepted, r.genAtOpen, r.active = start, proposed, accepted, gen, true
}

func (r *speculativeRoundObservation) close(gen int) {
	if !r.active {
		return
	}
	r.active = false
	r.tally.Rounds++
	r.tally.DraftTokens += r.proposed
	r.tally.AcceptedTokens += r.accepted
	enginestep.Default.ObserveSpeculativeRound(r.proposed, r.accepted, gen-r.genAtOpen, time.Since(r.start))
}

func speculativeProposalSize(p model.DraftProposal) int {
	if len(p.Tokens) > 0 || p.Tree == nil {
		return len(p.Tokens)
	}
	return len(p.Tree.Nodes)
}

func (p *InKernelPlanner) newSpeculativeSession() *model.Session {
	if p.backend != nil {
		return p.m.NewBackendSession(p.backend)
	}
	return p.m.NewSession()
}

func (p *InKernelPlanner) newVulkanMTPProposalGenerator(target *model.Session, depth int) (model.ProposalGenerator, func(), error) {
	if p.vulkanMTPDraftFactory != nil {
		return p.vulkanMTPDraftFactory(target, depth)
	}
	draft, err := model.NewQwen35MTPDraftSession(target, depth)
	if err != nil {
		return nil, nil, err
	}
	return model.NewMTPProposalGenerator(draft), draft.Close, nil
}

// verifyGreedySpeculativeRound selects the strict resident-device transaction
// for a bounded linear draft. A typed capability downgrade retains the existing
// verifier, whose linear fallback is ordinary target decode. Other device errors
// propagate instead of being hidden behind a second execution.
func verifyGreedySpeculativeRound(
	ctx context.Context,
	target *model.Session,
	committed []int,
	proposal model.DraftProposal,
	lastLogits []float32,
	sanitizer *model.RepetitionPenaltySanitizer,
	counts []int32,
) (model.VerificationResult, error) {
	result, _, err := verifyGreedySpeculativeRoundObserved(ctx, target, committed, proposal, lastLogits, sanitizer, counts)
	return result, err
}

func verifyGreedySpeculativeRoundObserved(
	ctx context.Context,
	target *model.Session,
	committed []int,
	proposal model.DraftProposal,
	lastLogits []float32,
	sanitizer *model.RepetitionPenaltySanitizer,
	counts []int32,
) (model.VerificationResult, *model.TargetVerificationReceipt, error) {
	if qwen35DeviceVerifierAvailable(target) && proposal.Tree == nil && len(proposal.Tokens) > 0 {
		device, err := target.VerifyGreedyDeviceDraft(ctx, proposal.Tokens, lastLogits)
		if err == nil {
			rollback := len(proposal.Tokens) - len(device.Accepted)
			if device.Receipt.TargetVerificationOperations == 0 && device.Receipt.TargetDecodeSteps == 0 {
				// A known-boundary first-token rejection performs no target
				// mutation, so there is no speculative state to roll back.
				rollback = 0
			}
			return model.VerificationResult{
				AcceptedTokens:  device.Accepted,
				CorrectionToken: device.Correction,
				NumAccepted:     len(device.Accepted),
				RollbackKVCount: rollback,
				TargetLogits:    device.TargetLogits,
			}, &device.Receipt, nil
		}
		if !errors.Is(err, model.ErrTargetVerificationDowngrade) {
			return model.VerificationResult{}, nil, err
		}
	}
	result, err := model.ParallelVerifyKernel(ctx, target, committed, proposal, lastLogits, sanitizer, counts)
	return result, nil, err
}

// speculativeVerificationChoiceLogits maps each emitted target decision back to
// the logits that selected it. Linear proposals use the round boundary followed
// by the preceding target row. Tree proposals follow the accepted parent chain.
func speculativeVerificationChoiceLogits(
	proposal model.DraftProposal,
	result model.VerificationResult,
	boundary []float32,
) (accepted [][]float32, bonus []float32, err error) {
	accepted = make([][]float32, len(result.AcceptedTokens))
	if proposal.Tree == nil {
		bonus = boundary
		for i := range result.AcceptedTokens {
			if i == 0 {
				accepted[i] = boundary
			} else {
				if i-1 >= len(result.TargetLogits) {
					return nil, nil, fmt.Errorf("speculative verification returned %d target rows for %d accepted tokens", len(result.TargetLogits), len(result.AcceptedTokens))
				}
				accepted[i] = result.TargetLogits[i-1]
			}
		}
		if len(result.AcceptedTokens) > 0 {
			row := len(result.AcceptedTokens) - 1
			if row >= len(result.TargetLogits) {
				return nil, nil, fmt.Errorf("speculative verification returned %d target rows for correction after %d accepted tokens", len(result.TargetLogits), len(result.AcceptedTokens))
			}
			bonus = result.TargetLogits[row]
		}
		return accepted, bonus, nil
	}

	parent := -1
	next := boundary
	for i, token := range result.AcceptedTokens {
		accepted[i] = next
		nodeIndex := -1
		for candidate, node := range proposal.Tree.Nodes {
			if node.Parent == parent && node.Token == token {
				nodeIndex = candidate
				break
			}
		}
		if nodeIndex < 0 || nodeIndex >= len(result.TargetLogits) {
			return nil, nil, fmt.Errorf("speculative tree result does not map accepted token %d at depth %d to a target row", token, i)
		}
		parent = nodeIndex
		next = result.TargetLogits[nodeIndex]
	}
	return accepted, next, nil
}

func qwen35DeviceVerifierAvailable(target *model.Session) bool {
	if target == nil || target.M == nil || target.Backend == nil || !target.M.Cfg.IsQwen35Hybrid() {
		return false
	}
	backend, ok := target.Backend.(compute.Qwen35SequenceAllLogitsBackend)
	return ok && backend.Qwen35SequenceAllLogitsPath() == compute.Qwen35SequenceAllLogitsPath
}

func (p *InKernelPlanner) generateReusedMetalMTP(
	ctx context.Context,
	ids []int,
	maxNew int,
	temp, topP float64,
	topK int,
	logitBias model.LogitBias,
	freqPenalty, presPenalty float64,
	stops map[int]bool,
	emit func(int) bool,
	measurementOpt ...*nativeInferenceMeasurement,
) (res inKernelGenerateResult, err error) {
	promptTok := len(ids)
	if promptTok == 0 {
		return inKernelGenerateResult{}, nil
	}
	if err = ctx.Err(); err != nil {
		return inKernelGenerateResult{}, err
	}
	p.metalMTPMu.Lock()
	defer p.metalMTPMu.Unlock()

	reuse := p.tree != nil && inKernelPlannerPrefixReuseSupported(p.m, p.backend)
	var s *model.Session
	var cachedLogits []float32
	var matched, cacheable int
	var sourceTier radixkv.SnapshotTier

	if reuse {
		p.mu.Lock()
		b, m := p.tree.Lookup(ids)
		cacheable = m
		matched = m
		if k := b.KV(); k != nil {
			s = p.sessionFromPrefixClone(k)
			if m >= len(ids) {
				cachedLogits = b.Logits()
			}
			sourceTier = radixkv.SnapshotTierDeviceL1
		}
		p.tree.Done(b)
		p.mu.Unlock()

		if s != nil && matched >= len(ids) && cachedLogits == nil {
			if inKernelRefeedLastTokenForExactHit(s, len(ids)) {
				matched = len(ids) - 1
			} else {
				s, matched = nil, 0
				sourceTier = radixkv.SnapshotTierMiss
			}
		}
	}

	if s == nil {
		matched = 0
		s = p.m.NewSession()
	}
	defer s.Close()
	p.configureNativeSession(s)

	p.recordTurnTax(promptTok, cacheable, matched)
	enginestep.Default.ObservePrefixQueried(promptTok)
	enginestep.Default.ObservePrefixMatched(matched)

	// Prefill divergent prompt tokens
	logits := cachedLogits
	var prefillS float64
	if logits == nil {
		tp := time.Now()
		prefillAt := matched
		if prefillAt < len(ids) {
			rawLogits, err := p.prefillDivergentSuffix(ctx, s, ids[prefillAt:], measurementOpt...)
			if err != nil {
				p.recordNativePhase(nativePhaseTraceID(ctx), NativePhasePrefill, tp, time.Since(tp), false)
				return inKernelGenerateResult{}, err
			}
			logits = append([]float32(nil), rawLogits...)
		}
		prefillS = time.Since(tp).Seconds()
		enginestep.Default.ObservePhase(enginestep.PhasePrefill, time.Since(tp))
		p.recordNativePhase(nativePhaseTraceID(ctx), NativePhasePrefill, tp, time.Since(tp), true)
	} else {
		logits = append([]float32(nil), cachedLogits...)
	}
	if err = ctx.Err(); err != nil {
		return inKernelGenerateResult{}, err
	}

	// Admit full prompt to prefix cache BEFORE speculative decode mutates cache
	if reuse {
		p.mu.Lock()
		b, m := p.tree.Lookup(ids)
		leaf := p.tree.InsertCloneWithLogits(b, ids[m:], s.Cache, logits)
		p.tree.Done(leaf)
		p.mu.Unlock()
		p.noteKVPrefixAdmitted()
	}

	td := time.Now()
	// metalMTPMu is already held for the duration of generateReusedMetalMTP
	// (locked above), so read the field directly rather than via the locking
	// accessor, which would self-deadlock.
	coord := p.metalMTPCoordinator
	coord.SetTargetSession(s)
	checkpointMgr := ctxmmu.NewCheckpointManager(nil, nil, nil, nil)
	const sessionID = "metal-mtp"
	if _, checkpointErr := checkpointMgr.SaveInPlaceCheckpoint(sessionID); checkpointErr != nil {
		return inKernelGenerateResult{}, fmt.Errorf("agent: initialize Metal MTP checkpoint %q: %w", sessionID, checkpointErr)
	}
	coord.SetMMU(checkpointMgr, sessionID)
	defer coord.SetMMU(nil, "")

	// Enforce greedy temperature-zero tripwire
	if tripErr := coord.CheckSamplingTripwire(temp, freqPenalty); tripErr != nil {
		if coord.Config().EnforceGreedyTripwire {
			temp = 0.0
			topP = 1.0
			topK = 0
			freqPenalty = 0.0
			presPenalty = 0.0
		}
	}

	committed := append([]int(nil), ids...)
	gen := 0
	stopped := false
	curLogits := logits
	var round speculativeRoundObservation

	for gen < maxNew {
		round.close(gen)
		if err = ctx.Err(); err != nil {
			break
		}

		before := coord.Stats()
		roundStart := time.Now()
		accTokens, bonusTok, nextLogits, stepErr := coord.StepRound(ctx, committed, curLogits)
		if stepErr != nil {
			err = stepErr
			break
		}
		if after := coord.Stats(); after.TotalProposed > before.TotalProposed {
			round.open(roundStart, after.TotalProposed-before.TotalProposed, after.TotalAccepted-before.TotalAccepted, gen)
		} else {
			enginestep.Default.ObserveDecodeStep(enginestep.PathSerial, 1, time.Since(roundStart))
		}

		roundStopped := false
		for _, tok := range accTokens {
			if tok < 0 || stops[tok] {
				roundStopped = true
				stopped = true
				break
			}
			emitStopped := emit != nil && emit(tok)
			gen++
			committed = append(committed, tok)
			if emitStopped || gen == maxNew {
				roundStopped = true
				stopped = emitStopped
				break
			}
		}

		if roundStopped || gen == maxNew {
			break
		}

		if bonusTok >= 0 {
			if stops[bonusTok] {
				stopped = true
				break
			}
			emitStopped := emit != nil && emit(bonusTok)
			gen++
			committed = append(committed, bonusTok)
			if emitStopped || gen == maxNew {
				stopped = emitStopped
				break
			}
		}

		curLogits = nextLogits
	}

	round.close(gen)
	decodeS := time.Since(td).Seconds()
	enginestep.Default.ObservePhase(enginestep.PhaseDecode, time.Since(td))
	p.recordNativePhase(nativePhaseTraceID(ctx), NativePhaseDecode, td, time.Since(td), err == nil)
	p.recordNativePhase(nativePhaseTraceID(ctx), NativePhaseTerminal, time.Now(), 0, err == nil)
	return inKernelGenerateResult{
		gen:        gen,
		promptTok:  promptTok,
		cacheable:  cacheable,
		matched:    matched,
		sourceTier: sourceTier,
		prefillS:   prefillS,
		decodeS:    decodeS,
		stopped:    stopped,
		spec:       round.tally,
	}, err
}
