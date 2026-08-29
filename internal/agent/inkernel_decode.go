package agent

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

const inKernelQwenQ4KPrefillChunkTokens = 512

type inKernelPrefillSession interface {
	PrefillNoLogits([]int)
	Prefill([]int) []float32
}

func (p *InKernelPlanner) configureNativeSession(s *model.Session) {
	s.Quant = p.quant
	// Resident Q4_K decode runs on both host and device paths. The slab selector is
	// session-scoped alongside that mode, so an explicit planner setting reaches every
	// real request session and never leaks between planners.
	s.Q4K = p.q4k
	s.Q4KGateUpOutputSlab = p.q4kGateUpOutputSlab
	s.CPUOffloadExperts = p.cpuOffloadExperts
	p.applyExpertSpill(s)
	if p.backend == nil && p.metal {
		s.Metal = true
		s.MetalQ4K = p.q4k
	}
}

func (p *InKernelPlanner) generateReused(ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool) {
	gen, promptTok, matched, prefillS, decodeS, stopped, _ = p.generateReusedContext(context.Background(), ids, maxNew, temp, topP, topK, stops, emit)
	return
}

func (p *InKernelPlanner) generateReusedContext(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool, err error) {
	gen, promptTok, _, matched, _, prefillS, decodeS, stopped, err = p.generateReusedContextWithBias(ctx, ids, maxNew, temp, topP, topK, nil, 0, 0, stops, emit)
	return
}

// generateReusedContextWithBias runs the decode loop, sampling each next token with
// sampleLogitsWithPenalty. freqPenalty/presPenalty are the OpenAI repetition
// penalties (#1705); both zero is a byte-for-byte no-op versus the pre-penalty
// path, so every existing caller (which passes 0, 0) is unaffected. The per-token
// generation-count histogram (counts) is built from THIS turn's decode loop only —
// it is sized to the logits vocab on first use and never persists across turns.
func (p *InKernelPlanner) generateReusedContextWithBias(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, measurementOpt ...*nativeInferenceMeasurement) (gen, promptTok, cacheable, matched int, sourceTier radixkv.SnapshotTier, prefillS, decodeS float64, stopped bool, err error) {
	p.qwen35MetalGDNExecuted.Store(false)
	var measurement *nativeInferenceMeasurement
	if len(measurementOpt) > 0 {
		measurement = measurementOpt[0]
	}
	promptTok = len(ids)
	if len(ids) == 0 {
		return
	}
	if err = ctx.Err(); err != nil {
		return
	}
	reuse := p.tree != nil && inKernelPlannerPrefixReuseSupported(p.m, p.backend)

	// 1) Acquire a session, reusing the longest cached KV prefix when enabled. The clone
	// (SessionFromPrefix) happens under the lock, so once we unlock our session owns an
	// independent copy and a concurrent tree eviction cannot affect this turn's decode.
	var s *model.Session
	closeSession := false
	var cachedLogits []float32
	skipExactDeviceL1Readmission := false
	if reuse {
		owner, scoped := prefixCacheIdentityFromContext(ctx)
		scopedLookup := scoped && p.scopedTree != nil
		var matchedKV *model.KVCache
		var matchedSnapshot *model.PrefixSnapshot
		var m int
		var tier radixkv.SnapshotTier
		if scopedLookup {
			if p.backend != nil {
				matchedSnapshot, cachedLogits, m, _, tier, err = p.scopedTree.LookupSnapshotTieredContext(ctx, owner, ids)
			} else {
				matchedKV, cachedLogits, m, _, err = p.scopedTree.Lookup(owner, ids)
			}
		} else {
			p.mu.Lock()
			if p.backend != nil {
				b, snap, legacyMatched, lookupTier, lookupErr := p.tree.LookupSnapshotTieredContext(ctx, ids)
				matchedSnapshot, m, err = snap, legacyMatched, lookupErr
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
			return
		}
		// The lookup-side (cacheability) half of the #3390 split: m tokens matched the
		// radix index at this instant, whether or not the servability checks below (nil
		// KV, exact-hit refeed, unsupported truncate) let all of them be served. The
		// realized `matched` can only stay at or fall below this.
		cacheable = m
		if matchedSnapshot != nil {
			s = p.m.NewBackendSession(p.backend)
			if err = matchedSnapshot.Restore(s); err != nil {
				matchedSnapshot.Close()
				s.Close()
				return
			}
			matchedSnapshot.Close()
			closeSession, matched, sourceTier = true, m, tier
		} else if matchedKV != nil {
			s = p.sessionFromPrefixClone(matchedKV)
			closeSession = p.backend != nil
			matched = m
			sourceTier = radixkv.SnapshotTierDeviceL1
		}
		// Fully cached (an exact-duplicate transcript): the cached KV has the prefix but
		// decode still needs the last-token logits to sample the first generated token. New
		// leaves carry those logits; older/split leaves may not. When absent, refeed only
		// the final token: evicting the final cached row leaves an exact prefix of len(ids)-1,
		// and Prefill below recomputes that one row/logits. If the cache cannot truncate
		// exactly (for example a recurrent hybrid), fail open to the old full-prefill path.
		if s != nil && matched >= len(ids) && cachedLogits == nil {
			if inKernelRefeedLastTokenForExactHit(s, len(ids)) {
				matched = len(ids) - 1
			} else {
				if closeSession {
					s.Close()
				}
				s, matched, closeSession = nil, 0, false
				sourceTier = radixkv.SnapshotTierMiss
			}
		}
		// The unscoped hot tree already owns the full snapshot and lookup restored a
		// deep clone into this session. Cached logits mean no prompt state changed, so
		// capturing and admitting that same state again only performs a second clone.
		// Scoped hits retain re-admission because lookup does not yet expose the source
		// scope needed to prove that tenant materialization is redundant.
		skipExactDeviceL1Readmission = !scopedLookup && matchedSnapshot != nil &&
			matched == len(ids) && cachedLogits != nil && sourceTier == radixkv.SnapshotTierDeviceL1
	}
	if s == nil {
		matched = 0
		if p.backend != nil {
			s = p.m.NewBackendSession(p.backend)
			closeSession = true
		} else {
			s = p.m.NewSession()
		}
	}
	if closeSession {
		defer s.Close()
	}
	p.configureNativeSession(s)
	qwen35MetalStateIdentityEnabled := false
	if shouldEnableQwen35MetalStateIdentity(p, measurement, ids, matched, cachedLogits) {
		if enableErr := s.EnableQwen35MetalStateIdentityReceipt(ids); enableErr != nil {
			var unavailable *model.Qwen35MetalStateIdentityUnavailableError
			if !errors.As(enableErr, &unavailable) {
				err = enableErr
				return
			}
		} else {
			qwen35MetalStateIdentityEnabled = true
			defer s.ResetQwen35MetalStateIdentityReceipt()
		}
	}
	if p.qwen35MetalGDNSequence && p.backend == nil && p.metal && p.q4k && p.m.Cfg.IsQwen35Hybrid() && cachedLogits == nil {
		if matched != 0 {
			err = &model.UnsupportedGDNPreprojectedSequenceError{
				Path:   model.Qwen35MetalGDNSequenceForwardPath,
				Reason: "native sequence requires a fresh prompt; restored host prefix state cannot initialize resident owners",
			}
			return
		}
		if err = s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			return
		}
		// The historical CPU-session path otherwise has no close requirement. The
		// candidate owns native state, so bind cleanup even when prefill panics.
		defer s.Close()
	}

	// 1b) RECORD this turn's cache decision (#1538, inkernel_turntax.go). This is the seam the
	// turn-tax planner is defined on: the lookup has run and every servability/trust gate above
	// has settled (`matched` is final, and a prefix that matched but could not be served is still
	// visible as cacheable > 0 with matched == 0), while NO prefill or decode compute has happened
	// yet — so the decision is made from signals known ahead of the work. One append per turn that
	// reaches this seam, on the reuse path and the cold path alike.
	p.recordTurnTax(promptTok, cacheable, matched)

	// 2) Prefill ONLY the divergent suffix (the whole prompt on a miss). Device hybrid
	// snapshots cannot be truncated when a radix edge later splits: recurrent GDN state is
	// position-dependent. Materialize one stable block boundary before the leaf so sibling
	// prompts can restore a complete snapshot rather than merely matching an unusable
	// mid-edge token run. The boundary is deliberately bounded to one per request; Qwen
	// snapshots own substantial recurrent state even when the token prefix is short.
	logits := cachedLogits
	if logits == nil {
		tp := time.Now()
		prefillAt := matched
		checkpoint := inKernelSnapshotCheckpoint(prefillAt, len(ids))
		if reuse && p.backend != nil && checkpoint > prefillAt {
			logits = s.Prefill(ids[prefillAt:checkpoint])
			var checkpointSnapshot *model.PrefixSnapshot
			checkpointSnapshot, err = s.PrefixSnapshot()
			if err != nil {
				return
			}
			if err = p.admitPrefixSnapshot(ctx, ids[:checkpoint], checkpointSnapshot, logits); err != nil {
				checkpointSnapshot.Close()
				return
			}
			prefillAt = checkpoint
		}
		if prefillAt < len(ids) {
			logits, err = p.prefillDivergentSuffix(ctx, s, ids[prefillAt:])
			if err != nil {
				return
			}
		}
		prefillS = time.Since(tp).Seconds()
	}
	if err = ctx.Err(); err != nil {
		return
	}
	if p.qwen35MetalGDNSequence && p.backend == nil && p.metal && p.q4k && p.m.Cfg.IsQwen35Hybrid() {
		var executed bool
		executed, err = s.FinalizeQwen35MetalGDNPreprojectedSequence()
		if err != nil {
			captureQwen35MetalForwardSequenceReceipt(p, s, measurement)
			return
		}
		p.qwen35MetalGDNExecuted.Store(executed)
	}
	captureQwen35MetalForwardSequenceReceipt(p, s, measurement)
	if qwen35MetalStateIdentityEnabled {
		if err = finalizeAndCaptureQwen35MetalStateIdentity(s, measurement); err != nil {
			return
		}
	}

	// 3) Snapshot the full-prompt KV (before decode mutates s.Cache) and cache it under a
	// fresh Lookup→Insert→Done. The snapshot covers the FULL ids prefix, so it is a valid
	// leaf kv no matter how much a concurrent turn may have inserted since step 1.
	if reuse {
		if p.backend != nil && !skipExactDeviceL1Readmission {
			var snap *model.PrefixSnapshot
			snap, err = s.PrefixSnapshot()
			if err != nil {
				return
			}
			if err = p.admitPrefixSnapshot(ctx, ids, snap, logits); err != nil {
				snap.Close()
				return
			}
		} else {
			snap := s.Cache.Clone()
			if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped && p.scopedTree != nil {
				if admitErr := p.scopedTree.AdmitPrivate(owner, ids, snap, logits); admitErr != nil {
					err = admitErr
					return
				}
			} else {
				p.mu.Lock()
				b, m := p.tree.Lookup(ids)
				leaf := p.tree.InsertWithLogits(b, ids[m:], snap, logits)
				p.tree.Done(leaf)
				p.mu.Unlock()
			}
		}
	}

	// 4) Decode. The per-token step (sample → token-ID stop → penalty count → string-suffix
	// emit → the maxNew-1 skip-the-unused-final-Step) is factored into decodeLane.decodeOne so
	// the SAME step drives both the serial forward (Session.Step, the default) and the opt-in
	// multi-lane batched forward (BatchSession.StepBatchActive). The forward is the ONLY
	// difference between the two drivers, and model.StepBatchActive is bit-for-bit identical to
	// serial Session.Step per lane, so an unset FAK_INKERNEL_BATCH decodes byte-identically to
	// the pre-seam loop (the #401 wiring seam; the batched glm_moe_dsa GEMM is a separate lever).
	rng := rand.New(rand.NewSource(p.seed))
	// counts is the per-token generation histogram this turn's frequency/presence
	// penalty is computed from (#1705): counts[t] is how many times token t has
	// already been generated in THIS response. Only allocated when a penalty is
	// actually requested, so the zero-penalty path (the overwhelming default) pays
	// no extra allocation or per-step bookkeeping versus the pre-#1705 code.
	var counts []int32
	if freqPenalty != 0 || presPenalty != 0 {
		counts = make([]int32, len(logits))
	}
	ln := &decodeLane{
		s:           s,
		logits:      logits,
		counts:      counts,
		rng:         rng,
		emit:        emit,
		stops:       stops,
		temp:        temp,
		topP:        topP,
		topK:        topK,
		logitBias:   logitBias,
		freqPenalty: freqPenalty,
		presPenalty: presPenalty,
		maxNew:      maxNew,
		measurement: measurement,
	}
	measurement.startDecodeTrace()
	td := time.Now()
	if coordinated, coordinateErr := coalescedDecode(ctx, ln); coordinated {
		err = coordinateErr
	} else if p.batchDecode {
		// Opt-in: drive this one request through the shared continuous-batch step. For B==1
		// StepBatchActive is exactly Seqs[0].Step, so the served tokens are unchanged.
		inKernelDecodeLanesBatched(ctx, []*decodeLane{ln}, p.m, p.quant)
	} else {
		inKernelDecodeSerial(ctx, ln)
	}
	gen, stopped, err = ln.gen, ln.stopped, ln.err
	decodeS = time.Since(td).Seconds()
	// 5) R6/#5617: fold this request's activated-expert residency into the serve-scoped ledger while
	// the session is still alive — the `defer s.Close()` above takes the ring, and with it every
	// counter the offload ladder built, the moment this function returns. The token count is what
	// was actually FORWARDED (the prompt suffix the prefix cache could not serve, plus what was
	// generated), because a token served from cache activated no expert and would make the ring's
	// bytes-per-token read cheaper than it is. Inert on a session with no ring, which is the default.
	p.noteMoEResidency(s, int64(len(ids)-matched+gen))
	return
}

// prefillDivergentSuffix bounds the temporary prompt panels used by the resident
// Qwen hybrid Q4_K path. #9066 proves that PrefillNoLogits and Prefill append the
// same KV, convolution, recurrent, and position state at nonzero cache positions;
// only the last chunk needs the distribution consumed by decode. Other forward
// paths keep the historical single Prefill call because they do not share that
// append proof.
func (p *InKernelPlanner) prefillDivergentSuffix(ctx context.Context, s inKernelPrefillSession, ids []int) ([]float32, error) {
	chunkTokens := p.effectiveQwenQ4KPrefillChunkTokens()
	if !p.qwenQ4KPrefillChunkTarget() || len(ids) <= chunkTokens {
		return s.Prefill(ids), nil
	}
	for len(ids) > chunkTokens {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		s.PrefillNoLogits(ids[:chunkTokens])
		ids = ids[chunkTokens:]
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s.Prefill(ids), nil
}

func (p *InKernelPlanner) qwenQ4KPrefillChunkTarget() bool {
	return p != nil && p.m != nil && p.backend == nil && p.q4k && p.m.Cfg.IsQwen35Hybrid()
}

func (p *InKernelPlanner) effectiveQwenQ4KPrefillChunkTokens() int {
	if p != nil && p.qwenQ4KPrefillChunkTokens > 0 {
		return p.qwenQ4KPrefillChunkTokens
	}
	return inKernelQwenQ4KPrefillChunkTokens
}

func (p *InKernelPlanner) nativeInferencePrefillChunkTokens() int {
	if !p.qwenQ4KPrefillChunkTarget() {
		return 0
	}
	return p.effectiveQwenQ4KPrefillChunkTokens()
}

// decodeLane is one request's live decode state. decodeOne runs one token's worth of the
// decode loop body EXCEPT the model forward, so the serial driver (Session.Step) and the
// opt-in batched driver (BatchSession.StepBatchActive) share identical per-token semantics —
// the property that makes the two paths bit-for-bit equivalent.
type decodeLane struct {
	ctx    context.Context
	s      *model.Session
	logits []float32
	counts []int32
	rng    *rand.Rand
	emit   func(int) bool
	stops  map[int]bool

	temp        float64
	topP        float64
	topK        int
	logitBias   model.LogitBias
	freqPenalty float64
	presPenalty float64
	maxNew      int
	measurement *nativeInferenceMeasurement

	gen     int
	stopped bool
	done    bool
	err     error
}

func captureQwen35MetalForwardSequenceReceipt(p *InKernelPlanner, s *model.Session, measurement *nativeInferenceMeasurement) {
	if p == nil || s == nil || measurement == nil || measurement.inferenceDisabled || p.m == nil ||
		p.backend != nil || !p.metal || !p.q4k || !p.m.Cfg.IsQwen35Hybrid() {
		return
	}
	measurement.qwen35MetalForwardSequence = s.Qwen35MetalForwardSequenceStatus()
}

func shouldEnableQwen35MetalStateIdentity(p *InKernelPlanner, measurement *nativeInferenceMeasurement, ids []int, matched int, cachedLogits []float32) bool {
	return p != nil && p.m != nil && measurement != nil && !measurement.inferenceDisabled &&
		p.backend == nil && p.metal && p.q4k && p.m.Cfg.IsQwen35Hybrid() &&
		cachedLogits == nil && matched == 0 && len(ids) == 32
}

type qwen35MetalStateIdentitySession interface {
	FinalizeQwen35MetalStateIdentityReceipt() (bool, error)
	Qwen35MetalStateIdentityReceipt() (model.Qwen35MetalStateIdentityReceipt, bool)
}

func finalizeAndCaptureQwen35MetalStateIdentity(s qwen35MetalStateIdentitySession, measurement *nativeInferenceMeasurement) error {
	if s == nil || measurement == nil {
		return fmt.Errorf("agent: opted-in Qwen Metal state identity has no request-local owner")
	}
	finalized, err := s.FinalizeQwen35MetalStateIdentityReceipt()
	if err != nil {
		return err
	}
	identity, available := s.Qwen35MetalStateIdentityReceipt()
	if !finalized || !available || !identity.Available {
		return fmt.Errorf("agent: opted-in Qwen Metal state identity did not finalize")
	}
	measurement.qwen35MetalStateIdentity = cloneQwen35MetalStateIdentityReceipt(identity)
	return nil
}

type nativeInferenceMeasurement struct {
	startedAt                           time.Time
	tokenIDs                            []int
	logprobs                            []float64
	ttftS                               float64
	inferenceDisabled                   bool
	traceNow                            func() time.Time
	traceStartedAt                      time.Time
	traceEvents                         []NativeDecodeTraceEvent
	decodeTokenIDsEnabled               bool
	decodeTokenIDs                      []int
	qwen35MetalForwardSequence          model.Qwen35MetalForwardSequenceReceipt
	qwen35MetalStateIdentity            *model.Qwen35MetalStateIdentityReceipt
	cudaImmutableWeightUploadsBefore    model.NativeCUDAImmutableWeightUploadCounters
	cudaImmutableWeightUploadsAvailable bool
}

func (m *nativeInferenceMeasurement) reset() {
	if m == nil {
		return
	}
	m.tokenIDs = m.tokenIDs[:0]
	m.logprobs = m.logprobs[:0]
	m.ttftS = 0
	m.traceStartedAt = time.Time{}
	m.traceEvents = m.traceEvents[:0]
	m.decodeTokenIDs = m.decodeTokenIDs[:0]
	m.qwen35MetalForwardSequence = model.Qwen35MetalForwardSequenceReceipt{}
	m.qwen35MetalStateIdentity = nil
}

func (m *nativeInferenceMeasurement) record(logits []float32, token int) error {
	if m == nil || m.inferenceDisabled {
		return nil
	}
	lp, err := chosenTokenLogprob(logits, token)
	if err != nil {
		return err
	}
	if len(m.tokenIDs) == 0 {
		m.ttftS = time.Since(m.startedAt).Seconds()
	}
	m.tokenIDs = append(m.tokenIDs, token)
	m.logprobs = append(m.logprobs, lp)
	return nil
}

func (m *nativeInferenceMeasurement) startDecodeTrace() {
	if m == nil || m.traceNow == nil {
		return
	}
	m.traceStartedAt = m.traceNow()
	m.traceEvents = m.traceEvents[:0]
}

func (m *nativeInferenceMeasurement) recordDecodeTrace(tokenIndex, tokenID int) error {
	if m == nil || (m.traceNow == nil && !m.decodeTokenIDsEnabled) {
		return nil
	}
	if m.traceNow != nil {
		wantIndex := len(m.traceEvents) + 1
		if tokenIndex != wantIndex {
			return fmt.Errorf("native decode trace token index %d, want %d", tokenIndex, wantIndex)
		}
		elapsed := m.traceNow().Sub(m.traceStartedAt).Nanoseconds()
		if elapsed < 0 || (len(m.traceEvents) > 0 && elapsed < m.traceEvents[len(m.traceEvents)-1].ElapsedNS) {
			return fmt.Errorf("native decode trace clock moved backwards at token %d", tokenIndex)
		}
		m.traceEvents = append(m.traceEvents, NativeDecodeTraceEvent{TokenIndex: tokenIndex, ElapsedNS: elapsed})
	}
	if m.decodeTokenIDsEnabled {
		wantIndex := len(m.decodeTokenIDs) + 1
		if tokenIndex != wantIndex {
			return fmt.Errorf("native decode token ID index %d, want %d", tokenIndex, wantIndex)
		}
		m.decodeTokenIDs = append(m.decodeTokenIDs, tokenID)
	}
	return nil
}

func (m *nativeInferenceMeasurement) recordForwardTiming(tokenIndex int, kind string, activeLanes int, duration time.Duration) error {
	if m == nil || m.traceNow == nil {
		return nil
	}
	if duration < 0 {
		return fmt.Errorf("native decode forward clock moved backwards at token %d", tokenIndex)
	}
	if tokenIndex <= 0 || tokenIndex > len(m.traceEvents) || m.traceEvents[tokenIndex-1].TokenIndex != tokenIndex {
		return fmt.Errorf("native decode forward token index %d has no committed trace event", tokenIndex)
	}
	if activeLanes <= 0 || (kind != NativeForwardSessionStep && kind != NativeForwardStepBatchActive) {
		return fmt.Errorf("native decode forward token %d has invalid kind/active lanes %q/%d", tokenIndex, kind, activeLanes)
	}
	event := &m.traceEvents[tokenIndex-1]
	if event.Forward != nil {
		return fmt.Errorf("native decode forward token %d already recorded", tokenIndex)
	}
	event.Forward = &NativeForwardTiming{Kind: kind, DurationNS: duration.Nanoseconds(), ActiveLanes: activeLanes}
	return nil
}

func chosenTokenLogprob(logits []float32, token int) (float64, error) {
	if token < 0 || token >= len(logits) || len(logits) == 0 {
		return 0, fmt.Errorf("chosen token %d outside logits size %d", token, len(logits))
	}
	maxLogit := math.Inf(-1)
	for _, raw := range logits {
		v := float64(raw)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return 0, fmt.Errorf("non-finite model logit")
		}
		if v > maxLogit {
			maxLogit = v
		}
	}
	var expSum float64
	for _, raw := range logits {
		expSum += math.Exp(float64(raw) - maxLogit)
	}
	lp := float64(logits[token]) - maxLogit - math.Log(expSum)
	if math.IsNaN(lp) || math.IsInf(lp, 0) {
		return 0, fmt.Errorf("non-finite chosen-token log probability")
	}
	return lp, nil
}

// decodeOne runs one decode iteration for a lane EXCEPT the forward step. It mirrors the body
// of the pre-seam serial decode loop exactly — the ctx check, the sample, the token-ID stop,
// the per-token count, the string-suffix emit, the emit-time cancel, and the maxNew-1
// skip-the-unused-final-Step — updating the lane's gen/stopped/err/done. It returns the
// sampled token and whether the caller should advance the lane with a forward. advance==false
// means the lane is finished this step and must not be stepped again (the caller drops it from
// the batch's active set). The gen accounting is identical to the old loop: gen counts exactly
// the tokens that passed the stop check and were emitted.
func (ln *decodeLane) decodeOne(ctx context.Context) (next int, advance bool) {
	if err := ctx.Err(); err != nil {
		ln.err, ln.done = err, true
		return 0, false
	}
	next = sampleLogitsWithPenalty(ln.logits, ln.temp, ln.topP, ln.topK, ln.logitBias, ln.freqPenalty, ln.presPenalty, ln.counts, ln.rng)
	if next < 0 || ln.stops[next] {
		ln.stopped, ln.done = true, true
		return 0, false
	}
	if err := ln.measurement.record(ln.logits, next); err != nil {
		ln.err, ln.done = err, true
		return 0, false
	}
	if ln.counts != nil && next < len(ln.counts) {
		ln.counts[next]++
	}
	emitStopped := ln.emit != nil && ln.emit(next)
	ln.gen++ // this non-stop token was emitted/generated; trace only after this count.
	if err := ln.measurement.recordDecodeTrace(ln.gen, next); err != nil {
		ln.err, ln.done = err, true
		return 0, false
	}
	if emitStopped {
		ln.stopped, ln.done = true, true
		return 0, false
	}
	if ln.emit != nil {
		if err := ctx.Err(); err != nil {
			ln.err, ln.done = err, true
			return 0, false
		}
	}
	if ln.gen == ln.maxNew {
		ln.done = true
		return 0, false
	}
	return next, true
}

// inKernelDecodeSerial is the DEFAULT per-request decode: one lane advanced by Session.Step,
// byte-identical to the pre-seam decode loop (the gen<maxNew guard preserves the maxNew<=0
// no-token contract). It is the path an unset FAK_INKERNEL_BATCH takes.
func inKernelDecodeSerial(ctx context.Context, ln *decodeLane) {
	for ln.gen < ln.maxNew {
		laneCtx := ctx
		if ln.ctx != nil {
			laneCtx = ln.ctx
		}
		next, advance := ln.decodeOne(laneCtx)
		if !advance {
			return
		}
		if ln.measurement == nil || ln.measurement.traceNow == nil {
			ln.logits = ln.s.Step(next)
			continue
		}
		started := ln.measurement.traceNow()
		ln.logits = ln.s.Step(next)
		if err := ln.measurement.recordForwardTiming(ln.gen, NativeForwardSessionStep, 1, ln.measurement.traceNow().Sub(started)); err != nil {
			ln.err, ln.done = err, true
			return
		}
	}
}

// inKernelDecodeLanesBatched advances every lane in lockstep through ONE shared
// BatchSession.StepBatchActive per step: each still-running lane samples its next token via the
// shared decodeOne, then a single StepBatchActive forwards exactly the active lanes (each over
// its own Session/KV) and scatters the per-lane logits back. Because StepBatchActive is
// bit-for-bit identical to serial Session.Step for every active lane, each lane's emitted token
// sequence is identical to inKernelDecodeSerial on the same prompt/seed/sampler — the
// continuous-batching WIRING, correctness-equivalent and GPU-free. A lane that finishes (a
// token-ID stop, a string-suffix stop, or maxNew) simply drops out of the active mask while the
// others keep batching. Each lane owns its own *Session, so per-lane KV is never shared.
func inKernelDecodeLanesBatched(ctx context.Context, lanes []*decodeLane, m *model.Model, quant bool) (sharedPanels int, sharedMACs int64) {
	if len(lanes) == 0 {
		return
	}
	seqs := make([]*model.Session, len(lanes))
	for i, ln := range lanes {
		seqs[i] = ln.s
	}
	bs := &model.BatchSession{M: m, Seqs: seqs, Quant: quant}
	ids := make([]int, len(lanes))
	active := make([]bool, len(lanes))
	for {
		anyActive := false
		for i, ln := range lanes {
			active[i] = false
			if ln.done || ln.gen >= ln.maxNew {
				continue // finished lane: dropped from the active set, never re-stepped.
			}
			laneCtx := ctx
			if ln.ctx != nil {
				laneCtx = ln.ctx
			}
			next, advance := ln.decodeOne(laneCtx)
			if !advance {
				continue
			}
			ids[i] = next
			active[i] = true
			anyActive = true
		}
		if !anyActive {
			return
		}
		activeLanes := 0
		var forwardNow func() time.Time
		for i, isActive := range active {
			if !isActive {
				continue
			}
			activeLanes++
			if forwardNow == nil && lanes[i].measurement != nil {
				forwardNow = lanes[i].measurement.traceNow
			}
		}
		var forwardStarted time.Time
		if forwardNow != nil {
			forwardStarted = forwardNow()
		}
		out, panels, macs, probed := runQwenSharedReceiptProbe(bs, ids, active)
		if !probed {
			out = bs.StepBatchActive(ids, active)
			panels = bs.LastStepSharedPanels()
			macs = bs.LastStepMACs()
		}
		if forwardNow != nil {
			duration := forwardNow().Sub(forwardStarted)
			for i, isActive := range active {
				if !isActive {
					continue
				}
				if err := lanes[i].measurement.recordForwardTiming(lanes[i].gen, NativeForwardStepBatchActive, activeLanes, duration); err != nil {
					lanes[i].err, lanes[i].done = err, true
				}
			}
		}
		sharedPanels += panels
		sharedMACs += macs
		for i := range lanes {
			if active[i] {
				lanes[i].logits = out[i]
			}
		}
	}
}

const inKernelSnapshotCheckpointTokens = 64

// inKernelSnapshotCheckpoint returns the deepest fixed block boundary strictly before
// the prompt and after the already restored prefix. A strict-before boundary preserves
// the full leaf for exact hits while creating a reusable ancestor for sibling suffixes.
func inKernelSnapshotCheckpoint(matched, promptTokens int) int {
	if promptTokens <= 1 {
		return 0
	}
	checkpoint := ((promptTokens - 1) / inKernelSnapshotCheckpointTokens) * inKernelSnapshotCheckpointTokens
	if checkpoint <= matched {
		return 0
	}
	return checkpoint
}

// admitPrefixSnapshot transfers snapshot ownership to the same scoped/unscoped tree
// used by lookup. On error ownership remains with the caller.
func (p *InKernelPlanner) admitPrefixSnapshot(ctx context.Context, ids []int, snap *model.PrefixSnapshot, logits []float32) error {
	if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped && p.scopedTree != nil {
		return p.scopedTree.AdmitPrivateSnapshot(owner, ids, snap, logits)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	b, matched := p.tree.Lookup(ids)
	leaf, err := p.tree.InsertSnapshot(b, ids[matched:], snap, logits)
	if leaf != nil {
		p.tree.Done(leaf)
	}
	return err
}

func (p *InKernelPlanner) sessionFromPrefixClone(prefix *model.KVCache) *model.Session {
	if p.backend != nil {
		s := p.m.NewBackendSession(p.backend)
		s.Cache = prefix
		return s
	}
	s := p.m.NewSession()
	s.Cache = prefix
	return s
}

func inKernelRefeedLastTokenForExactHit(s *model.Session, promptLen int) bool {
	if s == nil || s.Cache == nil || promptLen <= 0 || s.Cache.Len() < promptLen {
		return false
	}
	removed, err := s.Cache.TryEvict(promptLen-1, 1)
	return err == nil && removed == 1 && s.Cache.Len() == promptLen-1
}

func envInt(key string, def int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}

func envInt64(key string, def int64) int64 {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil {
			return n
		}
	}
	return def
}

func envFloat(key string, def float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}
