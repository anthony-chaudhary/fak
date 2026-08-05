package agent

import (
	"context"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func (p *InKernelPlanner) generateReused(ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool) {
	gen, promptTok, matched, prefillS, decodeS, stopped, _ = p.generateReusedContext(context.Background(), ids, maxNew, temp, topP, topK, stops, emit)
	return
}

func (p *InKernelPlanner) generateReusedContext(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, stops map[int]bool, emit func(int) bool) (gen, promptTok, matched int, prefillS, decodeS float64, stopped bool, err error) {
	gen, promptTok, _, matched, prefillS, decodeS, stopped, err = p.generateReusedContextWithBias(ctx, ids, maxNew, temp, topP, topK, nil, 0, 0, stops, emit)
	return
}

// generateReusedContextWithBias runs the decode loop, sampling each next token with
// sampleLogitsWithPenalty. freqPenalty/presPenalty are the OpenAI repetition
// penalties (#1705); both zero is a byte-for-byte no-op versus the pre-penalty
// path, so every existing caller (which passes 0, 0) is unaffected. The per-token
// generation-count histogram (counts) is built from THIS turn's decode loop only —
// it is sized to the logits vocab on first use and never persists across turns.
func (p *InKernelPlanner) generateReusedContextWithBias(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool) (gen, promptTok, cacheable, matched int, prefillS, decodeS float64, stopped bool, err error) {
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
	if reuse {
		owner, scoped := prefixCacheIdentityFromContext(ctx)
		var matchedKV *model.KVCache
		var matchedSnapshot *model.PrefixSnapshot
		var m int
		if scoped && p.scopedTree != nil {
			if p.backend != nil {
				matchedSnapshot, cachedLogits, m, _, err = p.scopedTree.LookupSnapshot(owner, ids)
			} else {
				matchedKV, cachedLogits, m, _, err = p.scopedTree.Lookup(owner, ids)
			}
		} else {
			p.mu.Lock()
			if p.backend != nil {
				b, snap, legacyMatched, lookupErr := p.tree.LookupSnapshot(ids)
				matchedSnapshot, m, err = snap, legacyMatched, lookupErr
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
			closeSession, matched = true, m
		} else if matchedKV != nil {
			s = p.sessionFromPrefixClone(matchedKV)
			closeSession = p.backend != nil
			matched = m
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
			}
		}
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
	s.Quant = p.quant
	// resident-Q4_K decode runs on BOTH the host (cpu-ref) AND the cuda backend: the device HAL
	// copies the raw Q4_K super-blocks resident and serves them with the dequant-fused k_q4k_gemm
	// tile (internal/compute/cuda.go MatMul/BatchedMatMul case Q4_K, #485), so a device session can
	// decode Q4_K directly — no f32/Q8 round-trip. (The old gate forced Q8/F32 on any backend.)
	s.Q4K = p.q4k
	s.CPUOffloadExperts = p.cpuOffloadExperts
	// …and the GRADED form of that same placement (#5612), when the operator sized one: which MoE
	// layers spill to host, and how many device bytes the routed-expert ring (#5611) may hold. No
	// grade -> no-op, so the line above remains the whole placement decision it was.
	p.applyExpertSpill(s)
	// Apple-Silicon Metal GPU forward (`fak serve --metal`): engage the metalgemm GPU
	// prefill + GPU-resident Q8 decode on the CPU session. Guarded to backend==nil — Metal
	// is the CPU-session seam (s.Backend stays nil), and setting s.Metal on a device session
	// is incoherent; serve also rejects --metal with --backend up front. s.MetalQ4K mirrors
	// cmd/fakchat (s.MetalQ4K = q4k && metal). Inert on non-Metal builds (the model
	// package's metal dispatch falls back to CPU) and the resident decode self-declines any
	// non-dense-Qwen-Q8 model, so this never forces an unsupported GPU path.
	if p.backend == nil && p.metal {
		s.Metal = true
		s.MetalQ4K = p.q4k
	}

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
		if p.backend != nil && checkpoint > prefillAt {
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
			logits = s.Prefill(ids[prefillAt:])
		}
		prefillS = time.Since(tp).Seconds()
	}
	if err = ctx.Err(); err != nil {
		return
	}

	// 3) Snapshot the full-prompt KV (before decode mutates s.Cache) and cache it under a
	// fresh Lookup→Insert→Done. The snapshot covers the FULL ids prefix, so it is a valid
	// leaf kv no matter how much a concurrent turn may have inserted since step 1.
	if reuse {
		if p.backend != nil {
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
	}
	td := time.Now()
	if p.batchDecode {
		// Opt-in: drive this one request through the shared continuous-batch step. For B==1
		// StepBatchActive is exactly Seqs[0].Step, so the served tokens are unchanged; this is
		// the wiring a cross-request coalescer builds on (aggregate throughput is box-gated).
		inKernelDecodeLanesBatched(ctx, []*decodeLane{ln}, p.m, p.quant)
	} else {
		inKernelDecodeSerial(ctx, ln)
	}
	gen, stopped, err = ln.gen, ln.stopped, ln.err
	decodeS = time.Since(td).Seconds()
	return
}

// decodeLane is one request's live decode state. decodeOne runs one token's worth of the
// decode loop body EXCEPT the model forward, so the serial driver (Session.Step) and the
// opt-in batched driver (BatchSession.StepBatchActive) share identical per-token semantics —
// the property that makes the two paths bit-for-bit equivalent.
type decodeLane struct {
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

	gen     int
	stopped bool
	done    bool
	err     error
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
	if ln.counts != nil && next < len(ln.counts) {
		ln.counts[next]++
	}
	if ln.emit != nil && ln.emit(next) {
		ln.gen++ // this token WAS generated; count it before finishing
		ln.stopped, ln.done = true, true
		return 0, false
	}
	if ln.emit != nil {
		if err := ctx.Err(); err != nil {
			ln.gen++ // this token was emitted before cancellation became visible
			ln.err, ln.done = err, true
			return 0, false
		}
	}
	if ln.gen == ln.maxNew-1 {
		ln.gen++ // this token was generated; avoid computing unused next-token logits.
		ln.done = true
		return 0, false
	}
	ln.gen++ // matches the serial loop's post-increment before the next Step.
	return next, true
}

// inKernelDecodeSerial is the DEFAULT per-request decode: one lane advanced by Session.Step,
// byte-identical to the pre-seam decode loop (the gen<maxNew guard preserves the maxNew<=0
// no-token contract). It is the path an unset FAK_INKERNEL_BATCH takes.
func inKernelDecodeSerial(ctx context.Context, ln *decodeLane) {
	for ln.gen < ln.maxNew {
		next, advance := ln.decodeOne(ctx)
		if !advance {
			return
		}
		ln.logits = ln.s.Step(next)
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
func inKernelDecodeLanesBatched(ctx context.Context, lanes []*decodeLane, m *model.Model, quant bool) {
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
			next, advance := ln.decodeOne(ctx)
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
		out := bs.StepBatchActive(ids, active)
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
