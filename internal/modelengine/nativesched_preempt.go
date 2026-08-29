package modelengine

// nativesched_preempt.go — issue #31's live scheduler pressure path.
//
// NativeScheduler keeps this disabled until a host sets a positive MaxBlocks. That positive
// bound is the structural paged-KV dependency: without a paged block budget there is no page
// unit to reclaim, so the scheduler behaves exactly as it did before. When armed, the loop
// checks the running set at step boundaries and preempts the most-recently-admitted lane until
// the live block estimate fits. Swap mode serializes the victim's real KV bytes through the
// model.PagedKV swap blob; recompute mode drops KV and re-prefills prompt+generated tokens on
// readmit. In both modes the lane's token stream stays open and resumes after readmit.

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelperfobs"
)

const (
	nativePreemptVictimMostRecent = "most-recent"
	nativePreemptVictimCostAware  = "cost-aware"
	nativeKVBMDefaultPinTTL       = time.Minute
)

// NativePreemptionMode selects how the scheduler releases a victim's KV under pressure.
type NativePreemptionMode uint8

const (
	// NativePreemptSwap snapshots the victim's KV into paged blocks, serializes those blocks
	// to host bytes, and restores them on readmit.
	NativePreemptSwap NativePreemptionMode = iota
	// NativePreemptRecompute drops the victim's KV and replays prompt+generated tokens on
	// readmit to rebuild the same cache state.
	NativePreemptRecompute
)

// NativePreemptionVictimRule selects which running lane is swapped/recomputed when
// the native scheduler's paged-KV block budget is exceeded.
type NativePreemptionVictimRule uint8

const (
	// NativePreemptVictimMostRecent preserves the pre-#2239 behavior: preempt the newest
	// running lane first.
	NativePreemptVictimMostRecent NativePreemptionVictimRule = 0
	// NativePreemptVictimCostAware uses compute.PickEvictionVictim over scheduler-local
	// KVBM hints. Metric code 1 is already used by the gateway preemptor for
	// lowest-priority, so native cost-aware is exported as 2.
	NativePreemptVictimCostAware NativePreemptionVictimRule = 2
)

func (r NativePreemptionVictimRule) String() string {
	switch r {
	case NativePreemptVictimCostAware:
		return nativePreemptVictimCostAware
	default:
		return nativePreemptVictimMostRecent
	}
}

// NativePreemptionPolicy arms the scheduler's paged-KV block pressure path.
type NativePreemptionPolicy struct {
	Mode        NativePreemptionMode
	VictimRule  NativePreemptionVictimRule
	MaxBlocks   int // <=0 disables preemption; positive means a paged-KV block budget exists
	BlockTokens int // tokens per paged-KV block; <=0 defaults to 16
	// UsageLedgerPath is the declared, reversible JSONL target for Qwen hybrid
	// swap codec invocations. Empty disables the writer without a default path.
	UsageLedgerPath string
}

// NativePreemptionStats is the scheduler-local cumulative preemption witness.
type NativePreemptionStats struct {
	Running           int
	UsedBlocks        int
	SwappedOut        int
	MaxBlocks         int
	MaxPreemptRounds  int64
	Preemptions       int64
	SwapPreemptions   int64
	RecomputeCount    int64
	SwapBytes         int64
	Readmitted        int64
	SwapRestoredBytes int64
	VictimReason      string
	VictimRule        NativePreemptionVictimRule
	CostAwareVictims  int64
	PinnedSkipped     int64
	ExpiredPins       int64
	LastVictimCost    float64
	LastVictimTokens  int
	LastVictimBlocks  int
	LastVictimHits    int
	LastCandidates    int
	LastPinned        int
	LastExpiredPins   int
}

// SetKVPreemptionPolicy configures the scheduler's opt-in paged-KV preemption path. It is
// intended to be set before first Admit; changing it live is safe but only affects future loop
// iterations.
func (s *NativeScheduler) SetKVPreemptionPolicy(p NativePreemptionPolicy) {
	if p.BlockTokens <= 0 {
		p.BlockTokens = 16
	}
	switch p.Mode {
	case NativePreemptSwap, NativePreemptRecompute:
	default:
		p.Mode = NativePreemptSwap
	}
	switch p.VictimRule {
	case NativePreemptVictimMostRecent, NativePreemptVictimCostAware:
	default:
		p.VictimRule = NativePreemptVictimMostRecent
	}
	s.mu.Lock()
	s.preemption = p
	s.mu.Unlock()
	s.signal()
}

// KVPreemptionStats returns a point-in-time view of the scheduler's pressure path.
func (s *NativeScheduler) KVPreemptionStats() NativePreemptionStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	st := s.preemptStats
	st.Running = s.runningKVLanesLocked()
	st.UsedBlocks = s.usedKVBlocksLocked()
	st.SwappedOut = len(s.preempted)
	st.MaxBlocks = s.preemption.MaxBlocks
	st.MaxPreemptRounds = s.maxPreemptRoundsLocked()
	st.VictimRule = s.preemption.VictimRule
	return st
}

func (s *NativeScheduler) runningKVLanesLocked() int {
	n := 0
	for _, ln := range s.lanes {
		if ln != nil && !ln.terminal && ln.sess != nil {
			n++
		}
	}
	return n
}

// WriteKVPreemptionMetrics renders the live native scheduler's #31 counters in the same
// fak_sched_preempt_* family the gateway preemptor exposes, so /metrics reports the actual
// scheduler preemptions when fak serve attaches this scheduler as the metric writer.
func (s *NativeScheduler) WriteKVPreemptionMetrics(b *strings.Builder) {
	if s == nil || b == nil {
		return
	}
	writeNativePreemptionMetrics(b, s.KVPreemptionStats())
}

func writeNativePreemptionMetrics(b *strings.Builder, st NativePreemptionStats) {
	const p = "fak_sched_preempt_"
	writeNativeHelpType(b, p+"running", "Sequences currently running (holding KV blocks) under the preemptor.", "gauge")
	fmt.Fprintf(b, "%srunning %d\n", p, st.Running)
	writeNativeHelpType(b, p+"used_blocks", "KV blocks currently held by the running set.", "gauge")
	fmt.Fprintf(b, "%sused_blocks %d\n", p, st.UsedBlocks)
	writeNativeHelpType(b, p+"max_blocks", "Configured paged-KV block capacity (0 = no paged pool; preemption disarmed).", "gauge")
	fmt.Fprintf(b, "%smax_blocks %d\n", p, st.MaxBlocks)
	writeNativeHelpType(b, p+"swapped_out", "Sequences currently swapped/awaiting readmit (preempted).", "gauge")
	fmt.Fprintf(b, "%sswapped_out %d\n", p, st.SwappedOut)
	writeNativeHelpType(b, p+"victim_rule", "Active victim-selection rule (0=most-recent, 1=lowest-priority, 2=cost-aware).", "gauge")
	fmt.Fprintf(b, "%svictim_rule %d\n", p, nativePreemptVictimRuleCode(st.VictimRule, st.VictimReason))
	writeNativeHelpType(b, p+"last_candidates", "Candidate lanes considered by the last victim-selection pass.", "gauge")
	fmt.Fprintf(b, "%slast_candidates %d\n", p, st.LastCandidates)
	writeNativeHelpType(b, p+"last_pinned", "Pinned candidate lanes skipped by the last victim-selection pass.", "gauge")
	fmt.Fprintf(b, "%slast_pinned %d\n", p, st.LastPinned)
	writeNativeHelpType(b, p+"last_expired_pins", "Pinned candidate lanes whose TTL had expired in the last victim-selection pass.", "gauge")
	fmt.Fprintf(b, "%slast_expired_pins %d\n", p, st.LastExpiredPins)
	writeNativeHelpType(b, p+"last_victim_cost", "Cost-aware score of the last selected victim; 0 when no cost-aware victim has been selected.", "gauge")
	fmt.Fprintf(b, "%slast_victim_cost %g\n", p, st.LastVictimCost)
	writeNativeHelpType(b, p+"max_wait_rounds", "Oldest current victim's age in readmit rounds (starvation visibility).", "gauge")
	fmt.Fprintf(b, "%smax_wait_rounds %d\n", p, st.MaxPreemptRounds)
	writeNativeCounter(b, p+"total", "Sequences preempted under KV-block exhaustion.", st.Preemptions)
	writeNativeCounter(b, p+"cost_aware_total", "Sequences preempted by the cost-aware KVBM victim picker.", st.CostAwareVictims)
	writeNativeCounter(b, p+"pinned_skipped_total", "Pinned lanes skipped by the cost-aware KVBM victim picker.", st.PinnedSkipped)
	writeNativeCounter(b, p+"pin_expired_total", "Pinned lanes whose KVBM pin TTL had expired when considered by the cost-aware victim picker.", st.ExpiredPins)
	writeNativeCounter(b, p+"swap_total", "Preemptions taken via KV swap-to-host.", st.SwapPreemptions)
	writeNativeCounter(b, p+"recompute_total", "Preemptions taken via drop-and-recompute.", st.RecomputeCount)
	writeNativeCounter(b, p+"swap_bytes_total", "KV bytes swapped out to host DRAM.", st.SwapBytes)
	writeNativeCounter(b, p+"readmitted_total", "Preempted sequences readmitted to the running set.", st.Readmitted)
	writeNativeCounter(b, p+"swap_restored_bytes_total", "KV bytes restored from host DRAM on readmit.", st.SwapRestoredBytes)
}

func writeNativeHelpType(b *strings.Builder, name, help, typ string) {
	fmt.Fprintf(b, "# HELP %s %s\n# TYPE %s %s\n", name, help, name, typ)
}

func writeNativeCounter(b *strings.Builder, name, help string, v int64) {
	writeNativeHelpType(b, name, help, "counter")
	fmt.Fprintf(b, "%s %d\n", name, v)
}

func nativePreemptVictimRuleCode(rule NativePreemptionVictimRule, reason string) int {
	switch rule {
	case NativePreemptVictimCostAware:
		return int(NativePreemptVictimCostAware)
	case NativePreemptVictimMostRecent:
		return 0
	}
	switch reason {
	case "", nativePreemptVictimMostRecent:
		return int(NativePreemptVictimMostRecent)
	case nativePreemptVictimCostAware:
		return int(NativePreemptVictimCostAware)
	default:
		return -1
	}
}

func (s *NativeScheduler) maxPreemptRoundsLocked() int64 {
	var oldest int64
	for _, ln := range s.preempted {
		if w := s.preemptRound - ln.preemptRound; w > oldest {
			oldest = w
		}
	}
	return oldest
}

func (s *NativeScheduler) preemptionEnabledLocked() bool {
	return s.preemption.MaxBlocks > 0
}

func (s *NativeScheduler) blockTokensLocked() int {
	if s.preemption.BlockTokens > 0 {
		return s.preemption.BlockTokens
	}
	return 16
}

func (s *NativeScheduler) laneKVBlocksLocked(ln *schedLane) int {
	if ln == nil {
		return 0
	}
	tokens := ln.promptLen + ln.emitted
	if ln.sess != nil && ln.sess.Cache != nil && ln.sess.Cache.Len() > tokens {
		tokens = ln.sess.Cache.Len()
	}
	if tokens <= 0 {
		return 0
	}
	bt := s.blockTokensLocked()
	return (tokens + bt - 1) / bt
}

func (s *NativeScheduler) usedKVBlocksLocked() int {
	used := 0
	for _, ln := range s.lanes {
		if !ln.terminal {
			used += s.laneKVBlocksLocked(ln)
		}
	}
	return used
}

func (s *NativeScheduler) dropCanceledPreemptedLocked() {
	if len(s.preempted) == 0 {
		return
	}
	kept := s.preempted[:0]
	for _, ln := range s.preempted {
		if ln.ctx.Err() != nil {
			ln.hostKV = nil
			ln.savedLogits = nil
			ln.finish(nil, ln.ctx.Err())
			continue
		}
		kept = append(kept, ln)
	}
	s.preempted = kept
}

func (s *NativeScheduler) readmitPreemptedLocked() {
	if !s.preemptionEnabledLocked() || len(s.preempted) == 0 {
		return
	}
	s.preemptRound++
	sort.SliceStable(s.preempted, func(i, j int) bool {
		if s.preempted[i].preemptRound != s.preempted[j].preemptRound {
			return s.preempted[i].preemptRound < s.preempted[j].preemptRound
		}
		return s.preempted[i].seqNo < s.preempted[j].seqNo
	})

	kept := s.preempted[:0]
	blocked := false
	for _, ln := range s.preempted {
		need := s.laneKVBlocksLocked(ln)
		if blocked || (s.maxRunning > 0 && len(s.lanes) >= s.maxRunning) {
			blocked = true
			kept = append(kept, ln)
			continue
		}
		used := s.usedKVBlocksLocked()
		if used > 0 && used+need > s.preemption.MaxBlocks {
			blocked = true
			kept = append(kept, ln)
			continue
		}
		if err := s.restorePreemptedLaneLocked(ln); err != nil {
			ln.hostKV = nil
			ln.savedLogits = nil
			ln.finish(nil, err)
			continue
		}
		s.lanes = append(s.lanes, ln)
		s.preemptStats.Readmitted++
	}
	s.preempted = append([]*schedLane(nil), kept...)
}

func (s *NativeScheduler) restorePreemptedLaneLocked(ln *schedLane) error {
	switch ln.preemptMode {
	case NativePreemptRecompute:
		sess := s.newLaneSession(ln.q4k)
		history := make([]int, 0, len(ln.prompt)+len(ln.gen))
		history = append(history, ln.prompt...)
		history = append(history, ln.gen...)
		ln.logits = copyF32(sess.Prefill(history))
		ln.sess = sess
	case NativePreemptSwap:
		var cache *model.KVCache
		qwenHybrid := s.m.Cfg.IsQwen35Hybrid()
		if qwenHybrid {
			var err error
			cache, err = model.QwenHybridKVCacheFromHost(s.m.Cfg, ln.hostKV)
			if err != nil {
				_ = s.recordQwenSwapUsage(modelperfobs.QwenSwapDirectionIn, modelperfobs.QwenSwapOutcomeError, modelperfobs.QwenSwapResultRefused, len(ln.hostKV))
				return err
			}
		} else {
			pool := model.NewPagedKVPoolWithRaw(s.m.Cfg, s.blockTokensLocked())
			seq, err := pool.RestoreFromHost(ln.hostKV)
			if err != nil {
				return err
			}
			cache = seq.ToKVCache(s.m.Cfg)
			seq.Free()
		}
		candidate := s.sessionFromCache(cache, ln.q4k)
		if qwenHybrid {
			history := make([]int, 0, len(ln.prompt)+len(ln.gen))
			history = append(history, ln.prompt...)
			history = append(history, ln.gen...)
			if _, err := candidate.RestoreTokenLineage(history); err != nil {
				candidate.Close()
				operationErr := fmt.Errorf("modelengine: restore Qwen swap token lineage: %w", err)
				_ = s.recordQwenSwapUsage(modelperfobs.QwenSwapDirectionIn, modelperfobs.QwenSwapOutcomeSuccess, modelperfobs.QwenSwapResultRefused, len(ln.hostKV))
				return operationErr
			}
			_ = s.recordQwenSwapUsage(modelperfobs.QwenSwapDirectionIn, modelperfobs.QwenSwapOutcomeSuccess, modelperfobs.QwenSwapResultCommitted, len(ln.hostKV))
		}
		ln.sess = candidate
		ln.logits = copyF32(ln.savedLogits)
		s.preemptStats.SwapRestoredBytes += int64(len(ln.hostKV))
	default:
		return fmt.Errorf("modelengine: unknown native preemption mode %d", ln.preemptMode)
	}
	ln.hostKV = nil
	ln.savedLogits = nil
	return nil
}

func (s *NativeScheduler) enforcePreemptionLocked() {
	if !s.preemptionEnabledLocked() {
		return
	}
	for s.usedKVBlocksLocked() > s.preemption.MaxBlocks {
		idx := s.preemptibleLaneLocked()
		if idx < 0 {
			return
		}
		ln := s.lanes[idx]
		if err := s.preemptLaneLocked(ln); err != nil {
			ln.finish(nil, err)
		}
		s.lanes = append(s.lanes[:idx], s.lanes[idx+1:]...)
	}
}

func (s *NativeScheduler) preemptibleLaneLocked() int {
	switch s.preemption.VictimRule {
	case NativePreemptVictimCostAware:
		return s.costAwarePreemptibleLaneLocked()
	default:
		return s.mostRecentPreemptibleLaneLocked()
	}
}

func (s *NativeScheduler) mostRecentPreemptibleLaneLocked() int {
	if len(s.lanes) <= 1 {
		return -1
	}
	best := -1
	var bestSeq int64
	for i, ln := range s.lanes {
		if ln == nil || ln.terminal || ln.ctx.Err() != nil || ln.sess == nil {
			continue
		}
		if best < 0 || ln.seqNo > bestSeq || (ln.seqNo == bestSeq && ln.tool < s.lanes[best].tool) {
			best = i
			bestSeq = ln.seqNo
		}
	}
	return best
}

func (s *NativeScheduler) costAwarePreemptibleLaneLocked() int {
	if len(s.lanes) <= 1 {
		return -1
	}
	spans := make([]compute.KVSpanStats, 0, len(s.lanes))
	laneIndexes := make([]int, 0, len(s.lanes))
	pinned, expired := 0, 0
	now := time.Now()
	for i, ln := range s.lanes {
		stats, ok := s.laneKVCostStatsLockedAt(ln, now)
		if !ok {
			continue
		}
		if stats.Pinned {
			pinned++
		} else if ln.kvPinExpired(now) {
			expired++
		}
		spans = append(spans, stats)
		laneIndexes = append(laneIndexes, i)
	}
	s.preemptStats.LastCandidates = len(spans)
	s.preemptStats.LastPinned = pinned
	s.preemptStats.LastExpiredPins = expired
	s.preemptStats.PinnedSkipped += int64(pinned)
	s.preemptStats.ExpiredPins += int64(expired)
	victim := compute.PickEvictionVictim(spans)
	if victim < 0 {
		return -1
	}
	stats := spans[victim]
	s.preemptStats.LastVictimCost = compute.KVEvictionCost(stats)
	s.preemptStats.LastVictimTokens = stats.Tokens
	s.preemptStats.LastVictimBlocks = blocksFromStats(stats, s.blockTokensLocked())
	s.preemptStats.LastVictimHits = stats.Hits
	return laneIndexes[victim]
}

func (s *NativeScheduler) laneKVCostStatsLockedAt(ln *schedLane, now time.Time) (compute.KVSpanStats, bool) {
	if ln == nil || ln.terminal || ln.ctx.Err() != nil || ln.sess == nil {
		return compute.KVSpanStats{}, false
	}
	tokens := ln.promptLen + ln.emitted
	if ln.sess.Cache != nil && ln.sess.Cache.Len() > tokens {
		tokens = ln.sess.Cache.Len()
	}
	if tokens <= 0 {
		return compute.KVSpanStats{}, false
	}
	blocks := (tokens + s.blockTokensLocked() - 1) / s.blockTokensLocked()
	return compute.KVSpanStats{
		Tokens:   tokens,
		Bytes:    int64(blocks * s.blockTokensLocked()),
		Hits:     ln.kvReuseHits,
		LastUsed: uint64(ln.seqNo),
		Pinned:   ln.kvPinActive(now),
	}, true
}

func (ln *schedLane) kvPinActive(now time.Time) bool {
	if ln == nil || !ln.kvPinned {
		return false
	}
	if ln.kvPinUntil.IsZero() {
		return true
	}
	return now.Before(ln.kvPinUntil)
}

func (ln *schedLane) kvPinExpired(now time.Time) bool {
	return ln != nil && ln.kvPinned && !ln.kvPinUntil.IsZero() && !now.Before(ln.kvPinUntil)
}

func blocksFromStats(stats compute.KVSpanStats, blockTokens int) int {
	if blockTokens <= 0 || stats.Tokens <= 0 {
		return 0
	}
	return (stats.Tokens + blockTokens - 1) / blockTokens
}

func (s *NativeScheduler) preemptLaneLocked(ln *schedLane) error {
	ln.preemptMode = s.preemption.Mode
	ln.preemptRound = s.preemptRound
	ln.savedLogits = copyF32(ln.logits)
	s.preemptStats.Preemptions++
	s.preemptStats.VictimReason = s.preemption.VictimRule.String()
	if s.preemption.VictimRule == NativePreemptVictimCostAware {
		s.preemptStats.CostAwareVictims++
	}
	switch s.preemption.Mode {
	case NativePreemptRecompute:
		s.preemptStats.RecomputeCount++
	case NativePreemptSwap:
		if ln.sess == nil || ln.sess.Cache == nil {
			return fmt.Errorf("modelengine: cannot swap preempt lane without resident KV")
		}
		var blob []byte
		var err error
		qwenHybrid := s.m.Cfg.IsQwen35Hybrid()
		if qwenHybrid {
			blob, err = model.QwenHybridKVCacheToHost(ln.sess.Cache, s.blockTokensLocked())
		} else {
			pool := model.NewPagedKVPoolWithRaw(s.m.Cfg, s.blockTokensLocked())
			seq, pageErr := model.KVCacheToPaged(pool, ln.sess.Cache)
			if pageErr != nil {
				return pageErr
			}
			blob, err = seq.SwapToHost()
			seq.Free()
		}
		if err != nil {
			if qwenHybrid {
				_ = s.recordQwenSwapUsage(modelperfobs.QwenSwapDirectionOut, modelperfobs.QwenSwapOutcomeError, modelperfobs.QwenSwapResultRefused, 0)
			}
			return err
		}
		ln.hostKV = blob
		if qwenHybrid {
			_ = s.recordQwenSwapUsage(modelperfobs.QwenSwapDirectionOut, modelperfobs.QwenSwapOutcomeSuccess, modelperfobs.QwenSwapResultCommitted, len(blob))
		}
		s.preemptStats.SwapPreemptions++
		s.preemptStats.SwapBytes += int64(len(blob))
	default:
		return fmt.Errorf("modelengine: unknown native preemption mode %d", s.preemption.Mode)
	}
	if ln.sess != nil {
		ln.sess.Close()
	}
	ln.sess = nil
	s.preempted = append(s.preempted, ln)
	return nil
}

func (s *NativeScheduler) recordQwenSwapUsage(direction, outcome, result string, bytes int) error {
	return modelperfobs.AppendQwenSwapUsage(s.preemption.UsageLedgerPath, modelperfobs.QwenSwapUsageRow{
		Schema: modelperfobs.QwenSwapUsageSchema, ObservedAt: time.Now().UTC(), Version: modelperfobs.QwenSwapCodecVersion,
		Direction: direction, Outcome: outcome, Result: result, Bytes: int64(bytes),
	})
}

func (s *NativeScheduler) newLaneSession(q4k bool) *model.Session {
	sess := s.m.NewSession()
	if q4k {
		sess.Quant = true
		sess.Q4K = true
		sess.Q4KGateUpOutputSlab = s.q4kGateUpOutputSlab
	}
	return sess
}

func nativeKVBMHintsFromMeta(meta map[string]string) (reuseHits int, pinned bool, pinUntil time.Time) {
	return nativeKVBMHintsFromMetaAt(meta, time.Now())
}

func nativeKVBMHintsFromMetaAt(meta map[string]string, now time.Time) (reuseHits int, pinned bool, pinUntil time.Time) {
	reuseHits = nonNegativeMetaInt(meta, "kv_reuse_hits", "kv.reuse_hits")
	pinned = metaBool(meta, "kv_pin", "kv.pin", "kv_preempt_pin", "kv.preempt_pin")
	if !pinned {
		return reuseHits, false, time.Time{}
	}
	ttl := nativeKVBMDefaultPinTTL
	if ttlMS := nonNegativeMetaInt(meta, "kv_pin_ttl_ms", "kv.pin_ttl_ms", "kv_preempt_pin_ttl_ms", "kv.preempt_pin_ttl_ms"); ttlMS > 0 {
		ttl = time.Duration(ttlMS) * time.Millisecond
	}
	pinUntil = now.Add(ttl)
	return reuseHits, pinned, pinUntil
}

func nonNegativeMetaInt(meta map[string]string, keys ...string) int {
	for _, key := range keys {
		if raw, ok := meta[key]; ok {
			n, err := strconv.Atoi(strings.TrimSpace(raw))
			if err == nil && n > 0 {
				return n
			}
			return 0
		}
	}
	return 0
}

func metaBool(meta map[string]string, keys ...string) bool {
	for _, key := range keys {
		raw, ok := meta[key]
		if !ok {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(raw)) {
		case "1", "t", "true", "y", "yes", "pin", "pinned":
			return true
		default:
			return false
		}
	}
	return false
}

func (s *NativeScheduler) sessionFromCache(cache *model.KVCache, q4k bool) *model.Session {
	sess := &model.Session{M: s.m, Cache: cache}
	if q4k {
		sess.Quant = true
		sess.Q4K = true
		sess.Q4KGateUpOutputSlab = s.q4kGateUpOutputSlab
	}
	return sess
}
