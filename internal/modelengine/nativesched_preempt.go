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
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const nativePreemptVictimMostRecent = "most-recent"

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

// NativePreemptionPolicy arms the scheduler's paged-KV block pressure path.
type NativePreemptionPolicy struct {
	Mode        NativePreemptionMode
	MaxBlocks   int // <=0 disables preemption; positive means a paged-KV block budget exists
	BlockTokens int // tokens per paged-KV block; <=0 defaults to 16
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
	st.Running = len(s.lanes)
	st.UsedBlocks = s.usedKVBlocksLocked()
	st.SwappedOut = len(s.preempted)
	st.MaxBlocks = s.preemption.MaxBlocks
	st.MaxPreemptRounds = s.maxPreemptRoundsLocked()
	return st
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
	writeNativeHelpType(b, p+"victim_rule", "Active victim-selection rule (0=most-recent).", "gauge")
	fmt.Fprintf(b, "%svictim_rule %d\n", p, nativePreemptVictimRuleCode(st.VictimReason))
	writeNativeHelpType(b, p+"max_wait_rounds", "Oldest current victim's age in readmit rounds (starvation visibility).", "gauge")
	fmt.Fprintf(b, "%smax_wait_rounds %d\n", p, st.MaxPreemptRounds)
	writeNativeCounter(b, p+"total", "Sequences preempted under KV-block exhaustion.", st.Preemptions)
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

func nativePreemptVictimRuleCode(reason string) int {
	switch reason {
	case "", nativePreemptVictimMostRecent:
		return 0
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
		pool := model.NewPagedKVPoolWithRaw(s.m.Cfg, s.blockTokensLocked())
		seq, err := pool.RestoreFromHost(ln.hostKV)
		if err != nil {
			return err
		}
		ln.sess = s.sessionFromCache(seq.ToKVCache(s.m.Cfg), ln.q4k)
		seq.Free()
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
		idx := s.mostRecentPreemptibleLaneLocked()
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

func (s *NativeScheduler) preemptLaneLocked(ln *schedLane) error {
	ln.preemptMode = s.preemption.Mode
	ln.preemptRound = s.preemptRound
	ln.savedLogits = copyF32(ln.logits)
	s.preemptStats.Preemptions++
	s.preemptStats.VictimReason = nativePreemptVictimMostRecent
	switch s.preemption.Mode {
	case NativePreemptRecompute:
		s.preemptStats.RecomputeCount++
	case NativePreemptSwap:
		if ln.sess == nil || ln.sess.Cache == nil {
			return fmt.Errorf("modelengine: cannot swap preempt lane without resident KV")
		}
		pool := model.NewPagedKVPoolWithRaw(s.m.Cfg, s.blockTokensLocked())
		seq, err := model.KVCacheToPaged(pool, ln.sess.Cache)
		if err != nil {
			return err
		}
		blob, err := seq.SwapToHost()
		seq.Free()
		if err != nil {
			return err
		}
		ln.hostKV = blob
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

func (s *NativeScheduler) newLaneSession(q4k bool) *model.Session {
	sess := s.m.NewSession()
	if q4k {
		sess.Quant = true
		sess.Q4K = true
	}
	return sess
}

func (s *NativeScheduler) sessionFromCache(cache *model.KVCache, q4k bool) *model.Session {
	sess := &model.Session{M: s.m, Cache: cache}
	if q4k {
		sess.Quant = true
		sess.Q4K = true
	}
	return sess
}
