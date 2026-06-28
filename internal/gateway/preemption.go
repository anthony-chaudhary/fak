package gateway

// preemption.go — the fak-NATIVE serving node's PREEMPTION + KV swap-to-host / recompute
// gate under memory pressure (issue #31, under #50). This is the scheduler-level escape
// valve admission.go (#35) names as a sibling seed and explicitly scopes OUT of itself
// ("The KV-block budget, preemption / KV swap-out ... are explicit non-goals here"): when
// the paged/block KV allocator (internal/model PagedKVPool, #277) reports exhaustion, the
// scheduler PREEMPTS a running sequence — swapping its KV blocks out to host DRAM and
// restoring them on readmit, or dropping them and recomputing on readmit — so a node makes
// forward progress instead of OOM-ing or deadlocking under bursty / long-context load.
//
// WHY IT EXISTS — the gap it closes. Admission (admission.go) can OVERCOMMIT a node:
// requests are admitted on the token budget, but their KV footprint grows during decode and
// can outrun the paged pool. With no preemption the only outcomes are OOM or a deadlocked
// step (the whole batch stalls because one lane cannot allocate its next block). This file
// is the missing escape valve — under KV-block exhaustion it sheds a VICTIM's KV (not the
// request) so the rest of the batch keeps stepping, then readmits the victim when blocks
// free, with a starvation guard so no victim waits forever.
//
// STRICTLY GATED behind the paged allocator (acceptance bullet 3). Preemption speaks in KV
// BLOCKS — the page unit only the paged allocator has; a contiguous-append cache (kvcache.go)
// has no page to swap or reclaim. The gate is structural: MaxBlocks is the paged pool's block
// capacity, and with no block bound (<=0, the contiguous-cache stand-in) Admit NEVER preempts
// — byte-identical to the pre-paged path (TestPreemptGatedBehindPagedPool).
//
// BIT-EXACT (the native value). swap holds the victim's serialized KV bytes VERBATIM and
// returns them byte-identical on readmit, so a swapped-and-restored sequence resumes from the
// exact same KV a never-preempted one held — the serializer round-trip is PagedKV
// GatherK/Append, proven byte-for-byte in paged_evict.go. recompute drops the KV and retains
// the prompt so readmit re-prefills the SAME tokens through the same deterministic kernel (the
// StepBatch f32 guarantee, witnessed end-to-end in native_sched_gateway_test.go) — re-deriving
// identical Kraw/K. This file owns the state-machine + accounting half; the model-forward
// equality rests on those two existing witnesses.
//
// HONEST FENCE — house form, like admission.go / batchsched.go. A pure, deterministic,
// stdlib-only policy: a value type + total methods, no hidden clock (a monotone seqNo orders
// "most-recently-admitted", a round counter drives readmit aging) and no randomness, so it is
// unit-testable to an exact preemption/readmit sequence. It moves no real KV and runs no model
// (the KV blob it round-trips models the host serializer's bytes); the host injects the real
// serializer and folds WriteMetrics into renderMetrics once the native scheduler is on the
// live serve loop — the same deferral admission.go documents.

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

// PreemptionMode is the swap-vs-recompute policy knob (acceptance bullet 4) — one knob, two
// strategies for reclaiming a victim's KV blocks.
type PreemptionMode uint8

const (
	// PreemptSwap serializes the victim's paged KV blocks to host DRAM and frees the GPU/HBM
	// pages; readmit restores the held bytes verbatim (bit-exact round-trip). The vLLM-V1
	// "swap" mode — cheaper readmit (no recompute) at the cost of host DRAM + transfer.
	PreemptSwap PreemptionMode = iota
	// PreemptRecompute drops the victim's KV outright and frees its pages; readmit re-prefills
	// the retained prompt+generated tokens through the deterministic kernel (re-deriving the
	// same Kraw/K). The vLLM-V1 "recompute" mode — no host DRAM, at the cost of re-prefill.
	PreemptRecompute
)

// String renders a mode as its lowercase token; an out-of-range value renders "unknown"
// rather than panicking, matching the package's other enums (AdmissionVerdict.String).
func (m PreemptionMode) String() string {
	switch m {
	case PreemptSwap:
		return "swap"
	case PreemptRecompute:
		return "recompute"
	}
	return "unknown"
}

// VictimRule is the deterministic victim-selection order (acceptance bullet 5). Both rules are
// total and clock-free, so the chosen victim is byte-deterministic for a given running set.
type VictimRule uint8

const (
	// VictimMostRecent preempts the most-recently-admitted running sequence first (highest
	// seqNo). This is vLLM V1's default: the newest sequence has the least sunk cost to redo
	// and preempting it leaves older, closer-to-done sequences to finish and free their blocks.
	VictimMostRecent VictimRule = iota
	// VictimLowestPriority preempts the least-important running sequence first (highest
	// Priority value, lower-is-higher like the rest of the gateway), and NEVER preempts a
	// sequence the newcomer does not outrank — so a low-priority arrival cannot evict a
	// high-priority incumbent (no priority inversion).
	VictimLowestPriority
)

// String renders a victim rule as its lowercase token; out-of-range renders "unknown".
func (r VictimRule) String() string {
	switch r {
	case VictimMostRecent:
		return "most-recent"
	case VictimLowestPriority:
		return "lowest-priority"
	}
	return "unknown"
}

// PreemptionPolicy holds the preemption knobs. The zero value is usable (swap, most-recent,
// no block bound → never preempts); build the shipping defaults with DefaultPreemptionPolicy.
type PreemptionPolicy struct {
	// Mode selects swap vs recompute for every preemption (acceptance bullet 4).
	Mode PreemptionMode
	// Victim selects the deterministic victim-selection order (acceptance bullet 5).
	Victim VictimRule
	// MaxBlocks is the paged KV pool's total block capacity — the exhaustion bound preemption
	// fires at. ≤0 means NO paged pool (the contiguous-append cache): Admit never preempts and
	// always admits, the structural gate that keeps preemption behind the paged allocator.
	MaxBlocks int
	// AgingRounds is the readmit starvation guard: a preempted sequence's effective readmit
	// priority improves by one for every AgingRounds readmit rounds it waits, so it climbs to
	// the head and is readmitted within a bounded number of rounds once blocks free. ≤0
	// disables aging (raw priority stands; a flood of higher-priority preemptions can starve a
	// low-priority victim — TestPreemptNoStarvation asserts both directions).
	AgingRounds int
}

// DefaultPreemptionPolicy returns the shipping defaults: swap mode (cheaper readmit),
// most-recently-admitted victim selection (the vLLM V1 default), and aging every round so no
// victim is ever starved. MaxBlocks is left 0 — the host sets it to the live paged pool's
// PhysicalBlocks capacity, which is also what arms the gate.
func DefaultPreemptionPolicy() PreemptionPolicy {
	return PreemptionPolicy{Mode: PreemptSwap, Victim: VictimMostRecent, AgingRounds: 1}
}

// KVSeq is one sequence in the paged KV pool: its KV-block footprint, its priority, the
// prompt-token count needed to re-prefill it (recompute), and its serialized KV bytes (the
// swap-to-host payload, held verbatim across a swap so readmit restores it bit-identically).
type KVSeq struct {
	TraceID      string
	Priority     int    // lower is higher (Priority-ascending, like SeqRequest/admission.go)
	Blocks       int    // KV-block footprint held in the paged pool
	PromptTokens int    // tokens to re-prefill on a recompute readmit
	KV           []byte // serialized KV; the swap payload, round-tripped byte-for-byte
}

// PreemptionEvent records one victim preemption — what was reclaimed and why — for the Admit
// caller and the metrics counters.
type PreemptionEvent struct {
	TraceID string
	Mode    PreemptionMode
	Blocks  int    // KV blocks freed back to the pool
	Bytes   int    // KV bytes swapped to host DRAM (0 for recompute)
	Reason  string // victim-selection reason token (the VictimRule that picked it)
}

// AdmitResult is the outcome of offering a sequence to the KV-block pool.
type AdmitResult struct {
	Admitted  bool              // the newcomer entered the running set
	Preempted []PreemptionEvent // victims preempted to make room, in preemption order
	Reason    string            // when !Admitted, why (e.g. "exceeds-capacity", "outranked")
}

// ReadmitTicket is one readmitted victim handed back to the caller to resume. A swap ticket
// carries the restored KV bytes (byte-identical to what was swapped out); a recompute ticket
// carries NeedsRecompute + the prompt to re-prefill.
type ReadmitTicket struct {
	TraceID        string
	Mode           PreemptionMode
	NeedsRecompute bool   // recompute mode: the caller must re-prefill PromptTokens
	PromptTokens   int    // tokens to re-prefill (recompute)
	KV             []byte // restored KV bytes (swap) — byte-identical to the swapped-out blob
}

// PreemptionStats is a snapshot of the preemptor's live gauges plus cumulative counters — the
// serving-metrics fragment a fleet router / autoscaler reads to see memory-pressure load.
type PreemptionStats struct {
	Running           int   // running sequences right now (gauge)
	UsedBlocks        int   // KV blocks held by the running set right now (gauge)
	SwappedOut        int   // sequences currently swapped/awaiting readmit (gauge)
	MaxPreemptRounds  int64 // oldest current victim's age in readmit rounds (starvation, gauge)
	Preemptions       int64 // cumulative preemptions (counter)
	SwapPreemptions   int64 // cumulative swap-mode preemptions (counter)
	RecomputeCount    int64 // cumulative recompute-mode preemptions (counter)
	SwapBytes         int64 // cumulative KV bytes swapped to host DRAM (counter)
	Readmitted        int64 // cumulative victims readmitted (counter)
	SwapRestoredBytes int64 // cumulative KV bytes restored from host on readmit (counter)
}

// KVPreemptor is the preemption + KV swap/recompute gate over the paged KV pool. The zero
// value is not usable — build one with NewKVPreemptor. It is safe for concurrent use (the
// gateway request path and the native loop both touch it).
type KVPreemptor struct {
	mu        sync.Mutex
	policy    PreemptionPolicy
	running   map[string]*runSeq // admitted, holding KV blocks, keyed by TraceID
	used      int                // Σ running[*].Blocks, maintained incrementally
	preempted []*victimSeq       // swapped-out / awaiting-recompute victims, awaiting readmit
	seqNo     int64              // monotone admit counter (orders "most-recently-admitted")
	round     int64              // monotone readmit-round counter (drives aging)
	stats     PreemptionStats    // cumulative counters (gauges derived in Stats)
}

// runSeq is one admitted sequence plus the order it was admitted, so VictimMostRecent has a
// total, clock-free ordering.
type runSeq struct {
	seq   KVSeq
	seqNo int64
}

// victimSeq is one preempted sequence: its identity + footprint, the round it was preempted
// (so aging can measure how long it has waited), and — for swap — the held KV bytes.
type victimSeq struct {
	seq          KVSeq
	mode         PreemptionMode
	preemptedRnd int64
	kv           []byte // swap: the held bytes, returned verbatim on readmit; nil for recompute
}

// NewKVPreemptor builds a preemptor under the given policy.
func NewKVPreemptor(p PreemptionPolicy) *KVPreemptor {
	return &KVPreemptor{policy: p, running: map[string]*runSeq{}}
}

// Admit offers a sequence to the paged KV pool. If its block footprint fits the free capacity
// it is admitted directly. If not, the preemptor selects victims in the policy's deterministic
// order and preempts them (swap or recompute) until the footprint fits, then admits — so the
// node makes forward progress instead of OOM-ing (acceptance bullets 1+2). It refuses without
// preempting anything when the footprint can NEVER fit (exceeds total capacity) or when no
// eligible victim outranks the newcomer (priority inversion guard) — the caller then queues or
// sheds the request via the admission gate.
//
// With MaxBlocks ≤ 0 (no paged pool — the contiguous-append cache) Admit ALWAYS admits and
// NEVER preempts: the structural gate that keeps preemption behind the paged allocator.
func (c *KVPreemptor) Admit(seq KVSeq) AdmitResult {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Gate: no block bound ⇒ no page granularity ⇒ no preemption (contiguous-cache path).
	if c.policy.MaxBlocks <= 0 {
		c.admitLocked(seq)
		return AdmitResult{Admitted: true}
	}
	if seq.Blocks > c.policy.MaxBlocks {
		return AdmitResult{Admitted: false, Reason: "exceeds-capacity"}
	}
	free := c.policy.MaxBlocks - c.used
	if seq.Blocks <= free {
		c.admitLocked(seq)
		return AdmitResult{Admitted: true}
	}

	// Need to reclaim (seq.Blocks - free) blocks. Choose victims deterministically; only
	// preempt if the eligible victims can free ENOUGH (no thrash: never preempt and still fail).
	need := seq.Blocks - free
	victims := c.eligibleVictimsLocked(seq)
	var reclaimable int
	for _, v := range victims {
		reclaimable += v.seq.Blocks
		if reclaimable >= need {
			break
		}
	}
	if reclaimable < need {
		return AdmitResult{Admitted: false, Reason: "outranked"}
	}

	var events []PreemptionEvent
	freed := 0
	for _, v := range victims {
		ev := c.preemptLocked(v)
		events = append(events, ev)
		freed += ev.Blocks
		if freed >= need {
			break
		}
	}
	c.admitLocked(seq)
	return AdmitResult{Admitted: true, Preempted: events}
}

// eligibleVictimsLocked returns the running sequences eligible to be preempted for the
// newcomer, ordered by the policy's victim rule (the order they will be preempted in). For
// VictimLowestPriority a victim must be strictly LESS important than the newcomer (higher
// Priority value) — the priority-inversion guard; VictimMostRecent has no such guard (any
// running sequence may be the newest). Caller holds c.mu.
func (c *KVPreemptor) eligibleVictimsLocked(newcomer KVSeq) []*runSeq {
	out := make([]*runSeq, 0, len(c.running))
	for _, r := range c.running {
		if r.seq.TraceID == newcomer.TraceID {
			continue
		}
		if c.policy.Victim == VictimLowestPriority && r.seq.Priority <= newcomer.Priority {
			continue // never evict an equal-or-higher-priority incumbent for this newcomer
		}
		out = append(out, r)
	}
	sort.SliceStable(out, func(i, j int) bool {
		switch c.policy.Victim {
		case VictimLowestPriority:
			if out[i].seq.Priority != out[j].seq.Priority {
				return out[i].seq.Priority > out[j].seq.Priority // least important first
			}
			// tie: newest first (least sunk cost), then TraceID for total determinism
			if out[i].seqNo != out[j].seqNo {
				return out[i].seqNo > out[j].seqNo
			}
			return out[i].seq.TraceID < out[j].seq.TraceID
		default: // VictimMostRecent
			if out[i].seqNo != out[j].seqNo {
				return out[i].seqNo > out[j].seqNo // most-recently-admitted first
			}
			return out[i].seq.TraceID < out[j].seq.TraceID
		}
	})
	return out
}

// preemptLocked preempts one victim: it frees the victim's KV blocks back to the pool and,
// per the mode, either holds its serialized KV bytes (swap) or drops them and retains the
// prompt to re-prefill (recompute), then moves it to the preempted queue. Returns the event
// for the caller and the counters. Caller holds c.mu.
func (c *KVPreemptor) preemptLocked(r *runSeq) PreemptionEvent {
	delete(c.running, r.seq.TraceID)
	c.used -= r.seq.Blocks
	if c.used < 0 {
		c.used = 0
	}
	v := &victimSeq{seq: r.seq, mode: c.policy.Mode, preemptedRnd: c.round}
	ev := PreemptionEvent{
		TraceID: r.seq.TraceID,
		Mode:    c.policy.Mode,
		Blocks:  r.seq.Blocks,
		Reason:  c.policy.Victim.String(),
	}
	c.stats.Preemptions++
	if c.policy.Mode == PreemptSwap {
		// Serialize-to-host: hold the bytes VERBATIM (a fresh copy so a later mutation of the
		// caller's slice cannot corrupt the swapped state) — restored byte-identical on readmit.
		v.kv = append([]byte(nil), r.seq.KV...)
		ev.Bytes = len(v.kv)
		c.stats.SwapPreemptions++
		c.stats.SwapBytes += int64(len(v.kv))
	} else {
		c.stats.RecomputeCount++
	}
	c.preempted = append(c.preempted, v)
	return ev
}

// Readmit runs ONE readmit round: it advances the aging clock and restores preempted victims
// — in starvation-guarded effective-priority order — into the running set while free KV-block
// capacity admits them, stopping at the first that does not fit (head-of-line, so an aged
// victim is never skipped by a smaller younger one). A swap ticket carries the restored KV
// bytes (byte-identical to what was swapped out); a recompute ticket carries NeedsRecompute +
// the prompt to re-prefill. Returns the readmitted tickets in readmit order. The host calls it
// each iteration and after a Complete frees blocks.
func (c *KVPreemptor) Readmit() []ReadmitTicket {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.round++
	if len(c.preempted) == 0 {
		return nil
	}
	sort.SliceStable(c.preempted, func(i, j int) bool {
		ei, ej := c.effectivePriorityLocked(c.preempted[i]), c.effectivePriorityLocked(c.preempted[j])
		if ei != ej {
			return ei < ej // lower effective priority served first
		}
		if c.preempted[i].preemptedRnd != c.preempted[j].preemptedRnd {
			return c.preempted[i].preemptedRnd < c.preempted[j].preemptedRnd // older victim first
		}
		return c.preempted[i].seq.TraceID < c.preempted[j].seq.TraceID
	})

	var tickets []ReadmitTicket
	kept := c.preempted[:0:0]
	blocked := false
	for _, v := range c.preempted {
		free := c.policy.MaxBlocks - c.used
		if blocked || v.seq.Blocks > free {
			blocked = true // head-of-line: do not let a smaller younger victim jump an aged one
			kept = append(kept, v)
			continue
		}
		c.admitLocked(v.seq)
		c.stats.Readmitted++
		t := ReadmitTicket{TraceID: v.seq.TraceID, Mode: v.mode}
		if v.mode == PreemptSwap {
			t.KV = v.kv // byte-identical to the swapped-out blob
			c.stats.SwapRestoredBytes += int64(len(v.kv))
		} else {
			t.NeedsRecompute = true
			t.PromptTokens = v.seq.PromptTokens
		}
		tickets = append(tickets, t)
	}
	c.preempted = append([]*victimSeq(nil), kept...)
	return tickets
}

// Complete releases a running sequence's KV blocks when its decode finishes (the loop's
// per-lane reclaim edge). Returns true if the trace was running. The freed blocks are taken by
// the next Readmit.
func (c *KVPreemptor) Complete(traceID string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	r, ok := c.running[traceID]
	if !ok {
		return false
	}
	delete(c.running, traceID)
	c.used -= r.seq.Blocks
	if c.used < 0 {
		c.used = 0
	}
	return true
}

// admitLocked moves a sequence into the running set and charges its KV blocks. Caller holds
// c.mu.
func (c *KVPreemptor) admitLocked(seq KVSeq) {
	c.seqNo++
	c.running[seq.TraceID] = &runSeq{seq: seq, seqNo: c.seqNo}
	c.used += seq.Blocks
}

// effectivePriorityLocked is a preempted victim's priority adjusted for how long it has waited
// to be readmitted: it improves (decreases) by one for every AgingRounds readmit rounds, so an
// aged victim climbs monotonically toward the head and is readmitted within a bounded number of
// rounds once blocks free — the no-starvation guarantee (mirrors admission.go). AgingRounds ≤ 0
// disables aging (raw Priority stands). Caller holds c.mu.
func (c *KVPreemptor) effectivePriorityLocked(v *victimSeq) int {
	if c.policy.AgingRounds <= 0 {
		return v.seq.Priority
	}
	waited := c.round - v.preemptedRnd
	if waited < 0 {
		waited = 0
	}
	return v.seq.Priority - int(waited/int64(c.policy.AgingRounds))
}

// Stats returns the live gauges plus cumulative counters.
func (c *KVPreemptor) Stats() PreemptionStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	st := c.stats
	st.Running = len(c.running)
	st.UsedBlocks = c.used
	st.SwappedOut = len(c.preempted)
	st.MaxPreemptRounds = c.maxPreemptRoundsLocked()
	return st
}

// maxPreemptRoundsLocked is the oldest current victim's age in readmit rounds (0 when none are
// preempted) — the starvation-visibility gauge. Caller holds c.mu.
func (c *KVPreemptor) maxPreemptRoundsLocked() int64 {
	var oldest int64
	for _, v := range c.preempted {
		if w := c.round - v.preemptedRnd; w > oldest {
			oldest = w
		}
	}
	return oldest
}

// WriteMetrics renders the preemptor's counters as Prometheus text into b — the serving-metrics
// fragment (acceptance bullet 4) a fleet router / autoscaler reads to see memory-pressure load.
// It reuses the gateway's metric writers and the schedMetricPrefix (admission.go) so the format
// matches the rest of /metrics; the preempt_ infix keeps it distinct from the admission family.
// The host folds this into renderMetrics once the native scheduler is on the live serve path.
func (c *KVPreemptor) WriteMetrics(b *strings.Builder) {
	st := c.Stats()
	p := schedMetricPrefix + "preempt_"
	writeHelpType(b, p+"running", "Sequences currently running (holding KV blocks) under the preemptor.", "gauge")
	fmt.Fprintf(b, "%srunning %d\n", p, st.Running)
	writeHelpType(b, p+"used_blocks", "KV blocks currently held by the running set.", "gauge")
	fmt.Fprintf(b, "%sused_blocks %d\n", p, st.UsedBlocks)
	writeHelpType(b, p+"max_blocks", "Configured paged-KV block capacity (0 = no paged pool; preemption disarmed).", "gauge")
	fmt.Fprintf(b, "%smax_blocks %d\n", p, c.policy.MaxBlocks)
	writeHelpType(b, p+"swapped_out", "Sequences currently swapped/awaiting readmit (preempted).", "gauge")
	fmt.Fprintf(b, "%sswapped_out %d\n", p, st.SwappedOut)
	writeHelpType(b, p+"victim_rule", "Active victim-selection rule (0=most-recent, 1=lowest-priority).", "gauge")
	fmt.Fprintf(b, "%svictim_rule %d\n", p, c.policy.Victim)
	writeHelpType(b, p+"max_wait_rounds", "Oldest current victim's age in readmit rounds (starvation visibility).", "gauge")
	fmt.Fprintf(b, "%smax_wait_rounds %d\n", p, st.MaxPreemptRounds)
	writeCounter(b, p+"total", "Sequences preempted under KV-block exhaustion.", st.Preemptions)
	writeCounter(b, p+"swap_total", "Preemptions taken via KV swap-to-host.", st.SwapPreemptions)
	writeCounter(b, p+"recompute_total", "Preemptions taken via drop-and-recompute.", st.RecomputeCount)
	writeCounter(b, p+"swap_bytes_total", "KV bytes swapped out to host DRAM.", st.SwapBytes)
	writeCounter(b, p+"readmitted_total", "Preempted sequences readmitted to the running set.", st.Readmitted)
	writeCounter(b, p+"swap_restored_bytes_total", "KV bytes restored from host DRAM on readmit.", st.SwapRestoredBytes)
}
