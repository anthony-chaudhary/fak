package gateway

// preemption_test.go — the issue-#31 acceptance witnesses for the KV preemption + swap /
// recompute gate (preemption.go). Each test names the acceptance bullet it proves:
//   - exhaustion triggers preempt + forward progress (bullets 1, 2)
//   - swap round-trip bit-exactness (bullet 2)
//   - recompute bit-exactness from the retained prompt (bullet 2)
//   - readmit-after-preempt restores the running set (bullets 1, 2)
//   - the gate stays behind the paged allocator (bullet 3)
//   - deterministic victim selection (bullet 5)
//   - the readmit starvation guard, both directions (bullet 5)
//   - the policy-driven swap/recompute choice is observable in the metrics schema (bullet 4)

import (
	"bytes"
	"strings"
	"testing"
)

// kvOf is a deterministic stand-in for the paged-prefill serializer: the same prompt always
// yields the same KV bytes (the StepBatch f32 determinism the recompute path relies on). It
// lets the recompute witness assert that re-deriving from the retained prompt reproduces the
// pre-preempt KV byte-for-byte, without running the model in a gateway-package unit test.
func kvOf(prompt int) []byte {
	out := make([]byte, prompt)
	for i := range out {
		out[i] = byte((prompt*131 + i*17) & 0xff)
	}
	return out
}

// TestPreemptExhaustionTriggersPreempt proves bullets 1+2: when the newcomer's KV footprint
// exceeds the free blocks, the preemptor evicts a victim (rather than failing) and admits the
// newcomer — the node makes forward progress instead of OOM-ing.
func TestPreemptExhaustionTriggersPreempt(t *testing.T) {
	c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimMostRecent, MaxBlocks: 10, AgingRounds: 1})
	if r := c.Admit(KVSeq{TraceID: "a", Priority: 0, Blocks: 6, KV: kvOf(6)}); !r.Admitted || len(r.Preempted) != 0 {
		t.Fatalf("first admit: got %+v, want admitted with no preemption", r)
	}
	// b needs 6 blocks; only 4 free → must preempt a (the only/newest victim) to fit.
	r := c.Admit(KVSeq{TraceID: "b", Priority: 0, Blocks: 6, KV: kvOf(60)})
	if !r.Admitted {
		t.Fatalf("exhausted admit: not admitted (%q) — preemption did not engage", r.Reason)
	}
	if len(r.Preempted) != 1 || r.Preempted[0].TraceID != "a" {
		t.Fatalf("expected to preempt victim a, got %+v", r.Preempted)
	}
	if st := c.Stats(); st.Running != 1 || st.UsedBlocks != 6 || st.SwappedOut != 1 || st.Preemptions != 1 {
		t.Fatalf("post-preempt stats = %+v, want running=1 used=6 swapped=1 preemptions=1", st)
	}
}

// TestPreemptSwapRoundTripBitExact proves bullet 2 (swap): a swapped-out sequence's KV is held
// VERBATIM and restored byte-identical on readmit — the bit-exact native-value guarantee. It
// also proves the held bytes are an independent copy (mutating the caller's slice after admit
// cannot corrupt the swapped state).
func TestPreemptSwapRoundTripBitExact(t *testing.T) {
	c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimMostRecent, MaxBlocks: 8, AgingRounds: 1})
	orig := kvOf(40)
	want := append([]byte(nil), orig...) // golden copy of the swapped-out KV
	c.Admit(KVSeq{TraceID: "victim", Priority: 5, Blocks: 5, KV: orig})
	// Force exhaustion: newcomer needs 5, only 3 free → victim's KV is serialized to host.
	r := c.Admit(KVSeq{TraceID: "new", Priority: 5, Blocks: 5, KV: kvOf(1)})
	if !r.Admitted || len(r.Preempted) != 1 || r.Preempted[0].Mode != PreemptSwap {
		t.Fatalf("swap preempt: got %+v", r)
	}
	if r.Preempted[0].Bytes != len(want) {
		t.Fatalf("swap event Bytes = %d, want %d", r.Preempted[0].Bytes, len(want))
	}
	// Mutate the caller's slice AFTER swap-out — the held bytes must be an independent copy,
	// not an alias, so a later overwrite of the source buffer cannot corrupt the swapped state.
	for i := range orig {
		orig[i] ^= 0xff
	}
	// Free the newcomer so a readmit round has capacity to restore the victim.
	if !c.Complete("new") {
		t.Fatal("Complete(new) = false, want the newcomer was running")
	}
	tickets := c.Readmit()
	if len(tickets) != 1 || tickets[0].TraceID != "victim" {
		t.Fatalf("readmit tickets = %+v, want one for victim", tickets)
	}
	if tickets[0].NeedsRecompute {
		t.Fatal("swap ticket marked NeedsRecompute")
	}
	if !bytes.Equal(tickets[0].KV, want) {
		t.Fatalf("restored KV not bit-identical:\n got %v\nwant %v", tickets[0].KV, want)
	}
}

// TestPreemptRecomputeBitExact proves bullet 2 (recompute): a recompute preemption DROPS the
// KV (holds no bytes) and retains the prompt, so re-prefilling the retained prompt through the
// deterministic kernel (kvOf) reproduces the pre-preempt KV byte-for-byte.
func TestPreemptRecomputeBitExact(t *testing.T) {
	c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptRecompute, Victim: VictimMostRecent, MaxBlocks: 8, AgingRounds: 1})
	const prompt = 33
	before := kvOf(prompt) // what a never-preempted sequence would hold
	c.Admit(KVSeq{TraceID: "victim", Priority: 5, Blocks: 5, PromptTokens: prompt, KV: kvOf(prompt)})
	r := c.Admit(KVSeq{TraceID: "new", Priority: 5, Blocks: 5, KV: kvOf(1)})
	if !r.Admitted || len(r.Preempted) != 1 || r.Preempted[0].Mode != PreemptRecompute {
		t.Fatalf("recompute preempt: got %+v", r)
	}
	if r.Preempted[0].Bytes != 0 {
		t.Fatalf("recompute held %d swap bytes, want 0 (KV is dropped, not swapped)", r.Preempted[0].Bytes)
	}
	c.Complete("new")
	tickets := c.Readmit()
	if len(tickets) != 1 || !tickets[0].NeedsRecompute || tickets[0].PromptTokens != prompt {
		t.Fatalf("recompute ticket = %+v, want NeedsRecompute with %d prompt tokens", tickets, prompt)
	}
	if tickets[0].KV != nil {
		t.Fatal("recompute ticket carried KV bytes, want nil (re-prefill, not restore)")
	}
	// The acceptance: re-prefill the retained prompt → bit-identical KV.
	if got := kvOf(tickets[0].PromptTokens); !bytes.Equal(got, before) {
		t.Fatalf("recomputed KV not bit-identical:\n got %v\nwant %v", got, before)
	}
}

// TestPreemptGatedBehindPagedPool proves bullet 3: with no block bound (MaxBlocks ≤ 0 — the
// contiguous-append cache that has no page to swap) Admit NEVER preempts and always admits.
// Preemption is structurally impossible without the paged allocator's block granularity.
func TestPreemptGatedBehindPagedPool(t *testing.T) {
	c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimMostRecent, MaxBlocks: 0, AgingRounds: 1})
	for _, id := range []string{"a", "b", "c"} {
		r := c.Admit(KVSeq{TraceID: id, Blocks: 1_000_000, KV: kvOf(4)})
		if !r.Admitted || len(r.Preempted) != 0 {
			t.Fatalf("admit %q on contiguous cache: got %+v, want admitted with NO preemption", id, r)
		}
	}
	if st := c.Stats(); st.Preemptions != 0 || st.SwappedOut != 0 {
		t.Fatalf("contiguous-cache path preempted: %+v, want zero preemptions", st)
	}
}

// TestPreemptVictimSelectionDeterministic proves bullet 5 (deterministic, documented victim
// choice). VictimMostRecent evicts the newest admitted; VictimLowestPriority evicts the least
// important and refuses to invert priority (a low-priority newcomer cannot evict a
// high-priority incumbent).
func TestPreemptVictimSelectionDeterministic(t *testing.T) {
	// Most-recent: a(0) then b(0) admitted; a newcomer that needs room evicts b (the newest).
	mr := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimMostRecent, MaxBlocks: 10, AgingRounds: 1})
	mr.Admit(KVSeq{TraceID: "a", Priority: 0, Blocks: 5, KV: kvOf(2)})
	mr.Admit(KVSeq{TraceID: "b", Priority: 0, Blocks: 5, KV: kvOf(2)})
	r := mr.Admit(KVSeq{TraceID: "c", Priority: 0, Blocks: 5, KV: kvOf(2)})
	if len(r.Preempted) != 1 || r.Preempted[0].TraceID != "b" || r.Preempted[0].Reason != "most-recent" {
		t.Fatalf("most-recent victim = %+v, want b with reason most-recent", r.Preempted)
	}

	// Lowest-priority: incumbents hi(prio 0) and lo(prio 9); a mid(prio 5) newcomer evicts lo
	// (it outranks lo) but never hi.
	lp := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimLowestPriority, MaxBlocks: 10, AgingRounds: 1})
	lp.Admit(KVSeq{TraceID: "hi", Priority: 0, Blocks: 5, KV: kvOf(2)})
	lp.Admit(KVSeq{TraceID: "lo", Priority: 9, Blocks: 5, KV: kvOf(2)})
	r = lp.Admit(KVSeq{TraceID: "mid", Priority: 5, Blocks: 5, KV: kvOf(2)})
	if len(r.Preempted) != 1 || r.Preempted[0].TraceID != "lo" {
		t.Fatalf("lowest-priority victim = %+v, want lo", r.Preempted)
	}

	// Priority inversion guard: a low-priority newcomer (prio 9) cannot evict the high-priority
	// incumbent (prio 0) — it is refused without preempting anything.
	inv := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimLowestPriority, MaxBlocks: 5, AgingRounds: 1})
	inv.Admit(KVSeq{TraceID: "hi", Priority: 0, Blocks: 5, KV: kvOf(2)})
	r = inv.Admit(KVSeq{TraceID: "low", Priority: 9, Blocks: 5, KV: kvOf(2)})
	if r.Admitted || r.Reason != "outranked" {
		t.Fatalf("priority inversion: got %+v, want refused with reason outranked", r)
	}
	if st := inv.Stats(); st.Preemptions != 0 {
		t.Fatalf("inversion guard preempted anyway: %+v", st)
	}
}

// TestPreemptNoStarvation proves bullet 5's starvation guard, both directions. One running
// slot (MaxBlocks==Blocks==1): a low-priority victim (prio 9) is preempted, then an endless
// stream of higher-priority sequences cycles through the slot — each fresh arrival preempts the
// incumbent rival, which rejoins the readmit queue ahead of the victim on RAW priority. With
// aging the victim still climbs to the head within a bounded number of rounds; with aging off,
// the flood starves it forever (the negative control).
func TestPreemptNoStarvation(t *testing.T) {
	run := func(aging int) bool {
		c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimMostRecent, MaxBlocks: 1, AgingRounds: aging})
		c.Admit(KVSeq{TraceID: "victim", Priority: 9, Blocks: 1, KV: kvOf(2)})
		c.Admit(KVSeq{TraceID: "rival", Priority: 0, Blocks: 1, KV: kvOf(2)}) // preempts the victim
		for i := 0; i < 64; i++ {
			// A fresh high-priority arrival preempts the running rival; the rival rejoins the
			// readmit queue, freshly preempted, ahead of the aging victim on raw priority.
			c.Admit(KVSeq{TraceID: "fresh" + itoa(uint64(i)), Priority: 0, Blocks: 1, KV: kvOf(2)})
			c.Complete("fresh" + itoa(uint64(i))) // free the single slot for exactly one readmit
			for _, tk := range c.Readmit() {
				if tk.TraceID == "victim" {
					return true
				}
			}
		}
		return false
	}
	if !run(1) {
		t.Fatal("with aging ON the victim was never readmitted — starvation guard failed")
	}
	if run(0) {
		t.Fatal("with aging OFF the victim was readmitted — the negative control should starve it")
	}
}

// TestPreemptMetricsObservable proves bullet 4: the swap/recompute choice and the preemption
// counters are observable in the serving-metrics schema. A swap preemption increments the swap
// + bytes counters; the recompute family stays zero, and the active victim rule is exported.
func TestPreemptMetricsObservable(t *testing.T) {
	c := NewKVPreemptor(PreemptionPolicy{Mode: PreemptSwap, Victim: VictimLowestPriority, MaxBlocks: 8, AgingRounds: 1})
	c.Admit(KVSeq{TraceID: "victim", Priority: 9, Blocks: 5, KV: kvOf(20)})
	c.Admit(KVSeq{TraceID: "new", Priority: 0, Blocks: 5, KV: kvOf(1)})
	var b strings.Builder
	c.WriteMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_sched_preempt_total 1",
		"fak_sched_preempt_swap_total 1",
		"fak_sched_preempt_recompute_total 0",
		"fak_sched_preempt_swap_bytes_total 20",
		"fak_sched_preempt_victim_rule 1", // VictimLowestPriority == 1
		"# TYPE fak_sched_preempt_total counter",
		"# TYPE fak_sched_preempt_running gauge",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics missing %q in:\n%s", want, out)
		}
	}
}
