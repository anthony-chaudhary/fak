package kvmmu_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestCompactRefusesPoisonedSummary is the kernel-mediated-compaction gate witness
// (issue #522). Compaction swaps old spans for a summary; the load-bearing property
// is that a POISONED summary cannot launder itself back into the cache through that
// swap: the gate refuses it, the old spans stay, and the summary never enters.
func TestCompactRefusesPoisonedSummary(t *testing.T) {
	ctx := context.Background()
	m := model.NewSynthetic(synthCfg())
	sys := []int{1, 2, 3, 4}
	old1 := []int{10, 11, 12}
	old2 := []int{20, 21, 22, 23}
	summaryIDs := []int{40, 41} // the tokens a harness-produced summary would occupy

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sys)
	c.Append("o1", "read_notes", old1)
	c.Append("o2", "read_policy", old2)
	wantLen := c.CacheLen() // sys+o1+o2 — nothing evicted yet

	// A poisoned summary: the gate must Quarantine it, so Compact refuses the swap.
	v, swapped := c.Compact(ctx, []string{"o1", "o2"}, "summary", "compact", summaryIDs, []byte(poisonBody))
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("poisoned summary verdict = %v, want Quarantine", v.Kind)
	}
	if swapped != 0 {
		t.Fatalf("poisoned summary swapped %d spans, want 0 (the swap must be refused)", swapped)
	}
	if c.CacheLen() != wantLen {
		t.Fatalf("after refusing a poisoned summary, cache len = %d, want %d (old spans must stay)",
			c.CacheLen(), wantLen)
	}
	// The summary segment must NOT have been appended.
	for _, seg := range c.Segments() {
		if seg.ID == "summary" {
			t.Fatalf("a refused summary was appended as segment %+v", seg)
		}
	}
}

// TestCompactSwapsCleanSummary proves the happy path: a clean summary is admitted,
// the named old spans are evicted, and the summary is appended in their place. The
// post-compaction next-token distribution must equal a reference session that only
// ever saw sys+summary+query — true iff the old spans were removed AND the summary
// appended at the compacted position.
func TestCompactSwapsCleanSummary(t *testing.T) {
	ctx := context.Background()
	m := model.NewSynthetic(synthCfg())
	sys := []int{1, 2, 3, 4}
	old1 := []int{10, 11, 12}
	old2 := []int{20, 21, 22, 23}
	query := []int{30, 31}
	summaryIDs := []int{40, 41, 42}

	s := m.NewSession()
	c := kvmmu.NewWithGate(s, ctxmmu.New())
	c.Append("sys", "system", sys)
	c.Append("o1", "read_notes", old1)
	c.Append("o2", "read_policy", old2)

	v, swapped := c.Compact(ctx, []string{"o1", "o2"}, "summary", "compact", summaryIDs, []byte(benignBody))
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("clean summary verdict = %v, want Allow", v.Kind)
	}
	if swapped != 2 {
		t.Fatalf("clean summary swapped %d spans, want 2 (o1 and o2)", swapped)
	}
	// sys + summary only: o1 and o2 evicted, summary appended.
	wantLen := len(sys) + len(summaryIDs)
	if c.CacheLen() != wantLen {
		t.Fatalf("after clean compaction, cache len = %d, want %d (sys+summary)", c.CacheLen(), wantLen)
	}
	if c.Evicted() != 2 {
		t.Fatalf("Evicted() = %d, want 2", c.Evicted())
	}

	// Reference: a session that only ever prefilled sys+summary+query.
	lGot, _ := c.Append("usr", "user", query)
	lRef := m.NewSession().Prefill(cat(sys, summaryIDs, query))
	if d := maxAbsDiff(lGot, lRef); d != 0 {
		t.Fatalf("post-compaction distribution != reference sys+summary (max|Δ|=%.3e); the span swap is incoherent", d)
	}
}
