package agent

import (
	"bytes"
	"testing"
)

// TestCompactSolvencyOverrideFiresRefusedBurst is the acceptance witness for the context-solvency
// override: the SAME body and horizon the burst economics REFUSE (CompactReasonBurstUnprofitable,
// identity — see TestCompactHeadAnchorRespectsBurstEconomics) must FIRE once the caller reports a
// resident occupancy at or above the armed floor, and the fire must be labeled SolvencyForced so it
// is never booked as a profitable one.
//
// Why the override exists: CacheBurstPaysBack prices a fire in cache dollars and has no term for
// running out of window, so it refuses hardest exactly where refusing is most expensive. Measured
// over 3191 real served turns in .dispatch-runs, the fire rate INVERTED against occupancy (33.4% at
// 96-110k, 33.9% at 110-125k, 24.7% at 125-140k, 14.3% at 140-155k, 3.4% at 155-170k, 0.0% above
// 170k) and of the traces that ever fired, 100% never fired again — a one-way latch whose median
// 9-turn tail carried resident a further +33.8k into the window.
func TestCompactSolvencyOverrideFiresRefusedBurst(t *testing.T) {
	raw := headOrderedBody(t, 120, 2)
	// The refused baseline: unknown horizon ⇒ a real one-time penalty ⇒ no repayment ⇒ identity.
	refused := CompactOptions{Budget: 1200, Anchor: CompactAnchorHead}
	if out, outcome := CompactAnthropicHistoryWithOptions(raw, refused); !bytes.Equal(out, raw) || outcome.Reason != CompactReasonBurstUnprofitable {
		t.Fatalf("baseline must bail burst_unprofitable (identity); got changed=%v reason=%q",
			!bytes.Equal(out, raw), outcome.Reason)
	}

	// Same refusal, but the trace has climbed to the solvency floor: fire anyway.
	armed := refused
	armed.SolvencyFloorTokens = 100000
	armed.ResidentTokens = 143000
	out, outcome := CompactAnthropicHistoryWithOptions(raw, armed)
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("solvency floor must override the refused burst and FIRE, got reason=%q (%+v)", outcome.Reason, outcome)
	}
	if !outcome.SolvencyForced {
		t.Fatalf("a fire the economics refused must be labeled SolvencyForced, got %+v", outcome)
	}
	if bytes.Equal(out, raw) || len(out) >= len(raw) {
		t.Fatalf("a forced fire must still shrink the body, got %d (in %d)", len(out), len(raw))
	}
	if outcome.Dropped <= 0 || outcome.ShedTokens <= 0 {
		t.Fatalf("a forced fire must report real dropped/shed work, got %+v", outcome)
	}
	// The override buys occupancy — it must NOT buy it by breaking any of the byte-level
	// guarantees. The stable head stays verbatim, the body still decodes, roles still alternate.
	start := messagesArrayContentStart(t, raw)
	if start > len(out) || !bytes.Equal(raw[:start], out[:start]) {
		t.Fatalf("forced fire changed the stable head bytes [0,%d) — the override must never relax a cache-safety guarantee", start)
	}
	if _, err := DecodeAnthropicMessagesRequest(out); err != nil {
		t.Fatalf("forced-fire body failed to decode: %v", err)
	}
	assertAlternation(t, out)
}

// TestCompactSolvencyOverrideStaysDisarmed proves the override is a FLOOR, not a switch, and that
// it needs BOTH halves. Every case here must reproduce the pure-economics bail byte-for-byte, so an
// existing caller (or an ablation row) that supplies neither value is provably unchanged.
func TestCompactSolvencyOverrideStaysDisarmed(t *testing.T) {
	raw := headOrderedBody(t, 120, 2)
	base := CompactOptions{Budget: 1200, Anchor: CompactAnchorHead}

	for _, tc := range []struct {
		name     string
		resident int
		floor    int
	}{
		{"neither supplied (the untouched default path)", 0, 0},
		{"occupancy known but no floor armed", 143000, 0},
		{"floor armed but occupancy unknown", 0, 100000},
		{"occupancy below the floor", 99999, 100000},
	} {
		opts := base
		opts.ResidentTokens, opts.SolvencyFloorTokens = tc.resident, tc.floor
		out, outcome := CompactAnthropicHistoryWithOptions(raw, opts)
		if !bytes.Equal(out, raw) || outcome.Reason != CompactReasonBurstUnprofitable {
			t.Fatalf("%s: must stay on pure economics (identity + burst_unprofitable); got changed=%v reason=%q",
				tc.name, !bytes.Equal(out, raw), outcome.Reason)
		}
		if outcome.SolvencyForced {
			t.Fatalf("%s: a bail must never be labeled SolvencyForced", tc.name)
		}
	}

	// Exactly AT the floor fires — the gate is >=, so the documented threshold is inclusive.
	at := base
	at.ResidentTokens, at.SolvencyFloorTokens = 100000, 100000
	if _, outcome := CompactAnthropicHistoryWithOptions(raw, at); outcome.Reason != CompactReasonNone || !outcome.SolvencyForced {
		t.Fatalf("resident exactly AT the floor must force a fire, got reason=%q forced=%v", outcome.Reason, outcome.SolvencyForced)
	}
}

// TestCompactSolvencyOverrideNeverMislabelsProfitableFire keeps the accounting honest in the other
// direction: a fire the economics APPROVE must not be labeled SolvencyForced just because the floor
// happens to be armed and cleared. Otherwise the cache-value ledger would write off genuine cache
// wins as survival spending.
func TestCompactSolvencyOverrideNeverMislabelsProfitableFire(t *testing.T) {
	raw := headOrderedBody(t, 120, 2)
	opts := CompactOptions{
		Budget: 1200, Anchor: CompactAnchorHead,
		TotalTurns: 1000, CurrentTurn: 1, // a horizon the burst comfortably repays
		ResidentTokens: 143000, SolvencyFloorTokens: 100000,
	}
	_, outcome := CompactAnthropicHistoryWithOptions(raw, opts)
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("a paying horizon must still fire, got reason=%q", outcome.Reason)
	}
	if outcome.SolvencyForced {
		t.Fatalf("an economics-APPROVED fire must not be labeled SolvencyForced, got %+v", outcome)
	}
}
