package agent

import (
	"encoding/json"
	"testing"
)

// compact_receipt_induced_test.go — #2785: the suffix-burst cost (induced cache_creation) a
// compaction fire buys is RECORDED per fire, not computed-and-discarded, and reconciles against
// the provider's own cache_creation on the turns those fires rewrote.
//
// Before this landed, headBurstEconomics computed invalidatedSuffixTokens purely as a gate input:
// the gate consumed it and threw it away, so every fire's one-time burst premium was invisible to
// the ledger and the netting read the shed saving as free. Each test below fails against that
// prior behaviour, where CompactOutcome/CompactReceipt carried no induced figure at all.

// messagesElements decodes a request body's messages[] into the element slice the compactor works
// on, so a test can re-derive a span (e.g. the invalidated suffix) independently of the value the
// compactor reported.
func messagesElements(t *testing.T, raw []byte) []json.RawMessage {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	elems, _, ok := decodeArrayElements(raw, obj["messages"])
	if !ok {
		t.Fatalf("decodeArrayElements failed")
	}
	return elems
}

// TestCompactFireRecordsInducedCacheCreation is the #2785 witness on the live path: a warm
// head-anchored fire — the exact shape whose burst the gate prices and refuses without a horizon
// (TestCompactHeadAnchorRespectsBurstEconomics) — now REPORTS the burst it bought, and the figure
// is the same invalidated-suffix span the gate priced, not an independent guess.
//
// keepStart is recoverable from the outcome on this path: head mode makes pfxEnd -1, so the
// compactor's `dropped = keepStart - (pfxEnd+1)` is exactly keepStart. That lets the test
// re-derive the expected span from the ORIGINAL body and assert equality rather than mere
// positivity — a nonzero-but-wrong figure would still fail.
func TestCompactFireRecordsInducedCacheCreation(t *testing.T) {
	raw := headOrderedBody(t, 120, 2) // recent breakpoint at index 117 — a real suffix to invalidate

	opts := CompactOptions{Budget: 1200, Anchor: CompactAnchorHead, TotalTurns: 1000, CurrentTurn: 1}
	_, outcome := CompactAnthropicHistoryWithOptions(raw, opts)
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("head anchor with a paying horizon must FIRE, got reason=%q", outcome.Reason)
	}

	elems := messagesElements(t, raw)
	want := invalidatedSuffixSpanTokens(elems, outcome.Dropped)
	if want <= 0 {
		t.Fatalf("fixture must have a surviving downstream breakpoint to invalidate; got span=%d (keepStart=%d)", want, outcome.Dropped)
	}
	if outcome.InducedCacheCreationTokens != want {
		t.Fatalf("fire must record the burst it bought: induced=%d, want the invalidated suffix span %d",
			outcome.InducedCacheCreationTokens, want)
	}

	// The receipt is the per-fire ledger row, so the figure must survive onto it — that is the
	// "persisted per fire" half of the acceptance.
	r := NewCompactReceipt(outcome)
	if !r.Fired || r.InducedCacheCreationTokens != want {
		t.Fatalf("receipt must carry the fire's induced creation: fired=%v induced=%d, want %d",
			r.Fired, r.InducedCacheCreationTokens, want)
	}
	if SumReceiptInducedCreation([]CompactReceipt{r}) != want {
		t.Fatalf("SumReceiptInducedCreation = %d, want %d", SumReceiptInducedCreation([]CompactReceipt{r}), want)
	}
}

// TestCompactColdFireInducesNoCreation pins the EARNED zero: on an observed-cold fire the suffix's
// cache has already expired, so the provider cold-writes it this turn with or without the drop.
// The fire causes no MARGINAL creation, and debiting one would charge compaction for a cost it did
// not cause. This is the same zeroing the burst gate applies, so the recorded cost and the gate
// can never disagree about a cold fire.
func TestCompactColdFireInducesNoCreation(t *testing.T) {
	raw := recentOnlyBreakpointBody(t, 120, 2)

	_, outcome := CompactAnthropicHistoryWithOptions(raw, CompactOptions{Budget: 1200, Anchor: CompactAnchorHead, ColdCache: true})
	if outcome.Reason != CompactReasonNone {
		t.Fatalf("observed-cold head anchor must fire horizon-free, got reason=%q", outcome.Reason)
	}
	if outcome.InducedCacheCreationTokens != 0 {
		t.Fatalf("an observed-cold fire induces no marginal creation, got induced=%d", outcome.InducedCacheCreationTokens)
	}
	// The span itself is nonzero — the zero above is the ColdCache attribution decision, not an
	// absent breakpoint. Without this the test would pass on a fixture with nothing to invalidate.
	if span := invalidatedSuffixSpanTokens(messagesElements(t, raw), outcome.Dropped); span <= 0 {
		t.Fatalf("fixture must have a nonzero suffix span for the cold zero to be meaningful, got %d", span)
	}
}

// TestCompactBailRecordsNoInducedCreation pins the other earned zero: a bail moves no bytes, so it
// invalidates nothing. Uses the warm no-horizon burst bail — the case that would be most tempting
// to debit, since the gate DID price a nonzero burst before refusing.
func TestCompactBailRecordsNoInducedCreation(t *testing.T) {
	raw := headOrderedBody(t, 120, 2)

	_, outcome := CompactAnthropicHistoryWithOptions(raw, CompactOptions{Budget: 1200, Anchor: CompactAnchorHead})
	if outcome.Reason != CompactReasonBurstUnprofitable {
		t.Fatalf("unknown horizon must bail burst_unprofitable, got reason=%q", outcome.Reason)
	}
	if outcome.InducedCacheCreationTokens != 0 {
		t.Fatalf("a bail rewrote nothing and can induce nothing, got induced=%d", outcome.InducedCacheCreationTokens)
	}
	if r := NewCompactReceipt(outcome); r.Fired || r.InducedCacheCreationTokens != 0 {
		t.Fatalf("bail receipt must carry no induced creation, got fired=%v induced=%d", r.Fired, r.InducedCacheCreationTokens)
	}
}

// TestReconcileInducedCreationAgainstProviderDelta is the acceptance's reconciliation half: the
// per-fire induced figure placed beside the provider's OWN cache_creation on the same fire turns.
// It covers the containment relation, the conservative clamp, and the two ways a fire is excluded
// from the reading rather than silently folded in as a zero.
func TestReconcileInducedCreationAgainstProviderDelta(t *testing.T) {
	fire := func(induced int, observedCreation uint64) CompactReceipt {
		return CompactReceipt{Fired: true, ShedTokens: 1000, InducedCacheCreationTokens: induced,
			ObservedCacheCreationTokens: observedCreation}
	}

	t.Run("contained fires reconcile", func(t *testing.T) {
		// Two stamped fires: 400+600 induced against 1000+2000 the provider actually created on
		// those turns. Containment holds, so the debit is the raw induced sum.
		got := ReconcileInducedCreation([]CompactReceipt{fire(400, 1000), fire(600, 2000)})
		if got.Fires != 2 || got.ReconciledFires != 2 {
			t.Fatalf("fires=%d reconciled=%d, want 2/2", got.Fires, got.ReconciledFires)
		}
		if got.InducedTokens != 1000 || got.ObservedCreationTokens != 3000 {
			t.Fatalf("induced=%d observed=%d, want 1000/3000", got.InducedTokens, got.ObservedCreationTokens)
		}
		if !got.WithinObserved || !got.Reconciled() {
			t.Fatalf("1000 <= 3000 must reconcile, got within=%v reconciled=%v", got.WithinObserved, got.Reconciled())
		}
		if got.DebitTokens != 1000 {
			t.Fatalf("debit=%d, want the raw induced sum 1000", got.DebitTokens)
		}
		if got.AttributedFraction < 0.3332 || got.AttributedFraction > 0.3334 {
			t.Fatalf("attributed fraction=%v, want ~1/3", got.AttributedFraction)
		}
	})

	t.Run("overshoot clamps and refuses to reconcile", func(t *testing.T) {
		// The estimator ran hot: fak claims it induced more creation than the provider billed on
		// the whole turn. The raw sum must NOT be debited (it would invent a cost), and the pass
		// must not report itself reconciled.
		got := ReconcileInducedCreation([]CompactReceipt{fire(5000, 1200)})
		if got.WithinObserved || got.Reconciled() {
			t.Fatalf("5000 > 1200 must fail containment, got within=%v reconciled=%v", got.WithinObserved, got.Reconciled())
		}
		if got.DebitTokens != 1200 {
			t.Fatalf("debit=%d, want the observed creation 1200 (clamped, never over-debited)", got.DebitTokens)
		}
	})

	t.Run("unstamped fires are excluded not zero-folded", func(t *testing.T) {
		// A byte-level fire with no provider usage stamped. Folding it in as a 0/0 pair would make
		// the fraction and the containment verdict claim coverage the row does not have.
		got := ReconcileInducedCreation([]CompactReceipt{fire(700, 0)})
		if got.Fires != 1 || got.ReconciledFires != 0 {
			t.Fatalf("fires=%d reconciled=%d, want 1/0", got.Fires, got.ReconciledFires)
		}
		if got.InducedTokens != 0 || got.DebitTokens != 0 || got.AttributedFraction != 0 {
			t.Fatalf("an unjoined fire must contribute nothing, got %+v", got)
		}
		if got.Reconciled() || got.WithinObserved {
			t.Fatalf("nothing reconciled must never read as a pass, got %+v", got)
		}
	})

	t.Run("bails are not fires", func(t *testing.T) {
		bail := CompactReceipt{Reason: CompactReasonBurstUnprofitable, ObservedCacheCreationTokens: 9000}
		got := ReconcileInducedCreation([]CompactReceipt{bail, fire(400, 1000)})
		if got.Fires != 1 || got.ReconciledFires != 1 {
			t.Fatalf("fires=%d reconciled=%d, want 1/1 (the bail is not a fire)", got.Fires, got.ReconciledFires)
		}
		if got.ObservedCreationTokens != 1000 {
			t.Fatalf("observed=%d, want 1000 — a bail's turn creation must not enter the fire-turn delta", got.ObservedCreationTokens)
		}
	})

	t.Run("empty is an abstain", func(t *testing.T) {
		if got := ReconcileInducedCreation(nil); got.Reconciled() || got.DebitTokens != 0 {
			t.Fatalf("no receipts must abstain, got %+v", got)
		}
	})
}

// TestInducedCreationSurvivesReceiptRoundTrip proves the figure is on the durable per-fire row, not
// just in memory: it serializes under the issue's `induced_cache_creation_tokens` key — the same
// name the usage-ledger counter (gatewayusageledger.Counters) already reserves for it — and comes
// back intact.
func TestInducedCreationSurvivesReceiptRoundTrip(t *testing.T) {
	b, err := json.Marshal(CompactReceipt{Fired: true, ShedTokens: 4000, InducedCacheCreationTokens: 950})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var probe map[string]any
	if err := json.Unmarshal(b, &probe); err != nil {
		t.Fatalf("unmarshal probe: %v", err)
	}
	if got, ok := probe["induced_cache_creation_tokens"]; !ok || got != float64(950) {
		t.Fatalf("receipt must persist induced_cache_creation_tokens=950, got %v (present=%v) in %s", got, ok, b)
	}
	var back CompactReceipt
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.InducedCacheCreationTokens != 950 {
		t.Fatalf("round-trip induced = %d, want 950", back.InducedCacheCreationTokens)
	}
}
