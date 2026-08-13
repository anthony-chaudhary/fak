package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// inkernel_turntax_test.go — the #1538 witness that the in-kernel turn path makes a cache
// decision BY DEFAULT and records why. It drives real turns through generateReused over the
// synthetic model (the same harness inkernel_reuse_test.go's parity arm uses) and reads the
// ledger back off the planner. Nothing here stubs the decision: the entries asserted on are
// produced by the live decode seam calling ctxplan.PlanTurnTax.

// TestInKernelTurnTaxRecordsReuseAndColdPrefill is the core arm: with the prefix cache ON, a
// first turn has nothing admitted and must book COLD PREFILL, and a second turn extending that
// exact prefix must book REUSE — the adaptive behavior the issue asks for, observed per turn.
func TestInKernelTurnTaxRecordsReuseAndColdPrefill(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	base := synthIDs(64, 40, 7)

	// Turn 1: the always-cold first prefill — the tree is empty, so nothing can match.
	decode(p, base, 4)
	// Turn 2: the same 40-token prefix plus a divergent tail. The radix index matches the
	// shared prefix, so only the tail needs computing.
	turn2 := append(append([]int(nil), base...), synthIDs(64, 12, 99)...)
	_, matched := decode(p, turn2, 4)
	if matched <= 0 {
		t.Fatalf("harness precondition failed: turn 2 reused %d prefix tokens, want > 0", matched)
	}

	ents := p.TurnTaxDecisions()
	if len(ents) != 2 {
		t.Fatalf("recorded %d decisions, want exactly one per turn (2)", len(ents))
	}
	if got := ents[0].Decision.Strategy; got != ctxplan.TurnTaxColdPrefill {
		t.Errorf("turn 1 strategy = %q, want %q (nothing admitted yet)", got, ctxplan.TurnTaxColdPrefill)
	}
	if got := ents[1].Decision.Strategy; got != ctxplan.TurnTaxReuse {
		t.Errorf("turn 2 strategy = %q, want %q (prefix matched and was served)", got, ctxplan.TurnTaxReuse)
	}
	// "…and records WHY": the reason is populated, not an empty string, on every turn.
	for i, e := range ents {
		if strings.TrimSpace(e.Decision.Reason) == "" {
			t.Errorf("turn %d recorded strategy %q with an empty reason", i+1, e.Decision.Strategy)
		}
	}
	// The recorded signals are the real ones this turn saw, not placeholders.
	if got := ents[1].Signals.PromptTokens; got != len(turn2) {
		t.Errorf("turn 2 recorded PromptTokens = %d, want %d", got, len(turn2))
	}
	if !ents[1].Signals.PrefixTrusted {
		t.Error("turn 2 served a matched prefix but recorded PrefixTrusted = false")
	}
	// Reuse must be the CHEAPER choice, or picking it was not a cost decision at all.
	if reuseTax, cold := ents[1].Decision.Tax(), ents[1].Signals.PromptTokens; reuseTax >= cold {
		t.Errorf("turn 2 reuse tax %d not below the cold-prefill tax %d", reuseTax, cold)
	}
}

// TestInKernelTurnTaxRecordsWithReuseDisabled is the cold-path arm the issue requires to stay
// explicit: with the prefix cache OFF every turn genuinely IS a cold prefill, and the ledger
// still records one decision per turn. "By default" has to mean recorded even when the cache
// contributes nothing — otherwise the ledger would only exist on the happy path.
func TestInKernelTurnTaxRecordsWithReuseDisabled(t *testing.T) {
	p := reusePlanner(false, false, tinyCfg())
	ids := synthIDs(64, 32, 11)
	const turns = 3
	for i := 0; i < turns; i++ {
		decode(p, ids, 3)
	}

	ents := p.TurnTaxDecisions()
	if len(ents) != turns {
		t.Fatalf("recorded %d decisions over %d turns, want one per turn", len(ents), turns)
	}
	for i, e := range ents {
		if e.Decision.Strategy != ctxplan.TurnTaxColdPrefill {
			t.Errorf("turn %d strategy = %q, want %q (reuse disabled)", i+1, e.Decision.Strategy, ctxplan.TurnTaxColdPrefill)
		}
		if e.Signals.MatchedPrefix != 0 {
			t.Errorf("turn %d recorded MatchedPrefix = %d with reuse disabled, want 0", i+1, e.Signals.MatchedPrefix)
		}
		if e.Decision.Saved() != 0 {
			t.Errorf("turn %d claims %d tokens saved on a cold prefill, want 0", i+1, e.Decision.Saved())
		}
	}
}

// TestInKernelTurnTaxLedgerIsDeterministicAndSummarized checks the two properties an operator
// readout depends on: every recorded decision re-derives from its OWN stored signals (so the
// ledger is auditable rather than merely asserted), and the summary folds to the same turns.
func TestInKernelTurnTaxLedgerIsDeterministicAndSummarized(t *testing.T) {
	p := reusePlanner(true, false, tinyCfg())
	base := synthIDs(64, 24, 3)
	decode(p, base, 2)
	decode(p, append(append([]int(nil), base...), synthIDs(64, 8, 5)...), 2)

	ents := p.TurnTaxDecisions()
	var log ctxplan.TurnTaxLog
	for _, e := range ents {
		log.Append(e.Signals)
	}
	if diverged, ok := log.Replay(); !ok {
		t.Errorf("replaying the live ledger diverged at entries %v; decisions are not reproducible from their signals", diverged)
	}

	sum := p.TurnTaxSummary()
	if total := sum.Reuse + sum.Query + sum.ColdPrefill; total != len(ents) {
		t.Errorf("summary counts total %d, want %d (one per recorded turn)", total, len(ents))
	}
	if sum.Saved != sum.PromptTokens-sum.Tax {
		t.Errorf("summary Saved = %d, want PromptTokens-Tax = %d", sum.Saved, sum.PromptTokens-sum.Tax)
	}
	// Every strategy the ledger reports is a member of the closed vocabulary.
	for i, e := range ents {
		if !ctxplan.ValidTurnTaxStrategy(e.Decision.Strategy) {
			t.Errorf("entry %d recorded out-of-vocabulary strategy %q", i, e.Decision.Strategy)
		}
	}
	// The operator readout names each turn's strategy.
	explain := p.ExplainTurnTax()
	if !strings.Contains(explain, string(ctxplan.TurnTaxColdPrefill)) {
		t.Errorf("ExplainTurnTax did not name the cold-prefill turn:\n%s", explain)
	}
}
