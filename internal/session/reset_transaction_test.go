package session

import "testing"

func TestRecontinueResetTransactionRowLivesOnChild(t *testing.T) {
	tbl := NewTable()
	tbl.SetBudget("parent", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10})
	drained := tbl.DebitUsage("parent", Usage{ContextTokens: 11})
	if drained.ContinuationID == "" {
		t.Fatalf("drain did not mint continuation: %+v", drained)
	}
	if v := tbl.Decide("parent"); v.State.Run != Stopped {
		t.Fatalf("expected stopped parent, got %+v", v)
	}

	inputTx := ResetTransaction{
		SeedDigest:       "seed-digest",
		Contributors:     []string{"warm_prefix", "task_distill"},
		OmittedSpans:     []ResetOmittedSpan{{Index: 2, Role: "assistant", Digest: "span-digest", Reason: "ephemeral_or_turn_scoped"}},
		WarmPrefixDigest: "warm-digest",
	}
	fresh := tbl.RecontinueWithTransaction("parent", drained.ContinuationID, Budget{
		TurnsLeft:         Unbounded,
		TokensLeft:        Unbounded,
		ContextTokensLeft: 50,
	}, inputTx)
	tx := fresh.ResetTransaction
	if tx.Schema != ResetTransactionSchema {
		t.Fatalf("schema = %q, want %q", tx.Schema, ResetTransactionSchema)
	}
	if tx.OldTrace != "parent" || tx.NewTrace != drained.ContinuationID {
		t.Fatalf("transaction lineage = %q -> %q, want parent -> continuation", tx.OldTrace, tx.NewTrace)
	}
	if tx.BudgetRearm.ContextTokensLeft != 50 || tx.BudgetRearm.ContextTokensCap != 50 {
		t.Fatalf("budget rearm = %+v, want context budget 50/50", tx.BudgetRearm)
	}
	if tx.SeedDigest != inputTx.SeedDigest || tx.WarmPrefixDigest != inputTx.WarmPrefixDigest {
		t.Fatalf("seed fields = %+v, want carried from caller %+v", tx, inputTx)
	}
	if len(tx.Contributors) != 2 || len(tx.OmittedSpans) != 1 {
		t.Fatalf("transaction proof fields = %+v, want contributors and omitted span", tx)
	}
	if parent := tbl.Get("parent"); !parent.ResetTransaction.IsZero() {
		t.Fatalf("parent should keep the drain record only, got reset transaction %+v", parent.ResetTransaction)
	}
}

func TestDescriptorRoundTripPreservesResetTransaction(t *testing.T) {
	tx := ResetTransaction{
		Schema:       ResetTransactionSchema,
		OldTrace:     "old",
		NewTrace:     "new",
		SeedDigest:   "seed",
		Contributors: []string{"task_distill"},
		BudgetRearm:  ResetBudgetRearm{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 80, ContextTokensCap: 80},
	}
	st := State{TraceID: "new", Run: Running, Budget: Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded}, ResetTransaction: tx, Rev: 7}
	d := descriptorFromState(st)
	restored := d.RestoredState()
	if restored.ResetTransaction.Schema != ResetTransactionSchema ||
		restored.ResetTransaction.OldTrace != "old" ||
		restored.ResetTransaction.SeedDigest != "seed" ||
		len(restored.ResetTransaction.Contributors) != 1 {
		t.Fatalf("restored reset transaction = %+v, want descriptor to preserve %+v", restored.ResetTransaction, tx)
	}
}
