package orchestration

import (
	"testing"
	"time"
)

func TestUltracodeEnvelopeThreeChildOverrunFailsClosed(t *testing.T) {
	started := time.Date(2026, 8, 22, 4, 27, 54, 0, time.UTC)
	receipt, err := NewUltracodeEnvelopeReceipt(65_536, 3*time.Minute, started, []string{"worker-1", "worker-2", "worker-3"})
	if err != nil {
		t.Fatal(err)
	}
	var reserved int64
	for _, child := range receipt.Children {
		reserved += child.ReservedTokens
	}
	if reserved != 65_536 {
		t.Fatalf("child reservations = %d, want one conserved 65536-token parent envelope", reserved)
	}

	got, err := FoldUltracodeEnvelopeReceipt(receipt, []UltracodeChildUsage{
		{ChildID: "worker-1", ProviderTokens: 59_630, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-2", ProviderTokens: 59_479, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-3", ProviderTokens: 118_615, Authority: UltracodeBudgetAuthorityProvider},
	}, started.Add(34*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if got.ConsumedTokens != 237_724 || got.RemainingTokens != 0 {
		t.Fatalf("aggregate token accounting = consumed %d remaining %d, want 237724/0", got.ConsumedTokens, got.RemainingTokens)
	}
	if got.CoveredChildren != 3 || got.TotalChildren != 3 || got.Authority != UltracodeBudgetAuthorityProvider {
		t.Fatalf("coverage/authority = %d/%d %q", got.CoveredChildren, got.TotalChildren, got.Authority)
	}
	if !got.Overrun || !got.TokenOverrun || got.WallOverrun || got.Admitted || got.Reason != UltracodeBudgetReasonTokenOverrun {
		t.Fatalf("over-budget fleet was admitted: %+v", got)
	}
}

func TestUltracodeEnvelopeIncompleteAndWallExpiredFailClosed(t *testing.T) {
	started := time.Unix(1_700_000_000, 0).UTC()
	base, err := NewUltracodeEnvelopeReceipt(65_536, 3*time.Minute, started, []string{"worker-1", "worker-2", "worker-3"})
	if err != nil {
		t.Fatal(err)
	}

	incomplete, err := FoldUltracodeEnvelopeReceipt(base, []UltracodeChildUsage{
		{ChildID: "worker-1", ProviderTokens: 10_000, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-2", ProviderTokens: 10_000, Authority: UltracodeBudgetAuthorityProvider},
	}, started.Add(time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if incomplete.Admitted || incomplete.Complete || incomplete.CoveredChildren != 2 || incomplete.Authority != UltracodeBudgetAuthorityIncomplete || incomplete.Reason != UltracodeBudgetReasonIncomplete {
		t.Fatalf("incomplete receipt did not fail closed: %+v", incomplete)
	}

	expired, err := FoldUltracodeEnvelopeReceipt(base, []UltracodeChildUsage{
		{ChildID: "worker-1", ProviderTokens: 10_000, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-2", ProviderTokens: 10_000, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-3", ProviderTokens: 10_000, Authority: UltracodeBudgetAuthorityProvider},
	}, started.Add(3*time.Minute))
	if err != nil {
		t.Fatal(err)
	}
	if expired.Admitted || !expired.Overrun || expired.TokenOverrun || !expired.WallOverrun || expired.Reason != UltracodeBudgetReasonWallOverrun {
		t.Fatalf("expired wall envelope did not fail closed: %+v", expired)
	}
}

func TestUltracodeEnvelopeRejectsTamperedSummary(t *testing.T) {
	started := time.Unix(1_700_000_000, 0).UTC()
	receipt, err := NewUltracodeEnvelopeReceipt(12_000, time.Minute, started, []string{"worker-1", "worker-2", "worker-3"})
	if err != nil {
		t.Fatal(err)
	}
	receipt, err = FoldUltracodeEnvelopeReceipt(receipt, []UltracodeChildUsage{
		{ChildID: "worker-1", ProviderTokens: 1_000, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-2", ProviderTokens: 1_000, Authority: UltracodeBudgetAuthorityProvider},
		{ChildID: "worker-3", ProviderTokens: 1_000, Authority: UltracodeBudgetAuthorityProvider},
	}, started.Add(10*time.Second))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateUltracodeEnvelopeReceipt(receipt); err != nil {
		t.Fatalf("valid receipt rejected: %v", err)
	}
	receipt.ConsumedTokens = 2_999
	if err := ValidateUltracodeEnvelopeReceipt(receipt); err == nil {
		t.Fatal("tampered aggregate was accepted")
	}
}
