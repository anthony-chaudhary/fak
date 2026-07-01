package sessionreset

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestResetTransactionCapturesSeedDigestContributorsAndOmissions(t *testing.T) {
	in := Input{
		Trace:          "trace-parent",
		Messages:       sampleTranscript(),
		FreshBudgetTok: 75,
	}
	seed := BuildSeed(in)
	tx := BuildResetTransaction(in, "trace-child", seed)

	if tx.Schema != session.ResetTransactionSchema {
		t.Fatalf("schema = %q, want %q", tx.Schema, session.ResetTransactionSchema)
	}
	if tx.OldTrace != "trace-parent" || tx.NewTrace != "trace-child" {
		t.Fatalf("lineage = %q -> %q, want trace-parent -> trace-child", tx.OldTrace, tx.NewTrace)
	}
	if tx.SeedDigest == "" || tx.SeedDigest != DigestSeed(seed) {
		t.Fatalf("seed digest = %q, want recomputable digest %q", tx.SeedDigest, DigestSeed(seed))
	}
	if tx.BudgetRearm.ContextTokensLeft != 75 || tx.BudgetRearm.ContextTokensCap != 75 {
		t.Fatalf("budget rearm = %+v, want context budget 75/75", tx.BudgetRearm)
	}
	if tx.WarmPrefixDigest == "" {
		t.Fatalf("warm prefix digest missing from transaction: %+v", tx)
	}
	if !hasContributor(tx.Contributors, "warm_prefix") || !hasContributor(tx.Contributors, "durability_facts") {
		t.Fatalf("contributors = %v, want fired seed contributors", tx.Contributors)
	}
	if len(tx.OmittedSpans) == 0 {
		t.Fatal("expected at least one omitted span handle")
	}
	for _, span := range tx.OmittedSpans {
		if span.Digest == "" {
			t.Fatalf("omitted span missing digest: %+v", span)
		}
		if span.Reason == "" {
			t.Fatalf("omitted span missing reason: %+v", span)
		}
	}

	again := BuildResetTransaction(in, "trace-child", BuildSeed(in))
	if !reflect.DeepEqual(tx, again) {
		t.Fatalf("transaction is not replayable:\n first=%+v\n again=%+v", tx, again)
	}
	tampered := seed
	tampered.Recap += "\nmutation"
	if DigestSeed(tampered) == tx.SeedDigest {
		t.Fatal("seed digest did not change after recap mutation")
	}
}

func hasContributor(contributors []string, want string) bool {
	for _, got := range contributors {
		if got == want {
			return true
		}
	}
	return false
}
