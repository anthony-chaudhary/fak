package vcachegov

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestContinuationAffinityFeedsProviderAffinityHeader(t *testing.T) {
	tbl := session.NewTable()
	tbl.SetBudget("trace-root", session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 10,
	})

	drained := tbl.DebitUsage("trace-root", session.Usage{ContextTokens: 11})
	if drained.CacheAffinity.Action != session.CacheAffinityPreserve || drained.CacheAffinity.AffinityKey == "" {
		t.Fatalf("continuation cache affinity = %+v, want preserve with key", drained.CacheAffinity)
	}
	parentHeader := AffinityHeader("tenant-a", drained.CacheAffinity.AffinityKey, "us-east-1")
	if parentHeader == "" || len(parentHeader) > 32 {
		t.Fatalf("provider affinity header = %q, want bounded non-empty", parentHeader)
	}

	child := tbl.Recontinue("trace-root", drained.ContinuationID, session.Budget{
		TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 10,
	})
	childHeader := AffinityHeader("tenant-a", child.CacheAffinity.AffinityKey, "us-east-1")
	if childHeader != parentHeader {
		t.Fatalf("child provider affinity header = %q, want preserved %q", childHeader, parentHeader)
	}

	again := tbl.DebitUsage(child.TraceID, session.Usage{ContextTokens: 11})
	againHeader := AffinityHeader("tenant-a", again.CacheAffinity.AffinityKey, "us-east-1")
	if againHeader != parentHeader {
		t.Fatalf("next continuation provider affinity header = %q, want original %q", againHeader, parentHeader)
	}
}
