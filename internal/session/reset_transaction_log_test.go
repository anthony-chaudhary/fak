package session

import (
	"reflect"
	"strings"
	"testing"
)

func TestResetTransactionLogAppendEntriesLatest(t *testing.T) {
	var log ResetTransactionLog
	if _, ok := log.Latest(); ok {
		t.Fatalf("empty log must report no latest entry")
	}

	tx1 := NewResetTransaction("trace-a", "trace-b", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 50})
	tx1.SeedDigest = "digest-1"
	got1 := log.Append(tx1)
	if !reflect.DeepEqual(got1, tx1) {
		t.Fatalf("Append returned %+v, want %+v", got1, tx1)
	}

	tx2 := NewResetTransaction("trace-b", "trace-c", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 40})
	tx2.SeedDigest = "digest-2"
	tx2.WarmPrefixDigest = "warm-2"
	tx2.OmittedSpans = []ResetOmittedSpan{{Index: 0, Role: "user", Digest: "span-1", Reason: "ephemeral_or_turn_scoped"}}
	log.Append(tx2)

	entries := log.Entries()
	if len(entries) != 2 {
		t.Fatalf("Entries() = %d rows, want 2", len(entries))
	}
	if entries[0].NewTrace != "trace-b" || entries[1].NewTrace != "trace-c" {
		t.Fatalf("entries out of order: %+v", entries)
	}

	latest, ok := log.Latest()
	if !ok || latest.NewTrace != "trace-c" {
		t.Fatalf("Latest() = %+v, ok=%v, want trace-c", latest, ok)
	}

	// Entries() must be a defensive copy: mutating it must not affect the log.
	entries[0].NewTrace = "tampered"
	if again, _ := log.Latest(); again.NewTrace != "trace-c" {
		t.Fatalf("mutating Entries() copy leaked into the log: latest = %+v", again)
	}
}

func TestResetTransactionLogReplayDetectsChainAndMalformedRows(t *testing.T) {
	var log ResetTransactionLog
	log.Append(NewResetTransaction("trace-a", "trace-b", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 50}))
	log.Append(NewResetTransaction("trace-b", "trace-c", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 40}))

	verdicts, allMatch := log.Replay()
	if !allMatch {
		t.Fatalf("well-formed chained log must replay clean: %+v", verdicts)
	}
	if len(verdicts) != 2 {
		t.Fatalf("Replay() = %d verdicts, want 2", len(verdicts))
	}
	if !verdicts[0].ChainLinked {
		t.Fatalf("first entry is always chain-linked (nothing precedes it): %+v", verdicts[0])
	}
	if !verdicts[1].ChainLinked {
		t.Fatalf("second entry's OldTrace should link to first entry's NewTrace: %+v", verdicts[1])
	}

	// A malformed row (missing schema token) must be flagged as diverged.
	log.Append(ResetTransaction{OldTrace: "trace-c", NewTrace: "trace-d"})
	verdicts, allMatch = log.Replay()
	if allMatch {
		t.Fatalf("log with a malformed row must not replay clean: %+v", verdicts)
	}
	if !verdicts[2].Diverged || verdicts[2].DivergeNote == "" {
		t.Fatalf("malformed row verdict = %+v, want diverged with a note", verdicts[2])
	}

	// A new, unrelated lineage is expected to show ChainLinked=false without being a divergence.
	var branched ResetTransactionLog
	branched.Append(NewResetTransaction("trace-x", "trace-y", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10}))
	branched.Append(NewResetTransaction("trace-unrelated", "trace-z", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 10}))
	verdicts, allMatch = branched.Replay()
	if !allMatch {
		t.Fatalf("an unlinked-but-well-formed second lineage must still replay clean: %+v", verdicts)
	}
	if verdicts[1].ChainLinked {
		t.Fatalf("unrelated lineage must not report ChainLinked=true: %+v", verdicts[1])
	}
}

func TestResetTransactionLogSummaryAndExplain(t *testing.T) {
	var log ResetTransactionLog
	tx1 := NewResetTransaction("trace-a", "trace-b", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 50})
	tx1.SeedDigest = "0123456789abcdef"
	tx1.OmittedSpans = []ResetOmittedSpan{{Index: 0, Digest: "d1"}, {Index: 1, Digest: "d2"}}
	log.Append(tx1)

	tx2 := NewResetTransaction("trace-b", "trace-c", Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded, ContextTokensLeft: 40})
	tx2.SeedDigest = "fedcba9876543210"
	tx2.WarmPrefixDigest = "warm"
	log.Append(tx2)

	sum := log.Summary()
	if sum.Total != 2 || sum.WithSeedDigest != 2 || sum.WithWarmPrefix != 1 || sum.OmittedSpans != 2 {
		t.Fatalf("Summary() = %+v, want total=2 seed=2 warm=1 omitted=2", sum)
	}

	explain := log.Explain()
	if explain == "" {
		t.Fatal("Explain() must render a non-empty report")
	}
	for _, want := range []string{"trace-a -> trace-b", "trace-b -> trace-c", "2 reset(s)"} {
		if !strings.Contains(explain, want) {
			t.Fatalf("Explain() = %q, missing expected substring %q", explain, want)
		}
	}
}
