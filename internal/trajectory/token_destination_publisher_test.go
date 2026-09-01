package trajectory

import (
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestLiveRecorderPublishesTokenDestinationSnapshotOnBoundedCadence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "token-destination.jsonl")
	var auditCalls atomic.Int64
	audit := func() (AuditResult, error) {
		call := auditCalls.Add(1)
		if call == 1 {
			return AuditResult{}, errors.New("startup corpus unavailable")
		}
		return AuditResult{Summary: tokenDestinationSummaryForTest(call)}, nil
	}

	r := New()
	if err := r.startTokenDestinationPublisher(tokenDestinationPublisherConfig{
		Path: path, Interval: 20 * time.Millisecond, Audit: audit,
	}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(r.StopTokenDestinationPublisher)

	waitForPublisherStats(t, r, func(stats TokenDestinationPublisherStats) bool {
		return stats.Attempts == 1 && stats.Failures == 1
	})
	if _, err := os.Stat(path); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed startup must remain unavailable, stat err = %v", err)
	}

	// An ordinary live recorder event dirties the projection. The event path stays
	// non-blocking; the publisher performs the transcript audit on its own cadence.
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("publisher", "search_kb", "find policy"), Verdict: allowVerdict()})
	waitForPublisherStats(t, r, func(stats TokenDestinationPublisherStats) bool {
		return stats.Published == 1
	})

	summary, err := ReadAuditBaseline(path)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Transcripts != 2 || summary.Schema != AuditSchema || summary.Kind != "summary" {
		t.Fatalf("published summary = %+v", summary)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	firstModTime := info.ModTime()
	firstCalls := auditCalls.Load()
	time.Sleep(3 * 20 * time.Millisecond)
	if got := auditCalls.Load(); got != firstCalls {
		t.Fatalf("idle recorder refreshed snapshot %d extra time(s), making stale data appear fresh", got-firstCalls)
	}
	info, err = os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if !info.ModTime().Equal(firstModTime) {
		t.Fatalf("idle recorder advanced staleness timestamp: %s -> %s", firstModTime, info.ModTime())
	}

	// Bursty live events coalesce into one bounded refresh rather than rescanning
	// transcripts once per kernel event.
	for i := 0; i < 20; i++ {
		r.Emit(abi.Event{Kind: abi.EvDecide, Call: mkCall("publisher", "search_kb", "find policy"), Verdict: allowVerdict()})
	}
	waitForPublisherStats(t, r, func(stats TokenDestinationPublisherStats) bool {
		return stats.Published == 2
	})
	if got := auditCalls.Load(); got != firstCalls+1 {
		t.Fatalf("burst caused %d audits, want exactly one coalesced refresh", got-firstCalls)
	}
	stats := r.TokenDestinationPublisherStats()
	if stats.LastDurationNS <= 0 || stats.MaxDurationNS < stats.LastDurationNS || stats.LastPublishedAt.IsZero() || stats.LastError != "" {
		t.Fatalf("publisher overhead/freshness stats = %+v", stats)
	}
	if matches, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".token-destination-*.tmp")); err != nil || len(matches) != 0 {
		t.Fatalf("atomic publisher left temp files: matches=%v err=%v", matches, err)
	}
}

func TestTokenDestinationAtomicFailurePreservesLastGoodSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "token-destination.jsonl")
	first := tokenDestinationSummaryForTest(1)
	if err := writeTokenDestinationSummaryAtomic(path, first, os.Rename); err != nil {
		t.Fatal(err)
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	renameFailure := errors.New("rename refused")
	err = writeTokenDestinationSummaryAtomic(path, tokenDestinationSummaryForTest(2), func(string, string) error {
		return renameFailure
	})
	if !errors.Is(err, renameFailure) {
		t.Fatalf("rename error = %v, want %v", err, renameFailure)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("failed atomic publish changed last good snapshot\n got: %s\nwant: %s", got, want)
	}
}

func tokenDestinationSummaryForTest(transcripts int64) AuditSummaryRow {
	return AuditSummaryRow{
		Schema: AuditSchema, Kind: "summary", Transcripts: int(transcripts),
		DistributionUnit:       AuditDistributionUnit,
		DistributionProvenance: "deterministic model-visible content UTF-8 bytes; not billed tokens",
		Distribution:           []AuditDistributionRow{{Name: "tool_result", Bytes: transcripts, Share: 1}},
	}
}

func waitForPublisherStats(t *testing.T, r *Recorder, ready func(TokenDestinationPublisherStats) bool) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if ready(r.TokenDestinationPublisherStats()) {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("publisher did not reach expected state: %+v", r.TokenDestinationPublisherStats())
}
