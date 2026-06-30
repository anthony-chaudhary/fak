package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// twoTrackReportNow is a fixed clock for the recompute comparison. The Track-2
// economics (NET, cumulative, break-even) do not depend on it — it only stamps
// GeneratedAt — so the re-fold reproduces regardless of the value.
func twoTrackReportNow() time.Time { return time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC) }

// writeTwoLedgers drops a WITNESSED Track-1 ledger and an OBSERVED-$ Track-2
// ledger into dir and returns their paths. The Track-2 fixture starts NET
// negative (cold writes) and crosses break-even the next week.
func writeTwoLedgers(t *testing.T, dir string) (track1, track2 string) {
	t.Helper()
	track1 = filepath.Join(dir, "cache-value.jsonl")
	t1 := `{"date":"2026-06-15","session_type":"guard","turns":10,"prompt_tokens":1000,"reused_tokens":600}
{"date":"2026-06-22","session_type":"guard","turns":10,"prompt_tokens":1000,"reused_tokens":800}
`
	if err := os.WriteFile(track1, []byte(t1), 0o600); err != nil {
		t.Fatal(err)
	}
	track2 = filepath.Join(dir, "cache-savings.jsonl")
	t2 := `{"schema":"fak-cache-savings-ledger/1","date":"2026-06-15","session_type":"guard","input_tokens":2000,"cache_creation_tokens":8000,"output_tokens":500,"rebate_usd":0.5,"write_premium_usd":2.0,"spend_usd":1.0,"compaction_saved_usd":0.25}
{"schema":"fak-cache-savings-ledger/1","date":"2026-06-22","session_type":"guard","input_tokens":1000,"cache_read_tokens":9000,"output_tokens":500,"rebate_usd":5.0,"write_premium_usd":0.1,"spend_usd":0.5,"compaction_saved_usd":0.4}
`
	if err := os.WriteFile(track2, []byte(t2), 0o600); err != nil {
		t.Fatal(err)
	}
	return track1, track2
}

// TestCachevalueReportPrintsBothTracksAndNet is the #1304 witness at the CLI seam:
// `fak cachevalue report --since` prints Track 1 (WITNESSED) and Track 2 (OBSERVED
// $) side by side plus a NET line, with the running total crossing break-even shown
// explicitly.
func TestCachevalueReportPrintsBothTracksAndNet(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)

	var out, errb bytes.Buffer
	code := runCachevalueReport(&out, &errb, []string{
		"--ledger", track1, "--savings-ledger", track2, "--since", "2026-06-01",
	})
	if code != 0 {
		t.Fatalf("report exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{
		"Track 1", "WITNESSED",
		"Track 2", "OBSERVED $",
		"net$", "break-even",
		"marginal-over-tuned-warm-KV", // the #1066 fence on Track 1
		"never blended",               // the provenance fence on the P&L
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("report missing %q:\n%s", want, got)
		}
	}
}

// TestCachevalueReportJSONReproducesFold is the recompute witness at the CLI seam:
// `--json` emits a TwoTrackReport that re-folds byte-for-byte from the same two
// ledgers (report == fold(ledgers)), and carries the break-even crossing.
func TestCachevalueReportJSONReproducesFold(t *testing.T) {
	dir := t.TempDir()
	track1, track2 := writeTwoLedgers(t, dir)

	var out, errb bytes.Buffer
	code := runCachevalueReport(&out, &errb, []string{
		"--ledger", track1, "--savings-ledger", track2, "--json",
	})
	if code != 0 {
		t.Fatalf("report --json exit = %d, stderr=%s", code, errb.String())
	}
	var rep cachevaluereport.TwoTrackReport
	if err := json.Unmarshal(out.Bytes(), &rep); err != nil {
		t.Fatalf("report --json is not a TwoTrackReport: %v\n%s", err, out.String())
	}
	if rep.Verdict != "MEASURED" {
		t.Fatalf("both tracks have evidence; want MEASURED, got %q", rep.Verdict)
	}
	if len(rep.Track2) != 2 {
		t.Fatalf("want 2 Track-2 buckets, got %d", len(rep.Track2))
	}
	if rep.Track2[0].NetUSD >= 0 || rep.Track2[0].BrokeEven {
		t.Fatalf("week 1 should be NET-negative and below break-even: %+v", rep.Track2[0])
	}
	if !rep.Track2[1].BrokeEven || !rep.BrokeEven {
		t.Fatalf("week 2 should cross break-even: %+v", rep.Track2[1])
	}

	// Re-fold the SAME two ledgers and assert the CLI JSON is exactly the pure
	// fold output, ignoring only the live GeneratedAt clock stamp.
	refold := cachevaluereport.FoldTwoTrack(
		cachevalueledger.ReadLedgerFile(track1),
		cachevaluereport.ReadSavingsLedgerFile(track2),
		twoTrackReportNow(),
	)
	refold.GeneratedAt = rep.GeneratedAt
	refold.Track1.GeneratedAt = rep.Track1.GeneratedAt
	want, err := json.Marshal(refold)
	if err != nil {
		t.Fatal(err)
	}
	got, err := json.Marshal(rep)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Fatalf("CLI report does not reproduce from fold(two ledgers):\n got=%s\nwant=%s", got, want)
	}
}

// TestCachevalueReportMissingTrack2IsHonest checks an absent Track-2 ledger folds
// to the honest "rung B not appending yet" report (Track 1 only), not a failure.
func TestCachevalueReportMissingTrack2IsHonest(t *testing.T) {
	dir := t.TempDir()
	track1, _ := writeTwoLedgers(t, dir)

	var out, errb bytes.Buffer
	code := runCachevalueReport(&out, &errb, []string{
		"--ledger", track1, "--savings-ledger", filepath.Join(dir, "absent.jsonl"),
	})
	if code != 0 {
		t.Fatalf("report with absent Track-2 should still render, exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "no OBSERVED-$ rows yet") {
		t.Fatalf("absent Track-2 should say so:\n%s", out.String())
	}
}

// TestCachevalueReportRejectsBadSince checks --since must be a valid date.
func TestCachevalueReportRejectsBadSince(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runCachevalueReport(&out, &errb, []string{"--since", "last-tuesday"}); code != 2 {
		t.Fatalf("bad --since should exit 2, got %d", code)
	}
}
