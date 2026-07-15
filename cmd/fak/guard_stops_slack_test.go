package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

func TestSummarizeGuardStopsSlackUsesRecentTrendThresholds(t *testing.T) {
	row := func(d guardStopDisposition, kind guardStopKind) string {
		return `{"schema":"` + guardStopRecordSchema + `","disposition":"` + string(d) + `","kind":"` + string(kind) + `"}`
	}
	content := strings.Join([]string{
		row(stopDispBlindGiveUp, stopKindStandDown), // lifetime-only outside recent window
		row(stopDispCleanCompletion, stopKindClean),
		row(stopDispFailOpenGaugeUnavailable, stopKindFailOpen),
	}, "\n") + "\n"
	tally := summarizeGuardStopsSlack(content, 2)
	if tally.Total != 3 || tally.StandDown != 1 || tally.FailOpen != 1 {
		t.Fatalf("lifetime tally wrong: %+v", tally)
	}
	if tally.RecentTotal != 2 || tally.RecentStandDown != 0 || tally.RecentFailOpen != 1 || tally.Status != "yellow" {
		t.Fatalf("recent trend wrong: %+v", tally)
	}

	red := summarizeGuardStopsSlack(row(stopDispFailOpenGaugeUnavailable, stopKindFailOpen)+"\n"+row(stopDispFailOpenGaugeUnavailable, stopKindFailOpen)+"\n", 20)
	if red.Status != "red" {
		t.Fatalf("two recent fail-opens should be red: %+v", red)
	}
}

func TestRunGuardStopsSlackDryRunDoesNotNeedCredentials(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "stops.jsonl")
	if err := os.WriteFile(ledger, []byte(`{"schema":"`+guardStopRecordSchema+`","disposition":"`+string(stopDispCleanCompletion)+`","kind":"clean"}`+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr strings.Builder
	if rc := runGuardStopsSlack(&stdout, &stderr, []string{"--ledger", ledger, "--dry-run"}); rc != 0 {
		t.Fatalf("dry-run rc=%d stderr=%q", rc, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Guard Stop health") || !strings.Contains(stdout.String(), "GREEN") {
		t.Fatalf("dry-run output=%q", stdout.String())
	}
}

func TestUpsertGuardStopsSlackPostsOnceThenUpdatesInPlace(t *testing.T) {
	dir := t.TempDir()
	if err := upsertGuardStopsSlack(dir, "C1", "first"); err != nil {
		t.Fatal(err)
	}
	if err := upsertGuardStopsSlack(dir, "C1", "pending duplicate"); err != nil {
		t.Fatal(err)
	}
	ob, err := slackoutbox.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 1 || snap.Rows[0].UpdateTS != "" {
		t.Fatalf("pending root duplicated: %+v", snap.Rows)
	}

	wire := newGuardStopsSlackWire()
	if _, err := ob.Drain(context.Background(), wire, slackoutbox.DrainOpts{}); err != nil {
		t.Fatal(err)
	}
	if err := upsertGuardStopsSlack(dir, "C1", "second"); err != nil {
		t.Fatal(err)
	}
	snap, err = ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 2 || snap.Rows[1].UpdateTS == "" || snap.Rows[1].Text != "second" {
		t.Fatalf("second refresh not update-in-place: %+v", snap.Rows)
	}
}

type guardStopsSlackWire struct{ next int }

func newGuardStopsSlackWire() *guardStopsSlackWire { return &guardStopsSlackWire{} }
func (w *guardStopsSlackWire) PostMessage(_ context.Context, _, _ string, _ []any, _ string) (string, error) {
	w.next++
	return fmt.Sprintf("%d.0", w.next), nil
}
func (w *guardStopsSlackWire) PostMessageIdem(ctx context.Context, channel, text string, blocks []any, threadTS, _ string) (string, error) {
	return w.PostMessage(ctx, channel, text, blocks, threadTS)
}
func (w *guardStopsSlackWire) UpdateMessage(_ context.Context, _, ts, _ string, _ []any) error {
	return nil
}
func (w *guardStopsSlackWire) History(context.Context, string, string, int) ([]slackwire.Message, error) {
	return nil, nil
}
func (w *guardStopsSlackWire) Replies(context.Context, string, string, int) ([]slackwire.Message, error) {
	return nil, nil
}
func (w *guardStopsSlackWire) DeleteMessage(context.Context, string, string) error { return nil }
