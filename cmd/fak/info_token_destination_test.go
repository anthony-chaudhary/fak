package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/guardvars"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func populatedTokenDestinationSnapshot() *infoTokenDestinationSnapshot {
	return &infoTokenDestinationSnapshot{
		Schema:       infoTokenDestinationSnapshotSchema,
		Availability: guardvars.AvailabilityObserved,
		RecordedAt:   "2026-08-27T18:04:05Z",
		Revision:     "turn-12",
		Window:       "last 12 recorded turns",
		Summary: trajectory.AuditSummaryRow{
			Schema:                 trajectory.AuditSchema,
			Kind:                   "summary",
			DistributionUnit:       trajectory.AuditDistributionUnit,
			DistributionProvenance: "deterministic transcript UTF-8 bytes; not provider-billed per-block tokens",
			Distribution: []trajectory.AuditDistributionRow{
				{Name: "tool_result", Bytes: 600, Share: .60},
				{Name: "assistant", Bytes: 250, Share: .25},
				{Name: "reasoning", Bytes: 150, Share: .15},
			},
			ToolDistribution: []trajectory.AuditDistributionRow{
				{Name: "exec_command", Bytes: 420, Share: .70, Calls: 4},
				{Name: "read_file", Bytes: 180, Share: .30, Calls: 2},
			},
		},
	}
}

func tokenDestinationTrendRows(snapshot *infoTokenDestinationSnapshot, width int) []string {
	tr := newGuardInfoTrend(8)
	tr.prefillPerTurn = []float64{100, 120}
	tr.decodePerTurn = []float64{20, 30}
	tr.costPerTurn = []float64{120, 150}
	v := provenVisualVars()
	v.TokenDestination = snapshot
	return guardInfoTrendsPanelRows(newGuardInfoPanelCtx(v, tr, width), guardPanelFull)
}

// TestGuardInfoTokenDestinationPopulatedCapturedRender proves the real recorded summary is
// rendered directly beside the #9408 marginal-cost row. The basis line prevents the utf8-byte
// attribution from being mistaken for provider-billed per-block tokens.
func TestGuardInfoTokenDestinationPopulatedCapturedRender(t *testing.T) {
	got := strings.Join(tokenDestinationTrendRows(populatedTokenDestinationSnapshot(), 160), "\n")
	for _, exact := range []string{
		" usage    ▁█  150 tok/reply · avg 135 · trend ↑ +25%",
		" tokens→ · tool_result 60% · assistant 25% · reasoning 15% · top-tool exec_command 70%",
		" basis   model-visible attributed UTF-8 bytes (not billed tokens) · last 12 recorded turns · recorded 2026-08-27T18:04:05Z",
	} {
		if !strings.Contains(got, exact) {
			t.Fatalf("populated token-destination capture missing %q:\n%s", exact, got)
		}
	}
}

// TestGuardInfoTokenDestinationStaleCapturedRender proves a retained distribution may remain
// inspectable only when STALE leads the percentages and the recorder's reason is explicit.
func TestGuardInfoTokenDestinationStaleCapturedRender(t *testing.T) {
	snapshot := populatedTokenDestinationSnapshot()
	snapshot.Availability = guardvars.AvailabilityStale
	snapshot.Reason = "recorder has not advanced for 3m"
	got := strings.Join(tokenDestinationTrendRows(snapshot, 220), "\n")
	for _, exact := range []string{
		" tokens→ STALE · tool_result 60% · assistant 25% · reasoning 15% · top-tool exec_command 70%",
		" basis   model-visible attributed UTF-8 bytes (not billed tokens) · last 12 recorded turns · recorded 2026-08-27T18:04:05Z · STALE: recorder has not advanced for 3m",
	} {
		if !strings.Contains(got, exact) {
			t.Fatalf("stale token-destination capture missing %q:\n%s", exact, got)
		}
	}
}

// TestGuardInfoTokenDestinationUnavailableCapturedRender pins omission to an explicit unavailable
// row. No category, percentage, or top tool may be invented when /debug/vars has no snapshot.
func TestGuardInfoTokenDestinationUnavailableCapturedRender(t *testing.T) {
	got := strings.Join(infoTokenDestinationRows(nil, 120), "\n")
	want := " tokens→ unavailable · live recorded snapshot not available"
	if !strings.Contains(got, want) {
		t.Fatalf("unavailable token-destination capture missing %q:\n%s", want, got)
	}
	for _, fabricated := range []string{"tool_result", "top-tool", "%"} {
		if strings.Contains(got, fabricated) {
			t.Fatalf("unavailable token-destination capture fabricated %q:\n%s", fabricated, got)
		}
	}
}

// TestGuardInfoTokenDestinationNarrowCapturedRender proves the destination rows remain one-row,
// rune-safe, and honesty-first in the 20% pane. The canonical compact mix is visibly truncated;
// it never wraps or hides stale/unavailable state past the right edge.
func TestGuardInfoTokenDestinationNarrowCapturedRender(t *testing.T) {
	const width = 36
	rows := infoTokenDestinationRows(populatedTokenDestinationSnapshot(), width)
	want := []string{
		" tokens→ · tool_result 60% · assist…",
		" top-tool exec_command 70% of attri…",
		" basis   model-visible attributed U…",
	}
	if !reflect.DeepEqual(rows, want) {
		t.Fatalf("narrow token-destination capture:\n got: %#v\nwant: %#v", rows, want)
	}
	for _, row := range rows {
		if got := dispWidthTUI(row); got > width {
			t.Fatalf("narrow row escaped %d cells (%d): %q", width, got, row)
		}
		if !utf8.ValidString(row) {
			t.Fatalf("narrow row split a UTF-8 rune: %q", row)
		}
	}
}

// TestGuardInfoTokenDestinationDebugVarsDataSeam proves this is a real /debug/vars decode path,
// not a renderer-only fixture: the recorded wrapper survives the same JSON projection info polls.
func TestGuardInfoTokenDestinationDebugVarsDataSeam(t *testing.T) {
	want := populatedTokenDestinationSnapshot()
	payload, err := json.Marshal(map[string]any{"token_destination": want})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/debug/vars" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(payload)
	}))
	defer srv.Close()

	got, err := readGuardInfoVars(&claudeMacDebugClient{base: srv.URL, hc: srv.Client()})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.TokenDestination, want) {
		t.Fatalf("token destination did not survive /debug/vars projection:\n got: %#v\nwant: %#v", got.TokenDestination, want)
	}
}

// TestGuardInfoTokenDestinationRecordedFileSource proves fak info can consume the summary row
// emitted by the existing trajectory audit without importing unexported parser code. The file's
// witnessed mtime drives the observed-to-stale transition; refreshing is never inferred.
func TestGuardInfoTokenDestinationRecordedFileSource(t *testing.T) {
	summary := populatedTokenDestinationSnapshot().Summary
	summary.Transcripts = 12
	payload, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "trajectory-audit.jsonl")
	if err := os.WriteFile(path, append(payload, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	recordedAt := time.Date(2026, 8, 27, 18, 4, 5, 0, time.UTC)
	if err := os.Chtimes(path, recordedAt, recordedAt); err != nil {
		t.Fatal(err)
	}

	now := recordedAt.Add(30 * time.Second)
	source := &infoTokenDestinationSource{
		Path:   path,
		MaxAge: time.Minute,
		now:    func() time.Time { return now },
	}
	var fresh guardInfoVars
	source.decorate(&fresh)
	if problem := infoTokenDestinationProblem(fresh.TokenDestination); problem != "" {
		t.Fatalf("fresh recorded artifact unavailable: %s (%+v)", problem, fresh.TokenDestination)
	}
	if fresh.TokenDestination.Availability != guardvars.AvailabilityObserved || fresh.TokenDestination.Window != "12 recorded sessions" {
		t.Fatalf("fresh recorded artifact = %+v", fresh.TokenDestination)
	}

	now = recordedAt.Add(2 * time.Minute)
	var stale guardInfoVars
	source.decorate(&stale)
	if stale.TokenDestination.Availability != guardvars.AvailabilityStale || !strings.Contains(stale.TokenDestination.Reason, "age 2m0s exceeds 1m0s") {
		t.Fatalf("stale recorded artifact = %+v", stale.TokenDestination)
	}
}
