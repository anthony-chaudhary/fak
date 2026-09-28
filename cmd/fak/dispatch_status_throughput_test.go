package main

// dispatch_status_throughput_test.go — the decision table that separates "idle" from
// "dead" (gateshare/fak/lesson-1).
//
// The whole point of the throughput headline is that ONE liveness number could not
// tell those two states apart: during 2026-07-28..08-03, 350 of 382 spawned workers
// died at argv parse and the card read "0 live worker(s)" for six days. So the case
// is pinned here as a table rather than left to observation — if a future refactor
// collapses any two rows back together, this fails.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDispatchThroughputSeparatesIdleFromDead(t *testing.T) {
	// spawned-and-finished, nothing landed, nothing still running == the DOA shape.
	spawned := &dispatchStatusSpawnHealth{Runs: 382, DOA: 350, Status: "alarm"}
	prog := func(commits int) *dispatchProgressSummary {
		return &dispatchProgressSummary{Available: true, Commits: commits}
	}

	cases := []struct {
		name    string
		live    int
		spawn   *dispatchStatusSpawnHealth
		prog    *dispatchProgressSummary
		want    string
		wantSub string
	}{
		{
			// THE REGRESSION THIS BLOCK EXISTS FOR. Pre-fix this rendered identically
			// to the idle row: "0 live worker(s)".
			name: "doa shape is an outage not an idle fleet", live: 0, spawn: spawned, prog: prog(0),
			want:    dispatchLandingOutageSuspect,
			wantSub: "382 spawn(s) finished, nothing landed, 0 still live",
		},
		{
			name: "genuinely idle", live: 0, spawn: nil, prog: prog(0),
			want: dispatchLandingIdle, wantSub: "genuinely idle",
		},
		{
			name: "work landed", live: 3, spawn: spawned, prog: prog(7),
			want: dispatchLandingLanding, wantSub: "7 commit(s) landed",
		},
		{
			name: "in flight, none due yet", live: 4, spawn: nil, prog: prog(0),
			want: dispatchLandingWorking, wantSub: "4 live worker(s)",
		},
		{
			// A fold that cannot read commits must NOT render as zero — zero is a claim
			// about the fleet, and a failed fold is not evidence of one.
			name: "unreadable commit count is unknown not zero", live: 0, spawn: spawned,
			prog:    &dispatchProgressSummary{Error: "not a git repository"},
			want:    dispatchLandingUnknown,
			wantSub: "commit count unavailable",
		},
		{
			name: "nil progress is unknown", live: 0, spawn: spawned, prog: nil,
			want: dispatchLandingUnknown, wantSub: "commit count unavailable",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := dispatchThroughputFold(tc.live, tc.spawn, tc.prog)
			if got.Landing != tc.want {
				t.Fatalf("landing = %q, want %q (reason=%q)", got.Landing, tc.want, got.Reason)
			}
			if !strings.Contains(got.Reason, tc.wantSub) {
				t.Fatalf("reason = %q, want substring %q", got.Reason, tc.wantSub)
			}
		})
	}
}

// The DOA row must point at the LAUNCHER, since a 91.6% dead-on-arrival rate is an
// argv/binary skew or a launch preflight refusal, not a selection-quality question.
func TestDispatchThroughputOutageNamesTheLaunchCause(t *testing.T) {
	spawn := &dispatchStatusSpawnHealth{Runs: 382, DOA: 350, Status: "alarm"}
	got := dispatchThroughputFold(0, spawn, &dispatchProgressSummary{Available: true})
	if got.Landing != dispatchLandingOutageSuspect {
		t.Fatalf("landing = %q, want outage_suspect", got.Landing)
	}
	if !strings.Contains(got.NextAction, "350 of 382") {
		t.Fatalf("next_action = %q, want the DOA count to be surfaced", got.NextAction)
	}
	if !strings.Contains(got.NextAction, "argv") {
		t.Fatalf("next_action = %q, want it to name the argv-vs-PATH skew", got.NextAction)
	}
}

// The rendered line must label liveness as secondary, so a reader who reads only the
// headline is reading the number that can fail.
func TestDispatchThroughputLineDemotesLiveness(t *testing.T) {
	line := dispatchThroughputLine(&dispatchThroughput{
		Landing: dispatchLandingOutageSuspect, Commits: 0, LiveWorkers: 0,
		Reason: "382 spawn(s) finished",
	})
	if !strings.Contains(line, "LANDING [outage_suspect]") {
		t.Fatalf("line must lead with the landing verdict, got %q", line)
	}
	if !strings.Contains(line, "live workers 0 (secondary)") {
		t.Fatalf("line must mark liveness secondary, got %q", line)
	}
	if dispatchThroughputLine(nil) != "" {
		t.Fatal("nil throughput must render as an empty line, not a panic or a zero")
	}
}

// The card's FIRST line must be the landing block, in both renderers. This is the
// actual contract: line order is the fix, not merely exposing a new field.
func TestDispatchStatusCardLeadsWithLanding(t *testing.T) {
	snap := dispatchStatusSnapshot{
		LiveWorkerCount: 0,
		RunsDir:         t.TempDir(),
		Throughput: &dispatchThroughput{
			Landing: dispatchLandingOutageSuspect, Commits: 0, LiveWorkers: 0,
			Reason: "382 spawn(s) finished, 0 landed nothing, 0 still live",
		},
	}
	text := renderDispatchStatus(snap)
	first := strings.SplitN(strings.TrimSpace(text), "\n", 2)[0]
	if !strings.HasPrefix(first, "LANDING [outage_suspect]") {
		t.Fatalf("text card line 1 = %q, want the landing block first", first)
	}
	md := renderDispatchStatusMarkdown(snap)
	firstMD := strings.SplitN(strings.TrimSpace(md), "\n", 2)[0]
	if !strings.HasPrefix(firstMD, "LANDING [outage_suspect]") {
		t.Fatalf("markdown card line 1 = %q, want the landing block first", firstMD)
	}
	// And the outage must be legible in the card body, not only in the verdict word.
	if !strings.Contains(text, "outage") {
		t.Fatalf("text card must say outage in words, got %q", text)
	}
}

// An empty snapshot (no throughput block at all) must not invent a landing claim.
func TestDispatchStatusCardWithNoThroughputStaysSilent(t *testing.T) {
	snap := dispatchStatusSnapshot{LiveWorkerCount: 2, RunsDir: t.TempDir()}
	text := renderDispatchStatus(snap)
	if strings.Contains(text, "LANDING") {
		t.Fatalf("card must not claim a landing verdict when the block is absent, got %q", text)
	}
}

// --- goal launch rate (lesson 3) ---

func TestDispatchGoalLaunchRateFoldCountsPlanOnly(t *testing.T) {
	dir := t.TempDir()
	write := func(name, outcome string) {
		t.Helper()
		raw, _ := json.Marshal(map[string]any{"outcome": outcome})
		if err := os.WriteFile(filepath.Join(dir, name), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("a.receipt.json", "launched")
	for _, n := range []string{"b", "c", "d"} {
		write(n+".receipt.json", "plan_only")
	}
	// A non-receipt file must be ignored, not counted.
	if err := os.WriteFile(filepath.Join(dir, "noise.txt"), []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	got := dispatchGoalLaunchRateFold(dir)
	if got.Receipts != 4 {
		t.Fatalf("receipts = %d, want 4 (the .txt must be ignored)", got.Receipts)
	}
	if got.Launched != 1 || got.PlanOnly != 3 {
		t.Fatalf("launched=%d plan_only=%d, want 1/3", got.Launched, got.PlanOnly)
	}
	// 1/4 == 250permille, the corpus shape that motivated this block (4% live).
	if got.RatePermille != 250 {
		t.Fatalf("rate = %d, want 250", got.RatePermille)
	}
	if got.Status != dispatchLaunchLow {
		t.Fatalf("status = %q, want low", got.Status)
	}
	if !strings.Contains(got.NextAction, "LAUNCHER") {
		t.Fatalf("next_action = %q, want it to blame the launcher not the selector", got.NextAction)
	}
}

func TestDispatchGoalLaunchRateHealthyAboveBoundary(t *testing.T) {
	dir := t.TempDir()
	for i := 0; i < 9; i++ {
		raw, _ := json.Marshal(map[string]any{"outcome": "launched"})
		name := filepath.Join(dir, string(rune('a'+i))+".receipt.json")
		if err := os.WriteFile(name, raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got := dispatchGoalLaunchRateFold(dir)
	if got.Status != dispatchLaunchOK {
		t.Fatalf("status = %q, want ok at 9/9", got.Status)
	}
}

// A missing corpus is UNMEASURED, not 0% — the same honesty rule as the throughput
// block. A launcher that never ran and a launcher that always refuses are different
// facts and must not render identically.
func TestDispatchGoalLaunchRateMissingDirIsUnknownNotZero(t *testing.T) {
	got := dispatchGoalLaunchRateFold(filepath.Join(t.TempDir(), "absent"))
	if got.Status != dispatchLaunchUnknown {
		t.Fatalf("status = %q, want unknown", got.Status)
	}
	if got.RatePermille != 0 || got.Receipts != 0 {
		t.Fatalf("expected zeroed counters, got rate=%d receipts=%d", got.RatePermille, got.Receipts)
	}
	if !strings.Contains(dispatchGoalLaunchLine(got), "not a zero rate") {
		t.Fatalf("line must say the rate is unmeasured, got %q", dispatchGoalLaunchLine(got))
	}
}

// Unparseable receipts must be EXCLUDED from the ratio and reported, so a corrupt
// corpus cannot silently lower the launch rate.
func TestDispatchGoalLaunchRateExcludesUnparsed(t *testing.T) {
	dir := t.TempDir()
	raw, _ := json.Marshal(map[string]any{"outcome": "launched"})
	if err := os.WriteFile(filepath.Join(dir, "a.receipt.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "b.receipt.json"), []byte("{not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	got := dispatchGoalLaunchRateFold(dir)
	if got.Receipts != 1 {
		t.Fatalf("receipts = %d, want 1 (corrupt file excluded from the denominator)", got.Receipts)
	}
	if got.Unparsed != 1 {
		t.Fatalf("unparsed = %d, want 1", got.Unparsed)
	}
	if !strings.Contains(got.NextAction, "1 receipt(s) unparseable") {
		t.Fatalf("next_action must disclose the exclusion, got %q", got.NextAction)
	}
}

func TestDispatchGoalLaunchLineNilIsEmpty(t *testing.T) {
	if got := dispatchGoalLaunchLine(nil); got != "" {
		t.Fatalf("nil fold must render empty, got %q", got)
	}
}
