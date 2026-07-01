package main

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// TestRenderSkippedHumanBlockedCardEmpty pins the all-clear line: an empty human-blocked set
// posts an honest "holding nothing for a human" rather than a bare header, so a scheduled run
// on a clean backlog is legible instead of silent.
func TestRenderSkippedHumanBlockedCardEmpty(t *testing.T) {
	got := renderSkippedHumanBlockedCard(nil, "")
	if !strings.Contains(got, "no human-blocked issues") {
		t.Fatalf("empty card = %q, want the all-clear line", got)
	}
	if strings.Contains(got, "•") {
		t.Fatalf("empty card should have no issue rows: %q", got)
	}
}

// TestRenderSkippedHumanBlockedCardRows pins the populated card: the headline count, one row per
// issue carrying the title and next-action, and a Slack #-link when a repo URL is set.
func TestRenderSkippedHumanBlockedCardRows(t *testing.T) {
	issues := []dispatchtick.SkippedIssue{
		{Number: 42, Title: "gateway: waiting on vendor token", NextAction: "wait for the named external/human blocker to clear before worker dispatch"},
		{Number: 7, Title: "compute: needs GPU-server hours"},
	}
	got := renderSkippedHumanBlockedCard(issues, "https://github.com/owner/repo/")
	if !strings.Contains(got, "*2 issue(s) blocked by a human*") {
		t.Fatalf("card missing headline count: %q", got)
	}
	// repo-url set → the linked ref form, trailing slash trimmed.
	if !strings.Contains(got, "<https://github.com/owner/repo/issues/42|#42> gateway: waiting on vendor token — wait for the named") {
		t.Fatalf("card missing linked row w/ next-action: %q", got)
	}
	if !strings.Contains(got, "<https://github.com/owner/repo/issues/7|#7> compute: needs GPU-server hours") {
		t.Fatalf("card missing second linked row: %q", got)
	}
}

// TestRenderSkippedHumanBlockedCardBareRef pins the no-repo-url path: a bare #N ref, no link.
func TestRenderSkippedHumanBlockedCardBareRef(t *testing.T) {
	got := renderSkippedHumanBlockedCard([]dispatchtick.SkippedIssue{{Number: 9, Title: "t"}}, "")
	if !strings.Contains(got, "• #9 t") {
		t.Fatalf("bare-ref row wrong: %q", got)
	}
	if strings.Contains(got, "<http") {
		t.Fatalf("no repo-url should mean no link: %q", got)
	}
}

// TestRenderSkippedHumanBlockedCardOverflow pins the cap: at most skippedMaxRows rows are listed
// and the remainder is summarised as a "… and N more" tail so a large backlog stays one post.
func TestRenderSkippedHumanBlockedCardOverflow(t *testing.T) {
	n := skippedMaxRows + 5
	issues := make([]dispatchtick.SkippedIssue, 0, n)
	for i := 0; i < n; i++ {
		issues = append(issues, dispatchtick.SkippedIssue{Number: i + 1, Title: "x"})
	}
	got := renderSkippedHumanBlockedCard(issues, "")
	if rows := strings.Count(got, "• #"); rows != skippedMaxRows {
		t.Fatalf("listed %d rows, want the cap %d", rows, skippedMaxRows)
	}
	if !strings.Contains(got, "… and 5 more") {
		t.Fatalf("overflow tail missing: %q", got)
	}
}

// TestHumanBlockedSkippedFiltersByReason pins the selection: only rows whose reason is
// BLOCKED_BY_HUMAN survive — epics, triage-only, duplicate-risk, and the unverified-label
// bucket are NOT a human's to clear and must not reach the human-blocked channel.
func TestHumanBlockedSkippedFiltersByReason(t *testing.T) {
	// Reasons are matched against the router's stable wire values (literals here, decoupled
	// from dispatchtick's internal reason constants).
	router := dispatchtick.RouterPayload{SkippedHumanBlocked: []dispatchtick.SkippedIssue{
		{Number: 5, Reason: "BLOCKED_BY_HUMAN"},
		{Number: 4, Reason: "ISSUE_HUMAN_BLOCK_UNVERIFIED"},
		{Number: 3, Reason: "ISSUE_NOT_DISPATCH_LEAF"},
		{Number: 2, Reason: "ISSUE_DUPLICATE_RISK"},
		{Number: 1, Reason: "BLOCKED_BY_HUMAN"},
	}}
	got := humanBlockedSkipped(router)
	if len(got) != 2 {
		t.Fatalf("filtered %d, want 2 BLOCKED_BY_HUMAN rows", len(got))
	}
	if got[0].Number != 5 || got[1].Number != 1 {
		t.Fatalf("filter changed order/selection: %+v", got)
	}
}

// TestRunDispatchSkippedDryRun pins the default-safe path: --dry-run renders the card and the
// "would post to <channel>" line and posts nothing, resolving the channel from the env var.
func TestRunDispatchSkippedDryRun(t *testing.T) {
	t.Setenv("FAK_SKIPPED_CHANNEL", "C0TEST")
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{SkippedHumanBlocked: []dispatchtick.SkippedIssue{
			{Number: 100, Title: "held for a human", Reason: "BLOCKED_BY_HUMAN"},
			{Number: 99, Title: "not a human's job", Reason: "ISSUE_NOT_DISPATCH_LEAF"},
		}}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })

	var out, errb bytes.Buffer
	code := runDispatchSkipped(&out, &errb, []string{"--workspace", t.TempDir(), "--dry-run"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "*1 issue(s) blocked by a human*") {
		t.Fatalf("dry-run card wrong count (should exclude the non-human skip): %q", s)
	}
	if !strings.Contains(s, "#100 held for a human") {
		t.Fatalf("dry-run card missing the human-blocked row: %q", s)
	}
	if strings.Contains(s, "#99") {
		t.Fatalf("dry-run card leaked a non-human skip: %q", s)
	}
	if !strings.Contains(s, "(dry-run: 1 human-blocked issue(s); would post to C0TEST)") {
		t.Fatalf("dry-run footer wrong: %q", s)
	}
}

// TestRunDispatchSkippedNoChannel pins the live-run guard: with no --channel and the env var
// unset, the command refuses with a usage error (exit 2) instead of misrouting.
func TestRunDispatchSkippedNoChannel(t *testing.T) {
	t.Setenv("FAK_SKIPPED_CHANNEL", "")
	old := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = old })

	var out, errb bytes.Buffer
	code := runDispatchSkipped(&out, &errb, []string{"--workspace", t.TempDir()})
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (no channel); stdout=%q stderr=%q", code, out.String(), errb.String())
	}
	if !strings.Contains(errb.String(), "no target channel") {
		t.Fatalf("stderr missing the no-channel diagnostic: %q", errb.String())
	}
}
