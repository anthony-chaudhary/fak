package main

import (
	"bytes"
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestSlackWalkIncludesNewsRefreshCommand(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackWalk(&out, &errb, nil)
	if code != 0 {
		t.Fatalf("slack walk exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "news") || !strings.Contains(got, "fak slack refresh --surface news [--news-title TITLE --news-file FILE]") {
		t.Fatalf("walk output missing news refresh command:\n%s", got)
	}
}

func TestSlackRefreshNewsAuditsWithoutPublishing(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "news"})
	if code == 0 {
		t.Fatalf("unwired news unexpectedly healthy: %s", out.String())
	}
	got := out.String()
	if !strings.Contains(got, "news: NEVER_POSTED_OR_UNWIRED") || !strings.Contains(got, "--news-title TITLE --news-file FILE") {
		t.Fatalf("refresh output omitted audited news state or next command:\n%s", got)
	}
	if strings.Contains(got, "DRY-RUN: would post") {
		t.Fatalf("audit path attempted publication:\n%s", got)
	}
}

func TestSlackRefreshGitHubPayloadAppearsInDryRun(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_BACKLOG_CHANNEL", "C-LEGACY-MUST-NOT-WIN")
	payload := []byte(`[{"number":8790,"title":"refresh blockers and backlog from GitHub","url":"https://github.com/anthony-chaudhary/fak/issues/8790","assignees":[],"labels":[{"name":"blocked"}]}]`)
	var out, errb bytes.Buffer
	code := runSlackRefreshWithGH(&out, &errb, []string{"--surface", "blockers,backlog", "--backlog-channel", "C-BACKLOG"}, func(args ...string) ([]byte, error) {
		got := strings.Join(args, " ")
		if !strings.Contains(got, "--limit 100") || !strings.Contains(got, "number,title,url,assignees,labels") {
			t.Fatalf("unbounded or incomplete gh request: %s", got)
		}
		return payload, nil
	})
	if code != 0 {
		t.Fatalf("refresh exit = %d, stderr=%s\nstdout=%s", code, errb.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "#8790") || !strings.Contains(got, "refresh blockers and backlog from GitHub") {
		t.Fatalf("dry-run omitted current issue data:\n%s", got)
	}
	if !strings.Contains(got, "== blockers: DRY-RUN ==") || !strings.Contains(got, "== backlog: DRY-RUN ==") {
		t.Fatalf("both GitHub-backed surfaces did not execute:\n%s", got)
	}
}

func TestSlackRefreshBacklogChannelReachesScoreboardCaller(t *testing.T) {
	old := slackRefreshRunScoreboardPost
	t.Cleanup(func() { slackRefreshRunScoreboardPost = old })
	var got []string
	slackRefreshRunScoreboardPost = func(_, _ io.Writer, argv []string) int {
		got = append([]string(nil), argv...)
		return 0
	}
	t.Setenv("FAK_BACKLOG_CHANNEL", "C-LEGACY")
	action := slackRefreshActions()["backlog"]
	if code := action.Run(io.Discard, io.Discard, true, slackRefreshOptions{BacklogIssues: `[]`, BacklogChannel: "C-EXPLICIT"}); code != 0 {
		t.Fatalf("backlog refresh code=%d", code)
	}
	joined := strings.Join(got, " ")
	if !strings.Contains(joined, "--channel C-EXPLICIT") || strings.Contains(joined, "C-LEGACY") {
		t.Fatalf("scoreboard argv=%q", joined)
	}
}

func TestSlackRefreshGitHubFailureIsTypedAndNeverAllClear(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefreshWithGH(&out, &errb, []string{"--surface", "blockers,backlog", "--json"}, func(args ...string) ([]byte, error) { return nil, fmt.Errorf("authentication required") })
	if code == 0 {
		t.Fatalf("GitHub failure unexpectedly succeeded: %s", out.String())
	}
	got := out.String()
	if !strings.Contains(got, `"error_type": "GITHUB_FETCH_FAILED"`) || !strings.Contains(got, "no all-clear rendered") {
		t.Fatalf("GitHub failure is not typed:\n%s", got)
	}
	if strings.Contains(strings.ToLower(got), "no standing blockers") {
		t.Fatalf("failure fabricated all-clear:\n%s", got)
	}
}

func TestSlackRefreshUnknownSurface(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "missing"})
	if code != 2 {
		t.Fatalf("unknown surface exit = %d, stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(errb.String(), "unknown surface") {
		t.Fatalf("unknown surface error not surfaced: %s", errb.String())
	}
}

func TestSlackRefreshAlertsAndGuardSessionsAreAuditable(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "alerts,guard-sessions", "--continue-on-error"})
	if code == 0 {
		t.Fatalf("unwired surfaces unexpectedly healthy: %s", out.String())
	}
	got := out.String()
	if !strings.Contains(got, "== alerts: FAIL ==") || !strings.Contains(got, "INCOMPLETE") {
		t.Fatalf("alerts audit did not preserve incomplete evidence:\n%s", got)
	}
	if !strings.Contains(got, "== guard-sessions: FAIL ==") || !strings.Contains(got, `"pending"`) {
		t.Fatalf("guard-session audit omitted durable outbox evidence:\n%s", got)
	}
	if strings.Contains(strings.ToLower(got), "all clear") {
		t.Fatalf("audit synthesized an all-clear alert:\n%s", got)
	}
}

func TestSlackWalkMarksAlertsAndGuardSessionsRunnable(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackWalk(&out, &errb, nil)
	if code != 0 {
		t.Fatalf("walk exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, surface := range []string{"alerts", "guard-sessions"} {
		if !strings.Contains(got, surface) || !strings.Contains(got, "audit-only") {
			t.Fatalf("walk did not expose %s audit action:\n%s", surface, got)
		}
	}
}

func TestSlackRefreshAuditOutputIsPortableASCII(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	_ = runSlackRefresh(&out, &errb, []string{"--surface", "alerts"})
	if strings.ContainsRune(out.String(), '\ufffd') {
		t.Fatalf("audit output contains replacement rune: %q", out.String())
	}
}

func TestSlackRefreshScoreboardUsesBuiltInOutboxRollup(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SLACK_OUTBOX_DIR", t.TempDir())
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C-SCOREBOARD")
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "scoreboard"})
	if code != 0 {
		t.Fatalf("scoreboard refresh exit = %d, stderr=%s, stdout=%s", code, errb.String(), out.String())
	}
	got := out.String()
	for _, want := range []string{"scoreboard: DRY-RUN", "slack-outbox-pending", "source=fak slack outbox status", "pending=0"} {
		if !strings.Contains(got, want) {
			t.Fatalf("built-in scoreboard omitted %q:\n%s", want, got)
		}
	}
}
