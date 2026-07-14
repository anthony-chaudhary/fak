package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

func sampleMessages() []slackwire.Message {
	// Deliberately out of order (newest first, as conversations.history returns) so the
	// chronological sort is exercised by the render tests.
	return []slackwire.Message{
		{Type: "message", TS: "1719600120.000200", BotID: "B42", Text: "deploy is green ✓"},
		{Type: "message", TS: "1719600000.000100", User: "U7", Text: "kicking off the run"},
	}
}

func TestRenderShotTextShape(t *testing.T) {
	got := renderShotText("#dispatch (C0ABC123)", sortMessagesChrono(sampleMessages()))
	// Header names the channel and the count.
	if !strings.Contains(got, "#dispatch (C0ABC123) — 2 message(s), oldest first") {
		t.Fatalf("header missing/wrong:\n%s", got)
	}
	// Oldest message must appear before the newest (chronological).
	iKick := strings.Index(got, "kicking off the run")
	iDeploy := strings.Index(got, "deploy is green")
	if iKick < 0 || iDeploy < 0 || iKick > iDeploy {
		t.Fatalf("messages not in chronological order:\n%s", got)
	}
	// Author labels: user id as-is, bot id prefixed.
	if !strings.Contains(got, "U7") || !strings.Contains(got, "bot:B42") {
		t.Fatalf("author labels missing:\n%s", got)
	}
	// Body lines are indented.
	if !strings.Contains(got, "    kicking off the run") {
		t.Fatalf("message body not indented:\n%s", got)
	}
}

func TestRenderShotTextNestsThreadReplies(t *testing.T) {
	msgs := []slackwire.Message{
		{Type: "message", TS: "1.0", BotID: "B1", Text: "session running"},
		{Type: "message", TS: "2.0", ThreadTS: "1.0", BotID: "B1", Text: "launch context"},
		{Type: "message", TS: "3.0", ThreadTS: "1.0", BotID: "B1", Text: "session failed"},
	}
	got := renderShotText("#guard-sessions", msgs)
	for _, want := range []string{"1 message(s), oldest first · 2 threaded replies", "↳", "launch context", "session failed"} {
		if !strings.Contains(got, want) {
			t.Fatalf("thread capture missing %q:\n%s", want, got)
		}
	}
	if strings.Index(got, "session running") > strings.Index(got, "launch context") {
		t.Fatalf("reply rendered before its root:\n%s", got)
	}
}

func TestRenderShotTextEmpty(t *testing.T) {
	got := renderShotText("#dispatch (C0ABC123)", nil)
	if !strings.Contains(got, "0 message(s)") || !strings.Contains(got, "channel is empty") {
		t.Fatalf("empty transcript wrong:\n%s", got)
	}
}

func TestRenderShotHTMLEscapesAndLinks(t *testing.T) {
	msgs := []slackwire.Message{{Type: "message", TS: "1719600000.000100", User: "U7", Text: "<script>alert(1)</script> & done"}}
	got := renderShotHTML("#dispatch (C0ABC123)", "https://app.slack.com/client/T1/C0ABC123", msgs)
	if strings.Contains(got, "<script>alert(1)</script>") {
		t.Fatalf("raw script tag leaked into HTML (XSS):\n%s", got)
	}
	if !strings.Contains(got, "&lt;script&gt;alert(1)&lt;/script&gt; &amp; done") {
		t.Fatalf("message text not escaped:\n%s", got)
	}
	if !strings.Contains(got, "https://app.slack.com/client/T1/C0ABC123") {
		t.Fatalf("web URL link missing:\n%s", got)
	}
	if !strings.HasPrefix(got, "<!doctype html>") {
		t.Fatalf("not a standalone HTML document:\n%.40s", got)
	}
}

func TestRenderShotHTMLNestsThreadReplies(t *testing.T) {
	msgs := []slackwire.Message{
		{Type: "message", TS: "1.0", BotID: "B1", Text: "guard session · RUNNING"},
		{Type: "message", TS: "2.0", ThreadTS: "1.0", BotID: "B1", Text: "terminal outcome: <failed>"},
	}
	got := renderShotHTML("#guard-sessions", "", msgs)
	for _, want := range []string{"1 threaded reply", `class="msg reply"`, "terminal outcome: &lt;failed&gt;"} {
		if !strings.Contains(got, want) {
			t.Fatalf("HTML thread capture missing %q:\n%s", want, got)
		}
	}
}

func TestSlackLaunchURLs(t *testing.T) {
	if got := slackDeepLink("T1", "C2"); got != "slack://channel?team=T1&id=C2" {
		t.Fatalf("deep link: %q", got)
	}
	if got := slackWebURL("T1", "C2"); got != "https://app.slack.com/client/T1/C2" {
		t.Fatalf("web url: %q", got)
	}
	// No team resolved → legible placeholder, still a usable shape.
	if got := slackDeepLink("", "C2"); got != "slack://channel?team=<team>&id=C2" {
		t.Fatalf("deep link placeholder: %q", got)
	}
}

func TestFormatSlackTS(t *testing.T) {
	want := time.Unix(1719600000, 0).UTC().Format("2006-01-02 15:04:05 UTC")
	if got := formatSlackTS("1719600000.000100"); got != want {
		t.Fatalf("formatSlackTS = %q, want %q", got, want)
	}
	if got := formatSlackTS("not-a-ts"); got != "not-a-ts" {
		t.Fatalf("bad ts should pass through, got %q", got)
	}
}

func TestLooksLikeChannelID(t *testing.T) {
	for _, ok := range []string{"C0ABC123", "G01234567", "DABCDEFGH"} {
		if !looksLikeChannelID(ok) {
			t.Errorf("%q should look like a channel id", ok)
		}
	}
	for _, no := range []string{"dispatch", "scoreboard", "C012", "c0abc123", "node-usage"} {
		if looksLikeChannelID(no) {
			t.Errorf("%q should NOT look like a channel id", no)
		}
	}
}

func TestRunSlackShotDryRunJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runSlackShot(&out, &errb, []string{"--channel", "C0ABC123", "--team", "T9", "--json", "--dry-run"})
	if code != 0 {
		t.Fatalf("dry-run exit=%d stderr=%s", code, errb.String())
	}
	var r shotResult
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("json: %v\nout=%s", err, out.String())
	}
	if r.Channel != "C0ABC123" || r.Team != "T9" || !r.DryRun {
		t.Fatalf("unexpected result: %+v", r)
	}
	if r.DeepLink != "slack://channel?team=T9&id=C0ABC123" {
		t.Fatalf("deep link wrong: %s", r.DeepLink)
	}
}

func TestRunSlackShotFlagsAfterPositional(t *testing.T) {
	// Regression: flags must parse when they follow the surface positional
	// (`fak slack shot dispatch --dry-run`), and dry-run must NOT touch the network.
	prev := slackShotHistory
	defer func() { slackShotHistory = prev }()
	called := false
	slackShotHistory = func(token, apiBase, channel string, limit int) ([]slackwire.Message, error) {
		called = true
		return nil, nil
	}
	var out, errb bytes.Buffer
	// Positional channel id FIRST, then flags — the ordering that a plain fs.Parse drops.
	code := runSlackShot(&out, &errb, []string{"C0ABC123", "--dry-run", "--json", "--team", "T9"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if called {
		t.Fatal("dry-run must not call history")
	}
	var r shotResult
	if err := json.Unmarshal(out.Bytes(), &r); err != nil {
		t.Fatalf("json: %v\nout=%s", err, out.String())
	}
	if !r.DryRun || r.Channel != "C0ABC123" || r.Team != "T9" {
		t.Fatalf("flags after positional not applied: %+v", r)
	}
}

func TestRunSlackShotDryRunNeedsTarget(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runSlackShot(&out, &errb, []string{"--dry-run"}); code != 2 {
		t.Fatalf("no target should be usage error 2, got %d", code)
	}
	if !strings.Contains(errb.String(), "surfaces:") {
		t.Fatalf("error should list surfaces:\n%s", errb.String())
	}
}

func TestRunSlackShotCapturesInjectedHistory(t *testing.T) {
	prevHist, prevTeam := slackShotHistory, slackShotTeam
	defer func() { slackShotHistory, slackShotTeam = prevHist, prevTeam }()

	var gotChannel string
	var gotLimit int
	slackShotHistory = func(token, apiBase, channel string, limit int) ([]slackwire.Message, error) {
		gotChannel, gotLimit = channel, limit
		return sampleMessages(), nil
	}
	slackShotTeam = func(token, apiBase string) string { return "TSTUB" }

	outPath := filepath.Join(t.TempDir(), "shot.html")
	var out, errb bytes.Buffer
	code := runSlackShot(&out, &errb, []string{"--channel", "C0ABC123", "--token", "xoxb-test", "-n", "5", "--out", outPath})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	if gotChannel != "C0ABC123" || gotLimit != 5 {
		t.Fatalf("history called with channel=%q limit=%d", gotChannel, gotLimit)
	}
	// stdout shows the transcript and points at the written HTML.
	if !strings.Contains(out.String(), "deploy is green") || !strings.Contains(out.String(), "wrote HTML screenshot") {
		t.Fatalf("stdout missing transcript/out pointer:\n%s", out.String())
	}
	// The HTML capture exists, is standalone, and rendered both messages.
	b, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("read out: %v", err)
	}
	htmlDoc := string(b)
	if !strings.HasPrefix(htmlDoc, "<!doctype html>") || !strings.Contains(htmlDoc, "kicking off the run") {
		t.Fatalf("HTML capture wrong:\n%s", htmlDoc)
	}
}

func TestRunSlackShotReadsAndCapturesReplies(t *testing.T) {
	prevHist, prevReplies, prevTeam := slackShotHistory, slackShotReplies, slackShotTeam
	defer func() { slackShotHistory, slackShotReplies, slackShotTeam = prevHist, prevReplies, prevTeam }()

	slackShotHistory = func(token, apiBase, channel string, limit int) ([]slackwire.Message, error) {
		return []slackwire.Message{{Type: "message", TS: "1.0", BotID: "B1", Text: "root", ReplyCount: 1}}, nil
	}
	slackShotReplies = func(token, apiBase, channel, threadTS string, limit int) ([]slackwire.Message, error) {
		if channel != "C0ABC123" || threadTS != "1.0" || limit != slackShotReplyLimit {
			t.Fatalf("reply read channel/thread/limit = %q/%q/%d", channel, threadTS, limit)
		}
		return []slackwire.Message{
			{Type: "message", TS: "1.0", BotID: "B1", Text: "root"},
			{Type: "message", TS: "2.0", ThreadTS: "1.0", BotID: "B1", Text: "useful outcome"},
		}, nil
	}
	slackShotTeam = func(token, apiBase string) string { return "T1" }

	var out, errb bytes.Buffer
	code := runSlackShot(&out, &errb, []string{"--channel", "C0ABC123", "--token", "xoxb-test"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	for _, want := range []string{"1 threaded reply", "↳", "useful outcome"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("capture missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunSlackShotHistoryErrorExits1(t *testing.T) {
	prev := slackShotHistory
	defer func() { slackShotHistory = prev }()
	slackShotHistory = func(token, apiBase, channel string, limit int) ([]slackwire.Message, error) {
		return nil, errTest
	}
	var out, errb bytes.Buffer
	code := runSlackShot(&out, &errb, []string{"--channel", "C0ABC123", "--token", "xoxb-test", "--team", "T1"})
	if code != 1 {
		t.Fatalf("history error should exit 1, got %d (stderr=%s)", code, errb.String())
	}
	if !strings.Contains(errb.String(), "history:") {
		t.Fatalf("stderr should name the history failure:\n%s", errb.String())
	}
}
