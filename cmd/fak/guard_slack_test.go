package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

func TestGuardSessionAgentNameNormalizesLauncher(t *testing.T) {
	cases := map[string]string{
		`C:\Program Files\Claude\claude.exe`: "claude.exe",
		"/usr/local/bin/codex":               "codex",
		"opencode":                           "opencode",
		"":                                   "unknown",
	}
	for in, want := range cases {
		if got := guardSessionAgentName([]string{in}); got != want {
			t.Fatalf("guardSessionAgentName(%q) = %q, want %q", in, got, want)
		}
	}
	if got := guardSessionAgentName(nil); got != "unknown" {
		t.Fatalf("guardSessionAgentName(nil) = %q, want unknown", got)
	}
}

func TestGuardSessionThreadRowIsRootPost(t *testing.T) {
	row := guardSessionThreadRow(guardSessionThreadMeta{
		Channel:   "CCHAN",
		Nonce:     "nonce-1",
		TraceID:   "trace one\ntrace two",
		Agent:     "claude",
		Provider:  "anthropic",
		AuditPath: "audit.jsonl",
		StartedAt: time.Unix(1000, 0),
		PID:       123,
	})
	if row.Channel != "CCHAN" || row.Nonce != "nonce-1" || row.Source != guardSessionThreadSource {
		t.Fatalf("row identity = %+v", row)
	}
	if row.ThreadTS != "" || row.UpdateTS != "" {
		t.Fatalf("guard session row must be a root post, got thread_ts=%q update_ts=%q", row.ThreadTS, row.UpdateTS)
	}
	for _, want := range []string{
		"fak guard session started",
		"session_thread_id: nonce-1",
		"trace_id: trace one trace two",
		"agent: claude",
		"provider: anthropic",
		"started_utc: 1970-01-01T00:16:40Z",
		"audit: audit.jsonl",
	} {
		if !strings.Contains(row.Text, want) {
			t.Fatalf("row text missing %q:\n%s", want, row.Text)
		}
	}
}

func TestEnqueueGuardSessionThreadDefaultsToDedicatedChannel(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)

	row, err := enqueueGuardSessionThread("trace-a", "anthropic", []string{"claude"}, "audit.jsonl", time.Unix(2000, 0))
	if err != nil {
		t.Fatalf("enqueueGuardSessionThread: %v", err)
	}
	if row.Channel != guardSessionsChannelDefault {
		t.Fatalf("channel = %q, want %q", row.Channel, guardSessionsChannelDefault)
	}
	if row.ThreadTS != "" || row.UpdateTS != "" {
		t.Fatalf("enqueued guard session must create a root thread, got thread_ts=%q update_ts=%q", row.ThreadTS, row.UpdateTS)
	}

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 1 {
		t.Fatalf("spool rows = %d, want 1", len(snap.Rows))
	}
	got := snap.Rows[0]
	if got.Channel != guardSessionsChannelDefault || got.Source != guardSessionThreadSource || got.ThreadTS != "" {
		t.Fatalf("spooled row = %+v", got)
	}
	if !strings.Contains(got.Text, "trace_id: trace-a") || !strings.Contains(got.Text, "agent: claude") {
		t.Fatalf("spooled row text missing guard identity:\n%s", got.Text)
	}
}

func TestGuardSessionControlPointRowsThreadUnderRoot(t *testing.T) {
	rows := guardSessionControlPointRows("root-nonce", "CCHAN", "line one\nline two", guardSessionControlContext{
		Command:    []string{"claude", "--flag"},
		Cwd:        `C:\work\fak`,
		Audit:      `C:\work\fak\.dispatch-runs\audit\session.jsonl`,
		Trace:      "trace-x",
		Provider:   "anthropic",
		GatewayURL: "http://127.0.0.1:8080",
	})
	if len(rows) != 2 {
		t.Fatalf("want 1 banner + 1 context reply, got %d rows", len(rows))
	}
	for _, r := range rows {
		if r.ParentNonce != "root-nonce" {
			t.Fatalf("reply not threaded under the root: parent_nonce=%q", r.ParentNonce)
		}
		if r.Channel != "CCHAN" || r.ThreadTS != "" || r.UpdateTS != "" {
			t.Fatalf("reply must be a deferred post: %+v", r)
		}
	}
	banner, context := rows[0], rows[1]
	if banner.Source != guardSessionThreadSource+":banner" || !strings.Contains(banner.Text, "line one\nline two") {
		t.Fatalf("banner reply wrong: %+v", banner)
	}
	if context.Source != guardSessionThreadSource+":context" {
		t.Fatalf("context reply source = %q", context.Source)
	}
	// Full paths (not basenames) so an operator can locate artifacts on disk.
	for _, want := range []string{`command: claude --flag`, `cwd: C:\work\fak`, `audit: C:\work\fak\.dispatch-runs\audit\session.jsonl`, "gateway: http://127.0.0.1:8080"} {
		if !strings.Contains(context.Text, want) {
			t.Fatalf("context reply missing %q:\n%s", want, context.Text)
		}
	}
}

func TestGuardSessionControlPointRowsChunkLongBanner(t *testing.T) {
	// A banner well over the reply ceiling splits into multiple labeled banner replies.
	big := strings.Repeat("x", guardSessionReplyTextLimit*2+10)
	rows := guardSessionControlPointRows("root", "CCHAN", big, guardSessionControlContext{})
	banners := 0
	for _, r := range rows {
		if r.Source == guardSessionThreadSource+":banner" {
			banners++
			if !strings.Contains(r.Text, "launch banner (") {
				t.Fatalf("multi-chunk banner not labeled: %q", r.Text[:40])
			}
		}
	}
	if banners < 2 {
		t.Fatalf("long banner not chunked: %d banner replies", banners)
	}
}

func TestGuardSessionControlPointRowsEmptyBannerStillPostsContext(t *testing.T) {
	rows := guardSessionControlPointRows("root", "CCHAN", "", guardSessionControlContext{Trace: "t"})
	if len(rows) != 1 || rows[0].Source != guardSessionThreadSource+":context" {
		t.Fatalf("empty banner should leave only the context reply, got %d rows", len(rows))
	}
}

type fakeGuardMetrics struct{ sum gateway.AdjudicationSummary }

func (f fakeGuardMetrics) AdjudicationSummary() gateway.AdjudicationSummary { return f.sum }

func TestEnqueueGuardSessionControlPointsSpoolsThreadedReplies(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	n, err := enqueueGuardSessionControlPoints("root-1", "C1", "banner text", guardSessionControlContext{
		Trace:   "t",
		Command: []string{"claude"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("queued %d replies, want 2 (banner + context)", n)
	}
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap.Rows) != 2 {
		t.Fatalf("spool rows = %d, want 2", len(snap.Rows))
	}
	for _, r := range snap.Rows {
		if r.ParentNonce != "root-1" {
			t.Fatalf("reply not deferred-threaded under the root: %+v", r)
		}
	}
}

func TestGuardSessionCardUpdaterStartsAndStopsCleanly(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	card := newGuardSessionCard("C1", "root-1", time.Now())
	card.startUpdater(fakeGuardMetrics{})
	card.stopUpdater()
	card.stopUpdater() // second stop is a safe no-op
	// The 20s tick never fires within the test, and the root is unposted anyway, so no
	// update row may have been enqueued.
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			t.Fatalf("updater enqueued against an unposted root: %+v", r)
		}
	}
}

func TestGuardSessionCardFinalizeWithoutTokenStaysDurable(t *testing.T) {
	clearSlackEnv(t) // no ambient token — finalize enqueues but does not drain
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}
	card := newGuardSessionCard("C1", "root-1", time.Now())
	card.finalize("status: completed (exit=0) · turns=1")
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, r := range snap.Rows {
		if r.UpdateTS == "1.23" && strings.Contains(r.Text, "completed") {
			found = true
		}
	}
	if !found {
		t.Fatal("final outcome edit was not enqueued durably")
	}
}

func TestGuardSessionLiveAndFinalLines(t *testing.T) {
	sum := gateway.AdjudicationSummary{Total: 7, InputTokens: 120, OutputTokens: 40, CachedPromptTokens: 90, Denied: 2}
	live := guardSessionLiveLine(sum, 65*time.Second)
	for _, want := range []string{"status: running", "turns=7", "in=120 out=40 cached=90", "denied=2", "elapsed=1m5s"} {
		if !strings.Contains(live, want) {
			t.Fatalf("live line missing %q: %s", want, live)
		}
	}
	if got := guardSessionFinalLine(0, sum, time.Second); !strings.Contains(got, "status: completed (exit=0)") {
		t.Fatalf("final(0) = %s", got)
	}
	if got := guardSessionFinalLine(2, sum, time.Second); !strings.Contains(got, "status: failed (exit=2)") {
		t.Fatalf("final(2) = %s", got)
	}
}

func TestGuardSessionCardEnqueueUpdateSkipsUnpostedRoot(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	card := newGuardSessionCard("C1", "root-1", time.Now())
	if err := card.enqueueUpdate("progress"); err != nil {
		t.Fatalf("enqueueUpdate: %v", err)
	}
	// The root has not posted, so no update row may be enqueued (no card ts to edit).
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			t.Fatalf("update row enqueued against an unposted root: %+v", r)
		}
	}
	if len(snap.Rows) != 1 {
		t.Fatalf("spool rows = %d, want 1 (root only)", len(snap.Rows))
	}
}

func TestGuardSessionCardEnqueueUpdateEditsPostedRoot(t *testing.T) {
	clearSlackEnv(t)
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "root", Nonce: "root-1"}); err != nil {
		t.Fatal(err)
	}
	// Drain the root so it reaches posted state (ts "1.23" from okSlackServer).
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}

	card := newGuardSessionCard("C1", "root-1", time.Now())
	if err := card.enqueueUpdate("progress a"); err != nil {
		t.Fatal(err)
	}
	if err := card.enqueueUpdate("progress b"); err != nil {
		t.Fatal(err)
	}
	snap, err := ob.Load()
	if err != nil {
		t.Fatal(err)
	}
	updates := 0
	for _, r := range snap.Rows {
		if r.UpdateTS != "" {
			updates++
			if r.UpdateTS != "1.23" {
				t.Fatalf("update row not bound to the root ts: %+v", r)
			}
			if r.Source != guardSessionThreadSource+":status" {
				t.Fatalf("update row source = %q", r.Source)
			}
		}
	}
	if updates != 2 {
		t.Fatalf("want 2 update rows (coalesced at drain), got %d", updates)
	}
}
