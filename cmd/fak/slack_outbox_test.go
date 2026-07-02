package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
)

// outboxTestDir points the outbox at a fresh temp spool for one test.
func outboxTestDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("FAK_SLACK_OUTBOX_DIR", dir)
	return dir
}

// okSlackServer answers every chat.postMessage/chat.update/history call with ok:true.
func okSlackServer(t *testing.T, posts *int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "conversations.history") {
			_, _ = io.WriteString(w, `{"ok":true,"messages":[]}`)
			return
		}
		if posts != nil {
			*posts++
		}
		_, _ = io.WriteString(w, `{"ok":true,"ts":"1.23"}`)
	}))
}

func TestSlackOutboxStatusEmptySpool(t *testing.T) {
	outboxTestDir(t)
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"status", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	var st slackoutbox.Status
	if err := json.Unmarshal(out.Bytes(), &st); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out.String())
	}
	if st.Pending != 0 || st.Dead != 0 {
		t.Fatalf("empty spool not empty: %+v", st)
	}
}

func TestSlackSendDurableEnqueuesAndDrains(t *testing.T) {
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	var out, errb bytes.Buffer
	rc := runSlackSend(&out, &errb, []string{
		"--durable", "--channel", "C1", "--text", "durable hello",
		"--token", "xoxb-test", "--api-base", srv.URL + "/",
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if posts != 1 {
		t.Fatalf("posts=%d, want 1", posts)
	}
	if !strings.Contains(out.String(), "enqueued durably") || !strings.Contains(out.String(), "posted 1") {
		t.Fatalf("output missing durable trail:\n%s", out.String())
	}
}

func TestSlackSendDurableSurvivesDeadWireThenDrains(t *testing.T) {
	outboxTestDir(t)
	// First send: the wire refuses (500s). The message must be spooled, exit 0.
	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = io.WriteString(w, `{"ok":false,"error":"internal_error"}`)
	}))
	var out, errb bytes.Buffer
	rc := runSlackSend(&out, &errb, []string{
		"--durable", "--channel", "C1", "--text", "outage survivor",
		"--token", "xoxb-test", "--api-base", dead.URL + "/",
	})
	dead.Close()
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "failed 1") {
		t.Fatalf("failure not recorded:\n%s", out.String())
	}

	// Recovery: a later drain against a healthy wire delivers the spooled row.
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()
	out.Reset()
	rc = runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/"})
	if rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}
	if posts != 1 || !strings.Contains(out.String(), "posted 1") {
		t.Fatalf("recovery drain did not deliver (posts=%d):\n%s", posts, out.String())
	}
}

func TestSlackOutboxDrainDryRunTouchesNothing(t *testing.T) {
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "planned"}); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--dry-run"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "1 send(s) planned") {
		t.Fatalf("plan missing:\n%s", out.String())
	}
	st, _ := ob.Status(time.Now())
	if st.Pending != 1 || st.Posted != 0 {
		t.Fatalf("dry-run mutated state: %+v", st)
	}
}

func TestSlackOutboxDeadAndRetryRoundTrip(t *testing.T) {
	outboxTestDir(t)
	ob, err := openOutbox()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "CBAD", Text: "will die", Source: "test"}); err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "conversations.history") {
			_, _ = io.WriteString(w, `{"ok":true,"messages":[]}`)
			return
		}
		_, _ = io.WriteString(w, `{"ok":false,"error":"channel_not_found"}`)
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/", "--max-attempts", "1"})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "dead 1") {
		t.Fatalf("row not dead-lettered:\n%s", out.String())
	}

	out.Reset()
	if rc := runSlackOutbox(&out, &errb, []string{"dead"}); rc != 0 {
		t.Fatalf("dead rc=%d", rc)
	}
	if !strings.Contains(out.String(), "channel_not_found") {
		t.Fatalf("dead listing missing reason:\n%s", out.String())
	}

	out.Reset()
	if rc := runSlackOutbox(&out, &errb, []string{"retry", "--all"}); rc != 0 {
		t.Fatalf("retry rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "re-armed 1") {
		t.Fatalf("retry did not re-arm:\n%s", out.String())
	}
	st, _ := ob.Status(time.Now())
	if st.Pending != 1 || st.Dead != 0 {
		t.Fatalf("retry state wrong: %+v", st)
	}
}

func TestOutboxHealthRungVerdicts(t *testing.T) {
	dir := outboxTestDir(t)
	now := time.Now()

	// Empty spool ⇒ OK.
	if hr := outboxHealthRung(now); hr.Verdict != verdictOK {
		t.Fatalf("empty spool: %+v", hr)
	}

	ob, err := slackoutbox.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	// Fresh pending row ⇒ still OK (within the stall budget).
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "fresh"}); err != nil {
		t.Fatal(err)
	}
	if hr := outboxHealthRung(now); hr.Verdict != verdictOK {
		t.Fatalf("fresh pending row: %+v", hr)
	}

	// A pending row past the stall budget ⇒ STALLED.
	old := now.Add(-3 * time.Hour).UTC().Format(time.RFC3339)
	if _, err := ob.Enqueue(slackoutbox.Row{Channel: "C1", Text: "ancient", EnqueuedAt: old}); err != nil {
		t.Fatal(err)
	}
	hr := outboxHealthRung(now)
	if hr.Verdict != verdictOutboxStalled {
		t.Fatalf("stalled backlog not flagged: %+v", hr)
	}
	if healthExit([]healthReport{hr}) != 1 {
		t.Fatal("STALLED must trip the health gate")
	}

	// A dead row outranks stalled ⇒ DEAD_ROWS.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "conversations.history") {
			_, _ = io.WriteString(w, `{"ok":true,"messages":[]}`)
			return
		}
		_, _ = io.WriteString(w, `{"ok":false,"error":"invalid_auth"}`)
	}))
	defer srv.Close()
	var out, errb bytes.Buffer
	if rc := runSlackOutbox(&out, &errb, []string{"drain", "--token", "xoxb-test", "--api-base", srv.URL + "/", "--max-attempts", "1"}); rc != 0 {
		t.Fatalf("drain rc=%d stderr=%s", rc, errb.String())
	}
	hr = outboxHealthRung(now)
	if hr.Verdict != verdictDeadRows || !strings.Contains(hr.Detail, "invalid_auth") {
		t.Fatalf("dead rows not flagged: %+v", hr)
	}
	if healthExit([]healthReport{hr}) != 1 {
		t.Fatal("DEAD_ROWS must trip the health gate")
	}
}

func TestScoreboardPostFlowEnqueuesOnFailure(t *testing.T) {
	outboxTestDir(t)
	// The post client is bound to a server that always refuses; the flow must spool
	// the card and exit 0 (fail open = delayed, not lost).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":false,"error":"internal_error"}`)
	}))
	defer srv.Close()
	orig := newScoreboardPostClient
	newScoreboardPostClient = func(tok string) (*scoreboard.Client, error) {
		return scoreboard.NewClient(tok, scoreboard.WithAPIBase(srv.URL+"/"), scoreboard.WithHTTPClient(srv.Client()))
	}
	defer func() { newScoreboardPostClient = orig }()

	var out, errb bytes.Buffer
	rc := scoreboardPostFlow(&out, &errb, scoreboard.Update{Title: "t", Score: "1"}, scoreboardPostOpts{
		prefix:      "test post",
		channelFlag: "C1",
		tokenFlag:   "xoxb-test",
		resolveChan: func() string { return "" },
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "enqueued durably") {
		t.Fatalf("failed post not spooled:\n%s", out.String())
	}
	ob, _ := openOutbox()
	st, _ := ob.Status(time.Now())
	if st.Pending != 1 {
		t.Fatalf("spool empty after failed post: %+v", st)
	}
}
