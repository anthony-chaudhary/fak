package main

// Tests for the shared durable post tail (slackPostTail). Every Slack feeder — bench,
// blockers, cachevalue, dojo, grafana, marketing, milestone — routes through this tail,
// so these two properties (deliver-and-confirm, and never-lose-on-failure) hold for all
// of them at once (#2262).

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fakeSlackCard is a minimal slackCard for the tail tests.
type fakeSlackCard struct {
	text   string
	blocks []any
}

func (c fakeSlackCard) Text() string  { return c.text }
func (c fakeSlackCard) Blocks() []any { return c.blocks }

// The happy path: the card is enqueued, the in-process drain posts it, and the historical
// `posted to CHANNEL ts=TS` confirmation is preserved (read back from the settled row).
func TestSlackPostTailDurablePostsAndConfirms(t *testing.T) {
	outboxTestDir(t)
	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()

	// Terse resolvers (no explicit source) — the shape the real feeders use, which yields
	// the historical `posted to CHANNEL ts=TS` line.
	var out, errb bytes.Buffer
	rc := slackPostTail(&out, &errb, slackPostSpec{
		card:           fakeSlackCard{text: "hello fak"},
		label:          "fak bench post",
		chanEnv:        "FAK_BENCH_CHANNEL",
		resolveChannel: func() string { return "C_TEST" },
		resolveToken:   func() string { return "xoxb-test" },
		apiBase:        srv.URL + "/",
	})
	if rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if got := out.String(); !strings.Contains(got, "posted to C_TEST ts=1.23") {
		t.Fatalf("missing preserved terse post confirmation, got: %q", got)
	}
	if posts != 1 {
		t.Fatalf("want exactly 1 post to Slack, got %d", posts)
	}
	ob, err := openOutbox()
	if err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	st, err := ob.Status(time.Now())
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if st.Posted != 1 || st.Pending != 0 {
		t.Fatalf("row did not settle as posted: %+v", st)
	}
}

// The no-loss property: when the send fails (Slack answers ok:false), the command must
// NOT hard-fail and the card must remain durably owed on the spool — the exact hole the
// old fire-and-forget tail left open.
func TestSlackPostTailDurableSurvivesSendFailure(t *testing.T) {
	outboxTestDir(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "conversations.history") {
			_, _ = io.WriteString(w, `{"ok":true,"messages":[]}`)
			return
		}
		_, _ = io.WriteString(w, `{"ok":false,"error":"channel_not_found"}`)
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	rc := slackPostTail(&out, &errb, slackPostSpec{
		card:    fakeSlackCard{text: "must survive a failed send"},
		channel: "C_TEST",
		token:   "xoxb-test",
		label:   "fak bench post",
		chanEnv: "FAK_BENCH_CHANNEL",
		apiBase: srv.URL + "/",
	})
	if rc != 0 {
		t.Fatalf("a failed send must not hard-fail the command: rc=%d stderr=%s", rc, errb.String())
	}
	ob, err := openOutbox()
	if err != nil {
		t.Fatalf("open outbox: %v", err)
	}
	st, err := ob.Status(time.Now())
	if err != nil {
		t.Fatalf("status: %v", err)
	}
	if st.Posted != 0 || st.Pending != 1 {
		t.Fatalf("failed post must remain durably owed, not lost or posted: %+v", st)
	}
}
