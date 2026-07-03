package dispatchpost

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/slackoutbox"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// fakeSlack is an httptest Slack Web API: chat.postMessage, chat.update, and
// conversations.history over real HTTP, so the card lifecycle is exercised
// through the REAL slackwire client and the REAL outbox drainer.
type fakeSlack struct {
	mu     sync.Mutex
	nextTS int
	msgs   []fakeMsg
}

type fakeMsg struct {
	Channel  string
	TS       string
	ThreadTS string
	Text     string
	Nonce    string
}

func (f *fakeSlack) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/chat.postMessage", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Channel  string                     `json:"channel"`
			Text     string                     `json:"text"`
			ThreadTS string                     `json:"thread_ts"`
			Metadata *slackwire.MessageMetadata `json:"metadata"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		f.mu.Lock()
		f.nextTS++
		ts := fmt.Sprintf("%d.0", f.nextTS)
		m := fakeMsg{Channel: body.Channel, TS: ts, ThreadTS: body.ThreadTS, Text: body.Text}
		if body.Metadata != nil {
			m.Nonce, _ = body.Metadata.EventPayload["nonce"].(string)
		}
		f.msgs = append(f.msgs, m)
		f.mu.Unlock()
		fmt.Fprintf(w, `{"ok":true,"ts":%q}`, ts)
	})
	mux.HandleFunc("/chat.update", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Channel string `json:"channel"`
			TS      string `json:"ts"`
			Text    string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		f.mu.Lock()
		ok := false
		for i := range f.msgs {
			if f.msgs[i].TS == body.TS && f.msgs[i].Channel == body.Channel {
				f.msgs[i].Text = body.Text
				ok = true
			}
		}
		f.mu.Unlock()
		if !ok {
			fmt.Fprint(w, `{"ok":false,"error":"message_not_found"}`)
			return
		}
		fmt.Fprint(w, `{"ok":true}`)
	})
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		msgs := make([]map[string]any, 0, len(f.msgs))
		for _, m := range f.msgs {
			row := map[string]any{"type": "message", "ts": m.TS, "text": m.Text, "thread_ts": m.ThreadTS}
			if m.Nonce != "" {
				row["metadata"] = map[string]any{
					"event_type":    slackwire.IdemEventType,
					"event_payload": map[string]any{"nonce": m.Nonce},
				}
			}
			msgs = append(msgs, row)
		}
		f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "messages": msgs})
	})
	return mux
}

// transcript renders the channel as a reader sees it: one line per top-level
// message, thread replies indented under their parent.
func (f *fakeSlack) transcript(channel string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	var b strings.Builder
	for _, m := range f.msgs {
		if m.Channel != channel || m.ThreadTS != "" {
			continue
		}
		fmt.Fprintf(&b, "• %s\n", m.Text)
		for _, r := range f.msgs {
			if r.Channel == channel && r.ThreadTS == m.TS {
				fmt.Fprintf(&b, "    ↳ %s\n", strings.ReplaceAll(r.Text, "\n", " / "))
			}
		}
	}
	return b.String()
}

func (f *fakeSlack) topLevelCount(channel string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	n := 0
	for _, m := range f.msgs {
		if m.Channel == channel && m.ThreadTS == "" {
			n++
		}
	}
	return n
}

func TestCardWitnessFoldsGitEvidenceNotSelfReport(t *testing.T) {
	// Exit 0 with no landed commit: NOT_SHIPPED, verify=none — a green exit
	// code cannot masquerade as a landed change.
	green := CardWitness(Result{RunID: "r-1", ExitCode: 0})
	if green.Verdict() != "NOT_SHIPPED" || green.VerifySource != slackoutbox.VerifyNone {
		t.Fatalf("no-commit run must fold NOT_SHIPPED/none: %+v", green)
	}
	// A landed commit WITH a bindable ship-stamp: grepped from the git delta.
	shipped := CardWitness(Result{RunID: "r-2", ExitCode: 0,
		Commits: []string{"ab12cd3 fix(gateway): treat same-tick ready as positive (fak gateway)"}})
	if shipped.Verdict() != "SHIPPED" || shipped.CommitSHA != "ab12cd3" ||
		shipped.ShipStamp != "(fak gateway)" || shipped.VerifySource != slackoutbox.VerifyGrep {
		t.Fatalf("stamped commit must fold SHIPPED/grep: %+v", shipped)
	}
	// A landed commit WITHOUT a stamp: the SHA is reported but the verdict
	// stays NOT_SHIPPED — a bare un-stamped subject is not bindable.
	bare := CardWitness(Result{RunID: "r-3", Commits: []string{"ab12cd3 fixed some stuff"}})
	if bare.Verdict() != "NOT_SHIPPED" || bare.CommitSHA != "ab12cd3" || bare.VerifySource != slackoutbox.VerifyNone {
		t.Fatalf("un-stamped commit must stay NOT_SHIPPED: %+v", bare)
	}
}

// TestRunCardLifecycleAgainstHTTPTestFake is the DoD integration witness for
// the migrated dispatch consumer: post once at run start, restart-resume
// without a second post, final witness edit in place, detail in the thread —
// all through the real slackwire HTTP client against an httptest Slack fake.
// It also captures the before/after channel transcript: the legacy consumer's
// terminal-only message vs the card.
func TestRunCardLifecycleAgainstHTTPTestFake(t *testing.T) {
	slack := &fakeSlack{}
	srv := httptest.NewServer(slack.handler())
	defer srv.Close()
	wire := slackwire.New("test-token", slackwire.WithAPIBase(srv.URL+"/"))

	res := Result{
		LoopID: "nightly-fix", RunID: "r-42", Command: "fix.ps1", ExitCode: 0,
		HeadBefore: "aaa1111", HeadAfter: "bbb2222", DurationMS: 65_000,
		Commits: []string{"bbb2222 fix(gateway): close the tick race #2263 (fak gateway)"},
	}

	// BEFORE (legacy consumer): the terminal-only post — the channel never sees
	// the run start, and every run appends a new message.
	if _, err := wire.PostMessage(context.Background(), "C-legacy", res.Text(), res.Blocks(), ""); err != nil {
		t.Fatal(err)
	}
	before := slack.transcript("C-legacy")

	// AFTER (migrated consumer): the run card.
	dir := t.TempDir()
	rc, err := OpenRunCard(dir, res.LoopID, res.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if err := rc.Start("C-dispatch", res); err != nil {
		t.Fatal(err)
	}
	drainOnce(t, rc, wire)
	if got := slack.topLevelCount("C-dispatch"); got != 1 {
		t.Fatalf("start must post exactly one card, got %d", got)
	}
	running := slack.transcript("C-dispatch")
	if !strings.Contains(running, "running `fix.ps1`") {
		t.Fatalf("start banner missing:\n%s", running)
	}

	// Process restart mid-run: a fresh RunCard over the same spool + state must
	// resume the SAME card, not post a second one.
	rc2, err := OpenRunCard(dir, res.LoopID, res.RunID)
	if err != nil {
		t.Fatal(err)
	}
	if err := rc2.Start("C-dispatch", res); err != nil {
		t.Fatal(err)
	}
	drainOnce(t, rc2, wire)
	if got := slack.topLevelCount("C-dispatch"); got != 1 {
		t.Fatalf("restart posted a second card (%d top-level messages)", got)
	}

	// Run ends: final witness edit in place + full result body in the thread.
	if err := rc2.Finalize(res); err != nil {
		t.Fatal(err)
	}
	drainOnce(t, rc2, wire)

	after := slack.transcript("C-dispatch")
	if got := slack.topLevelCount("C-dispatch"); got != 1 {
		t.Fatalf("channel must stay one line per run, got %d:\n%s", got, after)
	}
	wantFinal := "dispatch nightly-fix · run `r-42` — " + CardWitness(res).FinalText()
	if !strings.Contains(after, wantFinal) {
		t.Fatalf("final card line must be the witness fold\nwant contains: %s\ngot:\n%s", wantFinal, after)
	}
	if !strings.Contains(after, "SHIPPED · commit bbb2222 · (fak gateway) · verify=grep · exit=0") {
		t.Fatalf("witnessed verdict missing:\n%s", after)
	}
	if !strings.Contains(after, "↳") || !strings.Contains(after, "dispatch result nightly-fix") {
		t.Fatalf("result detail must ride in the thread:\n%s", after)
	}

	// The captured before/after transcript — the migration's channel-shape witness.
	t.Logf("BEFORE (legacy terminal-only post):\n%s\nDURING (live card):\n%s\nAFTER (final witness edit, same message):\n%s",
		before, running, after)
}

// drainOnce drains the card's outbox through the real wire, skipping pacing sleeps.
func drainOnce(t *testing.T, rc *RunCard, wire *slackwire.Client) {
	t.Helper()
	_, err := rc.Outbox.Drain(context.Background(), wire, slackoutbox.DrainOpts{
		Sleep: func(ctx context.Context, _ time.Duration) error { return ctx.Err() },
	})
	if err != nil {
		t.Fatal(err)
	}
}
