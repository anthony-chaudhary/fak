package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchpost"
)

// loopCardSlack is a minimal httptest Slack Web API for the `fak loop run`
// card path: chat.postMessage + chat.update + conversations.history.
type loopCardSlack struct {
	mu     sync.Mutex
	nextTS int
	msgs   []loopCardMsg
}

type loopCardMsg struct {
	Channel  string
	TS       string
	ThreadTS string
	Text     string
}

func (f *loopCardSlack) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/chat.postMessage", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Channel  string `json:"channel"`
			Text     string `json:"text"`
			ThreadTS string `json:"thread_ts"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		f.mu.Lock()
		f.nextTS++
		ts := fmt.Sprintf("%d.0", f.nextTS)
		f.msgs = append(f.msgs, loopCardMsg{Channel: body.Channel, TS: ts, ThreadTS: body.ThreadTS, Text: body.Text})
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
		for i := range f.msgs {
			if f.msgs[i].TS == body.TS && f.msgs[i].Channel == body.Channel {
				f.msgs[i].Text = body.Text
			}
		}
		f.mu.Unlock()
		fmt.Fprint(w, `{"ok":true}`)
	})
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		msgs := make([]map[string]any, 0, len(f.msgs))
		for _, m := range f.msgs {
			msgs = append(msgs, map[string]any{"type": "message", "ts": m.TS, "text": m.Text, "thread_ts": m.ThreadTS})
		}
		f.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"ok": true, "messages": msgs})
	})
	return mux
}

func (f *loopCardSlack) topLevel(channel string) []loopCardMsg {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []loopCardMsg
	for _, m := range f.msgs {
		if m.Channel == channel && m.ThreadTS == "" {
			out = append(out, m)
		}
	}
	return out
}

func (f *loopCardSlack) replies(channel, ts string) []loopCardMsg {
	f.mu.Lock()
	defer f.mu.Unlock()
	var out []loopCardMsg
	for _, m := range f.msgs {
		if m.Channel == channel && m.ThreadTS == ts {
			out = append(out, m)
		}
	}
	return out
}

// TestLoopRunCardPostsOnceAndFinalizesInPlace drives the REAL `fak loop run`
// Slack path (openDispatchRunCard -> postDispatchResult) against an httptest
// fake: the dispatch channel gets ONE card at run start, and the run's end
// EDITS that same message into the witness fold — with the full result body
// threaded under it — instead of appending a terminal message (#2263).
func TestLoopRunCardPostsOnceAndFinalizesInPlace(t *testing.T) {
	slack := &loopCardSlack{}
	srv := httptest.NewServer(slack.handler())
	defer srv.Close()
	oldBase := dispatchAPIBase
	dispatchAPIBase = srv.URL + "/"
	defer func() { dispatchAPIBase = oldBase }()
	t.Setenv("FAK_SLACK_OUTBOX_DIR", filepath.Join(t.TempDir(), "outbox"))

	var stderr bytes.Buffer
	res := dispatchpost.Result{LoopID: "nightly", RunID: "r-7", Command: "job.ps1"}
	card := openDispatchRunCard(&stderr, "C7", "test-token", res)
	if card == nil {
		t.Fatalf("card did not arm: %s", stderr.String())
	}
	top := slack.topLevel("C7")
	if len(top) != 1 || !strings.Contains(top[0].Text, "running `job.ps1`") {
		t.Fatalf("run start must post one card: %+v (%s)", top, stderr.String())
	}

	// Run ends: equal HEADs, exit 0 — an honest check-result, nothing landed.
	res.ExitCode = 0
	res.HeadBefore, res.HeadAfter = "abc1234", "abc1234"
	postDispatchResult(&stderr, false, "C7", "test-token", card, res)

	top = slack.topLevel("C7")
	if len(top) != 1 {
		t.Fatalf("channel must stay one line per run, got %d: %+v (%s)", len(top), top, stderr.String())
	}
	if !strings.Contains(top[0].Text, "dispatch nightly · run `r-7` — NOT_SHIPPED") ||
		!strings.Contains(top[0].Text, "verify=none · exit=0") {
		t.Fatalf("final edit must carry the witness fold: %q", top[0].Text)
	}
	reps := slack.replies("C7", top[0].TS)
	if len(reps) != 1 || !strings.Contains(reps[0].Text, "dispatch result nightly") {
		t.Fatalf("result body must ride in the thread: %+v", reps)
	}
	if !strings.Contains(stderr.String(), "dispatch run card finalized") {
		t.Fatalf("card path did not report success: %s", stderr.String())
	}
}

// TestLoopRunCardUnarmedWithoutChannel keeps the unconfigured-box behavior:
// no dispatch channel means no card and no error.
func TestLoopRunCardUnarmedWithoutChannel(t *testing.T) {
	t.Setenv("FAK_DISPATCH_CHANNEL", "")
	t.Setenv("FAK_SLACK_OUTBOX_DIR", filepath.Join(t.TempDir(), "outbox"))
	// Channel resolution falls back to a .env.slack.local walked up from the
	// cwd; run from an empty dir so a configured dev box stays hermetic.
	t.Chdir(t.TempDir())
	var stderr bytes.Buffer
	if card := openDispatchRunCard(&stderr, "", "", dispatchpost.Result{LoopID: "l", RunID: "r"}); card != nil {
		t.Fatalf("card must stay unarmed without a channel (stderr: %s)", stderr.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("unarmed card must be silent: %s", stderr.String())
	}
}
