package chatrelay

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

// fakeHub is one httptest server that plays BOTH the Slack Web API (conversations.history,
// chat.postMessage) and a served OpenAI-compatible endpoint (/v1/chat/completions), so the
// relay is exercised over the real HTTP + JSON wire with no network and no live model.
type fakeHub struct {
	mu         sync.Mutex
	history    []map[string]any                  // canned channel messages, snake_case like Slack
	posted     []postedMsg                       // what the relay sent back
	prompts    []string                          // user prompts the model received
	nextTS     int                               // monotonic ts source for posted-message ids
	completeFn func(prompt string) (string, int) // model reply + HTTP status (200 default)
}

type postedMsg struct {
	Channel  string
	ThreadTS string
	Text     string
}

func newFakeHub() *fakeHub {
	return &fakeHub{nextTS: 1000, completeFn: func(p string) (string, int) { return "ECHO: " + p, 200 }}
}

func (h *fakeHub) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/conversations.history", func(w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		msgs := append([]map[string]any(nil), h.history...)
		h.mu.Unlock()
		// Slack returns newest-first; emulate that so the relay's own sort is what
		// guarantees oldest-first answering.
		rev := make([]map[string]any, 0, len(msgs))
		for i := len(msgs) - 1; i >= 0; i-- {
			rev = append(rev, msgs[i])
		}
		writeJSON(w, map[string]any{"ok": true, "messages": rev})
	})
	mux.HandleFunc("/chat.postMessage", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Channel  string `json:"channel"`
			ThreadTS string `json:"thread_ts"`
			Text     string `json:"text"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		h.mu.Lock()
		h.posted = append(h.posted, postedMsg{body.Channel, body.ThreadTS, body.Text})
		h.nextTS++
		ts := h.nextTS
		h.mu.Unlock()
		writeJSON(w, map[string]any{"ok": true, "ts": fmt.Sprintf("%d.000100", ts)})
	})
	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req chatRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		var userContent string
		for _, m := range req.Messages {
			if m.Role == "user" {
				userContent = m.Content
			}
		}
		h.mu.Lock()
		h.prompts = append(h.prompts, userContent)
		fn := h.completeFn
		h.mu.Unlock()
		reply, status := fn(userContent)
		if status != 200 {
			w.WriteHeader(status)
			writeJSON(w, map[string]any{"error": map[string]any{"message": reply}})
			return
		}
		writeJSON(w, map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": reply}}},
		})
	})
	return mux
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v)
}

func (h *fakeHub) addHistory(m map[string]any) {
	h.mu.Lock()
	h.history = append(h.history, m)
	h.mu.Unlock()
}

func (h *fakeHub) postedCopy() []postedMsg {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]postedMsg(nil), h.posted...)
}

func (h *fakeHub) promptsCopy() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.prompts...)
}

// newRelay wires a Relay against the fakeHub server.
func newRelay(t *testing.T, srv *httptest.Server, channel string) *Relay {
	t.Helper()
	return &Relay{
		Slack:   &HTTPSlack{Token: "xoxb-test", APIBase: srv.URL + "/", HTTP: srv.Client()},
		Model:   &HTTPModel{Endpoint: srv.URL, Model: "glm-5.2", HTTP: srv.Client()},
		Channel: channel,
	}
}

func TestTickAnswersHumanSkipsBotAndDedups(t *testing.T) {
	hub := newFakeHub()
	srv := httptest.NewServer(hub.handler())
	defer srv.Close()

	// A prior bot reply (bot_id set) and a fresh human message.
	hub.addHistory(map[string]any{"type": "message", "ts": "1000.000100", "bot_id": "B01", "text": "an earlier bot reply"})
	hub.addHistory(map[string]any{"type": "message", "ts": "1001.000100", "user": "U_HUMAN", "text": "hello GLM"})

	r := newRelay(t, srv, "C_CHAT")
	n, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if n != 1 {
		t.Fatalf("handled = %d, want 1 (the human message only)", n)
	}
	posted := hub.postedCopy()
	if len(posted) != 1 {
		t.Fatalf("posted %d messages, want 1: %+v", len(posted), posted)
	}
	if posted[0].Channel != "C_CHAT" {
		t.Errorf("posted to channel %q, want C_CHAT", posted[0].Channel)
	}
	if posted[0].ThreadTS != "1001.000100" {
		t.Errorf("reply thread_ts = %q, want the human ts 1001.000100", posted[0].ThreadTS)
	}
	if posted[0].Text != "ECHO: hello GLM" {
		t.Errorf("reply text = %q, want %q", posted[0].Text, "ECHO: hello GLM")
	}

	// De-dup: a second Tick over the SAME history must not answer again.
	n2, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("second Tick: %v", err)
	}
	if n2 != 0 {
		t.Fatalf("second Tick handled = %d, want 0 (already answered)", n2)
	}
	if got := len(hub.postedCopy()); got != 1 {
		t.Fatalf("after dedup, total posts = %d, want 1", got)
	}

	// A NEW human message after the mark IS answered.
	hub.addHistory(map[string]any{"type": "message", "ts": "1002.000100", "user": "U_HUMAN", "text": "again"})
	n3, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("third Tick: %v", err)
	}
	if n3 != 1 {
		t.Fatalf("third Tick handled = %d, want 1 (the new message)", n3)
	}
}

func TestTickMentionGatingStripsToken(t *testing.T) {
	hub := newFakeHub()
	srv := httptest.NewServer(hub.handler())
	defer srv.Close()

	hub.addHistory(map[string]any{"type": "message", "ts": "1001.000100", "user": "U1", "text": "<@U07BOT> what is 2+2"})
	hub.addHistory(map[string]any{"type": "message", "ts": "1002.000100", "user": "U2", "text": "just chatting, not for the bot"})

	r := newRelay(t, srv, "C_CHAT")
	r.Mention = "<@U07BOT>"

	n, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if n != 1 {
		t.Fatalf("handled = %d, want 1 (only the addressed message)", n)
	}
	prompts := hub.promptsCopy()
	if len(prompts) != 1 {
		t.Fatalf("model saw %d prompts, want 1: %v", len(prompts), prompts)
	}
	if prompts[0] != "what is 2+2" {
		t.Errorf("prompt = %q, want the mention token stripped to %q", prompts[0], "what is 2+2")
	}
}

func TestTickSkipsOwnBotUser(t *testing.T) {
	hub := newFakeHub()
	srv := httptest.NewServer(hub.handler())
	defer srv.Close()

	// A message from our own bot user id must be skipped even if bot_id is absent.
	hub.addHistory(map[string]any{"type": "message", "ts": "1001.000100", "user": "U_SELF", "text": "echo of myself"})
	hub.addHistory(map[string]any{"type": "message", "ts": "1002.000100", "user": "U_HUMAN", "text": "real question"})

	r := newRelay(t, srv, "C_CHAT")
	r.BotUserID = "U_SELF"

	n, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if n != 1 {
		t.Fatalf("handled = %d, want 1 (own message skipped)", n)
	}
	if got := hub.promptsCopy(); len(got) != 1 || got[0] != "real question" {
		t.Fatalf("model prompts = %v, want [\"real question\"]", got)
	}
}

func TestPrimeSkipsBacklog(t *testing.T) {
	hub := newFakeHub()
	srv := httptest.NewServer(hub.handler())
	defer srv.Close()

	hub.addHistory(map[string]any{"type": "message", "ts": "1001.000100", "user": "U", "text": "old 1"})
	hub.addHistory(map[string]any{"type": "message", "ts": "1002.000100", "user": "U", "text": "old 2"})

	r := newRelay(t, srv, "C_CHAT")
	if err := r.Prime(context.Background()); err != nil {
		t.Fatalf("Prime: %v", err)
	}
	n, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if n != 0 {
		t.Fatalf("after Prime, handled = %d, want 0 (backlog skipped)", n)
	}
	hub.addHistory(map[string]any{"type": "message", "ts": "1003.000100", "user": "U", "text": "new one"})
	n2, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("Tick after new: %v", err)
	}
	if n2 != 1 {
		t.Fatalf("handled = %d, want 1 (only the post-Prime message)", n2)
	}
}

func TestHTTPSlackUsesSharedSlackwireErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"ok": false, "error": "channel_not_found"})
	}))
	defer srv.Close()

	slack := &HTTPSlack{Token: "xoxb-test", APIBase: srv.URL + "/", HTTP: srv.Client()}
	_, err := slack.History(context.Background(), "C_MISSING", "", 1)
	if err == nil {
		t.Fatal("History returned nil error, want Slack API error")
	}
	var apiErr *slackwire.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("History error = %T %v, want *slackwire.APIError", err, err)
	}
	if apiErr.Method != "conversations.history" || apiErr.Code != "channel_not_found" {
		t.Fatalf("API error fields = %+v", apiErr)
	}
}

func TestModelErrorDoesNotAdvanceMark(t *testing.T) {
	hub := newFakeHub()
	hub.completeFn = func(p string) (string, int) { return "boom", 500 }
	srv := httptest.NewServer(hub.handler())
	defer srv.Close()

	hub.addHistory(map[string]any{"type": "message", "ts": "1001.000100", "user": "U", "text": "q"})

	r := newRelay(t, srv, "C_CHAT")
	if _, err := r.Tick(context.Background()); err == nil {
		t.Fatalf("Tick: want an error from the failing model, got nil")
	}
	// The mark must NOT have advanced past the failed message: once the model recovers,
	// the same message is retried and answered.
	hub.completeFn = func(p string) (string, int) { return "OK: " + p, 200 }
	n, err := r.Tick(context.Background())
	if err != nil {
		t.Fatalf("retry Tick: %v", err)
	}
	if n != 1 {
		t.Fatalf("retry handled = %d, want 1 (the message was retried, not dropped)", n)
	}
}

func TestTSAfter(t *testing.T) {
	cases := []struct {
		a, b string
		want bool
	}{
		{"1001.000100", "1000.000100", true},
		{"1000.000100", "1001.000100", false},
		{"1000.000100", "1000.000100", false},
		{"1000000000.0", "999999999.9", true}, // numeric, not lexical
		{"5", "", true},
		{"", "5", false},
	}
	for _, c := range cases {
		if got := tsAfter(c.a, c.b); got != c.want {
			t.Errorf("tsAfter(%q,%q) = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestResolveChannelFromEnv(t *testing.T) {
	t.Setenv("FAK_CHATRELAY_CHANNEL", "C_FROM_ENV")
	if got := ResolveChannel(); got != "C_FROM_ENV" {
		t.Errorf("ResolveChannel() = %q, want C_FROM_ENV", got)
	}
	t.Setenv("FAK_CHATRELAY_TOKEN", "xoxb-from-env")
	if got := ResolveToken(); got != "xoxb-from-env" {
		t.Errorf("ResolveToken() = %q, want xoxb-from-env", got)
	}
}
