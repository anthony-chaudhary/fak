package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// beatHub is the in-memory Slack the beat verb talks to: auth.test (always OK), a per-channel
// conversations.history (so foldSlackHealth's staleness probe is exercised), AND chat.postMessage
// (the beat's own delivery). It is the witness that the beat reaches a real post endpoint, not a
// self-report. Mirrors slack_health_test.go's healthHub plus the post leg.
type beatHub struct {
	srv       *httptest.Server
	historyTS map[string]string // channel -> ts of its newest message ("" => empty channel)
	posted    bool              // set once chat.postMessage is hit
	lastText  string            // the body the beat posted
}

func newBeatHub(t *testing.T) *beatHub {
	t.Helper()
	h := &beatHub{historyTS: map[string]string{}}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, "auth.test"):
			_, _ = w.Write([]byte(`{"ok":true,"team":"Acme","user":"fakbot"}`))
		case strings.HasSuffix(r.URL.Path, "conversations.history"):
			ch := r.URL.Query().Get("channel")
			ts := h.historyTS[ch]
			if ts == "" {
				_, _ = w.Write([]byte(`{"ok":true,"messages":[]}`))
				return
			}
			_, _ = fmt.Fprintf(w, `{"ok":true,"messages":[{"type":"message","ts":%q,"text":"card"}]}`, ts)
		case strings.HasSuffix(r.URL.Path, "chat.postMessage"):
			h.posted = true
			var body struct {
				Channel string `json:"channel"`
				Text    string `json:"text"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			h.lastText = body.Text
			_, _ = fmt.Fprintf(w, `{"ok":true,"channel":%q,"ts":"1782000000.000100"}`, body.Channel)
		default:
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
		}
	}))
	t.Cleanup(h.srv.Close)
	return h
}

func (h *beatHub) base() string { return h.srv.URL + "/" }

// TestBeatLineGreenVsDown is the core acceptance witness for the liveness beat: on an all-OK
// fold the beat is a single green "alive" line with the OK count; with a down surface the beat
// names it and its mode. This is the whole point of #1426 — a quiet day still produces a
// provably-alive line, and a broken surface is announced in-channel rather than only as a
// once-a-day GH issue.
func TestBeatLineGreenVsDown(t *testing.T) {
	now := time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

	green := []healthReport{
		{Name: "scoreboard", Verdict: verdictOK, LastPostAgeS: 3 * 3600},
		{Name: "dojo", Verdict: verdictOK, LastPostAgeS: 6 * 3600},
		{Name: "dispatch", Verdict: verdictOK, LastPostAgeS: -1}, // on-demand: no age
	}
	line := beatLine(green, now)
	if !strings.HasPrefix(line, "✅") {
		t.Fatalf("all-OK beat must lead with the green glyph, got %q", line)
	}
	if !strings.Contains(line, "3/3 OK") {
		t.Fatalf("green beat must report the OK count 3/3, got %q", line)
	}
	if strings.Contains(line, "down:") {
		t.Fatalf("an all-OK beat must NOT carry a down clause, got %q", line)
	}
	// The freshest age across surfaces (3h) must be the one reported, not 6h.
	if !strings.Contains(line, "3h0m0s ago") {
		t.Fatalf("green beat must report the FRESHEST feeder age (3h), got %q", line)
	}

	down := []healthReport{
		{Name: "scoreboard", Verdict: verdictOK, LastPostAgeS: 3 * 3600},
		{Name: "cachevalue", Verdict: verdictStale, LastPostAgeS: -1},
		{Name: "bench", Verdict: verdictIncomplete, LastPostAgeS: -1},
	}
	dl := beatLine(down, now)
	if strings.HasPrefix(dl, "✅") {
		t.Fatalf("a beat with a down surface must NOT be green, got %q", dl)
	}
	if !strings.Contains(dl, "1/3 OK") {
		t.Fatalf("down beat must report 1/3 OK, got %q", dl)
	}
	for _, want := range []string{"down:", "cachevalue STALE", "bench INCOMPLETE"} {
		if !strings.Contains(dl, want) {
			t.Fatalf("down beat must name %q, got %q", want, dl)
		}
	}
}

// TestBeatGlyphAuthFailIsLoudest checks the severity ladder: an auth wall (the bot token itself
// rejected) is the loudest signal and gets the red glyph, while staleness/config drift gets the
// warning glyph.
func TestBeatGlyphAuthFailIsLoudest(t *testing.T) {
	now := time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)
	authWall := []healthReport{
		{Name: "scoreboard", Verdict: verdictAuthFail, LastPostAgeS: -1},
		{Name: "bench", Verdict: verdictIncomplete, LastPostAgeS: -1},
	}
	if got := beatLine(authWall, now); !strings.HasPrefix(got, "🔴") {
		t.Fatalf("an AUTH_FAIL surface must make the beat red, got %q", got)
	}
	warnOnly := []healthReport{
		{Name: "cachevalue", Verdict: verdictStale, LastPostAgeS: -1},
	}
	if got := beatLine(warnOnly, now); !strings.HasPrefix(got, "⚠️") {
		t.Fatalf("a STALE-only surface must make the beat a warning, got %q", got)
	}
}

// TestRunSlackBeatPostsLive drives the whole verb against the in-memory Slack hub (auth.test +
// conversations.history + chat.postMessage) and asserts the beat actually POSTS — the liveness
// guarantee end to end. The post lands even though every surface is OK (nothing to "update"),
// which is exactly the behavior a quiet day needs.
func TestRunSlackBeatPostsLive(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")
	t.Setenv("FAK_DISPATCH_CHANNEL", "C0DISP") // the beat's preferred status channel

	hub := newBeatHub(t)
	hub.historyTS["C0SCORE"] = slackTS(time.Now().Add(-3 * time.Hour))

	var out, errBuf bytes.Buffer
	code := runSlackBeat(&out, &errBuf, []string{"--json", "--api-base", hub.base()})
	if code != 0 {
		t.Fatalf("beat should exit 0 on a successful post, got %d (stderr=%s)", code, errBuf.String())
	}

	var res beatResult
	if err := json.Unmarshal(out.Bytes(), &res); err != nil {
		t.Fatalf("decode beat json: %v\nout=%s", err, out.String())
	}
	if !res.Posted {
		t.Fatalf("beat must POST on a quiet day (the liveness guarantee), got %+v", res)
	}
	if res.Channel != "C0DISP" {
		t.Fatalf("beat must prefer the dispatch status channel, got %q", res.Channel)
	}
	if res.TS == "" {
		t.Fatal("a posted beat must carry the message ts")
	}
	if !hub.posted {
		t.Fatal("beat must reach chat.postMessage; the hub saw no post")
	}
}

// TestRunSlackBeatDryRunPostsNothing confirms the fork-safe path: --dry-run resolves and renders
// but never reaches chat.postMessage, and still exits 0 (a dry-run is success, not a misconfig).
func TestRunSlackBeatDryRunPostsNothing(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_DISPATCH_CHANNEL", "C0DISP")

	hub := newBeatHub(t)
	var out, errBuf bytes.Buffer
	code := runSlackBeat(&out, &errBuf, []string{"--dry-run", "--api-base", hub.base()})
	if code != 0 {
		t.Fatalf("dry-run beat should exit 0, got %d", code)
	}
	if hub.posted {
		t.Fatal("dry-run must NOT post")
	}
	if !strings.Contains(out.String(), "dry-run") {
		t.Fatalf("dry-run output must say so, got %q", out.String())
	}
}

// TestRunSlackBeatSkipsWithoutChannel checks the misconfig path: no channel resolves anywhere =>
// skipped + exit 1, so a scheduled tick flags it rather than silently no-op'ing.
func TestRunSlackBeatSkipsWithoutChannel(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	// No channel env at all.

	hub := newBeatHub(t)
	var out, errBuf bytes.Buffer
	code := runSlackBeat(&out, &errBuf, []string{"--json", "--api-base", hub.base()})
	if code != 1 {
		t.Fatalf("a beat with no resolvable channel must exit 1, got %d", code)
	}
	var res beatResult
	_ = json.Unmarshal(out.Bytes(), &res)
	if res.Skipped == "" {
		t.Fatalf("missing channel must set a skipped reason, got %+v", res)
	}
	if hub.posted {
		t.Fatal("must not post when no channel resolves")
	}
}
