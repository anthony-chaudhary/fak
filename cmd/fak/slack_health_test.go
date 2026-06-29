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

	"github.com/anthony-chaudhary/fak/internal/chatrelay"
)

// healthByName finds one surface's folded health row.
func healthByName(rs []healthReport, name string) *healthReport {
	for i := range rs {
		if rs[i].Name == name {
			return &rs[i]
		}
	}
	return nil
}

// slackTS renders a time as a Slack message ts ("seconds.micro").
func slackTS(at time.Time) string {
	return fmt.Sprintf("%d.000100", at.Unix())
}

// healthHub is a single httptest server that plays BOTH Slack endpoints the health verb
// calls: auth.test (always OK here) and conversations.history (returns a single message
// whose ts the test controls per channel). It is the witness that the staleness verdict
// comes from a REAL conversations.history read, not a self-report.
type healthHub struct {
	srv         *httptest.Server
	historyTS   map[string]string // channel -> ts of its newest message ("" => empty channel)
	historyErr  map[string]string // channel -> slack error code (e.g. "not_in_channel")
	authOK      bool
	historyHits int
}

func newHealthHub(t *testing.T) *healthHub {
	t.Helper()
	h := &healthHub{historyTS: map[string]string{}, historyErr: map[string]string{}, authOK: true}
	h.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch {
		case strings.HasSuffix(r.URL.Path, "auth.test"):
			if h.authOK {
				_, _ = w.Write([]byte(`{"ok":true,"team":"Acme","user":"fakbot"}`))
			} else {
				_, _ = w.Write([]byte(`{"ok":false,"error":"invalid_auth"}`))
			}
		case strings.HasSuffix(r.URL.Path, "conversations.history"):
			h.historyHits++
			ch := r.URL.Query().Get("channel")
			if code, bad := h.historyErr[ch]; bad {
				_, _ = w.Write([]byte(`{"ok":false,"error":"` + code + `"}`))
				return
			}
			ts := h.historyTS[ch]
			if ts == "" { // empty channel: ok, but no messages
				_, _ = w.Write([]byte(`{"ok":true,"messages":[]}`))
				return
			}
			_, _ = fmt.Fprintf(w, `{"ok":true,"messages":[{"type":"message","ts":%q,"text":"card"}]}`, ts)
		default:
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
		}
	}))
	t.Cleanup(h.srv.Close)
	return h
}

func (h *healthHub) base() string { return h.srv.URL + "/" }

// TestSlackHealthFreshVsStale is the core acceptance witness: with the same auth and the
// same conversations.history transport, a freshly-posted channel grades OK and a long-quiet
// channel grades STALE — and the staleness verdict is driven by a real history read.
func TestSlackHealthFreshVsStale(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE") // daily cadence => 36h budget
	t.Setenv("FAK_PRODUCT_CHANNEL", "C0PROD")     // weekly cadence => 8d budget

	hub := newHealthHub(t)
	now := time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)
	// scoreboard: posted 3h ago => fresh against the 36h budget.
	hub.historyTS["C0SCORE"] = slackTS(now.Add(-3 * time.Hour))
	// product: posted 10 days ago => stale against the 8d budget.
	hub.historyTS["C0PROD"] = slackTS(now.Add(-10 * 24 * time.Hour))

	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := foldSlackHealth(reports, hub.base(), now)

	if hub.historyHits == 0 {
		t.Fatal("staleness must be witnessed by a real conversations.history read; server saw none")
	}

	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictOK {
		t.Fatalf("fresh scoreboard should be OK: %+v", sb)
	}
	if sb.LastPostAgeS < int64((2*time.Hour)/time.Second) || sb.LastPostAgeS > int64((4*time.Hour)/time.Second) {
		t.Fatalf("scoreboard age should be ~3h, got %ds", sb.LastPostAgeS)
	}
	if sb.BudgetS != int64((36*time.Hour)/time.Second) {
		t.Fatalf("scoreboard budget should be 36h, got %ds", sb.BudgetS)
	}

	pr := healthByName(health, "product")
	if pr == nil || pr.Verdict != verdictStale {
		t.Fatalf("quiet product should be STALE: %+v", pr)
	}
	if !strings.Contains(pr.Detail, "exceeds") {
		t.Fatalf("stale detail should explain the age vs budget: %q", pr.Detail)
	}
}

// TestSlackHealthEmptyChannelIsStale: a ready, authed channel with NO posts cannot witness a
// recent post, so it grades STALE (the silent-dead-feeder alarm).
func TestSlackHealthEmptyChannelIsStale(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0EMPTY")

	hub := newHealthHub(t)
	hub.historyTS["C0EMPTY"] = "" // empty channel

	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := foldSlackHealth(reports, hub.base(), time.Now())

	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictStale {
		t.Fatalf("empty channel should be STALE: %+v", sb)
	}
	if sb.LastPostAgeS != -1 {
		t.Fatalf("unwitnessed channel should report age -1, got %d", sb.LastPostAgeS)
	}
}

// TestSlackHealthHistoryErrorIsStale: a config error on the read (bot not in channel) is a
// real silent-failure mode and must grade STALE with the cause preserved, not crash.
func TestSlackHealthHistoryErrorIsStale(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0NOTMEMBER")

	hub := newHealthHub(t)
	hub.historyErr["C0NOTMEMBER"] = "not_in_channel"

	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := foldSlackHealth(reports, hub.base(), time.Now())

	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictStale {
		t.Fatalf("history error should grade STALE: %+v", sb)
	}
	if !strings.Contains(sb.Detail, "not_in_channel") {
		t.Fatalf("stale detail should preserve the read error: %q", sb.Detail)
	}
}

// TestSlackHealthIncompleteAndAuthFail: an unresolved surface is INCOMPLETE (never probed),
// and a rejected token is AUTH_FAIL — both before any history read.
func TestSlackHealthIncompleteAndAuthFail(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-bad")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")

	hub := newHealthHub(t)
	hub.authOK = false // token rejected

	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := foldSlackHealth(reports, hub.base(), time.Now())

	// scoreboard: token resolves but auth.test rejects it => AUTH_FAIL.
	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictAuthFail {
		t.Fatalf("rejected token should be AUTH_FAIL: %+v", sb)
	}
	if hub.historyHits != 0 {
		t.Fatalf("auth-failed surfaces must not be probed for history; saw %d reads", hub.historyHits)
	}

	// marketing: no channel default and none set => INCOMPLETE regardless of the token.
	mk := healthByName(health, "marketing")
	if mk == nil || mk.Verdict != verdictIncomplete {
		t.Fatalf("unresolved marketing should be INCOMPLETE: %+v", mk)
	}
}

// TestSlackHealthOnDemandSurfaceIsOK: a ready, authed surface with NO scheduled feeder
// (no cadence) is OK without any staleness probe — quiet is only judged where a feeder is
// supposed to be loud.
func TestSlackHealthOnDemandSurfaceIsOK(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_MARKETING_CHANNEL", "C0MKT") // marketing has no cadence budget

	hub := newHealthHub(t)
	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := foldSlackHealth(reports, hub.base(), time.Now())

	mk := healthByName(health, "marketing")
	if mk == nil || mk.Verdict != verdictOK {
		t.Fatalf("ready on-demand marketing should be OK: %+v", mk)
	}
	if mk.BudgetS != 0 {
		t.Fatalf("on-demand surface should have no budget, got %ds", mk.BudgetS)
	}
}

// TestSlackHealthExitNonZeroOnNonOK: the verb exits non-zero whenever any surface is non-OK
// (the gate the watchdog reads), and the JSON carries the acceptance contract fields.
func TestSlackHealthExitNonZeroOnNonOK(t *testing.T) {
	clearSlackEnv(t) // everything unset => every surface INCOMPLETE => exit 1
	var out, errb bytes.Buffer
	code := runSlackHealth(&out, &errb, []string{"--json"})
	if code != 1 {
		t.Fatalf("all-incomplete health must exit 1, got %d", code)
	}
	var health []healthReport
	if err := json.Unmarshal(out.Bytes(), &health); err != nil {
		t.Fatalf("decode health json: %v\n%s", err, out.String())
	}
	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictIncomplete || sb.Ready {
		t.Fatalf("scoreboard should be INCOMPLETE in a cleared env: %+v", sb)
	}
}

// healthExit is pure over the verdict set: any non-OK trips the gate.
func TestHealthExit(t *testing.T) {
	allOK := []healthReport{{Verdict: verdictOK}, {Verdict: verdictOK}}
	if healthExit(allOK) != 0 {
		t.Fatal("all OK should exit 0")
	}
	withStale := []healthReport{{Verdict: verdictOK}, {Verdict: verdictStale}}
	if healthExit(withStale) != 1 {
		t.Fatal("a STALE surface should exit 1")
	}
}

// compile-time check that the live reader satisfies the narrow probe interface.
var _ historyReader = (*chatrelay.HTTPSlack)(nil)
