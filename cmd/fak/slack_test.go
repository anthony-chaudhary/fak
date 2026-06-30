package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// clearSlackEnv blanks every Slack env key the surface table consults so a test runs in a
// deterministic environment regardless of what the dev box has exported.
func clearSlackEnv(t *testing.T) {
	t.Helper()
	keys := []string{
		"FAK_SCOREBOARD_TOKEN", "FAK_SCOREBOARD_CHANNEL", "FAK_PRODUCT_CHANNEL",
		"FAK_BLOCKERS_TOKEN", "FAK_BLOCKERS_CHANNEL",
		"FAK_CACHEVALUE_TOKEN", "FAK_CACHEVALUE_CHANNEL",
		"FAK_GRAFANA_TOKEN", "FAK_GRAFANA_CHANNEL",
		"FAK_BENCH_TOKEN", "FAK_BENCH_CHANNEL",
		"FAK_DISPATCH_TOKEN", "FAK_DISPATCH_CHANNEL",
		"FAK_DOJO_TOKEN", "FAK_DOJO_CHANNEL",
		"FAK_BACKLOG_CHANNEL",
		"FAK_MARKETING_TOKEN", "FAK_MARKETING_CHANNEL",
		"FAK_NEWS_CHANNEL",
		"FAK_NODE_USAGE_TOKEN", "FAK_NODE_USAGE_CHANNEL",
		"FAK_STEERING_CHANNEL",
		"FAK_CHATRELAY_TOKEN", "FAK_CHATRELAY_CHANNEL",
	}
	for _, k := range keys {
		t.Setenv(k, "")
	}
	// A temp cwd with no .env.slack.local removes the file fallback.
	t.Chdir(t.TempDir())
}

func reportByName(reports []*surfaceReport, name string) *surfaceReport {
	for _, r := range reports {
		if r.Name == name {
			return r
		}
	}
	return nil
}

func TestBuildSurfaceReportsScoreboardFallback(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-sb-token")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")

	reports := buildSurfaceReports()

	sb := reportByName(reports, "scoreboard")
	if !sb.TokenSet || sb.Channel != "C0SCORE" || !sb.Ready {
		t.Fatalf("scoreboard: %+v", sb)
	}
	if sb.SignalNoise.Signal == 0 || sb.SignalNoise.Noise == 0 {
		t.Fatalf("scoreboard missing S/N self-score: %+v", sb.SignalNoise)
	}
	if sb.Token == "bottok-sb-token" {
		t.Fatalf("token must be redacted in the report, got raw %q", sb.Token)
	}

	// bench has no own token set => it must fall back to the scoreboard token, and it now has
	// a public channel default (#1428) => channel resolves and the surface is ready.
	bench := reportByName(reports, "bench")
	if !bench.TokenSet || !strings.Contains(bench.TokenSource, "scoreboard-fallback") {
		t.Fatalf("bench should fall back to scoreboard token: %+v", bench)
	}
	if bench.Channel == "" || bench.ChannelSource != "built-in default" || !bench.Ready {
		t.Fatalf("bench should use its built-in channel default => ready: %+v", bench)
	}

	// blockers has a public channel default => channel resolves even with nothing set.
	blk := reportByName(reports, "blockers")
	if blk.Channel == "" || blk.ChannelSource != "built-in default" {
		t.Fatalf("blockers should use its built-in channel default: %+v", blk)
	}
}

func TestBuildSurfaceReportsBacklogUsesScoreboardTokenAndChannelVar(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-sb")
	t.Setenv("FAK_BACKLOG_CHANNEL", "C0BACKLOG")

	reports := buildSurfaceReports()
	bl := reportByName(reports, "backlog")
	if bl == nil {
		t.Fatal("backlog surface must be registered")
	}
	if !bl.TokenSet || !strings.Contains(bl.TokenSource, "scoreboard-fallback") {
		t.Fatalf("backlog should fall back to scoreboard token: %+v", bl)
	}
	if bl.Channel != "C0BACKLOG" || !strings.Contains(bl.ChannelSource, "FAK_BACKLOG_CHANNEL") || !bl.Ready {
		t.Fatalf("backlog should resolve FAK_BACKLOG_CHANNEL and be ready: %+v", bl)
	}
}

func TestBuildSurfaceReportsNewsUsesScoreboardTokenAndChannelVar(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-sb")
	t.Setenv("FAK_NEWS_CHANNEL", "C0NEWS")

	reports := buildSurfaceReports()
	news := reportByName(reports, "news")
	if news == nil {
		t.Fatal("news surface must be registered")
	}
	if !news.TokenSet || !strings.Contains(news.TokenSource, "scoreboard-fallback") {
		t.Fatalf("news should fall back to scoreboard token: %+v", news)
	}
	if news.Channel != "C0NEWS" || !strings.Contains(news.ChannelSource, "FAK_NEWS_CHANNEL") || !news.Ready {
		t.Fatalf("news should resolve FAK_NEWS_CHANNEL and be ready: %+v", news)
	}
}

func TestBuildSurfaceReportsOwnTokenWins(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-sb")
	t.Setenv("FAK_BLOCKERS_TOKEN", "bottok-blockers-own")

	reports := buildSurfaceReports()
	blk := reportByName(reports, "blockers")
	if !strings.Contains(blk.TokenSource, "FAK_BLOCKERS_TOKEN") {
		t.Fatalf("blockers should prefer its OWN token: %+v", blk)
	}
	if strings.Contains(blk.TokenSource, "scoreboard-fallback") {
		t.Fatalf("blockers own token set => must not report fallback: %+v", blk)
	}
}

func TestBuildSurfaceReportsAllUnset(t *testing.T) {
	clearSlackEnv(t)
	reports := buildSurfaceReports()
	for _, r := range reports {
		// Surfaces with a built-in channel default still show a channel, but never a token.
		if r.TokenSet {
			t.Fatalf("surface %s should have no token in a cleared env: %+v", r.Name, r)
		}
		if r.Ready {
			t.Fatalf("surface %s cannot be ready with no token: %+v", r.Name, r)
		}
	}
}

func TestRedactToken(t *testing.T) {
	cases := map[string]string{
		"":                "(unset)",
		"abc":             "****",
		"bottok-secret99": "****et99",
	}
	for in, want := range cases {
		if got := redactToken(in); got != want {
			t.Fatalf("redactToken(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestSlackSendDryRun(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-default-tok")
	var out, errb bytes.Buffer
	code := runSlackSend(&out, &errb, []string{"--channel", "C0ABC", "--text", "hello", "--dry-run"})
	if code != 0 {
		t.Fatalf("dry-run exit = %d, stderr=%s", code, errb.String())
	}
	s := out.String()
	if !strings.Contains(s, "C0ABC") || !strings.Contains(s, "hello") {
		t.Fatalf("dry-run output missing channel/text: %s", s)
	}
	if strings.Contains(s, "bottok-default-tok") {
		t.Fatalf("dry-run must not print the raw token: %s", s)
	}
}

func TestSlackSendRequiresChannelAndText(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-tok")
	var out, errb bytes.Buffer
	if code := runSlackSend(&out, &errb, []string{"--text", "hi", "--dry-run"}); code != 2 {
		t.Fatalf("missing --channel should exit 2, got %d", code)
	}
	out.Reset()
	errb.Reset()
	if code := runSlackSend(&out, &errb, []string{"--channel", "C0ABC", "--dry-run"}); code != 2 {
		t.Fatalf("missing --text should exit 2, got %d", code)
	}
}

func TestSlackSendLive(t *testing.T) {
	clearSlackEnv(t)
	var gotChannel, gotText, gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "chat.postMessage") {
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
			return
		}
		gotAuth = r.Header.Get("Authorization")
		var body struct {
			Channel string `json:"channel"`
			Text    string `json:"text"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotChannel, gotText = body.Channel, body.Text
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true,"ts":"1719600000.000100"}`))
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	code := runSlackSend(&out, &errb, []string{
		"--channel", "C0LIVE", "--text", "shipped", "--token", "bottok-live", "--api-base", srv.URL + "/",
	})
	if code != 0 {
		t.Fatalf("live send exit = %d, stderr=%s", code, errb.String())
	}
	if gotChannel != "C0LIVE" || gotText != "shipped" {
		t.Fatalf("server saw channel=%q text=%q", gotChannel, gotText)
	}
	if gotAuth != "Bearer bottok-live" {
		t.Fatalf("server saw auth header %q", gotAuth)
	}
	if !strings.Contains(out.String(), "1719600000.000100") {
		t.Fatalf("send output missing posted ts: %s", out.String())
	}
}

func TestSlackCheckAuthOK(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")

	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "auth.test") {
			calls++
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"ok":true,"team":"Acme","user":"fakbot"}`))
			return
		}
		http.Error(w, "unexpected", http.StatusNotFound)
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	code := runSlackCheck(&out, &errb, []string{"--auth", "--json", "--api-base", srv.URL + "/"})
	if code != 0 {
		t.Fatalf("check --auth (ok) exit = %d, stderr=%s", code, errb.String())
	}
	// Every surface shares the one scoreboard token => auth.test is called exactly once.
	if calls != 1 {
		t.Fatalf("auth.test should be called once per distinct token, got %d", calls)
	}
	var reports []*surfaceReport
	if err := json.Unmarshal(out.Bytes(), &reports); err != nil {
		t.Fatalf("decode json report: %v\n%s", err, out.String())
	}
	sb := reportByName(reports, "scoreboard")
	if sb.Auth == nil || !sb.Auth.OK || sb.Auth.Team != "Acme" || sb.Auth.User != "fakbot" {
		t.Fatalf("scoreboard auth not reported OK: %+v", sb.Auth)
	}
}

func TestSlackCheckAuthFailExitsNonZero(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-bad")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":false,"error":"invalid_auth"}`))
	}))
	defer srv.Close()

	var out, errb bytes.Buffer
	code := runSlackCheck(&out, &errb, []string{"--auth", "--api-base", srv.URL + "/"})
	if code != 1 {
		t.Fatalf("check --auth with a failing token must exit 1, got %d\n%s", code, out.String())
	}
	if !strings.Contains(out.String(), "invalid_auth") {
		t.Fatalf("expected the auth error surfaced in output: %s", out.String())
	}
}

func TestSlackCheckOfflineExitsZero(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	if code := runSlackCheck(&out, &errb, nil); code != 0 {
		t.Fatalf("offline check should always exit 0, got %d", code)
	}
	if !strings.Contains(out.String(), "scoreboard") {
		t.Fatalf("offline check should list surfaces: %s", out.String())
	}
	if !strings.Contains(out.String(), "S/N self-score") {
		t.Fatalf("offline check should carry S/N metadata: %s", out.String())
	}
}
