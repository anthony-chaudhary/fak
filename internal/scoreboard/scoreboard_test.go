package scoreboard

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func TestResolveTokenAndChannelFromEnv(t *testing.T) {
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-env-token")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_ENV")
	if got := ResolveToken(); got != "xoxb-env-token" {
		t.Fatalf("ResolveToken env = %q, want xoxb-env-token", got)
	}
	if got := ResolveChannel(); got != "C_ENV" {
		t.Fatalf("ResolveChannel env = %q, want C_ENV", got)
	}
}

func TestResolveFromEnvFileWhenEnvUnset(t *testing.T) {
	// Env vars unset -> the resolver must walk up to .env.slack.local.
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"export FAK_SCOREBOARD_TOKEN=xoxb-file-token\n" +
		"FAK_SCOREBOARD_CHANNEL=C_FILE\n" +
		"SLACK_BOT_TOKEN=xoxb-lab-token-must-not-leak\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	// run from a nested subdir to exercise the walk-up.
	sub := filepath.Join(dir, "a", "b")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	chdir(t, sub)

	if got := ResolveToken(); got != "xoxb-file-token" {
		t.Fatalf("ResolveToken file = %q, want xoxb-file-token", got)
	}
	if got := ResolveChannel(); got != "C_FILE" {
		t.Fatalf("ResolveChannel file = %q, want C_FILE", got)
	}
}

func TestResolveTokenNeverFallsBackToLabToken(t *testing.T) {
	// The whole point of the dedicated key: an unset scoreboard token must NOT
	// silently reuse the lab SLACK_BOT_TOKEN (that would cross workspaces).
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	t.Setenv("SLACK_BOT_TOKEN", "xoxb-lab-token")
	chdir(t, t.TempDir()) // no .env.slack.local here
	if got := ResolveToken(); got != "" {
		t.Fatalf("ResolveToken leaked the lab token: got %q, want empty", got)
	}
}

func TestNewClientRequiresToken(t *testing.T) {
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	chdir(t, t.TempDir())
	if _, err := NewClient(""); err == nil {
		t.Fatal("NewClient with no token should error")
	}
}

func TestPostSendsChannelTextAndBlocks(t *testing.T) {
	var gotBody map[string]any
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "chat.postMessage") {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"ts":"123.456","channel":"C1"}`)
	}))
	defer srv.Close()

	c, err := NewClient("xoxb-test", WithAPIBase(srv.URL+"/"), WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatal(err)
	}
	ts, err := c.Post(context.Background(), "C1", "hello", []any{map[string]any{"type": "section"}})
	if err != nil {
		t.Fatal(err)
	}
	if ts != "123.456" {
		t.Fatalf("ts = %q, want 123.456", ts)
	}
	if gotAuth != "Bearer xoxb-test" {
		t.Fatalf("auth = %q, want Bearer xoxb-test", gotAuth)
	}
	if gotBody["channel"] != "C1" || gotBody["text"] != "hello" {
		t.Fatalf("body channel/text wrong: %+v", gotBody)
	}
	if _, ok := gotBody["blocks"]; !ok {
		t.Fatalf("blocks not sent: %+v", gotBody)
	}
}

func TestPostSurfacesSlackError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":false,"error":"channel_not_found"}`)
	}))
	defer srv.Close()
	c, _ := NewClient("xoxb-test", WithAPIBase(srv.URL+"/"), WithHTTPClient(srv.Client()))
	_, err := c.Post(context.Background(), "CBAD", "hi", nil)
	if err == nil || !strings.Contains(err.Error(), "channel_not_found") {
		t.Fatalf("want channel_not_found error, got %v", err)
	}
}

func TestFromPayloadFoldsGradeScoreDebtAndKPIs(t *testing.T) {
	p := scorecard.Payload{
		Schema:  "fak-demo/1",
		Verdict: "OK",
		Finding: "all clear",
		Corpus:  map[string]any{"grade": "A", "score": 100.0, "demo_debt": 0},
		KPIs: []scorecard.KPI{
			{Key: "beta", Score: 80},
			{Key: "alpha", Score: 100},
		},
	}
	u := FromPayload("fak-demo/1", p, "demo_debt")
	if u.Grade != "A" || u.Score != "100" || u.Debt != "0" {
		t.Fatalf("fold wrong: grade=%q score=%q debt=%q", u.Grade, u.Score, u.Debt)
	}
	if u.Verdict != "OK" || u.Detail != "all clear" {
		t.Fatalf("fold verdict/detail wrong: %+v", u)
	}
	// KPI lines are sorted, so alpha precedes beta.
	if len(u.Lines) != 2 || u.Lines[0] != "alpha: 100" || u.Lines[1] != "beta: 80" {
		t.Fatalf("kpi lines wrong: %v", u.Lines)
	}

	txt := u.Text()
	for _, want := range []string{"fak-demo/1", "grade A", "score 100", "demo_debt 0", "all clear", "alpha: 100"} {
		if !strings.Contains(txt, want) {
			t.Fatalf("Text() missing %q in:\n%s", want, txt)
		}
	}
}

func TestTextHandlesAdHocUpdate(t *testing.T) {
	u := Update{Title: "code-debt", Score: "10", Grade: "A", Verdict: "OK", Detail: "14->10", Source: "agent"}
	txt := u.Text()
	for _, want := range []string{"code-debt", "grade A", "score 10", "14->10", "posted by agent"} {
		if !strings.Contains(txt, want) {
			t.Fatalf("Text() missing %q in:\n%s", want, txt)
		}
	}
}

func TestGradeEmoji(t *testing.T) {
	cases := []struct{ grade, verdict, want string }{
		{"A", "OK", ":large_green_circle:"},
		{"B", "OK", ":large_yellow_circle:"},
		{"", "OK", ":white_check_mark:"},
		{"F", "ACTION", ":red_circle:"},
		{"", "", ":bar_chart:"},
	}
	for _, c := range cases {
		if got := gradeEmoji(c.grade, c.verdict); got != c.want {
			t.Errorf("gradeEmoji(%q,%q) = %q, want %q", c.grade, c.verdict, got, c.want)
		}
	}
}

func TestBlocksCarrySameFacts(t *testing.T) {
	u := Update{Title: "t", Grade: "A", Score: "100", Debt: "0", DebtKey: "demo_debt", Verdict: "OK", Detail: "d", Lines: []string{"k: 1"}, Source: "ci"}
	b := u.Blocks()
	raw, _ := json.Marshal(b)
	s := string(raw)
	for _, want := range []string{"Grade", "Score", "demo_debt", "Verdict", "posted by ci"} {
		if !strings.Contains(s, want) {
			t.Fatalf("Blocks() missing %q in:\n%s", want, s)
		}
	}
}

func TestPostGatesOnNoChange(t *testing.T) {
	postCount := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		postCount++
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true,"ts":"123.456","channel":"C1"}`)
	}))
	defer srv.Close()

	c, err := NewClient("xoxb-test", WithAPIBase(srv.URL+"/"), WithHTTPClient(srv.Client()))
	if err != nil {
		t.Fatal(err)
	}

	up := Update{Title: "test", Grade: "A", Score: "100"}
	ctx := context.Background()

	// First post: should succeed
	ts, err := c.PostWithUpdate(ctx, "C1", up, "test", nil)
	if err != nil {
		t.Fatalf("first post failed: %v", err)
	}
	if ts == "" {
		t.Fatal("first post returned empty ts (should have posted)")
	}
	if postCount != 1 {
		t.Fatalf("first post: want 1 HTTP call, got %d", postCount)
	}

	// Second post with identical data: should skip
	ts, err = c.PostWithUpdate(ctx, "C1", up, "test", nil)
	if err != nil {
		t.Fatalf("second post (no change) failed: %v", err)
	}
	if ts != "" {
		t.Fatal("second post (no change) returned non-empty ts (should have skipped)")
	}
	if postCount != 1 {
		t.Fatalf("second post (no change): want still 1 HTTP call, got %d", postCount)
	}

	// Third post with changed grade: should succeed
	up.Grade = "B"
	ts, err = c.PostWithUpdate(ctx, "C1", up, "test", nil)
	if err != nil {
		t.Fatalf("third post (grade changed) failed: %v", err)
	}
	if ts == "" {
		t.Fatal("third post (grade changed) returned empty ts (should have posted)")
	}
	if postCount != 2 {
		t.Fatalf("third post (grade changed): want 2 HTTP calls, got %d", postCount)
	}

	// Different title: should post even with same values
	up2 := Update{Title: "other", Grade: "B", Score: "100"}
	ts, err = c.PostWithUpdate(ctx, "C1", up2, "test", nil)
	if err != nil {
		t.Fatalf("fourth post (different title) failed: %v", err)
	}
	if ts == "" {
		t.Fatal("fourth post (different title) returned empty ts (should have posted)")
	}
	if postCount != 3 {
		t.Fatalf("fourth post (different title): want 3 HTTP calls, got %d", postCount)
	}
}

func TestResolveProductChannelFromEnv(t *testing.T) {
	t.Setenv("FAK_PRODUCT_CHANNEL", "C_PRODUCT")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCOREBOARD")
	if got := ResolveProductChannel(); got != "C_PRODUCT" {
		t.Fatalf("ResolveProductChannel env = %q, want C_PRODUCT", got)
	}
	// The two channels must not collide: the product resolver must NOT return the
	// #scoreboard id (a product post landing in the number feed is the mistake we guard).
	if got := ResolveChannel(); got != "C_SCOREBOARD" {
		t.Fatalf("ResolveChannel env = %q, want C_SCOREBOARD", got)
	}
	if ResolveProductChannel() == ResolveChannel() {
		t.Fatal("product and scoreboard channels resolved to the same id")
	}
}

func TestResolveProductChannelFromEnvFileWhenEnvUnset(t *testing.T) {
	t.Setenv("FAK_PRODUCT_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "")

	dir := t.TempDir()
	envBody := "# comment\n" +
		"FAK_PRODUCT_CHANNEL=C_PRODUCT_FILE\n" +
		"export FAK_SCOREBOARD_CHANNEL=C_SCOREBOARD_FILE\n"
	if err := os.WriteFile(filepath.Join(dir, ".env.slack.local"), []byte(envBody), 0o600); err != nil {
		t.Fatal(err)
	}
	chdir(t, dir)

	if got := ResolveProductChannel(); got != "C_PRODUCT_FILE" {
		t.Fatalf("ResolveProductChannel file = %q, want C_PRODUCT_FILE", got)
	}
	if got := ResolveChannel(); got != "C_SCOREBOARD_FILE" {
		t.Fatalf("ResolveChannel file = %q, want C_SCOREBOARD_FILE", got)
	}
}

func TestNotesRenderInTextAndBlocks(t *testing.T) {
	body := "fak ships 11 durable products today.\nNext product surface: the disk cache tier (#986)."
	u := Update{Title: "fak product direction", Notes: body, Source: "agent"}

	txt := u.Text()
	for _, want := range []string{"fak product direction", "11 durable products", "#986", "posted by agent"} {
		if !strings.Contains(txt, want) {
			t.Fatalf("Text() missing %q in:\n%s", want, txt)
		}
	}

	blocks := u.Blocks()
	raw, _ := json.Marshal(blocks)
	if !strings.Contains(string(raw), "11 durable products") {
		t.Fatalf("Blocks() missing notes body in:\n%s", raw)
	}

	// An empty Notes leaves the card identical to today (no stray empty section).
	plain := Update{Title: "code-debt", Score: "10", Grade: "A"}
	if strings.Contains(plain.Text(), "\n\n") {
		t.Fatalf("empty Notes introduced a blank line:\n%q", plain.Text())
	}
}

// chdir switches to dir for the test and restores the prior cwd after.
func chdir(t *testing.T, dir string) {
	t.Helper()
	prev, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(prev) })
}
