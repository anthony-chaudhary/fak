package slackwire

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// newTestClient wires a Client to srv with the sleep seam capturing waits instead
// of serving them, so 429 tests run instantly and witness the exact backoff.
func newTestClient(t *testing.T, srv *httptest.Server, opts ...Option) (*Client, *[]time.Duration) {
	t.Helper()
	waits := &[]time.Duration{}
	opts = append([]Option{WithAPIBase(srv.URL + "/"), WithHTTPClient(srv.Client())}, opts...)
	c := New("xoxb-test", opts...)
	c.sleep = func(ctx context.Context, d time.Duration) error {
		*waits = append(*waits, d)
		return ctx.Err()
	}
	return c, waits
}

func TestPostMessageSendsChannelTextBlocksAndThread(t *testing.T) {
	var gotBody map[string]any
	var gotAuth, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = io.WriteString(w, `{"ok":true,"ts":"111.222"}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	ts, err := c.PostMessage(context.Background(), "C1", "hello", []any{map[string]any{"type": "section"}}, "99.100")
	if err != nil {
		t.Fatal(err)
	}
	if ts != "111.222" {
		t.Fatalf("ts = %q, want 111.222", ts)
	}
	if !strings.HasSuffix(gotPath, "chat.postMessage") {
		t.Fatalf("path = %q, want chat.postMessage", gotPath)
	}
	if gotAuth != "Bearer xoxb-test" {
		t.Fatalf("auth = %q", gotAuth)
	}
	if gotBody["channel"] != "C1" || gotBody["text"] != "hello" || gotBody["thread_ts"] != "99.100" {
		t.Fatalf("body wrong: %+v", gotBody)
	}
	if _, ok := gotBody["blocks"]; !ok {
		t.Fatalf("blocks not sent: %+v", gotBody)
	}
}

func TestPostMessageOmitsEmptyThreadAndBlocks(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = io.WriteString(w, `{"ok":true,"ts":"1.2"}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	if _, err := c.PostMessage(context.Background(), "C1", "hi", nil, ""); err != nil {
		t.Fatal(err)
	}
	if _, ok := gotBody["thread_ts"]; ok {
		t.Fatalf("empty threadTS was sent: %+v", gotBody)
	}
	if _, ok := gotBody["blocks"]; ok {
		t.Fatalf("nil blocks were sent: %+v", gotBody)
	}
}

func TestUpdateMessageSendsChannelTSText(t *testing.T) {
	var gotBody map[string]any
	var gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	if err := c.UpdateMessage(context.Background(), "C1", "111.222", "edited", nil); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(gotPath, "chat.update") {
		t.Fatalf("path = %q, want chat.update", gotPath)
	}
	if gotBody["channel"] != "C1" || gotBody["ts"] != "111.222" || gotBody["text"] != "edited" {
		t.Fatalf("body wrong: %+v", gotBody)
	}
}

func TestHistoryPassesOldestAndLimitAndDecodes(t *testing.T) {
	var gotQuery map[string][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query()
		_, _ = io.WriteString(w, `{"ok":true,"messages":[
			{"type":"message","ts":"2.0","thread_ts":"1.0","user":"U1","text":"hi"},
			{"type":"message","subtype":"bot_message","ts":"3.0","bot_id":"B9","text":"beep"}]}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	msgs, err := c.History(context.Background(), "C1", "1.5", 42)
	if err != nil {
		t.Fatal(err)
	}
	if got := gotQuery["channel"]; len(got) != 1 || got[0] != "C1" {
		t.Fatalf("channel query = %v", gotQuery)
	}
	if got := gotQuery["oldest"]; len(got) != 1 || got[0] != "1.5" {
		t.Fatalf("oldest query = %v", gotQuery)
	}
	if got := gotQuery["limit"]; len(got) != 1 || got[0] != "42" {
		t.Fatalf("limit query = %v", gotQuery)
	}
	if len(msgs) != 2 {
		t.Fatalf("got %d messages, want 2", len(msgs))
	}
	m := msgs[0]
	if m.Type != "message" || m.TS != "2.0" || m.ThreadTS != "1.0" || m.User != "U1" || m.Text != "hi" {
		t.Fatalf("message 0 decoded wrong: %+v", m)
	}
	if msgs[1].Subtype != "bot_message" || msgs[1].BotID != "B9" {
		t.Fatalf("message 1 decoded wrong: %+v", msgs[1])
	}
}

func TestHistoryOmitsEmptyOldestAndZeroLimit(t *testing.T) {
	var gotQuery map[string][]string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query()
		_, _ = io.WriteString(w, `{"ok":true,"messages":[]}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	if _, err := c.History(context.Background(), "C1", "", 0); err != nil {
		t.Fatal(err)
	}
	if _, ok := gotQuery["oldest"]; ok {
		t.Fatalf("empty oldest was sent: %v", gotQuery)
	}
	if _, ok := gotQuery["limit"]; ok {
		t.Fatalf("zero limit was sent: %v", gotQuery)
	}
}

func TestAuthTestDecodesIdentity(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":true,"url":"https://acme.slack.com/","team":"acme","user":"fakbot","team_id":"T1","user_id":"U1","bot_id":"B1"}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	info, err := c.AuthTest(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if info.Team != "acme" || info.User != "fakbot" || info.TeamID != "T1" || info.UserID != "U1" || info.BotID != "B1" {
		t.Fatalf("identity decoded wrong: %+v", info)
	}
}

// TestOKFalseIsTypedAPIError covers the error taxonomy: an ok:false envelope
// surfaces as *APIError carrying method, HTTP status, and Slack's error token.
func TestOKFalseIsTypedAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":false,"error":"channel_not_found"}`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	_, err := c.PostMessage(context.Background(), "CBAD", "hi", nil, "")
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.Method != "chat.postMessage" || apiErr.Code != "channel_not_found" || apiErr.Status != 200 {
		t.Fatalf("APIError fields wrong: %+v", apiErr)
	}
	if !strings.Contains(err.Error(), "channel_not_found") {
		t.Fatalf("Error() must carry the token: %q", err.Error())
	}
}

func TestNonJSONBodySurfacesDecodeErrorWithBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = io.WriteString(w, `<html>upstream sadness</html>`)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	_, err := c.History(context.Background(), "C1", "", 0)
	if err == nil {
		t.Fatal("want decode error")
	}
	for _, want := range []string{"conversations.history", "502", "upstream sadness"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("decode error missing %q: %v", want, err)
		}
	}
}

// Test429HonorsRetryAfterThenSucceeds is the core new capability: a rate-limited
// call waits the server-stated seconds and re-sends, invisibly to the caller.
func Test429HonorsRetryAfterThenSucceeds(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			w.Header().Set("Retry-After", "7")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_, _ = io.WriteString(w, `{"ok":true,"ts":"9.9"}`)
	}))
	defer srv.Close()

	c, waits := newTestClient(t, srv)
	ts, err := c.PostMessage(context.Background(), "C1", "hi", nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if ts != "9.9" {
		t.Fatalf("ts = %q", ts)
	}
	if calls != 2 {
		t.Fatalf("calls = %d, want 2", calls)
	}
	if len(*waits) != 1 || (*waits)[0] != 7*time.Second {
		t.Fatalf("waits = %v, want [7s]", *waits)
	}
}

func Test429BudgetExhaustedReturnsRatelimited(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c, waits := newTestClient(t, srv, WithRetryBudget(2))
	_, err := c.PostMessage(context.Background(), "C1", "hi", nil, "")
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("want *APIError, got %T: %v", err, err)
	}
	if apiErr.Status != http.StatusTooManyRequests || apiErr.Code != "ratelimited" {
		t.Fatalf("APIError fields wrong: %+v", apiErr)
	}
	if calls != 3 { // 1 initial + 2 retries
		t.Fatalf("calls = %d, want 3", calls)
	}
	if len(*waits) != 2 {
		t.Fatalf("waits = %v, want 2 entries", *waits)
	}
}

func Test429ZeroBudgetFailsImmediately(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c, waits := newTestClient(t, srv, WithRetryBudget(0))
	_, err := c.PostMessage(context.Background(), "C1", "hi", nil, "")
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "ratelimited" {
		t.Fatalf("want ratelimited *APIError, got %v", err)
	}
	if calls != 1 || len(*waits) != 0 {
		t.Fatalf("calls=%d waits=%v, want 1 call and no waits", calls, *waits)
	}
}

func Test429RetryCancelledByContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c, _ := newTestClient(t, srv)
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled: the sleep seam returns ctx.Err() and the call stops
	_, err := c.PostMessage(ctx, "C1", "hi", nil, "")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
}

func TestRetryAfterWaitBounds(t *testing.T) {
	cases := []struct {
		header string
		want   time.Duration
	}{
		{"7", 7 * time.Second},
		{"", retryAfterFallback},     // absent header
		{"soon", retryAfterFallback}, // garbled header
		{"-3", retryAfterFallback},   // negative is nonsense
		{"0", retryAfterFallback},    // zero would hot-loop
		{"86400", retryAfterCap},     // a day-long park is capped
		{"30", 30 * time.Second},     // exactly the cap passes
	}
	for _, tc := range cases {
		if got := retryAfterWait(tc.header); got != tc.want {
			t.Errorf("retryAfterWait(%q) = %v, want %v", tc.header, got, tc.want)
		}
	}
}

// TestEmptyTokenSurfacesSlackVerdict pins the deliberate design choice: the wire
// does not validate credentials; Slack's invalid_auth is the answer.
func TestEmptyTokenSurfacesSlackVerdict(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// net/http trims the trailing space an empty token leaves after "Bearer".
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer") {
			t.Errorf("auth header = %q", r.Header.Get("Authorization"))
		}
		_, _ = io.WriteString(w, `{"ok":false,"error":"invalid_auth"}`)
	}))
	defer srv.Close()

	c := New("", WithAPIBase(srv.URL+"/"), WithHTTPClient(srv.Client()))
	_, err := c.AuthTest(context.Background())
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Code != "invalid_auth" {
		t.Fatalf("want invalid_auth *APIError, got %v", err)
	}
}
