package dgxbridge

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// rtFunc adapts a func to http.RoundTripper.
type rtFunc func(*http.Request) (*http.Response, error)

func (f rtFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func jsonResp(body string) *http.Response {
	return &http.Response{
		StatusCode: 200,
		Body:       io.NopCloser(strings.NewReader(body)),
		Header:     make(http.Header),
	}
}

func testClient(t *testing.T, rt rtFunc) *Client {
	t.Helper()
	c, err := NewClient("xoxb-test", WithHTTPClient(&http.Client{Transport: rt}), WithAPIBase("https://slack.test/api/"))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestPost_RequestShape(t *testing.T) {
	var gotURL, gotAuth, gotBody string
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		gotURL = r.URL.String()
		gotAuth = r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		return jsonResp(`{"ok":true,"ts":"123.456"}`), nil
	})
	ts, err := c.Post(context.Background(), "C123", "999.111", "hello")
	if err != nil {
		t.Fatal(err)
	}
	if ts != "123.456" {
		t.Fatalf("ts=%q", ts)
	}
	if !strings.HasSuffix(gotURL, "chat.postMessage") {
		t.Fatalf("url=%q", gotURL)
	}
	if gotAuth != "Bearer xoxb-test" {
		t.Fatalf("auth=%q", gotAuth)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(gotBody), &body); err != nil {
		t.Fatal(err)
	}
	if body["channel"] != "C123" || body["thread_ts"] != "999.111" || body["text"] != "hello" {
		t.Fatalf("body=%v", body)
	}
}

func TestPost_OmitsThreadWhenEmpty(t *testing.T) {
	var gotBody string
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		return jsonResp(`{"ok":true,"ts":"1.0"}`), nil
	})
	if _, err := c.Post(context.Background(), "C1", "", "x"); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(gotBody, "thread_ts") {
		t.Fatalf("thread_ts should be omitted: %q", gotBody)
	}
}

func TestPost_SlackError(t *testing.T) {
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		return jsonResp(`{"ok":false,"error":"channel_not_found"}`), nil
	})
	_, err := c.Post(context.Background(), "C1", "", "x")
	if err == nil || !strings.Contains(err.Error(), "channel_not_found") {
		t.Fatalf("expected channel_not_found, got %v", err)
	}
}

func TestReplies_QueryParams(t *testing.T) {
	var gotURL string
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		gotURL = r.URL.String()
		return jsonResp(`{"ok":true,"messages":[{"ts":"1.1","text":"hi"}]}`), nil
	})
	msgs, err := c.Replies(context.Background(), "C9", "5.5", "4.4", 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(msgs) != 1 || msgs[0].Text != "hi" {
		t.Fatalf("msgs=%v", msgs)
	}
	for _, want := range []string{"channel=C9", "ts=5.5", "oldest=4.4", "limit=50", "conversations.replies"} {
		if !strings.Contains(gotURL, want) {
			t.Fatalf("url %q missing %q", gotURL, want)
		}
	}
}

func TestListFiles_ParsesFileFields(t *testing.T) {
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		return jsonResp(`{"ok":true,"files":[{"id":"F1","name":"transcript.jsonl","created":1781965000,"size":1234,"url_private_download":"https://files.slack.test/d"}]}`), nil
	})
	fs, err := c.ListFiles(context.Background(), "C1", 12)
	if err != nil {
		t.Fatal(err)
	}
	if len(fs) != 1 {
		t.Fatalf("len=%d", len(fs))
	}
	f := fs[0]
	if f.ID != "F1" || f.Name != "transcript.jsonl" || f.Created != 1781965000 || f.URLDownload == "" {
		t.Fatalf("file=%+v", f)
	}
}

func TestFindControlSession_NewestForHost(t *testing.T) {
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		return jsonResp(`{"ok":true,"messages":[
			{"ts":"100.0","text":"Slack control session started on ` + "`other.example.lab`" + `"},
			{"ts":"200.0","text":"Slack control session started on ` + "`dgx-a100.example.lab`" + ` for persistent"},
			{"ts":"150.0","text":"unrelated chatter"},
			{"ts":"300.0","text":"Slack control session started on ` + "`dgx-a100.example.lab`" + ` for persistent"}
		]}`), nil
	})
	cs, err := c.FindControlSession(context.Background(), "C1", "dgx-a100.example.lab")
	if err != nil {
		t.Fatal(err)
	}
	if cs == nil || cs.ThreadTS != "300.0" {
		t.Fatalf("expected newest 300.0, got %+v", cs)
	}
}

func TestFindControlSession_None(t *testing.T) {
	c := testClient(t, func(r *http.Request) (*http.Response, error) {
		return jsonResp(`{"ok":true,"messages":[{"ts":"1.0","text":"nothing here"}]}`), nil
	})
	cs, err := c.FindControlSession(context.Background(), "C1", "")
	if err != nil {
		t.Fatal(err)
	}
	if cs != nil {
		t.Fatalf("expected nil, got %+v", cs)
	}
}
