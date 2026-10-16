package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

// The real HTTP witness uses existing APIs so its partial-content case also
// diagnoses the original false-success path without a new production symbol.
// fak-test:runtime fast est=200ms lane=default
func TestCompleteStreamRequiresTerminalProof(t *testing.T) {
	const content = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"
	const finish = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	const finish0 = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	const finish1 = "data: {\"choices\":[{\"index\":1,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
	const done = "data: [DONE]\n\n"
	for _, tc := range []struct {
		name, body, extra, sink string
		complete, inband        bool
	}{
		{name: "empty"},
		{name: "heartbeat", body: ": keepalive\n\nevent: ping\n\ndata: malformed\n\n"},
		{name: "content_eof", body: content, sink: "partial"},
		{name: "reasoning_eof", body: "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n"},
		{name: "tools_eof", body: "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{}\"}}]}}]}\n\n"},
		{name: "finish_only", body: content + finish, sink: "partial", complete: true},
		{name: "done_only", body: content + done, sink: "partial", complete: true},
		{name: "finish_no_newline", body: content + strings.TrimSpace(finish), sink: "partial", complete: true},
		{name: "multi_one", body: content + finish0, extra: `{"n":2}`, sink: "partial"},
		{name: "multi_all", body: content + finish1 + finish0, extra: `{"n":2}`, sink: "partial", complete: true},
		{name: "multi_duplicate", body: finish0 + finish0, extra: `{"n":2}`},
		{name: "multi_missing_index", body: finish + finish1, extra: `{"n":2}`},
		{name: "wrong_index", body: finish1},
		{name: "invalid_cardinality", body: finish, extra: `{"n":"two"}`},
		{name: "usage_only", body: "data: {\"choices\":[],\"usage\":{\"completion_tokens\":1}}\n\n"},
		{name: "error_after_finish", body: content + finish + "data: {\"error\":{\"code\":\"truncated\",\"message\":\"cut short\"}}\n\n" + done, sink: "partial", inband: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			up, captured := sseServer(t, tc.body)
			p := NewHTTPPlanner(up.URL, "model", "")
			if tc.extra != "" {
				if err := p.SetExtraBodyJSON(tc.extra); err != nil {
					t.Fatal(err)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var sink strings.Builder
			comp, err := p.CompleteStream(ctx, func(s string) error { sink.WriteString(s); return nil }, []Message{{Role: RoleUser, Content: "hi"}}, nil)
			if sink.String() != tc.sink {
				t.Fatalf("sink=%q want %q", sink.String(), tc.sink)
			}
			var sent map[string]json.RawMessage
			if decodeErr := json.Unmarshal(*captured, &sent); decodeErr != nil || string(sent["stream"]) != "true" {
				t.Fatalf("stream request=%s error=%v", *captured, decodeErr)
			}
			if tc.extra != "" {
				var extra map[string]json.RawMessage
				if decodeErr := json.Unmarshal([]byte(tc.extra), &extra); decodeErr != nil || string(sent["n"]) != string(extra["n"]) {
					t.Fatalf("request cardinality changed: %s", *captured)
				}
			}
			if tc.complete {
				if err != nil || comp == nil || comp.Message.Content != tc.sink {
					t.Fatalf("completed stream=%+v error=%v", comp, err)
				}
				return
			}
			if tc.inband {
				if comp != nil || !errors.Is(err, ErrUpstreamStreamError) {
					t.Fatalf("in-band failure lost: completion=%+v error=%v", comp, err)
				}
				return
			}
			if comp != nil || err == nil || !strings.Contains(err.Error(), "upstream stream ended before completion") || errors.Unwrap(err) == nil {
				t.Fatalf("incomplete stream returned completion=%+v error=%v; want incomplete-stream refusal", comp, err)
			}
		})
	}
}

// A recognized finish cannot hide a subsequent transport failure. A transport
// error delivered with the bytes containing DONE also retains its identity.
// fak-test:runtime fast est=10ms lane=default
func TestCompleteStreamTerminalPreservesReadFailure(t *testing.T) {
	for _, terminal := range []string{"", "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n", "data: [DONE]\n\n"} {
		cause := errors.New("terminal body reset")
		body := "data: {\"choices\":[{\"delta\":{\"content\":\"once\"}}]}\n\n" + terminal
		reader := &completionErrorReader{body: []byte(body), err: cause}
		p := NewHTTPPlanner("http://upstream.test", "model", "")
		attempts := 0
		p.Client = &http.Client{Transport: issue10321RoundTripFunc(func(*http.Request) (*http.Response, error) {
			attempts++
			return issue10321Response("text/event-stream", reader), nil
		})}
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		var sink strings.Builder
		comp, err := p.CompleteStream(ctx, func(s string) error { sink.WriteString(s); return nil }, []Message{{Role: RoleUser, Content: "hi"}}, nil)
		cancel()
		if comp != nil || !errors.Is(err, cause) || attempts != 1 || sink.String() != "once" || !reader.closed {
			t.Fatalf("terminal=%q completion=%+v error=%v attempts=%d sink=%q closed=%t", terminal, comp, err, attempts, sink.String(), reader.closed)
		}
	}
}

type completionErrorReader struct {
	body   []byte
	err    error
	closed bool
}

func (r *completionErrorReader) Read(p []byte) (int, error) {
	if len(r.body) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.body)
	r.body = r.body[n:]
	if len(r.body) == 0 {
		return n, r.err
	}
	return n, nil
}

func (r *completionErrorReader) Close() error { r.closed = true; return nil }
