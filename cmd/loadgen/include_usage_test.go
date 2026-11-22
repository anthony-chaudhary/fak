package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

// TestIncludeUsageConformance pins the streaming usage rule: a streaming chat
// request carries stream_options.include_usage=true, caller-supplied
// stream_options are never clobbered, non-streaming requests carry no
// stream_options, and replayed stored payloads go out byte-for-byte.
func TestIncludeUsageConformance(t *testing.T) {
	t.Parallel()
	optOut := &streamOptions{IncludeUsage: false}
	optIn := &streamOptions{IncludeUsage: true}
	cases := []struct {
		name       string
		req        chatRequest
		wantOpts   bool
		wantUsage  bool
		wantSameIn *streamOptions
	}{
		{name: "streaming without options injects", req: chatRequest{Stream: true}, wantOpts: true, wantUsage: true},
		{name: "streaming explicit opt-out preserved", req: chatRequest{Stream: true, StreamOptions: optOut}, wantOpts: true, wantUsage: false, wantSameIn: optOut},
		{name: "streaming explicit opt-in preserved", req: chatRequest{Stream: true, StreamOptions: optIn}, wantOpts: true, wantUsage: true, wantSameIn: optIn},
		{name: "non-streaming adds nothing", req: chatRequest{Stream: false}, wantOpts: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.req.withIncludeUsage()
			if tc.wantSameIn != nil && got.StreamOptions != tc.wantSameIn {
				t.Fatalf("caller stream_options replaced: got %p want %p", got.StreamOptions, tc.wantSameIn)
			}
			raw, err := json.Marshal(got)
			if err != nil {
				t.Fatal(err)
			}
			var wire map[string]any
			if err := json.Unmarshal(raw, &wire); err != nil {
				t.Fatal(err)
			}
			so, present := wire["stream_options"]
			if present != tc.wantOpts {
				t.Fatalf("stream_options present=%v want %v in %s", present, tc.wantOpts, raw)
			}
			if !tc.wantOpts {
				return
			}
			v, ok := so.(map[string]any)["include_usage"].(bool)
			if !ok || v != tc.wantUsage {
				t.Fatalf("include_usage=%v (present=%v) want %v in %s", v, ok, tc.wantUsage, raw)
			}
		})
	}
	orig := chatRequest{Stream: true}
	_ = orig.withIncludeUsage()
	if orig.StreamOptions != nil {
		t.Fatal("withIncludeUsage mutated its receiver")
	}

	t.Run("wire", func(t *testing.T) {
		const replay = `{"model":"m","messages":[],"stream":true}`
		for _, override := range [][]byte{nil, []byte(replay)} {
			var sent []byte
			cfg := &Config{
				URL: "http://loadgen.test/v1/chat/completions", Model: "m", Prompt: "p", MaxTokens: 8,
				HTTPClient: &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
					sent, _ = io.ReadAll(r.Body)
					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     http.Header{"Content-Type": {"text/event-stream"}},
						Body:       io.NopCloser(strings.NewReader("data: {\"usage\":{\"completion_tokens\":3}}\n\ndata: [DONE]\n\n")),
					}, nil
				})},
			}
			r := oneRequest(context.Background(), cfg, &clientInFlight{}, override)
			if override != nil {
				if !bytes.Equal(sent, override) {
					t.Fatalf("replayed payload altered: sent %s want %s", sent, override)
				}
				continue
			}
			var wire struct {
				Stream        bool `json:"stream"`
				StreamOptions *struct {
					IncludeUsage *bool `json:"include_usage"`
				} `json:"stream_options"`
			}
			if err := json.Unmarshal(sent, &wire); err != nil {
				t.Fatalf("decode sent body %s: %v", sent, err)
			}
			if !wire.Stream || wire.StreamOptions == nil || wire.StreamOptions.IncludeUsage == nil || !*wire.StreamOptions.IncludeUsage {
				t.Fatalf("synthetic stream request lacks stream_options.include_usage=true: %s", sent)
			}
			if r.err != nil || r.completionTokens != 3 || r.tokenUsageMissing {
				t.Fatalf("usage not read from trailing chunk: err=%v tokens=%d missing=%v", r.err, r.completionTokens, r.tokenUsageMissing)
			}
		}
	})
}
