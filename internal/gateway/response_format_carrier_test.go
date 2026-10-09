package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fak-test:runtime fast est=500ms lane=default
func TestChatResponseFormatCarrierRejectsBeforeDispatch(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	for _, fanout := range []bool{false, true} {
		name := "without_followers"
		if fanout {
			name = "with_followers"
		}
		t.Run(name, func(t *testing.T) {
			var upstreamHits, followerHits atomic.Int32
			var forwardedMu sync.Mutex
			var forwarded json.RawMessage
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				upstreamHits.Add(1)
				var request struct {
					ResponseFormat json.RawMessage `json:"response_format"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode upstream request: %v", err)
				}
				forwardedMu.Lock()
				forwarded = request.ResponseFormat
				forwardedMu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{}}`)
			}))
			t.Cleanup(upstream.Close)
			follower := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				followerHits.Add(1)
				_, _ = io.Copy(io.Discard, r.Body)
				_, _ = io.WriteString(w, `{}`)
			}))
			t.Cleanup(follower.Close)
			addrs := ""
			if fanout {
				addrs = follower.URL
			}
			t.Setenv("FAK_EP_FANOUT_ADDRS", addrs)
			srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: upstream.URL + "/v1", Provider: "openai-compatible", VDSO: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			ts := httptest.NewServer(srv.Handler())
			t.Cleanup(ts.Close)
			client := &http.Client{Timeout: 5 * time.Second}

			for _, tc := range []struct {
				name, format, code, path string
				stream                  bool
			}{
				{name: "non_object", format: `"json_schema"`, code: "invalid_response_format"},
				{name: "missing_carrier", format: `{"type":"json_schema"}`, code: "json_schema_missing"},
				{name: "non_object_carrier", format: `{"type":"json_schema","json_schema":[]}`, code: "json_schema_not_object"},
				{name: "missing_schema", format: `{"type":"json_schema","json_schema":{}}`, code: "json_schema_missing"},
				{name: "array_schema_stream", format: `{"type":"json_schema","json_schema":{"schema":[]}}`, code: "json_schema_not_object", path: "#", stream: true},
				{name: "open_object", format: `{"type":"json_schema","json_schema":{"schema":{"type":"object"}}}`},
				{name: "boolean_true", format: `{"type":"json_schema","json_schema":{"schema":true}}`},
				{name: "boolean_false", format: `{"type":"json_schema","json_schema":{"schema":false}}`},
				{name: "draft_three", format: `{"type":"json_schema","json_schema":{"schema":{"$schema":"http://json-schema.org/draft-03/schema#","properties":{"a":{"type":"string","required":true}}}}}`},
				{name: "other_format", format: `{"type":"json_object"}`},
			} {
				t.Run(tc.name, func(t *testing.T) {
					beforeUpstream, beforeFollower := upstreamHits.Load(), followerHits.Load()
					body, err := json.Marshal(map[string]any{
						"model": "m", "messages": []map[string]string{{"role": "user", "content": "hi"}},
						"response_format": json.RawMessage(tc.format), "stream": tc.stream,
					})
					if err != nil {
						t.Fatal(err)
					}
					response, err := client.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
					if err != nil {
						t.Fatal(err)
					}
					defer response.Body.Close()
					raw, err := io.ReadAll(response.Body)
					if err != nil {
						t.Fatal(err)
					}
					if tc.code != "" {
						var envelope struct {
							Error struct {
								Code       string `json:"code"`
								Type       string `json:"type"`
								SchemaPath string `json:"schema_path"`
							} `json:"error"`
						}
						if err := json.Unmarshal(raw, &envelope); err != nil {
							t.Fatalf("decode refusal: %v: %s", err, raw)
						}
						if response.StatusCode != http.StatusUnprocessableEntity || envelope.Error.Code != tc.code || envelope.Error.Type != "invalid_request_error" || envelope.Error.SchemaPath != tc.path {
							t.Fatalf("status=%d refusal=%s, want 422 code=%s path=%s", response.StatusCode, raw, tc.code, tc.path)
						}
						if upstreamHits.Load() != beforeUpstream || followerHits.Load() != beforeFollower {
							t.Fatal("rejected carrier dispatched to upstream or EP follower")
						}
						return
					}
					if response.StatusCode != http.StatusOK || upstreamHits.Load() != beforeUpstream+1 {
						t.Fatalf("accepted carrier: status=%d upstream calls=%d: %s", response.StatusCode, upstreamHits.Load()-beforeUpstream, raw)
					}
					forwardedMu.Lock()
					got := append(json.RawMessage(nil), forwarded...)
					forwardedMu.Unlock()
					if !jsonEqual(t, got, json.RawMessage(tc.format)) {
						t.Fatalf("forwarded format = %s, want %s", got, tc.format)
					}
					wantFollowers := beforeFollower
					if fanout {
						wantFollowers++
					}
					if followerHits.Load() != wantFollowers {
						t.Fatalf("follower calls=%d, want %d", followerHits.Load(), wantFollowers)
					}
				})
			}
		})
	}
}
