package gateway

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

type unkeyedWireFixture struct {
	name, path, root, other, formatted, generation, continuation string
	changedSystem, changedTools, changedImage, largeA, largeB    string
	noUser, credential                                           string
}

func unkeyedWireFixtures() []unkeyedWireFixture {
	return []unkeyedWireFixture{
		{
			name: "anthropic", path: "/v1/messages", credential: "x-api-key",
			root:          `{"model":"test-model","max_tokens":32,"system":"setup","messages":[{"role":"user","content":[{"type":"text","text":"root"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQ=="}}]}]}`,
			other:         `{"model":"test-model","max_tokens":32,"system":"setup","messages":[{"role":"user","content":"other root"}]}`,
			formatted:     "{\n \"messages\": [{\"content\":[{\"text\":\"root\",\"type\":\"text\"},{\"source\":{\"data\":\"AQ==\",\"media_type\":\"image/png\",\"type\":\"base64\"},\"type\":\"image\"}],\"role\":\"user\"}], \"system\":\"setup\",\"max_tokens\":32,\"model\":\"test-model\"}",
			generation:    `,"temperature":0.5,"top_p":0.8,"stop_sequences":["stop"],"metadata":{"user_id":"generation-label"}`,
			continuation:  `{"model":"test-model","max_tokens":32,"system":"setup","messages":[{"role":"user","content":[{"type":"text","text":"root"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AQ=="}}]},{"role":"assistant","content":"ok"},{"role":"user","content":"next"}]}`,
			changedSystem: `"system":"different setup"`, changedTools: `,"tools":[{"name":"example","input_schema":{"type":"object"}}]`,
			changedImage: `"data":"Ag=="`,
			largeA:       `,"tools":[{"name":"example","input_schema":{"type":"integer","minimum":9007199254740992}}]`,
			largeB:       `,"tools":[{"name":"example","input_schema":{"type":"integer","minimum":9007199254740993}}]`,
			noUser:       `{"model":"test-model","max_tokens":32,"messages":[{"role":"assistant","content":"prior assistant only"}]}`,
		},
		{
			name: "gemini", path: "/v1beta/models/test-model:generateContent", credential: "x-goog-api-key",
			root:          `{"systemInstruction":{"parts":[{"text":"setup"}]},"contents":[{"role":"user","parts":[{"text":"root"},{"inlineData":{"mimeType":"image/png","data":"AQ=="}}]}],"generationConfig":{"maxOutputTokens":32}}`,
			other:         `{"systemInstruction":{"parts":[{"text":"setup"}]},"contents":[{"role":"user","parts":[{"text":"other root"}]}]}`,
			formatted:     "{\n\"generationConfig\":{\"maxOutputTokens\":32},\"contents\":[{\"parts\":[{\"text\":\"root\"},{\"inlineData\":{\"data\":\"AQ==\",\"mimeType\":\"image/png\"}}],\"role\":\"user\"}],\"systemInstruction\":{\"parts\":[{\"text\":\"setup\"}]}}",
			generation:    `"maxOutputTokens":64,"temperature":0.5,"topP":0.8,"stopSequences":["stop"]`,
			continuation:  `{"systemInstruction":{"parts":[{"text":"setup"}]},"contents":[{"role":"user","parts":[{"text":"root"},{"inlineData":{"mimeType":"image/png","data":"AQ=="}}]},{"role":"model","parts":[{"text":"ok"}]},{"role":"user","parts":[{"text":"next"}]}]}`,
			changedSystem: `"text":"different setup"`, changedTools: `,"tools":[{"functionDeclarations":[{"name":"example","parameters":{"type":"object"}}]}]`,
			changedImage: `"data":"Ag=="`,
			largeA:       `,"tools":[{"functionDeclarations":[{"name":"example","parameters":{"type":"integer","minimum":9007199254740992}}]}]`,
			largeB:       `,"tools":[{"functionDeclarations":[{"name":"example","parameters":{"type":"integer","minimum":9007199254740993}}]}]`,
			noUser:       `{"contents":[{"role":"model","parts":[{"text":"prior assistant only"}]}]}`,
		},
	}
}

func unkeyedWirePost(t *testing.T, handler http.Handler, path, body, remote string, headers map[string]string) (int, string, string) {
	t.Helper()
	r := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	r.RemoteAddr = remote
	r.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		r.Header.Set(key, value)
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, r)
	return w.Code, w.Header().Get(traceHeader), w.Body.String()
}

// Existing-API parent witness through both real wire adapters and TCP clients.
// fak-test:runtime medium est=1s lane=default
func TestUnkeyedWireConversationContinuity(t *testing.T) {
	for _, wire := range unkeyedWireFixtures() {
		t.Run(wire.name, func(t *testing.T) {
			srv := unkeyedContinuityServer(t, UnkeyedPerConnectionTraceID, 0)
			var mu sync.Mutex
			var peers []string
			handler := srv.Handler()
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				peers = append(peers, r.RemoteAddr)
				mu.Unlock()
				handler.ServeHTTP(w, r)
			}))
			defer ts.Close()
			a, b := &http.Client{Transport: &http.Transport{}}, &http.Client{Transport: &http.Transport{}}
			defer a.CloseIdleConnections()
			defer b.CloseIdleConnections()
			post := func(client *http.Client, body string) (int, string, string) {
				t.Helper()
				response, err := client.Post(ts.URL+wire.path, "application/json", strings.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				defer response.Body.Close()
				raw, err := io.ReadAll(response.Body)
				if err != nil {
					t.Fatal(err)
				}
				return response.StatusCode, response.Header.Get(traceHeader), string(raw)
			}
			code, first, body := post(a, wire.root)
			if code != http.StatusOK {
				t.Fatalf("initial status=%d: %s", code, body)
			}
			for _, client := range []*http.Client{a, b} {
				code, trace, body := post(client, wire.root)
				if code != http.StatusConflict || trace != first || !strings.Contains(body, sessionReasonBudgetContext) {
					t.Fatalf("exhausted/reconnected status=%d trace=%q first=%q: %s", code, trace, first, body)
				}
			}
			code, other, body := post(a, wire.other)
			if code != http.StatusOK || other == first {
				t.Fatalf("new root on reused connection: status=%d trace=%q first=%q: %s", code, other, first, body)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(peers) != 4 || peers[0] == peers[2] || peers[0] != peers[3] {
				t.Fatalf("fixture lost reconnect/reuse: %v", peers)
			}
		})
	}
}

// Generated provenance stays request-local; canonical raw tuples keep fidelity
// without changing either adapter's parser, auth, or explicit trace policy.
// fak-test:runtime medium est=1s lane=default
func TestUnkeyedWireConversationBoundaries(t *testing.T) {
	for _, wire := range unkeyedWireFixtures() {
		t.Run(wire.name, func(t *testing.T) {
			appendField := func(body, field string) string { return strings.TrimSuffix(body, "}") + field + "}" }
			generation := appendField(wire.root, wire.generation)
			if wire.name == "gemini" {
				generation = strings.Replace(wire.root, `"maxOutputTokens":32`, wire.generation, 1)
			}
			changedSystem := strings.Replace(wire.root, `"system":"setup"`, wire.changedSystem, 1)
			if wire.name == "gemini" {
				changedSystem = strings.Replace(wire.root, `"text":"setup"`, wire.changedSystem, 1)
			}
			for _, tc := range []struct {
				name, first, second, defaultTrace string
				headers                           map[string]string
				want                              int
			}{
				{"formatting and key order", wire.root, wire.formatted, UnkeyedPerConnectionTraceID, nil, http.StatusConflict},
				{"generation options excluded", wire.root, generation, UnkeyedPerConnectionTraceID, nil, http.StatusConflict},
				{"continuation", wire.root, wire.continuation, UnkeyedPerConnectionTraceID, nil, http.StatusConflict},
				{"system resets", wire.root, changedSystem, UnkeyedPerConnectionTraceID, nil, http.StatusOK},
				{"tools reset", wire.root, appendField(wire.root, wire.changedTools), UnkeyedPerConnectionTraceID, nil, http.StatusOK},
				{"image bytes reset", wire.root, strings.Replace(wire.root, `"data":"AQ=="`, wire.changedImage, 1), UnkeyedPerConnectionTraceID, nil, http.StatusOK},
				{"large numbers remain distinct", appendField(wire.root, wire.largeA), appendField(wire.root, wire.largeB), UnkeyedPerConnectionTraceID, nil, http.StatusOK},
				{"default disabled", wire.root, wire.root, "", nil, http.StatusOK},
				{"configured default preserved", wire.root, wire.other, "operator-session", nil, http.StatusConflict},
				{"explicit generated-looking trace", wire.root, wire.other, UnkeyedPerConnectionTraceID, map[string]string{"X-Trace-Id": "unkeyed-caller-chosen", "X-Fak-Unkeyed-Generated": "true"}, http.StatusConflict},
				{"request id preserved", wire.root, wire.other, UnkeyedPerConnectionTraceID, map[string]string{"X-Request-ID": "caller-request"}, http.StatusConflict},
				{"traceparent preserved", wire.root, wire.other, UnkeyedPerConnectionTraceID, map[string]string{"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}, http.StatusConflict},
				{"no user fallback", wire.noUser, wire.noUser, UnkeyedPerConnectionTraceID, nil, http.StatusOK},
			} {
				t.Run(tc.name, func(t *testing.T) {
					handler := unkeyedContinuityServer(t, tc.defaultTrace, 0).Handler()
					firstCode, first, firstBody := unkeyedWirePost(t, handler, wire.path, tc.first, "192.0.2.1:40001", tc.headers)
					code, trace, body := unkeyedWirePost(t, handler, wire.path, tc.second, "192.0.2.1:40002", tc.headers)
					if firstCode != http.StatusOK || code != tc.want || (first == trace) != (tc.want == http.StatusConflict) || code == http.StatusConflict && !strings.Contains(body, sessionReasonBudgetContext) {
						t.Fatalf("statuses=%d/%d traces=%q/%q bodies=%s/%s; want second=%d", firstCode, code, first, trace, firstBody, body, tc.want)
					}
				})
			}
			t.Run("model and eviction resets", func(t *testing.T) {
				handler := unkeyedContinuityServer(t, UnkeyedPerConnectionTraceID, 1).Handler()
				_, first, _ := unkeyedWirePost(t, handler, wire.path, wire.root, "192.0.2.1:40001", nil)
				path, body := wire.path, wire.root
				if wire.name == "gemini" {
					path = strings.Replace(path, "test-model", "other-model", 1)
				} else {
					body = strings.Replace(body, "test-model", "other-model", 1)
				}
				code, other, response := unkeyedWirePost(t, handler, path, body, "192.0.2.1:40001", nil)
				if code != http.StatusOK || other == first {
					t.Fatalf("model change did not reset accounting: status=%d trace=%q first=%q: %s", code, other, first, response)
				}
				code, again, response := unkeyedWirePost(t, handler, wire.path, wire.root, "192.0.2.1:40002", nil)
				if code != http.StatusOK || again != first {
					t.Fatalf("eviction reset changed: status=%d trace=%q first=%q: %s", code, again, first, response)
				}
			})
			t.Run("authenticated isolation and error precedence", func(t *testing.T) {
				srv := unkeyedContinuityServer(t, UnkeyedPerConnectionTraceID, 0)
				srv.keyset = newKeyset(map[string]string{"fixture-key-a": "tenant-a", "fixture-key-b": "tenant-b"})
				handler := srv.Handler()
				codeA, traceA, _ := unkeyedWirePost(t, handler, wire.path, wire.root, "192.0.2.1:40001", map[string]string{wire.credential: "fixture-key-a", "X-Fak-Principal": "tenant-b"})
				codeB, traceB, _ := unkeyedWirePost(t, handler, wire.path, wire.root, "192.0.2.1:40001", map[string]string{wire.credential: "fixture-key-b", "X-Fak-Principal": "tenant-a"})
				codeAgain, traceAgain, _ := unkeyedWirePost(t, handler, wire.path, wire.root, "192.0.2.1:40002", map[string]string{wire.credential: "fixture-key-a", "X-Fak-Principal": "spoofed-new"})
				if codeA != http.StatusOK || codeB != http.StatusOK || traceA == traceB || codeAgain != http.StatusConflict || traceAgain != traceA {
					t.Fatalf("principal isolation/continuity lost: status=%d/%d/%d trace=%q/%q/%q", codeA, codeB, codeAgain, traceA, traceB, traceAgain)
				}
				if code, _, _ := unkeyedWirePost(t, handler, wire.path, "{", "192.0.2.1:40003", nil); code != http.StatusUnauthorized {
					t.Fatalf("unauthenticated malformed request status=%d, want 401", code)
				}
				if code, _, _ := unkeyedWirePost(t, handler, wire.path, "{", "192.0.2.1:40003", map[string]string{wire.credential: "fixture-key-a"}); code != http.StatusBadRequest {
					t.Fatalf("authenticated malformed request status=%d, want 400", code)
				}
			})
		})
	}
}
