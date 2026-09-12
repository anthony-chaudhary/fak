package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

type chatRouteCall struct {
	method string
	path   string
	body   []byte
	header http.Header
}

type chatRouteProgress struct {
	mark    string
	release chan struct{}
}

type chatRouteUpstream struct {
	name     string
	server   *httptest.Server
	progress chan chatRouteProgress
	mu       sync.Mutex
	calls    []chatRouteCall
}

func newChatRouteUpstream(t *testing.T, name string) *chatRouteUpstream {
	t.Helper()
	u := &chatRouteUpstream{name: name, progress: make(chan chatRouteProgress, 16)}
	u.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		u.mu.Lock()
		u.calls = append(u.calls, chatRouteCall{method: r.Method, path: r.URL.Path, body: append([]byte(nil), raw...), header: r.Header.Clone()})
		u.mu.Unlock()

		var req struct {
			Stream bool `json:"stream"`
		}
		_ = json.Unmarshal(raw, &req)
		mark := "reply-" + name
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"model":"served-`+name+`","choices":[{"message":{"role":"assistant","content":"`+mark+`"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		_, _ = io.WriteString(w, `data: {"model":"served-`+name+`","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n"+
			`data: {"choices":[{"delta":{"content":"`+mark+`"},"finish_reason":null}]}`+"\n\n")
		flusher.Flush()
		release := make(chan struct{})
		u.progress <- chatRouteProgress{mark: mark, release: release}
		select {
		case <-release:
		case <-r.Context().Done():
			return
		case <-time.After(3 * time.Second):
			return
		}
		_, _ = io.WriteString(w, `data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`+"\n\n"+"data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(u.server.Close)
	return u
}

func (u *chatRouteUpstream) snapshot() []chatRouteCall {
	u.mu.Lock()
	defer u.mu.Unlock()
	out := make([]chatRouteCall, len(u.calls))
	copy(out, u.calls)
	return out
}

func chatRouteBody(path, model string, stream bool) []byte {
	var body map[string]any
	switch {
	case path == "/v1/messages":
		body = map[string]any{"max_tokens": 64, "messages": []map[string]string{{"role": "user", "content": "route me"}}, "stream": stream, "temperature": 0.25}
	case path == "/v1/completions":
		body = map[string]any{"prompt": "route me", "stream": stream, "temperature": 0.25}
	case path == "/v1/responses":
		body = map[string]any{"input": "route me", "stream": stream, "temperature": 0.25}
	case strings.HasPrefix(path, "/v1beta/models/"):
		body = map[string]any{"contents": []map[string]any{{"role": "user", "parts": []map[string]string{{"text": "route me"}}}}}
	default:
		body = map[string]any{"messages": []map[string]string{{"role": "user", "content": "route me"}}, "stream": stream, "temperature": 0.25}
	}
	if model != "" && !strings.HasPrefix(path, "/v1beta/models/") {
		body["model"] = model
	}
	raw, _ := json.Marshal(body)
	return raw
}

func chatRoutePublicPaths(model string) []string {
	return []string{
		"/v1/chat/completions",
		"/v1/messages",
		"/v1/completions",
		"/v1/responses",
		fmt.Sprintf("/v1beta/models/%s:generateContent", model),
	}
}

func postChatRoute(t *testing.T, url, path, model, key string, stream bool, progress <-chan chatRouteProgress) (int, []byte) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url+path, bytes.NewReader(chatRouteBody(path, model, stream)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST %s: %v", path, err)
	}
	defer resp.Body.Close()
	if !stream || resp.StatusCode/100 != 2 {
		raw, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, raw
	}

	var gate chatRouteProgress
	select {
	case gate = <-progress:
	case <-time.After(2 * time.Second):
		t.Fatal("routed upstream did not expose a live progress gate")
	}
	scanner := bufio.NewScanner(resp.Body)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		if strings.Contains(line, gate.mark) {
			close(gate.release)
			break
		}
	}
	if len(lines) == 0 || !strings.Contains(strings.Join(lines, "\n"), gate.mark) {
		close(gate.release)
		t.Fatalf("downstream did not expose %q before terminal release: %q", gate.mark, lines)
	}
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	return resp.StatusCode, []byte(strings.Join(lines, "\n"))
}

func TestChatWireResolvesRosterRoute(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	boot := newChatRouteUpstream(t, "boot")
	targetA := newChatRouteUpstream(t, "a")
	targetB := newChatRouteUpstream(t, "b")
	t.Setenv("FAK_CHAT_ROUTE_A_KEY", "target-a-secret")
	t.Setenv("FAK_CHAT_ROUTE_B_KEY", "target-b-secret")

	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "account-a", Kind: modelroute.KindOpenAI, BaseURL: targetA.server.URL, CredEnv: "FAK_CHAT_ROUTE_A_KEY", Principals: []string{"tenant-a"}},
			{ID: "account-b", Kind: modelroute.KindOpenAI, BaseURL: targetB.server.URL, CredEnv: "FAK_CHAT_ROUTE_B_KEY", Principals: []string{"tenant-b"}},
		},
		Bindings: []modelroute.Binding{
			{Model: "model-a", Account: "account-a", UpstreamModel: "upstream-alias-a"},
			{Model: "model-b", Account: "account-b", UpstreamModel: "upstream-alias-b"},
		},
		Default: "account-a",
	}
	keys := map[string]string{"gateway-a": "tenant-a", "gateway-b": "tenant-b"}
	newServer := func(routeAccounts *modelroute.Roster) *Server {
		srv, err := New(Config{EngineID: "test", Model: "boot-model", BaseURL: boot.server.URL, Provider: "openai-compatible", APIKey: "boot-secret", RouteAccounts: routeAccounts, KeyPrincipals: keys})
		if err != nil {
			t.Fatalf("New: %v", err)
		}
		t.Cleanup(srv.Close)
		return srv
	}
	routedHTTP := httptest.NewServer(newServer(roster).Handler())
	t.Cleanup(routedHTTP.Close)
	plainHTTP := httptest.NewServer(newServer(nil).Handler())
	t.Cleanup(plainHTTP.Close)

	routes := []struct {
		model, alias, key string
		upstream          *chatRouteUpstream
	}{
		{"model-a", "upstream-alias-a", "gateway-a", targetA},
		{"model-b", "upstream-alias-b", "gateway-b", targetB},
	}
	for _, route := range routes {
		for _, path := range []string{"/v1/chat/completions", "/v1/messages"} {
			for _, stream := range []bool{false, true} {
				name := route.model + strings.TrimPrefix(path, "/v1/")
				if stream {
					name += "-stream"
				}
				t.Run(name, func(t *testing.T) {
					before := len(route.upstream.snapshot())
					status, response := postChatRoute(t, routedHTTP.URL, path, route.model, route.key, stream, route.upstream.progress)
					if status != http.StatusOK || !bytes.Contains(response, []byte("reply-"+route.upstream.name)) {
						t.Fatalf("status=%d response=%s", status, response)
					}
					if stream {
						terminal := "[DONE]"
						if path == "/v1/messages" {
							terminal = "message_stop"
						}
						if !bytes.Contains(response, []byte(terminal)) {
							t.Fatalf("stream missing terminal %q: %s", terminal, response)
						}
					}
					calls := route.upstream.snapshot()
					if len(calls) != before+1 {
						t.Fatalf("target calls=%d, want %d", len(calls), before+1)
					}
					call := calls[len(calls)-1]
					if call.path != "/chat/completions" || call.header.Get("Authorization") != "Bearer target-"+route.upstream.name+"-secret" {
						t.Fatalf("target path/auth = %q/%q", call.path, call.header.Get("Authorization"))
					}
					var outbound map[string]any
					_ = json.Unmarshal(call.body, &outbound)
					if outbound["model"] != route.alias {
						t.Fatalf("outbound model=%v, want %q; client sampling options overrode the binding", outbound["model"], route.alias)
					}
				})
			}
		}
		for _, path := range chatRoutePublicPaths(route.model)[2:] {
			t.Run(route.model+strings.TrimPrefix(path, "/v1/"), func(t *testing.T) {
				before := len(route.upstream.snapshot())
				status, response := postChatRoute(t, routedHTTP.URL, path, route.model, route.key, false, nil)
				if status != http.StatusOK || !bytes.Contains(response, []byte("reply-"+route.upstream.name)) {
					t.Fatalf("status=%d response=%s", status, response)
				}
				calls := route.upstream.snapshot()
				if len(calls) != before+1 {
					t.Fatalf("target calls=%d, want %d", len(calls), before+1)
				}
				call := calls[len(calls)-1]
				if call.path != "/chat/completions" || call.header.Get("Authorization") != "Bearer target-"+route.upstream.name+"-secret" {
					t.Fatalf("target path/auth = %q/%q", call.path, call.header.Get("Authorization"))
				}
				var outbound map[string]any
				_ = json.Unmarshal(call.body, &outbound)
				if outbound["model"] != route.alias {
					t.Fatalf("outbound model=%v, want %q", outbound["model"], route.alias)
				}
			})
		}
	}
	if got := len(boot.snapshot()); got != 0 {
		t.Fatalf("roster hits reached boot upstream: %d calls", got)
	}

	beforeAll := len(boot.snapshot()) + len(targetA.snapshot()) + len(targetB.snapshot())
	for _, path := range chatRoutePublicPaths("model-b") {
		status, _ := postChatRoute(t, routedHTTP.URL, path, "model-b", "gateway-a", false, nil)
		if status/100 == 2 {
			t.Errorf("%s admitted tenant-a to account-b", path)
		}
	}
	afterAll := len(boot.snapshot()) + len(targetA.snapshot()) + len(targetB.snapshot())
	if afterAll != beforeAll {
		t.Fatalf("principal-refused requests made outbound calls: before=%d after=%d", beforeAll, afterAll)
	}

	stableHeaders := func(h http.Header) map[string]string {
		return map[string]string{"Authorization": h.Get("Authorization"), "Content-Type": h.Get("Content-Type"), "Accept": h.Get("Accept")}
	}
	for _, path := range []string{"/v1/chat/completions", "/v1/messages"} {
		for _, model := range []string{"unrostered-model", ""} {
			start := len(boot.snapshot())
			plainStatus, _ := postChatRoute(t, plainHTTP.URL, path, model, "gateway-a", false, nil)
			routedStatus, _ := postChatRoute(t, routedHTTP.URL, path, model, "gateway-a", false, nil)
			if plainStatus != routedStatus {
				t.Fatalf("roster miss changed %s model=%q status: plain=%d routed=%d", path, model, plainStatus, routedStatus)
			}
			calls := boot.snapshot()
			if len(calls) != start+2 {
				t.Fatalf("boot calls=%d, want %d", len(calls), start+2)
			}
			plain, miss := calls[start], calls[start+1]
			if plain.method != miss.method || plain.path != miss.path || !bytes.Equal(plain.body, miss.body) || !mapsEqual(stableHeaders(plain.header), stableHeaders(miss.header)) {
				t.Fatalf("roster miss changed outbound request:\nplain=%+v\nmiss=%+v", plain, miss)
			}
			// The OpenAI wire historically passes a requested model through. The
			// Anthropic-to-generic planner path only does so for DualPlanner; its
			// parity contract here is the exact no-roster/miss body comparison above.
			if path == "/v1/chat/completions" && model != "" && !bytes.Contains(miss.body, []byte(`"model":"`+model+`"`)) {
				t.Fatalf("roster miss lost client model %q: %s", model, miss.body)
			}
		}
	}
	for _, call := range boot.snapshot() {
		if call.header.Get("Authorization") != "Bearer boot-secret" || bytes.Contains(call.body, []byte("target-a-secret")) || bytes.Contains(call.body, []byte("target-b-secret")) {
			t.Fatalf("boot upstream credential boundary violated: header=%q body=%s", call.header.Get("Authorization"), call.body)
		}
	}
}

func TestChatWireRosterRoutePrecedesEPFanout(t *testing.T) {
	for _, path := range chatRoutePublicPaths("model-a") {
		path := path
		streams := []bool{false}
		if path == "/v1/chat/completions" || path == "/v1/messages" {
			streams = append(streams, true)
		}
		for _, stream := range streams {
			stream := stream
			name := strings.TrimPrefix(path, "/")
			if stream {
				name += "-streaming"
			}
			t.Run(name, func(t *testing.T) {
				abi.ResetForTest()
				abi.RegisterRegionBackend(inlineBackend{})
				abi.RegisterEngine("test", echoEngine{})
				abi.RegisterAdjudicator(0, toolAdj{})

				boot := newChatRouteUpstream(t, "boot")
				targetA := newChatRouteUpstream(t, "a")
				targetB := newChatRouteUpstream(t, "b")
				followerCalls := startRecordingFollowerRank(t)
				t.Setenv("FAK_CHAT_ROUTE_EP_A_KEY", "target-a-secret")
				// Leave B's credential empty. Admission must reject tenant A before
				// resolving that account's credential or starting any execution.
				t.Setenv("FAK_CHAT_ROUTE_EP_B_KEY", "")

				roster := &modelroute.Roster{
					Version: modelroute.RosterVersion,
					Accounts: []modelroute.Account{
						{ID: "account-a", Kind: modelroute.KindOpenAI, BaseURL: targetA.server.URL, CredEnv: "FAK_CHAT_ROUTE_EP_A_KEY", Principals: []string{"tenant-a"}},
						{ID: "account-b", Kind: modelroute.KindOpenAI, BaseURL: targetB.server.URL, CredEnv: "FAK_CHAT_ROUTE_EP_B_KEY", Principals: []string{"tenant-b"}},
					},
					Bindings: []modelroute.Binding{
						{Model: "model-a", Account: "account-a", UpstreamModel: "upstream-alias-a"},
						{Model: "model-b", Account: "account-b", UpstreamModel: "upstream-alias-b"},
					},
					Default: "account-a",
				}
				srv, err := New(Config{
					EngineID: "test", Model: "boot-model", BaseURL: boot.server.URL,
					Provider: "openai-compatible", APIKey: "boot-secret", RouteAccounts: roster,
					KeyPrincipals: map[string]string{"gateway-a": "tenant-a", "gateway-b": "tenant-b"},
				})
				if err != nil {
					t.Fatalf("New: %v", err)
				}
				t.Cleanup(srv.Close)
				ts := httptest.NewServer(srv.Handler())
				t.Cleanup(ts.Close)

				status, response := postChatRoute(t, ts.URL, path, "model-a", "gateway-a", stream, targetA.progress)
				if status != http.StatusOK || !bytes.Contains(response, []byte("reply-a")) {
					t.Fatalf("bound request status=%d response=%s", status, response)
				}
				if got := len(targetA.snapshot()); got != 1 {
					t.Fatalf("bound target calls=%d, want 1", got)
				}
				if got := len(boot.snapshot()); got != 0 {
					t.Fatalf("bound request reached boot execution: %d calls", got)
				}
				assertNoEPFollowerCall(t, followerCalls, "bound roster request")

				deniedPath := strings.Replace(path, "model-a", "model-b", 1)
				status, _ = postChatRoute(t, ts.URL, deniedPath, "model-b", "gateway-a", stream, nil)
				if status != http.StatusForbidden {
					t.Fatalf("denied request status=%d, want %d", status, http.StatusForbidden)
				}
				if got := len(boot.snapshot()) + len(targetA.snapshot()) + len(targetB.snapshot()); got != 1 {
					t.Fatalf("denied request started execution: aggregate calls=%d, want prior bound call only", got)
				}
				assertNoEPFollowerCall(t, followerCalls, "principal-denied roster request")

				unboundModel := "unrostered-model"
				unboundPath := strings.Replace(path, "model-a", unboundModel, 1)
				wantBody := chatRouteBody(unboundPath, unboundModel, stream)
				status, response = postChatRoute(t, ts.URL, unboundPath, unboundModel, "gateway-a", stream, boot.progress)
				if status != http.StatusOK || !bytes.Contains(response, []byte("reply-boot")) {
					t.Fatalf("unbound request status=%d response=%s", status, response)
				}
				if got := len(boot.snapshot()); got != 1 {
					t.Fatalf("unbound boot calls=%d, want 1", got)
				}
				select {
				case call := <-followerCalls:
					if call.path != chatRouteEPRoute(path) || call.header != "1" || call.body != string(wantBody) {
						t.Fatalf("unbound EP mirror = path %q header %q body %q, want path %q header 1 original body %q", call.path, call.header, call.body, chatRouteEPRoute(path), wantBody)
					}
				case <-time.After(time.Second):
					t.Fatal("unbound request did not preserve EP follower fanout")
				}
			})
		}
	}
}

func TestNativeMessagesResolvesRosterRoute(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	boot := newChatRouteUpstream(t, "boot")
	targetA := newChatRouteUpstream(t, "a")
	targetB := newChatRouteUpstream(t, "b")
	t.Setenv("FAK_CHAT_ROUTE_NATIVE_A_KEY", "target-a-secret")
	t.Setenv("FAK_CHAT_ROUTE_NATIVE_B_KEY", "target-b-secret")
	var logMu sync.Mutex
	var logs []string
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "account-a", Kind: modelroute.KindOpenAI, BaseURL: targetA.server.URL, CredEnv: "FAK_CHAT_ROUTE_NATIVE_A_KEY", Principals: []string{"tenant-a"}},
			{ID: "account-b", Kind: modelroute.KindOpenAI, BaseURL: targetB.server.URL, CredEnv: "FAK_CHAT_ROUTE_NATIVE_B_KEY", Principals: []string{"tenant-b"}},
		},
		Bindings: []modelroute.Binding{
			{Model: "model-a", Account: "account-a", UpstreamModel: "upstream-alias-a"},
			{Model: "model-b", Account: "account-b", UpstreamModel: "upstream-alias-b"},
		},
		Default: "account-a",
	}
	srv, err := New(Config{
		EngineID: "test", Model: "boot-model", BaseURL: boot.server.URL,
		Provider: "openai-compatible", APIKey: "boot-secret", RouteAccounts: roster,
		KeyPrincipals: map[string]string{"gateway-a": "tenant-a", "gateway-b": "tenant-b"},
		Native:        true, NativeMaxTurns: 2,
		Logf: func(format string, args ...any) {
			logMu.Lock()
			logs = append(logs, fmt.Sprintf(format, args...))
			logMu.Unlock()
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	for _, stream := range []bool{false, true} {
		name := "buffered"
		if stream {
			name = "streaming"
		}
		t.Run(name, func(t *testing.T) {
			beforeBoot := len(boot.snapshot())
			beforeTarget := len(targetA.snapshot())
			beforeMetrics := srv.metrics.adjudicationSummary()
			logMu.Lock()
			beforeLogs := len(logs)
			logMu.Unlock()
			status, response := postChatRoute(t, ts.URL, "/v1/messages", "model-a", "gateway-a", stream, targetA.progress)
			if status != http.StatusOK || !bytes.Contains(response, []byte("reply-a")) {
				t.Fatalf("bound native request status=%d response=%s", status, response)
			}
			if got := len(targetA.snapshot()); got != beforeTarget+1 {
				t.Fatalf("bound native target calls=%d, want %d", got, beforeTarget+1)
			}
			if got := len(boot.snapshot()); got != beforeBoot {
				t.Fatalf("bound native request reached boot planner: calls %d, want unchanged %d", got, beforeBoot)
			}
			call := targetA.snapshot()[beforeTarget]
			if call.header.Get("Authorization") != "Bearer target-a-secret" {
				t.Fatalf("bound native target auth=%q", call.header.Get("Authorization"))
			}
			var outbound map[string]any
			if err := json.Unmarshal(call.body, &outbound); err != nil {
				t.Fatalf("decode bound native outbound body: %v", err)
			}
			if outbound["model"] != "upstream-alias-a" {
				t.Fatalf("bound native outbound model=%v, want upstream alias", outbound["model"])
			}
			afterBoundMetrics := srv.metrics.adjudicationSummary()
			if afterBoundMetrics.VendorTurns != beforeMetrics.VendorTurns+1 {
				t.Fatalf("bound native vendor turns=%d, want %d", afterBoundMetrics.VendorTurns, beforeMetrics.VendorTurns+1)
			}
			logMu.Lock()
			boundLogs := append([]string(nil), logs[beforeLogs:]...)
			logMu.Unlock()
			boundLoggedAlias := false
			for _, line := range boundLogs {
				if strings.Contains(line, `"event":"gateway_inference_turn"`) && strings.Contains(line, `"wire":"anthropic_messages_native"`) && strings.Contains(line, `"model":"upstream-alias-a"`) {
					boundLoggedAlias = true
					break
				}
			}
			if !boundLoggedAlias {
				t.Fatalf("bound native completion did not log upstream alias: %q", boundLogs)
			}

			beforeAll := len(boot.snapshot()) + len(targetA.snapshot()) + len(targetB.snapshot())
			status, _ = postChatRoute(t, ts.URL, "/v1/messages", "model-b", "gateway-a", stream, nil)
			if status != http.StatusForbidden {
				t.Fatalf("denied native request status=%d, want %d", status, http.StatusForbidden)
			}
			afterAll := len(boot.snapshot()) + len(targetA.snapshot()) + len(targetB.snapshot())
			if afterAll != beforeAll {
				t.Fatalf("denied native request made outbound calls: before=%d after=%d", beforeAll, afterAll)
			}
			if afterDeniedMetrics := srv.metrics.adjudicationSummary(); afterDeniedMetrics.VendorTurns != afterBoundMetrics.VendorTurns {
				t.Fatalf("denied native request changed vendor turns: before=%d after=%d", afterBoundMetrics.VendorTurns, afterDeniedMetrics.VendorTurns)
			}

			beforeBoot = len(boot.snapshot())
			status, response = postChatRoute(t, ts.URL, "/v1/messages", "unrostered-model", "gateway-a", stream, boot.progress)
			if status != http.StatusOK || !bytes.Contains(response, []byte("reply-boot")) {
				t.Fatalf("unbound native request status=%d response=%s", status, response)
			}
			if got := len(boot.snapshot()); got != beforeBoot+1 {
				t.Fatalf("unbound native boot calls=%d, want %d", got, beforeBoot+1)
			}
			if afterMissMetrics := srv.metrics.adjudicationSummary(); afterMissMetrics.VendorTurns != afterBoundMetrics.VendorTurns {
				t.Fatalf("unbound native miss changed vendor turns: before=%d after=%d", afterBoundMetrics.VendorTurns, afterMissMetrics.VendorTurns)
			}
		})
	}
}

func assertNoEPFollowerCall(t *testing.T, calls <-chan epFollowerCall, requestKind string) {
	t.Helper()
	select {
	case call := <-calls:
		t.Fatalf("%s reached EP follower: path=%q body=%q", requestKind, call.path, call.body)
	default:
	}
}

func chatRouteEPRoute(path string) string {
	switch {
	case path == "/v1/messages":
		return epRouteMessages
	case path == "/v1/completions":
		return epRouteCompletions
	case path == "/v1/responses":
		return epRouteResponses
	case strings.HasPrefix(path, "/v1beta/models/"):
		return epRouteGeminiGenerateContent
	default:
		return epRouteChatCompletions
	}
}

func mapsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if b[k] != v {
			return false
		}
	}
	return true
}
