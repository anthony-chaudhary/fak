package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

type featureTrackerGenericFastPath struct{}

func (featureTrackerGenericFastPath) Caps() []abi.Capability { return nil }

func (featureTrackerGenericFastPath) Lookup(_ context.Context, call *abi.ToolCall) (*abi.Result, bool) {
	if call == nil || call.Tool != "allow_read" {
		return nil, false
	}
	return &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"source":"generic-hit"}`)},
		Status:  abi.StatusOK,
		Meta:    map[string]string{"served_by": "vdso", "tier": "2"},
	}, true
}

func TestFeatureTrackerHTTPRecordsOnlyExecutedContextTransforms(t *testing.T) {
	tests := []struct {
		name    string
		feature ServeFeature
		body    func(*testing.T) []byte
		config  func(*Config)
	}{
		{
			name:    "compaction",
			feature: FeatureCompactHistory,
			body: func(t *testing.T) []byte {
				return featureTrackerDistinctCompactBody(t)
			},
			config: func(cfg *Config) { cfg.CompactHistoryBudget = 120 },
		},
		{
			name:    "result elision",
			feature: FeatureElideResults,
			body:    featureTrackerElideBody,
			config:  func(cfg *Config) { cfg.ElideResultBytes = 64 },
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterEngine("mock", engine.MockEngine)
			abi.RegisterAdjudicator(0, toolAdj{})

			var mu sync.Mutex
			var outbound [][]byte
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				mu.Lock()
				outbound = append(outbound, append([]byte(nil), raw...))
				mu.Unlock()
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"msg_test","type":"message","role":"assistant","model":"claude-test","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
			}))
			defer upstream.Close()

			var logsMu sync.Mutex
			var logs []string
			cfg := Config{
				EngineID: "test", Model: "claude-test", BaseURL: upstream.URL,
				Provider: "anthropic", APIKey: "upstream-key", PinUpstreamCredential: true,
				Logf: func(format string, args ...any) {
					logsMu.Lock()
					logs = append(logs, formatLog(format, args...))
					logsMu.Unlock()
				},
			}
			tc.config(&cfg)
			srv, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer srv.Close()
			catalog, err := NewFeatureCatalog([]FeatureStatus{{
				Feature: tc.feature, State: FeatureConfiguredActive,
				Provenance: FeatureCLIFlag, Description: "integration witness",
			}})
			if err != nil {
				t.Fatal(err)
			}
			if err := srv.SetFeatureCatalog(catalog); err != nil {
				t.Fatal(err)
			}
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			idle := []byte(`{"model":"claude-test","max_tokens":16,"messages":[{"role":"user","content":"small"}]}`)
			idleResp := postFeatureTrackerMessage(t, ts.URL, idle)
			if got := idleResp.Trailer.Get(HeaderFeaturesUsedFinal); got != "" {
				t.Fatalf("eligible feature was configured but idle request reported use: %q", got)
			}
			active := tc.body(t)
			activeResp := postFeatureTrackerMessage(t, ts.URL, active)
			if got := activeResp.Trailer.Get(HeaderFeaturesUsedFinal); got != string(tc.feature) {
				t.Fatalf("executed transform trailer = %q, want %q", got, tc.feature)
			}

			mu.Lock()
			calls := append([][]byte(nil), outbound...)
			mu.Unlock()
			if len(calls) != 2 {
				t.Fatalf("upstream calls = %d, want 2", len(calls))
			}
			if !bytes.Equal(calls[0], idle) {
				t.Fatal("idle request changed despite no feature activation")
			}
			if bytes.Equal(calls[1], active) || len(calls[1]) >= len(active) {
				t.Fatalf("active request did not produce a smaller outbound body: before=%d after=%d", len(active), len(calls[1]))
			}

			metrics := getMetrics(t, ts.URL+"/metrics", "")
			wantMetric := `fak_gateway_feature_used_requests_total{feature="` + string(tc.feature) + `"} 1`
			if !strings.Contains(metrics, wantMetric) {
				t.Fatalf("metrics disagree with two request witnesses; missing %q", wantMetric)
			}
			logsMu.Lock()
			access := featureTrackerAccessEvents(t, append([]string(nil), logs...))
			logsMu.Unlock()
			if len(access) < 2 || len(featureTrackerUsed(access[len(access)-2])) != 0 || !equalStrings(featureTrackerUsed(access[len(access)-1]), []string{string(tc.feature)}) {
				t.Fatalf("access feature_used does not match idle/active execution: %#v", access)
			}
		})
	}
}

func featureTrackerDistinctCompactBody(t *testing.T) []byte {
	t.Helper()
	var body map[string]any
	if err := json.Unmarshal(compactWireBody(t, 20), &body); err != nil {
		t.Fatal(err)
	}
	body["stream"] = false
	messages, _ := body["messages"].([]any)
	for i := 1; i < len(messages); i++ {
		message, _ := messages[i].(map[string]any)
		blocks, _ := message["content"].([]any)
		if len(blocks) == 0 {
			continue
		}
		block, _ := blocks[0].(map[string]any)
		if text, ok := block["text"].(string); ok {
			block["text"] = text + " distinct-turn-" + string(rune('a'+i))
		}
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestFeatureTrackerHTTPWitnessesWarmVDSOAndManifestPick(t *testing.T) {
	srv := routeServer(t, pickManifest("allow_read", "routed2"))
	abi.RegisterEngine("mock", engine.MockEngine)
	v := vdso.New(vdso.DefaultCacheSize)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)
	var logsMu sync.Mutex
	var logs []string
	srv.logf = func(format string, args ...any) {
		logsMu.Lock()
		logs = append(logs, formatLog(format, args...))
		logsMu.Unlock()
	}
	catalog, err := NewFeatureCatalog([]FeatureStatus{
		{Feature: FeatureVDSO, State: FeatureConfiguredActive, Provenance: FeatureCLIFlag, Description: "integration witness"},
		{Feature: FeatureRouteManifest, State: FeatureConfiguredActive, Provenance: FeatureManifest, Description: "integration witness"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := srv.SetFeatureCatalog(catalog); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SyscallRequest{Tool: "allow_read", Arguments: json.RawMessage(`{"path":"same"}`), ReadOnly: true})
	first := postFeatureTrackerSyscall(t, ts.URL, body)
	second := postFeatureTrackerSyscall(t, ts.URL, body)
	if got := first.Trailer.Get(HeaderFeaturesUsedFinal); got != "route_manifest" {
		t.Fatalf("cold routed call used = %q, want route_manifest", got)
	}
	if got := second.Trailer.Get(HeaderFeaturesUsedFinal); got != "route_manifest,vdso" {
		t.Fatalf("warm routed call used = %q, want route_manifest,vdso", got)
	}

	metrics := getMetrics(t, ts.URL+"/metrics", "")
	for _, want := range []string{
		`fak_gateway_feature_used_requests_total{feature="route_manifest"} 2`,
		`fak_gateway_feature_used_requests_total{feature="vdso"} 1`,
	} {
		if !strings.Contains(metrics, want) {
			t.Errorf("metrics missing %q", want)
		}
	}
	logsMu.Lock()
	access := featureTrackerAccessEvents(t, append([]string(nil), logs...))
	logsMu.Unlock()
	if len(access) < 2 || !equalStrings(featureTrackerUsed(access[len(access)-2]), []string{"route_manifest"}) || !equalStrings(featureTrackerUsed(access[len(access)-1]), []string{"route_manifest", "vdso"}) {
		t.Fatalf("access log does not distinguish cold dispatch from actual warm hit: %#v", access)
	}
}

func TestFeatureTrackerHTTPDoesNotMislabelGenericFastPathAsVDSO(t *testing.T) {
	srv := routeServer(t, nil)
	abi.RegisterFastPath(1, featureTrackerGenericFastPath{})
	catalog, err := NewFeatureCatalog([]FeatureStatus{{
		Feature: FeatureVDSO, State: FeatureConfiguredActive,
		Provenance: FeatureCLIFlag, Description: "eligible but not used",
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := srv.SetFeatureCatalog(catalog); err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, _ := json.Marshal(SyscallRequest{Tool: "allow_read", Arguments: json.RawMessage(`{"path":"same"}`), ReadOnly: true})
	resp, err := http.Post(ts.URL+"/v1/fak/syscall", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, readErr := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if readErr != nil {
		t.Fatal(readErr)
	}
	if resp.StatusCode != http.StatusOK || !bytes.Contains(raw, []byte("generic-hit")) {
		t.Fatalf("generic fast-path response = %d %s, want accepted cached payload", resp.StatusCode, raw)
	}
	if got := resp.Trailer.Get(HeaderFeaturesUsedFinal); got != "" {
		t.Fatalf("generic fast path falsely reported vDSO activation: %q", got)
	}
	metrics := getMetrics(t, ts.URL+"/metrics", "")
	if strings.Contains(metrics, `fak_gateway_feature_used_requests_total{feature="vdso"} 1`) {
		t.Fatal("generic fast path falsely incremented vDSO usage metric")
	}
}

func postFeatureTrackerMessage(t *testing.T, base string, body []byte) *http.Response {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, base+"/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/messages = %d: %s", resp.StatusCode, raw)
	}
	return resp
}

func postFeatureTrackerSyscall(t *testing.T, base string, body []byte) *http.Response {
	t.Helper()
	resp, err := http.Post(base+"/v1/fak/syscall", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/fak/syscall = %d: %s", resp.StatusCode, raw)
	}
	return resp
}

func featureTrackerElideBody(t *testing.T) []byte {
	t.Helper()
	type object = map[string]any
	cc := object{"type": "ephemeral"}
	text := func(role, value string) object {
		return object{"role": role, "content": []object{{"type": "text", "text": value}}}
	}
	toolResult := func(id, value string) object {
		return object{
			"role": "user",
			"content": []object{{
				"type": "tool_result", "tool_use_id": id,
				"content": []object{{"type": "text", "text": value}},
			}},
		}
	}
	body := object{
		"model": "claude-test", "max_tokens": 16,
		"system": []object{{"type": "text", "text": "policy", "cache_control": cc}},
		"messages": []object{
			{"role": "user", "content": []object{{"type": "text", "text": "cached head", "cache_control": cc}}},
			{"role": "assistant", "content": []object{{"type": "tool_use", "id": "old", "name": "allow_read", "input": object{"path": "old"}}}},
			toolResult("old", strings.Repeat("x", 2048)), text("assistant", "a3"),
			text("user", "u4"), text("assistant", "a5"), text("user", "u6"), text("assistant", "a7"),
		},
	}
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func featureTrackerAccessEvents(t *testing.T, lines []string) []map[string]any {
	t.Helper()
	var events []map[string]any
	for _, line := range lines {
		var event map[string]any
		if json.Unmarshal([]byte(line), &event) == nil && event["event"] == "gateway_http_request" && (event["route"] == "/v1/messages" || event["route"] == "/v1/fak/syscall") {
			events = append(events, event)
		}
	}
	return events
}

func featureTrackerUsed(event map[string]any) []string {
	raw, _ := event["feature_used"].([]any)
	result := make([]string, 0, len(raw))
	for _, value := range raw {
		if name, ok := value.(string); ok {
			result = append(result, name)
		}
	}
	return result
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
