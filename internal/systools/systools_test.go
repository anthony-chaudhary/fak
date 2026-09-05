package systools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestCatalog(t *testing.T) {
	cat := Catalog()
	if len(cat) != 3 {
		t.Fatalf("Catalog() returned %d tools, want 3", len(cat))
	}
	expected := map[string]bool{
		ToolGetTime:   true,
		ToolFetchWeb:  true,
		ToolWebSearch: true,
	}
	for _, tool := range cat {
		if !expected[tool.Name] {
			t.Errorf("unexpected tool %s in catalog", tool.Name)
		}
		if !tool.ReadOnly {
			t.Errorf("tool %s should be ReadOnly", tool.Name)
		}
		if len(tool.Parameters) == 0 {
			t.Errorf("tool %s missing parameters", tool.Name)
		}
	}
}

func TestGetTime(t *testing.T) {
	ts, err := New(Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 1. Default (UTC)
	raw, isErr := ts.getTime(context.Background(), []byte(`{}`))
	if isErr {
		t.Fatalf("getTime failed: %s", string(raw))
	}
	var res map[string]any
	if err := json.Unmarshal(raw, &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if res["timezone"] != "UTC" {
		t.Errorf("timezone = %v, want UTC", res["timezone"])
	}
	if _, ok := res["epoch_seconds"]; !ok {
		t.Errorf("missing epoch_seconds")
	}
	if _, ok := res["epoch_millis"]; !ok {
		t.Errorf("missing epoch_millis")
	}

	// 2. Specific timezone
	rawTZ, isErr := ts.getTime(context.Background(), []byte(`{"timezone":"America/New_York"}`))
	if isErr {
		t.Fatalf("getTime with timezone failed: %s", string(rawTZ))
	}
	var resTZ map[string]any
	if err := json.Unmarshal(rawTZ, &resTZ); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resTZ["timezone"] != "America/New_York" {
		t.Errorf("timezone = %v, want America/New_York", resTZ["timezone"])
	}

	// 3. Invalid timezone
	rawErr, isErr := ts.getTime(context.Background(), []byte(`{"timezone":"Invalid/NonExistent"}`))
	if !isErr {
		t.Fatalf("expected error for invalid timezone, got: %s", string(rawErr))
	}

	// 4. Unknown fields disallowed
	rawUnk, isErr := ts.getTime(context.Background(), []byte(`{"unknown_field":"value"}`))
	if !isErr {
		t.Fatalf("expected error for unknown field, got: %s", string(rawUnk))
	}
}

func TestFetchWebLocalServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, "Hello from local server")
	}))
	defer srv.Close()

	// With AllowPrivateIPs: true, local test server is reachable.
	ts, err := New(Config{AllowPrivateIPs: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	reqBody := fmt.Sprintf(`{"url":%q}`, srv.URL)
	raw, isErr := ts.fetchWeb(context.Background(), []byte(reqBody))
	if isErr {
		t.Fatalf("fetchWeb failed: %s", string(raw))
	}
	var res map[string]any
	if err := json.Unmarshal(raw, &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if res["content"] != "Hello from local server" {
		t.Errorf("content = %v, want 'Hello from local server'", res["content"])
	}
	if res["status_code"] != float64(200) {
		t.Errorf("status_code = %v, want 200", res["status_code"])
	}
	if res["truncated"] != false {
		t.Errorf("truncated = %v, want false", res["truncated"])
	}
}

func TestFetchWebSSRFBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "should not be reached")
	}))
	defer srv.Close()

	// Default has AllowPrivateIPs: false.
	ts, err := New(Config{AllowPrivateIPs: false})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	testCases := []string{
		srv.URL,
		"http://127.0.0.1:8080/secret",
		"http://localhost:3000/api",
		"http://10.0.0.1/admin",
		"http://192.168.1.1/config",
		"http://172.16.0.1/status",
		"http://169.254.169.254/latest/meta-data",
	}

	for _, u := range testCases {
		reqBody := fmt.Sprintf(`{"url":%q}`, u)
		// 1. Engine check
		raw, isErr := ts.fetchWeb(context.Background(), []byte(reqBody))
		if !isErr {
			t.Errorf("fetchWeb unexpectedly allowed private IP: %s (got %s)", u, string(raw))
		}
		if !strings.Contains(string(raw), CodeSSRFBlock) {
			t.Errorf("fetchWeb expected %s for %s, got: %s", CodeSSRFBlock, u, string(raw))
		}

		// 2. Adjudicator rung check
		call := &abi.ToolCall{
			Tool: ToolFetchWeb,
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(reqBody), Len: int64(len(reqBody))},
		}
		v := ts.Adjudicate(context.Background(), call)
		if v.Kind != abi.VerdictDeny {
			t.Errorf("Adjudicate expected VerdictDeny for %s, got %+v", u, v)
		}
		if v.Reason != abi.ReasonPolicyBlock {
			t.Errorf("Adjudicate expected ReasonPolicyBlock for %s, got %+v", u, v)
		}
	}
}

func TestFetchWebDomainAllowlist(t *testing.T) {
	ts, err := New(Config{
		AllowPrivateIPs: true,
		AllowedDomains:  []string{"allowed.example.com", "*.trusted.org"},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 1. Denied domain
	deniedBody := `{"url":"https://denied.example.com/data"}`
	call := &abi.ToolCall{
		Tool: ToolFetchWeb,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(deniedBody), Len: int64(len(deniedBody))},
	}
	v := ts.Adjudicate(context.Background(), call)
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("Adjudicate expected VerdictDeny for denied domain, got %+v", v)
	}
	if v.Reason != abi.ReasonPolicyBlock {
		t.Errorf("expected ReasonPolicyBlock, got %v", v.Reason)
	}

	raw, isErr := ts.fetchWeb(context.Background(), []byte(deniedBody))
	if !isErr || !strings.Contains(string(raw), CodePolicyBlock) {
		t.Errorf("fetchWeb expected POLICY_BLOCK, got %s", string(raw))
	}

	// 2. Allowed domain should pass adjudicator check
	allowedBody := `{"url":"https://allowed.example.com/data"}`
	callAllowed := &abi.ToolCall{
		Tool: ToolFetchWeb,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(allowedBody), Len: int64(len(allowedBody))},
	}
	vAllowed := ts.Adjudicate(context.Background(), callAllowed)
	if vAllowed.Kind != abi.VerdictAllow {
		t.Errorf("Adjudicate expected VerdictAllow for allowed domain, got %+v", vAllowed)
	}

	// 3. Wildcard domain should pass
	wildcardBody := `{"url":"https://sub.trusted.org/info"}`
	callWildcard := &abi.ToolCall{
		Tool: ToolFetchWeb,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(wildcardBody), Len: int64(len(wildcardBody))},
	}
	vWildcard := ts.Adjudicate(context.Background(), callWildcard)
	if vWildcard.Kind != abi.VerdictAllow {
		t.Errorf("Adjudicate expected VerdictAllow for wildcard match, got %+v", vWildcard)
	}
}

func TestFetchWebByteCapTruncation(t *testing.T) {
	longText := strings.Repeat("A", 500)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprint(w, longText)
	}))
	defer srv.Close()

	ts, err := New(Config{
		AllowPrivateIPs:      true,
		DefaultMaxFetchBytes: 100,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 1. Default cap (100 bytes)
	reqBody := fmt.Sprintf(`{"url":%q}`, srv.URL)
	raw, isErr := ts.fetchWeb(context.Background(), []byte(reqBody))
	if isErr {
		t.Fatalf("fetchWeb failed: %s", string(raw))
	}
	var res map[string]any
	if err := json.Unmarshal(raw, &res); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if res["truncated"] != true {
		t.Errorf("truncated = %v, want true", res["truncated"])
	}
	if res["bytes"] != float64(100) {
		t.Errorf("bytes = %v, want 100", res["bytes"])
	}
	if len(res["content"].(string)) != 100 {
		t.Errorf("content len = %d, want 100", len(res["content"].(string)))
	}

	// 2. Explicit max_bytes parameter
	reqBody2 := fmt.Sprintf(`{"url":%q,"max_bytes":50}`, srv.URL)
	raw2, isErr2 := ts.fetchWeb(context.Background(), []byte(reqBody2))
	if isErr2 {
		t.Fatalf("fetchWeb failed: %s", string(raw2))
	}
	var res2 map[string]any
	if err := json.Unmarshal(raw2, &res2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if res2["bytes"] != float64(50) {
		t.Errorf("bytes = %v, want 50", res2["bytes"])
	}
	if res2["truncated"] != true {
		t.Errorf("truncated = %v, want true", res2["truncated"])
	}
}

func TestWebSearch(t *testing.T) {
	// 1. Unconfigured search returns explicit error
	ts, err := New(Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	raw, isErr := ts.webSearch(context.Background(), []byte(`{"query":"agent kernel"}`))
	if !isErr {
		t.Fatalf("expected error for unconfigured search, got: %s", string(raw))
	}
	if !strings.Contains(string(raw), "search backend unconfigured") {
		t.Errorf("expected 'search backend unconfigured', got: %s", string(raw))
	}

	// 2. Custom search adapter
	customTS, err := New(Config{
		SearchAdapter: func(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
			return []SearchResult{
				{Title: "Custom Result", URL: "https://example.com/custom", Snippet: "Custom snippet for " + query},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("New custom: %v", err)
	}

	rawCustom, isErr := customTS.webSearch(context.Background(), []byte(`{"query":"test custom"}`))
	if isErr {
		t.Fatalf("custom webSearch failed: %s", string(rawCustom))
	}
	var resCustom map[string]any
	if err := json.Unmarshal(rawCustom, &resCustom); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	customResults := resCustom["results"].([]any)
	if len(customResults) != 1 {
		t.Fatalf("custom results len = %d, want 1", len(customResults))
	}
	first := customResults[0].(map[string]any)
	if first["title"] != "Custom Result" {
		t.Errorf("title = %v, want Custom Result", first["title"])
	}

	// 3. Validation: missing query
	rawErr, isErr := ts.webSearch(context.Background(), []byte(`{}`))
	if !isErr {
		t.Fatalf("expected error on missing query, got: %s", string(rawErr))
	}
}

func TestAdjudicatorRung(t *testing.T) {
	ts, err := New(Config{
		AllowPrivateIPs: true,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 1. Unknown tool defers
	callUnknown := &abi.ToolCall{Tool: "unknown_tool"}
	vUnknown := ts.Adjudicate(context.Background(), callUnknown)
	if vUnknown.Kind != abi.VerdictDefer {
		t.Errorf("unknown tool expected VerdictDefer, got %+v", vUnknown)
	}

	// 2. Valid get_time call: allows, pins engine, sets readOnlyHint
	callTime := &abi.ToolCall{
		Tool: ToolGetTime,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"timezone":"UTC"}`)},
	}
	vTime := ts.Adjudicate(context.Background(), callTime)
	if vTime.Kind != abi.VerdictAllow {
		t.Errorf("get_time expected VerdictAllow, got %+v", vTime)
	}
	if callTime.Engine != EngineGetTime {
		t.Errorf("callTime.Engine = %q, want %q", callTime.Engine, EngineGetTime)
	}
	if callTime.Meta["readOnlyHint"] != "true" {
		t.Errorf("callTime.Meta[readOnlyHint] = %q, want true", callTime.Meta["readOnlyHint"])
	}

	// 3. Valid fetch_web call: allows, pins engine
	callFetch := &abi.ToolCall{
		Tool: ToolFetchWeb,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://example.com/test"}`)},
	}
	vFetch := ts.Adjudicate(context.Background(), callFetch)
	if vFetch.Kind != abi.VerdictAllow {
		t.Errorf("fetch_web expected VerdictAllow, got %+v", vFetch)
	}
	if callFetch.Engine != EngineFetchWeb {
		t.Errorf("callFetch.Engine = %q, want %q", callFetch.Engine, EngineFetchWeb)
	}

	// 4. Valid web_search call: allows, pins engine
	callSearch := &abi.ToolCall{
		Tool: ToolWebSearch,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"query":"test"}`)},
	}
	vSearch := ts.Adjudicate(context.Background(), callSearch)
	if vSearch.Kind != abi.VerdictAllow {
		t.Errorf("web_search expected VerdictAllow, got %+v", vSearch)
	}
	if callSearch.Engine != EngineWebSearch {
		t.Errorf("callSearch.Engine = %q, want %q", callSearch.Engine, EngineWebSearch)
	}

	// 5. Malformed args: denies with MALFORMED
	callBad := &abi.ToolCall{
		Tool: ToolFetchWeb,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":""}`)},
	}
	vBad := ts.Adjudicate(context.Background(), callBad)
	if vBad.Kind != abi.VerdictDeny {
		t.Errorf("malformed call expected VerdictDeny, got %+v", vBad)
	}
	if vBad.Reason != abi.ReasonMalformed {
		t.Errorf("malformed call expected ReasonMalformed, got %+v", vBad)
	}

	// 6. Policy denial when tool excluded from Allow
	tsRestricted, err := New(Config{
		Policy: Policy{Allow: map[string]bool{ToolGetTime: true}}, // fetch_web not allowed
	})
	if err != nil {
		t.Fatalf("New restricted: %v", err)
	}
	vPolicy := tsRestricted.Adjudicate(context.Background(), callFetch)
	if vPolicy.Kind != abi.VerdictDeny {
		t.Errorf("excluded tool expected VerdictDeny, got %+v", vPolicy)
	}
	if vPolicy.Reason != abi.ReasonDefaultDeny {
		t.Errorf("excluded tool expected ReasonDefaultDeny, got %+v", vPolicy)
	}
}

func TestCompleteEnginesThroughDriver(t *testing.T) {
	ts, err := New(Config{
		AllowPrivateIPs: true,
		SearchAdapter: func(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
			return []SearchResult{
				{Title: "Kernel Docs", URL: "https://example.com/kernel", Snippet: "Kernel documentation"},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	ts.RegisterEngines()

	driverTime := abi.Engine(EngineGetTime)
	if driverTime == nil {
		t.Fatalf("driver %s not registered", EngineGetTime)
	}
	callTime := &abi.ToolCall{
		Tool: ToolGetTime,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
	}
	resTime, err := driverTime.Complete(context.Background(), callTime)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resTime.Status != abi.StatusOK {
		t.Errorf("status = %v, want StatusOK", resTime.Status)
	}

	driverSearch := abi.Engine(EngineWebSearch)
	if driverSearch == nil {
		t.Fatalf("driver %s not registered", EngineWebSearch)
	}
	callSearch := &abi.ToolCall{
		Tool: ToolWebSearch,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"query":"kernel"}`)},
	}
	resSearch, err := driverSearch.Complete(context.Background(), callSearch)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resSearch.Status != abi.StatusOK {
		t.Errorf("status = %v, want StatusOK", resSearch.Status)
	}
}

func TestWebSearchUnconfiguredRegression(t *testing.T) {
	// 1. Unconfigured Toolset dispatch: verify unconfigured query never yields canned results
	tsUnconfigured, err := New(Config{})
	if err != nil {
		t.Fatalf("New unconfigured: %v", err)
	}
	tsUnconfigured.RegisterEngines()

	driver := abi.Engine(EngineWebSearch)
	if driver == nil {
		t.Fatalf("driver %s not registered", EngineWebSearch)
	}

	queries := []string{
		"agent kernel",
		"FAK Agent Kernel Overview",
		"Go Standard Library Documentation",
		"random unmatched search query",
	}

	for _, q := range queries {
		reqBody := fmt.Sprintf(`{"query":%q}`, q)

		// Direct engine method
		raw, isErr := tsUnconfigured.webSearch(context.Background(), []byte(reqBody))
		if !isErr {
			t.Fatalf("expected error for unconfigured search query %q, got: %s", q, string(raw))
		}
		rawStr := string(raw)
		if strings.Contains(rawStr, "FAK Agent Kernel Overview") ||
			strings.Contains(rawStr, "github.com/anthony-chaudhary/fak") ||
			strings.Contains(rawStr, "pkg.go.dev") {
			t.Fatalf("unconfigured search returned canned documents for query %q: %s", q, rawStr)
		}
		if !strings.Contains(rawStr, "search backend unconfigured") {
			t.Errorf("expected 'search backend unconfigured' for query %q, got: %s", q, rawStr)
		}

		// Toolset dispatch through abi.Engine
		call := &abi.ToolCall{
			Tool: ToolWebSearch,
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(reqBody), Len: int64(len(reqBody))},
		}
		v := tsUnconfigured.Adjudicate(context.Background(), call)
		if v.Kind != abi.VerdictAllow {
			t.Fatalf("Adjudicate expected VerdictAllow, got %+v", v)
		}
		res, err := driver.Complete(context.Background(), call)
		if err != nil {
			t.Fatalf("Complete: %v", err)
		}
		if res.Status != abi.StatusError {
			t.Fatalf("expected StatusError for unconfigured query %q, got %v", q, res.Status)
		}
		payload := string(bytesOf(context.Background(), res.Payload))
		if strings.Contains(payload, "FAK Agent Kernel Overview") ||
			strings.Contains(payload, "github.com/anthony-chaudhary/fak") ||
			strings.Contains(payload, "pkg.go.dev") {
			t.Fatalf("dispatch returned canned documents for query %q: %s", q, payload)
		}
		if !strings.Contains(payload, "search backend unconfigured") {
			t.Errorf("expected 'search backend unconfigured' in payload for query %q, got: %s", q, payload)
		}
	}

	// 2. Configured SearchAdapter still executes normally
	called := false
	tsCustom, err := New(Config{
		SearchAdapter: func(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
			called = true
			return []SearchResult{
				{
					Title:   "Live Search Result: " + query,
					URL:     "https://search.example.com?q=" + query,
					Snippet: "Fresh result for query " + query,
				},
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("New custom: %v", err)
	}
	tsCustom.RegisterEngines()
	driverCustom := abi.Engine(EngineWebSearch)
	if driverCustom == nil {
		t.Fatalf("driver %s not registered", EngineWebSearch)
	}

	callCustom := &abi.ToolCall{
		Tool: ToolWebSearch,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"query":"real world query"}`)},
	}
	resCustom, err := driverCustom.Complete(context.Background(), callCustom)
	if err != nil {
		t.Fatalf("Complete custom: %v", err)
	}
	if resCustom.Status != abi.StatusOK {
		t.Fatalf("expected StatusOK for configured search, got %v", resCustom.Status)
	}
	if !called {
		t.Fatalf("configured SearchAdapter was not called")
	}
	payloadCustom := string(bytesOf(context.Background(), resCustom.Payload))
	if !strings.Contains(payloadCustom, "Live Search Result: real world query") {
		t.Fatalf("expected configured adapter result in payload, got: %s", payloadCustom)
	}
}
