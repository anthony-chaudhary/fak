package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func openCodeSchemaTool(name string, properties ...string) agent.ToolDef {
	props := make(map[string]any, len(properties))
	for _, property := range properties {
		props[property] = map[string]string{"type": "string"}
	}
	parameters, _ := json.Marshal(map[string]any{"type": "object", "properties": props})
	return agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: name, Parameters: parameters}}
}

func openCodeSchemaServer(t *testing.T) (*Server, *adjudicator.Adjudicator) {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	monitor := adjudicator.New(adjudicator.DevAgentPolicy())
	abi.RegisterAdjudicator(100, monitor)
	srv, err := New(Config{EngineID: "mock", Model: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	return srv, monitor
}

func TestOpenCodeAdvertisedStandardToolSchemaTranslation(t *testing.T) {
	srv, _ := openCodeSchemaServer(t)
	tests := []struct {
		name       string
		source     string
		arguments  string
		tools      []agent.ToolDef
		wantName   string
		want       map[string]any
		wantAbsent []string
	}{
		{
			name: "read_file to camel read", source: "read_file",
			arguments: `{"file_path":"README.md","offset":2,"limit":4}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("read", "filePath", "offset", "limit")}, wantName: "read",
			want: map[string]any{"filePath": "README.md", "offset": float64(2), "limit": float64(4)}, wantAbsent: []string{"file_path"},
		},
		{
			name: "edit_file snake to camel edit", source: "edit_file",
			arguments: `{"file_path":"notes.txt","old_string":"old","new_string":"new","replaceAll":false}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("edit", "filePath", "oldString", "newString", "replaceAll")}, wantName: "edit",
			want:       map[string]any{"filePath": "notes.txt", "oldString": "old", "newString": "new", "replaceAll": false},
			wantAbsent: []string{"file_path", "old_string", "new_string"},
		},
		{
			name: "edit_file camel to snake edit", source: "edit_file",
			arguments: `{"filePath":"notes.txt","oldString":"old","newString":"new","count":2}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("edit", "file_path", "old_string", "new_string", "count")}, wantName: "edit",
			want:       map[string]any{"file_path": "notes.txt", "old_string": "old", "new_string": "new", "count": float64(2)},
			wantAbsent: []string{"filePath", "oldString", "newString"},
		},
		{
			name: "write_file to path write", source: "write_file",
			arguments: `{"file_path":"notes.txt","content":"hello","encoding":"utf-8"}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("write", "path", "content", "encoding")}, wantName: "write",
			want: map[string]any{"path": "notes.txt", "content": "hello", "encoding": "utf-8"}, wantAbsent: []string{"file_path"},
		},
		{
			name: "exact advertised source wins", source: "edit_file",
			arguments: `{"file_path":"notes.txt","old_string":"old","new_string":"new"}`,
			tools: []agent.ToolDef{
				openCodeSchemaTool("edit_file", "file_path", "old_string", "new_string"),
				openCodeSchemaTool("edit", "filePath", "oldString", "newString"),
			}, wantName: "edit_file",
			want: map[string]any{"file_path": "notes.txt", "old_string": "old", "new_string": "new"},
		},
		{
			name: "duplicate target is ambiguous", source: "edit_file",
			arguments: `{"file_path":"notes.txt","old_string":"old","new_string":"new"}`,
			tools: []agent.ToolDef{
				openCodeSchemaTool("edit", "filePath", "oldString", "newString"),
				openCodeSchemaTool("edit", "file_path", "old_string", "new_string"),
			}, wantName: "edit_file",
			want: map[string]any{"file_path": "notes.txt", "old_string": "old", "new_string": "new"},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := withClientToolSchemas(context.Background(), tc.tools)
			calls := []agent.ToolCall{{ID: "call-1", Type: "function", Function: agent.Func{Name: tc.source, Arguments: tc.arguments}}}
			kept, adjs, dropped := srv.adjudicateProposed(ctx, calls, "opencode-schema")
			if dropped != 0 || len(kept) != 1 || len(adjs) != 1 || !adjs[0].Admitted {
				t.Fatalf("proposal not admitted: kept=%+v adjs=%+v dropped=%d", kept, adjs, dropped)
			}
			if kept[0].Function.Name != tc.wantName {
				t.Fatalf("tool name = %q, want %q", kept[0].Function.Name, tc.wantName)
			}
			var got map[string]any
			if err := json.Unmarshal([]byte(kept[0].Function.Arguments), &got); err != nil {
				t.Fatal(err)
			}
			for key, want := range tc.want {
				if got[key] != want {
					t.Errorf("%s = %#v, want %#v; arguments=%s", key, got[key], want, kept[0].Function.Arguments)
				}
			}
			for _, key := range tc.wantAbsent {
				if _, exists := got[key]; exists {
					t.Errorf("obsolete alias %q survived: %s", key, kept[0].Function.Arguments)
				}
			}
		})
	}
}

func TestOpenCodeToolSchemaTranslationStaysConservative(t *testing.T) {
	srv, monitor := openCodeSchemaServer(t)

	ctx := withClientToolSchemas(context.Background(), []agent.ToolDef{openCodeSchemaTool("edit", "filePath", "oldString", "newString")})
	conflict := []agent.ToolCall{{ID: "conflict", Type: "function", Function: agent.Func{
		Name: "edit_file", Arguments: `{"file_path":"a.txt","filePath":"b.txt","old_string":"a","new_string":"b"}`,
	}}}
	kept, adjs, dropped := srv.adjudicateProposed(ctx, conflict, "opencode-conflict")
	if len(kept) != 0 || dropped != 1 || len(adjs) != 1 || adjs[0].Verdict.Reason != "MALFORMED" {
		t.Fatalf("conflicting aliases were not denied as malformed: kept=%+v adjs=%+v dropped=%d", kept, adjs, dropped)
	}

	equal := []agent.ToolCall{{ID: "equal", Type: "function", Function: agent.Func{
		Name: "edit_file", Arguments: `{"file_path":"notes.txt","filePath":"notes.txt","old_string":"a","oldString":"a","new_string":"b","newString":"b","count":1}`,
	}}}
	kept, _, dropped = srv.adjudicateProposed(ctx, equal, "opencode-equal")
	if dropped != 0 || len(kept) != 1 || kept[0].Function.Name != "edit" {
		t.Fatalf("equal aliases were not safely normalized: kept=%+v dropped=%d", kept, dropped)
	}
	var normalized map[string]any
	_ = json.Unmarshal([]byte(kept[0].Function.Arguments), &normalized)
	if normalized["filePath"] != "notes.txt" || normalized["oldString"] != "a" || normalized["newString"] != "b" || normalized["count"] != float64(1) {
		t.Fatalf("equal alias normalization lost values: %v", normalized)
	}

	protected := []agent.ToolCall{{ID: "protected", Type: "function", Function: agent.Func{
		Name: "edit_file", Arguments: `{"file_path":"internal/adjudicator/default_policy.go","old_string":"a","new_string":"b"}`,
	}}}
	kept, _, dropped = srv.adjudicateProposed(ctx, protected, "opencode-protected")
	if len(kept) != 0 || dropped != 1 {
		t.Fatalf("schema translation bypassed protected-path policy: kept=%+v dropped=%d", kept, dropped)
	}

	monitor.SetPolicy(adjudicator.Policy{Allow: map[string]bool{"custom_patch": true}})
	customArgs := `{"target":"notes.txt","before":"a","after":"b","optional":7}`
	customCtx := withClientToolSchemas(context.Background(), []agent.ToolDef{openCodeSchemaTool("custom_patch", "target", "before", "after", "optional")})
	custom := []agent.ToolCall{{ID: "custom", Type: "function", Function: agent.Func{Name: "custom_patch", Arguments: customArgs}}}
	kept, _, dropped = srv.adjudicateProposed(customCtx, custom, "opencode-custom")
	if dropped != 0 || len(kept) != 1 || kept[0].Function.Name != "custom_patch" || kept[0].Function.Arguments != customArgs {
		t.Fatalf("custom tool was widened or rewritten: kept=%+v dropped=%d", kept, dropped)
	}
}

func TestOpenCodeSchemaProjectionCannotBypassArgumentPolicy(t *testing.T) {
	srv, monitor := openCodeSchemaServer(t)
	tests := []struct {
		name      string
		tool      string
		policyArg string
		modelArgs string
		tools     []agent.ToolDef
	}{
		{
			name: "snake edit cannot bypass camel policy", tool: "edit", policyArg: "newString",
			modelArgs: `{"file_path":"notes.txt","old_string":"old","new_string":"blocked snake value"}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("edit", "filePath", "oldString", "newString")},
		},
		{
			name: "camel edit cannot bypass snake policy", tool: "edit", policyArg: "new_string",
			modelArgs: `{"filePath":"notes.txt","oldString":"old","newString":"blocked camel value"}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("edit", "file_path", "old_string", "new_string")},
		},
		{
			name: "snake write path cannot bypass camel policy", tool: "write", policyArg: "filePath",
			modelArgs: `{"file_path":"blocked.txt","content":"safe"}`,
			tools:     []agent.ToolDef{openCodeSchemaTool("write", "filePath", "content")},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			monitor.SetPolicy(adjudicator.Policy{
				Allow: map[string]bool{tc.tool: true},
				ArgPredicates: []adjudicator.ArgPredicate{{
					Tool: tc.tool, Arg: tc.policyArg, Kind: adjudicator.ArgDenyRegex,
					Re: regexp.MustCompile(`blocked`), Reason: abi.ReasonPolicyBlock,
				}},
			})
			ctx := withClientToolSchemas(context.Background(), tc.tools)
			source := tc.tool + "_file"
			calls := []agent.ToolCall{{ID: "policy", Type: "function", Function: agent.Func{Name: source, Arguments: tc.modelArgs}}}
			kept, adjs, dropped := srv.adjudicateProposed(ctx, calls, "opencode-policy-alias")
			if len(kept) != 0 || dropped != 1 || len(adjs) != 1 || adjs[0].Verdict.Reason != "POLICY_BLOCK" {
				t.Fatalf("argument alias bypassed policy: kept=%+v adjs=%+v dropped=%d", kept, adjs, dropped)
			}
		})
	}
}

func TestOpenCodeExactAdvertisedUppercaseReadUsesItsPathDialect(t *testing.T) {
	srv, monitor := openCodeSchemaServer(t)
	monitor.SetPolicy(adjudicator.Policy{Allow: map[string]bool{"Read": true}})
	ctx := withClientToolSchemas(context.Background(), []agent.ToolDef{openCodeSchemaTool("Read", "filePath", "offset")})
	calls := []agent.ToolCall{{ID: "read-uppercase", Type: "function", Function: agent.Func{
		Name: "Read", Arguments: `{"file_path":"README.md","offset":3}`,
	}}}
	kept, adjs, dropped := srv.adjudicateProposed(ctx, calls, "opencode-uppercase-read")
	if dropped != 0 || len(kept) != 1 || len(adjs) != 1 || !adjs[0].Admitted {
		t.Fatalf("uppercase Read not admitted: kept=%+v adjs=%+v dropped=%d", kept, adjs, dropped)
	}
	if kept[0].Function.Name != "Read" {
		t.Fatalf("exact advertised name changed to %q", kept[0].Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(kept[0].Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if args["filePath"] != "README.md" || args["offset"] != float64(3) {
		t.Fatalf("uppercase Read arguments=%v", args)
	}
	if _, exists := args["file_path"]; exists {
		t.Fatalf("uppercase Read retained unadvertised path alias: %v", args)
	}
}

func TestOpenCodeProjectionPreservesCanonicalRepairedValue(t *testing.T) {
	ctx := withClientToolSchemas(context.Background(), []agent.ToolDef{openCodeSchemaTool("read", "filePath", "offset")})
	original := `{"filePath":"old"}`
	repaired := `{"file_path":"repaired","filePath":"old","path":"old","offset":2}`
	projected := clientReadArguments(ctx, "read", original, repaired)
	var args map[string]any
	if err := json.Unmarshal([]byte(projected), &args); err != nil {
		t.Fatal(err)
	}
	if args["filePath"] != "repaired" || args["offset"] != float64(2) {
		t.Fatalf("projection discarded canonical repaired value: %s", projected)
	}
	for _, stale := range []string{"file_path", "path"} {
		if _, exists := args[stale]; exists {
			t.Fatalf("projection retained stale alias %q: %s", stale, projected)
		}
	}
}

func TestOpenCodeEditFileStreamUsesAdvertisedEditDialect(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		args := `{"file_path":"notes.txt","old_string":"old","new_string":"new","replaceAll":false}`
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"edit-1","type":"function","function":{"name":"edit_file","arguments":` + strconv.Quote(args) + `}}]}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
			"data: [DONE]", "",
		}, "\n\n"))
	}))
	defer upstream.Close()

	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(100, adjudicator.New(adjudicator.DevAgentPolicy()))
	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL, Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	gateway := httptest.NewServer(srv.Handler())
	defer gateway.Close()

	request := map[string]any{
		"model": "x:model", "stream": true,
		"messages": []map[string]string{{"role": "user", "content": "edit it"}},
		"tools": []map[string]any{{"type": "function", "function": map[string]any{
			"name": "edit", "parameters": map[string]any{"type": "object", "properties": map[string]any{
				"filePath": map[string]string{"type": "string"}, "oldString": map[string]string{"type": "string"},
				"newString": map[string]string{"type": "string"}, "replaceAll": map[string]string{"type": "boolean"},
			}},
		}}},
	}
	code, raw := postStream(t, gateway.URL, request)
	if code != http.StatusOK {
		t.Fatalf("status=%d body=%s", code, raw)
	}
	chunks, done := parseSSEChunks(t, raw)
	if !done {
		t.Fatalf("missing DONE: %s", raw)
	}
	var calls []ChatDeltaToolCall
	for _, chunk := range chunks {
		calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
	}
	if len(calls) != 1 || calls[0].Function.Name != "edit" {
		t.Fatalf("translated calls=%+v", calls)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if args["filePath"] != "notes.txt" || args["oldString"] != "old" || args["newString"] != "new" || args["replaceAll"] != false {
		t.Fatalf("translated stream arguments=%v", args)
	}
}
