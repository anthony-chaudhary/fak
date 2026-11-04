package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// dialectFoldCapturePlanner records the tool catalog each served turn hands the
// planner, so a handler test can assert which folded serving input a mounted
// dialect actually delivered.
type dialectFoldCapturePlanner struct {
	comp *agent.Completion
	mu   sync.Mutex
	got  [][]agent.ToolDef
}

func (p *dialectFoldCapturePlanner) Complete(_ context.Context, _ []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.got = append(p.got, append([]agent.ToolDef(nil), t...))
	return p.comp, nil
}

func (p *dialectFoldCapturePlanner) Model() string { return "capture" }

func (p *dialectFoldCapturePlanner) last() []agent.ToolDef {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.got) == 0 {
		return nil
	}
	return p.got[len(p.got)-1]
}

// TestDialectToolNormalizationParity pins the shared serving-input tool contract:
// the same logical tool catalog declared on the OpenAI chat wire (nested
// function objects), the OpenAI Responses wire (flat function items), and the
// Anthropic Messages wire (name/description/input_schema) must all fold through
// one normalization pass into byte-identical agent.ToolDef slices — empty names
// drop, exact duplicates keep the first occurrence, and an MCP legacy
// mcp__fak__X entry loses to its canonical mcp__fak_guard__X counterpart
// regardless of declaration order. Dialect capability checks stay distinct; this
// is the serving-input half of the seam only.
func TestDialectToolNormalizationParity(t *testing.T) {
	const params = `{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}`

	// One logical catalog, spelled in each dialect's own wire shape. Six
	// declarations: a plain tool, an exact duplicate of it, an empty-named
	// declaration, a legacy+canonical MCP pair, and a standalone legacy tool.
	chatFragment := `[` +
		`{"type":"function","function":{"name":"search_kb","description":"Search the knowledge base","parameters":` + params + `}},` +
		`{"type":"function","function":{"name":"search_kb","description":"Duplicate must lose","parameters":{}}},` +
		`{"type":"function","function":{"name":"","description":"Empty name must drop"}},` +
		`{"type":"function","function":{"name":"mcp__fak__fak_read","description":"Legacy read","parameters":{}}},` +
		`{"type":"function","function":{"name":"mcp__fak_guard__fak_read","description":"Canonical read","parameters":{}}},` +
		`{"type":"function","function":{"name":"mcp__fak__standalone_tool","description":"Standalone tool","parameters":{}}}` +
		`]`
	responsesFragment := `[` +
		`{"type":"function","name":"search_kb","description":"Search the knowledge base","parameters":` + params + `},` +
		`{"type":"function","name":"search_kb","description":"Duplicate must lose","parameters":{}},` +
		`{"type":"function","name":"","description":"Empty name must drop"},` +
		`{"type":"function","name":"mcp__fak__fak_read","description":"Legacy read","parameters":{}},` +
		`{"type":"function","name":"mcp__fak_guard__fak_read","description":"Canonical read","parameters":{}},` +
		`{"type":"function","name":"mcp__fak__standalone_tool","description":"Standalone tool","parameters":{}}` +
		`]`
	anthropicFragment := `[` +
		`{"name":"search_kb","description":"Search the knowledge base","input_schema":` + params + `},` +
		`{"name":"search_kb","description":"Duplicate must lose","input_schema":{}},` +
		`{"name":"","description":"Empty name must drop"},` +
		`{"name":"mcp__fak__fak_read","description":"Legacy read","input_schema":{}},` +
		`{"name":"mcp__fak_guard__fak_read","description":"Canonical read","input_schema":{}},` +
		`{"name":"mcp__fak__standalone_tool","description":"Standalone tool","input_schema":{}}` +
		`]`

	want := []agent.ToolDef{
		{
			Type:     "function",
			Function: agent.ToolDefFunction{Name: "search_kb", Description: "Search the knowledge base", Parameters: json.RawMessage(params)},
		},
		{
			Type:     "function",
			Function: agent.ToolDefFunction{Name: "mcp__fak_guard__fak_read", Description: "Canonical read", Parameters: json.RawMessage("{}")},
		},
		{
			Type:     "function",
			Function: agent.ToolDefFunction{Name: "mcp__fak__standalone_tool", Description: "Standalone tool", Parameters: json.RawMessage("{}")},
		},
	}

	t.Run("all three dialects fold to one serving input", func(t *testing.T) {
		var chatReq ChatRequest
		if err := json.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":`+chatFragment+`}`), &chatReq); err != nil {
			t.Fatalf("decode chat body: %v", err)
		}
		if len(chatReq.Tools) != 6 {
			t.Fatalf("chat wire decoded %d tools, want 6 (decoders must keep every declaration; the fold drops)", len(chatReq.Tools))
		}
		gotChat := normalizeServingToolDefs(chatReq.Tools)

		var respReq ResponsesRequest
		if err := json.Unmarshal([]byte(`{"model":"m","input":"hi","tools":`+responsesFragment+`}`), &respReq); err != nil {
			t.Fatalf("decode responses body: %v", err)
		}
		if len(respReq.Tools) != 6 {
			t.Fatalf("responses wire decoded %d tools, want 6", len(respReq.Tools))
		}
		gotResponses := responsesToolsToToolDefs(respReq.Tools)

		out, err := agent.DecodeAnthropicMessagesRequest([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"tools":` + anthropicFragment + `}`))
		if err != nil {
			t.Fatalf("decode anthropic body: %v", err)
		}
		if len(out.Tools) != 6 {
			t.Fatalf("anthropic wire decoded %d tools, want 6", len(out.Tools))
		}
		gotAnthropic := normalizeServingToolDefs(out.Tools)

		for name, got := range map[string][]agent.ToolDef{
			"openai-chat":      gotChat,
			"openai-responses": gotResponses,
			"anthropic":        gotAnthropic,
		} {
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s folded catalog = %+v, want %+v", name, got, want)
			}
		}
	})

	t.Run("canonical MCP namespace wins regardless of order", func(t *testing.T) {
		// The same legacy+canonical pair with the canonical entry declared FIRST.
		chatRev := `[` +
			`{"type":"function","function":{"name":"mcp__fak_guard__fak_read","description":"Canonical read","parameters":{}}},` +
			`{"type":"function","function":{"name":"mcp__fak__fak_read","description":"Legacy read","parameters":{}}}` +
			`]`
		respRev := `[` +
			`{"type":"function","name":"mcp__fak_guard__fak_read","description":"Canonical read","parameters":{}},` +
			`{"type":"function","name":"mcp__fak__fak_read","description":"Legacy read","parameters":{}}` +
			`]`
		anthRev := `[` +
			`{"name":"mcp__fak_guard__fak_read","description":"Canonical read","input_schema":{}},` +
			`{"name":"mcp__fak__fak_read","description":"Legacy read","input_schema":{}}` +
			`]`
		wantRev := []agent.ToolDef{
			{
				Type:     "function",
				Function: agent.ToolDefFunction{Name: "mcp__fak_guard__fak_read", Description: "Canonical read", Parameters: json.RawMessage("{}")},
			},
		}

		var chatReq ChatRequest
		if err := json.Unmarshal([]byte(`{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":`+chatRev+`}`), &chatReq); err != nil {
			t.Fatalf("decode chat body: %v", err)
		}
		if got := normalizeServingToolDefs(chatReq.Tools); !reflect.DeepEqual(got, wantRev) {
			t.Errorf("openai-chat reversed fold = %+v, want %+v", got, wantRev)
		}
		var respReq ResponsesRequest
		if err := json.Unmarshal([]byte(`{"model":"m","input":"hi","tools":`+respRev+`}`), &respReq); err != nil {
			t.Fatalf("decode responses body: %v", err)
		}
		if got := responsesToolsToToolDefs(respReq.Tools); !reflect.DeepEqual(got, wantRev) {
			t.Errorf("openai-responses reversed fold = %+v, want %+v", got, wantRev)
		}
		out, err := agent.DecodeAnthropicMessagesRequest([]byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}],"tools":` + anthRev + `}`))
		if err != nil {
			t.Fatalf("decode anthropic body: %v", err)
		}
		if got := normalizeServingToolDefs(out.Tools); !reflect.DeepEqual(got, wantRev) {
			t.Errorf("anthropic reversed fold = %+v, want %+v", got, wantRev)
		}
	})

	t.Run("mounted routes hand the planner the folded catalog", func(t *testing.T) {
		// Downstream of the fold the harness re-sorts the catalog and the
		// compaction machinery injects its own recall-page tools, so assert the
		// fold's closed contract as properties of whatever the planner received:
		// no empty names, no duplicate survivor, the legacy MCP pair member gone,
		// and each surviving client tool present exactly once with its winning
		// description and parameter bytes.
		assertFoldedWire := func(t *testing.T, wire string, got []agent.ToolDef) {
			t.Helper()
			count := make(map[string]int)
			var emptyNames, dupSurvivors int
			byName := make(map[string]agent.ToolDef)
			for _, d := range got {
				if d.Function.Name == "" {
					emptyNames++
					continue
				}
				count[d.Function.Name]++
				byName[d.Function.Name] = d
				if d.Function.Description == "Duplicate must lose" {
					dupSurvivors++
				}
			}
			if emptyNames != 0 {
				t.Errorf("%s route: planner saw %d empty-named tool(s); the fold drops them", wire, emptyNames)
			}
			if dupSurvivors != 0 {
				t.Errorf("%s route: %d duplicate-name survivor(s) reached the planner; first occurrence must win", wire, dupSurvivors)
			}
			if count["mcp__fak__fak_read"] != 0 {
				t.Errorf("%s route: legacy mcp__fak__fak_read reached the planner despite canonical mcp__fak_guard__fak_read", wire)
			}
			wantByName := map[string]agent.ToolDef{}
			for _, d := range want {
				wantByName[d.Function.Name] = d
			}
			for name, w := range wantByName {
				if count[name] != 1 {
					t.Errorf("%s route: tool %q reached the planner %d times, want exactly 1", wire, name, count[name])
					continue
				}
				if g := byName[name]; g.Function.Description != w.Function.Description || !reflect.DeepEqual(g.Function.Parameters, w.Function.Parameters) {
					t.Errorf("%s route: tool %q content = desc=%q params=%s, want desc=%q params=%s", wire, name, g.Function.Description, g.Function.Parameters, w.Function.Description, w.Function.Parameters)
				}
			}
		}

		srv := newTestServer(t)
		capture := &dialectFoldCapturePlanner{comp: &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
			FinishReason: "stop",
		}}
		srv.planner = capture
		ts := httptest.NewServer(srv.Handler())
		defer ts.Close()

		body := `{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":` + chatFragment + `}`
		resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatalf("POST chat: %v", err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("chat status = %d, want 200", resp.StatusCode)
		}
		assertFoldedWire(t, "openai-chat", capture.last())

		respBody := `{"model":"m","input":"hi","tools":` + responsesFragment + `}`
		resp2, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(respBody))
		if err != nil {
			t.Fatalf("POST responses: %v", err)
		}
		resp2.Body.Close()
		if resp2.StatusCode != http.StatusOK {
			t.Fatalf("responses status = %d, want 200", resp2.StatusCode)
		}
		assertFoldedWire(t, "openai-responses", capture.last())
	})
}
