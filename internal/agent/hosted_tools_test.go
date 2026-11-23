package agent

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestHostedToolTypeClassifier(t *testing.T) {
	cases := map[string]bool{
		"web_search":                    true,
		"web_search_preview":            true,
		"web_search_preview_2025_03_11": true,
		"web_search_2025_08_26":         true,
		"file_search":                   true,
		"code_interpreter":              true,
		"computer_use":                  true,
		"computer_use_preview":          true,
		"local_shell":                   true,
		"shell":                         true,
		"image_generation":              true,
		"mcp":                           true,
		"function":                      false,
		"custom":                        false,
		"":                              false,
		"web_search_":                   false,
		"websearch":                     false,
		"file_search_preview":           false,
		"mcp_server":                    false,
		"Web_Search":                    false,
	}
	for typ, want := range cases {
		if got := IsHostedToolType(typ); got != want {
			t.Errorf("IsHostedToolType(%q) = %v, want %v", typ, got, want)
		}
	}
}

func TestOpenAICompatibleToolsDropsHostedEntries(t *testing.T) {
	read := ToolDef{Type: "function", Function: ToolDefFunction{Name: "read", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}
	write := ToolDef{Type: "function", Function: ToolDefFunction{Name: "write", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}}
	in := []ToolDef{
		{Type: "web_search", ResponsesWire: json.RawMessage(`{"type":"web_search"}`)},
		read,
		{Type: "custom", ResponsesWire: json.RawMessage(`{"type":"custom","name":"apply_patch"}`)},
		{Type: "file_search"},
		{Type: "function"},
		write,
	}
	got := openAICompatibleTools(in)
	want := openAICompatibleTools([]ToolDef{read, write})
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("hosted/nameless entries leaked or reordered:\n got %+v\nwant %+v", got, want)
	}
	for _, td := range got {
		if td.Function.Name == "" || len(td.ResponsesWire) > 0 {
			t.Fatalf("malformed tool reached the chat wire: %+v", td)
		}
	}
}

func TestOpenAICompatibleToolsFunctionOnlyUnchanged(t *testing.T) {
	in := []ToolDef{
		{Type: "function", Function: ToolDefFunction{Name: "a", Description: "first", Parameters: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}},"required":["x"]}`)}},
		{Function: ToolDefFunction{Name: "b"}},
	}
	got := openAICompatibleTools(in)
	if len(got) != len(in) {
		t.Fatalf("function-only catalog changed length: %d, want %d", len(got), len(in))
	}
	for i := range in {
		gotWire, _ := json.Marshal(got[i])
		legacy := in[i]
		legacy.Function.Parameters = openAICompatibleSchema(in[i].Function.Parameters, true)
		wantWire, _ := json.Marshal(legacy)
		if string(gotWire) != string(wantWire) {
			t.Errorf("tool %d wire = %s, want %s", i, gotWire, wantWire)
		}
	}
}

func TestOpenAICompatibleToolsAllHostedOmitsToolChoice(t *testing.T) {
	body, err := openAIAdapter{}.MarshalRequest(adapterRequest{
		Model: "m",
		Tools: []ToolDef{{Type: "web_search", ResponsesWire: json.RawMessage(`{"type":"web_search"}`)}},
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if _, ok := wire["tools"]; ok {
		t.Errorf("hosted-only catalog sent tools to a chat upstream: %s", body)
	}
	if _, ok := wire["tool_choice"]; ok {
		t.Errorf("tool_choice sent with no surviving tools: %s", body)
	}
}

func TestOpenAICompatibleToolsFunctionCatalogKeepsAutoToolChoice(t *testing.T) {
	body, err := openAIAdapter{}.MarshalRequest(adapterRequest{
		Model: "m",
		Tools: []ToolDef{
			{Type: "web_search", ResponsesWire: json.RawMessage(`{"type":"web_search"}`)},
			{Type: "function", Function: ToolDefFunction{Name: "read", Parameters: json.RawMessage(`{"type":"object"}`)}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Tools      []ToolDef `json:"tools"`
		ToolChoice string    `json:"tool_choice"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.ToolChoice != "auto" || len(wire.Tools) != 1 || wire.Tools[0].Function.Name != "read" {
		t.Fatalf("function tool lost alongside a hosted one: %s", body)
	}
}
