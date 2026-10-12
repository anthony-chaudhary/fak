package agent

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// testdata/deepseek41_chat_template_golden.txt is the GGUF tokenizer.chat_template of
// vcruz305/DeepSeek-V4.1-Flash-GGUF (Q2_K shard 1) rendered by jinja2 3.1.6 with
// HF-style tojson (ensure_ascii=False, key order kept), from_json,
// bos_token=<｜begin▁of▁sentence｜> and add_generation_prompt=True over exactly this
// conversation. It pins the Go port to the template, not to itself: system folding,
// tool schemas, response_format, DSML tool-call history, merged tool results and the
// non-thinking generation turn.
// fak-test:justify why=contract when=changed:internal/agent/**
// fak-test:runtime fast est=5ms lane=default
func TestRenderDeepSeekV41ChatMatchesGGUFTemplate(t *testing.T) {
	want, err := os.ReadFile(filepath.Join("testdata", "deepseek41_chat_template_golden.txt"))
	if err != nil {
		t.Fatal(err)
	}
	msgs := []Message{
		{Role: RoleSystem, Content: "You are terse."},
		{Role: RoleSystem, Content: "Use metric units."},
		{Role: RoleUser, Content: "Weather in Paris?"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{Type: "function", Function: Func{Name: "get_weather", Arguments: `{"city": "Paris", "days": 3, "opts": {"units": ["c", "k"], "ok": true}}`}}}},
		{Role: RoleTool, Content: `{"temp": 18}`},
		{Role: RoleTool, Content: "cloudy"},
		{Role: RoleUser, Content: "And tomorrow?"},
		{Role: RoleAssistant, Content: "Rain, 15 C."},
		{Role: RoleUser, Content: "Thanks — ünïcode ok?"},
	}
	tools := []ToolDef{{Type: "function", Function: ToolDefFunction{
		Name: "get_weather", Description: "Get the weather <now>",
		Parameters: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}`),
	}}}
	sp := SampleParams{ResponseFormat: json.RawMessage(`{"type":"json_schema","json_schema":{"name":"w","schema":{"type":"object"}}}`)}
	if got := inKernelRenderPrompt(msgs, tools, model.Config{ModelType: "deepseek41"}, sp); got != string(want) {
		t.Fatalf("V4.1 chat render differs from the GGUF template:\ngot  %q\nwant %q", got, want)
	}
}
