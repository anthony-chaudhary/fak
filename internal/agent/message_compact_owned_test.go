package agent

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Exercise the actual native planner shrink/render seams with the owned catalogs.
// The HTTP client advertising a tool is insufficient if the model never sees it.
func TestNativeOwnedToolsRemainInModelPrompt(t *testing.T) {
	var owned []ToolDef
	for _, d := range codetools.Catalog() {
		owned = append(owned, ToolDef{Type: "function", Function: ToolDefFunction{Name: d.Name, Description: d.Description, Parameters: d.Parameters}})
	}
	owned = append(owned, taskToolDefs()...)
	for _, cold := range []bool{false, true} {
		t.Run(map[bool]string{false: "owned_only", true: "with_cold_extension"}[cold], func(t *testing.T) {
			tools := append([]ToolDef(nil), owned...)
			if cold {
				tools = append(tools, ToolDef{Type: "function", Function: ToolDefFunction{Name: "external_rare_analyzer", Parameters: json.RawMessage(`{"type":"object"}`)}})
			}
			before, _ := json.Marshal(tools)
			p := &InKernelPlanner{}
			p.SetPromptShrinkLevers(0, false, true)
			messages := []Message{{Role: RoleUser, Content: "Perform the work."}}
			shrunk, visible, outcome := p.ApplyPromptShrink(context.Background(), messages, tools)
			expectedCold := 0
			if cold {
				expectedCold = 1
			}
			if outcome.ColdToolsDeferred != expectedCold {
				t.Errorf("deferred %d tools, want only %d external tools", outcome.ColdToolsDeferred, expectedCold)
			}
			byName := map[string]ToolDef{}
			for _, d := range visible {
				byName[d.Function.Name] = d
			}
			prompt := renderInKernelChatMLRequest(shrunk, visible, model.Config{ModelType: "qwen3_5", LayerTypes: []string{"linear_attention", "full_attention"}}, nil, nil)
			for _, d := range owned {
				if got, ok := byName[d.Function.Name]; !ok || !reflect.DeepEqual(got, d) {
					t.Errorf("owned tool %q missing or schema changed", d.Function.Name)
				}
				marker := `"name":"` + d.Function.Name + `"`
				if !strings.Contains(prompt, marker) {
					t.Errorf("model-rendered prompt omits %s", d.Function.Name)
				}
			}
			if _, ok := byName["external_rare_analyzer"]; ok {
				t.Error("cold extension leaked into resident catalog")
			}
			_, hasSearch := byName["ToolSearch"]
			if hasSearch != cold {
				t.Errorf("ToolSearch inserted=%t, want %t", hasSearch, cold)
			}
			after, _ := json.Marshal(tools)
			if string(before) != string(after) {
				t.Error("source tool schemas mutated")
			}
		})
	}
}
