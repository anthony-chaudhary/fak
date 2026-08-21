package toolplugin

import (
	"encoding/json"
	"testing"
)

func TestResolveBudgetUsesNestedExplicitAlias(t *testing.T) {
	contracts := []BudgetContract{{Tool: "search", Pointers: []string{"/options/top_k", "/limit"}, Ceiling: 25}}
	got, err := ResolveBudget("search", []byte(`{"options":{"top_k":100},"limit":90}`), contracts)
	if err != nil {
		t.Fatal(err)
	}
	if got.Decision != "clamp" || got.Pointer != "/options/top_k" || got.Requested != 100 || got.Effective != 25 {
		t.Fatalf("resolution = %+v", got)
	}
	var args map[string]any
	if err := json.Unmarshal(got.Arguments, &args); err != nil {
		t.Fatal(err)
	}
	if args["limit"].(float64) != 90 || args["options"].(map[string]any)["top_k"].(float64) != 25 {
		t.Fatalf("arguments = %s", got.Arguments)
	}
}

func TestResolveBudgetNeverGuessesOrMutatesInvalidValues(t *testing.T) {
	contracts := []BudgetContract{{Tool: "search", Pointers: []string{"/request/max_results"}, Ceiling: 10}}
	cases := []struct {
		name, tool, args string
		wantErr          bool
	}{
		{"unknown tool", "other", `{"request":{"max_results":99}}`, false},
		{"unknown pointer", "search", `{"max_results":99}`, false},
		{"string", "search", `{"request":{"max_results":"99"}}`, true},
		{"float", "search", `{"request":{"max_results":9.5}}`, true},
		{"negative", "search", `{"request":{"max_results":-1}}`, true},
		{"overflow", "search", `{"request":{"max_results":9223372036854775808}}`, true},
		{"malformed", "search", `{"request":`, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveBudget(tc.tool, []byte(tc.args), contracts)
			if (err != nil) != tc.wantErr {
				t.Fatalf("err=%v wantErr=%v", err, tc.wantErr)
			}
			if string(got.Arguments) != tc.args {
				t.Fatalf("mutated invalid/unsupported input: %q", got.Arguments)
			}
		})
	}
}

func TestResolveBudgetDecodesJSONPointerEscapes(t *testing.T) {
	got, err := ResolveBudget("x", []byte(`{"a/b":{"~limit":8}}`), []BudgetContract{{Tool: "x", Pointers: []string{"/a~1b/~0limit"}, Ceiling: 3}})
	if err != nil || got.Decision != "clamp" || got.Effective != 3 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
}
