package streamrules

import "testing"

func TestParseScope(t *testing.T) {
	tests := []struct {
		spec string
		want Scope
		tool string
	}{
		{"text", ScopeText, ""},
		{"thinking", ScopeThinking, ""},
		{"any-tool", ScopeAnyTool, ""},
		{"named-tool:shell", ScopeNamedTool, "shell"},
	}
	for _, test := range tests {
		got, tool, err := ParseScope(test.spec)
		if err != nil || got != test.want || tool != test.tool {
			t.Errorf("ParseScope(%q) = (%q, %q, %v), want (%q, %q, nil)", test.spec, got, tool, err, test.want, test.tool)
		}
	}
	if _, _, err := ParseScope("named-tool:"); err == nil {
		t.Fatal("ParseScope(named-tool:) succeeded, want error")
	}
}
