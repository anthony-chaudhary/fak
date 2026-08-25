package trajectory

import (
	"reflect"
	"testing"
)

func TestQwenToolErrorFamilyClassification(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		content string
		want    string
	}{
		{name: "generic JSON timeout", content: `{"error":"request timeout"}`, want: "timeout"},
		{name: "deadline exceeded", content: `{"error":"context deadline exceeded"}`, want: "timeout"},
		{name: "permission denied", content: `{"error":"permission denied"}`, want: "permission"},
		{name: "policy block", content: `{"error":"POLICY_BLOCK"}`, want: "permission"},
		{name: "not found", content: `{"error":"tool not found"}`, want: "not_found"},
		{name: "malformed", content: `{"error":`, want: "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyQwenToolError(tt.content); got != tt.want {
				t.Fatalf("classifyQwenToolError(%q) = %q, want %q", tt.content, got, tt.want)
			}
		})
	}
}

func TestQwenToolErrorFamilyRanking(t *testing.T) {
	t.Parallel()

	events := []QwenToolErrorEvent{
		{Content: `{"error":"permission denied"}`, Index: 8, Tokens: 80},
		{Content: `{"error":"request timeout"}`, Index: 3, Tokens: 30},
		{Content: `{"error":"tool not found"}`, Index: 6, Tokens: 60},
		{Content: `{"error":"timeout waiting for tool"}`, Index: 11, Tokens: 110},
		{Content: `{"error":"permission denied"}`, Index: 14, Tokens: 140},
	}
	want := []QwenToolErrorFamily{
		{Family: "permission", Count: 2, FirstIndex: 8, LastIndex: 14, Tokens: 220},
		{Family: "timeout", Count: 2, FirstIndex: 3, LastIndex: 11, Tokens: 140},
		{Family: "not_found", Count: 1, FirstIndex: 6, LastIndex: 6, Tokens: 60},
	}

	if got := rankQwenToolErrorFamilies(events); !reflect.DeepEqual(got, want) {
		t.Fatalf("rankQwenToolErrorFamilies() = %#v, want %#v", got, want)
	}
}
