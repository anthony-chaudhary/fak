// Tests for parseAction, the pure model-response-to-action classifier.
//
// parseAction lowercases its input and returns the first matching action in
// precedence order: "click" wins over "fill"/"type", which win over
// "done"/"complete"; anything unmatched falls through to "wait". The cases
// below pin each branch, the case-insensitivity, and the precedence ordering.
package main

import "testing"

func TestParseAction(t *testing.T) {
	tests := []struct {
		name     string
		response string
		want     string
	}{
		{"click lowercase", "click(#submit)", "click"},
		{"click uppercase is lowercased", "CLICK the button", "click"},
		{"fill keyword", "fill(#name, value)", "fill"},
		{"type maps to fill", "type the password", "fill"},
		{"done keyword", "done", "done"},
		{"complete maps to done", "task is complete", "done"},
		{"no keyword falls through to wait", "scroll down a bit", "wait"},
		{"empty string is wait", "", "wait"},
		// Precedence: click is checked before fill/type.
		{"click precedence over fill", "click then fill the form", "click"},
		// Precedence: fill/type checked before done/complete.
		{"fill precedence over done", "fill the field, then done", "fill"},
		// Substring matching: "completed" contains "complete".
		{"completed substring matches done", "the action completed", "done"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseAction(tt.response)
			if got != tt.want {
				t.Errorf("parseAction(%q) = %q, want %q", tt.response, got, tt.want)
			}
		})
	}
}
