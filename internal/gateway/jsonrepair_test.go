package gateway

import (
	"encoding/json"
	"testing"
)

func TestRepairJSON(t *testing.T) {
	tests := []struct {
		name       string
		input      string
		want       string
		checkValid bool
	}{
		{
			name:       "already valid JSON",
			input:      `{"foo":"bar"}`,
			want:       `{"foo":"bar"}`,
			checkValid: true,
		},
		{
			name:       "JSON wrapped in markdown backticks with language",
			input:      "```json\n{\"foo\":\"bar\"}\n```",
			want:       `{"foo":"bar"}`,
			checkValid: true,
		},
		{
			name:       "JSON with markdown backticks without language",
			input:      "```\n{\"foo\":\"bar\"}\n```",
			want:       `{"foo":"bar"}`,
			checkValid: true,
		},
		{
			name:       "JSON with trailing commas in objects",
			input:      `{"a": 1, "b": 2,}`,
			want:       `{"a": 1, "b": 2}`,
			checkValid: true,
		},
		{
			name:       "JSON with trailing commas in arrays",
			input:      `[1, 2, 3,]`,
			want:       `[1, 2, 3]`,
			checkValid: true,
		},
		{
			name:       "JSON with nested trailing commas",
			input:      `{"a": [1, 2,], "b": {"c": 3,},}`,
			want:       `{"a": [1, 2], "b": {"c": 3}}`,
			checkValid: true,
		},
		{
			name:       "commas inside string literals",
			input:      `{"message": "hello, world,"}`,
			want:       `{"message": "hello, world,"}`,
			checkValid: true,
		},
		{
			name:       "unclosed brackets",
			input:      `{"items": [1, 2, 3`,
			want:       `{"items": [1, 2, 3]}`,
			checkValid: true,
		},
		{
			name:       "unclosed object and array",
			input:      `{"a": 1, "b": [2, 3`,
			want:       `{"a": 1, "b": [2, 3]}`,
			checkValid: true,
		},
		{
			name:       "escaped quotes inside strings",
			input:      `{"text": "quote: \"hello\", "}`,
			want:       `{"text": "quote: \"hello\", "}`,
			checkValid: true,
		},
		{
			name:       "empty input",
			input:      "",
			want:       "",
			checkValid: false,
		},
		{
			name:       "whitespace only",
			input:      "   \n\t  ",
			want:       "   \n\t  ",
			checkValid: false,
		},
		{
			name:       "unclosed string literal",
			input:      `{"status": "in-progress`,
			want:       `{"status": "in-progress"}`,
			checkValid: true,
		},
		{
			name:       "leading and trailing markdown prose",
			input:      `Here is the result: {"count": 42} Have a nice day!`,
			want:       `{"count": 42}`,
			checkValid: true,
		},
		{
			name:       "trailing comma in unclosed array",
			input:      `[1, 2, 3,`,
			want:       `[1, 2, 3]`,
			checkValid: true,
		},
		{
			name:       "unclosed object ending with colon",
			input:      `{"a": 1, "b":`,
			want:       `{"a": 1, "b": null}`,
			checkValid: true,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := RepairJSON([]byte(tc.input))
			if tc.want != "" && string(got) != tc.want {
				t.Fatalf("RepairJSON(%q) = %q, want %q", tc.input, string(got), tc.want)
			}
			if tc.checkValid {
				if !json.Valid(got) {
					t.Fatalf("RepairJSON(%q) produced invalid JSON: %q", tc.input, string(got))
				}
			}
		})
	}
}

func TestStripCodeFences(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "fenced json",
			input: "```json\n{\"foo\":\"bar\"}\n```",
			want:  `{"foo":"bar"}`,
		},
		{
			name:  "fenced without language",
			input: "```\n{\"foo\":\"bar\"}\n```",
			want:  `{"foo":"bar"}`,
		},
		{
			name:  "prose outside delimiters",
			input: `Here is the data: {"foo":"bar"} thank you!`,
			want:  `{"foo":"bar"}`,
		},
		{
			name:  "unclosed brackets left intact",
			input: `{"items": [1, 2`,
			want:  `{"items": [1, 2`,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := StripCodeFences([]byte(tc.input))
			if string(got) != tc.want {
				t.Fatalf("StripCodeFences(%q) = %q, want %q", tc.input, string(got), tc.want)
			}
		})
	}
}

func TestRemoveTrailingCommas(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "object comma",
			input: `{"a": 1,}`,
			want:  `{"a": 1}`,
		},
		{
			name:  "array comma",
			input: `[1, 2,]`,
			want:  `[1, 2]`,
		},
		{
			name:  "nested commas",
			input: `{"a": [1, 2,], "b": 3,}`,
			want:  `{"a": [1, 2], "b": 3}`,
		},
		{
			name:  "string commas preserved",
			input: `{"text": "a, b, c,"}`,
			want:  `{"text": "a, b, c,"}`,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := RemoveTrailingCommas([]byte(tc.input))
			if string(got) != tc.want {
				t.Fatalf("RemoveTrailingCommas(%q) = %q, want %q", tc.input, string(got), tc.want)
			}
		})
	}
}

func TestBalanceJSONDelimiters(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "unclosed array",
			input: `[1, 2, 3`,
			want:  `[1, 2, 3]`,
		},
		{
			name:  "unclosed nested",
			input: `{"a": [1, 2`,
			want:  `{"a": [1, 2]}`,
		},
		{
			name:  "unclosed string literal",
			input: `{"name": "hello`,
			want:  `{"name": "hello"}`,
		},
		{
			name:  "unclosed with trailing comma",
			input: `{"items": [1, 2,`,
			want:  `{"items": [1, 2]}`,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			got := BalanceJSONDelimiters([]byte(tc.input))
			if string(got) != tc.want {
				t.Fatalf("BalanceJSONDelimiters(%q) = %q, want %q", tc.input, string(got), tc.want)
			}
			if !json.Valid(got) {
				t.Fatalf("BalanceJSONDelimiters(%q) produced invalid JSON: %q", tc.input, string(got))
			}
		})
	}
}
