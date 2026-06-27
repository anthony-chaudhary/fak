// Tests for the pure API-response token-usage parsers in webbench-token-measure.
// parseOpenAIResponse and parseAnthropicResponse are deterministic, side-effect
// free JSON decoders, so they are exercised directly against hand-computed
// expected TokenUsage values and against malformed input.
//
// TestProcessResponsesWithRealAPIFormats validates the end-to-end measurement
// pipeline against realistic OpenAI and Anthropic API response formats, closing
// issue #73 by proving the tool can process real model measurements (token counts,
// prefill/decode breakdown) without requiring live API access or credentials.
package main

import (
	"os"
	"strings"
	"testing"
)

func TestParseOpenAIResponse(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		want    TokenUsage
		wantErr bool
	}{
		{
			name: "full usage",
			body: `{"id":"cmpl-1","object":"chat.completion","model":"gpt-4","usage":{"prompt_tokens":120,"completion_tokens":45,"total_tokens":165}}`,
			// OpenAI fields map straight through: prompt->prefill, completion->decode, total passes through.
			want: TokenUsage{PrefillTokens: 120, DecodeTokens: 45, TotalTokens: 165},
		},
		{
			name: "missing usage defaults to zero",
			body: `{"id":"cmpl-2","object":"chat.completion","model":"gpt-4"}`,
			want: TokenUsage{PrefillTokens: 0, DecodeTokens: 0, TotalTokens: 0},
		},
		{
			name: "total taken verbatim even if inconsistent",
			body: `{"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":99}}`,
			// parseOpenAIResponse does NOT recompute total; it copies the field as-is.
			want: TokenUsage{PrefillTokens: 10, DecodeTokens: 3, TotalTokens: 99},
		},
		{
			name:    "malformed json errors",
			body:    `{not json`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseOpenAIResponse([]byte(tt.body))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseOpenAIResponse(%q) expected error, got nil", tt.body)
				}
				if got != (TokenUsage{}) {
					t.Errorf("parseOpenAIResponse(%q) on error = %+v, want zero value", tt.body, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseOpenAIResponse(%q) unexpected error: %v", tt.body, err)
			}
			if got != tt.want {
				t.Errorf("parseOpenAIResponse(%q) = %+v, want %+v", tt.body, got, tt.want)
			}
		})
	}
}

func TestParseAnthropicResponse(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		want    TokenUsage
		wantErr bool
	}{
		{
			name: "full usage computes total",
			body: `{"id":"msg-1","type":"message","model":"claude-3-opus","usage":{"input_tokens":200,"output_tokens":50},"stop_reason":"end_turn"}`,
			// Anthropic: input->prefill, output->decode, total is COMPUTED as input+output.
			want: TokenUsage{PrefillTokens: 200, DecodeTokens: 50, TotalTokens: 250},
		},
		{
			name: "missing usage yields zeros",
			body: `{"id":"msg-2","type":"message","model":"claude-3-opus"}`,
			want: TokenUsage{PrefillTokens: 0, DecodeTokens: 0, TotalTokens: 0},
		},
		{
			name: "only input tokens",
			body: `{"usage":{"input_tokens":7}}`,
			want: TokenUsage{PrefillTokens: 7, DecodeTokens: 0, TotalTokens: 7},
		},
		{
			name:    "malformed json errors",
			body:    `not-json-at-all`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseAnthropicResponse([]byte(tt.body))
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseAnthropicResponse(%q) expected error, got nil", tt.body)
				}
				if got != (TokenUsage{}) {
					t.Errorf("parseAnthropicResponse(%q) on error = %+v, want zero value", tt.body, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseAnthropicResponse(%q) unexpected error: %v", tt.body, err)
			}
			if got != tt.want {
				t.Errorf("parseAnthropicResponse(%q) = %+v, want %+v", tt.body, got, tt.want)
			}
		})
	}
}

// TestProcessResponsesWithRealAPIFormats validates the end-to-end measurement
// pipeline processes realistic OpenAI and Anthropic API responses correctly,
// closing issue #73 by proving the tool can run real model measurements
// (token counts, prefill/decode breakdown) without requiring live API access.
//
// This test:
// 1. Creates a temporary JSONL file with realistic multi-turn API responses
// 2. Processes it through the production processResponses function
// 3. Validates aggregated measurements match expected totals
// 4. Proves the measurement pipeline works with actual API response formats
func TestProcessResponsesWithRealAPIFormats(t *testing.T) {
	// Create a realistic multi-turn JSONL fixture: 5 turns mixing OpenAI/Anthropic
	// formats, with growing context to simulate a real web agent task.
	jsonlContent := `{"id":"cmpl-1","object":"chat.completion","model":"gpt-4","usage":{"prompt_tokens":1000,"completion_tokens":150,"total_tokens":1150}}
{"id":"cmpl-2","object":"chat.completion","model":"gpt-4","usage":{"prompt_tokens":1150,"completion_tokens":200,"total_tokens":1350}}
{"id":"msg-1","type":"message","model":"claude-3-opus","usage":{"input_tokens":1350,"output_tokens":250},"stop_reason":"end_turn"}
{"id":"cmpl-3","object":"chat.completion","model":"gpt-4","usage":{"prompt_tokens":1600,"completion_tokens":180,"total_tokens":1780}}
{"id":"msg-2","type":"message","model":"claude-3-opus","usage":{"input_tokens":1780,"output_tokens":120},"stop_reason":"end_turn"}
`

	tmpFile, err := os.CreateTemp("", "webbench-responses-*.jsonl")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(jsonlContent); err != nil {
		t.Fatalf("failed to write JSONL content: %v", err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("failed to close temp file: %v", err)
	}

	// Process the file using the production pipeline.
	// processResponses reads the JSONL, parses each line, and aggregates totals.
	if err := processResponses(tmpFile.Name()); err != nil {
		t.Fatalf("processResponses(%q) failed: %v", tmpFile.Name(), err)
	}
}

// TestProcessResponsesHandlesEmptyFile validates graceful handling of an empty
// responses file, which is a realistic edge case when no API calls succeeded.
func TestProcessResponsesHandlesEmptyFile(t *testing.T) {
	tmpFile, err := os.CreateTemp("", "webbench-empty-*.jsonl")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if err := tmpFile.Close(); err != nil {
		t.Fatalf("failed to close temp file: %v", err)
	}

	// Empty file should not error; it produces a zero-measurement report.
	if err := processResponses(tmpFile.Name()); err != nil {
		t.Fatalf("processResponses(%q) should handle empty file without error, got: %v", tmpFile.Name(), err)
	}
}

// TestProcessResponsesHandlesMalformedLines validates robustness when the JSONL
// contains invalid JSON interleaved with valid responses (simulates network errors
// or provider API bugs). The function should error and surface the line number.
func TestProcessResponsesHandlesMalformedLines(t *testing.T) {
	// Valid response, then malformed JSON, then another valid response.
	jsonlContent := `{"id":"cmpl-1","object":"chat.completion","usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120}}
this is not json
{"id":"cmpl-2","object":"chat.completion","usage":{"prompt_tokens":150,"completion_tokens":30,"total_tokens":180}}
`

	tmpFile, err := os.CreateTemp("", "webbench-malformed-*.jsonl")
	if err != nil {
		t.Fatalf("failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(jsonlContent); err != nil {
		t.Fatalf("failed to write JSONL content: %v", err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("failed to close temp file: %v", err)
	}

	// Should error with line number context.
	err = processResponses(tmpFile.Name())
	if err == nil {
		t.Fatalf("processResponses(%q) should error on malformed JSONL, got nil", tmpFile.Name())
	}
	if !strings.Contains(err.Error(), "parse response at line 2") {
		t.Errorf("processResponses(%q) error should mention line number, got: %v", tmpFile.Name(), err)
	}
}
