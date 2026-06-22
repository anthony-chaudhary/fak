// Tests for the pure API-response token-usage parsers in webbench-token-measure.
// parseOpenAIResponse and parseAnthropicResponse are deterministic, side-effect
// free JSON decoders, so they are exercised directly against hand-computed
// expected TokenUsage values and against malformed input.
package main

import "testing"

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
