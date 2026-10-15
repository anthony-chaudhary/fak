package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"
)

// fak-test:justify why=regression when=changed:cmd/fak/ops_run*.go
// fak-test:runtime fast est=1s
func TestOpsProviderConformanceProbeStream(t *testing.T) {
	const start = `{"model":"exact-model","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"fak_inference_preflight","arguments":"{\"ok\":true}"}}]},"finish_reason":null}]}`
	const finish = `{"model":"exact-model","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`
	const usage = `{"model":"exact-model","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`
	const done = "data: [DONE]\n\n"
	data := func(s string) string { return "data: " + s + "\n\n" }
	valid := data(start) + data(finish) + done
	fragmentStart := strings.Replace(start, "fak_inference_preflight", "fak_inference_", 1)
	fragmentStart = strings.Replace(fragmentStart, `{\"ok\":true}`, `{\"ok\":`, 1)
	const fragment = `{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"preflight","arguments":"true}"}}]}}]}`
	for _, tc := range []struct{ name, wire, reason string }{
		{"valid", valid, ""},
		{"implicit-choice-zero", strings.ReplaceAll(valid, `"index":0,"delta"`, `"delta"`), ""},
		{"null-choice-index", strings.Replace(valid, `"index":0,"delta"`, `"index":null,"delta"`, 1), "malformed_tool_call"},
		{"string-choice-index", strings.Replace(valid, `"index":0,"delta"`, `"index":"0","delta"`, 1), "malformed_tool_call"},
		{"fragmented", data(fragmentStart) + data(fragment) + data(finish) + done, ""},
		{"usage-only", data(start) + data(finish) + data(usage) + done, ""},
		{"wrong-model", strings.Replace(valid, "exact-model", "EXACT-MODEL", 1), "model_mismatch"},
		{"no-model", strings.ReplaceAll(valid, `"model":"exact-model",`, ""), "incomplete_sse"},
		{"empty-choices", data(`{"model":"exact-model","choices":[]}`) + done, "malformed_sse"},
		{"null-choices", data(`{"model":"exact-model","choices":null}`) + done, "malformed_sse"},
		{"two-choices", data(`{"model":"exact-model","choices":[{"index":0},{"index":1}]}`) + done, "malformed_tool_call"},
		{"wrong-tool-index", strings.Replace(valid, `"index":0,"id"`, `"index":1,"id"`, 1), "malformed_tool_call"},
		{"wrong-tool-name", strings.Replace(valid, "fak_inference_preflight", "other", 1), "malformed_tool_call"},
		{"truncated-args", strings.Replace(valid, `{\"ok\":true}`, `{\"ok\":`, 1), "malformed_tool_call"},
		{"extra-args", strings.Replace(valid, `{\"ok\":true}`, `{\"ok\":true,\"extra\":1}`, 1), "malformed_tool_call"},
		{"duplicate-args", strings.Replace(valid, `{\"ok\":true}`, `{\"ok\":true,\"ok\":true}`, 1), "malformed_tool_call"},
		{"length-finish", strings.Replace(valid, `"tool_calls"}]}`, `"length"}]}`, 1), "incomplete_sse"},
		{"no-finish", data(start) + done, "incomplete_sse"},
		{"usage-not-terminal", data(start) + data(usage) + done, "malformed_sse"},
		{"usage-null", data(start) + data(finish) + data(`{"model":"exact-model","choices":[],"usage":null}`) + done, "malformed_sse"},
		{"usage-wrong-model", data(start) + data(finish) + data(strings.Replace(usage, "exact-model", "other", 1)) + done, "model_mismatch"},
		{"usage-negative", data(start) + data(finish) + data(strings.Replace(usage, `"prompt_tokens":1`, `"prompt_tokens":-1`, 1)) + done, "malformed_sse"},
		{"malformed", data("{not-json}") + done, "malformed_sse"},
		{"no-done", data(start) + data(finish), "incomplete_sse"},
		{"after-done", valid + data(usage), "malformed_sse"},
		{"after-done-byte", valid + "\n", "malformed_sse"},
		{"truncated-event", data(start) + data(finish) + "data: [DONE]\n", "incomplete_sse"},
		{"too-large", strings.Repeat("x", opsRunInferenceProbeMaxBytes+2), "response_too_large"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := opsRunReadInferenceProbe(context.Background(), strings.NewReader(tc.wire), "exact-model")
			if got.Reason != tc.reason {
				t.Fatalf("reason=%q, want %q", got.Reason, tc.reason)
			}
			observed := tc.wire
			if len(observed) > opsRunInferenceProbeMaxBytes+1 {
				observed = observed[:opsRunInferenceProbeMaxBytes+1]
			}
			sum := sha256.Sum256([]byte(observed))
			if got.ResponseDigest != "sha256:"+hex.EncodeToString(sum[:]) {
				t.Fatal("digest does not match observed wire bytes")
			}
		})
	}
}
