package agent

import "testing"

// TestPlaceCacheBreakpointFastPathEquivalence pins the exact-equivalence of the cache_control
// pre-scan that now runs BEFORE the whole-body json.Unmarshal in PlaceAnthropicCacheBreakpointWithOutcome.
// The old order decoded the body into a map[string]json.RawMessage first (accepting only a well-formed
// JSON object) and returned AlreadySet only for an object carrying cache_control; anything else bailed
// NonJSON. The hoisted scan must reproduce that byte-for-byte, so a valid-JSON-but-NON-object body that
// merely contains the literal (e.g. a JSON array) must still bail NonJSON, not AlreadySet — the reason
// the guard is skipSpace/'{' + json.Valid, not a bare json.Valid.
func TestPlaceCacheBreakpointFastPathEquivalence(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantReason string
	}{
		{"object_with_cache_control", `{"model":"m","system":[{"type":"text","text":"h","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`, BreakpointReasonAlreadySet},
		// A JSON ARRAY carrying the literal: valid JSON, but NOT an object — the old decode-into-map
		// failed, so this must stay NonJSON (the case a bare json.Valid guard would wrongly flip).
		{"array_with_cache_control", `["cache_control"]`, BreakpointReasonNonJSON},
		// A bare JSON string carrying the literal: valid JSON, not an object -> NonJSON.
		{"string_with_cache_control", `"a cache_control mention"`, BreakpointReasonNonJSON},
		// Malformed body carrying the literal: invalid JSON -> NonJSON (old decode failed too).
		{"malformed_with_cache_control", `{"x": cache_control`, BreakpointReasonNonJSON},
		// Empty body -> NonJSON (unchanged, handled before the scan).
		{"empty", ``, BreakpointReasonNonJSON},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, outcome := PlaceAnthropicCacheBreakpointWithOutcome([]byte(tc.body))
			if outcome.Reason != tc.wantReason {
				t.Fatalf("reason = %q, want %q", outcome.Reason, tc.wantReason)
			}
			// The fast path is identity — it never rewrites the body.
			if string(out) != tc.body {
				t.Fatalf("body was mutated on the fast path:\n in: %s\nout: %s", tc.body, out)
			}
		})
	}
}
