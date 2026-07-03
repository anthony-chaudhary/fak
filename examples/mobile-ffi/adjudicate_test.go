package main

import "testing"

// TestDecide pins the FFI core's deny/allow/fail-closed round-trip — the same
// three legs main() witnesses and the same three the C shim carries to
// Android/iOS. It runs with CGO_ENABLED=0 (this repo's CI default): the cgo
// shim is excluded, the decision logic is not.
func TestDecide(t *testing.T) {
	cases := []struct {
		name        string
		json        string
		wantAllow   bool
		wantVerdict string
		wantReason  string
	}{
		{"dangerous send_sms", `{"tool":"send_sms","args":{"to":"+1900"}}`, false, "DENY", "POLICY_BLOCK"},
		{"dangerous factory_reset", `{"tool":"factory_reset"}`, false, "DENY", "POLICY_BLOCK"},
		{"benign get_ prefix", `{"tool":"get_battery_level"}`, true, "ALLOW", ""},
		{"benign read_ prefix", `{"tool":"read_calendar"}`, true, "ALLOW", ""},
		{"unknown fails closed", `{"tool":"transfer_funds"}`, false, "DEFAULT_DENY", "DEFAULT_DENY"},
		{"empty tool denied", `{"tool":""}`, false, "DENY", "MALFORMED"},
		{"malformed json denied", `not json`, false, "DENY", "MALFORMED"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := Decide(tc.json)
			if d.Allow != tc.wantAllow {
				t.Errorf("Allow = %v, want %v (%+v)", d.Allow, tc.wantAllow, d)
			}
			if d.Verdict != tc.wantVerdict {
				t.Errorf("Verdict = %q, want %q", d.Verdict, tc.wantVerdict)
			}
			if tc.wantReason != "" && d.Reason != tc.wantReason {
				t.Errorf("Reason = %q, want %q", d.Reason, tc.wantReason)
			}
		})
	}
}
