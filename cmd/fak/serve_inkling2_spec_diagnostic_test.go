package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestInkling2ExplicitSpecDiagnostic(t *testing.T) {
	cases := []struct {
		name     string
		raw      string
		set      bool
		wantInfo bool
	}{
		{name: "unset"},
		{name: "zero", raw: "0", set: true},
		{name: "negative", raw: "-1", set: true},
		{name: "invalid", raw: "depth-four", set: true},
		{name: "one-alias", raw: "1", set: true, wantInfo: true},
		{name: "depth-four", raw: "4", set: true, wantInfo: true},
		{name: "maximum-depth", raw: "8", set: true, wantInfo: true},
		{name: "above-maximum-remains-observational", raw: "999", set: true, wantInfo: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			message := inkling2ExplicitSpecDiagnostic(func(name string) (string, bool) {
				if name != inkling2SpecEnv {
					t.Fatalf("lookup %q, want %q", name, inkling2SpecEnv)
				}
				return tc.raw, tc.set
			})
			if !tc.wantInfo {
				if message != (gateway.StartupMessage{}) {
					t.Fatalf("unexpected diagnostic: %+v", message)
				}
				return
			}
			if message.Source != "serve" || message.Kind != "inkling2-spec-config" || message.Level != "info" {
				t.Fatalf("diagnostic identity = %+v", message)
			}
			if !strings.Contains(message.Text, inkling2SpecEnv+"=\""+tc.raw+"\"") || !strings.Contains(message.Text, "active for greedy decode") {
				t.Fatalf("diagnostic text %q does not name the active setting", message.Text)
			}
		})
	}
	activeMsg := inkling2ExplicitSpecDiagnosticFor("greedy decode; spec=dspark depth=4; cohorts up to 4", func(string) (string, bool) { return "4", true })
	if activeMsg.Level != "info" || !strings.Contains(activeMsg.Text, "(spec=dspark depth=4)") || strings.Contains(activeMsg.Text, "cohorts") {
		t.Fatalf("active banner diagnostic = %+v", activeMsg)
	}
	warnMsg := inkling2ExplicitSpecDiagnosticFor("greedy decode; spec=off", func(string) (string, bool) { return "4", true })
	if warnMsg.Level != "warn" || !strings.Contains(warnMsg.Text, "selected plain decode") {
		t.Fatalf("fallback banner diagnostic = %+v", warnMsg)
	}
}
