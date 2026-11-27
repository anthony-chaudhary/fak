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
		wantWarn bool
	}{
		{name: "unset"},
		{name: "zero", raw: "0", set: true},
		{name: "negative", raw: "-1", set: true},
		{name: "invalid", raw: "depth-four", set: true},
		{name: "one-alias", raw: "1", set: true, wantWarn: true},
		{name: "depth-four", raw: "4", set: true, wantWarn: true},
		{name: "maximum-depth", raw: "8", set: true, wantWarn: true},
		{name: "above-maximum-remains-observational", raw: "999", set: true, wantWarn: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			message := inkling2ExplicitSpecDiagnostic(func(name string) (string, bool) {
				if name != inkling2SpecEnv {
					t.Fatalf("lookup %q, want %q", name, inkling2SpecEnv)
				}
				return tc.raw, tc.set
			})
			if !tc.wantWarn {
				if message != (gateway.StartupMessage{}) {
					t.Fatalf("unexpected diagnostic: %+v", message)
				}
				return
			}
			if message.Source != "serve" || message.Kind != "inkling2-spec-config" || message.Level != "warn" {
				t.Fatalf("diagnostic identity = %+v", message)
			}
			if !strings.Contains(message.Text, inkling2SpecEnv+"=\""+tc.raw+"\"") || !strings.Contains(message.Text, "selects plain decode") {
				t.Fatalf("diagnostic text %q does not name the request and selected path", message.Text)
			}
		})
	}
}
