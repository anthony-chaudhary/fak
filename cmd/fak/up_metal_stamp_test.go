package main

import (
	"bytes"
	"strings"
	"testing"
)

// TestMetalStampLine pins the stamp grammar: a live decision stamps
// backend=metal, a shaded skip reason stamps backend=cpu (<reason>), and an
// empty reason (Metal never targeted, e.g. non-darwin auto) stamps nothing so
// callers print no line at all.
func TestMetalStampLine(t *testing.T) {
	tests := []struct {
		name   string
		live   bool
		reason serveMetalSkipReason
		want   string
	}{
		{name: "live", live: true, want: "backend=metal"},
		{name: "not compiled", reason: skipReasonNotCompiled, want: "backend=cpu (metal-not-compiled)"},
		{name: "no device", reason: skipReasonNoDevice, want: "backend=cpu (metal-no-device)"},
		{name: "no decision prints nothing", want: ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := metalStampLine(tt.live, tt.reason); got != tt.want {
				t.Fatalf("metalStampLine(%v, %q) = %q, want %q", tt.live, tt.reason, got, tt.want)
			}
		})
	}
}

// TestMetalStampChatWord pins the mock-chat honesty swap: the forward is named
// Metal only when the session actually resolved Metal.
func TestMetalStampChatWord(t *testing.T) {
	if got := metalStampChatWord(true); got != "Metal" {
		t.Fatalf("metalStampChatWord(true) = %q, want Metal", got)
	}
	if got := metalStampChatWord(false); got != "CPU" {
		t.Fatalf("metalStampChatWord(false) = %q, want CPU", got)
	}
}

// TestPrintTurnkeyBackendStamp pins the one-line CPU-truth emission contract,
// including the reason clauses mirroring the modelbench/serve wording, and the
// silent no-decision case.
func TestPrintTurnkeyBackendStamp(t *testing.T) {
	tests := []struct {
		name           string
		decision       serveMetalDecision
		wantSubstrings []string
		wantEmpty      bool
	}{
		{
			name:           "live metal",
			decision:       serveMetalDecision{live: true},
			wantSubstrings: []string{"fak up: in-kernel chat forward: Metal GPU (backend=metal)"},
		},
		{
			name:           "cpu not compiled",
			decision:       serveMetalDecision{skippedBecause: skipReasonNotCompiled},
			wantSubstrings: []string{"backend=cpu (metal-not-compiled)", "no Metal support compiled in"},
		},
		{
			name:           "cpu no device",
			decision:       serveMetalDecision{skippedBecause: skipReasonNoDevice},
			wantSubstrings: []string{"backend=cpu (metal-no-device)", "no usable Metal device"},
		},
		{
			name:      "no decision stays silent",
			decision:  serveMetalDecision{},
			wantEmpty: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			printTurnkeyBackendStamp(&buf, tt.decision)
			if tt.wantEmpty {
				if buf.Len() != 0 {
					t.Fatalf("stamp emitted %q, want no output", buf.String())
				}
				return
			}
			out := buf.String()
			for _, want := range tt.wantSubstrings {
				if !strings.Contains(out, want) {
					t.Fatalf("stamp output %q missing %q", out, want)
				}
			}
		})
	}
}
