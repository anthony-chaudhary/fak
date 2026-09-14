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
			printTurnkeyBackendStamp(&buf, tt.decision, metalResidencyStamp{})
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

// TestPrintTurnkeyBackendStampQualifiesLiveMetal pins the #12875 honesty contract: a live Metal
// decision is NOT allowed to assert an unqualified "Metal GPU" when the observed device residency
// shows no Q8/Q6_K weights (decode silently falling back to CPU) or when the Q8 band was declined
// at load. The decision-time backend is preserved and the residency truth is appended.
func TestPrintTurnkeyBackendStampQualifiesLiveMetal(t *testing.T) {
	tests := []struct {
		name           string
		res            metalResidencyStamp
		wantSubstrings []string
	}{
		{
			name:           "zero residency qualifies",
			res:            metalResidencyStamp{ResidencyKnown: true},
			wantSubstrings: []string{"backend=metal", "device-resident-weights=0 (cpu-projection-fallback)"},
		},
		{
			name:           "q8 declined qualifies",
			res:            metalResidencyStamp{Q8Declined: true, ResidencyKnown: true},
			wantSubstrings: []string{"backend=metal", "q8-device-resident=declined"},
		},
		{
			name:           "resident weights reported",
			res:            metalResidencyStamp{Q8Resident: 272, Q6Resident: 16, ResidencyKnown: true},
			wantSubstrings: []string{"backend=metal", "device-resident-weights=q8:272,q6k:16"},
		},
		{
			name:           "unobserved stays plain",
			res:            metalResidencyStamp{},
			wantSubstrings: []string{"backend=metal"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			printTurnkeyBackendStamp(&buf, serveMetalDecision{live: true}, tt.res)
			out := buf.String()
			for _, want := range tt.wantSubstrings {
				if !strings.Contains(out, want) {
					t.Fatalf("stamp output %q missing %q", out, want)
				}
			}
			if !tt.res.ResidencyKnown && !tt.res.Q8Declined && strings.Contains(out, "device-resident-weights") {
				t.Fatalf("unobserved residency must not fabricate a device-resident claim: %q", out)
			}
		})
	}
}

// TestMetalResidencyStampFrom pins the report→stamp mapping, including the nil (no native model)
// case that must stay "unobserved" rather than a fabricated zero.
func TestMetalResidencyStampFrom(t *testing.T) {
	if got := metalResidencyStampFrom(nil); got.ResidencyKnown {
		t.Fatalf("nil report must be unobserved, got %+v", got)
	}
	got := metalResidencyStampFrom(map[string]any{
		"metal_live_q8_weights":    272,
		"metal_live_q6_weights":    16,
		"metal_q8_residency_error": "declined: over budget",
	})
	if !got.ResidencyKnown || got.Q8Resident != 272 || got.Q6Resident != 16 || !got.Q8Declined {
		t.Fatalf("metalResidencyStampFrom mapped wrong: %+v", got)
	}
	// A nil error string is not a decline.
	got = metalResidencyStampFrom(map[string]any{"metal_live_q8_weights": 0, "metal_live_q6_weights": 0})
	if got.Q8Declined {
		t.Fatalf("empty residency error must not read as a decline: %+v", got)
	}
}
