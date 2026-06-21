package toollint

import (
	"context"
	"strings"
	"testing"
)

// The steward abstains on a clean surface and fires (with a re-derivable witness)
// when the surface carries an error-severity finding.
func TestSurfaceViolation(t *testing.T) {
	// Clean: a read-shaped static + a reachable pure tool => no error finding.
	if v, _ := surfaceViolation([]ToolFacts{
		{Name: "list_all_airports", Kind: KindStatic},
		{Name: "calculate", Kind: KindPure, Hints: cacheableHints()},
	}); v {
		t.Fatalf("surfaceViolation fired on a clean surface")
	}
	// TL003: a static answer for a write-shaped tool is an error => fire.
	v, w := surfaceViolation([]ToolFacts{{Name: "send_alert", Kind: KindStatic}})
	if !v {
		t.Fatalf("surfaceViolation must fire on a TL003 error surface")
	}
	if !strings.HasPrefix(w, "TL003 send_alert:") {
		t.Fatalf("witness should name the finding; got %q", w)
	}
	// TL008: a policy-denied tool also on the fast path is an error => fire.
	v8, w8 := surfaceViolation([]ToolFacts{{Name: "exfiltrate", Kind: KindStatic, PolicyDenied: true}})
	if !v8 || !strings.Contains(w8, "TL008") {
		t.Fatalf("surfaceViolation must fire TL008 on a deny-bypass surface; got v=%v w=%q", v8, w8)
	}
}

// On the real default surface (the seeded vDSO + the default empty policy), the
// steward abstains — registering it is side-effect-free for a healthy kernel.
func TestSurfaceStewardAbstainsOnDefault(t *testing.T) {
	s := surfaceSteward{}
	if s.Name() != "tool-surface-sound" {
		t.Fatalf("steward name = %q, want tool-surface-sound", s.Name())
	}
	if v, w := s.Check(context.Background()); v {
		t.Fatalf("default surface must be lint-clean of errors; steward fired: %q", w)
	}
}
