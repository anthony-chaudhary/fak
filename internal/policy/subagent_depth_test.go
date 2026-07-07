package policy

import (
	"errors"
	"strings"
	"testing"
)

// TestSubagentDepthManifestRoundTrip is the end-to-end wiring witness: a manifest
// that declares subagent_depth parses (the field is a known key — before it was
// wired, DisallowUnknownFields rejected it), compiles, and the resolved Runtime
// carries an enforcing cap.
func TestSubagentDepthManifestRoundTrip(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"subagent_depth":{"max_depth":3}}`))
	if err != nil {
		t.Fatalf("manifest with subagent_depth should parse+compile, got: %v", err)
	}
	if rt.SubagentDepth == nil {
		t.Fatal("resolved Runtime.SubagentDepth is nil; the block did not compile through")
	}
	if got := rt.SubagentDepth.MaxDepth; got != 3 {
		t.Fatalf("MaxDepth = %d, want 3", got)
	}
	if err := rt.SubagentDepth.AdmitDepth(3); err != nil {
		t.Errorf("depth 3 is at the cap and must admit, got: %v", err)
	}
	if err := rt.SubagentDepth.AdmitDepth(4); !errors.Is(err, ErrSubagentDepthExceeded) {
		t.Errorf("depth 4 over cap 3 must be ErrSubagentDepthExceeded, got: %v", err)
	}
}

// TestSubagentDepthUnknownFieldRejectedPreWiring documents the pre-wiring
// behavior the round-trip test depends on: a manifest key the decoder does not
// know is a hard error. (This is what made declaring subagent_depth impossible
// before it became a first-class field.)
func TestSubagentDepthManifestRejectsUnknownSibling(t *testing.T) {
	_, err := ParseRuntime([]byte(`{"subagent_depth":{"max_depth":3,"bogus":1}}`))
	if err == nil {
		t.Fatal("an unknown nested key must fail loud (DisallowUnknownFields)")
	}
}

func TestSubagentDepthCompileValidation(t *testing.T) {
	// Absent block => nil, and the default cap applies via the nil receiver.
	got, err := compileSubagentDepth(nil)
	if err != nil || got != nil {
		t.Fatalf("nil rule => (nil,nil), got (%v,%v)", got, err)
	}
	// A valid cap compiles.
	if _, err := compileSubagentDepth(&SubagentDepthRule{MaxDepth: 5}); err != nil {
		t.Fatalf("max_depth=5 should compile, got: %v", err)
	}
	// A non-positive cap fails loud at load.
	for _, bad := range []int{0, -1} {
		if _, err := compileSubagentDepth(&SubagentDepthRule{MaxDepth: bad}); err == nil {
			t.Errorf("max_depth=%d must fail loud at load", bad)
		}
	}
}

func TestSubagentDepthAdmit(t *testing.T) {
	r := &SubagentDepthRule{MaxDepth: 2}
	cases := []struct {
		depth   int
		wantErr bool
	}{
		{-1, true}, // invalid
		{0, true},  // depth 0 is the root, not a subagent
		{1, false}, // first-level subagent
		{2, false}, // at the cap
		{3, true},  // over the cap
	}
	for _, c := range cases {
		err := r.AdmitDepth(c.depth)
		if c.wantErr && err == nil {
			t.Errorf("AdmitDepth(%d) admitted, want refusal", c.depth)
		}
		if !c.wantErr && err != nil {
			t.Errorf("AdmitDepth(%d) refused (%v), want admit", c.depth, err)
		}
		if c.wantErr && err != nil && !errors.Is(err, ErrSubagentDepthExceeded) {
			t.Errorf("AdmitDepth(%d) error not ErrSubagentDepthExceeded: %v", c.depth, err)
		}
	}
}

// TestSubagentDepthNilReceiverFailsClosed proves the security property: an
// unconfigured deployment (nil rule) is NOT unbounded — it enforces
// DefaultMaxSubagentDepth.
func TestSubagentDepthNilReceiverFailsClosed(t *testing.T) {
	var r *SubagentDepthRule // no manifest block
	if got := r.Cap(); got != DefaultMaxSubagentDepth {
		t.Fatalf("nil Cap() = %d, want default %d", got, DefaultMaxSubagentDepth)
	}
	if err := r.AdmitDepth(DefaultMaxSubagentDepth); err != nil {
		t.Errorf("depth at the default cap must admit on a nil rule, got: %v", err)
	}
	if err := r.AdmitDepth(DefaultMaxSubagentDepth + 1); !errors.Is(err, ErrSubagentDepthExceeded) {
		t.Errorf("depth over the default cap must refuse on a nil rule, got: %v", err)
	}
}

func TestSubagentDepthAdmitChildOf(t *testing.T) {
	r := &SubagentDepthRule{MaxDepth: 2}
	// root (depth 0) may spawn a child at depth 1.
	if err := r.AdmitChildOf(0); err != nil {
		t.Errorf("root may spawn a first-level child, got: %v", err)
	}
	// a depth-2 parent's child would be depth 3 > cap 2 -> refuse.
	if err := r.AdmitChildOf(2); !errors.Is(err, ErrSubagentDepthExceeded) {
		t.Errorf("a child of a depth-2 parent exceeds cap 2, got: %v", err)
	}
	if err := r.AdmitChildOf(-1); !errors.Is(err, ErrSubagentDepthExceeded) {
		t.Errorf("a negative parent depth is a caller bug, want refusal, got: %v", err)
	}
}

// TestSubagentDepthSummary confirms the operator-visible summary renders the cap
// in both the declared and defaulted cases.
func TestSubagentDepthSummary(t *testing.T) {
	rt, err := ParseRuntime([]byte(`{"subagent_depth":{"max_depth":4}}`))
	if err != nil {
		t.Fatal(err)
	}
	if s := SummaryRuntime(rt); !strings.Contains(s, "subagent depth     : max_depth=4") {
		t.Errorf("summary missing declared depth cap:\n%s", s)
	}
	def, err := ParseRuntime([]byte(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	if s := SummaryRuntime(def); !strings.Contains(s, "default — fail-closed") {
		t.Errorf("summary missing defaulted depth cap:\n%s", s)
	}
}
