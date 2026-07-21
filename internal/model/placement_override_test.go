package model

import (
	"errors"
	"testing"
)

// placement_override_test.go — witnesses for the user-supplied per-tensor placement override:
// (1) first-match-wins ordering, (2) no-match falls through to the built-in default, and
// (3) a bad regex (and other malformed input) fails closed with the typed *overrideParseError.
// All pure name-string logic: no GPU, no backend, no clock.

// TestPlacementOverrideFirstMatchWins pins the ordering contract: when two rules both match a name,
// the FIRST rule in the list decides. The same name is placed differently by two overrides that
// differ only in rule order — proving precedence is order, not specificity.
func TestPlacementOverrideFirstMatchWins(t *testing.T) {
	name := "model.layers.0.self_attn.o_proj.weight"

	// self_attn.* -> host wins because it precedes the broader .*proj rule.
	first, err := parsePlacementOverride(`self_attn\.=host;proj\.weight=device`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if onHost, matched := first.resolve(name); !matched || !onHost {
		t.Fatalf("first-order resolve(%q) = (onHost=%v, matched=%v); want (true, true)", name, onHost, matched)
	}

	// Reverse the order: now the broader proj rule fires first and pins it to the device.
	second, err := parsePlacementOverride(`proj\.weight=device;self_attn\.=host`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if onHost, matched := second.resolve(name); !matched || onHost {
		t.Fatalf("second-order resolve(%q) = (onHost=%v, matched=%v); want (false, true)", name, onHost, matched)
	}
}

// TestPlacementOverrideNoMatchDefault proves an unmatched name reports matched=false, and that
// onHostWith folds the override in FRONT of the built-in isExpertWeight default: a matched rule
// overrides it, an unmatched name falls through to it, and an empty override is the fallback itself.
func TestPlacementOverrideNoMatchDefault(t *testing.T) {
	o, err := parsePlacementOverride(`o_proj\.weight=host`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}

	// A name no rule matches -> matched=false (caller uses its default).
	if _, matched := o.resolve("model.layers.0.mlp.gate.weight"); matched {
		t.Fatalf("resolve of unmatched name reported matched=true; want false")
	}

	onHost := o.onHostWith(isExpertWeight)

	// Overridden: an o_proj (normally a device dense weight) is forced to host by the rule.
	if !onHost("model.layers.0.self_attn.o_proj.weight") {
		t.Fatalf("override should pin o_proj to host, got device")
	}
	// Fallthrough: an expert weight no rule matches still offloads via isExpertWeight.
	if !onHost("model.layers.0.mlp.experts.3.gate_proj.weight") {
		t.Fatalf("unmatched expert weight should fall through to isExpertWeight (host)")
	}
	// Fallthrough: the router, a dense weight no rule matches, stays on the device.
	if onHost(routerName(0)) {
		t.Fatalf("unmatched router weight should fall through to isExpertWeight (device)")
	}

	// Empty override returns the fallback unchanged.
	var empty placementOverride
	fb := isExpertWeight
	got := empty.onHostWith(fb)
	for _, n := range []string{routerName(0), "model.layers.0.mlp.experts.1.up_proj.weight"} {
		if got(n) != fb(n) {
			t.Fatalf("empty override changed placement of %q: got %v want %v", n, got(n), fb(n))
		}
	}
}

// TestPlacementOverrideParseFailsClosed proves malformed operator input is rejected with the typed
// *overrideParseError and yields NO partial override — a bad regex, empty pattern, missing '=', and
// unknown target each fail closed. Valid and empty specs still parse cleanly.
func TestPlacementOverrideParseFailsClosed(t *testing.T) {
	bad := []string{
		`experts\.[0-9=host`,       // unbalanced '[' — invalid regexp
		`=host`,                    // empty pattern
		`self_attn`,                // missing '=<target>'
		`o_proj\.weight=somewhere`, // unknown target
	}
	for _, spec := range bad {
		o, err := parsePlacementOverride(spec)
		if err == nil {
			t.Fatalf("parsePlacementOverride(%q) = nil error; want failure", spec)
		}
		var pe *overrideParseError
		if !errors.As(err, &pe) {
			t.Fatalf("parsePlacementOverride(%q) error type = %T; want *overrideParseError", spec, err)
		}
		if len(o.rules) != 0 {
			t.Fatalf("parsePlacementOverride(%q) returned %d rules on failure; want 0 (no partial override)", spec, len(o.rules))
		}
	}

	// A well-formed multi-rule spec parses into the stated number of ordered rules.
	ok, err := parsePlacementOverride(`experts\.=cpu;o_proj\.weight=gpu`)
	if err != nil {
		t.Fatalf("valid spec failed: %v", err)
	}
	if len(ok.rules) != 2 {
		t.Fatalf("valid spec parsed %d rules; want 2", len(ok.rules))
	}

	// Empty / whitespace spec is the empty override, no error.
	if o, err := parsePlacementOverride("   "); err != nil || len(o.rules) != 0 {
		t.Fatalf("empty spec = (%d rules, %v); want (0, nil)", len(o.rules), err)
	}
}
