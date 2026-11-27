package main

import (
	"errors"
	"strings"
	"testing"
)

// Serve binds fak's device-resident Metal forward; any other selector is
// refused before admission, and non-Metal routes never read the selector.
func TestServeInkling2MetalRouteHonoursTargetSelector(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	t.Setenv(inkling2MetalTargetEnv, "")
	routed, unset := serveInkling2MetalRoute(path, nil, true)
	if !routed {
		t.Fatal("unset: expected Metal route")
	}
	if inkling2MetalResidentAvailable {
		if unset != nil {
			t.Fatalf("unset: err=%v, want resident admitted", unset)
		}
	} else {
		var refusal *inkling2MetalServeRefusal
		if !errors.As(unset, &refusal) || !strings.Contains(unset.Error(), "not available in this build") {
			t.Fatalf("unset without resident: err=%v, want typed build refusal", unset)
		}
	}

	for _, selector := range []string{"vendored", " Vendored "} {
		t.Setenv(inkling2MetalTargetEnv, selector)
		routed, err := serveInkling2MetalRoute(path, nil, true)
		if !routed || err == nil || !strings.Contains(err.Error(), "removed") || strings.Contains(err.Error(), "fak-private") {
			t.Fatalf("%q: routed=%v err=%v, want removed refusal without fak-private", selector, routed, err)
		}
	}

	t.Setenv(inkling2MetalTargetEnv, "native")
	routed, err := serveInkling2MetalRoute(path, nil, true)
	var refusal *inkling2MetalServeRefusal
	if !routed || !errors.As(err, &refusal) || !strings.Contains(err.Error(), "not supported by serve yet") ||
		!strings.Contains(err.Error(), inkling2MetalTargetNativeLabel) {
		t.Fatalf("native: routed=%v err=%v, want the typed not-supported-by-serve refusal", routed, err)
	}
	if plan, err := serveInkling2MetalWeights(path, nil, true); plan != nil || !errors.As(err, &refusal) {
		t.Fatalf("native: admission plan = %v, %v, want the refusal before any plan", plan, err)
	}
	if routed, err := serveInkling2MetalRoute(path, nil, false); routed || err != nil {
		t.Fatalf("native without Metal: routed=%v err=%v, want the route left alone", routed, err)
	}

	t.Setenv(inkling2MetalTargetEnv, "mlx")
	if routed, err := serveInkling2MetalRoute(path, nil, true); !routed || err == nil || !strings.Contains(err.Error(), "unknown "+inkling2MetalTargetEnv) {
		t.Fatalf("unknown selector: routed=%v err=%v, want it named", routed, err)
	}
}
