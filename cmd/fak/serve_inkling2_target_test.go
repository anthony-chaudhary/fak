package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/inklingmetal"
)

// Serve binds only the vendored target, so the run selector either agrees with
// it or is refused before any admission; other routes never read the selector.
func TestServeInkling2MetalRouteHonoursTargetSelector(t *testing.T) {
	path := writeInkling2HeaderGGUF(t, serveInkling2LoopArch)
	t.Setenv(inkling2MetalTargetEnv, "")
	_, unset := serveInkling2MetalRoute(path, nil, true)
	for _, selector := range []string{"vendored", " Vendored "} {
		t.Setenv(inkling2MetalTargetEnv, selector)
		routed, err := serveInkling2MetalRoute(path, nil, true)
		if !routed || (err == nil) != (unset == nil) || (err != nil && err.Error() != unset.Error()) {
			t.Fatalf("%q: routed=%v err=%v, want the unset verdict %v", selector, routed, err, unset)
		}
	}

	t.Setenv(inkling2MetalTargetEnv, "native")
	routed, err := serveInkling2MetalRoute(path, nil, true)
	var refusal *inkling2MetalServeRefusal
	if !routed || !errors.As(err, &refusal) || !strings.Contains(err.Error(), "not supported by serve yet") ||
		!strings.Contains(err.Error(), inkling2MetalTargetNativeLabel) || errors.Is(err, inklingmetal.ErrUnavailable) {
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
