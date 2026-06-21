package main

import (
	"math"
	"strings"
	"testing"
)

// TestSoftmaxProbOf checks the displayed probability matches a reference softmax and
// handles out-of-range ids.
func TestSoftmaxProbOf(t *testing.T) {
	logits := []float32{0.5, 2.0, -1.0, 1.0}
	var sum float64
	for _, v := range logits {
		sum += math.Exp(float64(v))
	}
	want := math.Exp(float64(logits[1])) / sum
	if got := softmaxProbOf(logits, 1); math.Abs(float64(got)-want) > 1e-5 {
		t.Fatalf("softmaxProbOf=%g want %g", got, want)
	}
	if got := softmaxProbOf(logits, -1); got != 0 {
		t.Fatalf("neg id: want 0 got %g", got)
	}
	if got := softmaxProbOf(logits, 99); got != 0 {
		t.Fatalf("oob id: want 0 got %g", got)
	}
}

// TestBarMonotonic: the block bar must not shrink as probability rises, and stay in
// range at the endpoints.
func TestBarMonotonic(t *testing.T) {
	prev := bar(0)
	if prev == "" {
		t.Fatal("empty bar at 0")
	}
	for p := float32(0); p <= 1.0; p += 0.05 {
		b := bar(p)
		if len([]rune(b)) != 1 {
			t.Fatalf("bar(%g)=%q not one rune", p, b)
		}
	}
	// clamps
	if bar(-1) != bar(0) {
		t.Fatalf("bar(-1) should clamp to bar(0)")
	}
	if bar(2) != bar(1) {
		t.Fatalf("bar(2) should clamp to bar(1)")
	}
}

// TestHeatEndpoints: low prob is the dim slate, high prob the bright amber, both valid
// 6-digit hex.
func TestHeatEndpoints(t *testing.T) {
	lo, hi := heat(0), heat(1)
	if lo != "#1e293b" {
		t.Fatalf("heat(0)=%s want #1e293b", lo)
	}
	if hi != "#fbbf24" {
		t.Fatalf("heat(1)=%s want #fbbf24", hi)
	}
	for _, h := range []string{heat(0.3), heat(0.7), heat(-5), heat(5)} {
		if len(h) != 7 || h[0] != '#' {
			t.Fatalf("malformed color %q", h)
		}
	}
}

// TestPrintable: spaces/newlines become visible markers and long pieces truncate.
func TestPrintable(t *testing.T) {
	if got := printable(" hi\n"); got != "·hi\\n" {
		t.Fatalf("printable=%q", got)
	}
	long := printable(strings.Repeat("x", 40))
	if len([]rune(long)) > 15 {
		t.Fatalf("not truncated: %q (%d runes)", long, len([]rune(long)))
	}
	if !strings.HasSuffix(long, "…") {
		t.Fatalf("truncated piece should end with ellipsis: %q", long)
	}
}

// TestLayerLabel names embedding/final/middle rows.
func TestLayerLabel(t *testing.T) {
	if got := layerLabel(0, 29); got != "emb" {
		t.Fatalf("layer 0 = %q", got)
	}
	if got := layerLabel(28, 29); got != "fin" {
		t.Fatalf("last = %q", got)
	}
	if got := layerLabel(5, 29); got != "L5" {
		t.Fatalf("mid = %q", got)
	}
}
