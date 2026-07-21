package radixkv

import "testing"

// regimeA is a complete decode regime: every axis the issue enumerates is set,
// so DecodeRegime.Complete() holds.
func regimeA() DecodeRegime {
	return DecodeRegime{
		ModelID:    "model-A",
		Dtype:      "f16",
		QuantMode:  "int8",
		RoPEScheme: "rope:theta=1e4,scale=linear",
	}
}

// TestDecodeRegimeSameRegimeReuses witnesses that an identical regime keys to the
// same reuse key and admits reuse with AxisNone.
func TestDecodeRegimeSameRegimeReuses(t *testing.T) {
	a := regimeA()
	b := regimeA()

	if a.ReuseKey() != b.ReuseKey() {
		t.Fatalf("identical regimes must share a reuse key: %q vs %q", a.ReuseKey(), b.ReuseKey())
	}
	if ok, axis := a.Match(b); !ok || axis != AxisNone {
		t.Fatalf("identical regimes must match: ok=%v axis=%q", ok, axis)
	}
	if !a.Reusable(b) {
		t.Fatal("identical regimes must be reusable")
	}
}

// TestDecodeRegimeEveryAxisIsLoadBearing witnesses that flipping ANY single axis
// alone changes the reuse key AND refuses reuse with that axis's typed reason.
func TestDecodeRegimeEveryAxisIsLoadBearing(t *testing.T) {
	base := regimeA()

	cases := []struct {
		name   string
		mutate func(DecodeRegime) DecodeRegime
		want   MismatchAxis
	}{
		{"model", func(r DecodeRegime) DecodeRegime { r.ModelID = "model-B"; return r }, AxisModel},
		{"dtype", func(r DecodeRegime) DecodeRegime { r.Dtype = "bf16"; return r }, AxisDtype},
		{"quant", func(r DecodeRegime) DecodeRegime { r.QuantMode = "nf4"; return r }, AxisQuant},
		{"rope", func(r DecodeRegime) DecodeRegime { r.RoPEScheme = "rope:theta=1e6"; return r }, AxisRoPE},
	}
	for _, tc := range cases {
		other := tc.mutate(regimeA())
		if base.ReuseKey() == other.ReuseKey() {
			t.Fatalf("%s axis: a divergent regime must derive a different key, both were %q", tc.name, base.ReuseKey())
		}
		if ok, axis := base.Match(other); ok || axis != tc.want {
			t.Fatalf("%s axis: a divergent regime must refuse with %q, got ok=%v axis=%q", tc.name, tc.want, ok, axis)
		}
		if base.Reusable(other) {
			t.Fatalf("%s axis: a divergent regime must not be reusable", tc.name)
		}
		// The refusal is symmetric — reuse is refused in both directions.
		if other.Reusable(base) {
			t.Fatalf("%s axis: refusal must be symmetric", tc.name)
		}
	}
}

// TestDecodeRegimeKeyIsStableAndOrderIndependent witnesses that the reuse key is
// a pure function of the axis VALUES, not the order the struct was constructed
// in, and is byte-stable across repeated derivations (wall-clock-free).
func TestDecodeRegimeKeyIsStableAndOrderIndependent(t *testing.T) {
	// Construct the same regime with its fields written in different literal
	// orders — the derived key must be identical.
	byOne := DecodeRegime{
		ModelID:    "m",
		Dtype:      "f32",
		QuantMode:  "none",
		RoPEScheme: "linear",
	}
	byAnother := DecodeRegime{
		RoPEScheme: "linear",
		QuantMode:  "none",
		Dtype:      "f32",
		ModelID:    "m",
	}
	if byOne.ReuseKey() != byAnother.ReuseKey() {
		t.Fatalf("field construction order must not change the key: %q vs %q", byOne.ReuseKey(), byAnother.ReuseKey())
	}
	// Repeated derivation is byte-stable.
	first := byOne.ReuseKey()
	for i := 0; i < 100; i++ {
		if got := byOne.ReuseKey(); got != first {
			t.Fatalf("reuse key must be stable across derivations: %q != %q", got, first)
		}
	}
}

// TestDecodeRegimeZeroFailsClosed witnesses that a zero / partly-unset regime
// can neither key a prefix nor request one: reuse is refused with AxisIncomplete
// even against an identical incomplete regime.
func TestDecodeRegimeZeroFailsClosed(t *testing.T) {
	var zero DecodeRegime
	if zero.Complete() {
		t.Fatal("a zero regime must not be complete")
	}
	// A complete regime refuses a zero request.
	if ok, axis := regimeA().Match(zero); ok || axis != AxisIncomplete {
		t.Fatalf("a zero request must fail closed: ok=%v axis=%q", ok, axis)
	}
	// A zero tree refuses a complete request.
	if ok, axis := zero.Match(regimeA()); ok || axis != AxisIncomplete {
		t.Fatalf("a zero regime must fail closed against a complete request: ok=%v axis=%q", ok, axis)
	}
	// Two identical incomplete regimes still fail closed — equality is not a
	// proof of a valid regime.
	partial := DecodeRegime{ModelID: "only-model"}
	if partial.Reusable(partial) {
		t.Fatal("an incomplete regime must fail closed even against itself")
	}
	if ok, axis := partial.Match(partial); ok || axis != AxisIncomplete {
		t.Fatalf("an incomplete self-match must fail closed: ok=%v axis=%q", ok, axis)
	}
}
