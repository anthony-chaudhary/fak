package numfmt

import "testing"

func TestUpToOneDecimal(t *testing.T) {
	for value, want := range map[float64]string{2: "2", 2.25: "2.2", -1.5: "-1.5"} {
		if got := UpToOneDecimal(value); got != want {
			t.Errorf("UpToOneDecimal(%v) = %q, want %q", value, got, want)
		}
	}
}

func TestOneDecimal(t *testing.T) {
	if got := OneDecimal(2); got != "2.0" {
		t.Fatalf("OneDecimal(2) = %q", got)
	}
}
