package hostload

import (
	"math"
	"runtime"
	"testing"
)

func checkTriple(t *testing.T, name string, l1, l5, l15 float64, ok bool) {
	t.Helper()
	for _, v := range []float64{l1, l5, l15} {
		if ok {
			if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
				t.Fatalf("%s: ok=true but value %v not finite and >= 0", name, v)
			}
		} else if v != 0 {
			t.Fatalf("%s: ok=false but value %v != 0", name, v)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParse(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name        string
		in          string
		l1, l5, l15 float64
		ok          bool
	}{
		{"linux", "0.52 0.58 0.59 1/389 12345\n", 0.52, 0.58, 0.59, true},
		{"darwin-stripped", " 1.20 3.40 5.60 ", 1.20, 3.40, 5.60, true},
		{"extra-fields", "1 2 3 4 5 6", 1, 2, 3, true},
		{"empty", "", 0, 0, 0, false},
		{"two-fields", "1.0 2.0", 0, 0, 0, false},
		{"bad-first", "x 1 2", 0, 0, 0, false},
		{"bad-second", "1 y 2", 0, 0, 0, false},
		{"bad-third", "1 2 z", 0, 0, 0, false},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			l1, l5, l15, ok := parse(tc.in)
			if ok != tc.ok || l1 != tc.l1 || l5 != tc.l5 || l15 != tc.l15 {
				t.Fatalf("parse(%q) = (%v,%v,%v,%v), want (%v,%v,%v,%v)",
					tc.in, l1, l5, l15, ok, tc.l1, tc.l5, tc.l15, tc.ok)
			}
		})
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestParseDarwinBraces(t *testing.T) {
	t.Parallel()
	in := "{ 1.2 3.4 5.6 }\n"
	l1, l5, l15, ok := parseDarwin(in)
	if !ok || l1 != 1.2 || l5 != 3.4 || l15 != 5.6 {
		t.Fatalf("parseDarwin(%q) = (%v,%v,%v,%v), want (1.2,3.4,5.6,true)", in, l1, l5, l15, ok)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRead(t *testing.T) {
	t.Parallel()
	l1, l5, l15, ok := Read()
	checkTriple(t, "Read", l1, l5, l15, ok)
	switch runtime.GOOS {
	case "darwin":
		if _, _, _, dok := ReadDarwin(); dok != ok {
			t.Fatalf("Read ok=%v, ReadDarwin ok=%v", ok, dok)
		}
	case "linux":
		if _, _, _, lok := ReadLinux(); lok != ok {
			t.Fatalf("Read ok=%v, ReadLinux ok=%v", ok, lok)
		}
	case "windows":
		if !ok {
			t.Fatal("Read ok=false on windows, want a load sample")
		}
	default:
		if ok {
			t.Fatalf("Read ok=true on unsupported GOOS %s", runtime.GOOS)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestReadDarwin(t *testing.T) {
	t.Parallel()
	l1, l5, l15, ok := ReadDarwin()
	checkTriple(t, "ReadDarwin", l1, l5, l15, ok)
}

// fak-test:runtime fast est=20ms lane=default
func TestReadLinux(t *testing.T) {
	t.Parallel()
	l1, l5, l15, ok := ReadLinux()
	checkTriple(t, "ReadLinux", l1, l5, l15, ok)
}
