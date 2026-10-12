package compute

import (
	"math"
	"testing"
)

// Keep the bit witness available without a device so its negative control also
// runs in ordinary tests. Numeric equality must not hide a signed-zero change.
func v41MHCRequireBits(t interface {
	Helper()
	Fatalf(string, ...any)
}, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %g (%#08x) want %g (%#08x)", name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

type v41MHCBitsFailure struct{}

// Fatalf must stop the assertion just as testing.T.Fatalf does. A unique panic
// lets the negative control observe that failure without swallowing other bugs.
type v41MHCBitsReporter struct{}

func (v41MHCBitsReporter) Helper()               {}
func (v41MHCBitsReporter) Fatalf(string, ...any) { panic(v41MHCBitsFailure{}) }

// fak-test:runtime fast est=1ms lane=default
func TestV41MHCRequireBitsSignedZero(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name      string
		got, want uint32
		fail      bool
	}{
		{"positive unchanged", 0, 0, false},
		{"negative unchanged", 0x80000000, 0x80000000, false},
		{"negative lost", 0, 0x80000000, true},
		{"negative introduced", 0x80000000, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			failed := false
			func() {
				defer func() {
					if r := recover(); r != nil {
						if _, ok := r.(v41MHCBitsFailure); !ok {
							panic(r)
						}
						failed = true
					}
				}()
				v41MHCRequireBits(v41MHCBitsReporter{}, tc.name,
					[]float32{math.Float32frombits(tc.got)},
					[]float32{math.Float32frombits(tc.want)})
			}()
			if failed != tc.fail {
				t.Fatalf("got failure=%t, want %t for bits %#08x versus %#08x", failed, tc.fail, tc.got, tc.want)
			}
		})
	}
}
