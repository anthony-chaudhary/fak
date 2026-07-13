package cvregress

import (
	"reflect"
	"testing"
)

func TestPairedIntervalKnownDeltaAndReplay(t *testing.T) {
	b := []float64{1, 2, 3, 4, 5}
	c := []float64{1.5, 2.5, 3.5, 4.5, 5.5}
	a, e := PairedConfidenceInterval(b, c, .95, 2000, 7)
	if e != nil {
		t.Fatal(e)
	}
	z, _ := PairedConfidenceInterval(b, c, .95, 2000, 7)
	if !reflect.DeepEqual(a, z) {
		t.Fatalf("not replayable: %+v %+v", a, z)
	}
	if a.Mean != .5 || a.Lower != .5 || a.Upper != .5 || !a.Conclusive {
		t.Fatalf("unexpected known interval: %+v", a)
	}
}
func TestPairedIntervalMissingOrUnpairedNeverPasses(t *testing.T) {
	for _, tc := range []struct{ b, c []float64 }{{nil, nil}, {[]float64{1}, []float64{1}}, {[]float64{1, 2}, []float64{1}}} {
		if r, e := PairedConfidenceInterval(tc.b, tc.c, .95, 1000, 1); e == nil || r.Conclusive {
			t.Fatalf("invalid run conclusive: %+v %v", r, e)
		}
	}
}
