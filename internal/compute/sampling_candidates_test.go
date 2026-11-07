package compute

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestSamplingCandidateCertificateRetainsOriginalIDOrder(t *testing.T) {
	p := SamplingCandidatePacket{Vocab: 1000, Count: 4, Valid: true}
	p.IDs[0], p.Raw[0] = 700, 2
	p.IDs[1], p.Raw[1] = 400, 8
	p.IDs[2], p.Raw[2] = 7, 5
	p.IDs[3], p.Raw[3] = 2, 6
	if !p.Certify(.8, 3) || p.Count != 3 {
		t.Fatal("strict complete packet refused")
	}
	for i, want := range []int{2, 7, 400} {
		if p.IDs[i] != want {
			t.Fatalf("ID order %v, want ascending original IDs", p.IDs[:p.Count])
		}
	}
	for _, tc := range []struct {
		name   string
		temp   float64
		valid  bool
		lo, hi float32
	}{
		{"raw tie", 1, true, 1, 1},
		{"weight collapse", 1e300, true, 1, 2},
		{"invalid whole row", 1, false, 1, 2},
		{"positive infinity", 1, true, 1, float32(math.Inf(1))},
	} {
		p = SamplingCandidatePacket{Vocab: 9, Count: 2, Valid: tc.valid}
		p.IDs[0], p.Raw[0], p.IDs[1], p.Raw[1] = 3, tc.hi, 5, tc.lo
		if p.Certify(tc.temp, 1) {
			t.Fatalf("%s accepted", tc.name)
		}
	}
}
