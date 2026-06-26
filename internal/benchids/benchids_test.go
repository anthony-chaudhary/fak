package benchids

import (
	"reflect"
	"testing"
)

func TestLCGGolden(t *testing.T) {
	// Golden sequences pin the exact LCG the bench mains relied on, so the
	// extraction is provably behaviour-preserving (not just plausible).
	cases := []struct {
		seed uint64
		want []int
	}{
		{0, []int{227, 776, 265, 118, 119}},
		{7, []int{350, 503, 428, 765, 186}},
	}
	for _, tc := range cases {
		got := LCG(5, 1000, tc.seed)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("LCG(5, 1000, %d) = %v, want %v", tc.seed, got, tc.want)
		}
	}
}

func TestLCGShape(t *testing.T) {
	const n, vocab = 64, 256
	ids := LCG(n, vocab, 1)
	if len(ids) != n {
		t.Fatalf("len = %d, want %d", len(ids), n)
	}
	for i, id := range ids {
		if id < 0 || id >= vocab {
			t.Fatalf("ids[%d] = %d out of range [0,%d)", i, id, vocab)
		}
	}
}

func TestLCGDeterministicAndSeedSensitive(t *testing.T) {
	a := LCG(32, 1000, 991)
	b := LCG(32, 1000, 991)
	if !reflect.DeepEqual(a, b) {
		t.Errorf("same seed produced different sequences:\n %v\n %v", a, b)
	}
	c := LCG(32, 1000, 992)
	if reflect.DeepEqual(a, c) {
		t.Errorf("distinct seeds produced identical sequences: %v", a)
	}
}

func TestLCGEmpty(t *testing.T) {
	if got := LCG(0, 1000, 0); len(got) != 0 {
		t.Errorf("LCG(0,...) = %v, want empty", got)
	}
}
