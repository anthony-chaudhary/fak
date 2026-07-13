package mathx

import "testing"

func TestRound3(t *testing.T) {
	cases := []struct {
		in   float64
		want float64
	}{
		{1.23456, 1.235},
		{1.23443, 1.234},
		{-1.2345, -1.235}, // round-half-away-from-zero: math.Round(-1234.5) = -1235
		{2.0, 2.0},
		{0.0005, 0.001},
	}
	for _, c := range cases {
		if got := Round3(c.in); got != c.want {
			t.Errorf("Round3(%v) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestArgmaxF32(t *testing.T) {
	cases := []struct {
		name string
		in   []float32
		want int
	}{
		{"empty returns zero", nil, 0},
		{"single", []float32{5}, 0},
		{"last is max", []float32{1, 2, 3}, 2},
		{"first is max", []float32{9, 2, 3}, 0},
		{"middle is max", []float32{1, 8, 3}, 1},
		{"first max wins on tie", []float32{4, 4, 1}, 0},
		{"negatives", []float32{-3, -1, -2}, 1},
		{"all equal", []float32{2, 2, 2}, 0},
	}
	for _, c := range cases {
		if got := ArgmaxF32(c.in); got != c.want {
			t.Errorf("%s: ArgmaxF32(%v) = %d, want %d", c.name, c.in, got, c.want)
		}
	}
}
