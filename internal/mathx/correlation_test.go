package mathx

import "testing"

func TestPearson(t *testing.T) {
	for _, tc := range []struct {
		name string
		x, y []float64
		want float64
	}{
		{"positive", []float64{1, 2, 3}, []float64{2, 4, 6}, 1},
		{"negative", []float64{1, 2, 3}, []float64{6, 4, 2}, -1},
		{"constant", []float64{1, 1}, []float64{2, 3}, 0},
		{"mismatch", []float64{1}, nil, 0},
	} {
		if got := Pearson(tc.x, tc.y); got != tc.want {
			t.Errorf("%s: Pearson = %g, want %g", tc.name, got, tc.want)
		}
	}
}
