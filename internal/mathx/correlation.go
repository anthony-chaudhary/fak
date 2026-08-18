package mathx

import "math"

// Pearson returns the Pearson correlation coefficient for equal non-empty
// samples. It returns zero when the samples are invalid or either is constant.
func Pearson(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}
	var sx, sy float64
	for i := range x {
		sx += x[i]
		sy += y[i]
	}
	mx, my := sx/float64(len(x)), sy/float64(len(y))
	var numerator, dx, dy float64
	for i := range x {
		a, b := x[i]-mx, y[i]-my
		numerator += a * b
		dx += a * a
		dy += b * b
	}
	if dx == 0 || dy == 0 {
		return 0
	}
	return numerator / math.Sqrt(dx*dy)
}
