package mathx

import "math"

// ClampScore rounds half-to-even and clamps the result to the inclusive 0..100 range.
func ClampScore(score float64) int {
	rounded := math.RoundToEven(score)
	if rounded < 0 {
		return 0
	}
	if rounded > 100 {
		return 100
	}
	return int(rounded)
}
