package mathx

// MeanBy returns the arithmetic mean of values projected from items.
func MeanBy[T any](items []T, value func(T) float64) float64 {
	if len(items) == 0 {
		return 0
	}
	var sum float64
	for _, item := range items {
		sum += value(item)
	}
	return sum / float64(len(items))
}
