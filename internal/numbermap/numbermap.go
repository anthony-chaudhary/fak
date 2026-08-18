// Package numbermap normalizes JSON-decoded numeric maps.
package numbermap

// Ints copies map values into integers using convert for dynamically decoded values.
func Ints(value any, convert func(any) int) map[string]int {
	out := map[string]int{}
	switch values := value.(type) {
	case map[string]int:
		for key, number := range values {
			out[key] = number
		}
	case map[string]any:
		for key, number := range values {
			out[key] = convert(number)
		}
	}
	return out
}
