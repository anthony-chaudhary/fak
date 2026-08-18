// Package numfmt renders compact human-facing decimal values.
package numfmt

import (
	"math"
	"strconv"
)

// UpToOneDecimal renders integral values without a fraction and other values with one decimal.
func UpToOneDecimal(value float64) string {
	if value == math.Trunc(value) {
		return strconv.FormatFloat(value, 'f', 0, 64)
	}
	return strconv.FormatFloat(value, 'f', 1, 64)
}

// OneDecimal renders a value with exactly one digit after the decimal point.
func OneDecimal(value float64) string { return strconv.FormatFloat(value, 'f', 1, 64) }
