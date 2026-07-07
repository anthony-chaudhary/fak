package sessionaudit

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

func stat(xs []float64, includeMean bool, roundTenth bool, includeMax bool) StatSet {
	var st StatSet
	if len(xs) == 0 {
		return st
	}
	sort.Float64s(xs)
	med := median(xs)
	if roundTenth {
		med = round(med, 1)
	}
	st.Median = &med
	if includeMean {
		var sum float64
		for _, x := range xs {
			sum += x
		}
		mean := round(sum/float64(len(xs)), 1)
		st.Mean = &mean
	}
	p90 := pct(xs, 90)
	if roundTenth {
		p90 = round(p90, 1)
	}
	st.P90 = &p90
	if !includeMax && len(xs) > 0 {
		p10 := pct(xs, 10)
		if roundTenth {
			p10 = round(p10, 3)
		}
		st.P10 = &p10
	}
	if includeMax {
		max := xs[len(xs)-1]
		st.Max = &max
	}
	return st
}

func median(xs []float64) float64 {
	if len(xs)%2 == 1 {
		return xs[len(xs)/2]
	}
	return (xs[len(xs)/2-1] + xs[len(xs)/2]) / 2
}

func pct(xs []float64, p float64) float64 {
	k := int(math.Round((p / 100) * float64(len(xs)-1)))
	if k < 0 {
		k = 0
	}
	if k >= len(xs) {
		k = len(xs) - 1
	}
	return xs[k]
}

func round(v float64, places int) float64 {
	p := math.Pow10(places)
	return math.Round(v*p) / p
}

func sortedModelCounts(m map[string]ModelCounts) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]].Output == m[keys[j]].Output {
			return keys[i] < keys[j]
		}
		return m[keys[i]].Output > m[keys[j]].Output
	})
	return keys
}

func sortedCounts(m map[string]int64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if m[keys[i]] == m[keys[j]] {
			return keys[i] < keys[j]
		}
		return m[keys[i]] > m[keys[j]]
	})
	return keys
}

func ratio(n, d int64) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d)
	return &v
}

func floatRatio(n, d float64) *float64 {
	if d == 0 {
		return nil
	}
	v := n / d
	return &v
}

func floatRatioValue(n, d float64) float64 {
	if d == 0 {
		return 0
	}
	return n / d
}

func fmtPct(v *float64) string {
	if v == nil {
		return "-"
	}
	return fmt.Sprintf("%.1f%%", *v*100)
}

func fmtPctPtr(v *float64) string {
	return fmtPct(v)
}

func fmtInt(n int64) string {
	return groupThousands(strconv.FormatInt(n, 10))
}

// groupThousands inserts thousands-separator commas into a decimal digit
// string, scanning from the right. Shared by fmtInt and fmtFloat.
func groupThousands(s string) string {
	var out []byte
	for i, r := range reverse(s) {
		if i > 0 && i%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, byte(r))
	}
	return reverse(string(out))
}

func reverse(s string) string {
	b := []byte(s)
	for i, j := 0, len(b)-1; i < j; i, j = i+1, j-1 {
		b[i], b[j] = b[j], b[i]
	}
	return string(b)
}

func fmtFloat(v float64, places int) string {
	s := fmt.Sprintf("%."+strconv.Itoa(places)+"f", v)
	parts := strings.SplitN(s, ".", 2)
	intPart := parts[0]
	sign := ""
	if strings.HasPrefix(intPart, "-") {
		sign = "-"
		intPart = strings.TrimPrefix(intPart, "-")
	}
	grouped := groupThousands(intPart)
	if len(parts) == 2 {
		return sign + grouped + "." + parts[1]
	}
	return sign + grouped
}

func fmtStat(v *float64) string {
	if v == nil {
		return "null"
	}
	if math.Abs(*v-math.Round(*v)) < 1e-9 {
		return fmt.Sprintf("%.0f", *v)
	}
	return strconv.FormatFloat(*v, 'f', -1, 64)
}

func fmtStatInt(v *float64) string {
	if v == nil {
		return "0"
	}
	return fmtInt(int64(math.Round(*v)))
}

func trimFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
