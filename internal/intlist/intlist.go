// Package intlist parses a string into a list of non-negative integers.
//
// It exists to retire the byte-identical parseInts helper that was copy-pasted
// across the benchmark command mains (batchbench, longctxbench, fleetserve,
// sessionbench) — the worst-first duplication clone the code-slop scorecard
// named under #776.
package intlist

// Parse extracts every maximal run of decimal digits in s as a non-negative
// int, treating any non-digit byte as a separator. So "1,2,4,8", "1 2 4 8",
// and "[1,2,4,8]" all yield the same list; an empty or digit-free string yields
// nil. This is the exact behaviour of the bench harnesses' former parseInts.
func Parse(s string) []int {
	var out []int
	cur, has := 0, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= '0' && c <= '9' {
			cur = cur*10 + int(c-'0')
			has = true
		} else if has {
			out = append(out, cur)
			cur, has = 0, false
		}
	}
	if has {
		out = append(out, cur)
	}
	return out
}
