package model

// v41GroupedPanelChunks coalesces already-admitted 8/4/2/1 tiles behind fewer
// status/read fences. Only the bounded 16/32/64 extensions need new admission.
// Admission is per size: accepting 64 does not imply support for a 32-row tail.
func v41GroupedPanelChunks(rows int, supports func(int) bool) []int {
	var out []int
	var checked, admitted [7]bool
	for rows > 0 {
		n := 64
		for power := 6; ; power-- {
			if n <= rows {
				if n <= 8 {
					break
				}
				if !checked[power] {
					checked[power], admitted[power] = true, supports(n)
				}
				if admitted[power] {
					break
				}
			}
			n /= 2
		}
		out = append(out, n)
		rows -= n
	}
	return out
}
