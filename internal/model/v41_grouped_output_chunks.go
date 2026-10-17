package model

// v41GroupedPanelChunks coalesces already-admitted 8/4/2/1 tiles behind fewer
// status/read fences. Extensions above 8, bounded by 64, need new admission.
// Admission is per size: accepting 64 does not imply support for a 32-row tail.
func v41GroupedPanelChunks(rows int, supports func(int) bool) []int {
	var out []int
	var checked, admitted [7]bool
	for rows > 0 {
		// Probe the exact bounded tail before decomposing it. Non-power sizes
		// are visited at most once as rows decreases; refusal leaves the
		// independently admitted power-of-two path unchanged.
		if rows > 8 && rows < 64 && rows&(rows-1) != 0 && supports(rows) {
			out = append(out, rows)
			break
		}
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
