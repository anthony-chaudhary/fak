package metalgemm

import "fmt"

// TreeVerifyMaxNodes bounds a verify tree's rows: INK2T_MAX_NODES.
const TreeVerifyMaxNodes = 32

// TreeVerifyCheck reports whether parent is a verify tree: row 0 the root
// (-1) and every other row's parent an earlier row.
func TreeVerifyCheck(parent []int32) error {
	if len(parent) < 1 || len(parent) > TreeVerifyMaxNodes || parent[0] != -1 {
		return fmt.Errorf("%w: tree of %d rows", ErrInk2Shape, len(parent))
	}
	for t := 1; t < len(parent); t++ {
		if parent[t] < 0 || int(parent[t]) >= t {
			return fmt.Errorf("%w: tree row %d has parent %d", ErrInk2Shape, t, parent[t])
		}
	}
	return nil
}

// TreeVerifyPathCheck reports whether path is a kept root path of a tree of
// T rows: it starts at the root and its rows increase.
func TreeVerifyPathCheck(path []int, T int) error {
	if len(path) < 1 || len(path) > min(T, TreeVerifyMaxNodes) || path[0] != 0 {
		return fmt.Errorf("%w: path of %d rows in a tree of %d", ErrInk2Shape, len(path), T)
	}
	for i := 1; i < len(path); i++ {
		if path[i] <= path[i-1] || path[i] >= T {
			return fmt.Errorf("%w: path row %d is %d after %d in a tree of %d", ErrInk2Shape, i, path[i], path[i-1], T)
		}
	}
	return nil
}

// TreeConvCommitCols returns the log columns TreeVerifyConvCommit reads for
// a kept root path: the last past entries of the entry history (columns
// 0..past-1) followed by the path rows' inputs (column past+row). For a chain
// path 0..keep-1 they are keep..keep+past-1, VerifyConvCommit's.
func TreeConvCommitCols(path []int, past int) []int {
	seq := make([]int, 0, past+len(path))
	for j := range past {
		seq = append(seq, j)
	}
	for _, r := range path {
		seq = append(seq, past+r)
	}
	return seq[len(seq)-past:]
}
