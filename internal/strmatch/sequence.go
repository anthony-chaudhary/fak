package strmatch

// CommonSlicePrefixLen returns the length of the equal leading run.
func CommonSlicePrefixLen[T comparable](a, b []T) int {
	limit := len(a)
	if len(b) < limit {
		limit = len(b)
	}
	index := 0
	for index < limit && a[index] == b[index] {
		index++
	}
	return index
}
