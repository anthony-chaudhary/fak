// Package sortkeys compares records by source location and a stable tie-breaker.
package sortkeys

// FileLine reports whether a sorts before b by file, line, then key.
func FileLine(aFile string, aLine int, aKey string, bFile string, bLine int, bKey string) bool {
	if aFile != bFile {
		return aFile < bFile
	}
	if aLine != bLine {
		return aLine < bLine
	}
	return aKey < bKey
}
