// Package optsdefault carries the workspace/time defaults shared by the
// scorecard Options types: an empty root means the current workspace and a
// zero Now means the wall clock in UTC, so a score is deterministic whenever
// a test pins either field. Each scorer's Options.normalize applies this
// helper first, then layers its own package-specific defaults.
package optsdefault

import "time"

// RootNow applies the two shared defaults to root and now and returns the
// result. Non-empty roots and non-zero times pass through unchanged.
func RootNow(root string, now time.Time) (string, time.Time) {
	if root == "" {
		root = "."
	}
	if now.IsZero() {
		now = time.Now().UTC()
	}
	return root, now
}
