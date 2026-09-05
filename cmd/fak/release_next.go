package main

import (
	"io"
)

// runReleaseNext dispatches the "next" version draft management helper.
// It tracks in-flight work on trunk targeting the next release in docs/releases/NEXT.md.
func runReleaseNext(stdout, stderr io.Writer, argv []string) int {
	return releaseRunScript(repoRoot(), "release_next.py", argv, stdout, stderr)
}
