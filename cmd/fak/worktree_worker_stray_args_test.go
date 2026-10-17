package main

import (
	"io"
	"testing"
)

func TestWorktreeWorkerLandRefusesStrayPathArgs(t *testing.T) {
	res, code := runWorktreeWorkerLand(io.Discard, io.Discard, []string{
		"--worktree", t.TempDir(), "--paths", "a.go", "b.go", "c.go",
	})
	if code != 2 || res.OK || res.Code != "LAND_STRAY_ARGS" {
		t.Fatalf("got ok=%v code=%q exit=%d, want LAND_STRAY_ARGS exit 2", res.OK, res.Code, code)
	}
}
