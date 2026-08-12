package main

import "testing"

func TestKind(t *testing.T) {
	cases := map[string]string{
		"run.out.log": "trajectory-output", "run.err.log": "trajectory-error",
		"run.in.txt": "prompt-input", "run.worktree": "worktree-marker", "run.pid": "pid-marker",
	}
	for in, want := range cases {
		if got := kind(in); got != want {
			t.Errorf("kind(%q)=%q want %q", in, got, want)
		}
	}
}
