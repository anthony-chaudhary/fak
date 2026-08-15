package main

import (
	"os/exec"
	"strings"
	"testing"
)

func TestSelfcheck(t *testing.T) {
	cmd := exec.Command("go", "run", "./cmd/stackpreflightdemo", "-selfcheck")
	cmd.Dir = "../.."
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	for _, want := range []string{"MINIMUM PATH: allow", "recommended, not required", "T4 PATH: refuse", "ALTERNATIVE 1", "SELFCHECK PASS"} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("missing %q\n%s", want, out)
		}
	}
}
