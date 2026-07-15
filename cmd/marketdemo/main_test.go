package main

import (
	"os/exec"
	"strings"
	"testing"
)

func TestSelfcheck(t *testing.T) {
	out, err := exec.Command("go", "run", ".", "-selfcheck").CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s", err, out)
	}
	if !strings.Contains(string(out), "PASS fak-extension-conformance/1") {
		t.Fatalf("unexpected: %s", out)
	}
}
