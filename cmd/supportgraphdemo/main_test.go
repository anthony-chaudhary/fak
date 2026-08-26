package main

import (
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
)

func TestSelfcheck(t *testing.T) {
	c := exec.Command("go", "run", "./cmd/supportgraphdemo", "-selfcheck")
	c.Dir = "../.."
	o, e := c.CombinedOutput()
	if e != nil {
		t.Fatalf("%v %s", e, o)
	}
	for _, w := range []string{"L4 exact tuple: supported", "T4 exact tuple: unsupported", "old tuple: stale", "unknown layout: unknown", "SELFCHECK PASS"} {
		if !strings.Contains(string(o), w) {
			t.Fatalf("missing %s: %s", w, o)
		}
	}
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", o); err != nil {
		t.Fatal(err)
	}
}
