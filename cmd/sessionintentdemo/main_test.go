package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
)

func TestSelfcheckCapturesDistinctEffortSemantics(t *testing.T) {
	var out, err bytes.Buffer
	if code := run(&out, &err, []string{"-selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, err.String())
	}
	for _, want := range []string{`"kind": "minimum"`, `"duration": "2h0m0s"`, `"kind": "target"`, `"duration": "4h0m0s"`, `"kind": "maximum"`, `"duration": "10h0m0s"`, "DECISIONS: 1h_active=continue 2h_active=eligible 10h_elapsed=timeout", "SELFCHECK PASS"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in %s", want, out.String())
		}
	}
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", out.Bytes()); err != nil {
		t.Fatal(err)
	}
}
