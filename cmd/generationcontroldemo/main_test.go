package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/cmd/internal/democapture"
)

func TestCapturedSelfcheckTrace(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if err := run(w); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	b, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, want := range []string{
		`"event":"steering_point"`, `"kind":"redirect"`,
		`"number":2`, `"worker":"worker-gpu-7"`,
		`"accepted":"I will inspect the workspace. Inventory complete; no destructive action ran."`,
		"SELF_CHECK_PASS continuous_generation_redirect_handoff",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("trace missing %q:\n%s", want, got)
		}
	}
	if err := democapture.MatchMarkdown("EXAMPLE-OUTPUT.md", []byte(got)); err != nil {
		t.Fatal(err)
	}
}
