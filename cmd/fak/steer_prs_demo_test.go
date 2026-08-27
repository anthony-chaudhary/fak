package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSteerPRsDemoSelfcheckAndWorstFirstRender(t *testing.T) {
	path := filepath.Join("..", "..", "internal", "steerpr", "testdata", "lcd-demo.json")
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--demo", path, "--selfcheck"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	residual, unverifiable, cleared := strings.Index(out, "## [RESIDUAL]"), strings.Index(out, "## [UNVERIFIABLE]"), strings.Index(out, "## [CLEARED]")
	if residual < 0 || unverifiable < residual || cleared < unverifiable {
		t.Fatalf("bands not worst-first:\n%s", out)
	}
	for _, want := range []string{"gateway", "model", "docs", "⚠ unstamped", "2 commit(s)"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	if !strings.Contains(stderr.String(), "selfcheck OK") {
		t.Fatalf("stderr=%s", stderr.String())
	}
}

func TestSteerPRsDemoSelfcheckRedsOnPerturbedFixture(t *testing.T) {
	source := filepath.Join("..", "..", "internal", "steerpr", "testdata", "lcd-demo.json")
	b, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	b = bytes.Replace(b, []byte(`"verdict":"CLAIM_UNWITNESSED"`), []byte(`"verdict":"CLAIM_WITNESSED"`), 1)
	path := filepath.Join(t.TempDir(), "perturbed.json")
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runSteerPRs(&stdout, &stderr, []string{"--demo", path, "--selfcheck"}); code != 1 {
		t.Fatalf("code=%d want 1 stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "selfcheck FAILED") {
		t.Fatalf("stderr=%s", stderr.String())
	}
}
