package codesearch

import (
	"bytes"
	"strings"
	"testing"
)

// run drives the engine and returns (exitCode, stdout). These tests exercise the
// composed engine against the REAL fak repo tree (root ../.. from this package),
// so a green run is a dogfood proof that all four wired primitives work on real
// source, not just fixtures.
func run(t *testing.T, args ...string) (int, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := Run(args, &out, &errb)
	return code, out.String()
}

func TestGrepRealTree(t *testing.T) {
	// Regex search over a real package must find the file that defines the symbol.
	code, out := run(t, "grep", "--root", "../trigram", "func Test")
	if code != 0 {
		t.Fatalf("grep exit %d, out=%q", code, out)
	}
	if !strings.Contains(out, "trigram_test.go") {
		t.Errorf("grep 'func Test' over internal/trigram missed trigram_test.go; got:\n%s", out)
	}
}

func TestLitRealTree(t *testing.T) {
	code, out := run(t, "lit", "--root", ".", "func Run")
	if code != 0 {
		t.Fatalf("lit exit %d", code)
	}
	if !strings.Contains(out, "codesearch.go") {
		t.Errorf("lit 'func Run' over . missed codesearch.go; got:\n%s", out)
	}
}

func TestAstRealTree(t *testing.T) {
	// A shape query that really occurs in this package's own source.
	code, out := run(t, "ast", "--root", ".", "fmt.Fprintln($_, $_)")
	if code != 0 {
		t.Fatalf("ast exit %d", code)
	}
	if !strings.Contains(out, "codesearch.go:") {
		t.Errorf("ast query found no real Fprintln sites; got:\n%s", out)
	}
}

func TestCallsAndCallersRealTree(t *testing.T) {
	// Reaches: the exported Reaches method calls BFS in the same package.
	code, out := run(t, "calls", "--root", "../codegraph", "Reaches")
	if code != 0 {
		t.Fatalf("calls exit %d", code)
	}
	if !strings.Contains(out, "BFS") {
		t.Errorf("calls Reaches did not reach BFS; got:\n%s", out)
	}
	// Callers: BFS calls the neighbor helper, so BFS is a dependent of neighbor.
	code, out = run(t, "callers", "--root", "../codegraph", "neighbor")
	if code != 0 {
		t.Fatalf("callers exit %d", code)
	}
	if !strings.Contains(out, "BFS") {
		t.Errorf("callers neighbor did not surface BFS; got:\n%s", out)
	}
}

func TestFeatureRealCorpus(t *testing.T) {
	// Feature retrieval over the real repo card corpus, via the RRF fusion arm.
	code, out := run(t, "feature", "--root", "../..", "rotate logs")
	if code != 0 {
		t.Fatalf("feature exit %d, out=%q", code, out)
	}
	if strings.TrimSpace(out) == "" || strings.Contains(out, "no matching") {
		t.Errorf("feature 'rotate logs' returned nothing from the real corpus; got:\n%s", out)
	}
}

func TestUsageAndErrors(t *testing.T) {
	if code, _ := run(t); code != 2 {
		t.Errorf("no args exit = %d, want 2", code)
	}
	if code, _ := run(t, "bogus"); code != 2 {
		t.Errorf("bogus sub exit = %d, want 2", code)
	}
	if code, _ := run(t, "grep", "--root", "../trigram", "[unterminated"); code != 2 {
		t.Errorf("bad regexp exit = %d, want 2", code)
	}
}
