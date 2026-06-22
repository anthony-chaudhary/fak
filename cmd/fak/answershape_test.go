package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runAS is a test helper: drive runAnswerShape with a stdin string and captured
// streams, returning (exit, stdout, stderr).
func runAS(t *testing.T, stdin string, args ...string) (int, string, string) {
	t.Helper()
	var out, errb bytes.Buffer
	code := runAnswerShape(strings.NewReader(stdin), &out, &errb, args)
	return code, out.String(), errb.String()
}

func TestAnswerShapeCleanExitsZero(t *testing.T) {
	clean := "The gateway adjudicates each tool call in-process before it ever runs."
	code, out, _ := runAS(t, clean)
	if code != 0 {
		t.Fatalf("clean text: exit=%d, want 0\n%s", code, out)
	}
	if !strings.Contains(out, "answer-shape: OK") {
		t.Fatalf("clean text output missing OK verdict:\n%s", out)
	}
}

func TestAnswerShapeLoopExitsOne(t *testing.T) {
	code, out, _ := runAS(t, strings.Repeat("yes ", 40))
	if code != 1 {
		t.Fatalf("looping text: exit=%d, want 1\n%s", code, out)
	}
	if !strings.Contains(out, "DEGENERATE") {
		t.Fatalf("looping text output missing DEGENERATE verdict:\n%s", out)
	}
}

func TestAnswerShapeMaxCharsTrips(t *testing.T) {
	code, out, _ := runAS(t, "a perfectly coherent sentence that is over twenty characters long", "--max-chars", "20")
	if code != 1 {
		t.Fatalf("over-length text: exit=%d, want 1\n%s", code, out)
	}
	if !strings.Contains(out, "verbose") {
		t.Fatalf("expected a verbose reason:\n%s", out)
	}
}

func TestAnswerShapeJSON(t *testing.T) {
	code, out, _ := runAS(t, strings.Repeat("loop ", 40), "--json")
	if code != 1 {
		t.Fatalf("exit=%d, want 1", code)
	}
	var rep struct {
		Degenerate     bool    `json:"degenerate"`
		RepeatFraction float64 `json:"repeat_fraction"`
	}
	if err := json.Unmarshal([]byte(out), &rep); err != nil {
		t.Fatalf("output is not valid JSON: %v\n%s", err, out)
	}
	if !rep.Degenerate || rep.RepeatFraction <= 0.5 {
		t.Fatalf("JSON report not degenerate as expected: %+v", rep)
	}
}

func TestAnswerShapeFile(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "answer.txt")
	if err := os.WriteFile(p, []byte(strings.Repeat("abc", 80)), 0o644); err != nil {
		t.Fatal(err)
	}
	code, out, _ := runAS(t, "", "--file", p)
	if code != 1 {
		t.Fatalf("--file degenerate content: exit=%d, want 1\n%s", code, out)
	}
}

func TestAnswerShapeTextFlagLiteral(t *testing.T) {
	// An explicit --text literal is used verbatim (not stdin).
	code, _, _ := runAS(t, "IGNORED STDIN", "--text", "this is a fine short answer with enough length")
	if code != 0 {
		t.Fatalf("literal --text clean: exit=%d, want 0", code)
	}
}

func TestAnswerShapeBadFlagExitsTwo(t *testing.T) {
	code, _, errb := runAS(t, "x", "--nope")
	if code != 2 {
		t.Fatalf("bad flag: exit=%d, want 2", code)
	}
	if errb == "" {
		t.Fatalf("expected an error message on stderr for a bad flag")
	}
}
