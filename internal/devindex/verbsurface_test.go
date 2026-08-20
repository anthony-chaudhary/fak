package devindex

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestPreStateZeroIsUnverified(t *testing.T) {
	var state PreState
	if got := state.String(); got != "UNVERIFIED" {
		t.Fatalf("zero state=%q", got)
	}
}

func TestSourceSurfacePublishesGapsAndRefusals(t *testing.T) {
	root := repoRootForSurface(t)
	surface, err := ExtractVerbSurface(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(surface.Leaves) < vsVerbFloor {
		t.Fatalf("only %d rows", len(surface.Leaves))
	}
	rendered := string(surface.Markdown())
	if !strings.Contains(rendered, "unverified rows:") || !strings.Contains(rendered, "| REFUSES |") {
		t.Fatal("render omits gap/refusal contract")
	}
	refusalRows := 0
	sourceOnly := 0
	for _, leaf := range surface.Leaves {
		if len(leaf.Pre.Codes) > 0 {
			refusalRows++
		}
		if !leaf.InHelp {
			sourceOnly++
		}
	}
	if refusalRows == 0 {
		t.Fatal("REFUSES extraction found no rows")
	}
	// This is the failure-matched witness from #5934: source discovers paths help omits.
	// If help reaches parity later, update this assertion to name that landing SHA.
	if sourceOnly == 0 {
		t.Fatal("source/help drift set is empty; record the parity SHA before changing this witness")
	}
}

func TestReasonLexiconRejectsOrdinaryWords(t *testing.T) {
	got := refusalCodesInString("ALLOW nope OFF_TRUNK and LOCK_BUSY")
	joined := strings.Join(got, ",")
	if joined != "LOCK_BUSY,OFF_TRUNK" {
		t.Fatalf("codes=%q", joined)
	}
}

func TestFirstSentenceTruncatesAtRuneBoundary(t *testing.T) {
	input := strings.Repeat("a", 199) + "é" + strings.Repeat("b", 10)
	got := vsFirstSentence(input)
	if !utf8.ValidString(got) {
		t.Fatalf("truncated synopsis is not valid UTF-8: %q", got)
	}
	want := strings.Repeat("a", 199) + "…"
	if got != want {
		t.Fatalf("truncated synopsis = %q, want %q", got, want)
	}
}

func TestVerbSurfaceMarkdownHasNoTrailingWhitespace(t *testing.T) {
	rendered := string((&VerbSurface{Files: 1}).Markdown())
	wantCounts := "parsed files: 1<br>\nrows: 0<br>\nunverified rows: 0 / 0<br>\nsource-only rows absent from help: 0\n"
	if !strings.Contains(rendered, wantCounts) {
		t.Fatalf("count block lost its rendered line breaks:\n%s", rendered)
	}
	for lineNumber, line := range strings.Split(rendered, "\n") {
		if strings.HasSuffix(line, " ") {
			t.Fatalf("line %d has trailing whitespace: %q", lineNumber+1, line)
		}
	}
}
