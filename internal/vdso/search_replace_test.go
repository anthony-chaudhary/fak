package vdso

import "testing"

// search_replace_test.go — the tolerant matcher's contract. The load-bearing
// properties are (1) exact match wins and is never reinterpreted, (2) a tolerant
// match must be unique or the matcher refuses, and (3) the replacement keeps the
// TARGET's absolute indentation while preserving the model's relative shape.

func TestMatchAndReplaceRelativeIndentFourSpaceSourceTwoSpaceOld(t *testing.T) {
	// #10972's stated case: source is 4-space indented, the model's old_string is
	// 2-space indented. The tolerant path must still find and rewrite the block,
	// landing the replacement at the source's 4-space depth.
	content := "func f() {\n    if x {\n        y()\n    }\n}\n"
	oldStr := "  if x {\n      y()\n  }"
	newStr := "  if x {\n      z()\n  }"
	got, ok := MatchAndReplaceRelativeIndent(content, oldStr, newStr)
	if !ok {
		t.Fatal("tolerant match with 2-space old against 4-space source was refused")
	}
	want := "func f() {\n    if x {\n        z()\n    }\n}\n"
	if got != want {
		t.Fatalf("got  %q\nwant %q", got, want)
	}
}

func TestMatchAndReplaceRelativeIndentExactMatchPrecedence(t *testing.T) {
	// oldStr appears byte-exactly; the exact hit must win AND be returned with
	// only a single replacement even though a tolerant reading might be ambiguous.
	content := "a\n    keep\nb\n"
	oldStr := "    keep"
	newStr := "    changed"
	got, ok := MatchAndReplaceRelativeIndent(content, oldStr, newStr)
	if !ok {
		t.Fatal("exact match was refused")
	}
	if got != "a\n    changed\nb\n" {
		t.Fatalf("exact precedence got %q", got)
	}
}

func TestMatchAndReplaceRelativeIndentNonUniqueRefuses(t *testing.T) {
	// Two windows carry the same trimmed signature under drift; guessing which
	// the model meant would be a silent wrong edit. Must refuse.
	content := "    if x {\n        y()\n    }\n    if x {\n        y()\n    }\n"
	oldStr := "  if x {\n      y()\n  }"
	if _, ok := MatchAndReplaceRelativeIndent(content, oldStr, "  if x {\n      z()\n  }"); ok {
		t.Fatal("non-unique tolerant match was accepted")
	}
}

func TestMatchAndReplaceRelativeIndentNoMatchRefuses(t *testing.T) {
	content := "alpha\nbeta\ngamma\n"
	if _, ok := MatchAndReplaceRelativeIndent(content, "  delta\n  epsilon", "x"); ok {
		t.Fatal("absent block was matched")
	}
}

func TestMatchAndReplaceRelativeIndentPreservesIndentation(t *testing.T) {
	// The model's replacement is written flat (0-base) but the target sits two
	// levels deep; the rewrite must land at the target's depth, and the model's
	// own relative indentation between its lines must be preserved.
	content := "class C:\n    def m(self):\n        a = 1\n        b = 2\n"
	oldStr := "def m(self):\n    a = 1\n    b = 2"
	newStr := "def m(self):\n    a = 9\n    b = 8"
	got, ok := MatchAndReplaceRelativeIndent(content, oldStr, newStr)
	if !ok {
		t.Fatal("flat old_string against indented source was refused")
	}
	want := "class C:\n    def m(self):\n        a = 9\n        b = 8\n"
	if got != want {
		t.Fatalf("got  %q\nwant %q", got, want)
	}
}

func TestMatchAndReplaceRelativeIndentTabVsSpaceDrift(t *testing.T) {
	// Source uses tabs; the model emitted spaces. Leading whitespace spelling
	// must not defeat the match. The target's base indentation (a tab) is kept
	// and the model's own relative depth is preserved verbatim as extra spaces.
	content := "start\n\tif x {\n\t\tgo()\n\t}\nend\n"
	oldStr := "  if x {\n    go()\n  }"
	newStr := "  if x {\n    stop()\n  }"
	got, ok := MatchAndReplaceRelativeIndent(content, oldStr, newStr)
	if !ok {
		t.Fatal("tab-vs-space drift was refused")
	}
	want := "start\n\tif x {\n\t  stop()\n\t}\nend\n"
	if got != want {
		t.Fatalf("got  %q\nwant %q", got, want)
	}
}

func TestMatchAndReplaceRelativeIndentTrailingWhitespaceDrift(t *testing.T) {
	content := "x\n    foo   \n    bar\t\n y\n"
	oldStr := "foo\nbar"
	newStr := "baz\nqux"
	got, ok := MatchAndReplaceRelativeIndent(content, oldStr, newStr)
	if !ok {
		t.Fatal("trailing-whitespace drift was refused")
	}
	if got != "x\n    baz\n    qux\n y\n" {
		t.Fatalf("got %q", got)
	}
}

func TestMatchAndReplaceRelativeIndentBlankOldRefuses(t *testing.T) {
	if _, ok := MatchAndReplaceRelativeIndent("a\n\nb\n", "   \n\t", "x"); ok {
		t.Fatal("whitespace-only old_string was matched")
	}
}

func TestMatchAndReplaceRelativeIndentEmptyOldRefuses(t *testing.T) {
	if _, ok := MatchAndReplaceRelativeIndent("abc", "", "x"); ok {
		t.Fatal("empty old_string was matched")
	}
}

func TestNearestLineHintNamesStaleRegion(t *testing.T) {
	content := "package x\n\nfunc Alpha() {\n\treturn 1\n}\n\nfunc Beta() {\n\treturn 2\n}\n"
	// old_string is a stale revision of Beta's body (return 3 vs return 2).
	old := "func Beta() {\n\treturn 3\n}"
	line, score, ok := NearestLineHint(content, old)
	if !ok {
		t.Fatal("expected a hint for a same-shaped stale block")
	}
	if line != 7 {
		t.Errorf("NearestLineHint line = %d, want 7", line)
	}
	if score <= 0 || score > 100 {
		t.Errorf("NearestLineHint score = %d, want (0,100]", score)
	}
}

func TestNearestLineHintNoAnchorOnBlank(t *testing.T) {
	if _, _, ok := NearestLineHint("a\nb\n", "   "); ok {
		t.Error("blank old_string must not yield a hint")
	}
	if _, _, ok := NearestLineHint("", "x"); ok {
		t.Error("empty content must not yield a hint")
	}
}
