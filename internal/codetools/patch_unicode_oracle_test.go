package codetools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:justify why=independent-borrow-acceptance when=changed:internal/codetools/patch.go witness=internal/codetools/patch.go#applyHunks
// fak-test:runtime fast est=250ms lane=default
func TestBorrowOracleUnicodePatchContext(t *testing.T) {
	for _, tc := range []struct{ name, original, context string }{
		{"single-quotes", "say ‘yes’", "say 'yes'"},
		{"double-quotes", "say “yes”", `say "yes"`},
		{"equal-leading-spaces", "  say “yes”", `  say "yes"`},
		{"nbsp", "one\u00a0two", "one two"},
		{"en-space", "one\u2002two", "one two"},
		{"em-space", "one\u2003two", "one two"},
		{"thin-space", "one\u2009two", "one two"},
		{"narrow-nbsp", "one\u202ftwo", "one two"},
		{"ideographic-space", "one\u3000two", "one two"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			inserted := "new\t‘exact’\u00a0bytes"
			patch := "@@ -1,2 +1,2 @@\n " + tc.context + "\n-old\n+" + inserted + "\n"
			borrowOracleApply(t, tc.original+"\nold\n", patch, 0, tc.original+"\n"+inserted+"\n", false)
		})
	}
	for _, r := range []rune{'\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212'} {
		t.Run(fmt.Sprintf("dash-%04x", r), func(t *testing.T) {
			original := "left" + string(r) + "right"
			borrowOracleApply(t, original+"\nold\n", "@@ -1,2 +1,2 @@\n left-right\n-old\n+new\n", 0, original+"\nnew\n", false)
		})
	}
}

// fak-test:justify why=independent-borrow-acceptance when=changed:internal/codetools/patch.go witness=internal/codetools/patch.go#applyHunks
// fak-test:runtime fast est=100ms lane=default
func TestBorrowOracleStrongerPatchCandidateWins(t *testing.T) {
	for _, tc := range []struct{ name, original, want string }{
		{
			"later-ascii", "“target”\nseparator\n\"target\"\n",
			"“target”\nseparator\nchosen\n",
		},
		{
			"later-trailing-whitespace", "“target”\nseparator\n\"target\" \t\n",
			"“target”\nseparator\nchosen\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			borrowOracleApply(t, tc.original, "@@ -1 +1 @@\n-\"target\"\n+chosen\n", 2, tc.want, false)
		})
	}
	t.Run("earlier-trailing-whitespace", func(t *testing.T) {
		borrowOracleApply(t, "\"target\" \t\nseparator\n“target”\n", "@@ -3 +3 @@\n-\"target\"\n+chosen\n", 2, "chosen\nseparator\n“target”\n", false)
	})
	t.Run("stronger-outside-window-does-not-win", func(t *testing.T) {
		borrowOracleApply(t, "“target”\none\ntwo\n\"target\"\n", "@@ -1 +1 @@\n-\"target\"\n+chosen\n", 2, "chosen\none\ntwo\n\"target\"\n", false)
	})
}

// fak-test:justify why=independent-borrow-acceptance when=changed:internal/codetools/patch.go witness=internal/codetools/patch.go#applyHunks
// fak-test:runtime fast est=150ms lane=default
func TestBorrowOraclePatchMismatchAndBounds(t *testing.T) {
	for _, tc := range []struct {
		name, original, patch string
		fuzz                  int
	}{
		{"accent", "café\n", "@@ -1 +1 @@\n-cafe\n+wrong\n", 0},
		{"interior-tab", "one\ttwo\n", "@@ -1 +1 @@\n-one two\n+wrong\n", 0},
		{"leading-spaces-ascii", "  target\n", "@@ -1 +1 @@\n-target\n+wrong\n", 0},
		{"leading-spaces-unicode", "  “target”\n", "@@ -1 +1 @@\n-\"target\"\n+wrong\n", 0},
		{"leading-spaces-patch-only", "“target”\n", "@@ -1 +1 @@\n-  \"target\"\n+wrong\n", 0},
		{"unicode-outside-window", "one\ntwo\nthree\n“target”\n", "@@ -1 +1 @@\n-\"target\"\n+wrong\n", 2},
		{"zero-fuzz", "one\n“target”\n", "@@ -1 +1 @@\n-\"target\"\n+wrong\n", 0},
		{"forward-only-hunks", "“target”\npadding\nanchor\n", "@@ -3 +3 @@\n-anchor\n+changed\n@@ -1 +1 @@\n-\"target\"\n+wrong\n", 5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			borrowOracleApply(t, tc.original, tc.patch, tc.fuzz, tc.original, true)
		})
	}
	t.Run("unicode-at-window-edge", func(t *testing.T) {
		borrowOracleApply(t, "one\ntwo\n“target”\n", "@@ -1 +1 @@\n-\"target\"\n+chosen\n", 2, "one\ntwo\nchosen\n", false)
	})
}

func borrowOracleApply(t *testing.T, original, hunk string, fuzz int, want string, wantRefusal bool) {
	t.Helper()
	root := t.TempDir()
	ts, err := New(Config{Root: root})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(root, "example.txt")
	if err := os.WriteFile(path, []byte(original), 0o644); err != nil {
		t.Fatal(err)
	}
	// A map retains an explicit zero fuzz bound in the ordinary JSON API.
	body, err := json.Marshal(map[string]any{"patch": "--- a/example.txt\n+++ b/example.txt\n" + hunk, "fuzz_margin": fuzz})
	if err != nil {
		t.Fatal(err)
	}
	out, refused := ts.ApplyPatch(context.Background(), body)
	if refused != wantRefusal {
		t.Fatalf("refused=%v want=%v response=%s", refused, wantRefusal, out)
	}
	if refused {
		if code := errCode(t, out); code != CodeEditConflict {
			t.Fatalf("refusal code=%q want=%q", code, CodeEditConflict)
		}
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Fatalf("file bytes=%q want=%q", got, want)
	}
}

// fak-test:justify why=independent-borrow-acceptance when=changed:internal/codetools/patch.go witness=internal/codetools/patch.go#applyHunks
// fak-test:runtime fast est=100ms lane=default
func TestBorrowOracleUnicodePrefixSupplement(t *testing.T) {
	for _, tc := range []struct{ name, original, old string }{
		{"nbsp-versus-ascii-prefix", "\u00a0—target", " -target"},
		{"ascii-versus-nbsp-prefix", " —target", "\u00a0-target"},
		{"tab-versus-odd-space-prefix", "\t—target", "\u00a0-target"},
		{"odd-space-versus-tab-prefix", "\u00a0—target", "\t-target"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := tc.original + "\n"
			borrowOracleApply(t, original, "@@ -1 +1 @@\n-"+tc.old+"\n+wrong\n", 0, original, true)
		})
	}
	t.Run("equal-ascii-prefix-odd-space-inside-body", func(t *testing.T) {
		original := " \tleft\u00a0—right\nold\n"
		patch := "@@ -1,2 +1,2 @@\n  \tleft -right\n-old\n+new\n"
		borrowOracleApply(t, original, patch, 0, " \tleft\u00a0—right\nnew\n", false)
	})
}
