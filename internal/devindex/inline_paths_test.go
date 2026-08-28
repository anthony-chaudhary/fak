package devindex

import (
	"reflect"
	"testing"
)

func TestMarkdownInlinePathsIgnoreBareFilenames(t *testing.T) {
	line := "keep `keep_allowlist.txt`, review `internal/orphanscan/keep_allowlist.txt`, and read [missing](docs/missing.md)"
	got := markdownInlinePaths(line)
	want := []string{"internal/orphanscan/keep_allowlist.txt", "docs/missing.md"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("markdownInlinePaths() = %#v, want %#v", got, want)
	}
}

func TestMarkdownInlinePathsKeepQualifiedDocsPath(t *testing.T) {
	got := markdownInlinePaths("See `docs/guide.md` and `README.md`.")
	want := []string{"docs/guide.md"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("markdownInlinePaths() = %#v, want %#v", got, want)
	}
}
