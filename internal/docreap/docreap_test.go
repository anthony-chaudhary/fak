package docreap

import (
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func gitT(t *testing.T, root string, args ...string) {
	t.Helper()
	out, err := exec.Command("git", append([]string{"-C", root}, args...)...).CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func seed(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	gitT(t, dir, "init", "-q")
	gitT(t, dir, "config", "user.email", "t@example.invalid")
	gitT(t, dir, "config", "user.name", "t")
	for rel, body := range map[string]string{
		"README.md":              "See [old](docs/old.md) and `docs/old.md`, not `xdocs/old.md` or `docs/old.md-x`.\n",
		"docs/old.md":            "---\ntitle: old\n---\n\n# Old\n\nBack to [readme](../README.md) or [new](new.md#top).\n",
		"docs/new.md":            "# New\n\nReplaces [old](old.md).\n",
		"docs/sub/deep.md":       "Up: [old](../old.md).\n",
		"docs/_witnesses/w.json": "{\"path\": \"docs/old.md\"}\n",
		"docs/archive/README.md": "# Archive\n",
	} {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	gitT(t, dir, "add", "-A")
	gitT(t, dir, "commit", "-qm", "seed")
	root, err := RepoRoot(dir)
	if err != nil {
		t.Fatal(err)
	}
	return root
}

func read(t *testing.T, root, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func wantContains(t *testing.T, got, want string) {
	t.Helper()
	if !strings.Contains(got, want) {
		t.Errorf("missing %q in:\n%s", want, got)
	}
}

func checkClean(t *testing.T, root string) {
	t.Helper()
	bad, err := Check(root)
	if err != nil || len(bad) != 0 {
		t.Fatalf("check: bad=%v err=%v", bad, err)
	}
}

func day(s string) time.Time {
	d, _ := time.Parse(dateLayout, s)
	return d
}

func TestArchiveMovesBannersAndRepoints(t *testing.T) {
	root := seed(t)
	err := Archive(root, ArchiveOptions{
		Doc: "docs/old.md", Reason: "superseded", Successor: "docs/new.md",
		Scope: "the old design", Today: day("2026-11-23"),
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if exists(filepath.Join(root, "docs", "old.md")) {
		t.Error("docs/old.md still exists")
	}
	moved := read(t, root, "docs/archive/old.md")
	wantContains(t, moved, "> **Archived 2026-11-23** · reason: superseded · replacement: [docs/new.md](../new.md) · scope: the old design")
	if !strings.HasPrefix(moved, "---\ntitle: old\n---\n") {
		t.Errorf("front matter must stay first:\n%s", moved)
	}
	wantContains(t, moved, "[readme](../../README.md)")
	wantContains(t, moved, "[new](../new.md#top)")
	readme := read(t, root, "README.md")
	wantContains(t, readme, "[old](docs/archive/old.md)")
	wantContains(t, readme, "`docs/archive/old.md`,")
	wantContains(t, readme, "`xdocs/old.md`")
	wantContains(t, readme, "`docs/old.md-x`")
	wantContains(t, read(t, root, "docs/new.md"), "[old](archive/old.md)")
	wantContains(t, read(t, root, "docs/sub/deep.md"), "[old](../archive/old.md)")
	wantContains(t, read(t, root, "docs/_witnesses/w.json"), `"docs/old.md"`)
	checkClean(t, root)
}

func TestLinksBetweenArchivedDocsSurviveASecondArchive(t *testing.T) {
	root := seed(t)
	for _, o := range []ArchiveOptions{
		{Doc: "docs/old.md", Reason: "stale-snapshot"},
		{Doc: "docs/new.md", Reason: "retired"},
	} {
		o.Successor, o.Scope, o.Today = "none", "history", day("2026-11-23")
		if err := Archive(root, o, io.Discard); err != nil {
			t.Fatal(err)
		}
		gitT(t, root, "add", "-A")
	}
	wantContains(t, read(t, root, "docs/archive/old.md"), "[new](new.md#top)")
	checkClean(t, root)
}

func TestDryRunChangesNothing(t *testing.T) {
	root := seed(t)
	before := read(t, root, "README.md")
	var out strings.Builder
	err := Archive(root, ArchiveOptions{
		Doc: "docs/old.md", Reason: "retired", Successor: "none", Today: day("2026-11-23"), DryRun: true,
	}, &out)
	if err != nil {
		t.Fatal(err)
	}
	wantContains(t, out.String(), "repoint README.md")
	if read(t, root, "README.md") != before || !exists(filepath.Join(root, "docs", "old.md")) {
		t.Error("dry run modified the tree")
	}
}

func TestCheckFlagsArchivedDocWithoutBanner(t *testing.T) {
	root := seed(t)
	if err := os.WriteFile(filepath.Join(root, "docs", "archive", "bare.md"), []byte("# Bare\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitT(t, root, "add", "-A")
	bad, err := Check(root)
	if err != nil || len(bad) != 1 || bad[0] != "docs/archive/bare.md" {
		t.Fatalf("bad=%v err=%v", bad, err)
	}
}

func TestArchiveRefusals(t *testing.T) {
	root := seed(t)
	for name, o := range map[string]ArchiveOptions{
		"missing successor": {Doc: "docs/old.md", Reason: "superseded", Successor: "docs/nope.md"},
		"missing doc":       {Doc: "docs/nope.md", Reason: "superseded", Successor: "none"},
		"unknown reason":    {Doc: "docs/old.md", Reason: "bored", Successor: "none"},
	} {
		if err := Archive(root, o, io.Discard); !errors.Is(err, ErrRefused) {
			t.Errorf("%s: err=%v, want a refusal", name, err)
		}
	}
	if !exists(filepath.Join(root, "docs", "old.md")) {
		t.Error("refused archive moved the doc")
	}
}

func TestRankListsOrphansAndSkipsArchive(t *testing.T) {
	root := seed(t)
	rows, err := Rank(root, false, day("2027-06-01"), nil)
	if err != nil {
		t.Fatal(err)
	}
	byPath := map[string]Row{}
	for _, r := range rows {
		byPath[r.Path] = r
	}
	old, ok := byPath["docs/old.md"]
	if !ok {
		t.Fatalf("docs/old.md not ranked: %+v", rows)
	}
	if old.DocRefs != 3 || old.AgeDays < 90 || old.Action != "archive" {
		t.Errorf("unexpected row: %+v", old)
	}
	if _, ok := byPath["docs/archive/README.md"]; ok {
		t.Error("archive README ranked")
	}
	if _, ok := byPath["docs/sub/deep.md"]; ok {
		t.Error("non-recursive rank included a subdirectory doc")
	}
	scored, err := Rank(root, true, day("2027-06-01"), func(string, string) (float64, bool) { return 40, true })
	if err != nil {
		t.Fatal(err)
	}
	if scored[0].Path != "docs/sub/deep.md" || scored[0].Appeal == nil || *scored[0].Appeal != 40 {
		t.Errorf("orphan with appeal should rank first: %+v", scored[0])
	}
}

func TestInsertBannerKeepsCRLFFrontMatter(t *testing.T) {
	got := insertBanner("---\r\na: b\r\n---\r\nbody\r\n", "> B")
	if got != "---\r\na: b\r\n---\r\n\r\n> B\r\nbody\r\n" {
		t.Errorf("got %q", got)
	}
	if got := insertBanner("# T\n", "> B"); got != "> B\n\n# T\n" {
		t.Errorf("got %q", got)
	}
}
