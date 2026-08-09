package memoryindex

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func readIndex(t *testing.T, dir, tier string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, tier))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The round trip: a drifted store, one Apply, and the census findings are gone.
func TestApplyReconcilesTheIndex(t *testing.T) {
	dir := fixtureStore(t, map[string]string{
		IndexName: strings.Join([]string{
			"# Memory index",
			"",
			"- [Alpha](alpha.md) — the alpha fact",
			"- [Ghost](ghost.md) — a row whose file was deleted by hand",
			"",
		}, "\n"),
		"alpha.md":  memo("alpha", "project", "the alpha fact", "body"),
		"orphan.md": memo("orphan", "project", "written but never indexed", "body"),
	})

	ch, after, err := Apply(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ch.Added, []string{"orphan.md"}) {
		t.Errorf("Added = %v, want [orphan.md]", ch.Added)
	}
	if !reflect.DeepEqual(ch.Removed, []string{"ghost.md"}) {
		t.Errorf("Removed = %v, want [ghost.md]", ch.Removed)
	}
	if !reflect.DeepEqual(ch.Tiers, []string{IndexName}) {
		t.Errorf("Tiers = %v", ch.Tiers)
	}

	text := readIndex(t, dir, IndexName)
	if strings.Contains(text, "ghost.md") {
		t.Errorf("the dead row must be gone:\n%s", text)
	}
	if !strings.Contains(text, "- [Alpha](alpha.md) — the alpha fact") {
		t.Errorf("an untouched row must survive byte-for-byte:\n%s", text)
	}
	if !strings.Contains(text, UnfiledHeading) {
		t.Errorf("recovered rows go under a self-identifying heading:\n%s", text)
	}
	// The recovered row carries the memory's OWN description, so the fix is a
	// re-file rather than a re-read.
	if !strings.Contains(text, "- [orphan](orphan.md) — written but never indexed") {
		t.Errorf("recovered row is wrong:\n%s", text)
	}

	if after.Drifted() {
		t.Errorf("the post-write report must be clean; findings: %+v", after.Findings)
	}
	if after.Counts[KindMissingFromIndex] != 0 || after.Counts[KindIndexLineNoFile] != 0 {
		t.Errorf("census findings survived the write: %v", after.Counts)
	}
}

// The boundary that keeps -write from being a laundering machine: it reconciles
// the INDEX toward the FILES and never the reverse, so a source-side defect
// survives the write and the report still says so.
func TestApplyNeverEditsAMemoryFileAndCannotLaunderTheReport(t *testing.T) {
	dir := fixtureStore(t, map[string]string{
		IndexName: "# Memory index\n\n",
		"skew.md": memo("not-skew", "runbook", "a name that disagrees", "and a [[nowhere]] link"),
		"raw.md":  "# raw\n\nno frontmatter at all\n",
	})
	beforeSkew := readIndex(t, dir, "skew.md")
	beforeRaw := readIndex(t, dir, "raw.md")

	ch, after, err := Apply(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ch.Added, []string{"raw.md", "skew.md"}) {
		t.Errorf("Added = %v, want both memories indexed", ch.Added)
	}
	if got := readIndex(t, dir, "skew.md"); got != beforeSkew {
		t.Errorf("Apply must never rewrite a memory file:\n%s", got)
	}
	if got := readIndex(t, dir, "raw.md"); got != beforeRaw {
		t.Errorf("Apply must never rewrite a memory file:\n%s", got)
	}

	if !after.Drifted() {
		t.Fatal("the source-side drift must survive -write")
	}
	surviving := map[string]int{}
	for _, f := range after.Findings {
		surviving[f.Kind]++
	}
	want := map[string]int{
		KindSlugMismatch:   1,
		KindTypeVocabulary: 1,
		KindFrontmatter:    1,
		KindUnresolvedLink: 1,
	}
	if !reflect.DeepEqual(surviving, want) {
		t.Errorf("surviving findings = %v, want %v", surviving, want)
	}
	if after.Fixable() != 0 {
		t.Errorf("nothing left is fixable, yet Fixable() = %d", after.Fixable())
	}
	// A memory with no description says so IN the row rather than rendering a
	// pointer with nothing behind it.
	if !strings.Contains(readIndex(t, dir, IndexName), "(no description: in this file's frontmatter — write one)") {
		t.Errorf("a description-less memory must be flagged in its own row:\n%s", readIndex(t, dir, IndexName))
	}
}

// Idempotent: a second Apply over a reconciled store writes nothing, and does not
// stack a second Unfiled heading.
func TestApplyIsIdempotent(t *testing.T) {
	dir := fixtureStore(t, map[string]string{
		IndexName:   "# Memory index\n\n- [Ghost](ghost.md) — gone\n",
		"orphan.md": memo("orphan", "project", "d", "body"),
	})
	if _, _, err := Apply(dir, Options{}); err != nil {
		t.Fatal(err)
	}
	once := readIndex(t, dir, IndexName)

	ch, after, err := Apply(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if ch.Any() {
		t.Errorf("a second Apply must be a no-op; changes = %+v", ch)
	}
	if after.Drifted() {
		t.Errorf("still drifted: %+v", after.Findings)
	}
	if got := readIndex(t, dir, IndexName); got != once {
		t.Errorf("a no-op Apply rewrote the file:\n%q\n%q", once, got)
	}
	if n := strings.Count(once, UnfiledHeading); n != 1 {
		t.Errorf("Unfiled heading appears %d times, want 1", n)
	}

	// A THIRD memory appearing later reuses the same section rather than opening
	// a second one.
	if err := os.WriteFile(filepath.Join(dir, "later.md"), []byte(memo("later", "project", "d2", "b")), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Apply(dir, Options{}); err != nil {
		t.Fatal(err)
	}
	twice := readIndex(t, dir, IndexName)
	if n := strings.Count(twice, UnfiledHeading); n != 1 {
		t.Errorf("Unfiled heading appears %d times after a second recovery, want 1:\n%s", n, twice)
	}
	if !strings.Contains(twice, "(later.md)") {
		t.Errorf("the later memory was not indexed:\n%s", twice)
	}
}

// Dead rows are dropped from the tier they actually live in, and the surrounding
// lines keep their numbering.
func TestApplyDropsDeadRowsFromEveryTier(t *testing.T) {
	dir := fixtureStore(t, map[string]string{
		IndexName:           "# Memory index\n\n- [Alpha](alpha.md) — a\n- [Dead1](dead1.md) — x\n- [archive](MEMORY_archive.md) — spill\n",
		"MEMORY_archive.md": "# Archive\n\n- [Beta](beta.md) — b\n- [Dead2](dead2.md) — y\n",
		"alpha.md":          memo("alpha", "project", "a", "body"),
		"beta.md":           memo("beta", "project", "b", "body"),
	})
	ch, after, err := Apply(dir, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(ch.Removed, []string{"dead1.md", "dead2.md"}) {
		t.Errorf("Removed = %v", ch.Removed)
	}
	if !reflect.DeepEqual(ch.Tiers, []string{IndexName, "MEMORY_archive.md"}) {
		t.Errorf("Tiers = %v", ch.Tiers)
	}
	if got := readIndex(t, dir, IndexName); got != "# Memory index\n\n- [Alpha](alpha.md) — a\n- [archive](MEMORY_archive.md) — spill\n" {
		t.Errorf("primary tier =\n%q", got)
	}
	if got := readIndex(t, dir, "MEMORY_archive.md"); got != "# Archive\n\n- [Beta](beta.md) — b\n" {
		t.Errorf("spill tier =\n%q", got)
	}
	if after.Drifted() {
		t.Errorf("still drifted: %+v", after.Findings)
	}
}

// A written row must be indistinguishable in style from a hand-written one, or
// every -write shows up as a formatting diff on top of a content diff.
func TestRenderRowMatchesTheIndexStyle(t *testing.T) {
	fm := Frontmatter{Present: true, Terminated: true, Name: "alpha", Description: "the hook", Type: "project"}
	if got := RenderRow("alpha-two.md", fm, " — "); got != "- [alpha two](alpha-two.md) — the hook" {
		t.Errorf("RenderRow = %q", got)
	}
	if got := separatorOf("- [A](a.md) - one\n- [B](b.md) - two\n- [C](c.md) — three\n"); got != " - " {
		t.Errorf("separatorOf = %q, want the hyphen this index prefers", got)
	}
	if got := separatorOf("# empty index\n"); got != " — " {
		t.Errorf("separatorOf(empty) = %q, want the em dash default", got)
	}
	if got := Humanize("fak-oauth_token-family-2026-08-06"); got != "fak oauth token family 2026 08 06" {
		t.Errorf("Humanize = %q", got)
	}
}

func TestApplyRefusesAStoreWithNoIndex(t *testing.T) {
	dir := t.TempDir()
	ch, _, err := Apply(dir, Options{})
	if err == nil {
		t.Fatal("Apply must refuse rather than invent an index")
	}
	if ch.Any() {
		t.Errorf("nothing may be written: %+v", ch)
	}
	if _, statErr := os.Stat(filepath.Join(dir, IndexName)); statErr == nil {
		t.Error("Apply created an index where there was none")
	}
}
