package memoryindex

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fixtureStore writes a memory directory on disk. files maps basename -> body.
func fixtureStore(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func memo(stem, typ, desc, body string) string {
	return "---\nname: " + stem + "\ndescription: " + desc + "\nmetadata:\n  type: " + typ +
		"\n  recorded: 2026-08-08\n---\n\n" + body + "\n"
}

func TestParseFrontmatter(t *testing.T) {
	cases := []struct {
		name string
		body string
		want Frontmatter
	}{
		{
			name: "the full grammar",
			body: memo("alpha", "project", "what alpha is", "body"),
			want: Frontmatter{Present: true, Terminated: true, Name: "alpha", Description: "what alpha is", Type: "project"},
		},
		{
			name: "no fence at all",
			body: "# alpha\n\njust prose\n",
			want: Frontmatter{},
		},
		{
			name: "an unterminated block",
			body: "---\nname: alpha\ndescription: d\n",
			want: Frontmatter{Present: true, Name: "alpha", Description: "d"},
		},
		{
			// A description holding a colon must survive verbatim to the row the
			// writer renders; a naive split on ":" would truncate it.
			name: "a value containing a colon survives",
			body: "---\nname: alpha\ndescription: rule: never do the thing\nmetadata:\n  type: project\n---\n",
			want: Frontmatter{Present: true, Terminated: true, Name: "alpha", Description: "rule: never do the thing", Type: "project"},
		},
		{
			name: "quotes are stripped, inner content is not",
			body: "---\nname: \"alpha\"\ndescription: 'a \"quoted\" hook'\nmetadata:\n  type: project\n---\n",
			want: Frontmatter{Present: true, Terminated: true, Name: "alpha", Description: `a "quoted" hook`, Type: "project"},
		},
		{
			// type: is only metadata.type when it is nested under metadata:. A
			// top-level `type:` is a different key and must not satisfy the check.
			name: "a top-level type is not metadata.type",
			body: "---\nname: alpha\ndescription: d\ntype: project\n---\n",
			want: Frontmatter{Present: true, Terminated: true, Name: "alpha", Description: "d"},
		},
		{
			name: "CRLF line endings parse identically",
			body: "---\r\nname: alpha\r\ndescription: d\r\nmetadata:\r\n  type: project\r\n---\r\n",
			want: Frontmatter{Present: true, Terminated: true, Name: "alpha", Description: "d", Type: "project"},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ParseFrontmatter(tc.body); got != tc.want {
				t.Errorf("ParseFrontmatter = %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestParseRows(t *testing.T) {
	text := strings.Join([]string{
		"# Memory index",
		"",
		"- [Alpha](alpha.md) — the alpha fact",
		"- [Beta](beta.md#the-fact) — a row carrying a fragment",
		"- [Spec](https://example.test/spec.md) — a web link, not a local claim",
		"- [Mail](mailto:someone@example.test) — not a link target at all",
		"see also [archive](MEMORY_archive.md)",
		"prose that merely mentions (gamma.md) without linking it",
	}, "\n")

	got := ParseRows(IndexName, text)
	want := []Row{
		{Tier: IndexName, Line: 3, Title: "Alpha", Target: "alpha.md"},
		{Tier: IndexName, Line: 4, Title: "Beta", Target: "beta.md"},
		{Tier: IndexName, Line: 7, Title: "archive", Target: "MEMORY_archive.md"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseRows =\n%+v\nwant\n%+v", got, want)
	}
}

func TestParseWikilinks(t *testing.T) {
	got := ParseWikilinks("see [[alpha]] and [[beta|the other]] plus [[gamma#part]]; not [single] or [[bad\n]]")
	want := []string{"alpha", "beta|the other", "gamma#part"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ParseWikilinks = %v, want %v", got, want)
	}
}

// Slugify's rule is that EVERY non-alphanumeric rune becomes '-'. A
// separators-only transform points at a directory that does not exist, the
// existence guard passes, and the whole checker silently reports clean forever.
func TestSlugifyReplacesEveryNonAlphanumeric(t *testing.T) {
	for in, want := range map[string]string{
		`C:\work\fak`:                    "C--work-fak",
		`/home/test.user/fak`:            "-home-test-user-fak",
		`\\wsl.localhost\Ubuntu-24.04\h`: "--wsl-localhost-Ubuntu-24-04-h",
	} {
		if got := Slugify(in); got != want {
			t.Errorf("Slugify(%q) = %q, want %q", in, got, want)
		}
	}
	if got := AutoMemoryDir("/home/u", "/a b"); got != filepath.Join("/home/u", ".claude", "projects", "-a-b", "memory") {
		t.Errorf("AutoMemoryDir = %q", got)
	}
}

// The reconciliation core consumes values only: no fixture directory or filesystem I/O.
func TestReconcilePureCore(t *testing.T) {
	rep := Reconcile(Store{
		Dir:     "fixture",
		Tiers:   []string{IndexName},
		Present: []string{"alpha.md", "orphan.md", "wrong.md", "second.md"},
		Rows: []Row{
			{Tier: IndexName, Line: 1, Target: "alpha.md"},
			{Tier: IndexName, Line: 2, Target: "gone.md"},
		},
		Files: []File{
			{Name: "alpha.md", Front: Frontmatter{Name: "alpha", Description: "ok", Type: "project", Present: true, Terminated: true}},
			{Name: "orphan.md", Front: Frontmatter{Name: "orphan", Description: "ok", Type: "project", Present: true, Terminated: true}},
			{Name: "wrong.md", Front: Frontmatter{Name: "shared", Description: "ok", Type: "project", Present: true, Terminated: true}},
			{Name: "second.md", Front: Frontmatter{Name: "shared", Description: "ok", Type: "project", Present: true, Terminated: true}},
		},
	}, Options{})
	for kind, want := range map[string]int{
		KindMissingFromIndex: 3,
		KindIndexLineNoFile:  1,
		KindSlugMismatch:     2,
		KindDuplicateSlug:    1,
	} {
		if got := rep.Count(kind); got != want {
			t.Errorf("%s = %d, want %d", kind, got, want)
		}
	}
}

// "Nothing to check" is a real, distinct answer — not clean, not drifted.
func TestLoadNothingToCheck(t *testing.T) {
	if _, ok := Load(filepath.Join(t.TempDir(), "no-such-dir")); ok {
		t.Error("a missing directory must report nothing-to-check")
	}
	// A spill tier with no head is not an index: nothing points a cold session at it.
	dir := fixtureStore(t, map[string]string{
		"MEMORY_archive.md": "- [Alpha](alpha.md)\n",
		"alpha.md":          memo("alpha", "project", "d", "b"),
	})
	if _, ok := Load(dir); ok {
		t.Error("a store with no MEMORY.md must report nothing-to-check")
	}
	rep, ok := Check(dir, Options{})
	if ok || rep.Drifted() {
		t.Errorf("Check over a headless store: ok=%v drifted=%v, want false/false", ok, rep.Drifted())
	}
	if len(rep.Counts) != len(Kinds()) {
		t.Errorf("even a nothing-to-check report carries the whole vocabulary; got %d keys", len(rep.Counts))
	}
}

// The end-to-end read: a fixture directory carrying one of every defect.
func TestCheckOverAFixtureDirectory(t *testing.T) {
	dir := fixtureStore(t, map[string]string{
		IndexName: strings.Join([]string{
			"# Memory index",
			"",
			"- [Alpha](alpha.md) — indexed and well-formed",
			"- [Ghost](ghost.md) — a row whose file was deleted by hand",
			"- [archive](MEMORY_archive.md) — the spill tier",
			"",
		}, "\n"),
		"MEMORY_archive.md": "- [Beta](beta.md) — indexed only in the spill tier\n",
		"alpha.md":          memo("alpha", "project", "the alpha fact", "links to [[beta]] and [[nowhere]]"),
		"beta.md":           memo("beta", "project", "the beta fact", "body"),
		// unindexed: a memory nothing points at.
		"orphan.md": memo("orphan", "project", "written, never indexed", "body"),
		// mismatched slug + out-of-vocabulary type, on one file.
		"skew.md": memo("not-skew", "runbook", "a name that disagrees", "body"),
		// no frontmatter at all.
		"raw.md": "# raw\n\njust prose\n",
		// a non-memory file: present, never checked, and a valid link target.
		"README.md":    "not a memory\n",
		"_orphans.txt": "scratch\n",
	})

	rep, ok := Check(dir, Options{})
	if !ok {
		t.Fatal("Check must find the store")
	}
	if rep.Files != 6 {
		t.Errorf("Files = %d, want 6 memories (README.md and _orphans.txt are not memories)", rep.Files)
	}
	if !reflect.DeepEqual(rep.Tiers, []string{IndexName, "MEMORY_archive.md"}) {
		t.Errorf("Tiers = %v", rep.Tiers)
	}
	want := map[string]int{
		KindMissingFromIndex: 3, // orphan.md, skew.md, raw.md
		KindIndexLineNoFile:  1, // ghost.md
		KindSlugMismatch:     1, // skew.md
		KindDuplicateSlug:    0,
		KindFrontmatter:      1, // raw.md has no block
		KindTypeVocabulary:   1, // skew.md says "runbook"
		KindUnresolvedLink:   1, // alpha.md -> [[nowhere]]
	}
	if !reflect.DeepEqual(rep.Counts, want) {
		t.Fatalf("Counts = %v\nwant %v\nfindings: %+v", rep.Counts, want, rep.Findings)
	}
	if !rep.Drifted() || rep.Gating() != 7 {
		t.Errorf("drifted=%v gating=%d, want true/7 (the unresolved link does not gate)", rep.Drifted(), rep.Gating())
	}
	// beta.md is indexed ONLY in the spill tier and must not be reported.
	for _, f := range rep.Findings {
		if f.Kind == KindMissingFromIndex && f.Subject == "beta.md" {
			t.Error("a memory indexed in the spill tier is reachable and must not be an orphan")
		}
	}
	// The dead row must be located, so a human can go delete it.
	for _, f := range rep.Findings {
		if f.Kind == KindIndexLineNoFile && f.Where != IndexName+":4" {
			t.Errorf("dangling row Where = %q, want %s:4", f.Where, IndexName)
		}
	}
}
