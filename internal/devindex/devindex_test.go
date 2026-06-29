package devindex

import (
	"os"
	"path/filepath"
	"testing"
)

// writeSyntheticRepo lays down a tiny repo (dos.toml + INDEX.md) under a temp dir so
// the parser is tested against known, controlled bytes rather than the live tree.
func writeSyntheticRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := `[lanes]
concurrent = ["a", "b"]

[lanes.trees]
gateway = ["internal/gateway/**"]
session = ["internal/session/**"] # per-session DRIVE state + cost ring
cmd     = ["cmd/**"]
version = ["VERSION"]
# new-leaf:tree -- generated leaf trees inserted above this line

[other]
ignored = ["internal/ignored/**"]
`
	indexMd := "# INDEX\n" +
		"- [README](README.md) — what fak is, in one read.\n" +
		"- [`fleet` console](tools/FLEET.md) — watch the agent fleet on a host.\n" +
		"- [Gateway](docs/gateway.md) — the OpenAI-compatible front door.\n" +
		"this is prose, not a link, and must be skipped\n" +
		"- [Issue tracker](https://github.com/x/y/issues) — the live open-issue count.\n"
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "INDEX.md"), []byte(indexMd), 0o644); err != nil {
		t.Fatal(err)
	}
	// Make internal/gateway real so Exists is exercised in both polarities.
	if err := os.MkdirAll(filepath.Join(root, "internal", "gateway"), 0o755); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestParseLanes(t *testing.T) {
	c, err := Load(writeSyntheticRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	// Only [lanes.trees] entries — never the [other] section's "ignored".
	if _, ok := c.LeafByName("ignored"); ok {
		t.Error("leaf from a non-[lanes.trees] section leaked into the catalog")
	}
	wantNames := []string{"cmd", "gateway", "session", "version"}
	if len(c.Leaves) != len(wantNames) {
		t.Fatalf("got %d leaves %v, want %d %v", len(c.Leaves), names(c.Leaves), len(wantNames), wantNames)
	}
	for i, n := range wantNames {
		if c.Leaves[i].Name != n { // Load sorts by name
			t.Errorf("leaf[%d] = %q, want %q (sorted)", i, c.Leaves[i].Name, n)
		}
	}

	sess, ok := c.LeafByName("session")
	if !ok {
		t.Fatal("session leaf missing")
	}
	if sess.Desc != "per-session DRIVE state + cost ring" {
		t.Errorf("session desc = %q, want the inline dos.toml comment", sess.Desc)
	}
	if sess.Dir != "internal/session" {
		t.Errorf("session dir = %q, want internal/session", sess.Dir)
	}
	if sess.Exists {
		t.Error("session dir should NOT exist in the synthetic repo")
	}

	gw, _ := c.LeafByName("gateway")
	if !gw.Exists {
		t.Error("gateway dir was created and should report Exists=true")
	}
}

func TestLaneForPath(t *testing.T) {
	c, err := Load(writeSyntheticRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	cases := []struct{ path, want string }{
		{"internal/gateway/foo.go", "gateway"},       // subtree prefix
		{"internal\\gateway\\foo.go", "gateway"},     // windows separators
		{"./internal/session/x.go", "session"},       // leading ./ trimmed
		{"cmd/fak/index.go", "cmd"},                  // cmd/** tree
		{"VERSION", "version"},                       // exact-file entry
		{"internal/unknownleaf/x.go", "unknownleaf"}, // dir convention fallback
		{"docs/notes/x.md", "docs"},                  // top-level lane dir
		{"README.md", ""},                            // root file -> no lane
	}
	for _, tc := range cases {
		if got := c.LaneForPath(tc.path); got != tc.want {
			t.Errorf("LaneForPath(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

func TestSuggestStamp(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	if got := c.SuggestStamp("internal/session/x.go"); got != "(fak session)" {
		t.Errorf("SuggestStamp = %q, want (fak session)", got)
	}
	if got := c.SuggestStamp("README.md"); got != "" {
		t.Errorf("SuggestStamp(root file) = %q, want empty", got)
	}
}

func TestParseDocs(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	if len(c.Docs) != 4 {
		t.Fatalf("got %d docs, want 4: %+v", len(c.Docs), c.Docs)
	}
	var fleet *Doc
	for i := range c.Docs {
		if c.Docs[i].Path == "tools/FLEET.md" {
			fleet = &c.Docs[i]
		}
	}
	if fleet == nil {
		t.Fatal("FLEET.md doc missing")
	}
	if fleet.Title != "fleet console" { // surrounding backticks stripped
		t.Errorf("title = %q, want backtick-stripped 'fleet console'", fleet.Title)
	}
	if fleet.Blurb != "watch the agent fleet on a host." {
		t.Errorf("blurb = %q", fleet.Blurb)
	}
}

func TestSearchDocsRanking(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	hits := c.SearchDocs("gateway front door")
	if len(hits) == 0 {
		t.Fatal("expected a gateway doc hit")
	}
	if hits[0].Path != "docs/gateway.md" {
		t.Errorf("top hit = %q, want docs/gateway.md (title+blurb match)", hits[0].Path)
	}
	if got := c.SearchDocs(""); got != nil {
		t.Errorf("empty query should return nil, got %v", got)
	}
}

func TestSearchLeaves(t *testing.T) {
	c, _ := Load(writeSyntheticRepo(t))
	// Description token match.
	hits := c.SearchLeaves("DRIVE")
	if len(hits) != 1 || hits[0].Name != "session" {
		t.Errorf("SearchLeaves(DRIVE) = %v, want [session]", names(hits))
	}
	// Name match ranks above all; empty query returns the full set in order.
	if got := c.SearchLeaves(""); len(got) != len(c.Leaves) {
		t.Errorf("empty query returned %d leaves, want all %d", len(got), len(c.Leaves))
	}
	// A token nothing matches yields no hits.
	if got := c.SearchLeaves("nonexistenttoken"); len(got) != 0 {
		t.Errorf("expected no hits, got %v", names(got))
	}
}

func TestLoadMissingDosToml(t *testing.T) {
	if _, err := Load(t.TempDir()); err == nil {
		t.Error("Load with no dos.toml should error (no taxonomy to serve)")
	}
}

// TestRealRepoDogfood loads the live repo via FindRoot and asserts the catalog
// reflects reality — including this very leaf — so the index can't silently lose
// touch with the tree it indexes.
func TestRealRepoDogfood(t *testing.T) {
	root := FindRoot(".")
	c, err := Load(root)
	if err != nil {
		t.Skipf("no repo root found from test cwd (%v); skipping live dogfood", err)
	}
	for _, want := range []string{"devindex", "gateway", "cmd", "docs"} {
		l, ok := c.LeafByName(want)
		if !ok {
			t.Errorf("live catalog is missing the %q lane", want)
			continue
		}
		if want == "devindex" && !l.Exists {
			t.Errorf("devindex leaf should exist on disk (it is this package)")
		}
	}
	if c.LaneForPath("internal/gateway/gateway.go") != "gateway" {
		t.Error("live LaneForPath disagrees with the gateway tree")
	}
}

func names(ls []Leaf) []string {
	out := make([]string, len(ls))
	for i, l := range ls {
		out[i] = l.Name
	}
	return out
}
