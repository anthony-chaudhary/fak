package capindex

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestDigestStability proves the digest is a deterministic SHA-256: identical
// bytes always hash the same, different bytes hash differently, and the form is
// the "sha256:<hex>" the index keys on. This is the property a hot-swap compare
// depends on — same body, same digest, no re-read.
func TestDigestStability(t *testing.T) {
	a := Digest([]byte("hello world"))
	b := Digest([]byte("hello world"))
	if a != b {
		t.Fatalf("digest not stable: %q != %q for identical input", a, b)
	}
	// Known SHA-256 of "hello world".
	const want = "sha256:b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
	if a != want {
		t.Fatalf("digest mismatch: got %q want %q", a, want)
	}
	if c := Digest([]byte("hello world!")); c == a {
		t.Fatalf("digest collided on different input: both %q", c)
	}
	if d := Digest(nil); d == a {
		t.Fatalf("empty input digested same as non-empty: %q", d)
	}
}

// TestIndexDiffAddedRemovedChanged is the core hash-diff witness: build an index,
// snapshot it, mutate it (add one, remove one, change one body, leave one
// untouched), then diff the two snapshots and assert exactly the right rows in
// exactly the right categories — and that the untouched card produces NO row.
func TestIndexDiffAddedRemovedChanged(t *testing.T) {
	mk := func(name, body string) CapCard {
		return CapCard{
			Ref:       CapRef{Kind: CapKindSkill, Name: name},
			CardBytes: []byte(body),
		}
	}

	ix := NewIndex()
	ix.Register(mk("keep", "keep-v1"))     // untouched across the diff
	ix.Register(mk("change", "change-v1")) // body will change
	ix.Register(mk("drop", "drop-v1"))     // will be removed
	before := ix.Snapshot()

	// Mutate: change one body, remove one, add a new one. "keep" is untouched.
	ix.Register(mk("change", "change-v2")) // new digest
	if !ix.Remove(CapRef{Kind: CapKindSkill, Name: "drop"}) {
		t.Fatal("Remove(drop) reported not present")
	}
	ix.Register(mk("fresh", "fresh-v1")) // added
	after := ix.Snapshot()

	changes := before.Diff(after)

	got := map[string]ChangeKind{}
	for _, ch := range changes {
		got[ch.Ref.Name] = ch.Kind
	}

	if _, ok := got["keep"]; ok {
		t.Errorf("unchanged card 'keep' produced a diff row (re-index was not a no-op for it)")
	}
	if got["fresh"] != Added {
		t.Errorf("'fresh' = %v, want Added", got["fresh"])
	}
	if got["drop"] != Removed {
		t.Errorf("'drop' = %v, want Removed", got["drop"])
	}
	if got["change"] != Changed {
		t.Errorf("'change' = %v, want Changed", got["change"])
	}
	if len(changes) != 3 {
		t.Errorf("expected exactly 3 changes (added/removed/changed), got %d: %+v", len(changes), changes)
	}

	// The Changed row must carry both the old and new digest, and they must differ.
	for _, ch := range changes {
		if ch.Kind == Changed {
			if ch.OldDigest == "" || ch.NewDigest == "" {
				t.Errorf("Changed row missing a digest: %+v", ch)
			}
			if ch.OldDigest == ch.NewDigest {
				t.Errorf("Changed row has equal old/new digest: %+v", ch)
			}
		}
	}
}

// TestReindexUnchangedIsNoop proves the acceptance criterion directly:
// re-snapshotting an index that did not change yields an empty diff.
func TestReindexUnchangedIsNoop(t *testing.T) {
	ix := NewIndex()
	ix.Register(CapCard{Ref: CapRef{Kind: CapKindSkill, Name: "a"}, CardBytes: []byte("a-body")})
	ix.Register(CapCard{Ref: CapRef{Kind: CapKindSkill, Name: "b"}, CardBytes: []byte("b-body")})

	s1 := ix.Snapshot()
	// Re-register identical cards — same digests.
	ix.Register(CapCard{Ref: CapRef{Kind: CapKindSkill, Name: "a"}, CardBytes: []byte("a-body")})
	ix.Register(CapCard{Ref: CapRef{Kind: CapKindSkill, Name: "b"}, CardBytes: []byte("b-body")})
	s2 := ix.Snapshot()

	if changes := s1.Diff(s2); len(changes) != 0 {
		t.Fatalf("re-indexing an unchanged catalog should be a no-op, got %d changes: %+v", len(changes), changes)
	}
}

// TestLazyResolveFaultsOnce proves the lazy-fault discipline: a Capability's body
// is nil until Materialize is called, the underlying Resolve runs at most once
// even across repeated Materialize calls, and the body is correct.
func TestLazyResolveFaultsOnce(t *testing.T) {
	faults := 0
	cap := Capability{
		Ref: CapRef{Kind: CapKindSkill, Name: "lazy"},
		Resolve: func() []byte {
			faults++
			return []byte("the body")
		},
	}

	if cap.Body != nil {
		t.Fatal("body should be nil before Materialize (no eager fault)")
	}
	if faults != 0 {
		t.Fatalf("Resolve ran before Materialize: %d faults", faults)
	}

	b1 := cap.Materialize()
	if string(b1) != "the body" {
		t.Fatalf("Materialize returned %q, want %q", b1, "the body")
	}
	if faults != 1 {
		t.Fatalf("expected exactly 1 fault after first Materialize, got %d", faults)
	}

	// Second Materialize must be a cache hit, not a re-fault.
	b2 := cap.Materialize()
	if string(b2) != "the body" {
		t.Fatalf("second Materialize returned %q, want %q", b2, "the body")
	}
	if faults != 1 {
		t.Fatalf("Resolve re-ran on second Materialize: %d faults (want 1)", faults)
	}
	if cap.Resolve != nil {
		t.Error("Resolve should be cleared after the body is materialized")
	}
}

// writeSkill creates .claude/skills/<dir>/SKILL.md under root with the given
// frontmatter and body, returning the SKILL.md path.
func writeSkill(t *testing.T, root, dir, name, version, desc, body string) string {
	t.Helper()
	skillDir := filepath.Join(root, dir)
	if err := os.MkdirAll(skillDir, 0o755); err != nil {
		t.Fatal(err)
	}
	content := "---\nname: " + name + "\nversion: " + version +
		"\ndescription: " + desc + "\ntags: [alpha, beta]\n---\n\n" + body + "\n"
	path := filepath.Join(skillDir, "SKILL.md")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestSkillResolverReconciles is the witness test: a SkillResolver over a real
// on-disk skills directory indexes exactly the catalog (count + names), its card
// digests equal a SHA-256 over the raw SKILL.md bytes, and Fault pages the body
// lazily through Resolve. It also checks the Catalog seam: Lookup by ref and
// Query by intent both resolve, body still lazy.
func TestSkillResolverReconciles(t *testing.T) {
	root := t.TempDir()
	p1 := writeSkill(t, root, "first", "first-skill", "1", "does the first thing", "FIRST BODY")
	p2 := writeSkill(t, root, "second", "second-skill", "2", "does the second thing", "SECOND BODY")
	// A directory with no SKILL.md must be ignored.
	if err := os.MkdirAll(filepath.Join(root, "notaskill"), 0o755); err != nil {
		t.Fatal(err)
	}

	r := NewSkillResolver(root)
	cards := r.Index()
	if len(cards) != 2 {
		t.Fatalf("expected 2 skill cards, got %d: %+v", len(cards), cards)
	}

	// Reconcile each card's digest against a fresh SHA-256 of the file on disk.
	byName := map[string]CapCard{}
	for _, c := range cards {
		byName[c.Ref.Name] = c
		if c.Ref.Kind != CapKindSkill {
			t.Errorf("card %s has kind %v, want %v", c.Ref.Name, c.Ref.Kind, CapKindSkill)
		}
		if c.CardBytes == nil {
			t.Errorf("card %s holds no card bytes", c.Ref.Name)
		}
	}
	for name, path := range map[string]string{"first-skill": p1, "second-skill": p2} {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if got := byName[name].Digest; got != Digest(raw) {
			t.Errorf("card %s digest %q != on-disk SHA-256 %q", name, got, Digest(raw))
		}
	}

	// Fault must page lazily: tamper with the file's mtime is overkill; instead
	// fault, assert no body yet, then materialize and assert the real body.
	cap, err := r.Fault(CapRef{Kind: CapKindSkill, Name: "first-skill", Version: "1"})
	if err != nil {
		t.Fatalf("Fault(first-skill): %v", err)
	}
	if cap.Body != nil {
		t.Fatal("Fault paged the body eagerly; it should be lazy")
	}
	if cap.Scope != abi.ScopeAgent {
		t.Errorf("skill scope = %v, want ScopeAgent (private)", cap.Scope)
	}
	body := cap.Materialize()
	if string(body) == "" || !containsLine(body, "FIRST BODY") {
		t.Errorf("materialized body did not contain the skill body, got %q", body)
	}

	// Catalog seam: Lookup by ref and Query by intent.
	cat := NewCatalog()
	cat.AddResolver(CapKindSkill, r)
	changes := cat.Sync()
	if len(changes) != 2 { // both skills are Added on the first sync
		t.Fatalf("first Sync should report 2 Added, got %d: %+v", len(changes), changes)
	}
	if cat.Index().Len() != 2 {
		t.Fatalf("catalog index holds %d cards, want 2", cat.Index().Len())
	}
	// Re-sync with no change is a no-op.
	if again := cat.Sync(); len(again) != 0 {
		t.Fatalf("re-Sync of unchanged catalog should be a no-op, got %+v", again)
	}

	got, err := cat.Lookup(CapRef{Kind: CapKindSkill, Name: "second-skill", Version: "2"})
	if err != nil {
		t.Fatalf("Lookup(second-skill): %v", err)
	}
	if got.Body != nil {
		t.Error("Lookup paged the body eagerly; it should be lazy")
	}

	qd, err := cat.Query("the second thing")
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if qd.Ref.Name != "second-skill" {
		t.Errorf("Query('the second thing') resolved %q, want second-skill", qd.Ref.Name)
	}
}

// containsLine reports whether body contains want as a substring (line-agnostic).
func containsLine(body []byte, want string) bool {
	return len(body) > 0 && len(want) > 0 && indexOf(string(body), want) >= 0
}

func indexOf(haystack, needle string) int {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}
