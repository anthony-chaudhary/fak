package memq

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/recall"
)

// fixtureNotesStore writes a minimal markdown memory store (MEMORY.md index +
// fact files) and returns its dir. Files not passed are not written; entries in
// index without a file exercise the skip path.
func fixtureNotesStore(t *testing.T, index string, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "MEMORY.md"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

const notesFixtureIndex = `# Memory index

- [Fresh workaround](fresh.md) — a still-true note
- [Stale pointer](stale.md) — names a moved artifact
- [Prose preference](prose.md) — no concrete claims
`

var notesFixtureFiles = map[string]string{
	"fresh.md": "---\nname: fresh-workaround\ndescription: run the gate via the helper\nmetadata:\n  type: feedback\n---\n\nUse the helper in internal/memq/exec.go when the gate refuses.\n",
	"stale.md": "---\nname: stale-pointer\ndescription: an old location\nmetadata:\n  type: project\n---\n\nThe fix lives in internal/gonepkg/gone.go since commit deadbeef.\n",
	"prose.md": "---\nname: prose-preference\ndescription: terse answers preferred\nmetadata:\n  type: user\n---\n\nThe user prefers terse answers with the outcome first.\n",
}

// splitVerifier marks any claim mentioning "gone" stale and everything else
// fresh — deterministic, no git dependence (the reverify_test.go pattern).
func splitVerifier(_ context.Context, claims []recall.ArtifactClaim) []recall.ArtifactFinding {
	out := make([]recall.ArtifactFinding, 0, len(claims))
	for _, c := range claims {
		st := recall.ArtifactFresh
		if strings.Contains(c.Value, "gone") || c.Value == "deadbeef" {
			st = recall.ArtifactStale
		}
		out = append(out, recall.ArtifactFinding{Claim: c, Status: st, Detail: "split verifier"})
	}
	return out
}

// #2347: the index IS the curation — one cell per MEMORY.md-linked fact file, in
// index order, carrying safe metadata (kind, durability from metadata.type,
// provenance attr) and never the body.
func TestNotesBackend_cellsFromIndex(t *testing.T) {
	dir := fixtureNotesStore(t, notesFixtureIndex+"- [Ghost](missing.md) — file absent\n", notesFixtureFiles)
	// An unindexed file must stay invisible even though it is in the store dir.
	if err := os.WriteFile(filepath.Join(dir, "unindexed.md"), []byte("not curated"), 0o644); err != nil {
		t.Fatal(err)
	}
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	cells, err := b.Cells(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(cells) != 3 {
		t.Fatalf("cells = %d, want 3 (indexed+present only): %+v", len(cells), cells)
	}
	if cells[0].ID != "fresh.md" || cells[1].ID != "stale.md" || cells[2].ID != "prose.md" {
		t.Fatalf("cells must preserve index order, got %s/%s/%s", cells[0].ID, cells[1].ID, cells[2].ID)
	}
	if cells[0].Kind != KindMemoryNote || cells[0].Attrs["provenance"] != NotesProvenance {
		t.Fatalf("note cell must be stamped kind+provenance, got %+v", cells[0])
	}
	// metadata.type → durability: feedback=durable, project=bounded, user=durable.
	if cells[0].Durability != DurabilityDurable || cells[1].Durability != DurabilityBounded || cells[2].Durability != DurabilityDurable {
		t.Fatalf("durability mapping wrong: %s/%s/%s", cells[0].Durability, cells[1].Durability, cells[2].Durability)
	}
	if !strings.Contains(cells[0].Descriptor, "Fresh workaround") || !strings.Contains(cells[0].Descriptor, "run the gate via the helper") {
		t.Fatalf("descriptor must carry index title + frontmatter description, got %q", cells[0].Descriptor)
	}
}

// #2347 / #2077 done-condition: a note whose concrete claim no longer verifies is
// REFUSED at page-in (stale_recall_artifact) with the claim named; a still-true
// note renders. Run through the loop-recall driver, end to end.
func TestNotesBackend_staleNoteRefusedFreshNoteRendered(t *testing.T) {
	dir := fixtureNotesStore(t, notesFixtureIndex, notesFixtureFiles)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)

	d, ok := Get("loop-recall")
	if !ok {
		t.Fatal("loop-recall driver not registered")
	}
	res, err := Run(context.Background(), b, d.Build(Params{Intent: "gate refuses fix helper"}), Caps{})
	if err != nil {
		t.Fatal(err)
	}
	rendered := map[string]bool{}
	for _, it := range res.Rendered {
		rendered[it.ID] = true
	}
	if !rendered["fresh.md"] {
		t.Fatalf("fresh note must render; rendered=%+v", res.Rendered)
	}
	if rendered["stale.md"] {
		t.Fatal("stale note rendered into context — the #2077 failure this backend exists to stop")
	}
	var staleRefusal bool
	for _, rf := range res.Refused {
		if rf.ID == "stale.md" && rf.Reason == "stale_recall_artifact" {
			staleRefusal = true
		}
	}
	if !staleRefusal {
		t.Fatalf("stale note must be refused as stale_recall_artifact; refused=%+v", res.Refused)
	}
}

// The Verify seam: per-claim findings for tagging a rendered note fresh vs
// hedged — and an empty slice for a prose-only note (nothing checkable).
func TestNotesBackend_verifySeam(t *testing.T) {
	dir := fixtureNotesStore(t, notesFixtureIndex, notesFixtureFiles)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)
	ctx := context.Background()

	fresh, err := b.Verify(ctx, "fresh.md")
	if err != nil || len(fresh) == 0 {
		t.Fatalf("fresh.md must yield findings (it names a repo path); err=%v findings=%+v", err, fresh)
	}
	for _, f := range fresh {
		if f.Status != recall.ArtifactFresh {
			t.Fatalf("fresh.md finding not fresh: %+v", f)
		}
	}
	prose, err := b.Verify(ctx, "prose.md")
	if err != nil {
		t.Fatal(err)
	}
	if len(prose) != 0 {
		t.Fatalf("prose-only note must yield no findings (render hedged), got %+v", prose)
	}
}

// A secret-shaped note is sealed at scan and refused at page-in — memory is not
// a laundering path for credentials (the same screen the Codex backend runs).
func TestNotesBackend_secretShapedNoteSealed(t *testing.T) {
	dir := fixtureNotesStore(t,
		"# Memory index\n\n- [Leaky](leaky.md) — oops\n",
		map[string]string{"leaky.md": "---\nname: leaky\n---\n\nthe deploy key is AKIAIOSFODNN7EXAMPLE\n"})
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	cells, _ := b.Cells(context.Background())
	if len(cells) != 1 || !cells[0].Sealed {
		t.Fatalf("secret-shaped note must be sealed at scan, got %+v", cells)
	}
	if _, err := b.Materialize(context.Background(), "leaky.md"); err == nil {
		t.Fatal("sealed note must refuse page-in")
	}
}

// A missing store is an empty corpus, never an error — a fresh node must not
// crash the loop's recall step.
func TestNotesBackend_missingStoreIsEmpty(t *testing.T) {
	b, err := NewNotesBackend(filepath.Join(t.TempDir(), "nope"))
	if err != nil {
		t.Fatal(err)
	}
	cells, err := b.Cells(context.Background())
	if err != nil || len(cells) != 0 {
		t.Fatalf("missing store: cells=%v err=%v, want empty and nil", cells, err)
	}
}
