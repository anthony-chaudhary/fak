package memq

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
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

func notesRecallQueryForTest(intent string) Query {
	return Query{
		Intent: intent,
		Ops: []Op{
			{Kind: OpScan},
			{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
				{Op: PredEq, Field: "sealed", Value: "false"},
				{Op: PredEq, Field: "tombstoned", Value: "false"},
			}}},
			{Kind: OpRank, By: RankRelevance, Desc: true},
			{Kind: OpLimit, K: 5},
			{Kind: OpBudget, Bytes: 8192},
			{Kind: OpRender},
		},
	}
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
// note renders. Run through the same query shape the CLI uses, end to end.
func TestNotesBackend_staleNoteRefusedFreshNoteRendered(t *testing.T) {
	dir := fixtureNotesStore(t, notesFixtureIndex, notesFixtureFiles)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	b.WithVerifier(splitVerifier)

	res, err := Run(context.Background(), b, notesRecallQueryForTest("gate refuses fix helper"), Caps{})
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

// #2347 is a backend + CLI surface, not a new canned strategy: `fak memory
// drivers` must remain unchanged by the notes backend.
func TestNotesBackend_doesNotRegisterDriver(t *testing.T) {
	if _, ok := Get("loop-recall"); ok {
		t.Fatal("notes backend must not register loop-recall in the global driver catalog")
	}
	for _, d := range Drivers() {
		if d.Name == "loop-recall" {
			t.Fatal("loop-recall leaked into the memory driver list")
		}
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

// W2 (#2620): a note's forward [[wikilinks]] populate Cell.Refs, resolved to indexed
// cell IDs; an off-index link is flagged on refs_unresolved, never invented as an
// edge; and the graph is queryable — an authored query can filter cells to "references
// <id>" (hasref) and rank by reference count (refcount in-degree).
func TestNotesBackend_referencesFromWikilinks(t *testing.T) {
	index := "# Memory index\n\n" +
		"- [Hub](hub.md) — links the spokes\n" +
		"- [Spoke A](spoke-a.md) — links back to the hub\n" +
		"- [Spoke B](spoke-b.md) — links back to the hub\n"
	files := map[string]string{
		// hub links both spokes (the second via a [[slug|display]] alias) plus one
		// off-index (not-yet-written) note; [[hub]] is a self-link — the ghost link and
		// the self-link must not become edges.
		"hub.md":     "---\nname: hub\ndescription: the center\nmetadata:\n  type: reference\n---\n\nThe hub gathers [[spoke-a]] and [[spoke-b|Spoke B]], see also [[ghost-note]] and itself [[hub]].\n",
		"spoke-a.md": "---\nname: spoke-a\ndescription: a leaf\nmetadata:\n  type: reference\n---\n\nSpoke A points home to [[hub]].\n",
		"spoke-b.md": "---\nname: spoke-b\ndescription: a leaf\nmetadata:\n  type: reference\n---\n\nSpoke B points home to [[hub]] as well.\n",
	}
	dir := fixtureNotesStore(t, index, files)
	b, err := NewNotesBackend(dir)
	if err != nil {
		t.Fatal(err)
	}
	cells, err := b.Cells(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]Cell{}
	for _, c := range cells {
		byID[c.ID] = c
	}

	// Refs are the resolved forward edges, in first-appearance order, de-duplicated,
	// with the self-link and the off-index link excluded.
	if got := byID["hub.md"].Refs; !reflect.DeepEqual(got, []string{"spoke-a.md", "spoke-b.md"}) {
		t.Fatalf("hub.md Refs = %v, want [spoke-a.md spoke-b.md]", got)
	}
	if got := byID["spoke-a.md"].Refs; !reflect.DeepEqual(got, []string{"hub.md"}) {
		t.Fatalf("spoke-a.md Refs = %v, want [hub.md]", got)
	}
	// The off-index [[ghost-note]] is recorded on refs_unresolved, never in Refs.
	if got := byID["hub.md"].Attrs["refs_unresolved"]; got != "ghost-note" {
		t.Fatalf("hub.md refs_unresolved = %q, want %q", got, "ghost-note")
	}
	for _, r := range byID["hub.md"].Refs {
		if r == "ghost-note" || r == "ghost-note.md" {
			t.Fatalf("unresolved [[ghost-note]] leaked into Refs: %v", byID["hub.md"].Refs)
		}
	}

	// Determinism: a second scan of the same store yields byte-identical Refs.
	b2, _ := NewNotesBackend(dir)
	cells2, _ := b2.Cells(context.Background())
	for i := range cells {
		if !reflect.DeepEqual(cells[i].Refs, cells2[i].Refs) {
			t.Fatalf("Refs not deterministic for %s: %v vs %v", cells[i].ID, cells[i].Refs, cells2[i].Refs)
		}
	}

	// Acceptance #1a — filter cells to "references hub.md": only the two spokes.
	refsHub := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpFilter, Pred: &Pred{Op: PredHasRef, Value: "hub.md"}},
	}}
	res, err := Run(context.Background(), b, refsHub, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, c := range res.Working {
		got[c.ID] = true
	}
	if len(got) != 2 || !got["spoke-a.md"] || !got["spoke-b.md"] {
		t.Fatalf("hasref(hub.md) must keep exactly the two spokes, got %v", got)
	}

	// Acceptance #1b — rank by reference count (in-degree), most-backlinked first:
	// hub (2 backlinks) precedes the spokes (1 each).
	rankRefs := Query{Ops: []Op{
		{Kind: OpScan},
		{Kind: OpRank, By: RankRefcount, Desc: true},
	}}
	res, err = Run(context.Background(), b, rankRefs, Caps{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Working) == 0 || res.Working[0].ID != "hub.md" {
		t.Fatalf("rank by refcount desc must put hub.md (2 backlinks) first, got %+v",
			idsOf(res.Working))
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
