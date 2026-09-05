package memq

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/memoryread"
	"github.com/anthony-chaudhary/fak/internal/recall"
)

// NotesBackend is a READ-ONLY memq Backend over a markdown memory store — the
// MEMORY.md-indexed fact-file grammar internal/memoryread renders (the committed
// fleet mirror `.claude/memory`, or a harness auto-memory store passed by path).
// It is the loop-facing recall corpus (#2347, epic #2346 R1): the SAME algebra
// that runs over recall core images and Codex memories runs, unchanged, over the
// store the agent loop actually reads.
//
// Two gates run on every page-in, and neither trusts the note's prose:
//
//   - the content screen (ctxmmu.ScreenBytes, the screen recall and the Codex
//     backend run) — a secret/injection-shaped note is SEALED, never rendered;
//   - read-time artifact re-verification (recall.ExtractArtifactClaims + an
//     injectable recall.ArtifactVerifier, default DefaultArtifactVerifier): a note
//     naming a commit SHA, repo path, or flag that no longer verifies against the
//     current checkout is refused ErrStale — a frozen self-report from a past
//     session must not re-enter context wearing the authority of a fact (#2077).
//
// The index IS the curation: only fact files linked from MEMORY.md become cells
// (an unindexed file is invisible to recall, exactly as it is to the harness).
// The backend opens no file for write and implements neither Tombstoner nor
// Pruner; a missing/partial store yields an empty corpus, never an error.
type NotesBackend struct {
	dir      string
	verifier recall.ArtifactVerifier
	cells    []Cell
	bodies   map[string][]byte // by cell ID; frontmatter-stripped note bodies
}

// Provenance / kind vocabulary stamped on every note cell, selectable via
// attr:provenance / kind without the core knowing about the store.
const (
	// NotesProvenance labels a note as the agent's own accumulated memory: curated,
	// but still a generated self-report — verified at read time, never taken on faith.
	NotesProvenance = "memory-store/self-report"
	// KindMemoryNote tags one MEMORY.md-indexed fact file.
	KindMemoryNote = "memory-note"
)

// NewNotesBackend scans a memory store directory (MEMORY.md + linked fact files).
// A missing/empty store yields an EMPTY backend (no error) — a fresh node or a
// scrubbed clone must not crash the algebra. The verifier defaults to
// recall.DefaultArtifactVerifier; tests inject their own via WithVerifier.
func NewNotesBackend(dir string) (*NotesBackend, error) {
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		dir = filepath.Dir(dir)
	}
	b := &NotesBackend{dir: dir, verifier: recall.DefaultArtifactVerifier, bodies: map[string][]byte{}}
	if strings.TrimSpace(dir) == "" {
		return b, nil
	}
	indexBytes, err := os.ReadFile(filepath.Join(dir, "MEMORY.md"))
	if err != nil {
		return b, nil // no index — empty corpus, not an error
	}
	// Two passes so the [[wikilink]] resolver is complete before any note's References
	// are computed (W2, #2620). Pass 1 reads every indexed fact file and records its
	// resolution keys (lower filename-stem and frontmatter `name:`) -> the canonical
	// cell ID (its filename); pass 2 appends each cell, resolving its forward links
	// against that map. The index is the curation boundary: a link is an edge only when
	// its target is itself indexed — an off-index link is flagged, never invented.
	type indexedNote struct {
		step         int
		title, fname string
		raw          []byte
	}
	var notes []indexedNote
	resolver := map[string]string{}
	for i, fact := range memoryread.ParseIndex(string(indexBytes)) {
		title, fname := fact[0], fact[1]
		raw, err := os.ReadFile(filepath.Join(dir, fname))
		if err != nil {
			continue // index points at a missing file — fewer candidates, never a crash
		}
		notes = append(notes, indexedNote{step: i, title: title, fname: fname, raw: raw})
		resolver[strings.ToLower(strings.TrimSuffix(fname, ".md"))] = fname
		if name := noteName(string(raw)); name != "" {
			resolver[strings.ToLower(name)] = fname
		}
	}
	edges := map[string][]string{}
	order := make([]string, 0, len(notes))
	for _, n := range notes {
		order = append(order, n.fname)
		if targets := b.appendCell(n.step, n.title, n.fname, n.raw, resolver); len(targets) > 0 {
			edges[n.fname] = targets
		}
	}
	// W5 (#2624): a note reachable as the target of a live supersedes edge is
	// withheld from the working set by default — the same negative-only posture
	// as an OpTombstone suppression (bytes and row survive for audit; the default
	// recall query's tombstoned=false filter is what withholds it). The
	// deterministic resolution — chain collapse, the documented cycle tie-break —
	// is internal/recall's ResolveSupersession.
	withheldBy := recall.ResolveSupersession(edges, order)
	for i := range b.cells {
		if src, ok := withheldBy[b.cells[i].ID]; ok {
			b.cells[i].Tombstoned = true
			b.cells[i].Attrs["superseded_by"] = src
		}
	}
	return b, nil
}

// WithVerifier overrides the read-time artifact verifier (nil restores the
// default) — the same injectable seam recall.Session exposes.
func (b *NotesBackend) WithVerifier(v recall.ArtifactVerifier) *NotesBackend {
	if v == nil {
		v = recall.DefaultArtifactVerifier
	}
	b.verifier = v
	return b
}

// Dir reports the store directory this backend scanned (for the CLI label).
func (b *NotesBackend) Dir() string { return b.dir }

// appendCell turns one indexed fact file into a Cell carrying only SAFE metadata.
// The frontmatter-stripped body is held off-cell and surfaces only through the
// gated Materialize. Step preserves MEMORY.md index order (the curation order).
// resolver maps each indexed note's link keys to its cell ID, so the note's forward
// [[wikilinks]] become Cell.Refs — the graph edge set the algebra can filter/rank on.
// The returned slice is the note's resolved `supersedes:` targets (W5, #2624) — the
// typed retirement edges the caller folds into the supersession resolution.
func (b *NotesBackend) appendCell(step int, title, fname string, raw []byte, resolver map[string]string) []string {
	desc, mtype := parseNoteMeta(string(raw))
	body := []byte(memoryread.StripFrontmatter(string(raw)))

	_, caught := ctxmmu.ScreenBytes(body)
	sealed := caught

	descriptor := title
	if desc != "" {
		descriptor = title + ": " + desc
	}
	if sealed {
		descriptor = fmt.Sprintf("%s: [sealed memory note: %d bytes]", title, len(body))
	}

	id := fname
	refs, unresolved := resolveWikiLinks(body, id, resolver)
	attrs := map[string]string{
		"provenance":  NotesProvenance,
		"source_path": filepath.Join(b.dir, fname),
		"note_type":   mtype,
		"title":       title,
	}
	// An off-index forward reference is RECORDED, not dropped — the index is the
	// curation boundary, but a broken/not-yet-written link is real rot the graph must
	// keep visible (matching the W1 forward-reference grammar).
	if len(unresolved) > 0 {
		attrs["refs_unresolved"] = strings.Join(unresolved, ",")
	}
	// The frontmatter `supersedes: [[target]]` declarations (W5, #2624) resolve
	// through the same wikilink grammar as body links: dedup, self excluded, and a
	// dangling target — one that is not an indexed fact file — is flagged as rot on
	// supersedes_unresolved, never a crash and never an edge. Resolved targets are
	// surfaced on the superseder as the typed `supersedes` attr (the W4 graph reads
	// it) and returned for the working-set withholding fold.
	supersedes, supUnresolved := resolveWikiLinks([]byte(noteSupersedes(string(raw))), id, resolver)
	if len(supersedes) > 0 {
		attrs["supersedes"] = strings.Join(supersedes, ",")
	}
	if len(supUnresolved) > 0 {
		attrs["supersedes_unresolved"] = strings.Join(supUnresolved, ",")
	}
	b.cells = append(b.cells, Cell{
		ID:         id,
		Step:       step,
		Role:       "memory",
		Kind:       KindMemoryNote,
		Descriptor: descriptor,
		Digest:     Digest(body),
		Bytes:      int64(len(body)),
		Durability: noteDurability(mtype),
		Sealed:     sealed,
		Witness:    NotesProvenance,
		Refs:       refs,
		Attrs:      attrs,
	})
	b.bodies[id] = body
	return supersedes
}

// wikiLinkRE matches a [[target]] wikilink, tolerating a [[target|display]] alias and
// stripping a #anchor — the same grammar internal/memvaluescore flags broken links
// with. The shared extractor + backlink index is the W1 (internal/memgraph) lift;
// until that lands, W2 keeps a local copy so the memq algebra gains the edge set
// without depending on an unshipped package (#2620, epic #2618).
var wikiLinkRE = regexp.MustCompile(`\[\[([^\[\]|#]+?)(?:\|[^\[\]]*)?\]\]`)

// noteNameRE reads the top-level frontmatter `name:` slug — the alternate link key an
// authored [[name]] may use when it differs from the filename stem.
var noteNameRE = regexp.MustCompile(`(?m)^name:\s*(\S+)`)

// noteSupersedesRE reads a top-level frontmatter `supersedes:` declaration — one or
// many [[wikilinks]] on the line, repeatable across lines (W5, #2624).
var noteSupersedesRE = regexp.MustCompile(`(?m)^supersedes:\s*(.+?)\s*$`)

// noteSupersedes returns the raw wikilink text of every frontmatter `supersedes:`
// declaration, space-joined for the shared resolver, or "" when the block is absent
// or malformed (lexical and tolerant, like parseNoteMeta). Body-text `supersedes:`
// prose never counts — the edge is a frontmatter declaration, not a phrase.
func noteSupersedes(raw string) string {
	if !strings.HasPrefix(raw, "---") {
		return ""
	}
	end := strings.Index(raw[3:], "\n---")
	if end == -1 {
		return ""
	}
	var tails []string
	for _, m := range noteSupersedesRE.FindAllStringSubmatch(raw[:end+3], -1) {
		tails = append(tails, m[1])
	}
	return strings.Join(tails, " ")
}

// noteName returns the frontmatter `name:` slug, or "" when the block is absent or
// malformed (lexical and tolerant, like parseNoteMeta).
func noteName(raw string) string {
	if !strings.HasPrefix(raw, "---") {
		return ""
	}
	end := strings.Index(raw[3:], "\n---")
	if end == -1 {
		return ""
	}
	if m := noteNameRE.FindStringSubmatch(raw[:end+3]); m != nil {
		return m[1]
	}
	return ""
}

// resolveWikiLinks splits a note body's forward [[wikilinks]] into (resolved,
// unresolved) against the indexed-file resolver. A resolved link becomes the target's
// canonical cell ID (its filename) in Cell.Refs — deterministic first-appearance order
// with de-duplication, so a fixed body yields a byte-identical edge set. A self-link is
// not an edge (it would inflate the note's own in-degree). An unresolved link — one
// whose target is not an indexed fact file — is returned to be flagged, never dropped.
func resolveWikiLinks(body []byte, self string, resolver map[string]string) (refs, unresolved []string) {
	seenRef := map[string]bool{}
	seenUn := map[string]bool{}
	for _, m := range wikiLinkRE.FindAllStringSubmatch(string(body), -1) {
		target := strings.TrimSpace(m[1])
		if target == "" {
			continue
		}
		fname, ok := resolver[strings.ToLower(target)]
		if !ok {
			if !seenUn[target] {
				seenUn[target] = true
				unresolved = append(unresolved, target)
			}
			continue
		}
		if fname == self { // a note linking to itself is not a graph edge
			continue
		}
		if !seenRef[fname] {
			seenRef[fname] = true
			refs = append(refs, fname)
		}
	}
	return refs, unresolved
}

// noteDurability maps the store's frontmatter `metadata.type` onto the durability
// axis. This is a TEMPORAL statement (how long the fact intends to live), not a
// trust one — trust is the screen + the read-time verifier. Who the user is, how
// they want work done, and where external resources live are standing facts
// (durable); project state is time-bounded truth (bounded); anything untyped
// fails closed to session.
func noteDurability(mtype string) string {
	switch mtype {
	case "user", "feedback", "reference":
		return DurabilityDurable
	case "project":
		return DurabilityBounded
	default:
		return DurabilitySession
	}
}

// Cells returns the scanned page table (safe metadata only) as a snapshot copy.
func (b *NotesBackend) Cells(_ context.Context) ([]Cell, error) {
	return snapshotCells(b.cells), nil
}

// Materialize pages one note in through BOTH gates: the content re-screen (a
// note that turned secret/injection-shaped since the scan is sealed, not
// rendered) and the read-time artifact re-verification (a stale concrete claim
// refuses the whole note with the failing claim named). Fresh and unverifiable
// claims pass — the verifier is not an oracle over prose; tagging the hedge is
// the render surface's job via Verify.
func (b *NotesBackend) Materialize(ctx context.Context, id string) ([]byte, error) {
	cell, body, err := b.lookup(id)
	if err != nil {
		return nil, err
	}
	if cell.Sealed {
		return nil, fmt.Errorf("%w: memory note %s", ErrSealed, id)
	}
	if _, caught := ctxmmu.ScreenBytes(body); caught {
		return nil, fmt.Errorf("%w: memory note %s failed the read-time screen", ErrSealed, id)
	}
	for _, f := range b.verifyFindings(ctx, cell, body) {
		if f.Status == recall.ArtifactStale {
			return nil, fmt.Errorf("%w: memory note %s claims %s %q: %s",
				ErrStale, id, f.Claim.Kind, f.Claim.Value, f.Detail)
		}
	}
	return append([]byte(nil), body...), nil
}

// Verify re-runs the read-time artifact verification for one note and returns
// the per-claim findings — the seam a render surface uses to tag a rendered
// note fresh vs unverifiable (a note with no concrete claims returns an empty
// slice: nothing checkable, render hedged).
func (b *NotesBackend) Verify(ctx context.Context, id string) ([]recall.ArtifactFinding, error) {
	cell, body, err := b.lookup(id)
	if err != nil {
		return nil, err
	}
	return b.verifyFindings(ctx, cell, body), nil
}

func (b *NotesBackend) verifyFindings(ctx context.Context, cell Cell, body []byte) []recall.ArtifactFinding {
	claims := recall.ExtractArtifactClaims(string(body) + "\n" + cell.Descriptor)
	if len(claims) == 0 {
		return nil
	}
	return b.verifier(ctx, claims)
}

func (b *NotesBackend) lookup(id string) (Cell, []byte, error) {
	for _, c := range b.cells {
		if c.ID == id {
			return c, b.bodies[id], nil
		}
	}
	return Cell{}, nil, fmt.Errorf("memq: no memory note %s", id)
}

// Frontmatter fields the store grammar carries: a top-level `description:` and a
// `type:` nested under `metadata:`. Lexical and tolerant — a malformed block
// yields empty fields, never an error.
var (
	noteDescRE = regexp.MustCompile(`(?m)^description:\s*(.+?)\s*$`)
	noteTypeRE = regexp.MustCompile(`(?m)^\s+type:\s*(\S+)`)
)

func parseNoteMeta(raw string) (description, mtype string) {
	if !strings.HasPrefix(raw, "---") {
		return "", ""
	}
	end := strings.Index(raw[3:], "\n---")
	if end == -1 {
		return "", ""
	}
	front := raw[:end+3]
	if m := noteDescRE.FindStringSubmatch(front); m != nil {
		description = m[1]
	}
	if m := noteTypeRE.FindStringSubmatch(front); m != nil {
		mtype = m[1]
	}
	return description, mtype
}
