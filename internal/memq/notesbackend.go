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
	b := &NotesBackend{dir: dir, verifier: recall.DefaultArtifactVerifier, bodies: map[string][]byte{}}
	if strings.TrimSpace(dir) == "" {
		return b, nil
	}
	indexBytes, err := os.ReadFile(filepath.Join(dir, "MEMORY.md"))
	if err != nil {
		return b, nil // no index — empty corpus, not an error
	}
	for i, fact := range memoryread.ParseIndex(string(indexBytes)) {
		title, fname := fact[0], fact[1]
		raw, err := os.ReadFile(filepath.Join(dir, fname))
		if err != nil {
			continue // index points at a missing file — fewer candidates, never a crash
		}
		b.appendCell(i, title, fname, raw)
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
func (b *NotesBackend) appendCell(step int, title, fname string, raw []byte) {
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
		Attrs: map[string]string{
			"provenance":  NotesProvenance,
			"source_path": filepath.Join(b.dir, fname),
			"note_type":   mtype,
			"title":       title,
		},
	})
	b.bodies[id] = body
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
	out := make([]Cell, len(b.cells))
	copy(out, b.cells)
	return out, nil
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

func init() {
	// loop-recall — the loop-turn orientation query (#2346 R1): the top-K unsealed,
	// untombstoned notes most relevant to the turn's intent, trimmed to a byte
	// budget, rendered through the gated page-in (so a stale or secret-shaped note
	// is refused, not rendered). Corpus-agnostic like every driver; the name
	// documents the intended loop-turn-start use.
	Register(Driver{
		Name: "loop-recall",
		Doc:  "budget-bounded top-K verified orientation block for a loop turn (stale notes refused at page-in)",
		Build: func(p Params) Query {
			k := p.K
			if k <= 0 {
				k = 5
			}
			budget := p.Budget
			if budget <= 0 {
				budget = 8192
			}
			return Query{
				Intent: p.Intent,
				Ops: []Op{
					{Kind: OpScan},
					{Kind: OpFilter, Pred: &Pred{Op: PredAnd, Args: []Pred{
						{Op: PredEq, Field: "sealed", Value: "false"},
						{Op: PredEq, Field: "tombstoned", Value: "false"},
					}}},
					{Kind: OpRank, By: RankRelevance, Desc: true},
					{Kind: OpLimit, K: k},
					{Kind: OpBudget, Bytes: budget},
					{Kind: OpRender},
				},
			}
		},
	})
}
