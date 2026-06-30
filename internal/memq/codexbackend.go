package memq

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// CodexBackend is a READ-ONLY memq Backend over an external Codex memories home —
// the generated memory artifacts the OpenAI Codex agent writes under CODEX_HOME
// (normally ~/.codex). It surfaces those files as recall CANDIDATES, never as
// authoritative team rules: every cell is stamped provenance=external/untrusted,
// every page-in is re-screened through fak's trust gate (ctxmmu.ScreenBytes — the
// same screen recall runs), and a secret/injection-shaped memory is SEALED so it
// can never render into context. fak never writes a Codex file: the backend opens
// no file for write, implements neither Tombstoner nor Pruner, and treats a
// missing/partial home as an empty (not an error) corpus — the fail-closed posture
// for untrusted external state.
//
// Layout (authoritative per the Codex manual, https://developers.openai.com/codex/memories):
//   - <home>/memories/*.md                          → kind codex-memory
//   - <home>/memories_extensions/chronicle/*.md      → kind codex-chronicle
//
// Chronicle memories are generated from screen content and so carry a HIGHER
// prompt-injection risk; the backend tags them with a distinct source kind and a
// suspicion attr so a policy can exclude them wholesale, independent of the
// per-cell content screen.
type CodexBackend struct {
	home             string
	includeChronicle bool
	cells            []Cell
	bodies           map[string][]byte // by cell ID; the raw file bytes, paged in through the gate
}

// Provenance / attribute vocabulary stamped on every Codex cell. These are plain
// strings (memq carries no enum for external provenance) so a Pred can select on
// attr:provenance / attr:source / kind without the core knowing about Codex.
const (
	// CodexProvenance is the witness/provenance label: this is external, generated,
	// untrusted state — never a fak- or team-authored rule.
	CodexProvenance = "external/untrusted"
	// KindCodexMemory tags an ordinary generated memory file (<home>/memories).
	KindCodexMemory = "codex-memory"
	// KindCodexChronicle tags a Chronicle-origin memory (screen-derived; higher risk).
	KindCodexChronicle = "codex-chronicle"
)

// NewCodexBackend builds a read-only backend by scanning a Codex home. A
// missing/partial home yields an EMPTY backend (no error) — untrusted external
// state must never crash the algebra. includeChronicle gates the screen-derived
// Chronicle tree; when false those files are not enumerated at all.
//
// home is used verbatim (resolve flag > CODEX_HOME > the platform default at the
// CALL site, so this constructor never silently reaches into a real ~/.codex). An
// empty home is treated as "no home configured" and yields an empty backend.
func NewCodexBackend(home string, includeChronicle bool) (*CodexBackend, error) {
	b := &CodexBackend{home: home, includeChronicle: includeChronicle, bodies: map[string][]byte{}}
	if strings.TrimSpace(home) == "" {
		return b, nil
	}
	b.scan(filepath.Join(home, "memories"), KindCodexMemory)
	if includeChronicle {
		b.scan(filepath.Join(home, "memories_extensions", "chronicle"), KindCodexChronicle)
	}
	// Deterministic order: sort by (kind, id) so the cell table — and therefore every
	// Run/Explain over this backend — is byte-stable regardless of OS readdir order.
	sort.Slice(b.cells, func(i, j int) bool {
		if b.cells[i].Kind != b.cells[j].Kind {
			return b.cells[i].Kind < b.cells[j].Kind
		}
		return b.cells[i].ID < b.cells[j].ID
	})
	return b, nil
}

// scan enumerates the markdown files directly under dir (non-recursive — Codex
// writes a flat directory of generated files) and appends one cell per file. A
// missing dir, an unreadable entry, or a non-markdown file is skipped silently:
// the backend degrades to fewer candidates, never to a crash.
func (b *CodexBackend) scan(dir, kind string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return // missing/partial home — empty contribution, never an error
	}
	for _, e := range entries {
		if e.IsDir() || !isMarkdown(e.Name()) {
			continue
		}
		path := filepath.Join(dir, e.Name())
		body, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		info, err := e.Info()
		var mtime int64
		if err == nil {
			mtime = info.ModTime().UTC().Unix()
		}
		b.appendCell(kind, path, e.Name(), body, mtime)
	}
}

// appendCell turns one Codex file into a Cell carrying only SAFE metadata: a stable
// id, a content digest, a bounded extractive descriptor, the source path/kind, the
// mtime, and conservative durability. The raw bytes are held off-cell in bodies and
// surface only through the trust-gated Materialize. A content screen at scan time
// pre-SEALS a secret/injection-shaped file so it is refused on page-in.
func (b *CodexBackend) appendCell(kind, path, name string, body []byte, mtime int64) {
	digest := Digest(body)
	id := kind + ":" + name

	_, caught := ctxmmu.ScreenBytes(body)
	sealed := caught

	suspicion := "external"
	if kind == KindCodexChronicle {
		// Chronicle is generated from screen content — a strictly higher injection
		// surface; tag it so a policy can exclude the whole class up front.
		suspicion = "external-chronicle"
	}

	attrs := map[string]string{
		"provenance":  CodexProvenance,
		"source":      kind,
		"source_path": path,
		"suspicion":   suspicion,
		"mtime":       strconv.FormatInt(mtime, 10),
	}

	descriptor := name
	if sealed {
		descriptor = fmt.Sprintf("%s: [sealed external memory: %d bytes]", kind, len(body))
	} else if line := headLine(body, 120); line != "" {
		descriptor = kind + ": " + line
	}

	c := Cell{
		ID:         id,
		Step:       -1, // Codex files are unordered; ranking falls back to id ties
		Role:       kind,
		Kind:       kind,
		Descriptor: descriptor,
		Digest:     digest,
		Bytes:      int64(len(body)),
		// Codex memories are external generated candidates, NEVER durable team rules:
		// the strongest class a Codex file may claim is bounded (it can outlive a
		// session but is re-screened every page-in and never earns `durable` here).
		// Chronicle, the screen-derived higher-risk class, is capped at session.
		Durability: codexDurability(kind),
		Sealed:     sealed,
		Witness:    CodexProvenance,
		Attrs:      attrs,
	}
	b.cells = append(b.cells, c)
	b.bodies[id] = append([]byte(nil), body...)
}

// codexDurability caps the durability an external Codex memory may claim. An
// ordinary generated memory is at most `bounded`; a Chronicle (screen-derived,
// higher injection risk) is held to `session`. Neither can reach `durable` — a
// standing team rule belongs in AGENTS.md / a checked-in doc, never an external
// generated file (the issue's explicit posture).
func codexDurability(kind string) string {
	if kind == KindCodexChronicle {
		return DurabilitySession
	}
	return DurabilityBounded
}

// Cells returns the scanned page table (safe metadata only) — a snapshot copy so
// the executor never mutates the backend's slice.
func (b *CodexBackend) Cells(_ context.Context) ([]Cell, error) {
	out := make([]Cell, len(b.cells))
	copy(out, b.cells)
	return out, nil
}

// Materialize pages one Codex file in THROUGH THE TRUST GATE. A cell sealed at scan
// time is refused with ErrSealed; otherwise the raw bytes are RE-SCREENED at read
// time (the same independent content re-screen recall runs, so a file that changed
// shape since the scan, or that the scan-time heuristic missed, is still caught)
// before any byte crosses the gate. The Codex file itself is never modified.
func (b *CodexBackend) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, c := range b.cells {
		if c.ID != id {
			continue
		}
		if c.Sealed {
			return nil, fmt.Errorf("%w: codex cell %s", ErrSealed, id)
		}
		body, ok := b.bodies[id]
		if !ok {
			return nil, fmt.Errorf("memq: codex cell %s bytes absent", id)
		}
		// Read-time re-screen: poison never enters context even if it slipped the
		// scan-time seal. This is the page-in gate the issue requires.
		if _, caught := ctxmmu.ScreenBytes(body); caught {
			return nil, fmt.Errorf("%w: codex cell %s failed the read-time screen", ErrSealed, id)
		}
		return append([]byte(nil), body...), nil
	}
	return nil, fmt.Errorf("memq: no codex cell %s", id)
}

// Home reports the resolved Codex home this backend scanned ("" when none was
// configured). Read-only accessor for the CLI label.
func (b *CodexBackend) Home() string { return b.home }

func isMarkdown(name string) bool {
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, ".md") || strings.HasSuffix(lower, ".markdown")
}

func init() {
	// codex-recall — render the top-K most relevant UNSEALED, untrusted Codex
	// candidates. It is the `recall` driver scoped to the external corpus: an agent
	// pulls Codex-learned context as candidates, but every page-in still routes
	// through the gate, so a sealed (injection/secret-shaped) file is refused, not
	// rendered. The driver is corpus-agnostic — it runs over ANY backend — but its
	// name documents the intended external-recall use.
	Register(Driver{
		Name: "codex-recall",
		Doc:  "render the top-K most relevant external Codex memory candidates (untrusted; gated page-in)",
		Build: func(p Params) Query {
			k := p.K
			if k <= 0 {
				k = 5
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
					{Kind: OpRender},
				},
			}
		},
	})
}
