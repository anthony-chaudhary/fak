package ctxplan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// Span is one addressable unit of history the planner reasons over: SAFE metadata only —
// never the bytes of a sealed span (Descriptor is the extractive-or-sealed descriptor,
// exactly as recall.Page / memq.Cell record it). ctxplan defines it locally so the
// planner depends on NO mechanism package and builds as a self-contained leaf; an adapter
// that lowers a memq.Cell, a recall.Page, or a cdb.Frame into a Span is a thin,
// higher-tier follow-on (it imports both that package and ctxplan, so it lives above
// this leaf, not inside it).
type Span struct {
	ID         string            `json:"id"`                   // stable address in the store (e.g. "span:3")
	Step       int               `json:"step"`                 // ordinal position in the history
	Role       string            `json:"role,omitempty"`       // the producer (tool name, "user", "system")
	Descriptor string            `json:"descriptor"`           // SAFE extractive/sealed descriptor; never sealed bytes
	Digest     string            `json:"digest,omitempty"`     // content address — the recovery handle for an elided span
	Bytes      int64             `json:"bytes"`                // size (the token-cost proxy)
	Durability string            `json:"durability,omitempty"` // turn | session | bounded | durable
	Sealed     bool              `json:"sealed,omitempty"`     // quarantined by the trust gate — never resident
	Tombstoned bool              `json:"tombstoned,omitempty"` // suppressed by context control — never resident
	Attrs      map[string]string `json:"attrs,omitempty"`      // open bag; Attrs["utility"] carries a learned outcome-utility if present
}

// Durability classes — the temporal axis from CONTEXT-IS-NOT-MEMORY.md, mirrored here as
// plain strings (ctxplan imports no mechanism package). A missing/unknown class normalizes
// to the shortest-lived one (turn) — the fail-closed default of the expire-by-default
// posture.
const (
	DurabilityTurn    = "turn"
	DurabilitySession = "session"
	DurabilityBounded = "bounded"
	DurabilityDurable = "durable"
)

var durabilityRank = map[string]int{
	DurabilityTurn: 0, DurabilitySession: 1, DurabilityBounded: 2, DurabilityDurable: 3,
}

// NormDurability maps any class string to the canonical vocabulary, failing closed to turn
// for a missing/unknown value.
func NormDurability(s string) string {
	if _, ok := durabilityRank[s]; ok {
		return s
	}
	return DurabilityTurn
}

// ErrSealed is returned by a Store.Materialize that refuses a page-in because the span is
// quarantined by the trust gate. A real backend (a recall image, a memq backend) wraps its
// own seal error in this so a caller can branch on "the gate held" vs a lookup miss.
var ErrSealed = errors.New("ctxplan: span sealed by the trust gate")

// ErrTombstoned is returned by a Store.Materialize that refuses a page-in because the span
// was suppressed by context control. Tombstone is a SEPARATE gate from seal (recall.Resolve
// returns ErrTombstoned; memq's backend refuses both), and a Store MUST refuse a tombstoned
// span on the documented demand-page path — otherwise a span the planner elided as
// suppressed could be paged back in through the recovery handle, defeating the suppression.
var ErrTombstoned = errors.New("ctxplan: span suppressed by context control")

// Store is the history image the planner views: it supplies spans (SAFE metadata) and
// trust-gated byte access. Materialize is the gated page-in — a sealed/refused span
// returns an error wrapping ErrSealed, and its bytes never cross the gate. A real
// deployment backs this with a recall core image or a memq backend; the in-memory MemStore
// below is the zero-setup reference implementation for the demo and tests.
type Store interface {
	Spans(ctx context.Context) ([]Span, error)
	Materialize(ctx context.Context, id string) ([]byte, error)
}

// Rendered is one span materialized into the fresh history (its bytes paged in through the
// gate). Bytes/Tokens record the realized resident cost.
type Rendered struct {
	ID         string `json:"id"`
	Step       int    `json:"step"`
	Role       string `json:"role,omitempty"`
	Descriptor string `json:"descriptor,omitempty"`
	Bytes      int64  `json:"bytes"`
	Tokens     int    `json:"tokens"`
}

// Refusal is a selected span the gate declined to page in (sealed, or its bytes went
// missing) — reported, never rendered.
type Refusal struct {
	ID     string `json:"id"`
	Step   int    `json:"step"`
	Role   string `json:"role,omitempty"`
	Reason string `json:"reason"`
}

// Digest is the canonical content address (sha256 hex) — the same scheme recall/blob/memq
// use, so a ctxplan digest and a recall digest are interchangeable as recovery handles.
func Digest(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// ---------------------------------------------------------------------------
// MemStore — the in-memory reference Store: a span table plus a content-addressed blob
// map. It implements Store with zero setup (no disk, no recall image), so the whole
// planner runs in the demo and the tests. A sealed span's bytes stay in the CAS (audit),
// but Materialize refuses them, exactly as recall does.
// ---------------------------------------------------------------------------

// MemStore is the in-memory reference Store.
type MemStore struct {
	spans []Span
	cas   map[string][]byte
}

// NewMemStore returns an empty store.
func NewMemStore() *MemStore { return &MemStore{cas: map[string][]byte{}} }

// Add appends a span whose bytes are body, computing the digest and a safe descriptor (a
// sealed span gets a sealed-metadata descriptor, never its bytes). The id is assigned as
// "span:<n>" by insertion order.
func (m *MemStore) Add(role, durability string, body []byte, sealed bool) Span {
	digest := Digest(body)
	s := Span{
		ID:         fmt.Sprintf("span:%d", len(m.spans)),
		Step:       len(m.spans),
		Role:       role,
		Digest:     digest,
		Bytes:      int64(len(body)),
		Durability: NormDurability(durability),
		Sealed:     sealed,
	}
	if sealed {
		s.Descriptor = fmt.Sprintf("%s: [sealed: %d bytes]", role, len(body))
	} else {
		s.Descriptor = descriptorOf(role, body)
	}
	m.cas[digest] = append([]byte(nil), body...)
	m.spans = append(m.spans, s)
	return s
}

// Spans returns a snapshot of the span table (safe metadata only).
func (m *MemStore) Spans(_ context.Context) ([]Span, error) {
	out := make([]Span, len(m.spans))
	copy(out, m.spans)
	return out, nil
}

// Materialize pages a span's bytes in, refusing a sealed span (the trust gate).
func (m *MemStore) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, s := range m.spans {
		if s.ID != id {
			continue
		}
		if s.Sealed {
			return nil, fmt.Errorf("%w: span %s", ErrSealed, id)
		}
		if s.Tombstoned {
			return nil, fmt.Errorf("%w: span %s", ErrTombstoned, id)
		}
		b, ok := m.cas[s.Digest]
		if !ok {
			return nil, fmt.Errorf("ctxplan: span %s bytes absent from CAS", id)
		}
		return append([]byte(nil), b...), nil
	}
	return nil, fmt.Errorf("ctxplan: no span %s", id)
}

// descriptorOf builds a real extractive descriptor for a benign body: the role plus the
// first non-empty line, bounded — the recall.descriptorOf shape, kept local.
func descriptorOf(role string, body []byte) string {
	line := headLine(body, 120)
	if line == "" {
		return role
	}
	return role + ": " + line
}

func headLine(b []byte, max int) string {
	s := strings.TrimSpace(string(b))
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSpace(s)
	if len(s) > max {
		s = s[:max]
	}
	return s
}
