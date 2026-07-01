package memq

import (
	"context"
	"fmt"
)

// MemStore is the in-memory reference Backend: a page table plus a content-addressed
// blob map. It implements Backend, Tombstoner, and Pruner, so the whole algebra runs
// with zero setup (no disk, no recall image) — the substrate for the demo and the
// tests. A sealed cell's bytes stay in the CAS (audit), but Materialize refuses them,
// exactly as recall does. It also owns a PromotionLedger (#1595): every Add whose
// resulting durability crosses past turn-class mints a PromotionRecord, so a caller can
// later explain that write from structure alone.
type MemStore struct {
	cells  []Cell
	cas    map[string][]byte // by digest (so aliases share one blob, and orphans can exist)
	ledger *PromotionLedger
}

// NewMemStore returns an empty store with an empty promotion ledger.
func NewMemStore() *MemStore {
	return &MemStore{cas: map[string][]byte{}, ledger: NewPromotionLedger()}
}

// Promotions returns the store's promotion ledger — the audit trail behind every
// non-turn-class cell this store holds (#1595). Callers (notably `fak memory
// explain-promotion`) read it to explain a fact without asking a model to narrate.
func (m *MemStore) Promotions() *PromotionLedger { return m.ledger }

// Add appends a cell whose bytes are `body`, computing the digest and a safe
// descriptor. A sealed cell gets a sealed-metadata descriptor (never its bytes), just
// as recall.Recorder does. id is assigned as "cell:<n>" by insertion order. This is the
// ConsentInferred, producer-defaulted convenience used by the demo corpus and existing
// callers; a caller that has real consent/producer/expiry provenance should call
// AddPromoted instead so the ledger records it faithfully.
func (m *MemStore) Add(role, kind, durability string, body []byte, sealed bool) Cell {
	return m.AddPromoted(role, kind, durability, body, sealed, PromotionMeta{})
}

// PromotionMeta is the caller-supplied provenance for a promoting Add: whether the
// promotion was explicitly consented to, who/what produced it, and (for a bounded
// class) its expiry. A zero PromotionMeta defaults to ConsentInferred / an
// empty-producer-defaults-to-"unknown" / no expiry — the honest floor when a caller has
// no better information, never a silent upgrade to explicit consent.
type PromotionMeta struct {
	Consent  string
	Producer string
	Expiry   string
	Reason   string
}

// AddPromoted is Add plus an explicit PromotionMeta: it appends the cell exactly as Add
// does, then — if the resulting durability is not DurabilityTurn — records a
// PromotionRecord on the store's ledger capturing the source span, durability class,
// consent, producer, and expiry (#1595). A turn-class cell records nothing (it was
// never promoted; see PromotionLedger.Record).
func (m *MemStore) AddPromoted(role, kind, durability string, body []byte, sealed bool, meta PromotionMeta) Cell {
	digest := Digest(body)
	c := Cell{
		ID:         fmt.Sprintf("cell:%d", len(m.cells)),
		Step:       len(m.cells),
		Role:       role,
		Kind:       kind,
		Digest:     digest,
		Bytes:      int64(len(body)),
		Durability: NormDurability(durability),
		Sealed:     sealed,
	}
	if sealed {
		c.Descriptor = fmt.Sprintf("%s: [sealed: %d bytes]", role, len(body))
	} else {
		c.Descriptor = descriptorOf(role, body)
	}
	m.cas[digest] = append([]byte(nil), body...)
	m.cells = append(m.cells, c)

	producer := meta.Producer
	if producer == "" {
		producer = role
	}
	consent := meta.Consent
	if consent == "" {
		consent = ConsentInferred
	}
	m.ledger.Record(PromotionRecord{
		CellID: c.ID,
		SourceSpan: SourceSpan{
			Step: c.Step, Role: role, Descriptor: c.Descriptor, Digest: digest,
		},
		Durability: c.Durability,
		Consent:    consent,
		Producer:   producer,
		Expiry:     meta.Expiry,
		Reason:     meta.Reason,
	})
	return c
}

// AddOrphanBlob inserts a CAS blob that NO cell references — an unreferenced blob the
// prune op reclaims. Returns its digest.
func (m *MemStore) AddOrphanBlob(body []byte) string {
	d := Digest(body)
	m.cas[d] = append([]byte(nil), body...)
	return d
}

// Cells returns a snapshot of the page table (safe metadata only).
func (m *MemStore) Cells(_ context.Context) ([]Cell, error) {
	out := make([]Cell, len(m.cells))
	copy(out, m.cells)
	return out, nil
}

// Materialize pages a cell's bytes in, refusing a sealed cell (the trust gate).
func (m *MemStore) Materialize(_ context.Context, id string) ([]byte, error) {
	for _, c := range m.cells {
		if c.ID != id {
			continue
		}
		if c.Sealed {
			return nil, fmt.Errorf("%w: cell %s", ErrSealed, id)
		}
		b, ok := m.cas[c.Digest]
		if !ok {
			return nil, fmt.Errorf("memq: cell %s bytes absent from CAS", id)
		}
		return append([]byte(nil), b...), nil
	}
	return nil, fmt.Errorf("memq: no cell %s", id)
}

// Tombstone marks a cell suppressed (negative-only; the cell row and its bytes
// survive). Returns false if the cell is unknown or already tombstoned.
func (m *MemStore) Tombstone(_ context.Context, id, _, _ string) (bool, error) {
	for i := range m.cells {
		if m.cells[i].ID == id {
			if m.cells[i].Tombstoned {
				return false, nil
			}
			m.cells[i].Tombstoned = true
			return true, nil
		}
	}
	return false, nil
}

// Prune reclaims CAS blobs no cell references. With apply=false it only counts.
func (m *MemStore) Prune(_ context.Context, apply bool) (int, int64, error) {
	referenced := map[string]bool{}
	for _, c := range m.cells {
		referenced[c.Digest] = true
	}
	blobs := 0
	var bytes int64
	for d, b := range m.cas {
		if referenced[d] {
			continue
		}
		blobs++
		bytes += int64(len(b))
		if apply {
			delete(m.cas, d)
		}
	}
	return blobs, bytes, nil
}

// descriptorOf builds a real extractive descriptor for a benign body: the role plus
// the first non-empty line, bounded — the recall.descriptorOf shape, kept local.
func descriptorOf(role string, body []byte) string {
	line := headLine(body, 120)
	if line == "" {
		return role
	}
	return role + ": " + line
}
