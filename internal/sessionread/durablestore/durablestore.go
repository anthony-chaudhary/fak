// Package durablestore is the DURABLE, file-backed store that sits behind the gateway's
// span/context reads (fak_context_spans / fak_context_restore) — child C3 of the
// sessionread plane (epic #4176, issue #4194).
//
// # The gap this closes (verified)
//
// Today the gateway serves those MCP reads out of an IN-MEMORY per-trace stash —
// internal/gateway/ctxrestore.go's `s.ctxRestore` (map[string]*sessionCtxRestore). That
// map is generationally reset on overflow and, more decisively, LOST the instant the
// gateway process restarts or the session ends. So a span the compaction dropped — the
// content-addressed originating task a resuming model needs — evaporates with the process,
// and a finished/closed session becomes unreadable. There is no durable substrate under
// these tools. That is the whole gap: the recovery handle is only as durable as one
// process's heap.
//
// # The shape of the fix (compose, do not duplicate)
//
// The gateway restore path ALREADY falls through any stash miss to a content-addressed
// ctxplan.Store (internal/gateway/ctxrestore_store.go `restoreFromStore`): it matches a
// span by its content-address Digest and pages the bytes in THROUGH the store's own trust
// gate, mapping ctxplan.ErrSealed / ctxplan.ErrTombstoned to the same refusal shape the
// stash returns. So the highest-value, lowest-friction deliverable is a durable file-backed
// implementation OF ctxplan.Store — it drops into that extant seam with zero new routing
// code, and the in-memory stash becomes a hot cache over it (a stash miss resolves from
// disk instead of dying at the process boundary).
//
// This package therefore COMPOSES with internal/ctxplan rather than re-deriving a second
// content-address scheme: it reuses ctxplan.Digest (the canonical sha256-hex address that
// recall / blob / memq / the compaction tombstone all share, so a durable handle and a
// stash handle are the SAME address for the same bytes), ctxplan.Span (the SAFE byte-free
// projection Spans returns), and the ctxplan.ErrSealed / ctxplan.ErrTombstoned sentinels
// (so a caller's errors.Is branch is byte-for-byte identical whether the stash, a recall
// image, or this durable store served the digest). ctxplan was verified clean for this use:
// it builds standalone and drags in zero of the volatile internal/session or
// internal/gateway trees (`go list -deps` reports 0 internal/session deps), so importing it
// keeps this leaf hermetic and testable in isolation.
//
// # Durability is the point
//
// The store mirrors gateway.restoreEntry's span shape — { id (sha256-hex content-address),
// excerpt, bytes, sealed, tombstoned, cluster, kind } — and persists each span as one
// content-addressed JSON file at <dir>/<id>.json where id == Digest(bytes). Two properties
// carry the "survives a gateway restart" contract:
//
//   - Content-address integrity: a span's id IS the hash of its bytes. A caller-asserted id
//     that disagrees with the content hash is REFUSED at Put (ErrDigestMismatch), and a file
//     whose on-disk bytes no longer hash to its own name is REFUSED at read (the same
//     sentinel) — a tampered or truncated durable record can never masquerade as the span it
//     is named for. The address is not metadata about the bytes; it is a checkable function
//     OF the bytes.
//   - Persisted trust gate: seal and tombstone are written into the durable record, not held
//     in RAM. A span an operator sealed (trust quarantine) or tombstoned (context control)
//     STAYS refused after a restart — a fresh Store opened over the same dir rebuilds the
//     gate from disk and Materialize still returns the ctxplan sentinel. A restart must not
//     resurrect a suppressed span; if it did, durability would defeat the very suppression
//     that dropped the span.
//
// Writes are crash-safe-ish: each span is written to a temp file in the SAME directory and
// then os.Rename'd into place, so a reader opening the dir mid-write sees either the old
// record or the whole new one, never a half-written file. Reads work from a FRESH Store
// opened over the same dir with the in-memory index rebuilt from disk — that fresh-open is
// exactly the gateway-restart simulation the witness test exercises.
//
// This package imports ONLY the standard library and internal/ctxplan; it deliberately does
// NOT import internal/gateway or internal/session (both volatile), so it builds and tests as
// a self-contained leaf.
package durablestore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// fileExt is the suffix of a durable span record. Open loads only files with this
// extension, so the temp files a concurrent Put creates (tmpPrefix, below) are never
// mistaken for spans mid-write.
const fileExt = ".json"

// tmpPrefix names the temp file a write lands in before its atomic rename. It is dot-led and
// carries no fileExt so a reader walking the dir (Open) skips it: a reader sees either the
// prior <id>.json or the fully renamed new one, never the partial temp.
const tmpPrefix = ".durablestore-tmp-"

// ErrNotFound is returned by Get/Materialize when no durable record addresses the requested
// id. Distinct from a trust-gate refusal so a caller can tell "never persisted / evicted"
// from "persisted, the gate held" — the same never-had-it vs gate-held split the gateway
// stash draws with ErrRestoreMiss vs ErrRestoreRefused.
var ErrNotFound = errors.New("durablestore: no durable span for that id")

// ErrDigestMismatch is the content-address integrity refusal: the bytes do not hash to the
// id they are stored (or asserted) under. It fires at Put when a caller supplies an id that
// disagrees with Digest(bytes), and at read when a durable file's on-disk bytes no longer
// hash to its own filename (a tampered or truncated record). A content-address is a checkable
// function of the bytes, never a trusted label — this is the check that keeps it honest.
var ErrDigestMismatch = errors.New("durablestore: content-address does not match bytes")

// PutSpan is the durable write input: the payload bytes plus the SAFE span metadata that
// mirrors gateway.restoreEntry ({ id, excerpt, bytes, sealed, tombstoned, cluster, kind }).
// It is a distinct type from ctxplan.Span because ctxplan.Span carries only the byte-free
// SIZE (Bytes int64), never the payload — the payload rides here and is content-addressed
// into the store. ID is OPTIONAL: leave it empty to let Put assign the content-address, or
// set it to assert a content-address Put will VERIFY against Digest(Bytes) (a mismatch is
// ErrDigestMismatch). Descriptor is the safe excerpt/orientation line; the evidence edges and
// gate flags map straight onto the ctxplan.Span projection Spans returns.
type PutSpan struct {
	ID              string // OPTIONAL asserted content-address; if set, must equal Digest(Bytes)
	Descriptor      string // the safe excerpt / orientation line (gateway restoreEntry.excerpt)
	Bytes           []byte // the payload persisted durably and paged back by Get/Materialize
	Role            string // the producer (tool name, "user", "system"); optional
	Durability      string // ctxplan durability class; normalized, defaults to "durable" if empty
	Sealed          bool   // quarantined by the trust gate — persisted, refused on read
	Tombstoned      bool   // suppressed by context control — persisted, refused on read
	EvidenceCluster string // gateway restoreEntry.cluster → ctxplan.Span.EvidenceCluster
	EvidenceKind    string // gateway restoreEntry.kind    → ctxplan.Span.EvidenceKind
}

// persisted is the on-disk record: the byte-free ctxplan.Span projection (what Spans hands a
// caller) alongside the verbatim payload Body. Body is base64-encoded by encoding/json's
// []byte handling, so the record is a single self-describing JSON object. The content-address
// (Span.ID == Span.Digest == Digest(Body)) is stored redundantly with the filename so a load
// can cross-check name against content without trusting either alone.
type persisted struct {
	Span ctxplan.Span `json:"span"`
	Body []byte       `json:"body"`
}

// Store is a durable, content-addressed span store backed by one directory of <id>.json
// records. It implements ctxplan.Store (Spans + Materialize), so it drops directly into the
// gateway's restoreFromStore fall-through. The in-memory index is a hot cache of the SAFE
// byte-free metadata (rebuilt from disk on Open); payload bytes always come from disk, so a
// fresh Store over the same dir — a gateway restart — serves the same spans.
type Store struct {
	dir string

	mu    sync.RWMutex
	index map[string]ctxplan.Span // id → safe metadata (byte-free hot cache over the disk records)
}

// Open returns a Store backed by dir, creating the directory if absent and rebuilding the
// in-memory index from the durable records already on disk. The rebuild is the restart-
// survival path: every <id>.json is loaded and its content-address re-verified (a record
// whose bytes no longer hash to its filename is ErrDigestMismatch — a corrupt durable store
// fails closed rather than serving a span under the wrong address). A blank dir is refused;
// anything else present in the dir that is not a span record (temp files, unrelated files) is
// ignored.
func Open(dir string) (*Store, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, errors.New("durablestore: dir is required")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("durablestore: create dir: %w", err)
	}
	s := &Store{dir: dir, index: map[string]ctxplan.Span{}}
	if err := s.reindex(); err != nil {
		return nil, err
	}
	return s, nil
}

// reindex walks the backing dir and rebuilds the in-memory index from the durable records —
// the operation that makes a fresh Store over an existing dir replay the same spans (the
// gateway-restart witness). Only <id>.json files are considered; each is verified so a
// tampered or truncated record cannot silently enter the index under a wrong address.
func (s *Store) reindex() error {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return fmt.Errorf("durablestore: read dir: %w", err)
	}
	next := make(map[string]ctxplan.Span, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, fileExt) || strings.HasPrefix(name, tmpPrefix) {
			continue
		}
		id := strings.TrimSuffix(name, fileExt)
		rec, err := s.load(id)
		if err != nil {
			return err
		}
		next[id] = rec.Span
	}
	s.mu.Lock()
	s.index = next
	s.mu.Unlock()
	return nil
}

// path is the durable file for a content-address.
func (s *Store) path(id string) string { return filepath.Join(s.dir, id+fileExt) }

// load reads and integrity-checks one durable record: the on-disk bytes MUST hash to the id
// the file is named for and to the id the record carries, otherwise it is ErrDigestMismatch.
// This is the content-address contract enforced at read time — the address is re-derived from
// the bytes, never trusted from the name.
func (s *Store) load(id string) (persisted, error) {
	raw, err := os.ReadFile(s.path(id))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return persisted{}, fmt.Errorf("%w: %s", ErrNotFound, id)
		}
		return persisted{}, fmt.Errorf("durablestore: read %s: %w", id, err)
	}
	var rec persisted
	if err := json.Unmarshal(raw, &rec); err != nil {
		return persisted{}, fmt.Errorf("durablestore: decode %s: %w", id, err)
	}
	got := ctxplan.Digest(rec.Body)
	if got != id || rec.Span.ID != id || rec.Span.Digest != id {
		return persisted{}, fmt.Errorf("%w: file %q holds bytes hashing to %s", ErrDigestMismatch, id, got)
	}
	return rec, nil
}

// Put persists a span durably under its content-address and returns that address. The id is
// Digest(Bytes) — if the caller asserted a non-empty PutSpan.ID that disagrees, Put refuses
// with ErrDigestMismatch (content-address integrity: you cannot store bytes under an address
// that is not their hash). The write is crash-safe-ish: it lands in a temp file in the same
// dir and is os.Rename'd into place, so a concurrent reader sees the whole record or none.
// Re-Putting identical bytes is idempotent (same digest ⇒ same filename); a re-Put that
// flips the gate flags rewrites the record with the new gate.
func (s *Store) Put(in PutSpan) (string, error) {
	if len(in.Bytes) == 0 {
		return "", errors.New("durablestore: Put requires non-empty Bytes")
	}
	id := ctxplan.Digest(in.Bytes)
	if asserted := strings.TrimSpace(in.ID); asserted != "" && asserted != id {
		return "", fmt.Errorf("%w: asserted %s but bytes hash to %s", ErrDigestMismatch, asserted, id)
	}
	span := ctxplan.Span{
		ID:              id,
		Descriptor:      in.Descriptor,
		Digest:          id,
		Bytes:           int64(len(in.Bytes)),
		Role:            in.Role,
		Durability:      ctxplan.NormDurability(orDurable(in.Durability)),
		Sealed:          in.Sealed,
		Tombstoned:      in.Tombstoned,
		EvidenceCluster: in.EvidenceCluster,
		EvidenceKind:    in.EvidenceKind,
	}
	rec := persisted{Span: span, Body: append([]byte(nil), in.Bytes...)}
	if err := s.writeRecord(id, rec); err != nil {
		return "", err
	}
	s.mu.Lock()
	s.index[id] = span
	s.mu.Unlock()
	return id, nil
}

// writeRecord marshals a record and lands it atomically: temp file in the same dir, fsync-ish
// flush via Close, then os.Rename over the final name (MoveFileEx-replace on Windows, rename
// on POSIX). The temp is created in s.dir so the rename stays on one filesystem and is atomic.
func (s *Store) writeRecord(id string, rec persisted) error {
	body, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return fmt.Errorf("durablestore: encode %s: %w", id, err)
	}
	tmp, err := os.CreateTemp(s.dir, tmpPrefix+"*"+fileExt)
	if err != nil {
		return fmt.Errorf("durablestore: temp for %s: %w", id, err)
	}
	tmpName := tmp.Name()
	if _, err := tmp.Write(body); err != nil {
		tmp.Close()
		os.Remove(tmpName)
		return fmt.Errorf("durablestore: write temp for %s: %w", id, err)
	}
	if err := tmp.Close(); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("durablestore: close temp for %s: %w", id, err)
	}
	if err := os.Rename(tmpName, s.path(id)); err != nil {
		os.Remove(tmpName)
		return fmt.Errorf("durablestore: commit %s: %w", id, err)
	}
	return nil
}

// Materialize pages a span's bytes back in through the trust gate — the ctxplan.Store
// contract. It reads the DURABLE record (disk is the source of truth, so the gate flags a
// restart rebuilt from disk are the ones honored), refuses a sealed span with ctxplan.ErrSealed
// and a tombstoned one with ctxplan.ErrTombstoned (the SAME sentinels the gateway maps to its
// ErrRestoreRefused shape), and re-verifies the content-address before returning bytes. An id
// no record addresses is ErrNotFound.
func (s *Store) Materialize(_ context.Context, id string) ([]byte, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return nil, fmt.Errorf("%w: empty id", ErrNotFound)
	}
	rec, err := s.load(id)
	if err != nil {
		return nil, err
	}
	if rec.Span.Sealed {
		return nil, fmt.Errorf("%w: span %s", ctxplan.ErrSealed, id)
	}
	if rec.Span.Tombstoned {
		return nil, fmt.Errorf("%w: span %s", ctxplan.ErrTombstoned, id)
	}
	return append([]byte(nil), rec.Body...), nil
}

// Get is a convenience alias for Materialize with a background context — the plain read verb
// the witness test and non-ctxplan callers use.
func (s *Store) Get(id string) ([]byte, error) {
	return s.Materialize(context.Background(), id)
}

// Spans returns the SAFE, byte-free projection of every durable span (ctxplan.Store
// contract), sorted by content-address for a deterministic replay. The ctxplan.Span shape
// carries only the size (Bytes int64), never the payload — so this enumeration structurally
// cannot leak a sealed span's bytes, exactly as fak_context_spans requires. The snapshot
// comes from the in-memory index (the hot cache) rebuilt from disk on Open.
func (s *Store) Spans(_ context.Context) ([]ctxplan.Span, error) {
	s.mu.RLock()
	out := make([]ctxplan.Span, 0, len(s.index))
	for _, sp := range s.index {
		out = append(out, sp)
	}
	s.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out, nil
}

// Seal flips and PERSISTS the seal gate on a span: an operator trust-quarantine that must
// survive a restart. It rewrites the durable record atomically and updates the index, so a
// fresh Store over the same dir refuses the span with ctxplan.ErrSealed — a suppressed span is
// never resurrected by a restart.
func (s *Store) Seal(id string) error { return s.setGate(id, gateSeal) }

// Tombstone flips and PERSISTS the tombstone gate (context-control suppression), with the same
// restart-durable guarantee as Seal — Materialize then refuses with ctxplan.ErrTombstoned.
func (s *Store) Tombstone(id string) error { return s.setGate(id, gateTombstone) }

type gate int

const (
	gateSeal gate = iota
	gateTombstone
)

// setGate re-reads the durable record, flips the requested gate, and rewrites it atomically —
// keeping disk (the source of truth across a restart) and the in-memory index in step.
func (s *Store) setGate(id string, g gate) error {
	id = strings.TrimSpace(id)
	if id == "" {
		return fmt.Errorf("%w: empty id", ErrNotFound)
	}
	rec, err := s.load(id)
	if err != nil {
		return err
	}
	switch g {
	case gateSeal:
		rec.Span.Sealed = true
	case gateTombstone:
		rec.Span.Tombstoned = true
	}
	if err := s.writeRecord(id, rec); err != nil {
		return err
	}
	s.mu.Lock()
	s.index[id] = rec.Span
	s.mu.Unlock()
	return nil
}

// orDurable defaults an empty durability class to "durable": a span written to the durable
// store is, by construction, durable-class unless the caller says otherwise.
func orDurable(s string) string {
	if strings.TrimSpace(s) == "" {
		return ctxplan.DurabilityDurable
	}
	return s
}

// compile-time proof the durable Store satisfies the ctxplan.Store seam the gateway restore
// fall-through resolves against — the whole point of composing with ctxplan rather than
// duplicating a second content-address scheme.
var _ ctxplan.Store = (*Store)(nil)
