package gateway

import "strings"

// ctxrestore_gate.go — the PRODUCTION trust-propagation edge of the restore stash: the wire that
// finally connects an operator's seal/tombstone ACTION to the content-addressed handles ctxrestore.go
// stashed. restoreContext already REFUSES a sealed/tombstoned entry (ErrRestoreRefused wrapping the
// ctxplan sentinel) and contextSpans already reports it Restorable=false — but only once the flag is
// set. Until this file, nothing in the request path SET those flags: restoreEntry.sealed /
// restoreEntry.tombstoned were flipped only by unit tests, so an operator who tombstoned a span never
// caused its restore handle to be refused. That is the gap this file closes. The refusal side is
// unchanged; this is the SETTER side.
//
// The design axis (Law A2 — every value carries its owner, and a suppression is a value): a restore
// handle's id IS a content-address (the sha256-hex ctxplan.Digest scheme). A content-address names
// BYTES, not a location, so the SAME dropped span can be stashed under more than one trace — a
// head-anchored session that re-tombstoned after a window rewrite, two traces that dropped the
// identical originating task. Suppressing a span therefore cannot be a per-trace act: an operator who
// seals or tombstones a digest must suppress it EVERYWHERE it is recoverable, or the handle leaks the
// span back in under whichever trace the operator did not name. So propagation is deliberately
// trace-agnostic: we scan every trace's stash and flip the flag on every entry that shares the
// digest. This is the same reasoning ctxplan.Store applies to a span id, lifted to the gateway's
// per-trace stash.
//
// Fail-open on a miss, never on an error: a digest that addresses nothing stashed (a fresh session, an
// evicted handle, a tombstone of a page that was never a compaction span) flips zero entries and is a
// VALID no-op — not an error. Suppression must be idempotent and safe to fire unconditionally from the
// operator path, because the operator action (contextChange) is authoritative about the recall image
// whether or not a matching restore handle happens to be live.

// restoreGate names which trust flag an operator action flips on a matching stashed span: a TOMBSTONE
// (context control — recall.ContextActionTombstone, the shipped operator action) or a SEAL (trust
// quarantine). Keeping the two as one closed enum lets gateRestoreByDigest share a single locked scan,
// while restoreContext keeps its two DISTINCT refusal sentinels (ctxplan.ErrTombstoned vs
// ctxplan.ErrSealed) so a caller can still branch on which gate held. The gates are not mutually
// exclusive on an entry: an already-tombstoned span can also be sealed, and restoreContext checks
// sealed first — both still refuse.
type restoreGate int

const (
	gateTombstone restoreGate = iota // flip restoreEntry.tombstoned (context control)
	gateSeal                         // flip restoreEntry.sealed (trust quarantine)
)

// gateRestoreByDigest is the content-addressed trust-propagation primitive: under one
// Server.ctxRestoreMu hold it walks EVERY trace's stash and flips the requested gate on every
// restoreEntry whose id == digest, returning how many stashed handles now carry that suppression.
// It is trace-agnostic BY DESIGN (see the file header): a digest is a content-address, so the same
// span may be recoverable under several traces and suppressing it must suppress it in all of them.
//
// The returned count is the number of stashed handles now suppressed under this content-address, NOT
// a false→true transition delta: it is what the operator log line wants ("suppressed N restore
// handles on the wire") and it makes the no-op case legible — 0 means nothing was stashed under this
// digest yet, a valid no-op and not an error. Idempotent: flipping an already-set flag leaves it set,
// so a second call over the same digest re-reports the same count and never un-suppresses a span.
//
// A nil server, or a blank/whitespace digest, is a safe no-op (0) — a content-address that names no
// bytes can suppress nothing. The digest is trimmed so an operator action carrying incidental
// whitespace still matches the exact-string stash key restoreContext resolves on.
func (s *Server) gateRestoreByDigest(digest string, gate restoreGate) int {
	if s == nil {
		return 0
	}
	digest = strings.TrimSpace(digest)
	if digest == "" {
		return 0
	}
	s.ctxRestoreMu.Lock()
	defer s.ctxRestoreMu.Unlock()
	flipped := 0
	// Ranging a nil map is zero iterations, so a server that never stashed anything is a clean no-op.
	for _, sess := range s.ctxRestore {
		if sess == nil {
			continue
		}
		for i := range sess.entries {
			if sess.entries[i].id != digest {
				continue
			}
			switch gate {
			case gateTombstone:
				sess.entries[i].tombstoned = true
			case gateSeal:
				sess.entries[i].sealed = true
			}
			flipped++
		}
	}
	return flipped
}

// tombstoneRestore is the SHIPPED trust-propagation call: an operator context-control tombstone,
// keyed by the suppressed span's content-address digest, flips the tombstoned gate on every matching
// stashed restore handle so restoreContext refuses it (ctxplan.ErrTombstoned) and contextSpans lists
// it Restorable=false. Its one production caller is Server.contextChange, which fires it after the
// recall core image is persisted so the wire suppression cannot outlive the persisted one. Returns the
// number of handles suppressed (0 = the tombstoned digest was never a live restore handle — a no-op).
func (s *Server) tombstoneRestore(digest string) int {
	return s.gateRestoreByDigest(digest, gateTombstone)
}

// sealRestore is the SEAL (trust-quarantine) counterpart of tombstoneRestore: keyed by a span's
// content-address digest, it flips the sealed gate on every matching stashed restore handle so
// restoreContext refuses it (ctxplan.ErrSealed) and contextSpans lists it Restorable=false. Its
// mechanism is fully wired and unit-witnessed; what it lacks TODAY is a production caller.
//
// The honest seam: unlike the tombstone, there is no shipped operator "seal this span by
// content-address" action in the gateway. Seal/quarantine here is currently authored by the recall
// re-screening plane (recall.Session dream/re-screen seals a page whose witness was revoked or whose
// bytes fail a tightened screen — see internal/recall/dream.go and page_syndrome.go's quarantine
// axis), NOT by an operator naming a digest at the gateway edge. When such an operator-seal-by-digest
// action ships (the seal analogue of contextChange), it MUST call s.sealRestore(<span digest>) right
// after it persists the seal — exactly as contextChange calls s.tombstoneRestore(ch.Digest) — so a
// quarantined span cannot be paged back in through its restore handle. Exposing the method now keeps
// that future wire a one-line call against a tested primitive rather than a fresh scan-and-lock.
func (s *Server) sealRestore(digest string) int {
	return s.gateRestoreByDigest(digest, gateSeal)
}
