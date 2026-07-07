package cachemeta

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
)

// sysprompt_fingerprint.go — issue #1563 (cache-default[45], epic #1490 "O(1)
// context"): tie the system-prompt overlay's identity into the prefix cache
// fingerprint so that when the sys-prompt overlay changes, the affected prefix's
// fingerprint changes too and the prefix is INVALIDATED (miss → cold prefill of the
// changed span) rather than silently reused.
//
// The failure this closes. A served prefix carries a resident spine+policy head and,
// appended AFTER the cache breakpoint, a queried system-prompt overlay
// (syspromptmmu.SelectOverlay). Because the overlay lands past the breakpoint, an
// overlay edit leaves the resident bytes byte-identical (syspromptmmu.SpliceSystemOverlay
// copies the resident prefix verbatim). So a reuse key derived from the RESIDENT span
// alone cannot see an overlay change: under a default-on reuse gate (#1490) it would
// treat a prefix produced under overlay A as warm for a turn now running overlay B —
// the exact "silently poisoning reuse" this item names. Folding the active overlay's
// identity into the fingerprint makes that change observable, so the reuse decision
// misses instead of poisoning.
//
// Scope fence. This changes WHAT invalidates a prefix (the fingerprint the reuse
// decision keys on); it does not add a serving path, warm/evict actively, or re-score
// context value. It composes with the §A3/§A4 content-and-witness machinery
// (prefix_stability.go / prefix_coherence.go): those break a prefix on a resident-span
// byte change or a revoked witness; this catches the orthogonal case where the resident
// span is byte-identical but the after-breakpoint overlay identity moved. cachemeta stays
// a pure, wire-neutral consumer — it takes PromptSegments, never imports syspromptmmu.

// segmentIdentityDigest hashes an ordered segment list by (Kind, Content, Witness),
// NUL-separated so no field concatenation aliases another and a reorder is visible. The
// Witness is folded in so a revoked/rotated capability body (whose overlay segment
// carries Witness = card digest) also moves the digest, not only a raw content edit.
func segmentIdentityDigest(segs []PromptSegment) string {
	h := sha256.New()
	for _, s := range segs {
		_, _ = h.Write([]byte(s.Kind))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(strconv.Itoa(len(s.Content))))
		_, _ = h.Write([]byte{0})
		_, _ = h.Write(s.Content)
		_, _ = h.Write([]byte{0})
		_, _ = h.Write([]byte(s.Witness))
		_, _ = h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))
}

// PrefixFingerprint identifies a served prefix by BOTH its resident content AND the
// system-prompt overlay it was produced under. The reuse decision keys on Combined; the
// two component digests are exposed so a caller can tell an overlay-only change (the case
// this file adds) from a resident-span change (already caught by §A3).
type PrefixFingerprint struct {
	// ResidentDigest is the identity of the resident (spine+policy) prefix segments —
	// the span that sits before the cache breakpoint and is copied verbatim across turns.
	ResidentDigest string
	// OverlayDigest is the identity of the active system-prompt overlay segments appended
	// after the breakpoint. Empty overlay ⇒ the digest of an empty segment list (a stable,
	// non-empty sentinel), so "no overlay" is itself a distinct, comparable identity.
	OverlayDigest string
	// Combined is the single authoritative fingerprint the reuse decision compares: a
	// digest folding ResidentDigest and OverlayDigest together. Two prefixes reuse iff
	// their Combined fingerprints are equal.
	Combined string
}

// ComputePrefixFingerprint derives the authoritative prefix fingerprint from the resident
// prefix segments and the active system-prompt overlay segments. It is deterministic:
// the same (resident, overlay) yields the same fingerprint, and a change to EITHER the
// resident bytes or the overlay identity changes Combined. This is the fold that ties the
// sys-prompt overlay to the cache fingerprint (#1563).
func ComputePrefixFingerprint(resident, overlay []PromptSegment) PrefixFingerprint {
	fp := PrefixFingerprint{
		ResidentDigest: segmentIdentityDigest(resident),
		OverlayDigest:  segmentIdentityDigest(overlay),
	}
	h := sha256.New()
	_, _ = h.Write([]byte(fp.ResidentDigest))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(fp.OverlayDigest))
	fp.Combined = hex.EncodeToString(h.Sum(nil))
	return fp
}

// PrefixReuseState is the closed reuse-vs-invalidate verdict for a served prefix under a
// default-on reuse gate.
type PrefixReuseState string

const (
	// PrefixReuse means the current fingerprint matches the cached one: the prefix is warm
	// and safe to serve without a cold prefill.
	PrefixReuse PrefixReuseState = "reuse"
	// PrefixInvalidated means the fingerprint changed: the cached prefix must NOT be
	// reused; the changed span needs a cold prefill (miss).
	PrefixInvalidated PrefixReuseState = "invalidated"
)

// PrefixReuseDecision is the typed answer for whether a cached prefix may be reused for
// the current turn, plus enough provenance to see WHY it was invalidated.
type PrefixReuseDecision struct {
	State PrefixReuseState
	// CachedFingerprint / CurrentFingerprint are the Combined fingerprints compared.
	CachedFingerprint  string
	CurrentFingerprint string
	// OverlayChanged is true when the resident span is byte-identical but the system-prompt
	// overlay identity moved — the precise "silently poisoning reuse" case #1563 fixes: a
	// resident-only key would have called this a hit, but the folded fingerprint invalidates
	// it. False for a resident-span change (already caught by §A3) or a clean reuse.
	OverlayChanged bool
	Reason         string
}

// DecideReuse compares a cached prefix fingerprint (the receiver) against the current
// turn's fingerprint and returns the reuse-vs-invalidate verdict. Equal Combined
// fingerprints reuse; any divergence invalidates, and an overlay-only divergence is
// flagged (OverlayChanged) so the caller can attribute the miss to the sys-prompt overlay
// rather than a resident-span edit.
func (cached PrefixFingerprint) DecideReuse(current PrefixFingerprint) PrefixReuseDecision {
	d := PrefixReuseDecision{
		CachedFingerprint:  cached.Combined,
		CurrentFingerprint: current.Combined,
	}
	if cached.Combined == current.Combined {
		d.State = PrefixReuse
		d.Reason = "prefix fingerprint unchanged; overlay and resident span both match — safe reuse"
		return d
	}
	d.State = PrefixInvalidated
	if cached.ResidentDigest == current.ResidentDigest && cached.OverlayDigest != current.OverlayDigest {
		d.OverlayChanged = true
		d.Reason = "system-prompt overlay changed (resident span identical); affected prefix invalidated — cold prefill required, not silent reuse"
		return d
	}
	d.Reason = "resident prefix span changed; affected prefix invalidated — cold prefill required"
	return d
}
