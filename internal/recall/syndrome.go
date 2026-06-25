package recall

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
)

// ECC-STYLE MEMORY INTEGRITY (#783, #785). The CAS already self-verifies its
// BODY bytes — a persisted blob must hash to its key or LoadSession refuses the
// whole image (recall.go, "corrupt CAS entry"). But the page table itself carries
// security-critical METADATA the body digest does not protect: Taint, Quarantined,
// QID, Digest, Len. A tampered or bit-rotted manifest that flips Quarantined
// true->false, or repoints Digest at a benign blob, sails past the body check —
// the exact silent fault ECC exists to catch.
//
// Syndrome closes that gap with a per-page check value over the integrity-critical
// metadata, computed at persist time and re-derivable at load time. It is NOT a
// secret-keyed MAC (recall makes no confidentiality claim about a local image on
// the operator's own disk); it is a corruption/tamper-evidence syndrome, the
// memory-integrity analogue of an ECC syndrome word. Default-neutral: the field is
// omitempty, so a page persisted before this rung — or by a writer that does not
// stamp it — is byte-identical in manifest.json to today and classifies as
// FaultUnchecked (no syndrome to check), never as a false fault.

// Syndrome is the per-page integrity check value: hex(sha256(canonical metadata)),
// truncated to a fixed width. Stored on Page.Syndrome, recomputed by computeSyndrome.
// A mismatch between the stored and recomputed value is a metadata-corruption fault.
const syndromeWidth = 16 // 8 bytes hex — ample collision resistance for tamper-evidence

// computeSyndrome binds a page's integrity-critical metadata into one check value.
// Only the fields whose silent flip would change a TRUST or LOOKUP decision are
// folded in: Digest (which body), Len (truncation), Taint + Quarantined + QID (the
// poison-release surface). Descriptor/Reason/Utility are presentation/learning
// signals and are deliberately excluded so a benign descriptor repair (the dream
// path) does not invalidate the syndrome.
func computeSyndrome(p Page) string {
	h := sha256.New()
	// length-prefixed fields so concatenation is unambiguous (no "ab"+"c" == "a"+"bc").
	writeField := func(b []byte) {
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(len(b)))
		h.Write(n[:])
		h.Write(b)
	}
	writeField([]byte(p.Digest))
	var ln [8]byte
	binary.LittleEndian.PutUint64(ln[:], uint64(p.Len))
	writeField(ln[:])
	writeField([]byte{p.Taint})
	var q byte
	if p.Quarantined {
		q = 1
	}
	writeField([]byte{q})
	writeField([]byte(p.QID))
	sum := h.Sum(nil)
	return hex.EncodeToString(sum)[:syndromeWidth]
}

// stampSyndrome returns a copy of p with Syndrome set to its computed value. Used
// at record/persist time so a freshly written page carries its check word.
func stampSyndrome(p Page) Page {
	p.Syndrome = computeSyndrome(p)
	return p
}

// FaultClass is the #785 classification of a page's integrity state. The split is
// the ECC repairable-vs-erasure distinction applied to memory: a metadata
// corruption is REPAIRABLE because the authoritative bytes still exist in CAS and
// the true metadata can be re-derived from them; a missing body is an ERASURE
// because no amount of local recomputation recovers bytes that are simply gone —
// it needs a replica or witness, not a repair.
type FaultClass uint8

const (
	// FaultUnchecked: the page carries no syndrome (pre-rung or unstamped). Honest
	// absence of evidence — NOT a clean bill of health and NOT a fault.
	FaultUnchecked FaultClass = iota
	// FaultClean: syndrome matches and the body is present in CAS.
	FaultClean
	// FaultRepairable: the body bytes are present (and self-consistent against their
	// own digest), but the page's stored syndrome disagrees with its metadata — the
	// page table was corrupted/tampered. Recoverable: re-derive metadata from the
	// authoritative body and re-stamp.
	FaultRepairable
	// FaultErasure: the page's body bytes are absent from CAS (or do not hash to the
	// recorded Digest). Unrecoverable locally — the data itself is gone; needs a
	// replica or an external witness.
	FaultErasure
)

// String renders the class for logs and reports.
func (f FaultClass) String() string {
	switch f {
	case FaultUnchecked:
		return "unchecked"
	case FaultClean:
		return "clean"
	case FaultRepairable:
		return "repairable"
	case FaultErasure:
		return "erasure"
	}
	return "unknown"
}

// Repairable reports whether the fault can be fixed without an external replica.
func (f FaultClass) Repairable() bool { return f == FaultRepairable }

// ClassifyFault decides a page's integrity class against its body bytes. body is
// the CAS-resolved bytes for p.Digest, or nil if the lookup missed. The order of
// checks is load-bearing: an erasure (missing/wrong body) dominates a metadata
// mismatch, because you cannot trust the syndrome's verdict about a page whose
// authoritative bytes you cannot read.
func ClassifyFault(p Page, body []byte) FaultClass {
	// Erasure first: no body, or a body that does not hash to the recorded address,
	// means the authoritative source is gone — nothing local repairs it.
	if body == nil || Digest(body) != p.Digest {
		// A page with no syndrome AND a present, matching body is the only way past
		// here; an absent body is always an erasure regardless of syndrome presence.
		if body == nil {
			return FaultErasure
		}
		// body present but digest mismatch: the CAS gave us the wrong/rotted bytes.
		return FaultErasure
	}
	// Body is present and self-consistent. Now the metadata syndrome.
	if p.Syndrome == "" {
		return FaultUnchecked
	}
	if p.Syndrome == computeSyndrome(p) {
		return FaultClean
	}
	// Body authoritative, metadata disagrees with its check word: repairable.
	return FaultRepairable
}

// PageFault pairs a page index with its classification — the unit a scrub/verify
// pass reports and a repair pass consumes.
type PageFault struct {
	Step  int        `json:"step"`
	Class FaultClass `json:"class"`
}

// Verify scans a loaded session's pages and returns every page whose class is not
// FaultClean (faults AND unchecked pages), so an operator or a patrol-scrub pass
// (#784) sees exactly what is suspect without re-reading the whole table. It is
// read-only: it classifies, it does not repair. A nil/empty result means every
// stamped page verified clean.
func (s *Session) Verify() []PageFault {
	var faults []PageFault
	for i, p := range s.Manifest.Pages {
		body := s.cas[p.Digest] // nil if absent — ClassifyFault treats that as erasure
		if c := ClassifyFault(p, body); c != FaultClean {
			faults = append(faults, PageFault{Step: i, Class: c})
		}
	}
	return faults
}
