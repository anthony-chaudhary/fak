package recall

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// PER-PAGE MEMORY SYNDROME (#783, keystone of epic #782, builds on #82).
//
// A recall Page already carries strong, INDEPENDENT pieces of check information —
// its content digest, its quarantine state, its write-time durability class, the
// external witness it was admitted under, and the trust epoch observed at that
// moment. What the page table did NOT have until this rung is a single NAMED view
// that gathers those pieces and answers one operational question a scrub/read-back
// pass keeps asking:
//
//	what independent evidence must still hold before this page can become context again?
//
// PageSyndrome is that view. It is the memory-integrity analogue of an ECC syndrome
// word: a read-only roll-up over the page's per-axis evidence, not a new store. It
// COMPOSES the existing Page fields and the live vDSO revocation oracle — it adds no
// field to Page and no byte to the manifest. A page is reusable-as-context iff every
// piece of evidence it REQUIRES still holds; the first failing axis names exactly
// what an operator (or a #784 patrol scrub / #785 fault classifier / #786 parity
// check) must repair, witness, or erase before the bytes may re-enter a model's
// context.
//
// The axes are deliberately the same five #782 names — digest, quarantine,
// durability, witness, trust epoch — and the posture is the project's fail-closed
// default throughout: absent/unknown evidence reads as NOT-held, never as a free
// pass. This keystone gives the scrub/read-back children one place to ask "is this
// page's evidence intact?" instead of re-deriving the fold at every call site.

// EvidenceAxis names one independent piece of per-page integrity evidence. The set
// is closed and ordered so a syndrome's failed axes render deterministically.
type EvidenceAxis uint8

const (
	// EvidenceDigest: the page's bytes are present in the CAS and hash to the
	// recorded Digest. This is the "is the body even here, and is it the body we
	// addressed?" axis — an absent or rotted blob fails it (the #785 erasure case).
	EvidenceDigest EvidenceAxis = iota
	// EvidenceQuarantine: the page is not sealed, OR a witness clearance has been
	// recorded for its quarantine id. A still-sealed page is not reusable as context
	// until cleared (and, at page-in, re-screened — that second gate lives in Resolve,
	// not in this metadata view).
	EvidenceQuarantine
	// EvidenceDurability: the page's write-time class is durable enough to survive the
	// process boundary into a NEW context. This is the rung-1 default-expire inversion
	// (#82): a turn/session/unknown page was never durable enough to be re-served as a
	// timeless fact, so it fails this axis.
	EvidenceDurability
	// EvidenceWitness: the external trust witness the page was admitted under has not
	// been refuted in the vDSO revocation ledger. An unwitnessed page trivially holds
	// this axis (it makes no witness claim to refute).
	EvidenceWitness
	// EvidenceTrustEpoch: the integrity clock the page was admitted under is still
	// current for its witness — the recorded trust epoch has not been overtaken by a
	// refutation that retired the witness. A witnessed page whose witness is revoked
	// has a STALE trust epoch and fails this axis (the temporal companion to the
	// witness axis: witness names "who", trust epoch names "as-of when").
	EvidenceTrustEpoch
)

// String renders the axis for findings and logs.
func (a EvidenceAxis) String() string {
	switch a {
	case EvidenceDigest:
		return "digest"
	case EvidenceQuarantine:
		return "quarantine"
	case EvidenceDurability:
		return "durability"
	case EvidenceWitness:
		return "witness"
	case EvidenceTrustEpoch:
		return "trust_epoch"
	}
	return "unknown"
}

// Evidence is one axis's verdict: whether the page REQUIRES this piece of evidence,
// whether it currently HOLDS, and a short human reason when it does not. An axis a
// page does not require (Required=false) never blocks reuse — e.g. an unwitnessed
// page does not require the witness axis — but it is still reported so the view is a
// complete picture, not just the failures.
type Evidence struct {
	Axis     EvidenceAxis `json:"axis"`
	Required bool         `json:"required"`
	Held     bool         `json:"held"`
	Reason   string       `json:"reason,omitempty"`
}

// blocks reports whether this axis stands between the page and reuse: a required
// piece of evidence that does not currently hold.
func (e Evidence) blocks() bool { return e.Required && !e.Held }

// PageSyndrome is the named, read-only per-page integrity view: the five-axis
// evidence roll-up for one Page. It is computed by PageSyndromeFor and never
// mutates the page — it is pure metadata derivation plus a read of the live vDSO
// revocation oracle.
type PageSyndrome struct {
	Step     int        `json:"step"`
	Digest   string     `json:"digest"` // short content address the page references
	Evidence []Evidence `json:"evidence"`
}

// revocationOracle is the read-only slice of the vDSO the syndrome needs: "has this
// witness been refuted?". Abstracting it keeps PageSyndromeFor a PURE function of its
// inputs and lets a test pin a refuted witness without touching global state.
type revocationOracle interface {
	Revoked(witness string) bool
}

// PageSyndromeFor computes the per-page syndrome against the page's CAS body and the
// live vDSO revocation ledger. body is the bytes the page's Digest resolves to, or
// nil if the lookup missed — exactly what Session.cas[p.Digest] yields. It is the
// caller-facing entry point and the one a Session method (PageSyndrome) wraps.
func PageSyndromeFor(p Page, body []byte) PageSyndrome {
	return pageSyndromeWith(p, body, vdso.Default)
}

// pageSyndromeWith is the oracle-injected core, so the witness/trust-epoch axes can
// be tested against a pinned refutation without mutating the package-global vDSO.
func pageSyndromeWith(p Page, body []byte, oracle revocationOracle) PageSyndrome {
	s := PageSyndrome{Step: p.Step, Digest: short(p.Digest)}
	s.Evidence = []Evidence{
		digestEvidence(p, body),
		quarantineEvidence(p),
		durabilityEvidence(p),
		witnessEvidence(p, oracle),
		trustEpochEvidence(p, oracle),
	}
	return s
}

// digestEvidence: the body must be present and hash to the recorded address. Absent
// or digest-mismatched bytes fail closed — you cannot trust any other axis about a
// page whose authoritative bytes you cannot read.
func digestEvidence(p Page, body []byte) Evidence {
	e := Evidence{Axis: EvidenceDigest, Required: true}
	switch {
	case body == nil:
		e.Reason = "body absent from CAS"
	case Digest(body) != p.Digest:
		e.Reason = fmt.Sprintf("body does not hash to its address (actual=%s)", short(Digest(body)))
	default:
		e.Held = true
	}
	return e
}

// quarantineEvidence: a sealed page requires a recorded witness clearance for its
// QID before its bytes may re-enter context. (The page metadata cannot SEE the
// clearance map — that lives on the loaded Session — so the page-level helper reports
// the requirement; Session.PageSyndrome folds the clearance state in.)
func quarantineEvidence(p Page) Evidence {
	return quarantineEvidenceCleared(p, false)
}

// quarantineEvidenceCleared resolves the quarantine axis with the loaded session's
// clearance knowledge: cleared is true iff a witness Clear() has been recorded for
// this page's QID. A benign page trivially holds the axis; a sealed-but-cleared page
// holds it at the metadata layer (Resolve still re-screens the bytes — the second,
// independent gate this view deliberately does not duplicate).
func quarantineEvidenceCleared(p Page, cleared bool) Evidence {
	e := Evidence{Axis: EvidenceQuarantine, Required: true}
	switch {
	case !p.Quarantined:
		e.Held = true
	case cleared:
		e.Held = true
		e.Reason = "sealed; witness clearance recorded (bytes still re-screened on page-in)"
	default:
		e.Reason = fmt.Sprintf("page sealed (qid=%s); no witness clearance", p.QID)
	}
	return e
}

// durabilityEvidence: only a `durable`-classed page is meant to cross the process
// boundary as reusable context (#82). A turn/session/unknown/empty class fails the
// axis — the default-expire inversion: durability is the earned exception.
func durabilityEvidence(p Page) Evidence {
	e := Evidence{Axis: EvidenceDurability, Required: true}
	if promotionClass(p.Durability) == ctxmmu.DurabilityDurable {
		e.Held = true
		return e
	}
	cls := p.Durability
	if cls == "" {
		cls = "unset"
	}
	e.Reason = fmt.Sprintf("class %q is not durable enough to re-enter a new context", cls)
	return e
}

// witnessEvidence: a witnessed page requires its witness to still be live in the
// vDSO ledger. An unwitnessed page makes no witness claim, so the axis is NOT
// required and holds vacuously.
func witnessEvidence(p Page, oracle revocationOracle) Evidence {
	if p.Witness == "" {
		return Evidence{Axis: EvidenceWitness, Required: false, Held: true,
			Reason: "page carries no witness claim"}
	}
	e := Evidence{Axis: EvidenceWitness, Required: true}
	if oracle.Revoked(p.Witness) {
		e.Reason = fmt.Sprintf("witness %q refuted in the revocation ledger", p.Witness)
		return e
	}
	e.Held = true
	return e
}

// trustEpochEvidence: the temporal companion to the witness axis. A witnessed page
// records the trust epoch it was admitted under; if its witness has since been
// refuted, the recorded epoch is STALE relative to the integrity clock and the
// as-of-then trust no longer holds. An unwitnessed page has no epoch claim, so the
// axis is not required.
func trustEpochEvidence(p Page, oracle revocationOracle) Evidence {
	if p.Witness == "" {
		return Evidence{Axis: EvidenceTrustEpoch, Required: false, Held: true,
			Reason: "page carries no witness claim"}
	}
	e := Evidence{Axis: EvidenceTrustEpoch, Required: true}
	if oracle.Revoked(p.Witness) {
		e.Reason = fmt.Sprintf("recorded trust_epoch=%d is stale: witness %q refuted since admission", p.TrustEpoch, p.Witness)
		return e
	}
	e.Held = true
	return e
}

// Reusable reports whether the page may become context again: true iff every piece
// of evidence the page REQUIRES currently holds. The first blocking axis is the
// witness an operator must address; FailedEvidence enumerates all of them.
func (s PageSyndrome) Reusable() bool {
	for _, e := range s.Evidence {
		if e.blocks() {
			return false
		}
	}
	return true
}

// FailedEvidence returns the required-but-not-held axes, in EvidenceAxis order — the
// exact independent evidence that must still be repaired, witnessed, or erased before
// the page is reusable. Empty iff the syndrome is clean (Reusable() is true).
func (s PageSyndrome) FailedEvidence() []Evidence {
	var out []Evidence
	for _, e := range s.Evidence {
		if e.blocks() {
			out = append(out, e)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Axis < out[j].Axis })
	return out
}

// EvidenceFor returns the verdict for a single axis from a computed syndrome — a
// convenience for a caller that wants to inspect one piece (e.g. "did the durability
// axis hold?") without scanning the slice.
func (s PageSyndrome) EvidenceFor(axis EvidenceAxis) (Evidence, bool) {
	for _, e := range s.Evidence {
		if e.Axis == axis {
			return e, true
		}
	}
	return Evidence{}, false
}

// PageSyndrome computes the per-page integrity syndrome for a loaded page, folding in
// this session's witness-clearance knowledge so a sealed-but-cleared page resolves
// its quarantine axis correctly. It is read-only: it derives the view, it never
// mutates the page table or the CAS. body is resolved from the session's own CAS, so
// the digest axis is checked against the loaded image's actual bytes (nil ⇒ erasure).
func (s *Session) PageSyndrome(step int) (PageSyndrome, error) {
	if step < 0 || step >= len(s.Manifest.Pages) {
		return PageSyndrome{}, fmt.Errorf("recall: no page %d", step)
	}
	p := s.Manifest.Pages[step]
	body := s.cas[p.Digest] // nil if absent — digestEvidence treats that as an erasure
	syn := pageSyndromeWith(p, body, vdso.Default)
	// Replace the page-level (clearance-blind) quarantine axis with one that knows
	// this loaded session's recorded clearances — the only axis whose evidence lives
	// on the Session rather than on the Page itself.
	for i := range syn.Evidence {
		if syn.Evidence[i].Axis == EvidenceQuarantine {
			syn.Evidence[i] = quarantineEvidenceCleared(p, s.cleared[p.QID])
		}
	}
	return syn, nil
}
