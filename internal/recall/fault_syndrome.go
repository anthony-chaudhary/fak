package recall

import (
	"fmt"
)

// REPAIRABLE-vs-ERASURE FAULT CLASSIFICATION (#785, builds on the #783 PageSyndrome
// keystone and the #784 patrol scrub; epic #782).
//
// ECC memory draws one bright line that the per-page syndrome (#783) and the patrol
// scrub (#784) do not yet name explicitly: a CORRECTABLE error (the check bits let the
// controller reconstruct the true word in place) versus an UNCORRECTABLE error (the
// damage exceeds the code's reach — the line is poisoned and the read must fail). recall
// already detects faults (ClassifyFault → FaultClass; the five-axis FailedEvidence fold;
// the patrol ScrubClass), but a downstream repair/scrub child still has to re-derive, for
// each fault, the one question that decides whether to ACT or to REFUSE:
//
//	can this page be recovered to byte-faithful, trusted context — or is it lost?
//
// classifySyndrome answers exactly that. It is the READ-BACK classifier the scrub/repair
// children build on: a pure function from a page's failed integrity evidence (the #783
// PageSyndrome / FailedEvidence and the existing FaultClass) plus a small recovery oracle
// to a closed Syndrome verdict — REPAIRABLE or ERASURE — with a typed, auditable reason.
// It REUSES the existing detectors wholesale; it adds no new fault detection. Its only new
// idea is the SECOND dimension over the same faults: not "what is wrong" (FaultClass /
// EvidenceAxis already say that) but "is there a recovery path".
//
// The mapping is the project's fail-closed default throughout: a fault is REPAIRABLE only
// when a NARROW, byte-faithful recovery path is KNOWN to exist (a clean CAS replica for a
// digest miss; a re-derivable view; a live witness that re-establishes trust the revocation
// retired). Absent a known path, every fault is ERASURE — quarantine, tombstone, or refuse.
// This is the ECC posture: when in doubt, poison the line rather than hand back a guess.
//
//   - A metadata-only syndrome mismatch over a present, authoritative body (FaultRepairable)
//     is REPAIRABLE: the true metadata re-derives from bytes that are still here — this is
//     the in-place correction, no external help needed.
//   - A digest miss (body absent or rotted, FaultErasure / the digest axis) is ERASURE
//     UNLESS the oracle reports a clean replica of the same content address — then it is
//     REPAIRABLE by re-fetch (still byte-faithful: the replica must hash to the same address).
//   - A revoked witness / stale trust epoch is ERASURE UNLESS the oracle reports trust can be
//     RE-DERIVED (a live replacement witness re-establishes the same source) — then the page's
//     trust is repairable without rewriting a byte. A truly-retired witness with no re-deriv
//     path is uncorrectable: seal it.
//   - A sealed / poison quarantine fault is ERASURE, always: recall must never "repair" poison
//     into trusted text (an epic #782 non-goal). Fail closed — even when the page's metadata
//     syndrome is otherwise clean, an uncleared seal blocks reuse.
//
// The DURABILITY axis (#82) is deliberately OUT of scope here: a turn/session page that is not
// durable enough to re-enter a NEW context is a CURRENCY/reusability fact, not a corruption
// fault — its bytes are intact and its trust is live. Folding it into the erasure bucket would
// mislabel every healthy turn-scoped page as erased. Whether a healthy page may cross into a new
// context is the #784 reusability view's question (ScrubSyndrome / PageReusability), which
// already reports the durability axis; this classifier answers the orthogonal "is the cell
// corrupt, and if so can it be recovered" question.
//
// The classifier never rewrites page contents and never derives trusted bytes from anything
// but the page's own authoritative body or a hash-identical replica — the acceptance floor of
// #785: correctable cases preserve byte-faithful payloads or re-derive metadata only from
// trusted existing bytes; uncorrectable cases refuse with a typed, auditable reason.

// SyndromeClass is the #785 repairable-vs-erasure verdict for a memory page: the ECC
// correctable / uncorrectable distinction made explicit. It is a SECOND, orthogonal
// dimension over the existing FaultClass — FaultClass names WHAT failed, SyndromeClass
// names whether there is a recovery path — so the two compose: a fault carries both.
type SyndromeClass uint8

const (
	// SyndromeHealthy: no fault to classify — the page's evidence is intact (FaultClean or
	// the honest-absence FaultUnchecked). Not a repair and not an erasure; nothing to do.
	SyndromeHealthy SyndromeClass = iota
	// SyndromeRepairable: a CORRECTABLE fault — a narrow, byte-faithful recovery path is
	// known to exist (re-derive metadata from the present authoritative body; re-fetch a
	// clean replica that hashes to the same address; re-establish retired trust from a live
	// replacement witness). The page can become byte-faithful, trusted context again.
	SyndromeRepairable
	// SyndromeErasure: an UNCORRECTABLE fault — the body is gone or tampered with no known
	// recovery, or the only "fix" would launder poison / stale value into trusted text. The
	// page must be quarantined, tombstoned, or refused page-in. Fail closed.
	SyndromeErasure
)

// String renders the class for findings, logs, and audit rows.
func (c SyndromeClass) String() string {
	switch c {
	case SyndromeHealthy:
		return "healthy"
	case SyndromeRepairable:
		return "repairable"
	case SyndromeErasure:
		return "erasure"
	}
	return "unknown"
}

// Repairable reports whether the verdict is the correctable (in-place / replica) case.
func (c SyndromeClass) Repairable() bool { return c == SyndromeRepairable }

// Erasure reports whether the verdict is the uncorrectable (seal / tombstone / refuse) case.
func (c SyndromeClass) Erasure() bool { return c == SyndromeErasure }

// Recovery names WHY a fault is correctable — the specific narrow, byte-faithful path the
// classifier found. It is the closed, auditable reason a repair pass acts on; an erasure
// carries RecoveryNone. The set is deliberately small: every member is a path that preserves
// the original payload's bytes or derives metadata only from trusted existing bytes.
type Recovery uint8

const (
	// RecoveryNone: no recovery path — either the page is healthy (nothing to recover) or the
	// fault is an erasure (nothing recovers it).
	RecoveryNone Recovery = iota
	// RecoveryMetadataRederive: the body is present and authoritative; the corrupted metadata
	// re-derives from those trusted bytes (the FaultRepairable / metadata-only case). The
	// in-place ECC correction — no external help, no byte of payload rewritten.
	RecoveryMetadataRederive
	// RecoveryReplicaRefetch: the body is missing/rotted at this address but a CLEAN replica of
	// the SAME content address exists (per the oracle) and hashes back to it — re-fetch is
	// byte-faithful by construction (a replica that did not hash to the address would not be a
	// replica). This is the "a clean replica exists" digest-mismatch repair from #785.
	RecoveryReplicaRefetch
	// RecoveryWitnessRederive: a witness/trust-epoch fault whose trust can be RE-DERIVED — a live
	// replacement witness re-establishes the same source the revocation retired, so the page's
	// trust is restored without touching a byte of its body. The "revoked witness (re-derivable
	// trust)" case from #785, distinguished from a truly-erased cell.
	RecoveryWitnessRederive
)

// String renders the recovery path for findings and logs.
func (r Recovery) String() string {
	switch r {
	case RecoveryNone:
		return "none"
	case RecoveryMetadataRederive:
		return "metadata_rederive"
	case RecoveryReplicaRefetch:
		return "replica_refetch"
	case RecoveryWitnessRederive:
		return "witness_rederive"
	}
	return "unknown"
}

// FaultSyndrome is the #785 read-back classification for one page: the repairable-vs-erasure
// verdict, the named recovery path (RecoveryNone for an erasure), the underlying FaultClass it
// folded, the first blocking evidence axis (#783) when a syndrome fault drove the verdict, and
// a short human reason. It carries no page bytes — it is an audit-shaped row, the classifier
// twin of ScrubFinding (scrub.go) and PageReusability (scrub_syndrome.go).
type FaultSyndrome struct {
	Step     int           `json:"step"`
	Class    SyndromeClass `json:"class"`
	Recovery Recovery      `json:"recovery"`
	Fault    FaultClass    `json:"fault"`            // the underlying syndrome/CAS class (syndrome.go) that was folded
	Axis     EvidenceAxis  `json:"axis,omitempty"`   // first blocking #783 evidence axis, when a syndrome fault drove the verdict
	HasAxis  bool          `json:"has_axis"`         // whether Axis is meaningful (a syndrome-axis fault, not a metadata/CAS-only fault)
	Reason   string        `json:"reason,omitempty"` // closed, auditable reason for the verdict
}

// RepairOracle reports the narrow recovery facts the classifier needs to tell a CORRECTABLE
// fault from an ERASURE. It is the ONLY way a digest miss or a revoked witness becomes
// repairable — without an affirmative recovery fact, the classifier fails closed to erasure.
// It is small and pure by design so a repair pass supplies real knowledge (a replica index, a
// live-witness ledger) and a test can pin each branch deterministically.
type RepairOracle interface {
	// HasCleanReplica reports whether a clean replica of the content at `digest` exists and
	// hashes back to that address — the precondition for a byte-faithful replica re-fetch. An
	// implementation MUST verify the replica's bytes hash to `digest`; a replica that does not
	// is not a replica and must return false (the byte-faithfulness floor of #785).
	HasCleanReplica(digest string) bool
	// TrustRederivable reports whether the trust a revocation retired can be RE-DERIVED — a live
	// replacement witness re-establishes the same source `witness` named. False means the witness
	// is truly retired with no path back, which is an erasure of trust.
	TrustRederivable(witness string) bool
}

// noRecovery is the fail-closed default oracle: it knows of NO replica and NO re-derivable
// trust, so every digest miss and every revoked witness classifies as ERASURE. classifySyndrome
// uses it when the caller supplies no oracle, making "no knowledge" mean "no recovery" — the
// project's fail-closed posture, never a free pass.
type noRecovery struct{}

func (noRecovery) HasCleanReplica(string) bool  { return false }
func (noRecovery) TrustRederivable(string) bool { return false }

// classifyTrustAxis folds a blocking witness / trust-epoch axis into the verdict on the
// pre-populated `out`: REPAIRABLE via a byte-faithful witness re-derive when the oracle
// reports the retired trust is re-derivable, else an uncorrectable trust ERASURE. axis is
// the blocking evidence axis and repairReason/eraseReason are its two audit strings. It is
// the shared body of the witness and trust-epoch branches, which folded identically.
func classifyTrustAxis(out FaultSyndrome, axis EvidenceAxis, rederivable bool, repairReason, eraseReason string) FaultSyndrome {
	out.Axis, out.HasAxis = axis, true
	if rederivable {
		out.Class, out.Recovery = SyndromeRepairable, RecoveryWitnessRederive
		out.Reason = repairReason
		return out
	}
	out.Class, out.Recovery = SyndromeErasure, RecoveryNone
	out.Reason = eraseReason
	return out
}

// classifySyndrome maps a page's failed integrity evidence to the repairable-vs-erasure
// dimension. It is a PURE read: it computes the #783 syndrome and the existing FaultClass, then
// folds them — together with the recovery oracle — into one closed verdict. It mutates nothing
// (no page, no CAS, no manifest) and rewrites no contents.
//
// body is the CAS-resolved bytes for p.Digest (nil if the lookup missed). oracle supplies the
// recovery facts; pass nil to fail closed (no replica, no re-derivable trust). The order of the
// fold is load-bearing and mirrors scrubPage's security order: an erasure of the body dominates,
// then quarantine (poison must never be "repaired"), then the witness/trust axes, then the
// metadata-only repair, else healthy.
func classifySyndrome(p Page, body []byte, oracle RepairOracle) FaultSyndrome {
	return classifySyndromeCleared(p, body, oracle, false)
}

// classifySyndromeCleared is the clearance-aware core. cleared is true iff the loaded session
// recorded a witness Clear() for this page's QID — in which case the quarantine axis is folded
// as held (the seal does not block reuse), exactly as Session.PageSyndrome /
// quarantineEvidenceCleared do. The clearance is a session-level FACT, not a metadata edit, so
// it is applied by replacing the (clearance-blind) quarantine EVIDENCE — never by flipping the
// page's Quarantined bit, which would corrupt the page's own metadata syndrome (FaultClass) and
// misreport a cleared-clean page as a metadata mismatch.
func classifySyndromeCleared(p Page, body []byte, oracle RepairOracle, cleared bool) FaultSyndrome {
	if oracle == nil {
		oracle = noRecovery{}
	}
	fault := ClassifyFault(p, body)
	// Compute the #783 syndrome with this loaded session's recorded clearance folded into
	// the quarantine axis, mirroring Session.PageSyndrome (shared helper in page_syndrome.go).
	syn := pageSyndromeCleared(p, body, cleared)
	out := FaultSyndrome{Step: p.Step, Fault: fault}

	// 1) DIGEST / ERASURE dominates: the body is absent or no longer hashes to its address.
	//    The authoritative bytes are gone here, so no verdict about anything else can be
	//    trusted. REPAIRABLE only if a clean replica of the SAME address is known to exist —
	//    a byte-faithful re-fetch. Otherwise the cell is erased. (Checked before the
	//    body-present axes because none of them can be trusted without authoritative bytes.)
	if fault == FaultErasure {
		out.Axis, out.HasAxis = EvidenceDigest, true
		if oracle.HasCleanReplica(p.Digest) {
			out.Class, out.Recovery = SyndromeRepairable, RecoveryReplicaRefetch
			out.Reason = "body gone/rotted at this address, but a clean replica of the same content address exists — byte-faithful re-fetch"
			return out
		}
		out.Class, out.Recovery = SyndromeErasure, RecoveryNone
		if body == nil {
			out.Reason = "body absent from CAS and no clean replica known — uncorrectable, needs a replica or witness"
		} else {
			out.Reason = "body no longer hashes to its address and no clean replica known — uncorrectable, needs a replica or witness"
		}
		return out
	}

	// Body present and authoritative from here (fault is clean, unchecked, or the metadata-only
	// FaultRepairable). The bytes are recoverable; before deciding "healthy", a SECURITY axis
	// (#783) can still BLOCK reuse in a way the metadata dimension does not see — a sealed page,
	// or a refuted witness, can be FaultClean yet still not reusable. Those axes are checked here,
	// independent of FaultClass, so a syndrome-clean-but-sealed page is correctly an erasure
	// rather than a false "healthy".

	// 2) QUARANTINE: a sealed page is poison until cleared. recall must NEVER "repair" poison
	//    into trusted text (epic #782 non-goal), so an uncleared seal is an ERASURE for reuse —
	//    fail closed, regardless that the body is present and even if its metadata syndrome is
	//    clean. (A cleared seal does not block reuse; quarantineEvidenceCleared / the session
	//    helper fold that in, so it never reaches here as a blocking axis.)
	if e, ok := syn.EvidenceFor(EvidenceQuarantine); ok && e.blocks() {
		out.Class, out.Recovery = SyndromeErasure, RecoveryNone
		out.Axis, out.HasAxis = EvidenceQuarantine, true
		out.Reason = "page is sealed/poison — must not be repaired into trusted text; quarantine, do not reuse"
		return out
	}

	// 3) WITNESS / TRUST EPOCH: the admitting source was refuted in the revocation ledger. The
	//    body is byte-faithful, but its TRUST is retired. REPAIRABLE only if that trust can be
	//    RE-DERIVED (a live replacement witness re-establishes the same source) — then the page's
	//    trust is restored without rewriting a byte. Otherwise the trust is erased: seal it. The
	//    witness axis is the canonical one; the trust-epoch axis is its temporal companion and is
	//    reported under the same recovery path.
	if e, ok := syn.EvidenceFor(EvidenceWitness); ok && e.blocks() {
		return classifyTrustAxis(out, EvidenceWitness, oracle.TrustRederivable(p.Witness),
			"witness refuted, but trust is re-derivable from a live replacement witness — byte-faithful trust repair",
			"witness refuted and trust is not re-derivable — uncorrectable trust erasure; seal")
	}
	if e, ok := syn.EvidenceFor(EvidenceTrustEpoch); ok && e.blocks() {
		return classifyTrustAxis(out, EvidenceTrustEpoch, oracle.TrustRederivable(p.Witness),
			"trust epoch stale (witness refuted), but trust is re-derivable from a live replacement witness",
			"trust epoch stale and trust is not re-derivable — uncorrectable; seal")
	}

	// NOTE on the DURABILITY axis (#82): #783's FailedEvidence also reports a durability block —
	// a turn/session page is not durable enough to re-enter a NEW context as current truth. That
	// is a CURRENCY/reusability fact, not a CORRUPTION fault: the bytes are intact and the trust is
	// live; the page is simply scoped to its own session. The repairable-vs-erasure classifier is
	// about CORRUPTION/TAMPER/ERASURE faults, so it deliberately does NOT fold durability into the
	// erasure bucket here (that would mislabel every healthy turn-scoped page as erased). Whether a
	// healthy page may CROSS into a new context is the #784 reusability view's question
	// (ScrubSyndrome / PageReusability), which already reports the durability axis; this classifier
	// answers the orthogonal "is the cell corrupt, and if so can it be recovered" question.

	// 4) METADATA-ONLY REPAIR: the body is authoritative, no security axis blocks, and the fault
	//    is the FaultRepairable metadata-only mismatch. The true metadata re-derives from the
	//    present trusted bytes — the in-place ECC correction. No external help, no payload byte
	//    rewritten.
	if fault == FaultRepairable {
		out.Class, out.Recovery = SyndromeRepairable, RecoveryMetadataRederive
		out.Reason = "metadata syndrome mismatch over a present, authoritative body — re-derive metadata from the trusted bytes"
		return out
	}

	// 5) HEALTHY: the body is present and authoritative, no corruption/tamper/erasure axis blocks
	//    reuse, and the metadata syndrome is clean (FaultClean) or honestly absent (FaultUnchecked).
	//    Nothing to repair and nothing erased. (A non-durable page reaches here HEALTHY — its
	//    cross-context currency is the #784 reusability view's concern, not a corruption fault.)
	out.Class, out.Recovery = SyndromeHealthy, RecoveryNone
	out.Reason = "no corruption fault and every trust/integrity axis holds: " + fault.String()
	return out
}

// ClassifyFaultSyndrome is the Session-facing read-back classifier for one loaded page: it
// resolves the page's authoritative body from this session's own CAS and returns the #785
// repairable-vs-erasure verdict under the supplied recovery oracle (nil ⇒ fail closed). Pure:
// it derives the classification and mutates nothing.
func (s *Session) ClassifyFaultSyndrome(step int, oracle RepairOracle) (FaultSyndrome, error) {
	if step < 0 || step >= len(s.Manifest.Pages) {
		return FaultSyndrome{}, fmt.Errorf("recall: no page %d", step)
	}
	p := s.Manifest.Pages[step]
	body := s.cas[p.Digest] // nil if absent — classifySyndrome treats that as an erasure candidate
	// Fold in this session's recorded clearance for the quarantine axis: a sealed-but-cleared page
	// is not an erasure on the quarantine axis (page-in still re-screens its bytes — the second,
	// independent gate this view deliberately does not duplicate).
	fs := classifySyndromeCleared(p, body, oracle, s.cleared[p.QID])
	fs.Step = step
	return fs, nil
}
