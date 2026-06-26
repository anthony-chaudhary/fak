package recall

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// PATROL SCRUB (#784, parent #782). ECC memory does not wait for a read fault to
// discover corruption — it patrol-scrubs in the background. recall has the same need:
// a persisted core image can rot or be tampered with between the session that wrote it
// and a future session that relies on it, and three independent things can go stale
// without any single page-in noticing:
//
//  1. the per-page metadata SYNDROME (#783/#785) — a silent flip of an integrity field
//     (Quarantined, Len, Digest, Taint, QID) the body digest does not cover;
//  2. the witness LEDGER — a source trusted at persist time, refuted since;
//  3. the content GATE — a payload benign to the write-time detectors that today's
//     de-obfuscating gate (canon.Scan + the registered ResultAdmitter chain) catches.
//
// Scrub is the off-path pass that re-derives all three under TODAY's gates and reports
// every page by its integrity class, then seals/tombstones what now fails. It is the ECC
// sibling of Dream (dream.go): Dream consolidates and reclaims space and REQUIRES a
// Load-clean image; Scrub is deliberately MORE robust than Load — it surveys a
// partially-corrupt image per page rather than fail-closing the whole bundle, which is the
// only way an erasure or a metadata flip becomes a reported finding instead of a refused
// load.
//
// Byte handling is honest about its limits: a healthy blob is preserved byte-for-byte and
// a sealed page keeps its bytes, but a rotted (unhashable) blob CANNOT survive in a
// reloadable output — recall.Load fail-closes on it — so the scrubbed output drops it and
// the audit record fingerprints the actual rotted bytes (their real digest + length) so an
// operator can still identify the loss. The pass never summarizes poison into a trusted
// descriptor and never mints a clean syndrome over tampered security bits.

// ScrubClass is a page's patrol verdict — the action the scrubber took (or would take)
// after folding the syndrome, witness, and content checks. It is BROADER than FaultClass
// (syndrome.go), which sees only the syndrome/CAS dimension: a syndrome-clean page can
// still be sealed for a revoked witness or a tightened gate, and a syndrome-mismatched page
// can be an unrecoverable security tamper rather than a benign metadata rot.
type ScrubClass uint8

const (
	// ScrubClean: syndrome agrees, body present and authoritative, witness live, and the
	// content gate passes. The page is byte-identical after the pass.
	ScrubClean ScrubClass = iota
	// ScrubUnchecked: the page carried no syndrome (a pre-rung or unstamped image). Not a
	// fault — honest absence of evidence. The scrub re-derives the body-derivable field (Len)
	// and stamps a syndrome forward so the NEXT patrol can check it.
	ScrubUnchecked
	// ScrubRepaired: the metadata syndrome disagreed, but re-deriving the one body-derivable
	// integrity field (Len) fully restores the match — the corruption was confined to Len, a
	// benign rot. Re-stamped; CAS bytes untouched.
	ScrubRepaired
	// ScrubTampered: the syndrome disagrees and re-deriving Len does NOT restore it, so a
	// non-body-derivable security field (Taint/Quarantined/QID) was altered. A one-way
	// syndrome cannot recover the pre-tamper value, so the scrub MUST NOT mint a clean
	// syndrome (that would launder the tamper — e.g. a Quarantined true->false unseal or a
	// Taint escalation). Fail-closed: the page is sealed and stays flagged.
	ScrubTampered
	// ScrubErasure: the page's body is absent from CAS, or the bytes under its address no
	// longer hash to it (a flipped-byte blob). Unrecoverable locally — needs a replica or
	// witness, not a repair. The page is tombstoned (page-in suppressed); the actual rotted
	// bytes are fingerprinted in the finding before being dropped from the reloadable output.
	ScrubErasure
	// ScrubRevokedWitness: the page's admitting witness has been refuted in the vDSO ledger
	// since persist. Sealed (removed from model-visible page-in); bytes preserved.
	ScrubRevokedWitness
	// ScrubPoison: a page benign at write time that today's de-obfuscating content gate
	// quarantines (a tightened detector, or an obfuscation the write-time gate missed).
	// Sealed; its bytes and obfuscated text never reach the index.
	ScrubPoison
)

// String renders the class for reports and logs.
func (c ScrubClass) String() string {
	switch c {
	case ScrubClean:
		return "clean"
	case ScrubUnchecked:
		return "unchecked"
	case ScrubRepaired:
		return "repaired"
	case ScrubTampered:
		return "tampered"
	case ScrubErasure:
		return "erasure"
	case ScrubRevokedWitness:
		return "revoked_witness"
	case ScrubPoison:
		return "poison"
	}
	return "unknown"
}

// ScrubFinding is one page's patrol result — the deterministic, audit-shaped row a
// scrub reports and a downstream tool consumes. It carries no page bytes.
type ScrubFinding struct {
	Step         int        `json:"step"`
	Class        ScrubClass `json:"class"`
	Fault        FaultClass `json:"fault"`                   // the underlying syndrome/CAS class (syndrome.go)
	Digest       string     `json:"digest"`                  // short content address the page references
	ActualDigest string     `json:"actual_digest,omitempty"` // for an erasure: digest the rotted bytes actually hash to ("" if body absent)
	ActualLen    int64      `json:"actual_len,omitempty"`    // for an erasure: length of the rotted bytes
	Reason       string     `json:"reason,omitempty"`        // the seal reason, when sealed
	Sealed       bool       `json:"sealed,omitempty"`
	Tombstoned   bool       `json:"tombstoned,omitempty"`
	Detail       string     `json:"detail,omitempty"`
}

// ScrubOptions controls the patrol pass. With OutputDir empty, Scrub is a dry-run report.
// With OutputDir set, it writes a sealed/repaired copy of the core image (healthy CAS bytes
// copied byte-for-byte; only unhashable rot is dropped).
type ScrubOptions struct {
	OutputDir string
}

// ScrubReport is the patrol ledger: the per-page findings, the per-class counts that form
// the audit rows, and the before/after accounting. It carries no page bytes.
type ScrubReport struct {
	InputDir  string `json:"input_dir"`
	OutputDir string `json:"output_dir,omitempty"`
	DryRun    bool   `json:"dry_run"`

	Before Stats `json:"before"`
	After  Stats `json:"after"`

	Findings []ScrubFinding `json:"findings"`

	Clean        int `json:"clean"`
	Unchecked    int `json:"unchecked"`
	Repaired     int `json:"repaired"`
	Tampered     int `json:"tampered"`
	Erasures     int `json:"erasures"`
	RevokedSeals int `json:"revoked_seals"`
	PoisonSeals  int `json:"poison_seals"`

	Invariant string `json:"invariant"`
}

// Scrub runs the off-path ECC patrol over a persisted core image at dir and reports every
// page by its integrity class under today's gates. Unlike Load it does NOT fail-close on a
// corrupt blob: it reads the image raw so an erasure or metadata flip is surveyed and
// reported per page. Pages that now fail (erasure, refuted witness, tightened gate, security
// tamper) are sealed/tombstoned for model-visible page-in.
func Scrub(ctx context.Context, dir string, opt ScrubOptions) (ScrubReport, error) {
	m, cas, err := loadImageRaw(dir)
	if err != nil {
		return ScrubReport{}, err
	}
	if opt.OutputDir != "" && samePath(dir, opt.OutputDir) {
		return ScrubReport{}, fmt.Errorf("recall: scrub output dir must differ from input dir %q", dir)
	}

	// A Session so the content re-screen runs through a FRESH gate plus every registered
	// ResultAdmitter, exactly as page-in does — never the (possibly weaker) write-time gate.
	// Its cleared map is unused (Scrub never calls Resolve/Clear); the clearance state that
	// matters is the working `cleared` below, which is written to the output.
	s := &Session{Manifest: m, cas: cas, cleared: map[string]bool{}, gate: ctxmmu.New()}

	pages := append([]Page(nil), m.Pages...)
	changes := append([]ContextChange(nil), m.ContextChanges...)
	cleared := copyBoolMap(m.Cleared)
	nextQ := maxQID(pages)
	report := ScrubReport{
		InputDir: dir,
		DryRun:   opt.OutputDir == "",
		Before:   statsOf(m, cas),
		Invariant: "a healthy blob is preserved byte-for-byte and a sealed page keeps its bytes; an unhashable (rotted) blob " +
			"is dropped from the reloadable output but fingerprinted in the audit record; a metadata tamper over a security " +
			"field is sealed, never re-stamped clean; every later page-in still runs the witness gate plus a fresh content re-screen",
	}
	if opt.OutputDir != "" {
		report.OutputDir = opt.OutputDir
	}

	for i := range pages {
		f := scrubPage(ctx, s, &pages[i], cas, &changes, cleared, &nextQ)
		report.Findings = append(report.Findings, f)
		switch f.Class {
		case ScrubClean:
			report.Clean++
		case ScrubUnchecked:
			report.Unchecked++
		case ScrubRepaired:
			report.Repaired++
		case ScrubTampered:
			report.Tampered++
		case ScrubErasure:
			report.Erasures++
		case ScrubRevokedWitness:
			report.RevokedSeals++
		case ScrubPoison:
			report.PoisonSeals++
		}
	}

	out := m
	out.Pages = pages
	out.ContextChanges = changes
	out.Cleared = cleared

	// The reloadable output preserves every blob that still hashes (byte-for-byte) and
	// drops only the unhashable rot — so a future recall.Load (which fail-closes on a
	// corrupt blob) accepts the scrubbed image. Each drop is already recorded as a
	// ScrubErasure finding (with the rotted bytes' fingerprint) plus a tombstone, so no
	// evidence is silently deleted.
	outCAS := make(map[string][]byte, len(cas))
	for d, b := range cas {
		if Digest(b) == d {
			outCAS[d] = b
		}
	}
	report.After = statsOf(out, outCAS)

	if opt.OutputDir != "" {
		if err := writeImage(opt.OutputDir, out, outCAS); err != nil {
			return ScrubReport{}, err
		}
	}
	return report, nil
}

// scrubPage folds the integrity checks for one page in a load-bearing order and mutates *p
// (and, for an erasure, the context-control ledger; for a seal, the cleared map + nextQ) in
// place. The order is security-critical:
//
//   - ERASURE first: a missing/rotted body means the authoritative source is gone, so no
//     verdict about its metadata or content can be trusted — tombstone and stop.
//   - WITNESS next: a refuted source seals the page regardless of content.
//   - CONTENT next: a previously-benign page today's gate quarantines is sealed. This runs
//     BEFORE the metadata branches so a poisoned body whose Quarantined bit was flipped off
//     is re-quarantined here, never reclassified as a benign repair.
//   - METADATA last: a syndrome mismatch is REPAIRABLE only if re-deriving Len alone
//     restores the match (a benign Len rot). Any residual mismatch implicates a
//     non-derivable security field (Taint/Quarantined/QID) the one-way syndrome cannot
//     recover — fail-closed: seal and keep it flagged (ScrubTampered), never re-stamp clean.
func scrubPage(ctx context.Context, s *Session, p *Page, cas map[string][]byte, changes *[]ContextChange, cleared map[string]bool, nextQ *int) ScrubFinding {
	body := cas[p.Digest]
	fault := ClassifyFault(*p, body)
	find := ScrubFinding{Step: p.Step, Fault: fault, Digest: short(p.Digest)}

	// 1) ERASURE dominates — the bytes are gone or rotted; nothing local repairs them.
	if fault == FaultErasure {
		find.Class = ScrubErasure
		find.Tombstoned = true
		find.Reason = "erasure"
		if body == nil {
			find.Detail = "body absent from CAS; needs a replica or witness"
		} else {
			// Present but unhashable: fingerprint the ACTUAL rotted bytes so the loss is
			// identifiable from the report alone, even after the blob is dropped from output.
			find.ActualDigest = short(Digest(body))
			find.ActualLen = int64(len(body))
			find.Detail = fmt.Sprintf("body no longer hashes to its address (actual=%s, %d bytes); needs a replica or witness", find.ActualDigest, find.ActualLen)
		}
		tombstoneErasure(p, body, changes)
		return find
	}

	// Body present and authoritative from here.

	// 2) WITNESS: a source refuted since persist seals the page no matter its content.
	if p.Witness != "" && vdso.Default.Revoked(p.Witness) {
		find.Class = ScrubRevokedWitness
		find.Sealed = true
		find.Reason = abi.ReasonName(abi.ReasonTrustViolation)
		find.Detail = fmt.Sprintf("witness %q refuted since persist", p.Witness)
		sealInScrub(p, cleared, nextQ, find.Reason, "refuted witness")
		return find
	}

	// 3) CONTENT: a page resident as benign that today's de-obfuscating gate quarantines.
	// Only meaningful for a currently-benign page; a sealed page is already suppressed.
	if !p.Quarantined {
		if v := s.reScreen(ctx, p.Role, body); v.Kind == abi.VerdictQuarantine {
			find.Class = ScrubPoison
			find.Sealed = true
			find.Reason = abi.ReasonName(v.Reason)
			find.Detail = "benign at write time; today's content gate quarantines it"
			sealInScrub(p, cleared, nextQ, find.Reason, "tightened re-screen")
			return find
		}
	}

	// 4) METADATA: a syndrome mismatch. Re-derive the one body-derivable integrity field
	// (Len) and check whether that alone restores the stored check word.
	if fault == FaultRepairable {
		cand := *p
		cand.Len = int64(len(body))
		if computeSyndrome(cand) == cand.Syndrome {
			// The corruption was confined to Len — a benign rot. Re-stamp; bytes untouched.
			*p = stampSyndrome(cand)
			find.Class = ScrubRepaired
			find.Detail = "Len re-derived from the authoritative body; syndrome restored"
			return find
		}
		// Residual mismatch ⇒ a non-body-derivable security field (Taint/Quarantined/QID)
		// was altered. The one-way syndrome cannot recover the pre-tamper value, so re-stamping
		// would LAUNDER the tamper (an unseal, a taint escalation). Fail-closed: seal and flag.
		find.Class = ScrubTampered
		find.Sealed = true
		find.Reason = abi.ReasonName(abi.ReasonTrustViolation)
		find.Detail = "syndrome mismatch persists after re-deriving Len; a non-derivable security field (taint/quarantined/qid) was altered — sealed, original value unrecoverable"
		sealInScrub(p, cleared, nextQ, find.Reason, "metadata tamper")
		return find
	}

	// 5) UNCHECKED: a pre-rung page with no syndrome. Re-derive the body-derivable field and
	// stamp forward so the next patrol can check it (consistent with branch 4 — a corrupt Len
	// is never blessed as authoritative). Not a fault; benign body.
	if fault == FaultUnchecked {
		p.Len = int64(len(body))
		*p = stampSyndrome(*p)
		find.Class = ScrubUnchecked
		find.Detail = "no syndrome on the persisted page; Len re-derived and stamped forward for the next patrol"
		return find
	}

	// 6) CLEAN.
	find.Class = ScrubClean
	return find
}

// sealInScrub seals a page during the patrol with Dream's clearance discipline: it allocates
// a fresh, collision-free QID from the monotonic nextQ seed and deletes any stale clearance
// for that QID from the working cleared map (written to the output), so a freshly-sealed page
// can never be re-opened by a leftover witness clearance.
func sealInScrub(p *Page, cleared map[string]bool, nextQ *int, reason, detail string) {
	*nextQ++
	qid := ensureQID(p.QID, *nextQ)
	sealPage(p, qid, reason, detail)
	delete(cleared, qid)
}

// tombstoneErasure appends a system-originated tombstone for an erased page, suppressing it
// from future model-visible page-in without deleting any evidence. The reason carries the
// rotted bytes' fingerprint when present. Idempotent: a page already tombstoned is left as-is.
func tombstoneErasure(p *Page, body []byte, changes *[]ContextChange) {
	for _, ch := range *changes {
		if ch.Applied && ch.Action == ContextActionTombstone && ch.Step == p.Step {
			return
		}
	}
	reason := "recall/scrub: erasure — body absent"
	if body != nil {
		reason = fmt.Sprintf("recall/scrub: erasure — body rotted (actual=%s, %d bytes)", short(Digest(body)), len(body))
	}
	*changes = append(*changes, ContextChange{
		ID:          contextChangeID("scrub", p.Step, p.Digest),
		Action:      ContextActionTombstone,
		Step:        p.Step,
		Digest:      p.Digest,
		Reason:      reason,
		RequestedBy: "recall/scrub",
		TrustEpoch:  vdso.Default.TrustEpoch(),
		Applied:     true,
	})
}

// loadImageRaw reads a persisted core image WITHOUT the whole-bundle CAS integrity gate
// that Load enforces. Load fail-closes on the first corrupt blob (correct for serving); a
// patrol pass must instead survey corruption per page, so it reads the raw map and lets
// ClassifyFault classify each page (an unhashable blob surfaces as that page's erasure).
func loadImageRaw(dir string) (Manifest, map[string][]byte, error) {
	mb, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return Manifest{}, nil, err
	}
	var m Manifest
	if err := json.Unmarshal(mb, &m); err != nil {
		return Manifest{}, nil, fmt.Errorf("recall: bad manifest: %w", err)
	}
	if m.Version != ManifestVersion {
		return Manifest{}, nil, fmt.Errorf("recall: manifest version %q != %q", m.Version, ManifestVersion)
	}
	cb, err := os.ReadFile(filepath.Join(dir, "cas.json"))
	if err != nil {
		return Manifest{}, nil, err
	}
	var cas map[string][]byte
	if err := json.Unmarshal(cb, &cas); err != nil {
		return Manifest{}, nil, fmt.Errorf("recall: bad cas: %w", err)
	}
	return m, cas, nil
}
