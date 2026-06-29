package recall

import (
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// SYNDROME PATROL SCRUB (#784, builds on the #783 PageSyndrome keystone; epic #782).
//
// ECC memory does not wait for a read fault to discover corruption — it patrol-scrubs
// in the background. recall has the same need for persisted core images: a page that
// was reusable-as-context when its session persisted can quietly stop being reusable
// before a future session relies on it, because three of its evidence axes are NOT
// frozen into the bytes — they are re-evaluated against TODAY's world:
//
//  1. the WITNESS / TRUST-EPOCH axes — the source a page was admitted under can be
//     refuted in the vDSO revocation ledger AFTER the image was written, so a page
//     witnessed-and-current at persist is witness-stale now;
//  2. the DIGEST axis — the swap device can rot between persist and reuse, so the
//     bytes under a page's address may no longer hash to it (an erasure);
//  3. the QUARANTINE axis — a page sealed since persist, or whose recorded clearance
//     was retired, must not silently re-enter context.
//
// ScrubSyndrome is the OFF-PATH re-verification that answers, per persisted page, the
// one question the #783 PageSyndrome view was built to answer — "what independent
// evidence must still hold before this page can become context again?" — under the
// LIVE gates, and reports which pages are still reusable vs which now fail an evidence
// axis. It COMPOSES PageSyndromeFor (the five-axis digest/quarantine/durability/
// witness/trust-epoch fold from #783); it does not re-derive that fold.
//
// It is deliberately NOT the hot path and NOT the action-taking Scrub (scrub.go, which
// seals/tombstones and re-stamps): it is a pure READ over the persisted bodies. It
// mutates no page body, no metadata field, and no on-disk image — it reports and flags
// only. Any quarantine/seal action stays in the existing Scrub seam; this pass exists so
// an operator (or a periodic patrol) can SURVEY a persisted image's reusability under
// today's trust ledger without loading it for serving and without changing a byte.
//
// Where it differs from Scrub's per-page ScrubClass: ScrubClass names the ACTION the
// active scrubber took (clean/repaired/tampered/erasure/revoked_witness/poison), folding
// a fresh content re-screen and a metadata-tamper check. ScrubSyndrome reports the #783
// REUSABILITY EVIDENCE — including the DURABILITY axis, which the action-class scrub does
// not check at all: a page can be action-clean yet still not durable enough to re-enter a
// NEW context (the rung-1 default-expire inversion, #82). The two are complementary views
// of the same image: one says "what did I fix?", this one says "is the evidence still
// intact enough to reuse?".

// PageReusability is one persisted page's re-verification result: its computed #783
// syndrome, whether the page is still reusable-as-context under today's gates, and —
// when it is not — the exact evidence axes that no longer hold. It carries no page
// bytes; it is an audit-shaped row, the syndrome twin of ScrubFinding.
type PageReusability struct {
	Step int `json:"step"`
	// Syndrome is the full five-axis #783 evidence roll-up for the page (every axis,
	// not only the failures), so a consumer sees the complete picture.
	Syndrome PageSyndrome `json:"syndrome"`
	// Reusable mirrors Syndrome.Reusable(): true iff every REQUIRED evidence axis holds.
	Reusable bool `json:"reusable"`
	// Failed lists the axis name of each required-but-not-held axis, in EvidenceAxis
	// order — exactly Syndrome.FailedEvidence() projected to names. Empty iff Reusable.
	Failed []string `json:"failed,omitempty"`
}

// ScrubSyndromeReport is the read-only patrol ledger: one PageReusability row per
// persisted page plus the reusable/failed counts that form the audit summary. It
// carries no page bytes and reflects no mutation — a ScrubSyndrome over an image leaves
// the image byte-identical.
type ScrubSyndromeReport struct {
	InputDir string `json:"input_dir"`

	Rows []PageReusability `json:"rows"`

	Pages    int `json:"pages"`
	Reusable int `json:"reusable"`
	Failed   int `json:"failed"`

	// FailedByAxis counts, per evidence-axis name, how many pages that axis blocked —
	// the field-level summary an operator scans to see WHICH integrity dimension rotted
	// across the image (e.g. "12 pages lost the witness axis" after a mass revocation).
	FailedByAxis map[string]int `json:"failed_by_axis,omitempty"`

	Invariant string `json:"invariant"`
}

// ScrubSyndrome runs the off-path syndrome re-verification over a persisted core image
// at dir: it walks every persisted page, recomputes its #783 PageSyndrome against the
// page's actual CAS bytes and the LIVE vDSO revocation ledger, folding in the image's
// frozen witness-clearance state for the quarantine axis, and reports which pages are
// still reusable vs which now fail an evidence axis.
//
// It is a pure read: like Scrub it reads the image raw (loadImageRaw) so a partially
// corrupt image is surveyed per page rather than fail-closing the whole bundle, but
// unlike Scrub it NEVER mutates a page, seals/tombstones, re-stamps a syndrome, or
// writes an output image. The bodies it reads are left byte-identical. Any action a
// failing page warrants goes through the existing Scrub seam.
func ScrubSyndrome(dir string) (ScrubSyndromeReport, error) {
	m, cas, err := loadImageRaw(dir)
	if err != nil {
		return ScrubSyndromeReport{}, err
	}
	return scrubSyndromeImage(dir, m, cas, vdso.Default), nil
}

// scrubSyndromeImage is the oracle-injected core, so a test can pin a refuted witness
// without mutating the package-global vDSO. It walks the manifest's pages in order and
// composes PageSyndrome for each against its CAS body and the frozen clearance map.
func scrubSyndromeImage(dir string, m Manifest, cas map[string][]byte, oracle revocationOracle) ScrubSyndromeReport {
	rep := ScrubSyndromeReport{
		InputDir: dir,
		Pages:    len(m.Pages),
		Invariant: "a pure read over persisted page bodies: ScrubSyndrome recomputes each page's #783 evidence " +
			"syndrome under today's revocation ledger and reports reusable-vs-failed-axis, mutating no page body, " +
			"no metadata, and no on-disk image — any seal/tombstone action stays in the active Scrub seam",
	}
	for _, p := range m.Pages {
		body := cas[p.Digest] // nil if absent — digestEvidence treats that as an erasure
		syn := pageSyndromeWith(p, body, oracle)
		// Fold in the image's FROZEN clearance state for the quarantine axis — the one axis
		// whose evidence lives beside the page (in Manifest.Cleared) rather than on the page,
		// mirroring Session.PageSyndrome. A sealed-but-cleared page holds the axis at this
		// metadata layer; page-in still re-screens the bytes (the second, independent gate).
		for i := range syn.Evidence {
			if syn.Evidence[i].Axis == EvidenceQuarantine {
				syn.Evidence[i] = quarantineEvidenceCleared(p, m.Cleared[p.QID])
			}
		}
		row := PageReusability{Step: p.Step, Syndrome: syn, Reusable: syn.Reusable()}
		for _, e := range syn.FailedEvidence() {
			row.Failed = append(row.Failed, e.Axis.String())
		}
		rep.Rows = append(rep.Rows, row)
		if row.Reusable {
			rep.Reusable++
			continue
		}
		rep.Failed++
		for _, name := range row.Failed {
			if rep.FailedByAxis == nil {
				rep.FailedByAxis = map[string]int{}
			}
			rep.FailedByAxis[name]++
		}
	}
	// row.Failed is already in EvidenceAxis order (FailedEvidence sorts by axis), and the
	// rows preserve manifest order — so the whole report is deterministic without further
	// sorting. The per-axis order (digest < quarantine < durability < witness < trust_epoch)
	// is the meaningful one; an alphabetical re-sort would destroy it.
	return rep
}
