package sessionimage

// drive_projection.go — the DRIVE-PROJECTION sibling (issue #4126, guard-lifecycle epic
// #1193): the witnessed checkpoint packet already carries witnessed PROGRESS (witness.json,
// the keep-bit; see witness.go) but no BUDGET projection, so a resume watchdog that
// relaunches `claude --resume <uuid>` brings the child up with a FRESH drive — full budget
// again — because the real, spent-down budgets are stranded on the gateway-trace side of a
// disjoint keyspace (internal/resume/drivestate.go documents the transcript-UUID vs
// gateway-trace split, and that only the operator-hold token is smuggled across).
//
// The full drive State IS dumped verbatim to session.json, but that is the WHOLE
// session.State keyed to the gateway trace — there is no compact, resume-facing block a
// lean, transcript-UUID-side consumer can read without paging the full state. This sibling
// closes that the same way witness.go closed the keep-bit gap: drive.json carries a compact
// DriveProjection — ONLY the structural drive axes a relaunched governor needs to come up at
// the CARRIED (spent) budget instead of a fresh one — as a first-class, sha256-indexed part
// of the image, verified on Load exactly like every other part (verifyParts).
//
// It is a SEPARATE sibling of session.json, never a replacement: the full-State verbatim
// carrier stays; this is the lean view alongside it, DERIVED here (the session lane is
// untouched — no new field on session.State). Structural by construction: the projection
// carries budget/priority/pace/objective-pin/generation and NOTHING host- or account-tagged
// (no CacheAffinity, no Host/Account, no transcript bytes) — #4128 makes that a tested,
// enforced invariant (scrubForReHome + the leak-gate test).

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// DriveBudget projects ONLY the four remaining-allotment axes of session.Budget — the
// spent-down "left" values a relaunched governor must resume AT, never the caps (the caps
// are the denominators the base governor already holds; SeedState re-seeds the remaining
// onto them, #4127). Unbounded (-1) rides through unchanged, so an uncapped axis round-trips
// as uncapped. omitempty keeps a zero axis off the wire; a 0 decodes back to 0.
type DriveBudget struct {
	TurnsLeft           int   `json:"turns_left,omitempty"`
	TokensLeft          int   `json:"tokens_left,omitempty"`
	ContextTokensLeft   int   `json:"context_tokens_left,omitempty"`
	SpendMicroCentsLeft int64 `json:"spend_micro_cents_left,omitempty"`
}

// DrivePin projects the ObjectivePin identity+fingerprint ONLY (issue #4126): the stable
// PinID and the content Digest — the pair that makes "the objective was preserved" a
// checkable equality across a re-home. It deliberately drops Text/Step/SourceSpanID: the
// projection is an identity carrier, not the objective's content (the full pin, with Text,
// still rides session.json).
type DrivePin struct {
	PinID  string `json:"pin_id,omitempty"`
	Digest string `json:"digest,omitempty"`
}

// DriveProjection is the compact, resume-facing view of a session's drive written to
// drive.json — the smallest set of STRUCTURAL axes (Budget remaining, Priority, Pace,
// ObjectivePin identity, Generation lineage counter) a relaunched governor needs to come up
// at the carried (spent) budget instead of a fresh one. Version mirrors the image Version so
// a reader fails closed on a format it does not recognize (the same fail-closed discipline
// WitnessSet uses, witness.go:55-58).
type DriveProjection struct {
	Version      string       `json:"version"`
	Budget       DriveBudget  `json:"budget,omitempty,omitzero"`
	Priority     int          `json:"priority,omitempty"`
	Pace         session.Pace `json:"pace,omitempty,omitzero"`
	ObjectivePin DrivePin     `json:"objective_pin,omitempty,omitzero"`
	Generation   int          `json:"generation,omitempty"`
}

// IsZero reports whether the projection carries no structural drive signal — every axis at
// its zero value (Version excluded: projectDrive always stamps it). DumpDir skips writing
// drive.json for a zero projection, mirroring the optional-part pattern (a session with
// nothing to project stays byte-identical to a pre-#4126 image). It also drives the
// `omitzero` tag were the projection ever nested.
func (dp DriveProjection) IsZero() bool {
	return dp.Budget == (DriveBudget{}) &&
		dp.Priority == 0 &&
		dp.Pace == (session.Pace{}) &&
		dp.ObjectivePin == (DrivePin{}) &&
		dp.Generation == 0
}

// projectDrive derives the compact DriveProjection from a full drive State — the single
// place the session.State -> drive.json mapping lives. It copies ONLY the structural axes;
// nothing host/account-tagged is reachable from here by construction (#4128 makes that a
// tested invariant via scrubForReHome). Version is stamped so the written part fails closed
// on a format mismatch.
func projectDrive(st session.State) DriveProjection {
	return DriveProjection{
		Version: Version,
		Budget: DriveBudget{
			TurnsLeft:           st.Budget.TurnsLeft,
			TokensLeft:          st.Budget.TokensLeft,
			ContextTokensLeft:   st.Budget.ContextTokensLeft,
			SpendMicroCentsLeft: st.Budget.SpendMicroCentsLeft,
		},
		Priority:     st.Priority,
		Pace:         st.Pace,
		ObjectivePin: DrivePin{PinID: st.ObjectivePin.PinID, Digest: st.ObjectivePin.Digest},
		Generation:   st.Generation,
	}
}

// scrubForReHome returns the projection guaranteed origin-free — the enforced leak-gate a
// drive.json passes through before it crosses the model-agnostic offload boundary (issue
// #4128). The image is portable ACROSS hosts, accounts, and residencies by design (the
// sessionimage.go offload contract), and a re-home appends a Migration with FromHost/ToHost
// — so any origin identity riding the projection would leak into the destination image. This
// is an ALLOWLIST rebuild, not a blocklist strip: it reconstructs the projection from ONLY
// the origin-safe structural axes (Budget remaining, Priority, Pace, ObjectivePin identity,
// Generation), so a field added to DriveProjection later is dropped here unless its author
// consciously threads it through this allowlist — origin-freedom cannot regress by accident.
// Nothing host/account/trace-derived is reachable: the projection never carries
// CacheAffinity's AffinityKey/FromTraceID/ToTraceID or ParentTrace (session.State's
// account-scoped lineage), and the Host/Account/Residency provenance lives on image.json Meta,
// never here. Generation — the one structural lineage counter — rides through, so a resumed
// governor still knows how many re-continuations preceded it. writeDriveProjection applies
// this to EVERY drive.json written (the dump AND the re-home re-write), so no write can carry
// an origin token regardless of what the source session.State held.
func (dp DriveProjection) scrubForReHome() DriveProjection {
	return DriveProjection{
		Version:      dp.Version,
		Budget:       dp.Budget,
		Priority:     dp.Priority,
		Pace:         dp.Pace,
		ObjectivePin: DrivePin{PinID: dp.ObjectivePin.PinID, Digest: dp.ObjectivePin.Digest},
		Generation:   dp.Generation,
	}
}

// writeDriveProjection writes the drive sibling deterministically (MarshalIndent over a
// fixed-field struct is stable), so the part's digest — and therefore the packed archive —
// are byte-stable across runs, the same determinism the integrity index and the .faksession
// archive rely on. Every write passes through scrubForReHome (issue #4128): the drive.json
// that crosses the offload boundary is the enforced-origin-free projection, whatever the
// caller passes — the dump path and the re-home re-write are both gated here, in one place.
func writeDriveProjection(path string, dp DriveProjection) error {
	b, err := json.MarshalIndent(dp.scrubForReHome(), "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// rewriteDriveForReHome re-writes drive.json through the leak-gate for a re-home Migration
// and returns the re-hashed part index (issue #4128). A re-home crosses hosts/accounts, so
// the projection is re-derived from the (restored) drive and re-written through
// writeDriveProjection — which scrubs any origin identity — and the parts are re-hashed so
// image.json's sha256 integrity index still verifies the freshly written bytes. For a
// same-State re-home the projection is byte-identical (the enforced chokepoint is what a
// future migration that DID touch the drive would still have to pass); an image that carries
// no drive.json is left untouched — a pre-#4126 image grows no new part and the caller's
// existing Parts ride back unchanged.
func rewriteDriveForReHome(dir string, drive session.State, parts []Part) ([]Part, error) {
	if _, err := os.Stat(filepath.Join(dir, DriveFile)); err != nil {
		if os.IsNotExist(err) {
			return parts, nil
		}
		return nil, err
	}
	if err := writeDriveProjection(filepath.Join(dir, DriveFile), projectDrive(drive)); err != nil {
		return nil, err
	}
	return indexParts(dir)
}

// SeedState re-seeds a governor's drive from the checkpoint projection (issue #4127): it
// writes the carried (spent) Budget-remaining axes, Priority, Pace, ObjectivePin identity,
// and Generation onto a base session.State, returning the seeded state. This is the
// projection-only re-home path — a lean relaunch consumer that carries the witnessed
// checkpoint but NOT the full session.json seeds a governor from here, so the resumed child
// comes up at the CARRIED spent budget instead of a fresh full one (the reset-to-full-budget
// bug this closes). Only the four remaining-allotment axes are overwritten; the base's caps
// (ContextTokensCap/SpendMicroCentsCap and the clarification axis, the denominators) are
// preserved, so a re-seed narrows the remaining onto the base's limits rather than resetting
// them. The ObjectivePin re-seed is identity-only (PinID+Digest) — the pair the consumer
// checks for "the objective was preserved"; the pin's Text still rides session.json.
func (dp DriveProjection) SeedState(base session.State) session.State {
	base.Budget.TurnsLeft = dp.Budget.TurnsLeft
	base.Budget.TokensLeft = dp.Budget.TokensLeft
	base.Budget.ContextTokensLeft = dp.Budget.ContextTokensLeft
	base.Budget.SpendMicroCentsLeft = dp.Budget.SpendMicroCentsLeft
	base.Priority = dp.Priority
	base.Pace = dp.Pace
	base.ObjectivePin.PinID = dp.ObjectivePin.PinID
	base.ObjectivePin.Digest = dp.ObjectivePin.Digest
	base.Generation = dp.Generation
	return base
}

// DriveProjection reads the persisted drive projection back: (projection, present, err).
// present is false when the image carries no drive.json (a pre-#4126 image, or a session
// whose projection was zero) — the caller then re-seeds nothing and rehydrates
// byte-identically. The bytes were already integrity-checked by LoadDir/verifyParts before
// this is reachable, so a returned projection is proven whole; this re-reads them only to
// decode. A version mismatch fails closed.
func (img *Image) DriveProjection() (DriveProjection, bool, error) {
	b, err := os.ReadFile(filepath.Join(img.Dir, DriveFile))
	if err != nil {
		if os.IsNotExist(err) {
			return DriveProjection{}, false, nil
		}
		return DriveProjection{}, false, err
	}
	var dp DriveProjection
	if err := json.Unmarshal(b, &dp); err != nil {
		return DriveProjection{}, false, fmt.Errorf("sessionimage: bad %s: %w", DriveFile, err)
	}
	if dp.Version != Version {
		return DriveProjection{}, false, fmt.Errorf("sessionimage: drive projection version %q != %q", dp.Version, Version)
	}
	return dp, true, nil
}
