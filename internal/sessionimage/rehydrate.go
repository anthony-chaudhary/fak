package sessionimage

// rehydrate.go — RESUME: bring a loaded image back to life in THIS process, on this
// host, optionally under a different model. It re-attaches the three primitives the
// image carries:
//
//   - the DRIVE — session.Table.Restore puts the persisted State back verbatim (Rev
//     and all), so a paused session resumes paused and a stopped session resumes
//     stopped (never silently revived);
//   - the CONTENT — recall.Load re-arms the trust gate over the page table, and
//     recall.LoadOrAttachIndex re-attaches (or rebuilds) the ctxplan candidate index;
//   - the IDENTITY move — if the resume targets a different model or host, a Migration
//     is appended to the image's log (and optionally written back), so the change is an
//     audited fact. No content is transformed: the page table and index are
//     model-agnostic, so the new model simply re-prefills from the same logical bytes.
//
// Rehydrate never resurrects more than the image holds: a drive-only image rehydrates
// only its drive (Session/Index stay nil); the KV cache is never restored (it is a
// cache, rebuilt on the first turn — Portability.KVIncluded is false by design).

import (
	"context"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// RehydrateOptions configures a resume. Table, when non-nil, is the live drive table the
// session is re-attached into (keyed by SessionID). ToModel / ToHost, when set and
// different from the image's recorded Model / Host, record a Migration — the model swap
// or re-home is made explicit. WriteBack persists the updated Meta (with the new
// migration and Model/Host) back to image.json, so the next dump carries the lineage.
// Now is an injected unix clock for deterministic migration stamps AND the dormancy gap
// (0 = wall time). Gate, when non-nil, is the horizon-gated re-entry gate (internal/rehydrate,
// #1181): after the image is restored, Rehydrate computes the dormancy band from how long
// the image has been dormant (now − Meta.UpdatedUnix) and runs the staged gate before the
// resumed handle is admitted for its first post-wake action — a longer gap runs strictly
// more revalidation. A nil Gate is today's behavior: resume verbatim, admitted unconditionally.
type RehydrateOptions struct {
	Table     *session.Table
	ToModel   string
	ToHost    string
	Reason    string
	WriteBack bool
	Now       int64
	Gate      *rehydrate.Gate
}

// Resumed is the live session after a Rehydrate: the (possibly migrated) Meta, the drive
// State as re-attached, the content primitives paged in (Session/Index are nil for a
// drive-only image), and the persisted keep-bits (Witness is nil when the image carried
// none). Migrated reports whether a model/host change was recorded. Witness is the rung a
// resumed loop consults before re-firing an effect — the ACRFence distinction, restored.
type Resumed struct {
	Meta     Meta
	Drive    session.State
	Session  *recall.Session
	Index    *ctxplan.Index
	Witness  []WitnessEntry
	Migrated bool

	// Gated is true when a staged rehydration Gate ran (RehydrateOptions.Gate was set).
	// When false, no staging was configured and the resume is admitted unconditionally
	// (today's verbatim resume).
	Gated bool
	// Admission is the staged-gate verdict, meaningful only when Gated. A refused Admission
	// (Admission.Admitted false) means the caller must NOT fire the first post-wake action
	// until the rung named by Admission.RefusedBy clears — the CRaC afterRestore gate.
	Admission rehydrate.Admission
}

// Admitted reports whether the resumed session may fire its first post-wake action. With no
// staged gate configured it is always true (unconditional resume); with a gate it is the
// gate's verdict (every applicable rung cleared).
func (r *Resumed) Admitted() bool { return !r.Gated || r.Admission.Admitted }

// Rehydrate resumes the image in this process. It restores the drive into opt.Table (if
// given), loads the recall core image and its index (if the image carries content), and
// records a Migration when the resume targets a different model or host. The returned
// Resumed is the live handle; a follow-up turn reads its Drive each boundary
// (session.Table.Decide) and pages content through its Session's gate.
func (img *Image) Rehydrate(ctx context.Context, opt RehydrateOptions) (*Resumed, error) {
	out := &Resumed{Meta: img.Meta, Drive: img.Drive}

	// Dormancy is measured from when the image was last persisted (Meta.UpdatedUnix) to now —
	// the gap the staged gate (step 4) keys on. Captured up front because a write-back
	// migration below re-stamps UpdatedUnix to "now", which would erase the gap.
	dormantStamp := dormancy.FromUnix(img.Meta.UpdatedUnix)

	// (1) Re-attach the drive verbatim — the §5 persistence rung. A terminal session
	// restores terminal; Rev is preserved (a load is not a mutation).
	if opt.Table != nil {
		out.Drive = opt.Table.Restore(img.Meta.SessionID, img.Drive)
	}

	// (2) Page the content primitives back in, gate re-armed. A drive-only image skips
	// this (Session/Index stay nil) — there is nothing to resolve yet.
	if img.HasCoreImage() {
		s, err := recall.Load(img.Dir)
		if err != nil {
			return nil, err
		}
		out.Session = s
		ix, err := recall.LoadOrAttachIndex(ctx, img.Dir, s)
		if err != nil {
			return nil, err
		}
		out.Index = ix
	}

	// (2b) Re-attach the persisted keep-bits. The bytes were integrity-verified at Load;
	// this decodes them onto the live handle so a resumed loop can gate re-execution on
	// VerifiedDone (Resumed.Witness / Image.VerifiedDone) instead of replaying the effect.
	w, err := img.Witness()
	if err != nil {
		return nil, err
	}
	out.Witness = w

	// (3) Record an identity move. The content is model-agnostic, so a model change needs
	// no transform — only an honest entry in the log.
	if mig, changed := migrationFor(img.Meta, opt); changed {
		out.Meta.Migrations = append(out.Meta.Migrations, mig)
		if opt.ToModel != "" {
			out.Meta.Model = opt.ToModel
		}
		if opt.ToHost != "" {
			out.Meta.Host = opt.ToHost
		}
		out.Migrated = true
		if opt.WriteBack {
			// Re-stamp UpdatedUnix and persist the new Meta; Parts are unchanged (no
			// content moved), so the integrity index still verifies.
			out.Meta.UpdatedUnix = mig.WhenUnix
			img.Meta = out.Meta
			if err := writeImageJSON(img.Dir, out.Meta); err != nil {
				return nil, err
			}
		}
	}

	// (4) Horizon-gated admission (#1181): the longer the image was dormant, the more rungs
	// must clear before the resumed handle may fire its first post-wake action (the CRaC
	// afterRestore analog). A nil Gate skips this — resume verbatim, admitted unconditionally
	// (today's behavior). The dormancy band comes from the image's pre-migration UpdatedUnix.
	if opt.Gate != nil {
		now := time.Now()
		if opt.Now != 0 {
			now = time.Unix(opt.Now, 0)
		}
		out.Gated = true
		out.Admission = opt.Gate.Admit(ctx, dormantStamp.HorizonAt(now))
	}
	return out, nil
}

// migrationFor builds the Migration for a resume that changes the model or host, or
// reports changed=false when the resume stays on the same model and host. A ToModel /
// ToHost equal to the current value is a no-op (resuming "in place" records nothing).
func migrationFor(meta Meta, opt RehydrateOptions) (Migration, bool) {
	modelChanged := opt.ToModel != "" && opt.ToModel != meta.Model
	hostChanged := opt.ToHost != "" && opt.ToHost != meta.Host
	if !modelChanged && !hostChanged {
		return Migration{}, false
	}
	now := opt.Now
	if now == 0 {
		now = time.Now().Unix()
	}
	mig := Migration{WhenUnix: now, Reason: opt.Reason}
	if modelChanged {
		mig.FromModel, mig.ToModel = meta.Model, opt.ToModel
	}
	if hostChanged {
		mig.FromHost, mig.ToHost = meta.Host, opt.ToHost
	}
	return mig, true
}
