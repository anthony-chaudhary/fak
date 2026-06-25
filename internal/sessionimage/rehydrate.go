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
	"github.com/anthony-chaudhary/fak/internal/recall"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// RehydrateOptions configures a resume. Table, when non-nil, is the live drive table the
// session is re-attached into (keyed by SessionID). ToModel / ToHost, when set and
// different from the image's recorded Model / Host, record a Migration — the model swap
// or re-home is made explicit. WriteBack persists the updated Meta (with the new
// migration and Model/Host) back to image.json, so the next dump carries the lineage.
// Now is an injected unix clock for deterministic migration stamps (0 = wall time).
type RehydrateOptions struct {
	Table     *session.Table
	ToModel   string
	ToHost    string
	Reason    string
	WriteBack bool
	Now       int64
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
}

// Rehydrate resumes the image in this process. It restores the drive into opt.Table (if
// given), loads the recall core image and its index (if the image carries content), and
// records a Migration when the resume targets a different model or host. The returned
// Resumed is the live handle; a follow-up turn reads its Drive each boundary
// (session.Table.Decide) and pages content through its Session's gate.
func (img *Image) Rehydrate(ctx context.Context, opt RehydrateOptions) (*Resumed, error) {
	out := &Resumed{Meta: img.Meta, Drive: img.Drive}

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
