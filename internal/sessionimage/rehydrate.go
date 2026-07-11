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

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
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

	// DriveProjection is the compact, resume-facing drive block re-attached from drive.json
	// (#4127): the carried Budget-remaining / Priority / Pace / ObjectivePin identity /
	// Generation a lean relaunch consumer seeds a governor from (DriveProjection.SeedState),
	// so a resumed child comes up at the CARRIED spent budget instead of a fresh full one.
	// DriveProjectionPresent is false for a pre-#4126 image (no drive.json) — the consumer
	// then re-seeds nothing and the resume is byte-identical to today's behavior.
	DriveProjection        DriveProjection
	DriveProjectionPresent bool

	// CacheEntries is the payload-free cache-invalidation state of the rehydrated content
	// pages (#1536): one cachemeta.Entry per recall page, lowered through the shared
	// cachemeta.FromContextPage adapter, so a resumed session carries EXPLICIT invalidation
	// state (Coherence.InvalidationMode / Validity) alongside the queryable ctxplan Index —
	// not just the history. Nil for a drive-only image (no content pages to record). These
	// records are cold-path honest: FromContextPage places every entry at Residency.Tier
	// TierDisk, never a resident hot tier, so a rehydrated record NEVER implies a live cache
	// hit — the KV cache is not restored (Portability.KVIncluded is false) and the first
	// post-wake turn must still re-prefill from the logical bytes.
	CacheEntries []cachemeta.Entry

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

		// (2c) Record the cache-invalidation state of the rehydrated content pages (#1536).
		// The ctxplan Index above restores the QUERYABLE HISTORY; this restores the EXPLICIT
		// CACHE state the history alone cannot express — each page lowered through the shared
		// recall→cachemeta adapter into a cachemeta.Entry whose Coherence.InvalidationMode /
		// Validity are set from the page's own witness/trust-epoch. A page admitted under an
		// external witness records InvalidationExternalRefutation; the record stays cold-path
		// honest (TierDisk residency — never a hot hit), so item 19's serve gate (#1537) can
		// reason over fresh-vs-stale without this rehydrate ever implying a live hit.
		sid := s.Manifest.SessionID
		pages := s.Manifest.Pages
		if len(pages) > 0 {
			out.CacheEntries = make([]cachemeta.Entry, 0, len(pages))
			for _, p := range pages {
				out.CacheEntries = append(out.CacheEntries, p.CacheEntry(sid))
			}
		}
	}

	// (2b) Re-attach the persisted keep-bits. The bytes were integrity-verified at Load;
	// this decodes them onto the live handle so a resumed loop can gate re-execution on
	// VerifiedDone (Resumed.Witness / Image.VerifiedDone) instead of replaying the effect.
	w, err := img.Witness()
	if err != nil {
		return nil, err
	}
	out.Witness = w

	// (2d) Re-attach the compact drive projection (#4127). A lean relaunch consumer that
	// carries the witnessed checkpoint but not the full session.json seeds a governor from
	// this (Resumed.DriveProjection.SeedState) so the resumed drive comes up at the carried
	// spent budget. Absent for a pre-#4126 image — DriveProjectionPresent stays false and no
	// re-seed runs, so the resume is byte-identical to today. The bytes were integrity-verified
	// at Load; this decodes them onto the live handle.
	dp, present, err := img.DriveProjection()
	if err != nil {
		return nil, err
	}
	out.DriveProjection = dp
	out.DriveProjectionPresent = present

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
			// Re-stamp UpdatedUnix. A re-home crosses hosts/accounts, so re-write the drive
			// projection through the leak-gate (#4128) — rewriteDriveForReHome scrubs any origin
			// identity and re-hashes the parts so the integrity index still verifies the freshly
			// written bytes. The recall content parts are unchanged (no content moved); only
			// drive.json is re-written, byte-identically for a same-State re-home.
			out.Meta.UpdatedUnix = mig.WhenUnix
			parts, err := rewriteDriveForReHome(img.Dir, out.Drive, out.Meta.Parts)
			if err != nil {
				return nil, err
			}
			out.Meta.Parts = parts
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
