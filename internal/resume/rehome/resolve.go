package rehome

import (
	"os"
	"path/filepath"
	"time"
)

// maxTargetProbes bounds how many ranked re-home candidates are live-probed before
// falling back to the best-ranked one, so a fully-walled fleet does not probe every
// dir. Mirrors resume_resolver._MAX_TARGET_PROBES.
const maxTargetProbes = 4

// OwnerStatus is the owner account's availability, mirroring the fields
// resume_resolver.resolve reads out of fleet_accounts.runtime_status.
type OwnerStatus struct {
	Available    bool
	BlockReason  string
	BlockKind    string
	StatusSource string
}

// ResolveInput carries the facts and injected dependencies for a resume resolution.
// Availability / OwnerStatus / the *Fn callbacks are injectable so the decision is
// unit-testable with no live registry or real account dirs; the CLI wires the
// production bindings from internal/fleetaccounts.
type ResolveInput struct {
	SID  string
	Home string
	CWD  string // directory `claude --resume` will run from; "" => os.Getwd
	// DryRun decides and reports but does not copy the transcript.
	DryRun bool
	// ProbeOwner (mirrors resume_resolver probe_owner, default via NewResolveInput)
	// live-probes the owner before re-homing when its block is a carried usage
	// throttle, and enables the duplicate-owner reselection + target probe loop.
	ProbeOwner bool

	// OwnerStatus, when non-nil, is the owner's availability (skips OwnerStatusFn and,
	// mirroring the Python `owner_status is None` guard, disables duplicate reselect).
	OwnerStatus *OwnerStatus
	// Availability, when non-nil, is the live worker roster used for target ranking;
	// nil falls back to AvailabilityFn.
	Availability []Target

	// OwnerStatusFn looks up an account's runtime status (used when OwnerStatus is nil).
	OwnerStatusFn func(account string) OwnerStatus
	// AvailabilityFn discovers the live worker roster (used when Availability is nil).
	AvailabilityFn func() []Target
	// RehomeFn copies a transcript onto a target; defaults to RehomeTranscript.
	RehomeFn func(srcCfg, dstCfg, project, sid string, destProjects []string) bool
	// ProbeFn live-probes one account; required for the ProbeOwner paths.
	ProbeFn ProbeFunc
}

// ReselectMove records a duplicate-owner reselection (from -> to account).
type ReselectMove struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// OwnerProbe records the result of the owner carried-throttle re-probe.
type OwnerProbe struct {
	Available   bool   `json:"available"`
	BlockReason string `json:"block_reason,omitempty"`
}

// CapRelief records that the burst-spread cap was relaxed for a single resume.
type CapRelief struct {
	RehomeCap int    `json:"rehome_cap"`
	Note      string `json:"note"`
}

// TargetProbe records one probed re-home candidate.
type TargetProbe struct {
	Account     string `json:"account"`
	Available   bool   `json:"available"`
	BlockReason string `json:"block_reason,omitempty"`
}

// Decision is the resume resolution record, mirroring the dict resume_resolver.resolve
// returns. Action is one of NOT_FOUND | PIN | REHOME | PIN_BLOCKED.
type Decision struct {
	OK               bool          `json:"ok"`
	Action           string        `json:"action"`
	Session          string        `json:"session"`
	Project          string        `json:"project,omitempty"`
	OwnerAccount     string        `json:"owner_account,omitempty"`
	OwnerConfigDir   string        `json:"owner_config_dir,omitempty"`
	OwnerAvailable   bool          `json:"owner_available,omitempty"`
	OwnerBlockReason string        `json:"owner_block_reason,omitempty"`
	DupCount         int           `json:"dup_count,omitempty"`
	AllAccounts      []string      `json:"all_accounts,omitempty"`
	OwnerReselected  *ReselectMove `json:"owner_reselected,omitempty"`
	OwnerProbe       *OwnerProbe   `json:"owner_probe,omitempty"`
	Rehomed          bool          `json:"rehomed,omitempty"`
	WouldRehome      bool          `json:"would_rehome,omitempty"`
	PinAccount       string        `json:"pin_account,omitempty"`
	PinConfigDir     string        `json:"pin_config_dir"`
	SourceConfigDir  string        `json:"source_config_dir,omitempty"`
	MirroredToCwd    string        `json:"mirrored_to_cwd_slug,omitempty"`
	WouldMirrorToCwd string        `json:"would_mirror_to_cwd_slug,omitempty"`
	RehomeToSibling  string        `json:"rehome_to_sibling,omitempty"`
	CapRelief        *CapRelief    `json:"cap_relief,omitempty"`
	TargetProbes     []TargetProbe `json:"target_probes,omitempty"`
	DestProjectSlugs []string      `json:"dest_project_slugs,omitempty"`
	Reason           string        `json:"reason"`
}

// carriedThrottleBlock reports whether the owner is blocked by a usage throttle CARRIED
// in the registry (not confirmed by a fresh probe) — the stale-block risk a re-probe
// should clear. Mirrors resume_resolver._carried_throttle_block.
func carriedThrottleBlock(st OwnerStatus) bool {
	if st.Available {
		return false
	}
	if st.StatusSource == "probe" {
		return false
	}
	return st.BlockKind == "usage"
}

// Resolve decides where `claude --resume <sid>` should run, re-homing the transcript
// onto a healthy account when the owner is throttled. It is the Go port of
// resume_resolver.resolve.
func Resolve(in ResolveInput) Decision {
	home := in.Home
	cwd := in.CWD
	if cwd == "" {
		if wd, err := os.Getwd(); err == nil {
			cwd = wd
		}
	}
	cwdSlug := ProjectSlug(cwd)

	owner := LocateOwner(in.SID, home)
	if owner == nil {
		return Decision{
			OK: false, Action: "NOT_FOUND", Session: in.SID,
			Reason: "no ~/.claude* account holds this session id",
		}
	}

	// A session in more than one account dir is the signature of a prior re-home:
	// LocateOwner picks the newest-mtime copy, which is the re-home TARGET, not
	// necessarily a serving account. When owner_status was not injected, confirm the
	// pick actually serves and re-pick among the other copies if not.
	var reselect *ReselectMove
	var forcedTarget *Owner
	if in.ProbeOwner && in.OwnerStatus == nil && owner.DupCount > 1 && len(owner.AllAccounts) > 1 {
		d := ReselectDuplicateOwner(in.SID, home, in.ProbeFn)
		switch d.Mode {
		case ReselectPin:
			reselect = &ReselectMove{From: owner.Account, To: d.Owner.Account}
			owner = d.Owner
		case ReselectRehome:
			// Keep the freshest as the copy SOURCE (owner unchanged) and force the
			// landing onto the proven-serving sibling below.
			forcedTarget = d.Target
		}
	}

	var status OwnerStatus
	if in.OwnerStatus != nil {
		status = *in.OwnerStatus
	} else if in.OwnerStatusFn != nil {
		status = in.OwnerStatusFn(owner.Account)
	} else {
		status = OwnerStatus{Available: true}
	}
	ownerAvailable := status.Available
	blockReason := status.BlockReason
	if blockReason == "" {
		blockReason = "blocked"
	}

	rec := Decision{
		OK: true, Session: in.SID, Project: owner.Project,
		OwnerAccount: owner.Account, OwnerConfigDir: owner.ConfigDir,
		OwnerAvailable: ownerAvailable, OwnerBlockReason: status.BlockReason,
		DupCount: owner.DupCount, AllAccounts: owner.AllAccounts,
		OwnerReselected: reselect,
	}

	// Before trusting a CARRIED usage throttle, re-check the owner live — a stale
	// bare-time reset can keep an account marked throttled for hours after it cleared.
	if in.ProbeOwner && !ownerAvailable && carriedThrottleBlock(status) && in.ProbeFn != nil {
		probed := in.ProbeFn(owner.Account, owner.ConfigDir)
		if probed != nil {
			rec.OwnerProbe = &OwnerProbe{Available: probed.Available, BlockReason: probed.BlockReason}
			ownerAvailable = probed.Available
			if probed.BlockReason != "" {
				blockReason = probed.BlockReason
			}
			rec.OwnerAvailable = ownerAvailable
			if ownerAvailable {
				rec.OwnerBlockReason = ""
			}
		}
	}

	// Owner reachable -> pin to it, no cross-account copy. The owner may still store the
	// transcript under a different cwd-slug than the resume launches from, so mirror it
	// WITHIN the owner account into the cwd slug (same account, owner->owner).
	if ownerAvailable {
		reason := "owner account is available -- pin to it (no copy)"
		if rec.OwnerProbe != nil && rec.OwnerProbe.Available {
			reason = "owner's carried throttle was stale -- live probe OK, pin to owner (no re-home)"
		}
		if cwdSlug != "" && cwdSlug != owner.Project {
			if !in.DryRun {
				if in.RehomeFn(owner.ConfigDir, owner.ConfigDir, owner.Project, in.SID, []string{cwdSlug}) {
					rec.MirroredToCwd = cwdSlug
					reason += " (mirrored into cwd slug " + cwdSlug + ")"
				}
			} else {
				rec.WouldMirrorToCwd = cwdSlug
			}
		}
		rec.Action = "PIN"
		rec.Rehomed = false
		rec.PinAccount = owner.Account
		rec.PinConfigDir = owner.ConfigDir
		rec.Reason = reason
		return rec
	}

	// Owner blocked/throttled -> re-home its full transcript onto a healthy worker.
	var tgt Target
	if forcedTarget != nil {
		tgt = Target{Account: forcedTarget.Account, ConfigDir: forcedTarget.ConfigDir}
		rec.RehomeToSibling = forcedTarget.Account
	} else {
		availability := in.Availability
		if availability == nil && in.AvailabilityFn != nil {
			availability = in.AvailabilityFn()
		}
		targets := RehomeTargets(availability, owner.Account, nil, RehomeCap())
		if len(targets) == 0 {
			// The fleet burst-spread cap excluded every account by load. A single
			// interactive resume relaxes it onto the least-loaded healthy seat.
			relief := RehomeTargets(availability, owner.Account, nil, CapUnbounded)
			if len(relief) == 0 {
				rec.Action = "PIN_BLOCKED"
				rec.PinAccount = owner.Account
				rec.PinConfigDir = owner.ConfigDir
				rec.Reason = "owner blocked (" + blockReason + ") and no healthy Claude worker available -- pin to owner; resume waits for reset"
				return rec
			}
			rec.CapRelief = &CapRelief{
				RehomeCap: RehomeCap(),
				Note:      "all available accounts were over the fleet burst cap; a single interactive resume relaxes it onto the least-loaded healthy seat",
			}
			targets = relief
		}
		tgt = targets[0]
		if in.ProbeOwner && in.ProbeFn != nil {
			var checked []TargetProbe
			resolved := false
			limit := maxTargetProbes
			if limit > len(targets) {
				limit = len(targets)
			}
			for i := 0; i < limit; i++ {
				cand := targets[i]
				probed := in.ProbeFn(cand.Account, targetConfigDir(cand, home))
				if probed == nil {
					tgt = cand // cannot probe -> trust the ranking
					resolved = true
					break
				}
				checked = append(checked, TargetProbe{Account: cand.Account, Available: probed.Available, BlockReason: probed.BlockReason})
				if probed.Available {
					tgt = cand
					resolved = true
					break
				}
			}
			if !resolved {
				// Ran the whole bounded slice without a proven-serving target. If every
				// checked candidate probed blocked, re-homing only moves the resume from
				// one walled account to another -> pin to owner (PIN_BLOCKED).
				if allBlocked(checked) {
					rec.TargetProbes = checked
					rec.Action = "PIN_BLOCKED"
					rec.PinAccount = owner.Account
					rec.PinConfigDir = owner.ConfigDir
					rec.Reason = "owner blocked (" + blockReason + ") and every probed re-home target is also limited -- pin to owner; resume waits for reset"
					return rec
				}
				tgt = targets[0]
			}
			if len(checked) > 0 {
				rec.TargetProbes = checked
			}
		}
	}

	tgtCfg := targetConfigDir(tgt, home)
	var destSlugs []string
	if cwdSlug != "" && cwdSlug != owner.Project {
		destSlugs = []string{cwdSlug}
		rec.DestProjectSlugs = append([]string{owner.Project}, destSlugs...)
	}
	if !in.DryRun {
		if !in.RehomeFn(owner.ConfigDir, tgtCfg, owner.Project, in.SID, destSlugs) {
			rec.Action = "PIN_BLOCKED"
			rec.PinAccount = owner.Account
			rec.PinConfigDir = owner.ConfigDir
			rec.Reason = "re-home source transcript missing -- pin to owner"
			return rec
		}
		// A raw copy preserves the source mtime, so the re-homed copy would tie the
		// throttled original and the newest-mtime owner pick could re-select the walled
		// account. Stamp every re-homed copy as newest so the healthy target is the
		// unambiguous owner from now on.
		now := time.Now()
		for _, slug := range append([]string{owner.Project}, destSlugs...) {
			_ = os.Chtimes(filepath.Join(tgtCfg, "projects", slug, in.SID+".jsonl"), now, now)
		}
	}

	tgtTag := tgt.Tag
	if tgtTag == "" {
		tgtTag = tgt.Account
	}
	confirmNote := ""
	if targetProbeConfirmed(rec.TargetProbes, tgt.Account) {
		confirmNote = " (live-probe OK)"
	}
	verb := "re-homed"
	if in.DryRun {
		verb = "would re-home"
	}
	rec.Action = "REHOME"
	rec.Rehomed = !in.DryRun
	rec.WouldRehome = in.DryRun
	rec.PinAccount = tgt.Account
	rec.PinConfigDir = tgtCfg
	rec.SourceConfigDir = owner.ConfigDir
	rec.Reason = "owner blocked (" + blockReason + ") -- " + verb + " transcript onto " + tgtTag + confirmNote + " and pin there"
	return rec
}

func targetConfigDir(t Target, home string) string {
	if t.ConfigDir != "" {
		return t.ConfigDir
	}
	return filepath.Join(home, t.Account)
}

func allBlocked(checked []TargetProbe) bool {
	if len(checked) == 0 {
		return false
	}
	for _, c := range checked {
		if c.Available {
			return false
		}
	}
	return true
}

func targetProbeConfirmed(probes []TargetProbe, account string) bool {
	for _, p := range probes {
		if p.Account == account && p.Available {
			return true
		}
	}
	return false
}
