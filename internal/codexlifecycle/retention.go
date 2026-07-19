// retention.go — the #4765 bound: a retention contract and byte/age cap for the
// native Codex rollout archive (~/.codex/sessions), planned WITHOUT losing
// witnesses.
//
// THE PROBLEM IT FENCES. The 2026-07-14 audit found 2,125 sessions / 1.92 GB of
// append-only rollouts with no retention policy. Deleting blindly destroys the
// compaction/cost/task witnesses #2822/#3152/#2810 consume; keeping everything
// raw grows an unmanaged evidence store forever. This file is the PURE planning
// half: metadata in, a typed plan out. It never opens a rollout, never reads a
// body, and never touches the filesystem or the clock — Now is injected and a
// clockless policy is refused, so a plan is reproducible evidence.
//
// THE RETENTION CONTRACT (#4765's four classes):
//
//   - active               — a live writer or a recently-modified rollout.
//     Never selected: an open session may still be written.
//   - warm_evidence        — inside the warm window, OR protected by an active
//     goal, guard witness, unresolved issue, or explicit pin. Protection is
//     absolute: a protected session is NEVER selected for expiry, even when
//     that leaves the byte cap unmet. The plan reports the unmet bound
//     honestly (CapSatisfied=false + ProtectedOverCapBytes) instead of
//     sacrificing a witness to satisfy arithmetic.
//   - compacted  — already reduced to a scrubbed aggregate (bodies
//     gone, token/tool-shape/compaction analytics kept). Retained as-is.
//   - expired              — unprotected, not active, and either past the warm
//     window (reason age_past_warm_window) or selected oldest-first to bring
//     retained raw bytes under RawBytesCap (reason raw_bytes_over_cap).
//
// Expiry here means "eligible for compaction into an aggregate", not deletion:
// the destructive step lives behind the dry-run-first maintenance verb and its
// quarantine/grace store, which BoundQuarantine bounds by age and bytes with
// the same oldest-first, receipt-emitting discipline.
package codexlifecycle

import (
	"errors"
	"sort"
	"time"
)

// RetentionClass is the closed retention vocabulary for one rollout session.
type RetentionClass string

const (
	ClassActive       RetentionClass = "active"
	ClassWarmEvidence RetentionClass = "warm_evidence"
	ClassCompacted    RetentionClass = "compacted"
	ClassExpired      RetentionClass = "expired"
)

// ProtectReason says WHY a session may not expire. A session carrying any
// reason is warm evidence regardless of age or size.
type ProtectReason string

const (
	ProtectActiveGoal      ProtectReason = "active_goal"
	ProtectRefereeEvidence ProtectReason = "referee_evidence"
	ProtectUnresolvedIssue ProtectReason = "unresolved_issue"
	ProtectPin             ProtectReason = "pin"
)

// Expiry reason tokens, machine-readable in the manifest.
const (
	ReasonAgePastWarmWindow = "age_past_warm_window"
	ReasonRawBytesOverCap   = "raw_bytes_over_cap"
	ReasonQuarantineAge     = "quarantine_past_grace"
	ReasonQuarantineBytes   = "quarantine_bytes_over_cap"
)

// SessionRecord is the metadata a plan consumes — never prompt or tool-output
// bodies, which keeps the whole planning path privacy-safe by construction.
type SessionRecord struct {
	ID        string          `json:"id"`
	Bytes     int64           `json:"bytes"`
	ModTime   time.Time       `json:"mod_time"`
	Live      bool            `json:"live,omitempty"`      // open writer (#4785 Live outcome)
	Compacted bool            `json:"compacted,omitempty"` // already a scrubbed aggregate
	Protected []ProtectReason `json:"protected,omitempty"`
}

// RetentionContract bounds the archive. Now is REQUIRED: reading the clock here
// would make a plan unreproducible, so a zero Now is refused.
type RetentionContract struct {
	Now          time.Time     `json:"now"`
	ActiveWithin time.Duration `json:"active_within"` // mtime inside → active
	WarmWithin   time.Duration `json:"warm_within"`   // mtime inside → warm evidence
	// RawBytesCap bounds retained RAW bytes (active + warm, i.e. everything
	// not yet compacted). 0 means no byte cap: age expiry still applies.
	RawBytesCap int64 `json:"raw_bytes_cap"`
}

// Decision is one session's classification and (non-destructive) fate.
type Decision struct {
	ID        string          `json:"id"`
	Class     RetentionClass  `json:"class"`
	Bytes     int64           `json:"bytes"`
	Protected []ProtectReason `json:"protected,omitempty"`
	Expire    bool            `json:"expire,omitempty"`
	Reason    string          `json:"reason,omitempty"`
}

// RetentionManifest is the machine-readable manifest and receipt: every session's
// decision plus the before/after/reclaimed arithmetic, in input order.
type RetentionManifest struct {
	Decisions []Decision `json:"decisions"`

	BeforeBytes    int64 `json:"before_bytes"`    // raw (non-compacted) bytes going in
	ReclaimedBytes int64 `json:"reclaimed_bytes"` // raw bytes scheduled for compaction
	AfterBytes     int64 `json:"after_bytes"`     // raw bytes retained if applied
	RawBytesCap    int64 `json:"raw_bytes_cap"`

	// CapSatisfied is false ONLY when protected/active evidence alone keeps
	// AfterBytes over the cap — the honest "cannot bound without destroying
	// witnesses" signal, with the overage quantified.
	CapSatisfied          bool  `json:"cap_satisfied"`
	ProtectedOverCapBytes int64 `json:"protected_over_cap_bytes,omitempty"`
}

// DecideRetention classifies every session under the contract above and selects
// expiries — age first, then oldest-first by bytes until the cap holds. It is
// deterministic: same records + same policy ⇒ byte-identical plan.
func DecideRetention(sessions []SessionRecord, pol RetentionContract) (RetentionManifest, error) {
	if pol.Now.IsZero() {
		return RetentionManifest{}, errors.New("codexlifecycle: RetentionContract.Now is required (clockless plans are not reproducible)")
	}
	plan := RetentionManifest{RawBytesCap: pol.RawBytesCap, Decisions: make([]Decision, len(sessions))}
	for i, s := range sessions {
		d := Decision{ID: s.ID, Bytes: s.Bytes, Protected: s.Protected}
		age := pol.Now.Sub(s.ModTime)
		switch {
		case s.Compacted:
			d.Class = ClassCompacted
		case s.Live || age <= pol.ActiveWithin:
			d.Class = ClassActive
		case len(s.Protected) > 0:
			d.Class = ClassWarmEvidence
		case age <= pol.WarmWithin:
			d.Class = ClassWarmEvidence
		default:
			d.Class = ClassExpired
			d.Expire = true
			d.Reason = ReasonAgePastWarmWindow
		}
		if !s.Compacted {
			plan.BeforeBytes += s.Bytes
			if d.Expire {
				plan.ReclaimedBytes += s.Bytes
			}
		}
		plan.Decisions[i] = d
	}
	plan.AfterBytes = plan.BeforeBytes - plan.ReclaimedBytes

	// Byte cap: take unprotected warm sessions oldest-first until the bound
	// holds. Active and protected sessions are never candidates.
	if pol.RawBytesCap > 0 && plan.AfterBytes > pol.RawBytesCap {
		type cand struct {
			idx int
			mod time.Time
		}
		var cands []cand
		for i, d := range plan.Decisions {
			if d.Class == ClassWarmEvidence && len(d.Protected) == 0 {
				cands = append(cands, cand{i, sessions[i].ModTime})
			}
		}
		sort.Slice(cands, func(a, b int) bool {
			if !cands[a].mod.Equal(cands[b].mod) {
				return cands[a].mod.Before(cands[b].mod)
			}
			return plan.Decisions[cands[a].idx].ID < plan.Decisions[cands[b].idx].ID
		})
		for _, c := range cands {
			if plan.AfterBytes <= pol.RawBytesCap {
				break
			}
			d := &plan.Decisions[c.idx]
			d.Class = ClassExpired
			d.Expire = true
			d.Reason = ReasonRawBytesOverCap
			plan.ReclaimedBytes += d.Bytes
			plan.AfterBytes -= d.Bytes
		}
	}
	plan.CapSatisfied = pol.RawBytesCap <= 0 || plan.AfterBytes <= pol.RawBytesCap
	if !plan.CapSatisfied {
		plan.ProtectedOverCapBytes = plan.AfterBytes - pol.RawBytesCap
	}
	return plan, nil
}

// QuarantineItem is one raw payload parked in the grace store by the
// maintenance verb, awaiting either restore or final purge.
type QuarantineItem struct {
	ID            string    `json:"id"`
	Bytes         int64     `json:"bytes"`
	QuarantinedAt time.Time `json:"quarantined_at"`
}

// QuarantineBound bounds the grace store by age and bytes.
type QuarantineBound struct {
	Now      time.Time     `json:"now"`
	MaxAge   time.Duration `json:"max_age"`   // grace period; older entries purge
	MaxBytes int64         `json:"max_bytes"` // 0 = no byte bound
}

// QuarantineReceipt reports what the bound keeps and purges, with the same
// before/after/reclaimed arithmetic as a retention plan.
type QuarantineReceipt struct {
	Keep           []QuarantineItem `json:"keep"`
	Purge          []Decision       `json:"purge,omitempty"`
	BeforeBytes    int64            `json:"before_bytes"`
	ReclaimedBytes int64            `json:"reclaimed_bytes"`
	AfterBytes     int64            `json:"after_bytes"`
}

// BoundQuarantine applies the age bound, then the byte bound oldest-first, and
// returns the receipt. Like DecideRetention it is pure and clock-refusing.
func BoundQuarantine(items []QuarantineItem, b QuarantineBound) (QuarantineReceipt, error) {
	if b.Now.IsZero() {
		return QuarantineReceipt{}, errors.New("codexlifecycle: QuarantineBound.Now is required")
	}
	sorted := make([]QuarantineItem, len(items))
	copy(sorted, items)
	sort.Slice(sorted, func(i, j int) bool {
		if !sorted[i].QuarantinedAt.Equal(sorted[j].QuarantinedAt) {
			return sorted[i].QuarantinedAt.Before(sorted[j].QuarantinedAt)
		}
		return sorted[i].ID < sorted[j].ID
	})
	var rec QuarantineReceipt
	var kept []QuarantineItem
	for _, it := range sorted {
		rec.BeforeBytes += it.Bytes
		if b.MaxAge > 0 && b.Now.Sub(it.QuarantinedAt) > b.MaxAge {
			rec.Purge = append(rec.Purge, Decision{ID: it.ID, Class: ClassExpired, Bytes: it.Bytes, Expire: true, Reason: ReasonQuarantineAge})
			rec.ReclaimedBytes += it.Bytes
			continue
		}
		kept = append(kept, it)
		rec.AfterBytes += it.Bytes
	}
	// Byte bound: purge oldest survivors first until under MaxBytes.
	if b.MaxBytes > 0 {
		i := 0
		for rec.AfterBytes > b.MaxBytes && i < len(kept) {
			it := kept[i]
			rec.Purge = append(rec.Purge, Decision{ID: it.ID, Class: ClassExpired, Bytes: it.Bytes, Expire: true, Reason: ReasonQuarantineBytes})
			rec.ReclaimedBytes += it.Bytes
			rec.AfterBytes -= it.Bytes
			i++
		}
		kept = kept[i:]
	}
	rec.Keep = kept
	return rec, nil
}
