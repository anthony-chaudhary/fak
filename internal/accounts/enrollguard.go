package accounts

// Enrollment-time collision + servability guards (#3954) — the pre-write checks that make
// `fak accounts add` / `enroll-current` collision-safe instead of silently clobbering.
//
// Two distinct footguns, two distinct checks:
//
//   1. DetectEnrollCollision — before the registry write, compare the account the just-logged-in
//      credential ACTUALLY serves (its probed identity) against every seat already in the registry.
//      Enrolling a login whose account is already held by a DIFFERENT active seat silently creates a
//      duplicate — two seats collapsing onto one rate-limit bucket, one of which the rotation will
//      drop. That is the "identity hijack" a fresh enroll can commit against an existing seat, and it
//      is refused unless forced. A reconcile that rebinds a seat onto its OWN, different account is
//      enroll-current doing its job (correcting a stale binding), so it is surfaced, not refused.
//
//   2. VerifySeatServable — after the views are regenerated, confirm the new seat is actually usable:
//      serveable in the canonical (dos) registry projection AND present in both rendered roster
//      views. An enroll that "succeeds" but leaves a seat no rotation will call is a silent failure;
//      this turns it into a loud, advisory warning.
//
// Both are pure (no I/O — the caller supplies the registry and the rendered view texts) so the
// policy is unit-tested here rather than only through the CLI.

import (
	"fmt"
	"strings"
)

// EnrollCollisionKind is the closed classification of a pre-enrollment identity conflict.
type EnrollCollisionKind string

const (
	// EnrollOK: the probed account is not held by any other seat — safe to enroll.
	EnrollOK EnrollCollisionKind = "ok"
	// EnrollDuplicate: the probed account is ALREADY held by a DIFFERENT active seat. Enrolling
	// it again duplicates one rate-limit bucket across two seats; the rotation will silently drop
	// one. This is the collision a fresh add/enroll must refuse (unless forced).
	EnrollDuplicate EnrollCollisionKind = "duplicate"
	// EnrollRebind: a reconcile (--force) of an existing seat onto a DIFFERENT account than the
	// registry currently records for it. This is enroll-current legitimately correcting a stale
	// binding, so it is surfaced as an advisory note, never refused.
	EnrollRebind EnrollCollisionKind = "rebind"
	// EnrollUnknown: the credential yielded no identity to check (offline / scope-limited probe),
	// so a collision cannot be judged. Never conflated with OK — the caller warns instead of
	// refusing on no evidence.
	EnrollUnknown EnrollCollisionKind = "unknown"
)

// EnrollCollision is the result of the pre-write identity-collision check for one enrollment.
type EnrollCollision struct {
	Kind         EnrollCollisionKind `json:"kind"`
	TargetSeat   string              `json:"target_seat"`
	Account      string              `json:"account,omitempty"`       // AccountKey of the probed identity
	ConflictSeat string              `json:"conflict_seat,omitempty"` // the OTHER seat that holds the account (duplicate) or the target (rebind)
	PriorAccount string              `json:"prior_account,omitempty"` // rebind only: the account the target seat was bound to
	Detail       string              `json:"detail"`
}

// Refuse reports whether this collision should BLOCK the enroll absent an explicit --force. Only a
// duplicate (a different seat already owns the account) is a hard refusal; a rebind is enroll-current
// doing its job, and unknown/ok never block.
func (c EnrollCollision) Refuse() bool { return c.Kind == EnrollDuplicate }

// DetectEnrollCollision checks whether enrolling `targetName` with the just-probed identity `id`
// collides with an existing seat. `reg` must be the registry as recorded (its Home.Identity carrying
// the REGISTERED account for each seat) — do NOT Refresh() it first, or every seat's identity
// collapses to current disk and the binding authority is lost. The target's own same-name row is
// excluded from the duplicate scan so a legitimate reconcile is never mistaken for a collision.
func DetectEnrollCollision(reg Registry, targetName string, id ProbedIdentity) EnrollCollision {
	key := Identity{Email: id.Email, AccountUUID: id.AccountUUID}.AccountKey()
	out := EnrollCollision{TargetSeat: targetName, Account: key}
	if key == "" {
		out.Kind = EnrollUnknown
		out.Detail = "the enrolled credential yielded no identity to check for a collision (offline or scope-limited probe)"
		return out
	}
	for _, h := range reg.Homes {
		if !h.Active() {
			continue
		}
		hk := h.Identity.AccountKey()
		if h.Name == targetName {
			// Reconcile onto the target's own row: a different recorded account means we are
			// rebinding this seat (enroll-current correcting a stale binding) — surface, don't refuse.
			if hk != "" && hk != key {
				out.Kind = EnrollRebind
				out.ConflictSeat = targetName
				out.PriorAccount = hk
				out.Detail = fmt.Sprintf("seat %q was bound to %s; the live credential now serves %s — reconciling the binding", targetName, hk, key)
				return out
			}
			continue
		}
		if hk != "" && hk == key {
			out.Kind = EnrollDuplicate
			out.ConflictSeat = h.Name
			out.Detail = fmt.Sprintf("the enrolled credential serves account %s, which is already seat %q — enrolling it as %q would duplicate one rate-limit bucket across two seats", key, h.Name, targetName)
			return out
		}
	}
	out.Kind = EnrollOK
	out.Detail = fmt.Sprintf("account %s is not held by any other seat", key)
	return out
}

// ServableReport says whether a just-enrolled seat is actually usable across both roster views.
type ServableReport struct {
	Seat        string      `json:"seat"`
	Servable    bool        `json:"servable"` // dos-serveable AND present in both rendered views
	DosServable bool        `json:"dos_servable"`
	LoginStatus LoginStatus `json:"login_status"`
	InDosView   bool        `json:"in_dos_view"`
	InJobView   bool        `json:"in_job_view"`
	Detail      string      `json:"detail"`
}

// VerifySeatServable confirms the seat is usable after a sync: it must be serveable in the canonical
// registry projection (the dos view's authority — active, enabled, credentialed, identity-true) AND
// appear in both rendered roster view texts. `reg` should be Refresh()ed by the caller so LoginStatus
// reflects current disk. `dosViewText`/`jobViewText` are the just-written view files' contents; an
// empty string for either means "not checked" (unreadable view), recorded but not fatal. This is an
// advisory witness — a false here means the enroll landed but the seat is not yet callable, which the
// caller surfaces as a warning rather than failing the enroll it already committed.
func VerifySeatServable(reg Registry, seat, dosViewText, jobViewText string) ServableReport {
	rep := ServableReport{Seat: seat}
	var found *Home
	for i := range reg.Homes {
		if reg.Homes[i].Name == seat {
			found = &reg.Homes[i]
			break
		}
	}
	if found == nil {
		rep.Detail = fmt.Sprintf("seat %q is not in the registry after sync", seat)
		return rep
	}
	rep.LoginStatus = found.LoginStatus()
	rep.DosServable = found.CanServe()
	rep.InDosView = seatInViewText(dosViewText, seat)
	rep.InJobView = seatInViewText(jobViewText, seat)
	rep.Servable = rep.DosServable && rep.InDosView && rep.InJobView
	switch {
	case rep.Servable:
		rep.Detail = fmt.Sprintf("seat %q is serveable and present in both roster views", seat)
	case !rep.DosServable:
		rep.Detail = fmt.Sprintf("seat %q is enrolled but not serveable: login status %q", seat, rep.LoginStatus)
	case !rep.InDosView:
		rep.Detail = fmt.Sprintf("seat %q serves but is missing from the dos roster view", seat)
	case !rep.InJobView:
		rep.Detail = fmt.Sprintf("seat %q serves but is missing from the job roster view", seat)
	}
	return rep
}

// seatInViewText reports whether a rendered roster view names the seat. The views render the seat as
// a `name: <seat>` (or `- <seat>`) row, so a whole-token match on the name is a cheap, render-format
// agnostic presence check. An empty text is "not checked" (false).
func seatInViewText(viewText, seat string) bool {
	if viewText == "" || seat == "" {
		return false
	}
	for _, line := range strings.Split(viewText, "\n") {
		if strings.Contains(line, seat) {
			return true
		}
	}
	return false
}
