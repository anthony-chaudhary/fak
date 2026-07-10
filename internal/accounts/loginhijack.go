package accounts

// Login-time identity-hijack detection (#3953) — the root-cause guard for the failure #3987 only
// makes recoverable and #3215 only makes visible after the fact. An in-harness `/login` (or any
// `claude setup-token`) into a seat's config dir rewrites its .credentials.json IN PLACE. If it is
// pointed at the WRONG dir, that dir's credential now serves a DIFFERENT account than the registry
// bound the seat to — silently. The roster keeps calling the seat by its old name while it burns a
// different account's quota, and the displaced account may have just lost its only live credential.
//
// This file adds the primitive that catches it the moment it happens: compare the seat's REGISTERED
// identity (Home.Identity — what the registry says this seat is) against the account its live
// credential ACTUALLY serves now (the #3215 probe). A disagreement is a hijack. It is distinct from
// #3215's `identity_metadata_stale`, which compares the dir's own .claude.json metadata against its
// credential; here the authority is the REGISTRY's binding, so a login that rewrites BOTH the
// credential and the metadata in lockstep — invisible to the stale check — is still caught.

import "fmt"

// LoginWarningIdentityHijack is the warning surfaced when a seat's live credential serves a
// different account than the registry bound the seat to — a /login rebound the dir. It sits beside
// #3215's LoginWarningIdentityStale in the login-warning vocabulary; the two answer different
// questions (registry-vs-credential here, disk-metadata-vs-credential there).
const LoginWarningIdentityHijack LoginWarning = "identity_login_hijack"

// HijackVerdict is the closed classification of a login-time identity check for one seat.
type HijackVerdict string

const (
	// HijackOK: the live credential serves the account the registry bound the seat to.
	HijackOK HijackVerdict = "ok"
	// HijackDetected: the live credential serves a DIFFERENT account than the registry expects —
	// a login rebound this dir. This is the one verdict a caller must act on.
	HijackDetected HijackVerdict = "hijacked"
	// HijackUnbound: the registry has no identity recorded for the seat, so there is nothing to
	// check a login against (a freshly-added seat before its first identity derive).
	HijackUnbound HijackVerdict = "unbound"
	// HijackUnprobed: no live session credential was present, or no prober was supplied, or the
	// probe failed — the check could not run. Never conflated with OK.
	HijackUnprobed HijackVerdict = "unprobed"
)

// HijackReport is the result of one seat's login-time hijack check.
type HijackReport struct {
	Seat     string         `json:"seat"`
	Dir      string         `json:"dir"`
	Verdict  HijackVerdict  `json:"verdict"`
	Expected Identity       `json:"expected"` // the registry-bound identity for the seat
	Actual   ProbedIdentity `json:"actual"`   // what the live credential actually serves
	Detail   string         `json:"detail"`   // one-line human explanation
	ProbeErr string         `json:"probe_error,omitempty"`
}

// Hijacked reports whether this check found a rebind — the single bit a guard/CLI acts on.
func (r HijackReport) Hijacked() bool { return r.Verdict == HijackDetected }

// DetectLoginHijack checks whether the seat's live credential in `dir` still serves `expected` (the
// registry-bound identity). It reuses the #3215 credential probe: `expected` is the authority, the
// probed credential is ground truth for what the dir serves NOW, and a disagreement is a hijack. A
// missing expected identity, missing credential, or probe failure yields a non-OK, non-hijack
// verdict — the check refuses to guess, never a false alarm and never a false all-clear.
func DetectLoginHijack(seat, dir string, expected Identity, probe IdentityProber) HijackReport {
	rep := HijackReport{Seat: seat, Dir: dir, Expected: expected}
	if expected.Email == "" && expected.AccountUUID == "" {
		rep.Verdict = HijackUnbound
		rep.Detail = "seat has no registered identity to check a login against"
		return rep
	}
	res := ResolveCredentialIdentity(dir, probe)
	if !res.Probed {
		rep.Verdict = HijackUnprobed
		if res.ProbeErr != nil {
			rep.ProbeErr = res.ProbeErr.Error()
			rep.Detail = "live credential could not be probed: " + res.ProbeErr.Error()
		} else {
			rep.Detail = "no live session credential present to probe"
		}
		return rep
	}
	rep.Actual = res.Credential
	if identitiesDisagree(expected, res.Credential) {
		rep.Verdict = HijackDetected
		rep.Detail = fmt.Sprintf("seat %q is registered to %s but its live credential now serves %s — a /login rebound this dir to a different account",
			seat, identityLabel(expected), identityLabel(Identity{Email: res.Credential.Email, AccountUUID: res.Credential.AccountUUID}))
		return rep
	}
	rep.Verdict = HijackOK
	rep.Detail = "live credential serves the registered account"
	return rep
}

// ScanLoginHijacks runs DetectLoginHijack over every live seat in the registry — the fleet-wide
// read a guard hook or `fak accounts status --probe`/`check-hijack` consumes. Only seats with a
// config dir are checked. Reports are returned in registry order so the output is stable.
func ScanLoginHijacks(reg Registry, probe IdentityProber) []HijackReport {
	var out []HijackReport
	for _, h := range reg.Homes {
		if !h.Active() || h.Dir == "" {
			continue
		}
		out = append(out, DetectLoginHijack(h.Name, h.Dir, h.Identity, probe))
	}
	return out
}

// AnyHijacked reports whether any report in the set is a detected hijack — the one-bit gate a guard
// hook uses to decide whether to warn.
func AnyHijacked(reports []HijackReport) bool {
	for _, r := range reports {
		if r.Hijacked() {
			return true
		}
	}
	return false
}
