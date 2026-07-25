package executionroute

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

// ---------------------------------------------------------------------------
// LIVE HARNESS HEALTH — a measured input that excludes unhealthy candidates.
// ---------------------------------------------------------------------------

// HealthState is the measured serving posture of a harness candidate. It is a
// CLOSED vocabulary: only HealthAvailable is eligible; the other three are
// deterministic exclusions, each carrying its own rejection reason. The three
// signals the envelope names map onto it — availability (Unavailable), cooldown
// (Cooldown), and saturation (Draining, plus the zero-free-slots gate below).
type HealthState string

const (
	// HealthAvailable: the candidate is serving and has spare capacity.
	HealthAvailable HealthState = "available"
	// HealthDraining: the candidate is not blocked but has no free capacity
	// (process saturation) — every seat is at its session cap.
	HealthDraining HealthState = "draining"
	// HealthCooldown: the candidate is throttled behind a usage cap and returns
	// only after its reset window elapses.
	HealthCooldown HealthState = "cooldown"
	// HealthUnavailable: the candidate cannot serve at all (auth/access block).
	HealthUnavailable HealthState = "unavailable"
)

// HarnessHealth is one MEASURED health reading for a harness candidate. It carries
// the three routing signals (availability via State, cooldown via State+Detail,
// saturation via FreeSlots/SessionCap) plus EXPLICIT freshness (AgeSeconds) and
// provenance (Source). It is pure data: the reading's age is a supplied number,
// not a clock read, so harness selection stays deterministic and testable.
type HarnessHealth struct {
	State      HealthState `json:"state"`
	FreeSlots  int         `json:"free_slots"`
	SessionCap int         `json:"session_cap,omitempty"`
	// Detail explains a non-available state (block/cooldown reason, reset time).
	Detail string `json:"detail,omitempty"`
	// Source names where the reading came from (provenance), e.g. the fleet
	// health source and its status_source.
	Source string `json:"source,omitempty"`
	// AgeSeconds is how old the reading is. A negative age means the source
	// stamped no timestamp — treated as unfresh whenever a freshness bound is set.
	AgeSeconds int64 `json:"age_seconds"`
}

// HealthReport is the live health input consulted during harness selection. When
// Candidates is empty the health gate is inert and selection falls back to the
// static requirements alone, so existing callers are unaffected. Keys are
// candidate ids or harness-profile names/aliases, matched after normalization.
type HealthReport struct {
	Candidates map[string]HarnessHealth `json:"candidates,omitempty"`
	// MaxAgeSeconds bounds freshness: a reading older than this (or one with an
	// unknown age) is stale, and its candidate is excluded. 0 disables the bound.
	MaxAgeSeconds int64 `json:"max_age_seconds,omitempty"`
	// RequireEvidence, when true, excludes a candidate that has NO reading (fail
	// closed). When false, an unmeasured candidate passes the health gate.
	RequireEvidence bool `json:"require_evidence,omitempty"`
}

// HarnessRejection records one skipped harness candidate and the deterministic
// reason it was excluded — an unknown profile, an unsatisfied static requirement,
// or a live-health exclusion. Every candidate skipped before the winner is
// recorded, in operator candidate order, so the trace is auditable.
type HarnessRejection struct {
	Candidate string `json:"candidate"`
	Reason    string `json:"reason"`
}

// active reports whether the report gates selection at all.
func (r HealthReport) active() bool { return len(r.Candidates) > 0 }

// lookup finds the reading for a candidate, trying each supplied key (the
// candidate id, the profile's canonical name, and its aliases) after
// normalization. The first hit wins.
func (r HealthReport) lookup(keys ...string) (HarnessHealth, bool) {
	for _, k := range keys {
		if h, ok := r.Candidates[normalize(k)]; ok {
			return h, true
		}
	}
	return HarnessHealth{}, false
}

// healthReason returns the rejection reason for a candidate whose live health
// excludes it, and false when the candidate passes the health gate. The order of
// checks is deterministic: freshness first (a stale reading is never trusted),
// then the state vocabulary, then saturation for an otherwise-available seat.
func (r HealthReport) healthReason(keys ...string) (string, bool) {
	if !r.active() {
		return "", false
	}
	h, ok := r.lookup(keys...)
	if !ok {
		if r.RequireEvidence {
			return "no live health evidence", true
		}
		return "", false
	}
	prov := h.Source
	if prov == "" {
		prov = "unknown source"
	}
	if r.MaxAgeSeconds > 0 {
		if h.AgeSeconds < 0 {
			return fmt.Sprintf("stale health: no timestamp (source %s)", prov), true
		}
		if h.AgeSeconds > r.MaxAgeSeconds {
			return fmt.Sprintf("stale health: age %ds exceeds %ds bound (source %s)", h.AgeSeconds, r.MaxAgeSeconds, prov), true
		}
	}
	switch h.State {
	case HealthAvailable:
		// serving — fall through to the saturation check below.
	case HealthDraining:
		return withDetail("draining", h.Detail), true
	case HealthCooldown:
		return withDetail("cooldown", h.Detail), true
	case HealthUnavailable:
		return withDetail("unavailable", h.Detail), true
	default:
		return withDetail("unavailable: unrecognized state "+string(h.State), h.Detail), true
	}
	if h.SessionCap > 0 && h.FreeSlots <= 0 {
		return fmt.Sprintf("saturated: 0/%d free slots", h.SessionCap), true
	}
	return "", false
}

func withDetail(state, detail string) string {
	if strings.TrimSpace(detail) == "" {
		return state
	}
	return state + ": " + detail
}

// requirementReason returns why a profile fails the static harness requirements,
// or false when it satisfies them. Split out of selectHarness so a skipped
// candidate can name the specific requirement it missed.
func requirementReason(p harnessprofile.HarnessProfile, req HarnessRequirements) (string, bool) {
	if req.Wire != "" && p.Wire != req.Wire {
		return fmt.Sprintf("wire %q does not satisfy required %q", p.Wire, req.Wire), true
	}
	if req.Repoint != "" && !p.HasRepoint(req.Repoint) {
		return fmt.Sprintf("no %q repoint mechanism", req.Repoint), true
	}
	if req.Rotatable && (p.ConfigHomeGlob == "" || p.Identity == harnessprofile.IdentityNone) {
		return "not rotatable (no account-rotation identity)", true
	}
	return "", false
}

// healthKeys is the ordered set of names a candidate's health may be keyed under:
// the operator's harness id first, then the resolved profile's canonical name and
// its declared aliases. This is what lets a fleet source keyed by product
// ("opencode") populate a candidate named by its profile id ("openai-generic").
func healthKeys(harnessID string, p harnessprofile.HarnessProfile) []string {
	keys := make([]string, 0, 2+len(p.Names))
	keys = append(keys, harnessID, p.Name)
	keys = append(keys, p.Names...)
	return keys
}
