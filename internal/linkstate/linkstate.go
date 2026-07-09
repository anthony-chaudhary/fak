// Package linkstate is the general "what state is my channel with a peer in
// right now?" record — the reusable comms-protocol view that the lab dispatch
// gate (internal/fleet) is one specialization of.
//
// It replaces an older five-state lab vocabulary (one ready + four different
// not-ready states) that left agents "wedged": unsure which not-ready they were
// in or what to do next. The model here is deliberately three-phase:
//
//   - CLEAR   — the channel is up and the peer is idle & healthy; safe to dispatch.
//   - WORKING — a job/exchange is actively in flight; the peer is busy, wait it out.
//   - WAITING — not ready: blocked on recovery, gateway, auth, or simply unknown.
//
// "I don't know yet" is not a fourth phase — it is a form of WAITING (wait and
// re-check). The fine cause is demoted to a closed `detail` sub-vocabulary for
// operators. Every phase carries a mandatory next_action, so an agent reading a
// record can ALWAYS name its next step — being wedged is structurally impossible.
//
// Two load-bearing invariants carry over from the vocabulary this generalizes:
//   - admit_dispatch is ALWAYS derived (phase == CLEAR), never trusted from the
//     serialized record — a lying admit bit cannot open the gate.
//   - every string field is a generic token (see Tokenish): no host, channel,
//     token, thread id, transcript, or private path can ride across the boundary.
package linkstate

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// Schema is the versioned wire identity written by this package.
const Schema = "fak.link_state/v1"

// Phase is the primary, three-value link-state vocabulary.
type Phase string

const (
	// Waiting: the channel is not ready — blocked on recovery/gateway/auth, or
	// indeterminate. Never admits dispatch. The specific cause is in Detail.
	Waiting Phase = "WAITING"
	// Clear: channel up, peer idle & healthy — the only phase that admits dispatch.
	Clear Phase = "CLEAR"
	// Working: a job/exchange is in flight; the peer is busy. Does not admit new work.
	Working Phase = "WORKING"
)

// The closed `detail` sub-vocabulary — the demoted fine cause. Exactly one detail
// belongs to CLEAR (ready) and one to WORKING (job-in-flight); the rest are the
// distinct flavors of WAITING an operator may want to act on.
const (
	DetailReady           = "ready"                   // CLEAR
	DetailJobInFlight     = "job-in-flight"           // WORKING
	DetailGatewayDown     = "gateway-unreachable"     // WAITING
	DetailAuthBlocked     = "auth-or-channel-blocked" // WAITING
	DetailPrivateRecovery = "private-recovery"        // WAITING
	DetailIndeterminate   = "indeterminate"           // WAITING
)

// State is the generic link-state record. It is intentionally free of any
// domain identifiers: Subject names the peer/channel/box the state describes
// with a generic label only (a machine class, not a real host or channel).
type State struct {
	Schema        string `json:"schema"`
	Subject       string `json:"subject"`
	CheckedAt     string `json:"checked_at"`
	Phase         Phase  `json:"phase"`
	Detail        string `json:"detail"`
	NextAction    string `json:"next_action"`
	Evidence      string `json:"evidence"`
	AdmitDispatch bool   `json:"admit_dispatch"`
}

// New builds a validated-shape State, filling the phase-appropriate defaults for
// any empty field and deriving admit_dispatch from the phase. An empty phase
// fails safe to WAITING (never CLEAR).
func New(subject string, phase Phase, detail, nextAction, evidence string, checkedAt time.Time) State {
	if subject == "" {
		subject = "peer"
	}
	if checkedAt.IsZero() {
		checkedAt = time.Now()
	}
	if phase == "" {
		phase = Waiting
	}
	if detail == "" {
		detail = DefaultDetail(phase)
	}
	if nextAction == "" {
		nextAction = defaultNextAction(phase, detail)
	}
	if evidence == "" {
		evidence = defaultEvidence(detail)
	}
	return State{
		Schema:        Schema,
		Subject:       subject,
		CheckedAt:     checkedAt.UTC().Format(time.RFC3339),
		Phase:         phase,
		Detail:        detail,
		NextAction:    nextAction,
		Evidence:      evidence,
		AdmitDispatch: phase == Clear,
	}
}

// Indeterminate is the fail-safe "no usable signal" record: WAITING with the
// indeterminate detail, so it never admits dispatch but still names a next step.
func Indeterminate(subject, nextAction, evidence string, checkedAt time.Time) State {
	return New(subject, Waiting, DetailIndeterminate, nextAction, evidence, checkedAt)
}

// Load decodes a native fak.link_state/v1 record, rejecting any unknown/private
// field and re-deriving admit_dispatch from the phase (never trusting the file).
// Legacy fak.lab_readiness/v1 records carry lab-specific fields (commands, the old
// status vocabulary) and are read via the shim in internal/fleet, which owns that
// schema and uses Coarsen to fold the old five states onto a phase.
func Load(r io.Reader) (State, error) {
	var out State
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&out); err != nil {
		return State{}, err
	}
	if out.Schema == "" {
		out.Schema = Schema
	}
	if out.Subject == "" {
		out.Subject = "peer"
	}
	out.AdmitDispatch = out.Phase == Clear
	if probs := out.Validate(); len(probs) > 0 {
		return State{}, fmt.Errorf("%s", strings.Join(probs, "; "))
	}
	return out, nil
}

// Validate enforces the closed vocabularies, phase/detail consistency, and the
// generic-token scrub. It returns a slice of human-readable problems (empty == ok).
func (s State) Validate() []string {
	var probs []string
	if s.Schema != "" && s.Schema != Schema {
		probs = append(probs, fmt.Sprintf("unsupported schema %q (want %s)", s.Schema, Schema))
	}
	if !knownPhase(s.Phase) {
		probs = append(probs, fmt.Sprintf("phase %q is not in the closed link-state vocabulary", s.Phase))
	}
	if !knownDetail(s.Detail) {
		probs = append(probs, fmt.Sprintf("detail %q is not in the closed link-state detail vocabulary", s.Detail))
	} else if knownPhase(s.Phase) && phaseForDetail(s.Detail) != s.Phase {
		probs = append(probs, fmt.Sprintf("detail %q is inconsistent with phase %q", s.Detail, s.Phase))
	}
	for field, value := range map[string]string{
		"subject":     s.Subject,
		"next_action": s.NextAction,
		"evidence":    s.Evidence,
	} {
		if value == "" {
			probs = append(probs, field+" is required")
		} else if !Tokenish(value) {
			probs = append(probs, field+" must be a generic token-like value")
		}
	}
	if s.CheckedAt != "" {
		if _, err := time.Parse(time.RFC3339, s.CheckedAt); err != nil {
			probs = append(probs, "checked_at must be RFC3339")
		}
	}
	return probs
}

// Coarsen folds a legacy fak.lab_readiness/v1 status onto a (phase, detail). It is
// the migration bridge for records written before this package existed. Any status
// that is not the single ready state — including an unrecognized one — coarsens to
// WAITING, so a stale or malformed legacy record can never re-open the gate.
func Coarsen(legacyStatus string) (Phase, string) {
	switch legacyStatus {
	case "READY_FOR_DEV_WORK":
		return Clear, DetailReady
	case "WAIT_PRIVATE_RECOVERY":
		return Waiting, DetailPrivateRecovery
	case "GATEWAY_UNREACHABLE":
		return Waiting, DetailGatewayDown
	case "AUTH_OR_CHANNEL_BLOCKED":
		return Waiting, DetailAuthBlocked
	default: // INDETERMINATE and anything unknown fail safe to WAITING.
		return Waiting, DetailIndeterminate
	}
}

// DefaultDetail is the canonical detail for a phase when none is supplied.
func DefaultDetail(phase Phase) string {
	switch phase {
	case Clear:
		return DetailReady
	case Working:
		return DetailJobInFlight
	default:
		return DetailIndeterminate
	}
}

func knownPhase(p Phase) bool {
	switch p {
	case Waiting, Clear, Working:
		return true
	default:
		return false
	}
}

func knownDetail(d string) bool {
	switch d {
	case DetailReady, DetailJobInFlight, DetailGatewayDown, DetailAuthBlocked, DetailPrivateRecovery, DetailIndeterminate:
		return true
	default:
		return false
	}
}

// phaseForDetail maps each detail to the one phase it may appear under.
func phaseForDetail(detail string) Phase {
	switch detail {
	case DetailReady:
		return Clear
	case DetailJobInFlight:
		return Working
	default:
		return Waiting
	}
}

func defaultNextAction(phase Phase, detail string) string {
	switch phase {
	case Clear:
		return "admit-dispatch"
	case Working:
		return "await-completion"
	default: // Waiting: steer by the specific cause.
		switch detail {
		case DetailGatewayDown:
			return "recover-gateway"
		case DetailAuthBlocked:
			return "fix-auth-or-channel"
		case DetailPrivateRecovery:
			return "confirm-control-session"
		default:
			return "publish-link-state"
		}
	}
}

func defaultEvidence(detail string) string {
	if detail == DetailIndeterminate {
		return "no-state-record"
	}
	return "scrubbed-readback"
}

// Tokenish reports whether value is a single generic token — trimmed, non-empty,
// and made only of [A-Za-z0-9_-]. It is the scrub gate that keeps hosts, channels,
// tokens, thread ids, and paths (which carry ':', '.', '/', or spaces) out of a
// record that crosses the public boundary.
func Tokenish(value string) bool {
	if strings.TrimSpace(value) != value || value == "" {
		return false
	}
	for _, ch := range value {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			continue
		}
		return false
	}
	return true
}
