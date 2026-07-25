// Package executionroute composes independently useful routing decisions into one
// execution envelope. It chooses the harness, model plan, and session lifecycle
// together while preserving each leaf's own policy and vocabulary.
package executionroute

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// SessionAction is the lifecycle operation selected for this invocation.
type SessionAction string

const (
	SessionStart         SessionAction = "start"
	SessionResume        SessionAction = "resume"
	SessionFork          SessionAction = "fork"
	SessionCompactResume SessionAction = "compact_resume"
)

// HarnessRequirements describes execution-plane properties the selected harness
// must provide. Empty requirements accept every declared profile.
type HarnessRequirements struct {
	Wire      harnessprofile.Wire             `json:"wire,omitempty"`
	Repoint   harnessprofile.RepointMechanism `json:"repoint,omitempty"`
	Rotatable bool                            `json:"rotatable,omitempty"`
}

// SessionSubject carries only signals needed to choose session lifecycle. The
// caller owns persistence and compaction; this package chooses, it does not execute.
//
// Portability has two channels, and the descriptor channel wins. When both Source
// and Target descriptors are supplied, eligibility is COMPUTED from them by
// RouteCompat — a caller can no longer assert Portable over a move whose state
// could never survive. The Portable / PreserveContinuity booleans remain the
// fallback for callers that have not yet built a descriptor.
type SessionSubject struct {
	ID                 string  `json:"id,omitempty"`
	PreserveContinuity bool    `json:"preserve_continuity,omitempty"`
	Portable           bool    `json:"portable,omitempty"`
	ContextUtilization float64 `json:"context_utilization,omitempty"`
	CompactAt          float64 `json:"compact_at,omitempty"`

	// Source describes the existing session's own execution envelope; Target
	// describes the envelope it would move into. Both must be present to compute
	// eligibility; either alone leaves the boolean fallback in charge.
	Source *SessionDescriptor `json:"source,omitempty"`
	Target *SessionDescriptor `json:"target,omitempty"`
}

// Request is the cross-plane subject routed as one unit. Health, when populated,
// is the live harness-candidate health input: it excludes unavailable / draining
// / cooldown / stale candidates before the static requirements pick a winner. An
// empty Health leaves selection on the static requirements alone.
type Request struct {
	HarnessCandidates []string            `json:"harness_candidates,omitempty"`
	Harness           HarnessRequirements `json:"harness_requirements,omitempty"`
	Health            HealthReport        `json:"health,omitempty"`
	Model             modelroute.Subject  `json:"model"`
	Session           SessionSubject      `json:"session"`
}

// HarnessDecision records the selected profile, why it won, and — for an audit of
// the live-health/requirement gate — every candidate skipped ahead of it with the
// deterministic reason it was excluded. Rejected preserves operator order.
type HarnessDecision struct {
	Profile  harnessprofile.HarnessProfile `json:"profile"`
	Reason   string                        `json:"reason"`
	Rejected []HarnessRejection            `json:"rejected,omitempty"`
}

// SessionDecision is declarative: downstream session machinery performs Action.
// Compat, when present, is the field-by-field compatibility record the Action was
// derived from; it is nil when the caller supplied no descriptor pair and the
// boolean fallback decided.
type SessionDecision struct {
	Action SessionAction `json:"action"`
	ID     string        `json:"id,omitempty"`
	Compat *CompatResult `json:"compat,omitempty"`
	Reason string        `json:"reason"`
}

// Decision is the execution envelope. Keeping the three decisions visible avoids
// flattening distinct choices into an overloaded model id. Roles, when present,
// records the inspectable sub-model delegation plan (scout / worker / judge /
// primary) resolved for this execution; it is nil when no role plan was routed.
type Decision struct {
	Harness HarnessDecision     `json:"harness"`
	Model   modelroute.Decision `json:"model"`
	Session SessionDecision     `json:"session"`
	Roles   *RoleEnvelope       `json:"roles,omitempty"`
}

// Route composes harness selection, the existing model-routing oracle, and session
// lifecycle selection. Candidate order is operator policy and is therefore stable.
func Route(req Request, profiles []harnessprofile.HarnessProfile, manifest modelroute.Manifest) (Decision, error) {
	harness, err := selectHarness(req.HarnessCandidates, req.Harness, req.Health, profiles)
	if err != nil {
		return Decision{}, err
	}
	session, err := routeSession(req.Session)
	if err != nil {
		return Decision{}, err
	}
	return Decision{
		Harness: harness,
		Model:   manifest.Route(req.Model),
		Session: session,
	}, nil
}

// selectHarness walks the candidates in operator order and returns the first that
// satisfies BOTH the static requirements AND the live health gate, recording a
// deterministic rejection reason for every candidate skipped ahead of it. Order
// is preserved: among equally eligible profiles the earliest operator candidate
// wins, exactly as before health was consulted. An empty candidate list falls
// back to the declared profile order.
func selectHarness(candidates []string, requirements HarnessRequirements, health HealthReport, profiles []harnessprofile.HarnessProfile) (HarnessDecision, error) {
	if len(profiles) == 0 {
		return HarnessDecision{}, fmt.Errorf("execution route: no harness profiles are declared")
	}
	type entry struct {
		name    string
		profile harnessprofile.HarnessProfile
		found   bool
	}
	var ordered []entry
	explicitOrder := len(candidates) > 0
	if explicitOrder {
		for _, candidate := range candidates {
			p, ok := findProfile(candidate, profiles)
			ordered = append(ordered, entry{name: candidate, profile: p, found: ok})
		}
	} else {
		for _, p := range profiles {
			ordered = append(ordered, entry{name: p.Name, profile: p, found: true})
		}
	}
	var rejected []HarnessRejection
	for _, e := range ordered {
		if !e.found {
			rejected = append(rejected, HarnessRejection{Candidate: e.name, Reason: "unknown harness profile"})
			continue
		}
		if reason, bad := requirementReason(e.profile, requirements); bad {
			rejected = append(rejected, HarnessRejection{Candidate: e.name, Reason: reason})
			continue
		}
		if reason, bad := health.healthReason(healthKeys(e.name, e.profile)...); bad {
			rejected = append(rejected, HarnessRejection{Candidate: e.name, Reason: reason})
			continue
		}
		reason := "first declared profile satisfying requirements"
		if explicitOrder {
			reason = "first candidate satisfying requirements"
		}
		if health.active() {
			reason += " and live health"
		}
		return HarnessDecision{Profile: e.profile, Reason: reason, Rejected: rejected}, nil
	}
	return HarnessDecision{Rejected: rejected}, fmt.Errorf("execution route: no harness candidate satisfies wire=%q repoint=%q rotatable=%t and live health", requirements.Wire, requirements.Repoint, requirements.Rotatable)
}

func findProfile(candidate string, profiles []harnessprofile.HarnessProfile) (harnessprofile.HarnessProfile, bool) {
	candidate = normalize(candidate)
	for _, p := range profiles {
		if normalize(p.Name) == candidate {
			return p, true
		}
		for _, name := range p.Names {
			if normalize(name) == candidate {
				return p, true
			}
		}
	}
	return harnessprofile.HarnessProfile{}, false
}

func normalize(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.TrimSuffix(s, ".exe")
	return s
}

func routeSession(s SessionSubject) (SessionDecision, error) {
	if s.ID == "" {
		return SessionDecision{Action: SessionStart, Reason: "no prior session was supplied"}, nil
	}
	compactAt := s.CompactAt
	if compactAt <= 0 {
		compactAt = 0.80
	}
	if s.Source != nil && s.Target != nil {
		return routeByDescriptor(s, compactAt)
	}
	if s.PreserveContinuity {
		if s.ContextUtilization >= compactAt {
			return SessionDecision{Action: SessionCompactResume, ID: s.ID, Reason: "continuity required and context reached compaction threshold"}, nil
		}
		return SessionDecision{Action: SessionResume, ID: s.ID, Reason: "continuity required and context remains below compaction threshold"}, nil
	}
	if s.Portable {
		return SessionDecision{Action: SessionFork, ID: s.ID, Reason: "state is portable and continuity is not required"}, nil
	}
	return SessionDecision{Action: SessionStart, Reason: "prior state is neither required nor portable"}, nil
}

// routeByDescriptor derives the lifecycle from computed compatibility
// rather than the caller's booleans. Compaction stays orthogonal: it is a property
// of how full the context is, so it refines an eligible resume into a
// compact_resume without touching the portability verdict.
func routeByDescriptor(s SessionSubject, compactAt float64) (SessionDecision, error) {
	source := *s.Source
	if source.ID == "" {
		source.ID = s.ID
	}
	compat, err := RouteCompat(source, *s.Target)
	if err != nil {
		return SessionDecision{}, err
	}
	dec := SessionDecision{Action: compat.Action, ID: compat.ID, Compat: &compat, Reason: compat.Reason}
	if compat.Refused {
		// A refused move carries no prior state, so it names no session to reopen.
		dec.ID = ""
		return dec, nil
	}
	if compat.Action == SessionResume && s.ContextUtilization >= compactAt {
		dec.Action = SessionCompactResume
		dec.Reason = compat.Reason + "; context reached the compaction threshold"
	}
	return dec, nil
}
