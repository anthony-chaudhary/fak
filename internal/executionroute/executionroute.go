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
type SessionSubject struct {
	ID                 string  `json:"id,omitempty"`
	PreserveContinuity bool    `json:"preserve_continuity,omitempty"`
	Portable           bool    `json:"portable,omitempty"`
	ContextUtilization float64 `json:"context_utilization,omitempty"`
	CompactAt          float64 `json:"compact_at,omitempty"`
}

// Request is the cross-plane subject routed as one unit.
type Request struct {
	HarnessCandidates []string            `json:"harness_candidates,omitempty"`
	Harness           HarnessRequirements `json:"harness_requirements,omitempty"`
	Model             modelroute.Subject  `json:"model"`
	Session           SessionSubject      `json:"session"`
}

// HarnessDecision records both the selected profile and why it won.
type HarnessDecision struct {
	Profile harnessprofile.HarnessProfile `json:"profile"`
	Reason  string                        `json:"reason"`
}

// SessionDecision is declarative: downstream session machinery performs Action.
type SessionDecision struct {
	Action SessionAction `json:"action"`
	ID     string        `json:"id,omitempty"`
	Reason string        `json:"reason"`
}

// Decision is the execution envelope. Keeping the three decisions visible avoids
// flattening distinct choices into an overloaded model id.
type Decision struct {
	Harness HarnessDecision     `json:"harness"`
	Model   modelroute.Decision `json:"model"`
	Session SessionDecision     `json:"session"`
}

// Route composes harness selection, the existing model-routing oracle, and session
// lifecycle selection. Candidate order is operator policy and is therefore stable.
func Route(req Request, profiles []harnessprofile.HarnessProfile, manifest modelroute.Manifest) (Decision, error) {
	harness, err := selectHarness(req.HarnessCandidates, req.Harness, profiles)
	if err != nil {
		return Decision{}, err
	}
	return Decision{
		Harness: harness,
		Model:   manifest.Route(req.Model),
		Session: routeSession(req.Session),
	}, nil
}

func selectHarness(candidates []string, requirements HarnessRequirements, profiles []harnessprofile.HarnessProfile) (HarnessDecision, error) {
	if len(profiles) == 0 {
		return HarnessDecision{}, fmt.Errorf("execution route: no harness profiles are declared")
	}
	ordered := profiles
	if len(candidates) > 0 {
		ordered = make([]harnessprofile.HarnessProfile, 0, len(candidates))
		for _, candidate := range candidates {
			if p, ok := findProfile(candidate, profiles); ok {
				ordered = append(ordered, p)
			}
		}
	}
	for _, p := range ordered {
		if requirements.Wire != "" && p.Wire != requirements.Wire {
			continue
		}
		if requirements.Repoint != "" && !p.HasRepoint(requirements.Repoint) {
			continue
		}
		if requirements.Rotatable && (p.ConfigHomeGlob == "" || p.Identity == harnessprofile.IdentityNone) {
			continue
		}
		reason := "first declared profile satisfying requirements"
		if len(candidates) > 0 {
			reason = "first candidate satisfying requirements"
		}
		return HarnessDecision{Profile: p, Reason: reason}, nil
	}
	return HarnessDecision{}, fmt.Errorf("execution route: no harness candidate satisfies wire=%q repoint=%q rotatable=%t", requirements.Wire, requirements.Repoint, requirements.Rotatable)
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

func routeSession(s SessionSubject) SessionDecision {
	if s.ID == "" {
		return SessionDecision{Action: SessionStart, Reason: "no prior session was supplied"}
	}
	compactAt := s.CompactAt
	if compactAt <= 0 {
		compactAt = 0.80
	}
	if s.PreserveContinuity {
		if s.ContextUtilization >= compactAt {
			return SessionDecision{Action: SessionCompactResume, ID: s.ID, Reason: "continuity required and context reached compaction threshold"}
		}
		return SessionDecision{Action: SessionResume, ID: s.ID, Reason: "continuity required and context remains below compaction threshold"}
	}
	if s.Portable {
		return SessionDecision{Action: SessionFork, ID: s.ID, Reason: "state is portable and continuity is not required"}
	}
	return SessionDecision{Action: SessionStart, Reason: "prior state is neither required nor portable"}
}
