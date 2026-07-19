package scmbridge

import (
	"github.com/anthony-chaudhary/fak/internal/serviceledger"
	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

// The #4698 destructive matrix, extended by #4756: each probe kills the
// supervised chain a different way and must be INDEPENDENTLY corroborated —
// the stop and the recovery both proven by rows a sensor other than the
// supervisor itself ledgered (the supervisor's self-report alone proves
// nothing about its own death). Judge is the deterministic referee an
// isolated lab run feeds its captured ledger into.
type Probe string

const (
	// ProbeTerminalKill kills the interactive terminal hosting the work.
	ProbeTerminalKill Probe = "terminal-kill"
	// ProbeTermSvcReset resets TermService, tearing down the desktop
	// session; the broker must resume into a NEW session identity.
	ProbeTermSvcReset Probe = "termservice-reset"
	// ProbeHostReboot reboots the host; boot recovery must restart the chain.
	ProbeHostReboot Probe = "host-reboot"
	// ProbeSCMKill kills the SCM-owned service process; SCM recovery
	// actions must replace it.
	ProbeSCMKill Probe = "scm-process-kill"
)

// AllProbes enumerates the matrix (stable order).
var AllProbes = []Probe{ProbeTerminalKill, ProbeTermSvcReset, ProbeHostReboot, ProbeSCMKill}

// Missing-requirement vocabulary for ProbeVerdict.
const (
	MissingStopEvidence    = "independent-stop-evidence"
	MissingResumeEvidence  = "resume-evidence"
	MissingResumedIdentity = "resumed-session-identity"
)

// ProbeVerdict is Judge's answer for one probe over one captured ledger.
type ProbeVerdict struct {
	Probe Probe `json:"probe"`
	// Corroborated: the destructive stop is proven by a non-fak source.
	Corroborated bool `json:"corroborated"`
	// Resumed: recovery evidence follows the corroborated stop.
	Resumed bool     `json:"resumed"`
	Missing []string `json:"missing,omitempty"`
}

// Passed reports the full verdict: independently corroborated AND resumed.
func (v ProbeVerdict) Passed() bool { return v.Corroborated && v.Resumed && len(v.Missing) == 0 }

// independent reports whether the row came from a sensor other than the
// supervisor's own self-reports.
func independent(e serviceledger.Event) bool { return e.Source != serviceledger.SourceFak }

// Judge referees one probe against the captured event ledger:
//
//   - terminal-kill / scm-process-kill: an independent crash-classed
//     process-exit, then a manager-restart or readiness row after it.
//   - host-reboot: an independent boot-change, then restart/readiness after.
//   - termservice-reset: an independent process-exit, then a RESUME row after
//     it that carries the resumed session identity.
//
// Deterministic over the slice; order is ledger append order.
func Judge(p Probe, events []serviceledger.Event) ProbeVerdict {
	v := ProbeVerdict{Probe: p}
	stopAt := -1
	for i, e := range events {
		if !independent(e) {
			continue
		}
		switch p {
		case ProbeHostReboot:
			if e.Type == serviceledger.EventBootChange {
				stopAt = i
			}
		case ProbeTermSvcReset:
			if e.Type == serviceledger.EventProcessExit {
				stopAt = i
			}
		default: // terminal-kill, scm-process-kill
			if e.Type == serviceledger.EventProcessExit && e.Exit != nil && e.Exit.Class == servicespec.ExitCrash {
				stopAt = i
			}
		}
		if stopAt == i {
			break
		}
	}
	if stopAt < 0 {
		v.Missing = append(v.Missing, MissingStopEvidence)
		return v
	}
	v.Corroborated = true
	for _, e := range events[stopAt+1:] {
		switch p {
		case ProbeTermSvcReset:
			if e.Type == serviceledger.EventResume {
				if e.Correlation.Session == "" {
					v.Missing = append(v.Missing, MissingResumedIdentity)
					return v
				}
				v.Resumed = true
				return v
			}
		default:
			if e.Type == serviceledger.EventManagerRestart ||
				e.Type == serviceledger.EventResume ||
				(e.Type == serviceledger.EventReadiness && independent(e)) {
				v.Resumed = true
				return v
			}
		}
	}
	v.Missing = append(v.Missing, MissingResumeEvidence)
	return v
}
