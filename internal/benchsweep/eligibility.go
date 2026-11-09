// eligibility.go implements the pluggable eligibility chain of a sweep: for
// each executed combo's outcome it decides whether the outcome counts toward
// the sweep summary (accept), is refused with a closed reason token (reject),
// or is carried flagged and logged (rescue).
//
// The standard chain is ordered and first-decisive-check-wins:
//
//	artifacts-exist -> hard-exclude statuses -> status whitelist -> duration rescue
//
// Each check answers Pass (inconclusive, keep walking) or a decisive verdict;
// once a check decides, later checks never run, so every verdict names exactly
// one decisive check. The status sets and the duration floor are data
// parameters (Config), never literals, and Append lets a campaign extend the
// chain with its own checks without touching this file.
package benchsweep

import "time"

// Outcome is one executed combo's raw result record: the status the runner
// reported, the artifact paths it wrote, and its wall-clock duration. The
// caller measures the duration; this package only compares it against the
// data-parameter floor, so no clock is read here.
type Outcome struct {
	Combo     Combo         `json:"combo"`
	Status    string        `json:"status"`
	Artifacts []string      `json:"artifacts,omitempty"`
	Duration  time.Duration `json:"duration"`
}

// VerdictKind is the closed vocabulary of chain verdicts.
type VerdictKind string

const (
	// VerdictAccept: no check decided; the outcome counts normally.
	VerdictAccept VerdictKind = "accept"
	// VerdictReject: a decisive check refused the outcome; Verdict.Reason
	// carries the closed token naming why.
	VerdictReject VerdictKind = "reject"
	// VerdictRescue: the outcome passed the gates but its wall-clock duration
	// sat below the floor; it is carried flagged and logged, never silently.
	VerdictRescue VerdictKind = "rescue"
)

// Closed reason tokens of the standard chain. Callers branch on these, never
// on prose; campaign-appended checks may introduce their own tokens.
const (
	// ReasonNoArtifacts: the outcome wrote no artifact evidence.
	ReasonNoArtifacts = "NO_ARTIFACTS"
	// ReasonHardExcludedStatus: the status is on the hard-exclude list.
	ReasonHardExcludedStatus = "HARD_EXCLUDED_STATUS"
	// ReasonStatusNotWhitelisted: the status is not on the whitelist.
	ReasonStatusNotWhitelisted = "STATUS_NOT_WHITELISTED"
)

// CheckAnswer is one check's answer for one outcome. A non-decisive answer
// means "pass, keep walking"; a decisive answer ends the chain immediately.
type CheckAnswer struct {
	// Decisive is false for "pass". When true, Kind is VerdictReject or
	// VerdictRescue and, for a reject, Reason names the closed token.
	Decisive bool
	Kind     VerdictKind
	Reason   string
}

// Pass is the shared inconclusive answer (the zero CheckAnswer).
var Pass = CheckAnswer{}

// RejectAnswer builds a decisive reject carrying a closed reason token.
func RejectAnswer(reason string) CheckAnswer {
	return CheckAnswer{Decisive: true, Kind: VerdictReject, Reason: reason}
}

// RescueAnswer builds a decisive rescue.
func RescueAnswer() CheckAnswer {
	return CheckAnswer{Decisive: true, Kind: VerdictRescue}
}

// Check is one step of the chain: a named inspection function over outcomes.
type Check struct {
	Name    string
	Inspect func(o Outcome) CheckAnswer
}

// Config carries the standard chain's data parameters. None of the three is a
// literal inside a check: a campaign tunes the chain by passing a different
// config. An empty Whitelist is fail-closed: no status is eligible.
type Config struct {
	// HardExcluded statuses are rejected outright, ahead of the whitelist, so
	// an excluded status can never be admitted by also listing it there.
	HardExcluded []string
	// Whitelist lists the statuses an outcome may carry to reach the duration
	// gate. Empty means no status is eligible.
	Whitelist []string
	// MinDuration is the wall-clock floor. An otherwise-eligible outcome that
	// ran strictly shorter is rescued: flagged on its verdict, logged, and
	// counted separately in the ledger.
	MinDuration time.Duration
}

// Logger receives one entry per rescued outcome, in evaluation order. Nil is a
// valid no-op logger: the rescue is still flagged on the verdict and in the
// ledger; the logger is an extra channel, never the only record.
type Logger func(entry RescueLogEntry)

// RescueLogEntry is one logged rescue.
type RescueLogEntry struct {
	ComboID  string        `json:"combo_id"`
	Status   string        `json:"status"`
	Duration time.Duration `json:"duration"`
	Check    string        `json:"check"`
}

// Chain is an ordered sequence of checks with a rescue logger.
type Chain struct {
	checks []Check
	log    Logger
}

// NewChain builds the standard ordered chain:
// artifacts-exist -> hard-exclude statuses -> status whitelist -> duration rescue.
func NewChain(cfg Config, log Logger) *Chain {
	return &Chain{checks: standardChecks(cfg), log: log}
}

// Append adds c after the existing checks and returns the chain, so a campaign
// can extend the standard chain with its own decisive inspections.
func (ch *Chain) Append(c Check) *Chain {
	ch.checks = append(ch.checks, c)
	return ch
}

// Checks returns the chain's checks in evaluation order (a copy).
func (ch *Chain) Checks() []Check {
	out := make([]Check, len(ch.checks))
	copy(out, ch.checks)
	return out
}

// Evaluate walks the checks in order and returns the first decisive verdict;
// when no check decides, the outcome is accepted. Exactly one check is ever
// decisive for a given outcome — first-decisive-check-wins.
func (ch *Chain) Evaluate(o Outcome) Verdict {
	for _, c := range ch.checks {
		a := c.Inspect(o)
		if !a.Decisive {
			continue
		}
		v := Verdict{Kind: a.Kind, Reason: a.Reason, Check: c.Name}
		if a.Kind == VerdictRescue {
			v.Rescued = true
			if ch.log != nil {
				ch.log(RescueLogEntry{
					ComboID:  o.Combo.ID,
					Status:   o.Status,
					Duration: o.Duration,
					Check:    c.Name,
				})
			}
		}
		return v
	}
	return Verdict{Kind: VerdictAccept}
}

// Verdict is the typed chain result for one outcome.
type Verdict struct {
	// Kind is one of VerdictAccept, VerdictReject, VerdictRescue.
	Kind VerdictKind `json:"kind"`
	// Reason is the rejection token; set iff Kind == VerdictReject.
	Reason string `json:"reason,omitempty"`
	// Check names the decisive check; empty for an accept.
	Check string `json:"check,omitempty"`
	// Rescued is true iff Kind == VerdictRescue: the outcome is carried in the
	// ledger flagged, never silently included.
	Rescued bool `json:"rescued,omitempty"`
}

// IsAccepted reports whether the outcome counts normally.
func (v Verdict) IsAccepted() bool { return v.Kind == VerdictAccept }

// IsRejected reports whether a decisive check refused the outcome.
func (v Verdict) IsRejected() bool { return v.Kind == VerdictReject }

// IsRescued reports whether the outcome was carried flagged.
func (v Verdict) IsRescued() bool { return v.Kind == VerdictRescue }

// standardChecks materializes the four named checks from cfg. The status sets
// are precomputed into maps so each check stays O(1) per outcome.
func standardChecks(cfg Config) []Check {
	hard := make(map[string]bool, len(cfg.HardExcluded))
	for _, s := range cfg.HardExcluded {
		hard[s] = true
	}
	white := make(map[string]bool, len(cfg.Whitelist))
	for _, s := range cfg.Whitelist {
		white[s] = true
	}
	return []Check{
		{
			Name: "artifacts-exist",
			Inspect: func(o Outcome) CheckAnswer {
				if len(o.Artifacts) == 0 {
					return RejectAnswer(ReasonNoArtifacts)
				}
				return Pass
			},
		},
		{
			Name: "hard-exclude-statuses",
			Inspect: func(o Outcome) CheckAnswer {
				if hard[o.Status] {
					return RejectAnswer(ReasonHardExcludedStatus)
				}
				return Pass
			},
		},
		{
			Name: "status-whitelist",
			Inspect: func(o Outcome) CheckAnswer {
				if !white[o.Status] {
					return RejectAnswer(ReasonStatusNotWhitelisted)
				}
				return Pass
			},
		},
		{
			Name: "duration-rescue",
			Inspect: func(o Outcome) CheckAnswer {
				if o.Duration < cfg.MinDuration {
					return RescueAnswer()
				}
				return Pass
			},
		},
	}
}
