package resume

// Per-source (per-machine) launch admission. resume.Plan answers "what does THIS
// resume cost"; AdmitSource answers the orthogonal concurrency question "may the BOX
// take one more live resume right now". It is a pure decision in the spirit of
// loopmgr.Admit: it takes an already-folded host snapshot plus an operator-tunable
// policy and returns an advisory admit/refuse verdict. It performs no I/O, reads no
// clock (now is supplied), trusts no launcher's self-report — every input is a fold
// the cmd/fak shell computes. The caller (a launch tick, or `fak resume admit`)
// enforces the verdict.
//
// # The gap it closes
//
// The server-side 529 "Server is temporarily limiting requests (not your usage
// limit)" is a per-SOURCE (machine/IP) burst wall, not a per-account usage cap. N
// concurrent `claude --resume` processes on one box — even on different healthy
// accounts — trip it within seconds and every session re-strands (#1341/#1344). Every
// existing cap is per-tick (FAK_MAX_PER_TICK) or per-account (REHOME_CAP), so nothing
// counts the total live resumes on the box across accounts. This is that missing
// count, turned into one admission gate a launcher self-gates on before it spawns.
//
// The two dimensions are independent and neither trusts the other: the LIVE-CONCURRENCY
// ceiling is measured from an OS process census (how many `claude --resume` are alive
// right now, the truth that actually correlates with the burst wall), and the
// LAUNCH-RATE window is measured from the durable launch ledger's timestamps (a faithful
// generalization of the per-account launch_admission gate to the whole source). The
// spacing floor is the across-ticks form of FAK_LAUNCH_SPACING_SEC.

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"
)

// SourcePolicy is the operator-tunable per-source admission policy. The zero value is
// permissive (admit always): every gate is opt-in, so an unconfigured host behaves
// exactly as it did before a policy existed (the same fail-open default loopmgr.Policy
// takes). A launcher that wants the burst-wall protection sets MaxLiveResumes; the rest
// are finer tuning.
type SourcePolicy struct {
	// MaxLiveResumes is the host-wide ceiling on currently-LIVE `claude --resume`
	// processes across ALL accounts: refuse a launch once this many are already alive.
	// 0 disables the gate (no live ceiling). This is the dimension every prior cap
	// lacked — it bounds the standing concurrency the per-source 529 burst wall keys on,
	// not the per-tick or per-account rate. The live count is read off an OS census in
	// the fold; AdmitSource only compares it to the ceiling.
	MaxLiveResumes int `json:"max_live_resumes,omitempty"`

	// MaxLaunchesPerWindow + WindowSeconds bound the launch RATE across the whole source:
	// refuse once this many launches have been recorded in the trailing WindowSeconds.
	// A faithful port of launch_admission's GLOBAL_LAUNCH_CAP, generalized from
	// per-account to per-source. Either being 0 disables the gate (a rate needs both a
	// count and a window to mean anything).
	MaxLaunchesPerWindow int   `json:"max_launches_per_window,omitempty"`
	WindowSeconds        int64 `json:"window_seconds,omitempty"`

	// MinLaunchSpacingSeconds is the host-wide spacing floor: refuse a launch that lands
	// sooner than this many seconds after the most recent recorded launch on the box. 0
	// disables. This is the across-ticks generalization of FAK_LAUNCH_SPACING_SEC (which
	// only spaces launches WITHIN one tick) — it also spaces a launch against the burst a
	// prior tick or a different launcher already fired.
	MinLaunchSpacingSeconds int64 `json:"min_launch_spacing_seconds,omitempty"`
}

// Structured refusal reasons. Closed vocabulary, emittable and verifiable, in the
// spirit of the DOS refusal set: a refusal carries a reason a downstream can route on,
// never free-text drift. LAUNCH_RATE_EXCEEDED deliberately reuses the token
// launch_admission.py already emits, so the gate stays one member of the same closed
// vocabulary across the Go and python implementations.
const (
	ReasonSourceAdmitted  = "SOURCE_ADMITTED"
	ReasonSourceSaturated = "SOURCE_SATURATED"
	ReasonLaunchRate      = "LAUNCH_RATE_EXCEEDED"
	ReasonLaunchSpacing   = "LAUNCH_SPACING_FLOOR"
)

// SourceSnapshot is the folded host state AdmitSource decides on: pure data, no I/O.
// The cmd/fak shell builds it — LiveResumeCount from an OS process census, the launch
// times from the durable launch ledger.
type SourceSnapshot struct {
	// LiveResumeCount is the number of `claude --resume` processes alive on the host
	// right now, across all accounts (the OS census the fold runs).
	LiveResumeCount int `json:"live_resume_count"`

	// LaunchUnixTimes are the unix-second timestamps of recorded launches on the host
	// (from the launch ledger). Order is not assumed; AdmitSource filters to the window.
	LaunchUnixTimes []int64 `json:"launch_unix_times,omitempty"`

	// LastLaunchUnix is the most recent recorded launch time on the host (0 if none),
	// the anchor the spacing floor measures against. It is carried explicitly rather
	// than re-derived from LaunchUnixTimes so a caller can supply it independently.
	LastLaunchUnix int64 `json:"last_launch_unix,omitempty"`
}

// SourceDecision is the advisory verdict AdmitSource returns. Admit is true to proceed.
// RetryAfterUnix, when > 0, is the earliest time the refused launch could be retried
// (the oldest in-window launch + the window for a rate refuse; last-launch + spacing
// for a spacing refuse) — the machine-checkable retry_after launch_admission carries,
// so a launcher can wait the exact amount rather than poll.
type SourceDecision struct {
	Admit          bool   `json:"admit"`
	Reason         string `json:"reason"`
	Summary        string `json:"summary"`
	RetryAfterUnix int64  `json:"retry_after_unix,omitempty"`
	LiveResumes    int    `json:"live_resumes"`
	WindowLaunches int    `json:"window_launches"`
}

// AdmitSource applies a per-source policy to a folded host snapshot at time now. It is
// pure: no I/O, no clock read (now is supplied), no mutation. Gates are checked in a
// fixed order so the reason is deterministic; the first failing gate wins. The order is
// live-ceiling → rate-window → spacing-floor: the live ceiling is the hard structural
// bound on standing concurrency (the thing the burst wall keys on), checked first; the
// rate window bounds a burst of launches; the spacing floor is the finest smoothing,
// checked last.
func AdmitSource(snap SourceSnapshot, policy SourcePolicy, now time.Time) SourceDecision {
	nowUnix := now.UTC().Unix()
	windowLaunches := 0
	var oldestInWindow int64
	if policy.MaxLaunchesPerWindow > 0 && policy.WindowSeconds > 0 {
		cutoff := nowUnix - policy.WindowSeconds
		for _, t := range snap.LaunchUnixTimes {
			if t >= cutoff && t <= nowUnix {
				windowLaunches++
				if oldestInWindow == 0 || t < oldestInWindow {
					oldestInWindow = t
				}
			}
		}
	}

	d := SourceDecision{
		Admit:          true,
		Reason:         ReasonSourceAdmitted,
		LiveResumes:    snap.LiveResumeCount,
		WindowLaunches: windowLaunches,
	}

	// 1. LIVE CEILING: the host-wide standing-concurrency bound. At or over the ceiling,
	//    one more live resume would push the source further past the burst wall, so refuse.
	if policy.MaxLiveResumes > 0 && snap.LiveResumeCount >= policy.MaxLiveResumes {
		d.Admit = false
		d.Reason = ReasonSourceSaturated
		d.Summary = fmt.Sprintf("%d live resumes at/over the %d host ceiling — wait for one to finish",
			snap.LiveResumeCount, policy.MaxLiveResumes)
		return d
	}

	// 2. RATE WINDOW: bound a burst of launches across the whole source. retry_after is
	//    the oldest in-window launch plus the window — when it ages out, a slot frees.
	if policy.MaxLaunchesPerWindow > 0 && policy.WindowSeconds > 0 && windowLaunches >= policy.MaxLaunchesPerWindow {
		d.Admit = false
		d.Reason = ReasonLaunchRate
		d.Summary = fmt.Sprintf("%d launches in the last %ds at/over the %d-per-window cap",
			windowLaunches, policy.WindowSeconds, policy.MaxLaunchesPerWindow)
		if oldestInWindow > 0 {
			d.RetryAfterUnix = oldestInWindow + policy.WindowSeconds
		}
		return d
	}

	// 3. SPACING FLOOR: smooth launches against the box's most recent one, across ticks
	//    and across launchers (not just within a single tick).
	if policy.MinLaunchSpacingSeconds > 0 && snap.LastLaunchUnix > 0 {
		elapsed := nowUnix - snap.LastLaunchUnix
		if elapsed >= 0 && elapsed < policy.MinLaunchSpacingSeconds {
			d.Admit = false
			d.Reason = ReasonLaunchSpacing
			d.Summary = fmt.Sprintf("launched %ds into a %ds host spacing floor",
				elapsed, policy.MinLaunchSpacingSeconds)
			d.RetryAfterUnix = snap.LastLaunchUnix + policy.MinLaunchSpacingSeconds
			return d
		}
	}

	d.Summary = fmt.Sprintf("%d live + %d in-window launches — admitted",
		snap.LiveResumeCount, windowLaunches)
	return d
}

// SourcePolicies is the on-disk tunable policy document: a single default applied to the
// whole host. Unlike loopmgr.Policies it has no per-key overrides — a source governor
// governs ONE box, so one policy is the whole surface an operator edits.
type SourcePolicies struct {
	Schema  string       `json:"schema"`
	Default SourcePolicy `json:"default"`
}

// SchemaSourcePolicy is the policy-document schema tag.
const SchemaSourcePolicy = "fak.resume-source-policy.v1"

// LoadSourcePolicy reads a per-source policy document from path. A missing file is NOT
// an error: it returns an empty (permissive) policy set, so a host with no policy gets
// the same admit-always behavior it had before. This fail-open default is deliberate —
// the governor adds backpressure when asked, never silently throttles a launcher nobody
// configured. A present-but-malformed file IS an error: a typo should be loud, not
// silently permissive. Mirrors loopmgr.LoadPolicies exactly.
func LoadSourcePolicy(path string) (SourcePolicies, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return SourcePolicies{}, nil
	}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return SourcePolicies{}, nil
	}
	if err != nil {
		return SourcePolicies{}, fmt.Errorf("read resume source policy: %w", err)
	}
	var p SourcePolicies
	if err := json.Unmarshal(b, &p); err != nil {
		return SourcePolicies{}, fmt.Errorf("decode resume source policy %s: %w", path, err)
	}
	if p.Schema != "" && p.Schema != SchemaSourcePolicy {
		return SourcePolicies{}, fmt.Errorf("resume source policy schema = %q, want %q", p.Schema, SchemaSourcePolicy)
	}
	return p, nil
}
