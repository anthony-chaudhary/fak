package toolproc

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// A MONITOR is an event-stream watcher run as a tool process: a `tail -f log |
// grep`, an `inotifywait -m`, a poll loop that emits one line per occurrence.
// Its doctrine is "silence is not success": a filter matching only its progress
// pattern stays silent through a crashloop, and to an observer that silence is
// indistinguishable from steady progress. This leaf makes the doctrine
// STRUCTURAL rather than advisory in two places:
//
//   - at ARM time (ArmMonitor): a filter that cannot match at least one
//     failure-signature class alongside its progress pattern is REFUSED with the
//     closed MONITOR_NO_FAILURE_COVERAGE token — you cannot arm a monitor that
//     is silent-on-failure by construction;
//   - at FOLD time (finalizeProc): a monitor whose stream goes quiet past its
//     declared heartbeat cadence folds to TOOL_HEARTBEAT_STALLED with KILL
//     advice (a generic long-runner's stall is only a probe — it may be slow but
//     alive; a monitor that stops emitting is not doing its one job), so the
//     supervisor kills and journals it and the silence becomes a typed verdict.

// ReasonMonitorNoFailureCoverage is the arm-time refusal code. It extends the
// toolproc verdict block (1040–1043; see the reason-code note in toolproc.go)
// with 1044 — still below the egress family's reserved region, registered by the
// same consumer that registers the rest of the toolproc vocabulary.
const ReasonMonitorNoFailureCoverage abi.ReasonCode = 1044

// ReasonMonitorNoFailureCoverageName is its closed token: a monitor filter that
// declares no failure-signature class is refused, not armed.
const ReasonMonitorNoFailureCoverageName = "MONITOR_NO_FAILURE_COVERAGE"

// ErrMonitorNoFailureCoverage is the sentinel ArmMonitor wraps when a filter
// covers no failure-signature class, so a caller can errors.Is it and cite the
// closed token without string-matching the message.
var ErrMonitorNoFailureCoverage = errors.New(ReasonMonitorNoFailureCoverageName)

// monitorFailureClasses is the CLOSED set of terminal-failure signature classes
// a monitor filter must cover at least one of. Each class lists the lowercase
// substrings that WITNESS it in a filter string; a filter "covers" the class
// when it contains any of them (case-insensitively). The set is deliberately
// broad and additive: the check exists to catch the happy-path-only filter, so
// over-acceptance (a filter that names a failure token it never actually emits)
// is the safe direction and a false REFUSAL of a real failure filter is the
// costly one — widen the tokens here rather than narrow them.
var monitorFailureClasses = []struct {
	Name   string
	Tokens []string
}{
	{"crash", []string{"traceback", "panic", "segfault", "core dump", "fatal", "stack trace"}},
	{"error", []string{"error", "exception", "assert"}},
	{"failure", []string{"fail", "non-zero", "exit code", "unsuccessful"}},
	{"killed", []string{"killed", "sigkill", "sigterm", "oom", "out of memory", "terminated"}},
	{"timeout", []string{"timeout", "timed out", "deadline", "hung", "unreachable", "refused"}},
	{"cancel", []string{"cancel", "abort", "denied", "rejected"}},
}

// MonitorFailureClassNames returns the closed class vocabulary, for a caller
// that wants to render the "cover at least one of" set in a refusal message.
func MonitorFailureClassNames() []string {
	out := make([]string, len(monitorFailureClasses))
	for i, c := range monitorFailureClasses {
		out[i] = c.Name
	}
	return out
}

// FilterFailureCoverage reports which failure-signature classes the given
// monitor filter covers. Deterministic and pure: same filter ⇒ same set, in the
// fixed class order. An empty result means the filter is progress-only.
func FilterFailureCoverage(filter string) []string {
	low := strings.ToLower(filter)
	var covered []string
	for _, c := range monitorFailureClasses {
		for _, tok := range c.Tokens {
			if strings.Contains(low, tok) {
				covered = append(covered, c.Name)
				break
			}
		}
	}
	return covered
}

// MonitorSpec is the arm-time request to run an event-stream monitor as a
// supervised liveness process.
type MonitorSpec struct {
	CallID           string // the process identity (required)
	Tool             string // a human label for the monitor command (defaults to "monitor")
	Session          string // owning session/lease id (optional)
	Filter           string // the grep/alternation the monitor's stream is piped through (required)
	HeartbeatEveryMS int64  // the cadence past which silence is a stall (required, > 0)
	DeadlineMS       int64  // optional hard runtime ceiling (0 = unbounded)
	AtMS             int64  // the spawn instant (required, > 0)
}

// ArmMonitor validates a monitor spec against the observed-coverage doctrine and
// returns the spawn Event that arms it as a liveness process. It REFUSES
// (wrapping ErrMonitorNoFailureCoverage / citing MONITOR_NO_FAILURE_COVERAGE) a
// filter that covers no failure-signature class — a progress-only filter is
// silent through a crashloop and must not be armed. A missing identity, filter,
// spawn instant, or a non-positive cadence is a caller error (a monitor with no
// cadence can never stall, which defeats the whole point) reported as a plain
// error, distinct from the doctrine refusal.
func ArmMonitor(spec MonitorSpec) (Event, error) {
	if spec.CallID == "" {
		return Event{}, fmt.Errorf("toolproc: monitor arm requires a call id")
	}
	if strings.TrimSpace(spec.Filter) == "" {
		return Event{}, fmt.Errorf("toolproc: monitor %s arm requires a filter", spec.CallID)
	}
	if spec.HeartbeatEveryMS <= 0 {
		return Event{}, fmt.Errorf("toolproc: monitor %s arm requires a positive heartbeat cadence (a monitor with no cadence can never stall)", spec.CallID)
	}
	if spec.AtMS <= 0 {
		return Event{}, fmt.Errorf("toolproc: monitor %s arm requires a positive spawn instant", spec.CallID)
	}
	if spec.DeadlineMS < 0 {
		return Event{}, fmt.Errorf("toolproc: monitor %s arm has a negative deadline", spec.CallID)
	}
	if covered := FilterFailureCoverage(spec.Filter); len(covered) == 0 {
		return Event{}, fmt.Errorf(
			"toolproc: monitor %s filter covers no failure-signature class (progress-only filters are silent through a crashloop; cover at least one of %s): %w",
			spec.CallID, strings.Join(MonitorFailureClassNames(), "|"), ErrMonitorNoFailureCoverage)
	}
	tool := spec.Tool
	if tool == "" {
		tool = "monitor"
	}
	return Event{
		Kind: EvSpawn, CallID: spec.CallID, Tool: tool, Session: spec.Session,
		AtMS: spec.AtMS, DeadlineMS: spec.DeadlineMS, HeartbeatEveryMS: spec.HeartbeatEveryMS,
		Monitor: true,
	}, nil
}
