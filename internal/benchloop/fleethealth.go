package benchloop

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Fleet health accounting for the recurring bench-fleet loop (#6503).
//
// The 15-minute loop reported a healthy result whenever the current tick happened
// to claim nothing, so a durable queue holding only failed rows and unconfigured
// nodes -- eight requests, zero benchmark numbers -- was indistinguishable from a
// fleet that had just measured everything. Every rule below reads the DURABLE
// queue rather than the tick, so the result the loop reports is a function of the
// witnesses on disk: zero successful measurements can never report result 0.
//
// cmd/fak owns the on-disk request/witness JSON and the process/session read-back;
// this file owns the decisions taken from them so they can be witnessed by a test.

// FleetUtilitySchema versions the utility block emitted with each dispatch report.
const FleetUtilitySchema = "fak.bench-fleet.utility.v1"

// Durable queue states a bench-fleet request can carry. Unavailable nodes are
// typed with a waiting_ prefix that names the missing configuration (session,
// credentials, provisioning, route); FleetHeld normalizes all of them.
const (
	FleetQueued    = "queued"
	FleetRunning   = "running"
	FleetSucceeded = "succeeded"
	FleetFailed    = "failed"
	FleetHeld      = "held"
)

// Result codes the recurring loop reports to its scheduler. 2 stays reserved for
// the usage error every fak verb reports, so a fleet that produced no measurement
// at all reports 3 rather than reusing it.
const (
	FleetResultHealthy       = 0
	FleetResultFailed        = 1
	FleetResultNoMeasurement = 3
)

const (
	// FleetHeldRetry bounds how often the loop spends a dispatch re-probing a node
	// whose session or credential is simply not configured. Re-probing every tick is
	// what turned two unconfigured nodes into ninety-six failed dispatches a day.
	FleetHeldRetry = 6 * time.Hour
	// FleetFailureRetry is the first backoff after a genuine execution failure; each
	// consecutive failure doubles it up to FleetFailureRetryMax so a permanently
	// broken cell decays instead of re-running every fifteen minutes.
	FleetFailureRetry = time.Hour
	// FleetFailureRetryMax caps that doubling.
	FleetFailureRetryMax = 8 * time.Hour
	// FleetRunningLease bounds how long a claimed cell may sit in "running" before an
	// independent read-back requeues it. The dispatcher's own subprocess timeout is
	// ten minutes, so anything past this lease is a claim whose owner died.
	FleetRunningLease = 30 * time.Minute
)

// FleetCell is one durable queue row projected into the fields the health rules
// read. Seconds is the wall-clock compute the fleet has already spent on the cell
// and Measured records whether its witness carried a numeric benchmark marker.
type FleetCell struct {
	Machine     string
	Benchmark   string
	State       string
	HeldReason  string
	HeldSince   time.Time
	LastAttempt time.Time
	Attempts    int
	Failures    int
	Seconds     float64
	Measured    bool
}

// FleetUtility is the per-tick utility report: what the loop attempted, what it
// actually measured, which cells are held on a configuration gap, and what the
// churn cost. Result is the exit code the loop reports to its scheduler.
type FleetUtility struct {
	Schema           string         `json:"schema"`
	Cells            int            `json:"cells"`
	Attempted        int            `json:"attempted_cells"`
	Successful       int            `json:"successful_measurements"`
	Measured         int            `json:"numeric_measurements"`
	Failed           int            `json:"failed_cells"`
	Held             int            `json:"held_configuration_gaps"`
	Running          int            `json:"running_cells"`
	Queued           int            `json:"queued_cells"`
	RepeatedFailures int            `json:"repeated_failures"`
	ComputeSeconds   float64        `json:"compute_seconds"`
	HeldReasons      map[string]int `json:"held_reasons,omitempty"`
	Healthy          bool           `json:"healthy"`
	Result           int            `json:"result"`
	Reason           string         `json:"reason"`
}

// NormalizeFleetState folds every unavailable-node spelling onto FleetHeld and
// returns the configuration gap it names. A row typed waiting_session is held on
// "session"; the terminal states pass through unchanged.
func NormalizeFleetState(state string) (string, string) {
	if gap, ok := strings.CutPrefix(state, "waiting_"); ok {
		if gap == "" {
			gap = "unknown"
		}
		return FleetHeld, gap
	}
	if state == FleetHeld {
		return FleetHeld, ""
	}
	return state, ""
}

// SummarizeFleet reads the whole durable queue and reports what the loop is
// actually worth. The result is healthy only when at least one cell holds a
// successful measurement and nothing failed; an empty queue is idle, not healthy.
func SummarizeFleet(cells []FleetCell) FleetUtility {
	u := FleetUtility{Schema: FleetUtilitySchema, Cells: len(cells)}
	for _, cell := range cells {
		state, gap := NormalizeFleetState(cell.State)
		u.ComputeSeconds += cell.Seconds
		if cell.Attempts > 0 {
			u.Attempted++
		}
		if cell.Failures > 1 {
			u.RepeatedFailures++
		}
		switch state {
		case FleetSucceeded:
			u.Successful++
			if cell.Measured {
				u.Measured++
			}
		case FleetFailed:
			u.Failed++
		case FleetRunning:
			u.Running++
		case FleetQueued:
			u.Queued++
		case FleetHeld:
			u.Held++
			if gap == "" {
				gap = cell.HeldReason
			}
			if gap == "" {
				gap = "unknown"
			}
			if u.HeldReasons == nil {
				u.HeldReasons = map[string]int{}
			}
			u.HeldReasons[gap]++
		}
	}
	switch {
	case u.Failed > 0:
		u.Result = FleetResultFailed
		u.Reason = fmt.Sprintf("%d of %d bench-fleet cell(s) failed", u.Failed, u.Cells)
	case u.Cells == 0:
		u.Result = FleetResultHealthy
		u.Reason = "idle: no bench-fleet cells are queued"
	case u.Successful == 0:
		u.Result = FleetResultNoMeasurement
		u.Reason = fmt.Sprintf("no successful measurement in %d durable cell(s): %d held, %d running, %d queued", u.Cells, u.Held, u.Running, u.Queued)
	default:
		u.Result = FleetResultHealthy
		u.Reason = fmt.Sprintf("%d successful measurement(s) of %d cell(s), %d held", u.Successful, u.Cells, u.Held)
	}
	u.Healthy = u.Result == FleetResultHealthy && u.Successful > 0
	return u
}

// ShouldDispatchFleetCell reports whether this tick should spend a claim on the
// row, and when it should not, the reason an operator reads off the report. Held
// cells and failed cells are re-probed on a backoff instead of every tick; rows
// another dispatcher is running, and rows already measured, are left alone.
func ShouldDispatchFleetCell(cell FleetCell, now time.Time) (bool, string) {
	state, gap := NormalizeFleetState(cell.State)
	switch state {
	case FleetQueued:
		return true, ""
	case FleetHeld:
		if gap == "" {
			gap = cell.HeldReason
		}
		if cell.HeldSince.IsZero() || !now.Before(cell.HeldSince.Add(FleetHeldRetry)) {
			return true, ""
		}
		return false, fmt.Sprintf("held on %s configuration; re-probe after %s", gap, cell.HeldSince.Add(FleetHeldRetry).UTC().Format(time.RFC3339))
	case FleetFailed:
		backoff := FleetFailureBackoff(cell.Failures)
		if cell.LastAttempt.IsZero() || !now.Before(cell.LastAttempt.Add(backoff)) {
			return true, ""
		}
		return false, fmt.Sprintf("failed %d time(s); retry after %s", cell.Failures, cell.LastAttempt.Add(backoff).UTC().Format(time.RFC3339))
	case FleetRunning:
		return false, "claimed by a running dispatcher"
	default:
		return false, "already measured"
	}
}

// FleetFailureBackoff doubles the retry delay per consecutive failure, capped at
// FleetFailureRetryMax. Zero or one failure carries the base delay.
func FleetFailureBackoff(failures int) time.Duration {
	backoff := FleetFailureRetry
	for i := 1; i < failures && backoff < FleetFailureRetryMax; i++ {
		backoff *= 2
	}
	if backoff > FleetFailureRetryMax {
		return FleetFailureRetryMax
	}
	return backoff
}

// FleetClaim is the independent read-back of a running cell's claim lock: the
// lock file itself, the dispatcher that took it, and -- only when that dispatcher
// ran on this host -- whether its process is still alive. Local is false for a
// claim taken elsewhere, where Alive carries no information and only the lease
// can decide.
type FleetClaim struct {
	Present bool
	PID     int
	Host    string
	Local   bool
	Alive   bool
	Started time.Time
	Lease   time.Duration
}

// ReconcileFleetRunning decides what a row stuck in "running" should carry, from
// the claim read-back rather than from the row's own say-so. A claim whose owner
// is gone, or whose lock vanished without a terminal state, returns to the queue
// so the cell can be measured instead of being stranded forever.
func ReconcileFleetRunning(claim FleetClaim, now time.Time) (string, string) {
	lease := claim.Lease
	if lease <= 0 {
		lease = FleetRunningLease
	}
	switch {
	case !claim.Present:
		return FleetQueued, "orphaned claim: lock released without a terminal state"
	case claim.Local && claim.PID > 0 && !claim.Alive:
		return FleetQueued, fmt.Sprintf("stale claim: dispatcher pid %d is gone", claim.PID)
	case !claim.Started.IsZero() && !now.Before(claim.Started.Add(lease)):
		return FleetQueued, fmt.Sprintf("stale claim: running past the %s lease", lease)
	default:
		return FleetRunning, ""
	}
}

// FleetMeasurement returns the first numeric benchmark marker a node emitted. The
// fleet's remote recipes announce their identity with FAK_BENCH_NODE and their
// numbers with FAK_BENCH_<KEY>=<number> (HTTP status, seconds, token counts), so
// a transcript carrying only the identity marker proves the node ran and proves
// nothing was measured -- which is exactly the state this loop sat in.
func FleetMeasurement(output string) (string, float64, bool) {
	for _, field := range strings.Fields(output) {
		key, value, ok := strings.Cut(strings.TrimSpace(field), "=")
		if !ok || key == "FAK_BENCH_NODE" || !strings.HasPrefix(key, "FAK_BENCH_") {
			continue
		}
		number, err := strconv.ParseFloat(strings.TrimRight(value, ".,;"), 64)
		if err != nil {
			continue
		}
		return key, number, true
	}
	return "", 0, false
}

// FleetReenableAllowed reports whether the recurring schedule may be re-armed.
// The loop is only worth repeating once one real node has produced a witnessed
// numeric benchmark; until then re-arming just re-spends compute on held and
// failing cells.
func FleetReenableAllowed(u FleetUtility) (bool, string) {
	if u.Measured > 0 {
		return true, fmt.Sprintf("%d witnessed numeric measurement(s)", u.Measured)
	}
	return false, fmt.Sprintf("no witnessed numeric measurement in %d durable cell(s); run one node explicitly before re-arming the schedule", u.Cells)
}
