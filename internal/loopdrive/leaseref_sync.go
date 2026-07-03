package loopdrive

import (
	"fmt"
	"strings"
)

// LeaseRefSyncSurface names the caller surface whose tick is being wrapped with
// ambient lease-ref convergence. It lets the shell record why a sync ran without
// re-deriving boundary semantics from free text.
type LeaseRefSyncSurface string

const (
	LeaseRefSyncSurfaceGeneric           LeaseRefSyncSurface = "generic"
	LeaseRefSyncSurfaceDispatchPreflight LeaseRefSyncSurface = "dispatch_preflight"
	LeaseRefSyncSurfaceLoopDriveTick     LeaseRefSyncSurface = "loop_drive_tick"
	LeaseRefSyncSurfaceGardenStaleLease  LeaseRefSyncSurface = "garden_stale_lease"
)

// LeaseRefSyncBoundary names the loop-drive lease-ref sync boundary. The shell
// owns the actual `fak leaseref sync` call; this package only models when that
// call belongs in a tick.
type LeaseRefSyncBoundary string

const (
	// LeaseRefSyncBeforeDecide converges peer lease refs before admission or
	// termination policy reads the world.
	LeaseRefSyncBeforeDecide LeaseRefSyncBoundary = "before_decide"
	// LeaseRefSyncAfterWrite publishes local lease-ref writes after acquire,
	// release, or reap changed the namespace.
	LeaseRefSyncAfterWrite LeaseRefSyncBoundary = "after_write"
)

// LeaseRefSyncDirection is the safe direction for a boundary. Fetch-before-decide
// sees peers; push-after-write publishes local state without force-fetching over
// a just-written local lease.
type LeaseRefSyncDirection string

const (
	LeaseRefSyncFetchOnly LeaseRefSyncDirection = "fetch_only"
	LeaseRefSyncPushOnly  LeaseRefSyncDirection = "push_only"
)

// LeaseRefSyncStep is one ambient convergence operation a loop-drive tick should
// run around the lease/admission boundary.
type LeaseRefSyncStep struct {
	Surface   LeaseRefSyncSurface
	Boundary  LeaseRefSyncBoundary
	Direction LeaseRefSyncDirection
	Remote    string
	Required  bool
	Summary   string
}

// LeaseRefSyncPlanInput is the pure state needed to decide which convergence
// calls belong around a tick. LeaseRefsWritten means the tick acquired,
// released, or reaped at least one refs/fak/locks/* record.
type LeaseRefSyncPlanInput struct {
	Surface          LeaseRefSyncSurface
	Remote           string
	LeaseRefsWritten bool
}

// LeaseRefSyncPlan returns the ambient sync plan for a loop-drive tick:
// converge-before-decide every time, publish-after-write only when the tick
// changed lease refs. Steps are advisory: transport failures are surfaced by
// LeaseRefSyncReport but do not become fatal loop policy.
func LeaseRefSyncPlan(in LeaseRefSyncPlanInput) []LeaseRefSyncStep {
	surface := in.Surface
	if surface == "" {
		surface = LeaseRefSyncSurfaceGeneric
	}
	remote := strings.TrimSpace(in.Remote)
	if remote == "" {
		remote = "origin"
	}
	steps := []LeaseRefSyncStep{{
		Surface:   surface,
		Boundary:  LeaseRefSyncBeforeDecide,
		Direction: LeaseRefSyncFetchOnly,
		Remote:    remote,
		Required:  false,
		Summary:   "converge lease refs before deciding admission or completion",
	}}
	if in.LeaseRefsWritten {
		steps = append(steps, LeaseRefSyncStep{
			Surface:   surface,
			Boundary:  LeaseRefSyncAfterWrite,
			Direction: LeaseRefSyncPushOnly,
			Remote:    remote,
			Required:  false,
			Summary:   "publish lease-ref writes after acquire, release, or reap",
		})
	}
	return steps
}

// LeaseRefSyncPlanForSurface is the explicit profile constructor for the three
// #2302 wiring surfaces: dispatch preflight, loop-drive ticks, and garden's
// stale-lease rung. Each uses the same safe boundary rule: fetch before deciding;
// push after the tick writes lease refs.
func LeaseRefSyncPlanForSurface(surface LeaseRefSyncSurface, in LeaseRefSyncPlanInput) []LeaseRefSyncStep {
	in.Surface = surface
	return LeaseRefSyncPlan(in)
}

// LeaseRefSyncCommandArgs renders the argv after the `fak` binary for one sync
// step. The shell may call runLeaserefSync directly or exec these args; either
// way it does not need to reinterpret boundary directions.
func LeaseRefSyncCommandArgs(step LeaseRefSyncStep, dir string) ([]string, error) {
	remote := strings.TrimSpace(step.Remote)
	if remote == "" {
		remote = "origin"
	}
	args := []string{"leaseref", "sync"}
	if strings.TrimSpace(dir) != "" {
		args = append(args, "--dir", strings.TrimSpace(dir))
	}
	args = append(args, "--remote", remote)
	switch step.Direction {
	case LeaseRefSyncFetchOnly:
		args = append(args, "--fetch-only")
	case LeaseRefSyncPushOnly:
		args = append(args, "--push-only")
	default:
		return nil, fmt.Errorf("lease-ref sync step has unknown direction %q", step.Direction)
	}
	return args, nil
}

// LeaseRefSyncDirections returns the internal/leaseref Sync booleans for a step.
func LeaseRefSyncDirections(step LeaseRefSyncStep) (doPush, doFetch bool, err error) {
	switch step.Direction {
	case LeaseRefSyncFetchOnly:
		return false, true, nil
	case LeaseRefSyncPushOnly:
		return true, false, nil
	default:
		return false, false, fmt.Errorf("lease-ref sync step has unknown direction %q", step.Direction)
	}
}

// LeaseRefSyncAttempt records the observed result of one planned sync step. Err
// is a short error string from the shell/transport layer; empty means success.
type LeaseRefSyncAttempt struct {
	Step LeaseRefSyncStep
	Err  string
}

// LeaseRefSyncOutcome is the aggregate health of the advisory sync boundary.
type LeaseRefSyncOutcome string

const (
	LeaseRefSyncOK       LeaseRefSyncOutcome = "ok"
	LeaseRefSyncDegraded LeaseRefSyncOutcome = "degraded"
)

const ReasonLeaseRefSyncTransport = "LEASEREF_SYNC_TRANSPORT"

// LeaseRefSyncReport is the nonfatal sync-boundary fold. Fatal is intentionally
// false for transport failures: a node that cannot reach origin should continue
// using local lease evidence while surfacing the convergence failure.
type LeaseRefSyncReport struct {
	Outcome  LeaseRefSyncOutcome
	Reason   string
	Fatal    bool
	Summary  string
	Failures []string
}

// ReportLeaseRefSync folds sync attempts into the loop-drive boundary verdict.
// It never turns transport failure into a fatal loop decision; callers can log or
// expose Failures while continuing local admission.
func ReportLeaseRefSync(attempts []LeaseRefSyncAttempt) LeaseRefSyncReport {
	var failures []string
	for _, a := range attempts {
		if strings.TrimSpace(a.Err) == "" {
			continue
		}
		failures = append(failures, fmt.Sprintf("%s %s %s: %s",
			a.Step.Boundary, a.Step.Direction, firstNonEmpty(a.Step.Remote, "origin"), strings.TrimSpace(a.Err)))
	}
	if len(failures) == 0 {
		return LeaseRefSyncReport{Outcome: LeaseRefSyncOK, Summary: "lease-ref sync boundary clean"}
	}
	return LeaseRefSyncReport{
		Outcome:  LeaseRefSyncDegraded,
		Reason:   ReasonLeaseRefSyncTransport,
		Fatal:    false,
		Summary:  "lease-ref sync failed; continuing with local lease evidence",
		Failures: failures,
	}
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
