package main

// Two-phase fit commit (ticket T03 "fit confirm"): the Fit -> Alloc -> Commit
// back-off that turns an allocation failure at load time into a re-plan against
// a shrunken believed-free budget instead of a dead process.
//
// The planner sizes against a BELIEVED device free reading; a backend that
// over-reports free (a split aperture, a stale probe, a peer tenant) passes the
// pre-load fit and then fails the real allocation. Rather than exit, the
// driver cuts the believed free by 1% of the original reading on the first
// retry and doubles the cut each subsequent retry (1%, 2%, 4%, ...), re-runs
// the load with the shrunken layout, and stops when either:
//   - the load succeeds (commit),
//   - the same shrunken layout repeats (the planner's granularity means no
//     further shrink changes the plan; continuing would only burn retries),
//   - the retry cap (5) is reached, or
//   - the failure is not an allocation failure (shrinking cannot help).
//
// A user-pinned context token count is NEVER shrunk: the pinned axis stays
// byte-identical across attempts, so a pinned layout that fails its first
// allocation attempt repeats immediately and the driver returns the typed
// refusal naming that last layout. Only auto-derived values (auto-sized
// context tokens, believed-free budgets) may move.

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// serveFitCommitMaxRetries caps the back-off: attempt 0 is the original
// layout, attempts 1..5 carry the doubled cuts. Six loads total, bounded.
const serveFitCommitMaxRetries = 5

// ErrServeFitCommit is the typed refusal returned when the fit-commit back-off
// stops without a committed load. It is reachable via errors.Is and wraps the
// last attempt's cause via Unwrap, so callers can test both the class
// (errors.Is(err, ErrServeFitCommit)) and the concrete failure
// (errors.As(err, &*compute.FitError{})).
var ErrServeFitCommit = errors.New("fak serve: fit-commit back-off stopped without a committed load")

// serveFitCommitError names the LAST layout attempted, per the ticket's
// operator-visible contract: a refusal that does not say which layout was last
// tried cannot be diagnosed without re-running the boot.
type serveFitCommitError struct {
	LastLayout string
	Attempts   int
	Cause      error
}

func (e *serveFitCommitError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("%v after %d attempt(s); last layout: %s; cause: %v", ErrServeFitCommit, e.Attempts, e.LastLayout, e.Cause)
	}
	return fmt.Sprintf("%v after %d attempt(s); last layout: %s", ErrServeFitCommit, e.Attempts, e.LastLayout)
}

// Is makes errors.Is(err, ErrServeFitCommit) true for every back-off refusal.
func (e *serveFitCommitError) Is(target error) bool { return target == ErrServeFitCommit }

// Unwrap exposes the last attempt's cause for errors.As classification.
func (e *serveFitCommitError) Unwrap() error { return e.Cause }

// serveFitCommitLayout is one attempt's shrunken planning inputs. Everything
// in it is derived deterministically from the ORIGINAL readings plus the
// attempt index, so a repeated layout is detectable from the plan it produced.
type serveFitCommitLayout struct {
	ContextTokens int
	SizingFit     serveFitBudget
	DeviceFit     *serveFitBudget
}

// serveFitCommitResult carries one attempt's outcome. Plan is the memory plan
// the attempt planned against (empty on arms that plan host-side only); it is
// the fingerprint's primary content because two attempts with different
// believed-free readings that PLAN IDENTICALLY are the repeated-layout stop.
type serveFitCommitResult struct {
	Model   *fakmodel.Model
	Q4K     bool
	Profile *gateway.ModelLoadProfile
	Phase   gateway.StartupPhase
	Plan    compute.MemoryPlan
}

// serveFitCommitConfig pins what the driver owns: the original readings, the
// pin state, and the arm/ring provenance folded into the fingerprint.
type serveFitCommitConfig struct {
	// PinnedContext is true when the operator explicitly set
	// --native-context-tokens: the token axis must never move.
	PinnedContext bool
	// OriginalTokens is the resolved context token count of attempt 0.
	OriginalTokens int
	// ContextCSC and Weights let the driver re-run the SAME auto-sizer the
	// boot used (compute.AutoSizeContextPlan) against the shrunken budget.
	ContextCSC compute.ContextSizeConfig
	Weights    compute.MemoryPlan
	// SizingFit is the budget the context auto-sizer ran against at boot
	// (possibly the shared-pool form, see resolveServeNativeContextBudget).
	SizingFit serveFitBudget
	// DeviceFit is the load-time admission budget (nil on arms that admit
	// against host RAM only).
	DeviceFit *serveFitBudget
	// Arm labels the load arm in the fingerprint ("device-resident-q4k",
	// "cpu-offload-experts", ...) so an arm switch is never mistaken for a
	// repeat.
	Arm string
	// Log receives the per-attempt diagnostics; nil silences them.
	Log func(format string, args ...any)
}

// layoutFor derives attempt k's shrunken inputs. Attempt 0 is the original
// layout, byte-identical to today's boot. For k >= 1 the believed free is cut
// by 1% of the ORIGINAL reading, doubled each retry (1%, 2%, 4%, ...), clamped
// to the whole budget. The auto-sized context follows the shrunken budget
// unless the context axis is pinned.
func (c serveFitCommitConfig) layoutFor(attempt int) serveFitCommitLayout {
	l := serveFitCommitLayout{ContextTokens: c.OriginalTokens, SizingFit: c.SizingFit, DeviceFit: c.DeviceFit}
	if attempt == 0 || c.SizingFit.Base <= 0 {
		return l
	}
	base := c.SizingFit.Base
	cut := base / 100
	if cut < 1 {
		cut = 1
	}
	for i := 1; i < attempt; i++ {
		cut *= 2
	}
	if cut > base {
		cut = base
	}
	shrunk := c.SizingFit
	shrunk.Base = base - cut
	l.SizingFit = shrunk
	if !c.PinnedContext {
		if tokens, _ := compute.AutoSizeContextPlan(c.ContextCSC, c.Weights, shrunk.avail(), -1); tokens > 0 {
			l.ContextTokens = tokens
		}
	}
	if c.DeviceFit != nil && c.DeviceFit.Base > 0 {
		dcut := cut
		if dcut > c.DeviceFit.Base {
			dcut = c.DeviceFit.Base
		}
		df := *c.DeviceFit
		df.Base = c.DeviceFit.Base - dcut
		l.DeviceFit = &df
	}
	return l
}

// runServeFitCommit drives the Fit -> Alloc -> Commit loop. step runs one load
// attempt against the given layout and returns its outcome; on success the
// result is returned as-is (byte-identical to a single-shot load when attempt
// 0 succeeds), and on exhaustion/repeat a *serveFitCommitError wrapping the
// last cause is returned with the last layout named.
func runServeFitCommit(cfg serveFitCommitConfig, step func(l serveFitCommitLayout) (serveFitCommitResult, error)) (serveFitCommitResult, error) {
	seen := make(map[string]bool)
	var res serveFitCommitResult
	var err error
	var fp string
	for attempt := 0; attempt <= serveFitCommitMaxRetries; attempt++ {
		layout := cfg.layoutFor(attempt)
		res, err = step(layout)
		fp = serveFitCommitFingerprint(res.Plan, layout, cfg.Arm)
		if err == nil {
			return res, nil
		}
		if seen[fp] {
			if cfg.Log != nil {
				cfg.Log("fak serve: fit-commit back-off: repeated layout %s; stopping", fp)
			}
			return res, &serveFitCommitError{LastLayout: fp, Attempts: attempt + 1, Cause: err}
		}
		seen[fp] = true
		if !serveLoadFailureIsAllocation(err) {
			// Not an allocation failure: shrinking the believed free cannot
			// cure it, so surface it unchanged rather than burn retries.
			return res, err
		}
		if cfg.Log != nil {
			next := cfg.layoutFor(attempt + 1)
			cutPct := 0.0
			if cfg.SizingFit.Base > 0 {
				cutPct = float64(cfg.SizingFit.Base-next.SizingFit.Base) / float64(cfg.SizingFit.Base) * 100
			}
			cfg.Log("fak serve: fit-commit back-off: attempt %d failed (%v); next attempt cuts believed free to %.1f%% of the original reading", attempt+1, err, 100-cutPct)
		}
	}
	return res, &serveFitCommitError{LastLayout: fp, Attempts: serveFitCommitMaxRetries + 1, Cause: err}
}

// serveFitCommitFingerprint canonicalizes the layout an attempt actually
// planned: sorted plan rows (scope/class/bytes/detail) plus the token count
// and arm label. Believed-free readings that produce the SAME plan map to the
// SAME fingerprint, which is exactly the repeated-layout stop condition.
func serveFitCommitFingerprint(plan compute.MemoryPlan, l serveFitCommitLayout, arm string) string {
	rows := make([]compute.MemoryDemand, len(plan))
	copy(rows, plan)
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Scope != rows[j].Scope {
			return rows[i].Scope < rows[j].Scope
		}
		if rows[i].Class != rows[j].Class {
			return rows[i].Class < rows[j].Class
		}
		return rows[i].Detail < rows[j].Detail
	})
	var b strings.Builder
	fmt.Fprintf(&b, "ctx=%d;arm=%s;rows=", l.ContextTokens, arm)
	for _, d := range rows {
		fmt.Fprintf(&b, "%s/%s:%d:%s;", d.Scope, d.Class, d.Bytes, d.Detail)
	}
	return b.String()
}

// serveLoadFailureIsAllocation reports whether a load failure is the kind a
// shrunken believed-free budget can cure: a typed pre-load fit refusal
// (FitTooBig), a recovered device allocation panic, or the in-kernel OOM
// sentinel. Anything else (corrupt GGUF, tokenizer failure, ...) is surfaced
// unchanged by the driver.
func serveLoadFailureIsAllocation(err error) bool {
	if err == nil {
		return false
	}
	var oom *agent.InKernelOOMError
	if errors.As(err, &oom) {
		return true
	}
	var dae *compute.DeviceAllocError
	if errors.As(err, &dae) {
		return true
	}
	var fe *compute.FitError
	if errors.As(err, &fe) && fe.Verdict == compute.FitTooBig {
		return true
	}
	return false
}

// logServeFitCommitAttempt is the driver's default diagnostic sink: one line
// per failed attempt on stderr, matching the launch terminal's existing
// "fak serve:" prefix convention.
func logServeFitCommitAttempt(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
}

// loadServeInKernelModelFitted drives the two-phase fit commit around the eager
// GGUF load. Attempt 0 is today's byte-identical single-shot load; an allocation
// failure re-plans against a shrunken believed-free budget (1% of the original
// reading, doubling per retry, capped at 5 retries) until the load commits, a
// layout repeats, or the cap is reached. CPU-only serves keep the historical
// single-shot fatal path: their allocation failures are kernel OOM kills, not
// recoverable errors.
func (rt *serveRuntime) loadServeInKernelModelFitted(sf *serveFlags, expertShard *ggufload.ExpertShard, expertRanks int) (*fakmodel.Model, bool, *gateway.ModelLoadProfile, gateway.StartupPhase) {
	if rt.chatBackend == nil || strings.TrimSpace(*sf.ggufPath) == "" {
		return loadServeInKernelModelPlaced(*sf.ggufPath, rt.chatBackend, *sf.cpuOffloadExperts, rt.requireDeviceExecution, rt.nativeContext.ResolvedTokens, expertShard, expertRanks, rt.fitBudget)
	}
	cfg := serveFitCommitConfig{
		PinnedContext:  rt.contextPinned,
		OriginalTokens: rt.nativeContext.ResolvedTokens,
		ContextCSC:     rt.contextCSC,
		Weights:        rt.contextWeights,
		SizingFit:      rt.sizingFit,
		DeviceFit:      rt.fitBudget,
		Arm:            "device",
		Log:            logServeFitCommitAttempt,
	}
	res, err := runServeFitCommit(cfg, func(l serveFitCommitLayout) (serveFitCommitResult, error) {
		m, q4k, prof, phase, plan, err := loadServeInKernelModelPlacedErr(*sf.ggufPath, rt.chatBackend, *sf.cpuOffloadExperts, rt.requireDeviceExecution, l.ContextTokens, expertShard, expertRanks, l.DeviceFit)
		return serveFitCommitResult{Model: m, Q4K: q4k, Profile: prof, Phase: phase, Plan: plan}, err
	})
	if err != nil {
		must(err)
	}
	rt.lastLoadProfile = res.Profile
	return res.Model, res.Q4K, res.Profile, res.Phase
}
