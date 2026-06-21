package model

// budget.go — modular, machine-portable benchmark limits.
//
// FAK_WORKERS pins matmul parallelism to an ABSOLUTE core count (e.g. 8). That is
// not portable across a fleet of differently-sized machines: "8" is 25% of a 32-core
// box but 100% of an 8-core one, so an operator who wants "leave headroom because
// this box is also running agentic work" has to know each machine's core count and
// do the arithmetic by hand.
//
// FAK_BUDGET expresses the limit as a FRACTION of the machine instead:
//
//	FAK_BUDGET=0.75   -> use up to 75% of the logical cores (24 of 32, 6 of 8, ...)
//	FAK_BUDGET=75     -> same, percent form
//	FAK_BUDGET=75%    -> same, percent form with the sign
//
// The fraction is resolved against GOMAXPROCS once at package init (see parallel.go),
// so it is STATIC and reproducible — a recorded run states exactly the parallelism it
// was taken at. Live-load sensing is deliberately out of scope: a fraction of total
// cores is deterministic; reading OS load would make a run's worker count depend on
// whatever else happened to be busy at the moment, which is not something a benchmark
// number should silently absorb.
//
// Precedence (first match wins), so every existing invocation is byte-for-byte
// unchanged and the fraction is a strictly new path:
//
//	1. FAK_WORKERS=<n>   (n>=1) — explicit absolute override
//	2. FAK_BUDGET=<f>    — fraction in (0,1], or percent (>1)
//	3. default           — GOMAXPROCS(0) (all cores)

import (
	"fmt"
	"math"
	"runtime"
	"strconv"
	"strings"
)

// SetWorkerBudget re-resolves the matmul worker count from a fractional budget AFTER
// package init — the path a bench's -budget flag uses. The env-driven numWorkers var is
// resolved at package load (before main() parses flags), so a flag cannot reach it by
// setting an env var; it calls this instead. frac is a fraction in (0,1] (0.75) or a
// percent (75, "75%"); the resulting count is recorded by WorkerBudget() with a
// "-budget" source so the JSON report distinguishes a flag-set budget from an env one.
// Returns an error (leaving the count untouched) if frac doesn't resolve to (0,1].
//
// Must be called before the first matmul (the worker pool is built lazily on first
// parFor). A bench calls it in main() right after flag.Parse(), which is well before any
// Session work, so the new count is the one every forward pass sees.
func SetWorkerBudget(frac float64) error {
	w, ok := budgetToWorkers(frac, runtime.GOMAXPROCS(0))
	if !ok {
		return fmt.Errorf("budget %v is not a fraction in (0,1] or a percent in (0,100]", frac)
	}
	numWorkers = w
	workerBudgetSource = fmt.Sprintf("-budget=%g", frac)
	return nil
}

// SetWorkers pins an ABSOLUTE matmul worker count after init (n>=1), the post-init
// analogue of FAK_WORKERS for a bench's absolute -jobs flag. Like SetWorkerBudget it
// must run before the first matmul. Returns an error (count untouched) if n < 1.
func SetWorkers(n int) error {
	if n < 1 {
		return fmt.Errorf("workers %d must be >= 1", n)
	}
	numWorkers = n
	workerBudgetSource = fmt.Sprintf("-jobs=%d", n)
	return nil
}

// budgetToWorkers maps a raw fraction-or-percent number + machine width to a worker
// count in [1,cores], shared by the env path (parseBudgetFraction) and the flag path
// (SetWorkerBudget) so both round and floor identically.
func budgetToWorkers(raw float64, cores int) (int, bool) {
	if cores < 1 {
		cores = 1
	}
	frac := raw
	if frac > 1 {
		frac = frac / 100.0 // a value >1 is a percent, same rule as the env form
	}
	if frac <= 0 || frac > 1 {
		return 0, false
	}
	w := int(math.Floor(float64(cores)*frac + 0.5))
	if w < 1 {
		w = 1
	}
	if w > cores {
		w = cores
	}
	return w, true
}

// resolveBudgetWorkers turns the two env strings + the machine's core count into the
// resolved matmul worker count and a short source label describing HOW it was derived
// (so a bench can record it). It is pure — it touches no process state — which is what
// makes the precedence table-testable. `cores` is the machine width (GOMAXPROCS at the
// call site); it is clamped to >=1 so a degenerate 0 can never zero the result.
//
// notify, when non-nil, receives a one-line human note for a malformed FAK_BUDGET that
// was ignored (so a typo'd budget surfaces instead of silently running at full width).
func resolveBudgetWorkers(envWorkers, envBudget string, cores int, notify func(string)) (workers int, source string) {
	if cores < 1 {
		cores = 1
	}

	// 1. FAK_WORKERS — absolute override, unchanged historical behavior.
	if s := strings.TrimSpace(envWorkers); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n >= 1 {
			return n, "FAK_WORKERS=" + s
		}
	}

	// 2. FAK_BUDGET — fractional/percent budget of the machine.
	if s := strings.TrimSpace(envBudget); s != "" {
		if frac, ok := parseBudgetFraction(s); ok {
			// budgetToWorkers rounds half-up and floors at 1 — a positive budget always
			// yields >=1 worker (0.01 * 32 -> 1, never 0), and never more than the machine
			// has. parseBudgetFraction has already normalized to a fraction in (0,1].
			w, _ := budgetToWorkers(frac, cores)
			return w, "FAK_BUDGET=" + s
		}
		if notify != nil {
			notify(fmt.Sprintf("FAK_BUDGET=%q is not a fraction in (0,1] or a percent; ignoring (using all %d cores)", s, cores))
		}
	}

	// 3. default — all cores.
	return cores, "default(GOMAXPROCS)"
}

// parseBudgetFraction reads "0.75", "75", or "75%" into a fraction in (0,1]. A value
// <=1 is taken as a fraction directly; a value >1 (with or without a trailing '%') is
// taken as a percent — consistently, so a bare "1.5" is 1.5% (a valid tiny budget that
// floors to one worker), NOT 150%. Anything that doesn't land in (0,1] after that — 0,
// negatives, >100%, non-numeric — is rejected (ok=false) so the caller falls through
// to default.
func parseBudgetFraction(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	percentSign := strings.HasSuffix(s, "%")
	num := strings.TrimSpace(strings.TrimSuffix(s, "%"))
	v, err := strconv.ParseFloat(num, 64)
	if err != nil {
		return 0, false
	}
	if percentSign || v > 1 {
		v = v / 100.0
	}
	if v <= 0 || v > 1 {
		return 0, false
	}
	return v, true
}
