package dispatchtick

// footprint_ceiling.go -- the host_footprint admission cap term (#3600): a hard
// per-host ceiling on CONCURRENT workers derived from what a worker ACTUALLY costs
// the OS, not from a guessed constant.
//
// WHY. The proc resource guard (EvaluateProcGuard) gates only on a per-process
// runaway threshold (MaxThreads=2000, Handles/WorkingSet opt-in), so it happily
// admits new workers while the host is already near its REAL aggregate resource
// ceiling -- #3153 documents 41 worker processes never tripping the per-process
// threshold while the box thrashes. The missing notion is "this host holds at most
// N concurrent workers", where N is keyed on the measured per-worker resource draw
// (handles + threads + working-set, plus the conhost count a worker drags in), not
// on any single process being individually pathological.
//
// The shape: the impure shell attributes rows of the snapshot procguard.Collect
// already takes to worker run-ids (FootprintSampleFromProcs sums one worker's
// process tree), folds recent samples into a rolling per-worker footprint
// (MeasureWorkerFootprint), and prices it against configured host budgets:
//
//	max_workers = min(handle_budget/hpw, thread_budget/tpw, ws_budget/wspw, conhost_budget/cpw)
//
// (WorkerFootprint.MaxWorkers). ApplyHostFootprintCeiling then folds that ceiling
// into an already-evaluated preflight: once live workers sit AT/OVER the ceiling the
// admission refuses with the structured RefuseHostFootprint verdict, and below
// it admission is unchanged (the term only ever LOWERS the effective cap, and the
// zero value -- no budgets configured or no measurement -- is a byte-for-byte
// no-op, so the existing thread gate keeps its behavior).
//
// NOT A DUP. #3157's churn term is a spawn-RATE signal (processes born per sample
// interval); this is a resident-footprint LEVEL ceiling -- they compose (rate gate +
// level gate). The #1337 HostCapacityWith gradient prices GUESSED per-worker charges
// (Host* consts) against free cores/RAM; this term prices the MEASURED footprint
// against explicit host budgets, so a box whose workers run fat is capped by what
// they actually cost. FootprintCheck.Vars carries the {live_workers, worker_ceiling,
// limiting_dimension} triple the shell surfaces on /debug/vars.

import (
	"fmt"
	"strings"
)

// RefuseHostFootprint is the verdict a spawn preflight returns when the measured
// host-footprint ceiling is the sole binding admission term: live workers already sit
// at/over the number of workers the host's resource budgets can hold. It is not
// safe-to-spawn, so the sweep stops on it exactly as it does on REFUSE_AT_CAP /
// REFUSE_NO_SEAT / REFUSE_GATE / REFUSE_RATE_LIMIT / REFUSE_HOST_CHURN. The string is
// the WIRE token (#3600's structured reason); the Go symbol deliberately does not
// carry the sibling verdict constants' prefixed spelling (concept-family escape) and
// renaming it must never change the wire string (pinned by test).
const RefuseHostFootprint = "REFUSE_HOST_FOOTPRINT"

// DefaultFootprintMinWorkers is the cold-start floor: the fewest workers the
// footprint term admits even when the priced ceiling reads zero. A rolling
// measurement skewed by one bloated worker must throttle GROWTH, not liveness -- a
// floor of 0 would freeze a cold fleet at a zero cap with no probe left to refresh
// the measurement that imposed it. Matches the churn/rate cold-start convention.
const DefaultFootprintMinWorkers = 1

// FootprintSample is one worker-attributed OS footprint observation: the summed
// handles / threads / working-set of every process in one worker's tree at one
// snapshot, plus the console hosts (conhost/openconsole) that tree drags in. The
// impure shell builds these from the procguard.Collect snapshot keyed by worker
// run-id; tests build them synthetically.
type FootprintSample struct {
	Handles      int `json:"handles"`
	Threads      int `json:"threads"`
	WorkingSetMB int `json:"ws_mb"`
	Conhosts     int `json:"conhosts"`
}

// FootprintSampleFromProcs sums one worker's attributed process rows into a single
// footprint sample. Missing dimensions (nil pointers on ProcInfo) contribute zero
// rather than poisoning the sum, and console-host rows (conhost / openconsole) are
// additionally counted so the per-worker console cost -- a real handle/thread sink
// on Windows -- is visible as its own dimension.
func FootprintSampleFromProcs(procs []ProcInfo) FootprintSample {
	var s FootprintSample
	for _, p := range procs {
		s.Handles += ptrIntValue(p.Handles)
		s.Threads += ptrIntValue(p.Threads)
		s.WorkingSetMB += ptrIntValue(p.WorkingSetMB)
		name := strings.ToLower(strings.TrimSpace(p.Name))
		name = strings.TrimSuffix(name, ".exe")
		if name == "conhost" || name == "openconsole" {
			s.Conhosts++
		}
	}
	return s
}

// WorkerFootprint is the rolling per-worker OS cost the ceiling is priced from:
// each dimension is the ceiling-rounded mean of recent worker-attributed samples.
// A zero dimension means "unmeasured" and never binds (the term refuses to divide
// by a guess). The zero value as a whole means "no measurement" and disables the
// term entirely.
type WorkerFootprint struct {
	Handles      int `json:"handles"`
	Threads      int `json:"threads"`
	WorkingSetMB int `json:"ws_mb"`
	Conhosts     int `json:"conhosts"`
}

// MeasureWorkerFootprint folds worker-attributed samples into the rolling
// per-worker footprint. Each dimension is the mean rounded UP (a conservative
// per-worker price yields a conservative -- lower -- ceiling; truncating would let
// fractional cost disappear and overbook the box). No samples means no measurement:
// the zero value, which disables the term.
func MeasureWorkerFootprint(samples []FootprintSample) WorkerFootprint {
	if len(samples) == 0 {
		return WorkerFootprint{}
	}
	var handles, threads, ws, conhosts int
	for _, s := range samples {
		handles += s.Handles
		threads += s.Threads
		ws += s.WorkingSetMB
		conhosts += s.Conhosts
	}
	n := len(samples)
	ceilDiv := func(sum int) int {
		if sum <= 0 {
			return 0
		}
		return (sum + n - 1) / n
	}
	return WorkerFootprint{
		Handles:      ceilDiv(handles),
		Threads:      ceilDiv(threads),
		WorkingSetMB: ceilDiv(ws),
		Conhosts:     ceilDiv(conhosts),
	}
}

// HostFootprintBudgets are the host-wide resource budgets the measured per-worker
// footprint is priced against. Every dimension is OPT-IN: a zero/negative budget
// leaves that dimension unbounded, and the zero value disables the term, so a
// caller that wires nothing keeps the existing admission behavior byte-for-byte.
// The impure shell resolves these from the FAK_FOOTPRINT_MAX_* env knobs
// (DefaultHostFootprintBudgets); the pure fold itself never reads env.
type HostFootprintBudgets struct {
	MaxHandles      int `json:"max_handles"`
	MaxThreads      int `json:"max_threads"`
	MaxWorkingSetMB int `json:"max_ws_mb"`
	MaxConhosts     int `json:"max_conhosts"`
}

// DefaultHostFootprintBudgets resolves the host budgets from the env knobs
// (FAK_FOOTPRINT_MAX_HANDLES, FAK_FOOTPRINT_MAX_THREADS, FAK_FOOTPRINT_MAX_WS_MB,
// FAK_FOOTPRINT_MAX_CONHOSTS). All default to 0 -- unset -- so the ceiling is
// strictly opt-in per host: budgets are a property of the box, not the binary.
func DefaultHostFootprintBudgets() HostFootprintBudgets {
	return HostFootprintBudgets{
		MaxHandles:      envPosInt("FAK_FOOTPRINT_MAX_HANDLES", 0),
		MaxThreads:      envPosInt("FAK_FOOTPRINT_MAX_THREADS", 0),
		MaxWorkingSetMB: envPosInt("FAK_FOOTPRINT_MAX_WS_MB", 0),
		MaxConhosts:     envPosInt("FAK_FOOTPRINT_MAX_CONHOSTS", 0),
	}
}

// FootprintCeiling is the priced result of MaxWorkers: how many concurrent workers
// the host budgets hold at the measured per-worker cost, and which dimension binds.
type FootprintCeiling struct {
	// Bound reports whether any dimension had BOTH a positive budget and a positive
	// measured per-worker cost. Unbound means the term abstains entirely.
	Bound bool `json:"bound"`
	// MaxWorkers is the limiting-dimension ceiling: min over the bounded dimensions
	// of budget/per-worker cost. Zero is a real answer ("not even one worker fits
	// the budget"), distinct from Bound=false.
	MaxWorkers int `json:"max_workers"`
	// Limiting names the dimension that set the ceiling: "handles", "threads",
	// "ws_mb", or "conhost". On a tie the canonical order above wins so the
	// reported limiter is deterministic across identical inputs.
	Limiting string `json:"limiting"`
	// Components carries each bounded dimension's individual ceiling for
	// observability (the same shape HostCapacityInfo.Components uses).
	Components map[string]int `json:"components"`
}

// MaxWorkers prices the measured per-worker footprint against the host budgets:
//
//	max_workers = min(handle_budget/hpw, thread_budget/tpw, ws_budget/wspw, conhost_budget/cpw)
//
// over the dimensions where both a budget and a measurement exist. A dimension with
// no budget is unbounded; a dimension with no measurement cannot bind (never divide
// by a guess). Integer division truncates toward zero deliberately: a budget that
// holds 4.7 workers holds 4. Iteration is a fixed canonical order (handles, threads,
// ws_mb, conhost) so a tie for the minimum names a deterministic limiter.
func (fp WorkerFootprint) MaxWorkers(b HostFootprintBudgets) FootprintCeiling {
	c := FootprintCeiling{Components: map[string]int{}}
	for _, dim := range []struct {
		name      string
		budget    int
		perWorker int
	}{
		{name: "handles", budget: b.MaxHandles, perWorker: fp.Handles},
		{name: "threads", budget: b.MaxThreads, perWorker: fp.Threads},
		{name: "ws_mb", budget: b.MaxWorkingSetMB, perWorker: fp.WorkingSetMB},
		{name: "conhost", budget: b.MaxConhosts, perWorker: fp.Conhosts},
	} {
		if dim.budget <= 0 || dim.perWorker <= 0 {
			continue
		}
		ceiling := dim.budget / dim.perWorker
		c.Components[dim.name] = ceiling
		if !c.Bound || ceiling < c.MaxWorkers {
			c.MaxWorkers = ceiling
			c.Limiting = dim.name
		}
		c.Bound = true
	}
	return c
}

// FootprintCheck carries the MEASURED per-worker footprint and the host budgets the
// admission fold prices it against. The zero value (no budgets, no measurement) is
// a strict no-op, so a caller that wires nothing keeps the existing behavior
// byte-for-byte.
type FootprintCheck struct {
	// PerWorker is the rolling per-worker OS cost (MeasureWorkerFootprint over
	// procguard-snapshot samples attributed by worker run-id) -- a measurement,
	// never a worker self-report.
	PerWorker WorkerFootprint
	// Budgets are the host-wide resource budgets (DefaultHostFootprintBudgets in
	// the shell; explicit values in tests).
	Budgets HostFootprintBudgets
	// MinWorkers is the cold-start floor the ceiling holds to even when the priced
	// ceiling reads zero; <= 0 means DefaultFootprintMinWorkers.
	MinWorkers int
}

// floor resolves the cold-start allowance, defaulting a zero/negative MinWorkers to
// DefaultFootprintMinWorkers so the zero-value keeps the liveness-preserving default.
func (f FootprintCheck) floor() int {
	if f.MinWorkers <= 0 {
		return DefaultFootprintMinWorkers
	}
	return f.MinWorkers
}

// Ceiling prices the check's measurement against its budgets.
func (f FootprintCheck) Ceiling() FootprintCeiling {
	return f.PerWorker.MaxWorkers(f.Budgets)
}

// Vars is the /debug/vars payload for this term: the {live_workers, worker_ceiling,
// limiting_dimension} triple the shell publishes so an operator (and the multi-host
// placer) can read the host's truthful capacity at a glance. An unbound term
// publishes a nil ceiling and an empty limiting dimension: "no ceiling configured"
// must stay distinguishable from "ceiling is huge".
func (f FootprintCheck) Vars(live int) map[string]any {
	c := f.Ceiling()
	var ceiling any
	if c.Bound {
		ceiling = c.MaxWorkers
	}
	return map[string]any{
		"live_workers":       live,
		"worker_ceiling":     ceiling,
		"limiting_dimension": c.Limiting,
	}
}

// ApplyHostFootprintCeiling folds the measured host-footprint ceiling into an
// already-evaluated preflight as the host_footprint cap term. The ceiling (held to
// the cold-start floor) can only LOWER the effective cap:
//
//   - Live workers AT/OVER the ceiling: admitting one more would push the host past
//     the budget its measured per-worker cost supports, so the fold refuses with the
//     structured RefuseHostFootprint verdict and the sweep stops on it.
//   - Live workers BELOW the ceiling: the launch is still admitted (verdict stays
//     SPAWN_OK); when the ceiling is tighter than the existing cap it becomes the
//     visible cap term (cap_terms.Limiting "footprint") so headroom reads truthfully.
//
// The fold is a no-op when the preflight ALREADY refused for a higher-precedence
// reason (the fleet is then not growing, so the ceiling is not the sole binding
// term), when nothing is wired (no budgets or no measurement -- the zero value),
// and when the ceiling meets/exceeds the existing cap (the term never manufactures
// capacity and never RAISES a cap).
func ApplyHostFootprintCeiling(res PreflightResult, f FootprintCheck) PreflightResult {
	// Bottom-up backpressure on a SAFE-to-spawn preflight only.
	if !res.OK {
		return res
	}
	c := f.Ceiling()
	if !c.Bound {
		return res
	}
	// Hold at max(ceiling, floor): a zero priced ceiling still keeps one cold-start
	// probe live so a skewed rolling measurement can be refreshed, never a frozen
	// zero cap.
	hold := c.MaxWorkers
	if floor := f.floor(); hold < floor {
		hold = floor
	}
	if hold >= res.Cap {
		return res
	}
	res.Cap = hold
	res.Headroom = hold - res.Live
	res.CapTerms.EffectiveCap = hold
	res.CapTerms.Limiting = "footprint"
	if res.Headroom > 0 {
		// Below the ceiling: admission unchanged (still SPAWN_OK), just a tighter,
		// truthful cap.
		return res
	}
	res.OK = false
	res.Verdict = RefuseHostFootprint
	res.Reason = hostFootprintReason(c, f, res.Live, hold)
	return res
}

// hostFootprintReason cites the evidence behind a footprint refusal: the live count
// against the ceiling, the limiting dimension with its budget/per-worker price, and
// the full measured footprint -- so a reader can re-derive the ceiling instead of
// trusting it.
func hostFootprintReason(c FootprintCeiling, f FootprintCheck, live, hold int) string {
	limiting := c.Limiting
	if limiting == "" {
		limiting = "unknown"
	}
	return fmt.Sprintf("host footprint ceiling reached: %d live worker(s) >= ceiling %d (limiting dimension %s; priced ceiling %d, cold-start floor %d). Measured per-worker footprint handles=%d threads=%d ws_mb=%d conhost=%d against host budgets handles=%d threads=%d ws_mb=%d conhost=%d -- the host cannot hold another worker at what a worker actually costs. Wait for a worker to exit or retune the FAK_FOOTPRINT_MAX_* budgets; this is the resident-footprint LEVEL gate that composes with the spawn-churn RATE gate.",
		live, hold, limiting, c.MaxWorkers, f.floor(),
		f.PerWorker.Handles, f.PerWorker.Threads, f.PerWorker.WorkingSetMB, f.PerWorker.Conhosts,
		f.Budgets.MaxHandles, f.Budgets.MaxThreads, f.Budgets.MaxWorkingSetMB, f.Budgets.MaxConhosts)
}
