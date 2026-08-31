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
	"sync"
)

const defaultWorkerBudgetSource = "default(GOMAXPROCS)"

var (
	workerBudgetMu sync.Mutex
	gomaxprocsNow  = func() int { return runtime.GOMAXPROCS(0) }
)

// currentWorkerCount snapshots the process worker budget at an operation boundary.
// Only the runtime default is dynamic: explicit environment and flag overrides stay
// pinned until the operator changes them explicitly. Callers retain the returned
// integer for the whole dispatch, so a runtime update never resizes in-flight work.
func currentWorkerCount() int {
	workerBudgetMu.Lock()
	defer workerBudgetMu.Unlock()
	if workerBudgetSource == defaultWorkerBudgetSource {
		w := gomaxprocsNow()
		if w < 1 {
			w = 1
		}
		numWorkers = w
	}
	return numWorkers
}

func currentWorkerBudget() (int, string) {
	workerBudgetMu.Lock()
	defer workerBudgetMu.Unlock()
	if workerBudgetSource == defaultWorkerBudgetSource {
		w := gomaxprocsNow()
		if w < 1 {
			w = 1
		}
		numWorkers = w
	}
	return numWorkers, workerBudgetSource
}

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
	workerBudgetMu.Lock()
	numWorkers = w
	workerBudgetSource = fmt.Sprintf("-budget=%g", frac)
	workerBudgetMu.Unlock()
	return nil
}

// SetWorkers pins an ABSOLUTE matmul worker count after init (n>=1), the post-init
// analogue of FAK_WORKERS for a bench's absolute -jobs flag. Like SetWorkerBudget it
// must run before the first matmul. Returns an error (count untouched) if n < 1.
func SetWorkers(n int) error {
	if n < 1 {
		return fmt.Errorf("workers %d must be >= 1", n)
	}
	workerBudgetMu.Lock()
	numWorkers = n
	workerBudgetSource = fmt.Sprintf("-jobs=%d", n)
	workerBudgetMu.Unlock()
	return nil
}

// Q8DecodeWorkers reports the worker count used by Q8 batch-1 decode GEMVs. On
// darwin/arm64, the default all-core budget over-saturates the shared memory system:
// the M3 Pro scorecard peaks at 4-6 workers, while prefill still benefits from the
// global worker count. Explicit FAK_WORKERS/FAK_BUDGET/-budget choices remain exact.
func Q8DecodeWorkers() int {
	workers, source := currentWorkerBudget()
	w, _ := q8DecodeWorkersFor(workers, source, runtime.GOOS, runtime.GOARCH)
	return w
}

// Q8DecodeWorkerBudget reports how Q8DecodeWorkers was derived, so benchmark JSON can
// state when decode used the Apple Silicon default cap instead of the global budget.
func Q8DecodeWorkerBudget() string {
	workers, budgetSource := currentWorkerBudget()
	_, source := q8DecodeWorkersFor(workers, budgetSource, runtime.GOOS, runtime.GOARCH)
	return source
}

func q8DecodeWorkers() int {
	return Q8DecodeWorkers()
}

// Q8DecodeKernel reports the resolved Q8_0 decode inner-kernel tier for this host and whether
// the FUSED fast decode GEMV (qMatRowsRangeFast) is the active decode path. It answers #3176
// Q1 — "is the EPYC AVX2/AVX-512 Q8 SIMD decode lane actually engaged, or did it fall back to
// the reference path?" — as a queryable fact, no wall-clock guess needed. kernel is
// "avx512"/"avx2"/"scalar" on amd64, "neon-amort"/"neon"/"scalar" on arm64, and "scalar" on a
// no-SIMD build; fused is true only where the fused row kernel fires (amd64 + AVX-512). It is a
// static host property (the tier resolves once at package init via CPUID+XGETBV), so it is
// cheap to read on every decode witness line. Pair it with Q8DecodeWorkers()/
// Q8DecodeWorkerBudget() for the full "engaged and parallel?" picture.
func Q8DecodeKernel() (kernel string, fused bool) { return q8DecodeKernel() }

// decodeWorkersFor is the shared batch-1 decode worker-budget cap for the Q8 and Q4_K decode
// GEMVs. It clamps to >=1, lets an explicit FAK_WORKERS/FAK_BUDGET source bypass the cap, and
// otherwise applies two host caps parameterized by (amd64Div, amd64Max) and a `label` prefix so
// the two decode paths share ONE body (the only real difference is the amd64 divisor/ceiling):
//   - Apple Silicon (darwin/arm64, workers>=8): half-cap to <=6 (shared batch-1 pathology).
//   - Many-core amd64 (workers>=64): cap to workers/amd64Div, held in [8, amd64Max].
//
// The many-core amd64 cap exists because those servers (esp. multi-socket / NUMA) hit the same
// batch-1 decode pathology as Apple Silicon, amplified. A decode GEMV is memory-bound (see
// qMatRows: "decode is memory-bound … spreading rows taps aggregate bandwidth"), and the q/k/v
// and gate/up projections are small (k/v are 256 rows) and "parallelize poorly split 12 ways"
// (quant_gemv_many.go). Splitting them across ~all cores on a 256-thread box hands each worker a
// trivial slice while the parFor barrier — a shared cursor + busy-wait across cores, worse across
// NUMA — dominates, collapsing decode to a fraction of a token/s (#3176). Capping to a modest
// stream count still saturates multi-channel server memory bandwidth without the all-core sync
// collapse. The exact optimum is host-specific, so an operator tunes it with FAK_WORKERS/FAK_BUDGET
// (which set a non-default source and return above); only the genuinely many-core regime (>=64) is
// capped, so desktop/small-server amd64 is byte-for-byte unchanged.
func decodeWorkersFor(workers int, source, goos, goarch, label string, amd64Div, amd64Max int) (int, string) {
	if workers < 1 {
		workers = 1
	}
	if source != defaultWorkerBudgetSource {
		return workers, source
	}
	if goos == "darwin" && goarch == "arm64" && workers >= 8 {
		w := (workers + 1) / 2
		if w > 6 {
			w = 6
		}
		if w < 1 {
			w = 1
		}
		return w, defaultWorkerBudgetSource + "; " + label + "_decode=darwin/arm64-half-cap"
	}
	if goarch == "amd64" && workers >= 64 {
		w := workers / amd64Div
		if w > amd64Max {
			w = amd64Max
		}
		if w < 8 {
			w = 8
		}
		return w, defaultWorkerBudgetSource + "; " + label + "_decode=amd64-manycore-cap"
	}
	return workers, source
}

// q8DecodeWorkersFor caps the Q8 batch-1 decode budget: many-core amd64 to workers/8 (<=16).
func q8DecodeWorkersFor(workers int, source, goos, goarch string) (int, string) {
	return decodeWorkersFor(workers, source, goos, goarch, "q8", 8, 16)
}

// Q4KDecodeWorkers reports the worker count for resident-Q4_K batch-1 decode GEMVs
// (q4kMatRowsInto). Like Q8DecodeWorkers it caps the many-core amd64 default to dodge the
// parFor barrier collapse across NUMA nodes, but at a HIGHER cap than Q8. Witnessed on a
// dual EPYC-7742 (256 threads, 8 NUMA) with real Qwen3.6-27B-Q4_K_M weights
// (experiments/qwen36/cpu-decode-int8-q4k-numa-witness-2026-07-14.md): int8 Q4_K decode
// tok/s vs FAK_WORKERS under `numactl --interleave=all` was 256->0.593 (the UNCAPPED default,
// the worst config), 16->0.566, 32->0.971, 64->1.395, 128->1.392. The uncapped 256-worker
// default collapses and ~64 (= workers/4) is 2.35x faster. The Q8 cap of workers/8 (<=16) is
// too aggressive here (16->0.566): the int8 Q4_K path streams half the bytes/token, so it
// tolerates ~4x the workers before the barrier dominates. A NUMA-node-aware size is the C2
// (#4625) follow-up; this flat workers/4 (<=64) matches the witnessed knee (= 8 workers/node
// x 8 nodes). Operators override with FAK_WORKERS/FAK_BUDGET (non-default source bypasses it).
func Q4KDecodeWorkers() int {
	workers, source := currentWorkerBudget()
	w, _ := q4kDecodeWorkersFor(workers, source, runtime.GOOS, runtime.GOARCH)
	return w
}

func q4kDecodeWorkers() int { return Q4KDecodeWorkers() }

// KQuantDecodeWorkers reports the worker count for resident Q5_K/Q6_K batch-1 decode GEMVs
// (kQuantMatRowsInto). #4974: a q4_k_m artifact is a MIXTURE — the Q4_K majority and the
// Q5_K/Q6_K minority (ffn_down / lm_head, plus mixed-quant routed experts) are streamed by two
// different kernels on the SAME decode step — so the witnessed regime is a property of the
// artifact, not of one kernel: both take the one measured knee. Deliberately delegates to
// Q4KDecodeWorkers rather than declaring a second divisor, because the DA33 sweep witnessed ONE
// knee for the whole model (64 workers on a 256-thread / 8-NUMA host) and a k-quant-specific
// divisor would be an unwitnessed guess. Same derivation, same FAK_WORKERS/FAK_BUDGET override.
func KQuantDecodeWorkers() int { return Q4KDecodeWorkers() }

func kQuantDecodeWorkers() int { return KQuantDecodeWorkers() }

// Q4KDecodeWorkerBudget reports how Q4KDecodeWorkers was derived (for a bench JSON line).
func Q4KDecodeWorkerBudget() string {
	workers, budgetSource := currentWorkerBudget()
	_, source := q4kDecodeWorkersFor(workers, budgetSource, runtime.GOOS, runtime.GOARCH)
	return source
}

// q4kDecodeWorkersFor caps the Q4_K batch-1 decode budget: many-core amd64 to workers/4 (<=64),
// the witnessed int8 Q4_K knee (see Q4KDecodeWorkers). Q4_K streams roughly half the bytes/token
// of Q8, so it tolerates ~4x the workers before the parFor barrier dominates — hence /4 (<=64) vs
// Q8's /8 (<=16). Apple Silicon reuses the shared half-cap.
func q4kDecodeWorkersFor(workers int, source, goos, goarch string) (int, string) {
	return decodeWorkersFor(workers, source, goos, goarch, "q4k", 4, 64)
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
	return cores, defaultWorkerBudgetSource
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
