package programreport

// Live runners for `fak program`: the impure reads behind the two ongoing programs'
// frontier signals. Kept separate from the pure fold (programreport.go) so the
// interpreter/fold/ledger stay unit-testable with no process and no repo.
//
//   - KERNEL OPTIMIZATION — the frontier is "is throughput/parity work still
//     landing?". We witness it from git: ships stamped on the perf/kernel leaves
//     (the lanes where decode/prefill/quant/parity work lands) over a trailing
//     window. A window with shipped frontier-moves is ADVANCING; an empty window is
//     HOLDING (not regressed — a quiet week is not a frontier loss). This is an
//     activity proxy, honestly labeled: it does NOT assert a tok/s number (that lives
//     in the benchmark authority rows), it asserts that the program is being worked.
//   - CACHE OPTIMIZATION — the frontier is the realized KV-reuse ratio over the
//     dogfood cache-value ledger, read through cachevalueledger's #1066-fenced trend
//     gate (the SAME gate `fak cachevalue` enforces, so the program report and the
//     cache scorecard never disagree). REGRESSED when realized reuse fell beyond
//     tolerance; ADVANCING when it rose; HOLDING/INSUFFICIENT otherwise. The honesty
//     fence (the marginal-over-tuned-warm-KV value family) is carried onto the signal
//     so a reader can never mistake it for the forbidden vs-naive multiple.

import (
	"fmt"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/hooks"
	"github.com/anthony-chaudhary/fak/internal/worktype"
)

// DefaultWindowDays is the trailing window the kernel-opt activity signal counts over.
const DefaultWindowDays = 7

// perfLeaves is the set of ship-stamp leaves whose commits move the kernel-optimization
// frontier — the decode/prefill/quant/parity hot-path lanes. A ship stamped on one of
// these (e.g. `(fak metalgemm)`, `(fak compute)`) over the window is a witnessed
// frontier-move. The set is intentionally small + named (not a fuzzy substring match)
// so the activity count is deterministic and auditable.
var perfLeaves = map[string]bool{
	"compute":     true, // CUDA/host compute kernels
	"metalgemm":   true, // Metal GEMM / quant matmul
	"metal":       true, // Metal backend
	"cuda":        true, // CUDA backend
	"simd":        true, // CPU SIMD reducers
	"model":       true, // in-kernel model fusion (the parity work)
	"modelengine": true, // model execution engine
	"engine":      true, // the engine client / decode loop
	"enginecache": true, // engine KV path
	"kernel":      true, // the kernel walker
	"spec":        true, // speculative decode
}

// Collect measures both ongoing programs' frontier signals: kernel-opt from the
// perf-lane git ship window, cache-opt from the cache-value reuse trend gate. A
// signal that cannot be read carries Err (never a silent zero), and the pure
// InterpretPrograms folds the partial/whole-failure verdict.
func Collect(root, cacheLedgerPath string, windowDays int) Programs {
	if windowDays <= 0 {
		windowDays = DefaultWindowDays
	}
	signals := []Signal{
		kernelSignal(root, windowDays),
		cacheSignal(cacheLedgerPath),
	}
	return InterpretPrograms(signals)
}

// kernelSignal witnesses the kernel-optimization frontier from git: the count of
// ships stamped on a perf leaf over the trailing window. Direction is ADVANCING when
// the window carried frontier-moves, HOLDING when it was quiet (an empty window is not
// a regression). The Metric is the activity count, so the trend ledger trends "is the
// program still being worked" across ticks.
func kernelSignal(root string, windowDays int) Signal {
	p, _ := worktype.ProgramFor(worktype.KernelOptimization)
	s := Signal{
		Class: worktype.KernelOptimization,
		Label: worktype.KernelOptimization.Label(),
		Doc:   p.OperatingDoc,
		Note:  "activity proxy: counts perf-lane ships, not a tok/s number (the authority rows hold the throughput claim)",
	}
	moves, err := perfShipsInWindow(root, windowDays)
	if err != "" {
		s.Err = err
		return s
	}
	s.Activity = moves
	s.Metric = float64(moves)
	s.Window = windowLabel(windowDays)
	if moves > 0 {
		s.Direction = "advancing"
		s.Frontier = "perf/parity work landing"
	} else {
		s.Direction = "holding"
		s.Frontier = "no perf-lane ship in window"
	}
	s.OK = true
	return s
}

// cacheSignal witnesses the cache-optimization frontier from the dogfood cache-value
// ledger via the #1066-fenced trend gate. The frontier is the realized reuse ratio;
// the direction follows the gate verdict (REGRESSED -> regressed, OK with a positive
// delta -> advancing, else holding). An absent/thin ledger is HOLDING with the honest
// "insufficient corpus" frontier, never an errored signal — a thin corpus is not a
// measurement failure (matching the gate's fall-open posture).
func cacheSignal(ledgerPath string) Signal {
	if ledgerPath == "" {
		ledgerPath = cachevalueledger.DefaultLedgerRel
	}
	p, _ := worktype.ProgramFor(worktype.CacheOptimization)
	s := Signal{
		Class: worktype.CacheOptimization,
		Label: worktype.CacheOptimization.Label(),
		Doc:   p.OperatingDoc,
		OK:    true,
	}
	g := cachevalueledger.ScoreTrendGate(ledgerPath)
	s.Note = g.PublishableValueFamily
	s.Metric = round3(g.RecentReuseRatio)
	switch g.Verdict {
	case "REGRESSED":
		s.Direction = "regressed"
		s.Frontier = ratioFrontier("realized reuse fell", g.BaselineReuseRatio, g.RecentReuseRatio)
	case "OK":
		if g.DeltaReuseRatio > 0 {
			s.Direction = "advancing"
		} else {
			s.Direction = "holding"
		}
		s.Frontier = ratioFrontier("realized reuse", g.BaselineReuseRatio, g.RecentReuseRatio)
	default: // INSUFFICIENT
		s.Direction = "unknown"
		s.Frontier = "insufficient cache-value corpus to trend (need a thicker multi-turn dogfood window)"
	}
	return s
}

// perfShipsInWindow counts non-merge commits in the trailing window whose subject
// carries a real per-leaf ship-stamp (hooks.StampOf grades trailer|direct) on a perf
// leaf. It reuses the SAME grammar the pre-commit lint + the cadence work dimension
// bind to, so this counts exactly what `dos verify` can bind on a perf lane.
func perfShipsInWindow(root string, windowDays int) (int, string) {
	since := windowSince(windowDays)
	cmd := exec.Command("git", "log", "--no-merges", "--since="+since, "--format=%s", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return 0, "git log failed: " + gitErr(err)
	}
	moves := 0
	for _, line := range strings.Split(string(out), "\n") {
		subj := strings.TrimSpace(line)
		if subj == "" {
			continue
		}
		kind, leaf := hooks.StampOf(subj)
		if (kind == "trailer" || kind == "direct") && perfLeaves[leaf] {
			moves++
		}
	}
	return moves, ""
}

// HeadCommit returns the short HEAD commit of root, or "unknown" — inlined git
// plumbing so this tier-1 leaf imports no sibling composer (the milestonereport
// pattern).
func HeadCommit(root string) string {
	cmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return "unknown"
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return "unknown"
	}
	return s
}

func windowSince(windowDays int) string {
	return fmt.Sprintf("%d days ago", windowDays)
}

func windowLabel(windowDays int) string {
	return fmt.Sprintf("%dd", windowDays)
}

func ratioFrontier(prefix string, from, to float64) string {
	return fmt.Sprintf("%s %.3f -> %.3f", prefix, from, to)
}

func gitErr(err error) string {
	if ee, ok := err.(*exec.ExitError); ok {
		s := strings.TrimSpace(string(ee.Stderr))
		if s != "" {
			return lastLine(s)
		}
	}
	return err.Error()
}

func lastLine(s string) string {
	s = strings.TrimRight(s, "\n")
	if s == "" {
		return ""
	}
	if i := strings.LastIndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[i+1:])
	}
	return strings.TrimSpace(s)
}
