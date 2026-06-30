// Package worktype names the closed set of WORK CLASSES the project-management
// surfaces sort work into — the single source of truth that lets the milestone
// roadmap and the `fak program` report draw the same line between an ONGOING
// OPTIMIZATION PROGRAM and a DISCRETE DELIVERABLE EPIC.
//
// The distinction this package exists to make. The fleet's planning surfaces
// (internal/milestonereport, internal/cadencereport) historically folded every
// tracked epic into one undifferentiated "roadmap" measured by child-completion
// percent. That is a category error for the two long-horizon programs at the core
// of fak's value:
//
//   - KERNEL OPTIMIZATION — pushing decode/prefill throughput and numeric parity
//     toward and past SOTA on each model x backend cell. It is never "done": there
//     is always a faster kernel. Its honest measure is a FRONTIER (the best number
//     witnessed so far) and a TREND (is the frontier still advancing?), not a
//     percent-complete bar. A "60% complete" line on kernel-opt is meaningless.
//   - CACHE OPTIMIZATION — the agent-memory-and-reuse value-add: multi-agent KV
//     reuse, O(1) bounded context + queryable history, provider-cache preservation,
//     addressable KV deletion. Like kernel-opt it is an ongoing frontier program
//     with its own operating spine (docs/CACHE-FRONTIER-OPERATING-PLAN.md) and its
//     own evidence ledgers (the cache-value roll-up + the cache-frontier review
//     ledger), NOT a deliverable that completes.
//
// A DISCRETE EPIC, by contrast, is a deliverable with a definition of done — the
// native agent harness, release-at-agentic-speed, support-maturity disambiguation.
// Child-completion percent IS the right lens for those: they converge on 100% and
// then close.
//
// Keeping the two classes apart in the planning surfaces stops an ongoing program
// from being mis-read as "stalled at 40%" (it has no 100%) and stops a discrete
// epic from hiding inside a frontier trend. This package is the one table that
// names the classes, defines them, and classifies an epic by number — so both
// reports sort the same way and a reclassification is a one-line edit here, not a
// scattered set of magic numbers across two report packages.
//
// The package is pure and stdlib-only (tier 1): it imports nothing internal and
// reads no disk, so the report packages can fold it without a process or a repo.
package worktype

import "sort"

// Class is the closed work-class vocabulary. A value outside this set is a bug,
// not a lower-priority bucket — the same closed-vocabulary discipline the kernel
// applies to a refusal reason or a maturity rung.
type Class string

const (
	// KernelOptimization is the ONGOING throughput/parity program over the
	// model x backend grid. Measured by a frontier + trend, never a completion %.
	KernelOptimization Class = "kernel-optimization"
	// CacheOptimization is the ONGOING agent-memory-and-reuse program (the cache
	// frontier). Measured by a frontier + trend, never a completion %.
	CacheOptimization Class = "cache-optimization"
	// DiscreteEpic is a deliverable with a definition of done — measured by
	// child-issue completion percent, which converges on 100% and then closes.
	DiscreteEpic Class = "discrete-epic"
)

// Ongoing reports whether a class is an ongoing optimization PROGRAM (frontier +
// trend, never "done") rather than a discrete deliverable. It is the one predicate
// the reports branch on to decide whether a completion percent is meaningful for a
// row. An unknown class is treated as discrete (the conservative default: show the
// percent, do not invent a frontier we cannot measure).
func (c Class) Ongoing() bool {
	return c == KernelOptimization || c == CacheOptimization
}

// Label is the short human label for a class, for a render line or a Slack card.
func (c Class) Label() string {
	switch c {
	case KernelOptimization:
		return "kernel-optimization"
	case CacheOptimization:
		return "cache-optimization"
	case DiscreteEpic:
		return "discrete-epic"
	default:
		return string(c)
	}
}

// Definition is the one-line definition of a class — the written distinction the
// disambiguation discipline owes for any confusable concept. Rendered in the report
// header so an operator reading the split sees WHY a program is not on a % bar.
func (c Class) Definition() string {
	switch c {
	case KernelOptimization:
		return "ongoing throughput/parity program over the model x backend grid; measured by a frontier + trend, never 'done'"
	case CacheOptimization:
		return "ongoing agent-memory-and-reuse program (the cache frontier); measured by a frontier + trend, never 'done'"
	case DiscreteEpic:
		return "a deliverable with a definition of done; measured by child-issue completion %, which converges on 100% and closes"
	default:
		return "unknown work class"
	}
}

// Programs is the ordered list of the ongoing-program classes (kernel-opt then
// cache-opt). The `fak program` report iterates this so a new program is added by
// extending this slice (and the registry below), never by editing the report loop.
var Programs = []Class{KernelOptimization, CacheOptimization}

// Program is one ongoing-program definition: its class, the GitHub track label that
// scopes its work, and the canonical doc that operates it. The reports read this to
// route an epic and to point an operator at the program's operating spine.
type Program struct {
	// Class is the work class (KernelOptimization | CacheOptimization).
	Class Class
	// TrackLabel is the GitHub label whose issues belong to this program (the same
	// label the milestone roadmap can resolve children by). Empty when the program
	// is tracked by epic membership alone.
	TrackLabel string
	// OperatingDoc is the repo-relative canonical operating plan for the program —
	// the page that defines its frontier and its decision fences.
	OperatingDoc string
	// Blurb is a one-line description of the program's product outcome.
	Blurb string
}

// programRegistry is the canonical per-program metadata, keyed by class. Edit this
// (not a report's logic) to add a program or re-point its track/doc.
var programRegistry = map[Class]Program{
	KernelOptimization: {
		Class:        KernelOptimization,
		TrackLabel:   "track/B-performance",
		OperatingDoc: "docs/perf-parity-rsi-loop.md",
		Blurb:        "decode/prefill throughput + numeric parity toward and past SOTA per model x backend cell",
	},
	CacheOptimization: {
		Class:        CacheOptimization,
		TrackLabel:   "agentic-serving",
		OperatingDoc: "docs/CACHE-FRONTIER-OPERATING-PLAN.md",
		Blurb:        "agent memory + reuse as a kernel service: multi-agent KV reuse, O(1) bounded context, provider-cache preservation, addressable KV deletion",
	},
}

// ProgramFor returns the program metadata for an ongoing-program class, and ok=false
// for a class that is not an ongoing program (DiscreteEpic or an unknown value).
func ProgramFor(c Class) (Program, bool) {
	p, ok := programRegistry[c]
	return p, ok
}

// epicClass is the canonical epic-number -> work-class map. It is the one place the
// project says "this tracked epic is an ongoing program, not a deliverable." An epic
// absent from this map classifies as DiscreteEpic — the conservative default, so a
// newly-tracked epic shows a completion % until someone consciously declares it a
// program here. Seeded from the live fleet epics:
//
//	#1010 GLM-5.2 through fak's kernel        -> kernel-optimization (the kernel perf program's flagship cell)
//	#1301 cache-effectiveness P&L roll-up     -> cache-optimization  (the cache frontier's evidence ledger)
//	#1351 vDSO live on the served-turn hot path-> cache-optimization (dedup duplicate read executions = a reuse win)
//	#1217 context safety as a first-class prop -> cache-optimization  (the O(1) bounded-context value floor)
//
// Every other tracked epic is a discrete deliverable.
var epicClass = map[int]Class{
	1010: KernelOptimization,
	1301: CacheOptimization,
	1351: CacheOptimization,
	1217: CacheOptimization,
}

// ClassifyEpic returns the work class for a tracked epic number. An epic not in the
// declared map is a DiscreteEpic (the conservative default).
func ClassifyEpic(number int) Class {
	if c, ok := epicClass[number]; ok {
		return c
	}
	return DiscreteEpic
}

// DeclaredEpics returns the epic numbers explicitly declared as ongoing programs, in
// ascending order. Exported so a test (or a report) can assert the declared set
// without reaching into the unexported map.
func DeclaredEpics() []int {
	nums := make([]int, 0, len(epicClass))
	for n := range epicClass {
		nums = append(nums, n)
	}
	sort.Ints(nums)
	return nums
}
