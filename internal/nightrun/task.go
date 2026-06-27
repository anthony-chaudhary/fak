package nightrun

import (
	"regexp"
	"sort"
	"strings"
	"time"
)

// Source records where a collection Task came from, so a reader can tell a
// benchmark-grid cell from a curated open-measurement and from an operator's
// overlay file. It is informational; the selector treats all sources uniformly.
type Source string

const (
	// SourceBenchmark is a Task derived from the in-binary benchmark catalog
	// (internal/benchcatalog) — the menu of runnable benchmarks.
	SourceBenchmark Source = "benchmark"
	// SourceWitness is a curated, still-open measured witness — a named datum the
	// project is waiting on (a PENDING_MEASUREMENT / on-box re-measure). It is a
	// TASK (work to do), never a result, so it cannot overclaim.
	SourceWitness Source = "witness"
	// SourceOverlay is a Task an operator or agent added via the optional overlay
	// file, additive over the built-ins (no recompile needed to enqueue data).
	SourceOverlay Source = "overlay"
)

// Requirement is one thing the box must HAVE for a Task to be feasible. A Task's
// Requires is ANDed: every requirement must be satisfied by the box's probed
// Capabilities. Credentials are modelled separately (CredEnv) because they are
// matched by env-var NAME, not by a fixed enum.
type Requirement string

const (
	// ReqOffline asserts nothing — the Task runs with no weights, GPU, dataset,
	// cred, or network. An empty Requires is treated as offline.
	ReqOffline Requirement = "offline"
	// ReqWeights needs local model weights (an export dir / GGUF / HF snapshot).
	ReqWeights Requirement = "weights"
	// ReqDataset needs an external dataset checked out under testdata/ (e.g. a
	// WebVoyager export), not committed to the repo.
	ReqDataset Requirement = "dataset"
	// ReqCUDA needs an NVIDIA GPU (a CUDA-tagged benchmark or an H200/A100 run).
	ReqCUDA Requirement = "cuda"
	// ReqMetal needs an Apple GPU (a darwin && metal build — the Mac verify node).
	ReqMetal Requirement = "metal"
	// ReqNet needs outbound network (a live credentialed endpoint run).
	ReqNet Requirement = "net"
)

// Value is the importance class of a Task — what makes a datum worth collecting
// before another. The classes are ordered (Frontier is most valuable) and carry
// a normalized weight the selector blends into the score.
type Value string

const (
	// ValueFrontier is a first-of-its-kind number: a capability the project has
	// never measured (a new node's first run, a parity claim still open). Highest.
	ValueFrontier Value = "frontier"
	// ValueWitness is a specific open measured witness the project is blocked on
	// (a residual to re-measure after a fix, an on-box re-measure of a load time).
	ValueWitness Value = "witness"
	// ValueRegression is a recorded baseline aging past its re-check interval —
	// re-measure to catch drift before it surprises us.
	ValueRegression Value = "regression"
	// ValueCoverage fills an empty (capability × workload) hole — useful breadth,
	// but not a frontier or a blocking witness.
	ValueCoverage Value = "coverage"
	// ValueSmoke is a cheap offline re-runnable check — the floor; collected only
	// when nothing more valuable is feasible.
	ValueSmoke Value = "smoke"
)

// weight returns the normalized importance of a Value in [0,1]. The gaps are
// deliberate (a frontier datum is worth ~2.5× a coverage hole) so the class
// dominates the blend without fully swamping novelty/staleness.
func (v Value) weight() float64 {
	switch v {
	case ValueFrontier:
		return 1.0
	case ValueWitness:
		return 0.8
	case ValueRegression:
		return 0.55
	case ValueCoverage:
		return 0.4
	case ValueSmoke:
		return 0.15
	default:
		return 0.3
	}
}

// rank gives a stable integer ordering of the Value classes (higher = more
// important), used only for deterministic tie-breaking and display grouping.
func (v Value) rank() int {
	switch v {
	case ValueFrontier:
		return 5
	case ValueWitness:
		return 4
	case ValueRegression:
		return 3
	case ValueCoverage:
		return 2
	case ValueSmoke:
		return 1
	default:
		return 0
	}
}

// Task is one unit of data to collect. It is the atom the whole package operates
// on: the backlog is a set of Tasks, next() ranks Tasks, the ledger records
// collected Tasks. A Task carries everything a newcomer needs to run it — the
// exact command and what counts as "collected" — so the door is genuinely
// trivial.
type Task struct {
	// ID is the stable, kebab-case key (also the ledger join key). Unique across
	// the whole backlog; a duplicate id fails the backlog assembly loud.
	ID string `json:"id"`
	// Title is the one-line "what datum does this collect."
	Title string `json:"title"`
	// Source records the origin (benchmark / witness / overlay).
	Source Source `json:"source"`
	// Value is the importance class that drives the selector's weighting.
	Value Value `json:"value"`
	// Requires is the ANDed set of box capabilities the Task needs. Empty means
	// offline (always feasible).
	Requires []Requirement `json:"requires,omitempty"`
	// CredEnv is the set of credential env-var NAMES that must be present in the
	// environment for the Task to run (matched by name, never by value — a dump is
	// safe to print). ANDed with Requires.
	CredEnv []string `json:"cred_env,omitempty"`
	// Run is the exact, copy-pasteable command that collects the datum.
	Run string `json:"run"`
	// Acceptance is the human statement of what proves the datum was collected —
	// the artifact path or the headline number to look for. Recorded, not asserted.
	Acceptance string `json:"acceptance"`
	// RecheckDays is how long a collected datum stays "fresh" before a re-collect
	// is overdue. Zero falls back to DefaultRecheckDays.
	RecheckDays int `json:"recheck_days,omitempty"`
	// TimeoutSec is the per-task wall-clock budget for ONE --apply attempt. A task
	// that exceeds it is killed and recorded OBSERVED as a timeout (never as a
	// success), so one slow/hung task can never stall an unattended --loop. Zero
	// falls back to DefaultTaskTimeoutSec. A live serving/throughput collection can
	// raise it via the overlay (`"timeout_sec": 3600`) without recompiling.
	TimeoutSec int `json:"timeout_sec,omitempty"`
	// Manual marks a curated witness whose Run is a HUMAN RECIPE, not an executable
	// command — an operator-setup datum that must never be auto-run regardless of how
	// its Run string happens to read. It is the AUTHORITATIVE form of the autoRunnable
	// placeholder/arrow heuristic: a recipe whose Run is a bare `script.sh   # comment`
	// (no placeholder, no arrow) is still NOT auto-runnable when Manual is set, closing
	// the gap where such a row was exec'd every sweep and recorded a spurious failure.
	// A manual Task stays surfaced by plan/next (an operator runs it by hand) and is
	// recorded OutcomeSkipped by run --apply — never a ledger failed/collected row,
	// never counted against --max. The heuristic remains a backstop for un-flagged rows.
	Manual bool `json:"manual,omitempty"`
	// Doc is the in-repo methodology / issue / authority pointer, or "".
	Doc string `json:"doc,omitempty"`
}

// recheckDays returns the Task's re-check interval, defaulting when unset.
func (t Task) recheckDays() int {
	if t.RecheckDays > 0 {
		return t.RecheckDays
	}
	return DefaultRecheckDays
}

// timeout returns the Task's per-attempt wall-clock budget, defaulting when unset.
func (t Task) timeout() time.Duration {
	if t.TimeoutSec > 0 {
		return time.Duration(t.TimeoutSec) * time.Second
	}
	return time.Duration(DefaultTaskTimeoutSec) * time.Second
}

// placeholderRE matches a `<token>` placeholder in a Run command (e.g.
// `<glm-5.2.gguf>`, `<official-suite>`). The token must start with a non-space so a
// shell input-redirect (`< file`) and a heredoc (`<<EOF`) are NOT matched — only a
// genuine fill-me-in placeholder is.
var placeholderRE = regexp.MustCompile(`<[^>\s][^>]*>`)

// autoRunnable reports whether a Task's Run is a real, executable command the loop
// may auto-run, vs a MANUAL recipe that only describes how to collect the datum. A
// curated witness often carries a prose hint — a `<placeholder>` to fill in, or a
// `→`/` -> ` prose arrow ("script.sh → fak serve + fak agent") — which is not a
// command: exec-ing it just records a spurious failure every sweep. Such a Task is
// surfaced (plan/next still show it as the manual recipe to run by hand) but skipped
// by `run --apply`. Conservative by construction: a concrete command (no placeholder,
// no arrow, non-empty) is auto-runnable, so a real benchmark is never wrongly skipped.
//
// The registry's explicit Manual flag is AUTHORITATIVE: a row marked Manual is never
// auto-run even when its Run reads as a clean command (the `script.sh   # comment`
// shape that has no placeholder and no arrow but still needs operator setup). The
// placeholder/arrow heuristic remains the backstop for any un-flagged recipe row.
func (t Task) autoRunnable() bool {
	if t.Manual {
		return false
	}
	run := strings.TrimSpace(t.Run)
	if run == "" {
		return false
	}
	if strings.Contains(run, "→") || strings.Contains(run, " -> ") {
		return false
	}
	return !placeholderRE.MatchString(run)
}

// DefaultRecheckDays is the fall-back staleness horizon for a Task that does not
// declare its own — the same 14-day default tools/bench_plan.py uses, so the two
// planners agree on what "stale" means.
const DefaultRecheckDays = 14

// DefaultTaskTimeoutSec is the fall-back per-attempt wall-clock budget for a Task
// that does not declare its own. It bounds ONE --apply attempt so an unattended
// --loop cannot be stalled by a single slow or hung command (a full benchmark
// grid, a wedged process); the task is killed and recorded as a timeout, and the
// loop moves on. 15 minutes is generous for the offline/smoke lane while still
// catching a genuinely stuck run. A heavier collection raises it per-task.
const DefaultTaskTimeoutSec = 900

// sortTasks orders a slice of Tasks deterministically by id, in place, and
// returns it — the canonical order the backlog is presented in before scoring.
func sortTasks(ts []Task) []Task {
	sort.Slice(ts, func(i, j int) bool { return ts[i].ID < ts[j].ID })
	return ts
}
