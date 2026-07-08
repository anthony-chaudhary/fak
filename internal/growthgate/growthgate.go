// Package growthgate classifies unbounded-growth "bloat" fingerprints from a
// cheap census of append-only artifacts — the pure, OS-independent heart of
// `fak growthgate`.
//
// WHY THIS EXISTS. A busy multi-session box accumulates append-only ledgers and
// run logs that NOTHING rotates: the dos hook's `.dos/metrics/observations.jsonl`
// (one fsync'd line per hook call, forever), the `.dos/lane-journal.jsonl` WAL,
// `.dispatch-runs/*.log`, `.fak/loops.jsonl`, `fleet-runs/**`. On the reference
// box these had grown to 119 MB, 22 MB, and 47 MB respectively — ~1.1 GB in the
// working tree — silently. Unbounded growth is not just a disk problem: an
// ever-larger fsync'd appendee is itself an I/O-thrash contributor, and folding
// a 119 MB WAL on every read burns CPU. This is the growth twin of the churn
// stall that internal/stallscan classifies: both are invisible to a usage
// dashboard until the box is already hurting.
//
// This package is a RATCHET, in the spirit of windowgate: it takes a census the
// caller gathered (path, size, modification age) and returns a Report — a
// verdict (ok/watch/action) plus the flagged offenders and per-class totals,
// decided by fixed, documented per-class byte budgets. It reads nothing and
// walks nothing (the caller hands it numbers already gathered), so it is fully
// testable and cannot itself add to the growth it measures.
//
// HOT vs COLD. A finding also carries whether the file is still being written
// (modified recently). That distinction is load-bearing for remediation: a COLD
// oversized ledger is safe to rotate/reap; a HOT one is live and must be capped
// at the write site or rotated with care. The classifier reports the fact; it
// never reaps.
package growthgate

import (
	"sort"
	"strings"
)

// Artifact is one file the census found. Size is in bytes; ModAgeSec is seconds
// since the file was last modified (the caller computes now-mtime, keeping this
// package clock-free and deterministic). A negative or zero ModAgeSec is treated
// as "just modified" (hot).
type Artifact struct {
	Path      string  `json:"path"`
	Size      int64   `json:"size_bytes"`
	ModAgeSec float64 `json:"mod_age_sec"`
}

// Class names the growth family a file belongs to. The set is closed and ordered
// by specificity: Classify walks the matchers top-down and takes the first hit,
// so the narrow runtime paths win over the broad extension fallbacks.
type Class string

const (
	ClassDosMetrics  Class = "dos-metrics"  // .dos/metrics/observations.jsonl — telemetry, one line/fsync per hook call
	ClassLaneJournal Class = "lane-journal" // .dos/lane-journal.jsonl — the lease WAL
	ClassDispatchLog Class = "dispatch-log" // .dispatch-runs/* — per-run dispatcher traces
	ClassGoalLog     Class = "goal-log"     // .goal-runs/* — long-running goal traces
	ClassLoops       Class = "loops"        // .fak/loops.jsonl — the loop ledger
	ClassToolproc    Class = "toolproc"     // .fak/toolproc/*.jsonl — tool-process journals
	ClassFleetRun    Class = "fleet-run"    // fleet-runs/** — nightrun poll/watch logs
	ClassLog         Class = "log"          // any other *.log / *.err
	ClassLedger      Class = "ledger"       // any other *.jsonl append-only ledger
	ClassOther       Class = "other"        // fallback (still counted, budgeted by default)
)

// classMatcher maps a normalized-path substring (and/or suffix) to a Class. Order
// matters — the first match wins.
type classMatcher struct {
	sub    string // path substring that must be present (already lower-cased, '/'-separated)
	suffix string // if non-empty, Path must also end with this suffix
	class  Class
}

// matchers is the ordered classification table. Runtime paths are matched by
// their stable substring so an absolute OR repo-relative path both resolve.
var matchers = []classMatcher{
	{sub: "/metrics/observations.jsonl", class: ClassDosMetrics},
	{sub: "metrics/observations.jsonl", class: ClassDosMetrics},
	{sub: "lane-journal.jsonl", class: ClassLaneJournal},
	{sub: ".dispatch-runs/", class: ClassDispatchLog},
	{sub: ".goal-runs/", class: ClassGoalLog},
	{sub: ".fak/loops.jsonl", class: ClassLoops},
	{sub: ".fak/toolproc/", class: ClassToolproc},
	{sub: "fleet-runs/", class: ClassFleetRun},
	{suffix: ".log", class: ClassLog},
	{suffix: ".err", class: ClassLog},
	{suffix: ".jsonl", class: ClassLedger},
}

// ClassifyPath returns the growth Class of a single path. Exported so the
// gatherer and tests can label without duplicating the table.
func ClassifyPath(path string) Class {
	p := strings.ToLower(strings.ReplaceAll(path, "\\", "/"))
	for _, m := range matchers {
		if m.sub != "" && !strings.Contains(p, m.sub) {
			continue
		}
		if m.suffix != "" && !strings.HasSuffix(p, m.suffix) {
			continue
		}
		return m.class
	}
	return ClassOther
}

// Remedy is the one-line, class-specific fix the renderer prints next to a
// finding. It names WHERE the cap belongs, since these differ: telemetry caps at
// the write site, run logs rotate by age, WALs fold/compact.
func (c Class) Remedy() string {
	switch c {
	case ClassDosMetrics:
		return "cap/rotate at the dos write site (observe.go: size-bounded append, drop fsync-per-line)"
	case ClassLaneJournal:
		return "compact the lease WAL (fold resolved leases; keep a bounded tail)"
	case ClassDispatchLog, ClassGoalLog, ClassFleetRun:
		return "rotate by age/size; reap COLD run traces"
	case ClassLoops, ClassToolproc:
		return "rotate the ledger; archive+truncate past a bounded tail"
	case ClassLog:
		return "rotate by size; reap when COLD"
	case ClassLedger:
		return "bound the append (rotate or archive+truncate)"
	default:
		return "review the writer for a missing size/age bound"
	}
}

// Disposable reports whether a COLD, over-budget file of this class is safe to
// HARD-DELETE. Only pure sink classes qualify — append-only logs and advisory
// telemetry that carry no state anyone reads back. WALs and hash-chained ledgers
// (lane-journal, loops, toolproc, and the catch-all ledger class whose chaining
// is unknown) are NEVER disposable: deleting them can fork a chain or drop durable
// state, so they must be bounded at their write site (rotate/compact), not reaped.
func (c Class) Disposable() bool {
	switch c {
	case ClassDosMetrics, ClassDispatchLog, ClassGoalLog, ClassFleetRun, ClassLog:
		return true
	default: // ClassLaneJournal, ClassLoops, ClassToolproc, ClassLedger, ClassOther
		return false
	}
}

// Severity is the coarse band for a single artifact and for the whole report.
type Severity string

const (
	SevOK     Severity = "ok"     // under the warn budget
	SevWatch  Severity = "watch"  // at/over warn, under fail
	SevAction Severity = "action" // at/over fail — remediation owed
)

// rank orders severities for aggregation (report verdict = worst finding).
func (s Severity) rank() int {
	switch s {
	case SevAction:
		return 2
	case SevWatch:
		return 1
	default:
		return 0
	}
}

// ClassBudget is the warn/fail byte pair for one class.
type ClassBudget struct {
	Warn int64 `json:"warn_bytes"`
	Fail int64 `json:"fail_bytes"`
}

// Budget is the decision boundary set. PerClass overrides DefaultWarn/DefaultFail
// for the classes that should be held to a different bound (telemetry + WALs are
// held tight; human run logs are allowed to be larger). HotAgeSec is the age
// below which a file is "hot" (still being written).
type Budget struct {
	DefaultWarn int64
	DefaultFail int64
	PerClass    map[Class]ClassBudget
	HotAgeSec   float64
}

const (
	mb = 1 << 20
)

// DefaultBudget returns the calibrated defaults. Telemetry and the lease WAL are
// held tight (they should be rotated aggressively); human run logs are allowed a
// larger envelope before they count as bloat. These bands flag every offender
// seen on the reference box (119 MB / 47 MB / 22 MB) as ACTION while leaving a
// fresh few-MB ledger OK.
func DefaultBudget() Budget {
	return Budget{
		DefaultWarn: 8 * mb,
		DefaultFail: 32 * mb,
		HotAgeSec:   300, // modified within 5 min ⇒ still growing
		PerClass: map[Class]ClassBudget{
			ClassDosMetrics:  {Warn: 4 * mb, Fail: 16 * mb},
			ClassLaneJournal: {Warn: 4 * mb, Fail: 16 * mb},
			ClassLoops:       {Warn: 8 * mb, Fail: 24 * mb},
			ClassToolproc:    {Warn: 8 * mb, Fail: 24 * mb},
			ClassDispatchLog: {Warn: 16 * mb, Fail: 64 * mb},
			ClassGoalLog:     {Warn: 16 * mb, Fail: 64 * mb},
			ClassFleetRun:    {Warn: 16 * mb, Fail: 64 * mb},
		},
	}
}

// budgetFor resolves the warn/fail pair for a class, falling back to the defaults.
func (b Budget) budgetFor(c Class) ClassBudget {
	if cb, ok := b.PerClass[c]; ok {
		if cb.Warn > 0 || cb.Fail > 0 {
			return cb
		}
	}
	return ClassBudget{Warn: b.DefaultWarn, Fail: b.DefaultFail}
}

// Finding is one flagged artifact plus its verdict and the class remedy.
type Finding struct {
	Path      string   `json:"path"`
	Class     Class    `json:"class"`
	Size      int64    `json:"size_bytes"`
	Severity  Severity `json:"severity"`
	Hot       bool     `json:"hot"`
	ModAgeSec float64  `json:"mod_age_sec"`
	Remedy    string   `json:"remedy"`
}

// Reapable reports whether a finding is safe to hard-delete right now: it is over
// the ACTION budget, COLD (not recently written, so no live writer depends on it),
// and of a disposable class (a pure log/telemetry sink, never a WAL/chained
// ledger). A reaper must delete ONLY findings for which this is true.
func (f Finding) Reapable() bool {
	return f.Severity == SevAction && !f.Hot && f.Class.Disposable()
}

// ReapPlan partitions a report's findings into what a reaper may delete now
// (Reap) versus what is over budget but must be left in place (Protected: HOT, or
// a non-disposable WAL/chained ledger that has to be bounded at its write site).
// Pure — the caller performs the actual deletion. Reap is sorted largest-first so
// the biggest reclaim leads.
func ReapPlan(rep Report) (reap, protected []Finding) {
	for _, f := range rep.Findings {
		if f.Reapable() {
			reap = append(reap, f)
		} else {
			protected = append(protected, f)
		}
	}
	sort.SliceStable(reap, func(i, j int) bool { return reap[i].Size > reap[j].Size })
	return reap, protected
}

// ClassTotal is the per-class rollup across the whole census (every scanned file,
// not only the flagged ones), so the report can show where the bytes actually live.
type ClassTotal struct {
	Class    Class `json:"class"`
	Bytes    int64 `json:"bytes"`
	Count    int   `json:"count"`
	MaxBytes int64 `json:"max_bytes"`
	Flagged  int   `json:"flagged"` // how many of this class crossed >= warn
}

// Report is the classification result.
type Report struct {
	Verdict    Severity     `json:"verdict"`     // worst finding severity (ok if none)
	Scanned    int          `json:"scanned"`     // artifacts classified
	TotalBytes int64        `json:"total_bytes"` // sum across all scanned artifacts
	Findings   []Finding    `json:"findings"`    // watch+action only, worst-first then size desc
	ByClass    []ClassTotal `json:"by_class"`    // all classes present, bytes desc
}

// severityFor decides one artifact's band against its class budget.
func severityFor(size int64, cb ClassBudget) Severity {
	switch {
	case cb.Fail > 0 && size >= cb.Fail:
		return SevAction
	case cb.Warn > 0 && size >= cb.Warn:
		return SevWatch
	default:
		return SevOK
	}
}

// Classify decides the report from a census and a budget. It is pure: same input,
// same output, no I/O. Findings include only WATCH and ACTION artifacts (an OK
// file is counted in totals but not listed); the report Verdict is the worst
// finding severity, or OK when nothing crossed a warn budget.
func Classify(arts []Artifact, b Budget) Report {
	r := Report{Verdict: SevOK}
	byClass := map[Class]*ClassTotal{}

	for _, a := range arts {
		c := ClassifyPath(a.Path)
		cb := b.budgetFor(c)
		sev := severityFor(a.Size, cb)

		r.Scanned++
		r.TotalBytes += a.Size

		ct := byClass[c]
		if ct == nil {
			ct = &ClassTotal{Class: c}
			byClass[c] = ct
		}
		ct.Bytes += a.Size
		ct.Count++
		if a.Size > ct.MaxBytes {
			ct.MaxBytes = a.Size
		}

		if sev == SevOK {
			continue
		}
		ct.Flagged++
		if sev.rank() > r.Verdict.rank() {
			r.Verdict = sev
		}
		r.Findings = append(r.Findings, Finding{
			Path:      a.Path,
			Class:     c,
			Size:      a.Size,
			Severity:  sev,
			Hot:       a.ModAgeSec <= b.HotAgeSec,
			ModAgeSec: a.ModAgeSec,
			Remedy:    c.Remedy(),
		})
	}

	sortFindings(r.Findings)
	r.ByClass = flattenClassTotals(byClass)
	return r
}

// sortFindings orders worst-first, then largest-first, then path — deterministic.
func sortFindings(fs []Finding) {
	sort.SliceStable(fs, func(i, j int) bool {
		if fs[i].Severity.rank() != fs[j].Severity.rank() {
			return fs[i].Severity.rank() > fs[j].Severity.rank()
		}
		if fs[i].Size != fs[j].Size {
			return fs[i].Size > fs[j].Size
		}
		return fs[i].Path < fs[j].Path
	})
}

// flattenClassTotals returns the per-class rollups sorted by bytes desc (ties by
// class name) for a stable render.
func flattenClassTotals(m map[Class]*ClassTotal) []ClassTotal {
	out := make([]ClassTotal, 0, len(m))
	for _, v := range m {
		out = append(out, *v)
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Bytes != out[j].Bytes {
			return out[i].Bytes > out[j].Bytes
		}
		return out[i].Class < out[j].Class
	})
	return out
}
