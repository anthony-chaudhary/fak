// reconcile.go — the #10673 started-vs-terminal reconciliation gate: fold a
// corpus or day directory counting task_started against OBSERVED terminal
// records (task_complete, turn_aborted), then against the fold's SYNTHESIZED
// terminals (Superseded, ProcessDeath — codexlifecycle.go), and report BOTH
// numbers so neither can hide the other.
//
// THE TWO NUMBERS AND WHY BOTH. The audited day (2026-09-01) shows 498
// task_started against 288 task_complete and 25 turn_aborted — ~185 started
// tasks with no terminal record of any kind (#10673). That RAW delta is the
// honest producer-side gap: it is computed from event counts alone and a
// healthy run reports zero. The post-synthesis RESIDUAL is the reader-side
// number: after Fold types every start (Complete/Aborted observed;
// Superseded/ProcessDeath synthesized; Live kept open), the residual is what
// synthesis could NOT close — a live turn stays unaccounted because
// synthesizing an end for a running task is the fabrication this package
// fences. A corpus whose raw delta is large but residual is zero has a
// journaling gap, not an accounting gap; a nonzero residual means the fold
// itself could not close the corpus and the report must be read with that
// caveat (AllStartsTyped).
//
// Read-side only: this scan never writes to the journals (fak is read-only
// over Codex rollout stores). Aggregate counts and provider/version rows
// only — no paths, session ids, prompts, or content in the report (the root
// directory is the operator's own --root argument, the same scrubbing
// contract as ScanCorpus/ScanPostToolCorpus).
package codexlifecycle

import (
	"os"
	"sort"
	"time"
)

// ReconcileSchema identifies the report shape.
const ReconcileSchema = "fak-codex-reconcile/1"

// ReconcileCounts is one aggregate row (the whole corpus, or one
// provider/version). The raw-event columns and the outcome columns are two
// views of the SAME rollouts; RawUnaccounted and ResidualUnaccounted are the
// two deltas defined above.
type ReconcileCounts struct {
	Rollouts int `json:"rollouts"`

	// Raw event counts, from the records themselves (includes orphan events:
	// a truncated head can carry a task_complete whose start is in an earlier
	// file, so terminals may legitimately exceed starts and the delta below
	// may go negative — that honesty is the point).
	TaskStarted  int `json:"task_started"`
	TaskComplete int `json:"task_complete"`
	TurnAborted  int `json:"turn_aborted"`

	// RawUnaccounted = TaskStarted − TaskComplete − TurnAborted: starts with
	// no OBSERVED terminal record. The audited 498−288−25 = 185 shape. Never
	// clamped; a healthy run reports 0.
	RawUnaccounted    int     `json:"raw_unaccounted"`
	RawUnaccountedPct float64 `json:"raw_unaccounted_pct"` // of TaskStarted

	// Post-synthesis outcomes from the fold, observed and synthesized in
	// separate columns so a consumer never conflates read evidence with
	// inference. Live is the open tail: it is NOT a terminal.
	Complete     int `json:"complete"`
	Aborted      int `json:"aborted"`
	Superseded   int `json:"superseded"`    // synthesized
	ProcessDeath int `json:"process_death"` // synthesized
	Live         int `json:"live"`          // synthesized, open

	// ClosedTotal = Complete+Aborted+Superseded+ProcessDeath.
	ClosedTotal int `json:"closed_total"`
	// ResidualUnaccounted = TaskStarted − ClosedTotal: what synthesis could
	// not close (exactly the Live count on a healthy fold). ResidualPct is of
	// TaskStarted.
	ResidualUnaccounted int     `json:"residual_unaccounted"`
	ResidualPct         float64 `json:"residual_pct"`

	// Fold integrity classes, carried through from Report.
	UnclassifiedAfter  int `json:"unclassified_after"`
	Orphans            int `json:"orphans,omitempty"`
	Reused             int `json:"reused,omitempty"`
	MultiplyTerminated int `json:"multiply_terminated,omitempty"`
}

// ReconcileReport is the whole-corpus reconciliation report.
type ReconcileReport struct {
	Schema         string                      `json:"schema"`
	Root           string                      `json:"root"`
	Scanned        int                         `json:"scanned"`
	Unreadable     int                         `json:"unreadable,omitempty"`
	Totals         ReconcileCounts             `json:"totals"`
	ByProvider     map[string]*ReconcileCounts `json:"by_provider_version"`
	AllStartsTyped bool                        `json:"all_starts_typed"`
}

// ProviderVersions returns the table's row keys in a stable order.
func (r ReconcileReport) ProviderVersions() []string {
	keys := make([]string, 0, len(r.ByProvider))
	for k := range r.ByProvider {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func (c *ReconcileCounts) fold(raw map[string]int, rep Report) {
	c.Rollouts++
	c.TaskStarted += raw[KindStarted]
	c.TaskComplete += raw[KindComplete]
	c.TurnAborted += raw[KindAborted]
	c.Orphans += len(rep.Orphans)
	c.Reused += len(rep.Reused)
	c.MultiplyTerminated += len(rep.MultiplyTerminated)
	c.UnclassifiedAfter += len(rep.Unclassified())
	oc := rep.CountByOutcome()
	c.Complete += oc[Complete]
	c.Aborted += oc[Aborted]
	c.Superseded += oc[Superseded]
	c.ProcessDeath += oc[ProcessDeath]
	c.Live += oc[Live]
}

// finalize derives the two deltas. Kept separate from fold so the totals row
// derives from aggregate columns (never from a second, divergent pass).
func (c *ReconcileCounts) finalize() {
	c.RawUnaccounted = c.TaskStarted - c.TaskComplete - c.TurnAborted
	c.ClosedTotal = c.Complete + c.Aborted + c.Superseded + c.ProcessDeath
	c.ResidualUnaccounted = c.TaskStarted - c.ClosedTotal
	if c.TaskStarted > 0 {
		c.RawUnaccountedPct = 100 * float64(c.RawUnaccounted) / float64(c.TaskStarted)
		c.ResidualPct = 100 * float64(c.ResidualUnaccounted) / float64(c.TaskStarted)
	}
}

// ScanReconcileCorpus folds every rollout under root through the reconciler
// and reports the raw and post-synthesis accounting. Durability and scoping
// match ScanCorpus: unreadable rollouts are counted, never fatal; opt.CWD
// scopes to one repository's sessions; opt.FreshWithin/Now decide Live vs
// ProcessDeath exactly as there.
func ScanReconcileCorpus(root string, opt ScanOptions) (ReconcileReport, error) {
	now := opt.Now
	if now.IsZero() {
		now = time.Now()
	}
	rep := ReconcileReport{Schema: ReconcileSchema, Root: root, ByProvider: map[string]*ReconcileCounts{}}

	paths, err := rolloutPaths(root, opt.Limit)
	if err != nil {
		return rep, err
	}
	for _, p := range paths {
		info, statErr := os.Stat(p)
		if statErr != nil {
			rep.Unreadable++
			continue
		}
		fh, openErr := os.Open(p)
		if openErr != nil {
			rep.Unreadable++
			continue
		}
		meta, events, parseErr := ReadRollout(fh)
		_ = fh.Close()
		if parseErr != nil {
			rep.Unreadable++
			continue
		}
		if opt.CWD != "" && !sameDir(meta.CWD, opt.CWD) {
			continue
		}
		raw := make(map[string]int, 3)
		for _, ev := range events {
			raw[ev.Kind]++
		}
		fresh := opt.FreshWithin > 0 && now.Sub(info.ModTime()) <= opt.FreshWithin
		folded := Fold(events, fresh)
		if len(folded.Tasks) == 0 && len(folded.Orphans) == 0 {
			continue // not a task-bearing rollout
		}
		rep.Scanned++
		rep.Totals.fold(raw, folded)
		key := meta.ProviderVersion()
		row := rep.ByProvider[key]
		if row == nil {
			row = &ReconcileCounts{}
			rep.ByProvider[key] = row
		}
		row.fold(raw, folded)
	}
	rep.Totals.finalize()
	for _, row := range rep.ByProvider {
		row.finalize()
	}
	rep.AllStartsTyped = rep.Totals.UnclassifiedAfter == 0
	return rep, nil
}
