package contextq

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/cdb"
)

// The workflow-memory benchmark (acceptance #5 of issue #437). Where the other
// contextq pieces each prove ONE property in isolation — IndexViews emits the
// multi-view set, Query reaches all five verdict kinds, kvview scopes a KV
// materialization — this benchmark drives a whole multi-turn agentic workload
// through the derived-view path and reports the system-level numbers the memo
// names: resident bytes, page/view fault rate, source coverage, stale reuse,
// poison leakage, and task success.
//
// It is not a microbenchmark of wall-clock; it is a TRUST benchmark. The two
// numbers that must stay zero — StaleReuse and PoisonLeakage — are the
// fail-closed witnesses: a stale view is never served as a HIT (it RECOMPUTEs),
// and no sealed/tombstoned source page ever enters a materialized view. They are
// measured from the actual verdicts and the image's own page table, not asserted.
//
// The workload is a deterministic, model-free sequence of turns over one
// attached recall image, sharing a single ViewCache so the cache's economics
// (cold build -> warm reuse -> policy-drift recompute) are exercised end to end.

// BenchRequest configures a workflow-memory benchmark run.
type BenchRequest struct {
	// GoalQuery is the answer-bearing request the agent repeats across turns
	// (e.g. "refund fee account"). TaskSuccess is whether it materialized at
	// least one benign view over the run.
	GoalQuery string `json:"goal_query"`
	// PoisonQuery deliberately targets a sealed/poisoned page's descriptor. A
	// correct system REFUSES it and materializes nothing; PoisonLeakage is the
	// count of times it (or any turn) leaked a quarantined page into a view.
	PoisonQuery string `json:"poison_query"`
	// Producer stamps every emitted view (default "contextq-bench").
	Producer string `json:"producer,omitempty"`
	// PolicyA and PolicyB are the two freshness epochs. The workload runs
	// cold(A) -> warm(A) -> drift(B) -> warm(B) so the cache crosses a policy
	// boundary; a view built under A and reused under B must RECOMPUTE, not HIT.
	PolicyA string `json:"policy_a,omitempty"`
	PolicyB string `json:"policy_b,omitempty"`
}

// WorkflowMemoryReport is the system-level summary acceptance #5 names. Every
// field is derived from real per-turn Results and the image page table.
type WorkflowMemoryReport struct {
	Turns int `json:"turns"`

	// ResidentBytes is the demand-pageable universe — benign distinct bytes the
	// workload could fault. It is the denominator the rest is judged against.
	ResidentBytes int64 `json:"resident_bytes"`
	BytesPagedIn  int64 `json:"bytes_paged_in"` // raw bytes actually faulted across the run

	// Materialization mix across every turn.
	Materializations int `json:"materializations"` // HIT + FAULT + RECOMPUTE
	Faults           int `json:"faults"`           // FAULT + RECOMPUTE (a raw page was paged in)
	Hits             int `json:"hits"`             // HIT (served from the view cache, zero raw fault)
	Recomputes       int `json:"recomputes"`       // stale views rebuilt under the new policy

	// FaultRate is Faults / Materializations in [0,1]. A warm, fresh workload
	// drives this toward 0; a cold one toward 1.
	FaultRate float64 `json:"fault_rate"`

	// SourceCoverage is the fraction of benign source pages represented by at
	// least one emitted view over the run.
	BenignPages    int     `json:"benign_pages"`
	PagesCovered   int     `json:"pages_covered"`
	SourceCoverage float64 `json:"source_coverage"`

	// StaleReuse MUST be 0: a cached view whose policy epoch differs from the
	// serving request was served as a HIT instead of RECOMPUTE. Measured by
	// comparing each HIT's served view PolicyVersion to its turn's policy.
	StaleReuse int `json:"stale_reuse"`

	// PoisonLeakage MUST be 0: a sealed/tombstoned source page entered a
	// materialized view, OR a turn's working set flagged poison resident.
	PoisonLeakage int `json:"poison_leakage"`

	SealedRefused     int `json:"sealed_refused"`
	TombstonedSkipped int `json:"tombstoned_skipped"`

	// TaskSuccess is whether the goal query surfaced at least one benign view
	// over the run — the workload could actually answer.
	TaskSuccess bool `json:"task_success"`
}

// RunWorkflowMemoryBench drives the multi-turn workload over an attached image
// and returns the aggregated report. It pages nothing the trust gate would
// refuse; every metric is read back from the per-turn Results and the page
// table, so the report cannot claim a property the substrate did not enforce.
func RunWorkflowMemoryBench(ctx context.Context, im *cdb.Image, req BenchRequest) WorkflowMemoryReport {
	if req.Producer == "" {
		req.Producer = "contextq-bench"
	}
	if req.PolicyA == "" {
		req.PolicyA = "wfmem-p1"
	}
	if req.PolicyB == "" {
		req.PolicyB = "wfmem-p2"
	}

	// Page table: map every source step to whether it is a quarantined page, so
	// a materialized view over a sealed/tombstoned step is detectable as leakage.
	quarantined := map[int]bool{}
	for _, f := range im.Backtrace() {
		if f.Sealed || f.Tombstoned {
			quarantined[f.Step] = true
		}
	}

	info := im.Info()
	rep := WorkflowMemoryReport{
		ResidentBytes: info.ResidentBytes,
		BenignPages:   info.Benign,
	}

	cache := NewViewCache()
	covered := map[int]bool{}

	// turn is one materialization pass; policy is its freshness epoch.
	type turn struct {
		query  string
		policy string
	}
	turns := []turn{
		{req.GoalQuery, req.PolicyA},   // cold build under A
		{req.GoalQuery, req.PolicyA},   // warm reuse under A -> HITs
		{req.GoalQuery, req.PolicyB},   // policy drift -> RECOMPUTE (fail-closed staleness)
		{req.GoalQuery, req.PolicyB},   // warm reuse under B -> HITs
		{req.PoisonQuery, req.PolicyB}, // poison probe -> REFUSE, no view
	}

	for _, tn := range turns {
		res := Query(ctx, im, Request{
			Query:         tn.query,
			PreferView:    ViewSummary,
			ViewCache:     cache,
			PolicyVersion: tn.policy,
			Producer:      req.Producer,
		})
		rep.Turns++
		rep.BytesPagedIn += res.Stats.BytesPagedIn
		if res.Stats.TombstonedSkipped > rep.TombstonedSkipped {
			rep.TombstonedSkipped = res.Stats.TombstonedSkipped
		}

		// A working set that ever flagged poison resident is leakage, full stop.
		if res.Stats.PoisonInSet {
			rep.PoisonLeakage++
		}

		// Index served views by id so a HIT verdict can be checked for staleness.
		viewByID := make(map[string]MemoryViewRecord, len(res.Views))
		for _, v := range res.Views {
			viewByID[v.ViewID] = v
		}

		for _, vd := range res.Verdicts {
			switch vd.Kind {
			case MaterializationHit:
				rep.Materializations++
				rep.Hits++
				// Fail-closed witness: a HIT served under a policy epoch the view
				// was not built for is a stale reuse.
				if v, ok := viewByID[vd.ViewID]; ok && v.PolicyVersion != "" && v.PolicyVersion != tn.policy {
					rep.StaleReuse++
				}
			case MaterializationFault:
				rep.Materializations++
				rep.Faults++
			case MaterializationRecompute:
				rep.Materializations++
				rep.Faults++
				rep.Recomputes++
			case MaterializationRefuse:
				rep.SealedRefused++
			}
		}

		// Coverage + leakage from the materialized slices themselves.
		for _, sl := range res.Slices {
			if quarantined[sl.Step] {
				rep.PoisonLeakage++ // a quarantined page reached a view
				continue
			}
			covered[sl.Step] = true
			if tn.query == req.GoalQuery {
				rep.TaskSuccess = true
			}
		}
	}

	rep.PagesCovered = len(covered)
	if rep.BenignPages > 0 {
		rep.SourceCoverage = float64(rep.PagesCovered) / float64(rep.BenignPages)
	}
	if rep.Materializations > 0 {
		rep.FaultRate = float64(rep.Faults) / float64(rep.Materializations)
	}
	return rep
}
