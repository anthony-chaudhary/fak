package flowmetrics

import (
	"fmt"
	"strings"
	"time"
)

// The KPI fold. Every threshold in this file is a FIXED constant, per the
// repo's scorecard anti-gaming law: a defect is retired by changing reality, not
// by moving the detector. Each threshold carries the reasoning for its value so
// a later reader can argue with the number instead of guessing at it.
const (
	// FlowEfficiencyFloor is the median active/lead ratio below which the
	// backlog is dominated by waiting rather than working. 0.30 is
	// deliberately lenient — mature Kanban systems target 0.40+, and the
	// literature treats sub-0.15 as pathological — so tripping this floor
	// is unambiguous rather than aspirational.
	FlowEfficiencyFloor = 0.30

	// UnstartedShareCeiling is the share of OPEN issues that may have zero
	// referencing commits. Some unstarted backlog is healthy (it is the
	// option pool); a majority means intake has outrun capacity and the
	// backlog has stopped being a queue and become a wish list.
	UnstartedShareCeiling = 0.60

	// AgingWIPDays is when started-but-unclosed work counts as stalled. It
	// is set to 7 to sit well above this repo's measured p50 lead time
	// (~0.8d) and p90 (~9.8d): a started issue quiet for a full week is
	// outside normal cadence, not merely slow.
	AgingWIPDays = 7

	// AgingWIPCeiling is how many stalled items are tolerable at once. It
	// mirrors the spirit of the existing focusscore WIP cap: past roughly a
	// dozen, no operator is tracking them and each one is pure carry cost.
	AgingWIPCeiling = 12

	// AtomicShareFloor is the share of closed issues that must have landed
	// in exactly one commit and one lane, per AGENTS.md "one issue, one
	// commit; one commit, one leaf". Set at 0.50 because multi-commit
	// landings are legitimate sometimes (a revert, a follow-up fix), but
	// they should not be the majority shape.
	AtomicShareFloor = 0.50

	// ArrivalServiceRatioCeiling is arrivals divided by closes over the
	// window. Above 1.0 the backlog grows without bound by Little's Law; the
	// 1.10 ceiling allows normal burstiness before calling it a defect.
	ArrivalServiceRatioCeiling = 1.10

	// WitnessedProgressFloor is the share of aggregate (epic) issues whose
	// completion is witnessed by child-issue closes rather than
	// self-reported checkboxes. Below this, epic "progress" is narration.
	WitnessedProgressFloor = 0.50

	// UnmeasurableLimit caps the unmeasurable aggregate issues named in
	// the defect detail.
	UnmeasurableLimit = 20
)

// KPI is one graded criterion, matching the field shape the repo's other
// scorecards emit (kpi/group/score/detail/defects/soft) so this payload drops
// into the existing control pane without a new reader.
type KPI struct {
	KPI     string   `json:"kpi"`
	Group   string   `json:"group"`
	Score   int      `json:"score"`
	Value   float64  `json:"value"`
	Detail  string   `json:"detail"`
	Defects []string `json:"defects"`
	Soft    []string `json:"soft"`
}

// Report is the emitted envelope. Corpus carries the flat headline numbers the
// control pane reads (`flow_debt`, `grade`); KPIs carry the per-axis detail.
type Report struct {
	Schema     string         `json:"schema"`
	OK         bool           `json:"ok"`
	Verdict    string         `json:"verdict"`
	Finding    string         `json:"finding"`
	Reason     string         `json:"reason"`
	NextAction string         `json:"next_action"`
	Workspace  string         `json:"workspace,omitempty"`
	Corpus     map[string]any `json:"corpus"`
	KPIs       []KPI          `json:"kpis"`
	// Tree is the structured working-tree census behind local_wip, including
	// bounded recent paths, planned overlaps, and duplicate symbols.
	Tree TreeWIP `json:"tree_wip"`

	// Aging is the actionable list: started work nobody finished, oldest
	// first, holding the same rows the aging_wip KPI counts. It is truncated
	// by Input.AgingLimit so the payload stays bounded.
	Aging []Span `json:"aging_wip,omitempty"`
	// AgingTotal is how many rows existed BEFORE that truncation. Without it
	// a bounded list reports its own limit as the size of the problem — a
	// 25-row cap would make 86 rotting issues read as 25.
	AgingTotal int `json:"aging_wip_total,omitempty"`
	// Curve is the WIP-over-time series when a window was requested.
	Curve []DayWIP `json:"wip_curve,omitempty"`

	// Epics holds the per-aggregate progress information.
	Epics []EpicProgress `json:"epics,omitempty"`
	// Unmeasurable is the actionable list: unmeasurable aggregate issues, worst first,
	// capped at UnmeasurableLimit.
	Unmeasurable []int `json:"unmeasurable_epics,omitempty"`
	// UnmeasurableTotal is how many unmeasurable aggregates existed before truncation.
	UnmeasurableTotal int `json:"unmeasurable_total,omitempty"`
}

// Input is everything the fold needs. Keeping it a struct means adding a future
// signal never breaks callers.
type Input struct {
	Issues  []Issue
	Commits []Commit
	// Tree is the local uncommitted-WIP census; the zero value means "not
	// gathered" and its KPI is then reported as unmeasured rather than
	// clean, so a missing gather can never look like a green tree.
	Tree TreeWIP
	// AboutToTouch names repository-relative paths the current session plans to
	// edit, so the readout can distinguish a real overlap from general churn.
	AboutToTouch []string
	// Now anchors every age computation. Required: a zero Now would make
	// every open issue look brand new.
	Now time.Time
	// WindowDays bounds the closed-issue cohort used for the rate and
	// efficiency KPIs, and the length of the emitted WIP curve.
	WindowDays int
	// AgingLimit caps the emitted Aging list; <=0 means 25.
	AgingLimit int
	Workspace  string
}

// Build folds the input into the report.
func Build(in Input) Report {
	if in.WindowDays <= 0 {
		in.WindowDays = 30
	}
	if in.AgingLimit <= 0 {
		in.AgingLimit = 25
	}
	spans := BuildSpans(in.Issues, in.Commits)
	tree := withRecentWriterOverlap(in.Tree, in.AboutToTouch)
	byNumber := make(map[int]Issue, len(in.Issues))
	for _, i := range in.Issues {
		byNumber[i.Number] = i
	}
	since := in.Now.Add(-time.Duration(in.WindowDays) * 24 * time.Hour)

	// The stalled set is folded ONCE and handed to both the KPI and the
	// emitted list, so the graded number and the actionable rows are the same
	// set by construction rather than by two agreeing filters.
	stalled := stalledWIP(spans, in.Now)
	aging := stalled
	if in.AgingLimit > 0 && len(aging) > in.AgingLimit {
		aging = aging[:in.AgingLimit]
	}

	epics := EpicsProgress(in.Issues, byNumber)
	unmeasurableAll := UnmeasurableAggregates(epics, 0)
	unmeasurable := unmeasurableAll
	if len(unmeasurable) > UnmeasurableLimit {
		unmeasurable = unmeasurable[:UnmeasurableLimit]
	}
	var unmeasurableNums []int
	for _, u := range unmeasurable {
		unmeasurableNums = append(unmeasurableNums, u.Issue)
	}

	var kpis []KPI
	kpis = append(kpis,
		kpiFlowEfficiency(spans, since),
		kpiQueueTime(spans, since),
		kpiUnstartedBacklog(spans),
		kpiAgingWIP(stalled, in.Now),
		kpiAtomicity(spans, since),
		kpiArrivalVsService(spans, since, in.Now),
		kpiWitnessedProgress(in.Issues, byNumber),
		kpiLocalWIP(tree),
	)

	debt := 0
	for _, k := range kpis {
		debt += len(k.Defects)
	}
	rep := Report{
		Schema:            Schema,
		Workspace:         in.Workspace,
		KPIs:              kpis,
		Tree:              tree,
		Aging:             aging,
		AgingTotal:        len(stalled),
		Curve:             WIPCurve(spans, since, in.Now),
		Epics:             epics,
		Unmeasurable:      unmeasurableNums,
		UnmeasurableTotal: len(unmeasurableAll),
		Corpus: map[string]any{
			"flow_debt":   debt,
			"grade":       GradeLetter(debt),
			"issues":      len(in.Issues),
			"commits":     len(in.Commits),
			"window_days": in.WindowDays,
			"spans":       len(spans),
		},
	}
	rep.OK = debt == 0
	if rep.OK {
		rep.Verdict, rep.Finding = "OK", "flow_healthy"
		rep.Reason = "every flow axis is inside its fixed threshold"
		rep.NextAction = "none"
		return rep
	}
	rep.Verdict, rep.Finding = "ACTION", "flow_debt"
	worst := kpis[0]
	for _, k := range kpis {
		if len(k.Defects) > 0 && k.Score < worst.Score {
			worst = k
		}
	}
	rep.Reason = fmt.Sprintf("%d flow defect(s); weakest axis %s at %d/100", debt, worst.KPI, worst.Score)
	if len(worst.Defects) > 0 {
		rep.NextAction = worst.Defects[0]
	} else {
		rep.NextAction = "review the defect list"
	}
	return rep
}

// GradeLetter maps flow debt to the letter band the control pane renders,
// matching the convention used by the repo's other scorecards.
func GradeLetter(debt int) string {
	switch {
	case debt == 0:
		return "A"
	case debt <= 2:
		return "B"
	case debt <= 4:
		return "C"
	case debt <= 6:
		return "D"
	default:
		return "F"
	}
}

// score01 converts a ratio in [0,1] to a 0-100 integer score.
func score01(v float64) int {
	if v < 0 {
		return 0
	}
	if v > 1 {
		v = 1
	}
	return int(v*100 + 0.5)
}

// closedSince returns the spans closed inside the window that also have a start,
// i.e. the cohort for which the full four-term decomposition is defined.
func closedSince(spans []Span, since time.Time) []Span {
	var out []Span
	for _, s := range spans {
		if s.Closed() && s.Started() && !s.ClosedAt.Before(since) {
			out = append(out, s)
		}
	}
	return out
}

func kpiFlowEfficiency(spans []Span, since time.Time) KPI {
	k := KPI{KPI: "flow_efficiency", Group: "time_in_state", Defects: []string{}, Soft: []string{}}
	cohort := closedSince(spans, since)
	if len(cohort) == 0 {
		k.Score, k.Value = 0, -1
		k.Detail = "no issue closed in the window had a referencing commit, so active time is unmeasurable"
		k.Soft = append(k.Soft, "flow_efficiency: unmeasured — no closed span carried a commit reference")
		return k
	}
	var effs []float64
	for _, s := range cohort {
		effs = append(effs, s.FlowEfficiency)
	}
	med := Percentile(effs, 50)
	k.Value = med
	k.Score = score01(med)
	k.Detail = fmt.Sprintf("median flow efficiency %.0f%% over %d closed issues (p90 %.0f%%)",
		med*100, len(cohort), Percentile(effs, 90)*100)
	if med < FlowEfficiencyFloor {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"flow_efficiency: median %.0f%% is below the %.0f%% floor — work spends most of its life queued, not worked; cut intake or start fewer items",
			med*100, FlowEfficiencyFloor*100))
	}
	return k
}

func kpiQueueTime(spans []Span, since time.Time) KPI {
	k := KPI{KPI: "queue_time", Group: "time_in_state", Defects: []string{}, Soft: []string{}}
	cohort := closedSince(spans, since)
	if len(cohort) == 0 {
		k.Score, k.Value = 0, -1
		k.Detail = "no closed span with a start in the window"
		k.Soft = append(k.Soft, "queue_time: unmeasured")
		return k
	}
	var q, a []float64
	for _, s := range cohort {
		q = append(q, s.QueueHours)
		a = append(a, s.ActiveHours)
	}
	qp50, ap50 := Percentile(q, 50), Percentile(a, 50)
	k.Value = qp50
	// Reported as a ratio so the score is comparable across repos: queue
	// time only matters relative to how long the work itself takes.
	if qp50+ap50 > 0 {
		k.Score = score01(ap50 / (qp50 + ap50))
	}
	k.Detail = fmt.Sprintf("p50 queue %.1fh vs p50 active %.1fh (p90 queue %.1fh)",
		qp50, ap50, Percentile(q, 90))
	// No hard defect: queue time is already gated via flow_efficiency, and
	// double-charging one reality as two defects would inflate the debt.
	if qp50 > ap50*3 && qp50 > 24 {
		k.Soft = append(k.Soft, fmt.Sprintf(
			"queue_time: median wait %.1fh is over 3x median work %.1fh", qp50, ap50))
	}
	return k
}

func kpiUnstartedBacklog(spans []Span) KPI {
	k := KPI{KPI: "unstarted_backlog", Group: "intake", Defects: []string{}, Soft: []string{}}
	open, unstarted := 0, 0
	for _, s := range spans {
		if s.Closed() {
			continue
		}
		open++
		if !s.Started() {
			unstarted++
		}
	}
	if open == 0 {
		k.Score, k.Value = 100, 0
		k.Detail = "no open issues"
		return k
	}
	share := float64(unstarted) / float64(open)
	k.Value = share
	k.Score = score01(1 - share)
	k.Detail = fmt.Sprintf("%d of %d open issues (%.0f%%) have no referencing commit", unstarted, open, share*100)
	if share > UnstartedShareCeiling {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"unstarted_backlog: %.0f%% of open issues were never touched by a commit, over the %.0f%% ceiling — intake is outrunning capacity; stop filing or start closing as no-longer-wanted",
			share*100, UnstartedShareCeiling*100))
	}
	return k
}

// kpiAgingWIP grades the ALREADY-FILTERED stalled set from stalledWIP, rather
// than re-filtering the spans itself. Taking the rows means the graded count and
// the rows the readout prints are the same slice, so the two can never disagree.
func kpiAgingWIP(rows []Span, now time.Time) KPI {
	k := KPI{KPI: "aging_wip", Group: "wip", Defects: []string{}, Soft: []string{}}
	stalled := len(rows)
	var oldest float64
	if stalled > 0 {
		// The rows arrive oldest-first, so the head is the maximum age.
		oldest = rows[0].AgeHours(now)
	}
	k.Value = float64(stalled)
	if stalled <= AgingWIPCeiling {
		k.Score = 100
	} else {
		// Degrade linearly to 0 at four times the ceiling, so the score
		// keeps discriminating instead of saturating immediately.
		k.Score = score01(1 - float64(stalled-AgingWIPCeiling)/float64(AgingWIPCeiling*3))
	}
	k.Detail = fmt.Sprintf("%d issues started but unclosed for over %dd (oldest %.0fd)",
		stalled, AgingWIPDays, oldest/24)
	if stalled > AgingWIPCeiling {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"aging_wip: %d issues have commits but no close after %dd, over the cap of %d — finish these before starting anything new; they are the cheapest throughput available",
			stalled, AgingWIPDays, AgingWIPCeiling))
	}
	return k
}

func kpiAtomicity(spans []Span, since time.Time) KPI {
	k := KPI{KPI: "atomicity", Group: "shape", Defects: []string{}, Soft: []string{}}
	cohort := closedSince(spans, since)
	if len(cohort) == 0 {
		k.Score, k.Value = 0, -1
		k.Detail = "no closed span with commits in the window"
		k.Soft = append(k.Soft, "atomicity: unmeasured")
		return k
	}
	atomic, multiLane := 0, 0
	for _, s := range cohort {
		if s.Atomic() {
			atomic++
		}
		if len(s.Leaves) > 1 {
			multiLane++
		}
	}
	share := float64(atomic) / float64(len(cohort))
	k.Value = share
	k.Score = score01(share)
	k.Detail = fmt.Sprintf("%d of %d closed issues (%.0f%%) landed as one commit in one lane; %d spanned multiple lanes",
		atomic, len(cohort), share*100, multiLane)
	if share < AtomicShareFloor {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"atomicity: only %.0f%% of closed issues landed as a single commit, below the %.0f%% floor — issues are being filed larger than one landable change; decompose before dispatch",
			share*100, AtomicShareFloor*100))
	}
	return k
}

func kpiArrivalVsService(spans []Span, since, now time.Time) KPI {
	k := KPI{KPI: "arrival_vs_service", Group: "intake", Defects: []string{}, Soft: []string{}}
	m := MeasureArrivalService(spans, since, now)
	if m.Closed == 0 {
		k.Score, k.Value = 0, -1
		k.Detail = fmt.Sprintf("%d opened and 0 closed over %.0fd", m.Opened, m.WindowDays)
		if m.Opened > 0 {
			k.Defects = append(k.Defects, fmt.Sprintf(
				"arrival_vs_service: %d issues arrived and none closed in %.0fd — the queue is write-only", m.Opened, m.WindowDays))
		}
		return k
	}
	ratio := *m.Ratio
	k.Value = ratio
	k.Score = score01(1 / ratio)
	k.Detail = fmt.Sprintf("%.1f arrivals/day vs %.1f closes/day over %.0fd (ratio %.2f, net %+d)",
		m.ArrivalRate, m.ServiceRate, m.WindowDays, ratio, m.Opened-m.Closed)
	if ratio > ArrivalServiceRatioCeiling {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"arrival_vs_service: arrivals exceed closes by %.0f%% (net %+d over %.0fd) — by Little's Law the backlog and its lead time grow without bound; cap intake",
			(ratio-1)*100, m.Opened-m.Closed, m.WindowDays))
	}
	return k
}

func kpiWitnessedProgress(issues []Issue, byNumber map[int]Issue) KPI {
	k := KPI{KPI: "witnessed_progress", Group: "shape", Defects: []string{}, Soft: []string{}}
	epics := EpicsProgress(issues, byNumber)
	if len(epics) == 0 {
		k.Score, k.Value = 100, 1
		k.Detail = "no open aggregate issues"
		return k
	}

	witnessed, selfReported := 0, 0
	for _, ep := range epics {
		switch ep.Basis {
		case "children":
			witnessed++
		case "checkbox":
			selfReported++
		}
		k.Soft = append(k.Soft, fmt.Sprintf("witnessed_progress: %s", ep.Summary()))
	}

	aggregates := len(epics)
	share := float64(witnessed) / float64(aggregates)
	k.Value = share
	k.Score = score01(share)

	unmeasurableAll := UnmeasurableAggregates(epics, 0)
	var names []string
	for _, u := range unmeasurableAll {
		names = append(names, fmt.Sprintf("#%d", u.Issue))
	}

	namedList := ""
	if len(names) > 0 {
		capped := names
		more := 0
		if len(capped) > UnmeasurableLimit {
			more = len(capped) - UnmeasurableLimit
			capped = capped[:UnmeasurableLimit]
		}
		namedList = strings.Join(capped, ", ")
		if more > 0 {
			namedList += fmt.Sprintf(" (and %d more)", more)
		}
	}

	if len(unmeasurableAll) == 0 {
		k.Detail = fmt.Sprintf("%d of %d open aggregate issues (%.0f%%) report progress via child issues; %d rely on unwitnessed checkboxes",
			witnessed, aggregates, share*100, selfReported)
	} else {
		k.Detail = fmt.Sprintf("%d of %d open aggregate issues (%.0f%%) report progress via child issues; %d unmeasurable (%s)",
			witnessed, aggregates, share*100, len(unmeasurableAll), namedList)
	}

	if share < WitnessedProgressFloor {
		k.Defects = append(k.Defects, fmt.Sprintf(
			"witnessed_progress: only %.0f%% of open epics track progress through child issues, below the %.0f%% floor — unmeasurable epics (%s) report completion as checkboxes nobody ticks or lack child issues, so percent-complete is unmeasurable; convert task-list lines into real child issues",
			share*100, WitnessedProgressFloor*100, namedList))
	}
	return k
}
