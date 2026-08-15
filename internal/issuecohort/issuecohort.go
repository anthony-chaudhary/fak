// Package issuecohort plans a whole BATCH of machine-created GitHub issue
// candidates at creation time, before any of them is synced to GitHub.
//
// internal/issuecontract reviews one candidate at a time: is THIS issue a
// scoped, routeable, witnessed dispatch leaf? That is necessary but not
// sufficient when an agent emits many issues in a single run (1..1000). A batch
// of individually-perfect issues can still be a bad cohort: two of them may
// route to overlapping file trees, so dispatching them concurrently would put
// two workers in the same files — exactly the collision `dos arbitrate` refuses
// at wave-launch time. Discovering that only at dispatch time is late; the fix
// is to see the collision structure of the batch when it is CREATED.
//
// issuecohort folds a []issuepolicy.Candidate into a Plan that:
//
//   - reviews each candidate through the shared contract;
//   - partitions the dispatchable leaves into concurrency-safe WAVES using the
//     same disjoint-tree rule the dispatch arbiter uses (first-fit graph
//     colouring over lane+path overlap), so each wave is safe to launch at once
//     and the number of waves is the number of sequential rounds the batch needs;
//   - pulls oversized / non-leaf rows into a SUBDIVIDE queue with a child-issue
//     budget, so "this batch is 1000 issues" does not hide "40 of them are
//     really epics that must be split first";
//   - buckets the rest into TRIAGE by dispatchability;
//   - reports DUPLICATE keys, so a rerun that would create instead of update is
//     visible as cleanup, not silent spam.
//
// The package is pure (stdlib + issuecontract only): no gh, no disk, no clock.
// The overlap semantics mirror internal/dispatchtick's duplicatePathOverlaps so
// a cohort deconflicted here is deconflicted the same way the dispatcher will
// re-check it later.
package issuecohort

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// Schema is the stable schema tag stamped on the machine-readable plan.
const Schema = "fak.issue-cohort-plan.v1"

// Options carries the batch-level knobs plus the per-candidate contract options
// (Live / DedupeChecked / DedupeCap) forwarded to issuepolicy.
type Options struct {
	issuepolicy.Options

	// MaxWave caps how many leaves a single concurrency-safe wave may hold, e.g.
	// the serving-seat pool or an operator concurrency ceiling. 0 means the wave
	// is bounded only by the disjoint-tree rule (the natural peak concurrency).
	MaxWave int
}

// WaveMember is one dispatchable leaf placed in a wave.
type WaveMember struct {
	Key string `json:"key"`
	// IssueNumber is the member's live GitHub issue number when the cohort was
	// planned over EXISTING issues (`fak issue cohort --from-issues`), and 0 when
	// it was planned over not-yet-filed candidates. It is carried, not derived:
	// it is the review's own identity field, and it is what lets a downstream
	// reader bind a landed commit's `#N` back to the wave the work was decided in
	// (internal/steerpr's wave grouping, #5040) without inventing a second key.
	IssueNumber   int                      `json:"issue_number,omitempty"`
	Title         string                   `json:"title,omitempty"`
	Lane          string                   `json:"lane,omitempty"`
	Paths         []string                 `json:"paths,omitempty"`
	ExpectedSteps int                      `json:"expected_steps,omitempty"`
	ProblemFrame  issuepolicy.ProblemFrame `json:"problem_frame"`
}

// Wave is one set of leaves whose expected trees are mutually disjoint, so all
// of its members can be dispatched concurrently without a file-tree collision.
type Wave struct {
	Index      int          `json:"index"`
	Size       int          `json:"size"`
	StepBudget int          `json:"step_budget"`
	Members    []WaveMember `json:"members"`
	// LeaseRegion is the minimal normalized union of member path roots — the
	// exact tree set to hand `dos arbitrate --tree` for the whole wave, so a
	// concurrency-safe wave maps to one lease call. It covers every path-scoped
	// member; lane-only members are covered by LeaseLanes instead.
	LeaseRegion []string `json:"lease_region,omitempty"`
	// LeaseLanes are the lanes of members that named no explicit paths (they take
	// their whole lane); pass each as `dos arbitrate --lane`.
	LeaseLanes []string `json:"lease_lanes,omitempty"`
}

// SubdivideRow is a candidate that is routeable but declared (or detected) as an
// epic / non-leaf / oversized row: it must be decomposed before dispatch.
type SubdivideRow struct {
	Key              string   `json:"key"`
	Title            string   `json:"title,omitempty"`
	Reasons          []string `json:"reasons"`
	ExpectedSteps    int      `json:"expected_steps,omitempty"`
	ChildIssueBudget int      `json:"child_issue_budget"`
	Lane             string   `json:"lane,omitempty"`
	Paths            []string `json:"paths,omitempty"`
}

// TriageRow is a candidate that is not yet dispatchable and is not an obvious
// split target: it needs scope, routing, or private-boundary repair first.
type TriageRow struct {
	Key             string   `json:"key"`
	Title           string   `json:"title,omitempty"`
	Dispatchability string   `json:"dispatchability"`
	Reasons         []string `json:"reasons,omitempty"`
	MissingFields   []string `json:"missing_fields,omitempty"`
}

// DuplicateGroup is one marker key that appears more than once in the batch. The
// first occurrence is planned; the rest are the rerun-should-have-updated spam
// this catches before it reaches GitHub.
type DuplicateGroup struct {
	Key   string `json:"key"`
	Count int    `json:"count"`
}

// PortfolioRow keeps centrality visible without folding it into a priority score.
type PortfolioRow struct {
	IssueNumber       int                         `json:"issue_number,omitempty"`
	Key               string                      `json:"key"`
	Title             string                      `json:"title,omitempty"`
	Dispatchability   string                      `json:"dispatchability"`
	Priority          string                      `json:"priority,omitempty"`
	SpinePriority     issuepolicy.SpinePriority   `json:"spine_priority"`
	Dependencies      []issuepolicy.DependencyRef `json:"dependencies,omitempty"`
	RiskBoundaryNotes []string                    `json:"risk_boundary_notes,omitempty"`
	Reversibility     string                      `json:"reversibility,omitempty"`
	ExpectedSteps     int                         `json:"expected_steps,omitempty"`
	Centrality        string                      `json:"centrality"`
	CentralityTarget  string                      `json:"centrality_target,omitempty"`
	ProblemFrame      issuepolicy.ProblemFrame    `json:"problem_frame"`
	ProblemFrameReady bool                        `json:"problem_frame_ready"`
	SelectionNote     string                      `json:"selection_note"`
}

type CentralityCounts struct {
	Core         int `json:"core"`
	Enabling     int `json:"enabling"`
	Stewardship  int `json:"stewardship"`
	Peripheral   int `json:"peripheral"`
	Unclassified int `json:"unclassified"`
}

// Plan is the machine-readable cohort fold.
type Plan struct {
	Schema          string           `json:"schema"`
	Total           int              `json:"total"`
	Dispatchable    int              `json:"dispatchable"`
	Subdividable    int              `json:"subdividable"`
	TriageOnly      int              `json:"triage_only"`
	Refused         int              `json:"refused"`
	DuplicateKeys   int              `json:"duplicate_keys"`
	CollisionPairs  int              `json:"collision_pairs"`
	NumWaves        int              `json:"num_waves"`
	PeakConcurrency int              `json:"peak_concurrency"`
	ChildIssueTotal int              `json:"child_issue_total"`
	Waves           []Wave           `json:"waves,omitempty"`
	Subdivide       []SubdivideRow   `json:"subdivide,omitempty"`
	Triage          []TriageRow      `json:"triage,omitempty"`
	Duplicates      []DuplicateGroup `json:"duplicates,omitempty"`
	Portfolio       []PortfolioRow   `json:"portfolio"`
	Centrality      CentralityCounts `json:"centrality"`
}

// leaf is the internal projection of a dispatchable candidate used for wave
// partitioning: the original display fields plus normalized paths for overlap.
type leaf struct {
	member    WaveMember
	normPaths []string
}

// Build folds candidates into a cohort Plan. Input order is preserved for stable
// wave assignment; the derived queues are sorted for determinism.
func Build(candidates []issuepolicy.Candidate, opt Options) Plan {
	plan := Plan{Schema: Schema, Total: len(candidates)}

	keyCounts := map[string]int{}
	for _, c := range candidates {
		if key := strings.TrimSpace(c.Key); key != "" {
			keyCounts[key]++
		}
	}

	seen := map[string]bool{}
	var leaves []leaf
	for _, c := range candidates {
		key := strings.TrimSpace(c.Key)
		if key != "" {
			if seen[key] {
				continue // counted in Duplicates below; do not plan the duplicate
			}
			seen[key] = true
		}
		review := issuepolicy.ReviewCandidate(c, opt.Options)
		appendPortfolioRow(&plan, c, review)
		switch {
		case review.OK && review.Dispatchability == issuepolicy.Dispatchable:
			plan.Dispatchable++
			leaves = append(leaves, leaf{
				member: WaveMember{
					Key:           strmatch.FirstTrimmed(review.Key, key),
					IssueNumber:   review.IssueNumber,
					Title:         strings.TrimSpace(c.Title),
					Lane:          review.Lane,
					Paths:         append([]string(nil), review.Paths...),
					ExpectedSteps: review.ExpectedSteps,
					ProblemFrame:  review.ProblemFrame,
				},
				normPaths: normalizePaths(review.Paths),
			})
		case IsSplitTarget(review):
			plan.Subdividable++
			budget := childIssueBudget(review.ExpectedSteps)
			plan.ChildIssueTotal += budget
			plan.Subdivide = append(plan.Subdivide, SubdivideRow{
				Key:              strmatch.FirstTrimmed(review.Key, key),
				Title:            strings.TrimSpace(c.Title),
				Reasons:          splitReasons(review.Reasons),
				ExpectedSteps:    review.ExpectedSteps,
				ChildIssueBudget: budget,
				Lane:             review.Lane,
				Paths:            append([]string(nil), review.Paths...),
			})
		default:
			if review.Dispatchability == issuepolicy.Refused {
				plan.Refused++
			} else {
				plan.TriageOnly++
			}
			plan.Triage = append(plan.Triage, TriageRow{
				Key:             strmatch.FirstTrimmed(review.Key, key),
				Title:           strings.TrimSpace(c.Title),
				Dispatchability: review.Dispatchability,
				Reasons:         append([]string(nil), review.Reasons...),
				MissingFields:   append([]string(nil), review.MissingFields...),
			})
		}
	}

	plan.CollisionPairs = collisionPairs(leaves)
	plan.Waves = partition(leaves, opt.MaxWave)
	plan.NumWaves = len(plan.Waves)
	for _, w := range plan.Waves {
		if w.Size > plan.PeakConcurrency {
			plan.PeakConcurrency = w.Size
		}
	}

	for key, count := range keyCounts {
		if count > 1 {
			plan.DuplicateKeys += count - 1
			plan.Duplicates = append(plan.Duplicates, DuplicateGroup{Key: key, Count: count})
		}
	}
	sort.Slice(plan.Duplicates, func(i, j int) bool {
		if plan.Duplicates[i].Count != plan.Duplicates[j].Count {
			return plan.Duplicates[i].Count > plan.Duplicates[j].Count
		}
		return plan.Duplicates[i].Key < plan.Duplicates[j].Key
	})
	sort.Slice(plan.Subdivide, func(i, j int) bool {
		if plan.Subdivide[i].ChildIssueBudget != plan.Subdivide[j].ChildIssueBudget {
			return plan.Subdivide[i].ChildIssueBudget > plan.Subdivide[j].ChildIssueBudget
		}
		return plan.Subdivide[i].Key < plan.Subdivide[j].Key
	})
	sort.Slice(plan.Triage, func(i, j int) bool {
		return plan.Triage[i].Key < plan.Triage[j].Key
	})
	return plan
}

// partition places each leaf into the first existing wave it does not collide
// with (first-fit greedy graph colouring over the tree-overlap graph). A wave is
// therefore internally collision-free by construction, and the number of waves
// is the number of sequential dispatch rounds the batch needs.
func partition(leaves []leaf, maxWave int) []Wave {
	var buckets [][]leaf
	for _, lf := range leaves {
		placed := false
		for wi := range buckets {
			if maxWave > 0 && len(buckets[wi]) >= maxWave {
				continue
			}
			if !collidesWithAny(lf, buckets[wi]) {
				buckets[wi] = append(buckets[wi], lf)
				placed = true
				break
			}
		}
		if !placed {
			buckets = append(buckets, []leaf{lf})
		}
	}
	waves := make([]Wave, 0, len(buckets))
	for i, bucket := range buckets {
		wave := Wave{Index: i, Size: len(bucket)}
		var paths []string
		laneSet := map[string]bool{}
		for _, lf := range bucket {
			wave.Members = append(wave.Members, lf.member)
			wave.StepBudget += stepUnits(lf.member.ExpectedSteps)
			if len(lf.normPaths) > 0 {
				paths = append(paths, lf.normPaths...)
			} else if lf.member.Lane != "" {
				laneSet[lf.member.Lane] = true
			}
		}
		wave.LeaseRegion = minimalRoots(paths)
		wave.LeaseLanes = sortedKeys(laneSet)
		waves = append(waves, wave)
	}
	return waves
}

// minimalRoots reduces a set of normalized paths to the smallest set of tree
// roots that still covers all of them: a path is dropped when an ancestor of it
// is already a root. Sorting first guarantees ancestors precede descendants.
func minimalRoots(paths []string) []string {
	uniq := map[string]bool{}
	for _, p := range paths {
		if p != "" {
			uniq[p] = true
		}
	}
	sorted := sortedKeys(uniq)
	var roots []string
	for _, p := range sorted {
		covered := false
		for _, r := range roots {
			if p == r || strings.HasPrefix(p, r+"/") {
				covered = true
				break
			}
		}
		if !covered {
			roots = append(roots, p)
		}
	}
	return roots
}

func sortedKeys(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func appendPortfolioRow(plan *Plan, c issuepolicy.Candidate, review issuepolicy.Review) {
	centrality := review.ProblemFrame.Centrality
	if centrality == "" {
		centrality = issuepolicy.CentralityUnclassified
	}
	priority := candidatePriority(c)
	row := PortfolioRow{
		IssueNumber: review.IssueNumber, Key: strmatch.FirstTrimmed(review.Key, c.Key), Title: strings.TrimSpace(c.Title),
		Dispatchability: review.Dispatchability, Priority: priority, SpinePriority: review.SpinePriority,
		Dependencies:      append([]issuepolicy.DependencyRef(nil), review.Dependencies...),
		RiskBoundaryNotes: append([]string(nil), c.BoundaryNotes...), Reversibility: strings.TrimSpace(c.Reversibility),
		ExpectedSteps: review.ExpectedSteps,
		Centrality:    centrality, CentralityTarget: review.ProblemFrame.CentralityTarget,
		ProblemFrame: review.ProblemFrame, ProblemFrameReady: review.ProblemFrame.Ready,
		SelectionNote: selectionNote(centrality, priority, review.Dispatchability),
	}
	plan.Portfolio = append(plan.Portfolio, row)
	switch centrality {
	case issuepolicy.CentralityCore:
		plan.Centrality.Core++
	case issuepolicy.CentralityEnabling:
		plan.Centrality.Enabling++
	case issuepolicy.CentralityStewardship:
		plan.Centrality.Stewardship++
	case issuepolicy.CentralityPeripheral:
		plan.Centrality.Peripheral++
	default:
		plan.Centrality.Unclassified++
	}
}

func candidatePriority(c issuepolicy.Candidate) string {
	if priority := strings.TrimSpace(c.Priority); priority != "" {
		return priority
	}
	for _, label := range c.Labels {
		lower := strings.ToLower(strings.TrimSpace(label))
		if lower == "urgent" || lower == "critical" || strings.HasPrefix(lower, "priority/") || strings.HasPrefix(lower, "priority:") {
			return strings.TrimSpace(label)
		}
	}
	return ""
}

func selectionNote(centrality, priority, dispatchability string) string {
	if centrality == issuepolicy.CentralityStewardship && isUrgentPriority(priority) {
		return "urgent stewardship obligation may outrank ready Core work; centrality is non-scoring"
	}
	if dispatchability != issuepolicy.Dispatchable {
		return "repair readiness before selection; centrality remains visible and non-scoring"
	}
	return "centrality is non-scoring; order still considers urgency, dependencies, effort, and reversibility"
}

func isUrgentPriority(priority string) bool {
	p := strings.ToLower(strings.TrimSpace(priority))
	return p == "urgent" || p == "critical" || p == "p0" || p == "priority/p0" || p == "priority:critical" || p == "priority:urgent"
}

func collidesWithAny(lf leaf, bucket []leaf) bool {
	for _, other := range bucket {
		if collide(lf, other) {
			return true
		}
	}
	return false
}

// collide reports whether two dispatchable leaves would write overlapping file
// trees if dispatched concurrently.
//
//   - If both name explicit paths, they collide only when a path of one contains
//     or equals a path of the other (same lane with disjoint paths is safe — the
//     dispatch runbook's rule exactly).
//   - If either names no explicit paths, the leaf is treated as taking its whole
//     lane, so two such leaves collide when they share a non-empty lane.
func collide(a, b leaf) bool {
	if len(a.normPaths) > 0 && len(b.normPaths) > 0 {
		return pathsOverlap(a.normPaths, b.normPaths)
	}
	return a.member.Lane != "" && a.member.Lane == b.member.Lane
}

func collisionPairs(leaves []leaf) int {
	pairs := 0
	for i := 0; i < len(leaves); i++ {
		for j := i + 1; j < len(leaves); j++ {
			if collide(leaves[i], leaves[j]) {
				pairs++
			}
		}
	}
	return pairs
}

func pathsOverlap(a, b []string) bool {
	for _, ap := range a {
		for _, bp := range b {
			if pathOverlap(ap, bp) {
				return true
			}
		}
	}
	return false
}

// pathOverlap treats each path as a tree root: two paths overlap when they are
// equal or one is an ancestor directory of the other. Distinct files in the same
// directory do not overlap, so they may share a wave.
func pathOverlap(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	if a == b {
		return true
	}
	return strings.HasPrefix(b, a+"/") || strings.HasPrefix(a, b+"/")
}

func normalizePaths(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if n := normPath(p); n != "" {
			out = append(out, n)
		}
	}
	return out
}

func normPath(p string) string {
	p = strings.ReplaceAll(strings.TrimSpace(p), "\\", "/")
	p = strings.TrimSuffix(p, "/**")
	p = strings.TrimSuffix(p, "/*")
	return strings.Trim(p, "/")
}

// IsSplitTarget reports whether a non-OK review is best handled by decomposition
// (an epic/non-leaf or an oversized-step leaf) rather than scope/route repair. The
// two triggering reasons are the always-on STRUCTURAL gates issuecontract raises for
// a unit that must be broken up before dispatch; every other reason is repairable in
// place, so it is not a split target.
//
// Exported because "is this an epic to split?" must have ONE answer: the cohort planner
// here and `fak issue decompose` both route on it, and the cmd side carried a
// byte-identical private copy openly labelled a mirror. A reason added to one list and
// not the other would have made the planner and the verb disagree about the same issue.
func IsSplitTarget(review issuepolicy.Review) bool {
	for _, r := range review.Reasons {
		if r == issuepolicy.ReasonNotDispatchLeaf || r == issuepolicy.ReasonOversizedSteps {
			return true
		}
	}
	return false
}

func splitReasons(reasons []string) []string {
	out := make([]string, 0, len(reasons))
	for _, r := range reasons {
		if r == issuepolicy.ReasonNotDispatchLeaf || r == issuepolicy.ReasonOversizedSteps {
			out = append(out, r)
		}
	}
	if len(out) == 0 {
		out = append(out, issuepolicy.ReasonNotDispatchLeaf)
	}
	return out
}

// childIssueBudget mirrors issuecontract's split budget: ceil(steps / cap), and
// at least one child when the step budget is unknown.
func childIssueBudget(steps int) int {
	if steps <= 0 {
		return 1
	}
	return (steps + issuepolicy.MaxDispatchExpectedSteps - 1) / issuepolicy.MaxDispatchExpectedSteps
}

func stepUnits(steps int) int {
	if steps > 0 {
		return steps
	}
	return 1
}
