package trajctl

// rollup.go — issue #2549, the hierarchical rollup over the flat curve fold (epic
// #2533). Objectives carry parent ids but curve.go folds each one in isolation: a parent
// with three children shows no aggregate, and a met child moves nothing upstream. This
// fold turns the flat per-(objective) curve into a TREE — each parent's progress is the
// budget-weighted aggregate of its children, a met child LOCKS its contribution at
// complete, and a child's DETOUR_OVERRUN propagates up to its (paused) parent so a tree
// view shows where the time went.
//
// It is pure and tier-1 like curve.go: it reads the already-folded State (objectives plus
// the per-objective curves CurveFor derives) and never steers or acts. The aggregate rule
// is FIXED and documented so the tree fold is golden-testable:
//
//   - weight(o)      = o.Budget.Turns, or 1 when unbudgeted (every node counts once).
//   - aggProgress(o) = 1.0 when o is MET (done is done — a late re-score cannot regress
//                      it); else o's own witnessed progress when o is a LEAF; else the
//                      budget-weighted mean of each non-abandoned child's aggProgress. An
//                      ABANDONED child is dropped from the fold entirely (dead work is not
//                      progress and not a drag).
//   - aggSignal(o)   = the WORST (highest-severity) signal in o's subtree: o's own signal
//                      folded with each still-OPEN child's aggSignal. Because curve.go only
//                      raises DETOUR_OVERRUN on a child whose parent is PAUSED, this
//                      worst-first fold is exactly how a detour overrun surfaces on the
//                      paused parent — the tree view of where the time went.

import (
	"fmt"
	"math"
	"sort"
)

// RollupSchema is the pinned schema id for a rollup report. Downstream consumers
// (tree views, status) pin to this string.
const RollupSchema = "fak-trajctl-rollup/1"

// ObjectiveRollup is one objective folded together with its subtree. Own* are the flat
// curve (what curve.go derives in isolation); Signal / AggProgress are the tree fold.
type ObjectiveRollup struct {
	ObjectiveID string          `json:"objective_id"`
	ParentID    string          `json:"parent_id,omitempty"`
	Status      ObjectiveStatus `json:"status"`
	OwnSignal   Signal          `json:"own_signal"`
	OwnProgress float64         `json:"own_progress"`
	Signal      Signal          `json:"signal"`
	AggProgress float64         `json:"agg_progress"`
	Weight      int             `json:"weight"`
	Children    []string        `json:"children,omitempty"`
	Descendants int             `json:"descendants"`
	Detail      string          `json:"detail"`
}

// RollupReport is the schema-pinned forest: every objective's rollup (Nodes, in id
// order) plus the worst-first root ids (objectives with no live parent).
type RollupReport struct {
	Schema string            `json:"schema"`
	Nodes  []ObjectiveRollup `json:"nodes"`
	Roots  []string          `json:"roots"`
}

// objectiveWeight is a node's declared budget weight for the fold: its turn budget, or 1
// when unbudgeted so every node still counts once.
func objectiveWeight(o Objective) int {
	if o.Budget.Turns > 0 {
		return o.Budget.Turns
	}
	return 1
}

// childrenOf returns the ids whose ParentID == id, in lexical order (deterministic).
func (s State) childrenOf(id string) []string {
	kids := make([]string, 0)
	for cid, o := range s.Objectives {
		if o.ParentID == id {
			kids = append(kids, cid)
		}
	}
	sort.Strings(kids)
	return kids
}

// rollupNode folds one objective together with its subtree. onPath guards a malformed
// parent ring (a cycle is broken by folding the repeat as a leaf so the fold can't hang);
// memo caches each fully-folded node so a shared child is folded once.
func (s State) rollupNode(id string, onPath map[string]bool, memo map[string]ObjectiveRollup) ObjectiveRollup {
	if r, ok := memo[id]; ok {
		return r
	}
	obj := s.Objectives[id]
	curve, _ := s.CurveFor(id)
	r := ObjectiveRollup{
		ObjectiveID: id,
		ParentID:    obj.ParentID,
		Status:      obj.Status,
		OwnSignal:   curve.Signal,
		OwnProgress: curve.Latest,
		Signal:      curve.Signal,
		Weight:      objectiveWeight(obj),
	}
	if onPath[id] {
		// Cycle: fold the repeat as a leaf (own curve only) so a parent ring cannot hang
		// the fold. Not memoized — the real fold of this id still runs at its own root.
		r.AggProgress = leafProgress(obj.Status, curve.Latest)
		r.Detail = "cycle: parent ring broken at " + id
		return r
	}
	onPath[id] = true

	worst := curve.Signal
	var sumW, sumWV float64
	descendants := 0
	for _, cid := range s.childrenOf(id) {
		cr := s.rollupNode(cid, onPath, memo)
		r.Children = append(r.Children, cid)
		descendants += 1 + cr.Descendants
		if cr.Status == StatusAbandoned {
			continue // dead work: dropped from BOTH the progress and the signal fold.
		}
		w := float64(cr.Weight)
		sumW += w
		sumWV += w * cr.AggProgress // a met child's AggProgress is already locked at 1.0
		if objectiveOpen(cr.Status) && signalSeverity(cr.Signal) > signalSeverity(worst) {
			worst = cr.Signal // only a still-open child's trajectory trouble surfaces up
		}
	}
	delete(onPath, id)

	r.Descendants = descendants
	r.Signal = worst
	switch {
	case obj.Status == StatusMet:
		r.AggProgress = 1 // met locks at complete regardless of subtree
	case len(r.Children) == 0 || sumW == 0:
		r.AggProgress = leafProgress(obj.Status, curve.Latest)
	default:
		r.AggProgress = rollupRound(sumWV / sumW)
	}
	r.Detail = fmt.Sprintf("%s: agg progress %.2f, worst signal %s, over %d child(ren) / %d descendant(s)",
		obj.Status, r.AggProgress, r.Signal, len(r.Children), r.Descendants)
	memo[id] = r
	return r
}

// leafProgress is the progress a childless (or every-child-abandoned) node contributes:
// 1.0 when met (locked complete), else its own witnessed progress.
func leafProgress(st ObjectiveStatus, latest float64) float64 {
	if st == StatusMet {
		return 1
	}
	return latest
}

// rollupRound rounds a weighted mean to 3 decimals so the tree fold is a stable,
// golden-comparable value.
func rollupRound(v float64) float64 { return math.Round(v*1000) / 1000 }

// RollupFor folds one objective and its subtree. ok is false if the objective was never
// declared. Mirrors CurveFor's single-objective shape.
func (s State) RollupFor(id string) (ObjectiveRollup, bool) {
	if _, ok := s.Objectives[id]; !ok {
		return ObjectiveRollup{}, false
	}
	return s.rollupNode(id, map[string]bool{}, map[string]ObjectiveRollup{}), true
}

// Rollup folds every declared objective into the hierarchical forest: one ObjectiveRollup
// per objective (Nodes, id order) plus the worst-first root ids (objectives with no live
// parent — an empty or dangling ParentID). Roots rank by subtree-worst signal severity,
// then by lower aggregate progress, then by id, so the tree most off-course lists first.
func (s State) Rollup() RollupReport {
	rep := RollupReport{Schema: RollupSchema, Nodes: make([]ObjectiveRollup, 0), Roots: make([]string, 0)}
	memo := map[string]ObjectiveRollup{}
	for _, id := range s.ObjectiveIDs() {
		rep.Nodes = append(rep.Nodes, s.rollupNode(id, map[string]bool{}, memo))
	}

	roots := make([]ObjectiveRollup, 0)
	for _, n := range rep.Nodes {
		if _, hasParent := s.Objectives[n.ParentID]; n.ParentID == "" || !hasParent {
			roots = append(roots, n) // no parent, or a dangling parent -> a root of the forest
		}
	}
	sort.SliceStable(roots, func(i, j int) bool {
		a, b := roots[i], roots[j]
		if sa, sb := signalSeverity(a.Signal), signalSeverity(b.Signal); sa != sb {
			return sa > sb
		}
		if a.AggProgress != b.AggProgress {
			return a.AggProgress < b.AggProgress
		}
		return a.ObjectiveID < b.ObjectiveID
	})
	for _, r := range roots {
		rep.Roots = append(rep.Roots, r.ObjectiveID)
	}
	return rep
}
