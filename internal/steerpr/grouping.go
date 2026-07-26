package steerpr

// grouping.go — issue #5040 (child of epic #5015): make the operator's unit of
// ATTENTION match the fleet's unit of DECISION.
//
// The overlay's default unit is one per (fak <leaf>) ship-stamp, inherited from
// `release prplan`, where it is right: a promotion PR is a lane artifact and the
// lane owns it. But the fleet does not dispatch by leaf — it dispatches by WAVE.
// `fak issue cohort` partitions a batch into concurrency-safe waves of
// tree-disjoint candidates, and the wave is what actually gets spawned together.
// So an operator watching a wave land sees it scattered across several leaf
// units, and an operator reading a leaf unit sees commits from several unrelated
// waves. Neither view is the thing that was decided, which makes the steering act
// the operator most wants — "this WAVE is going wrong, stop it" — inexpressible.
//
// The fix is one level of regrouping, decided explicitly: commits whose
// subject-bound issue belongs to a known cohort wave fold into ONE unit per wave;
// everything else keeps folding by leaf. Two rules hold this honest:
//
//   - No new grouping key is invented. The wave key is the cohort plan's own wave
//     index; membership is the plan's own members, matched by the issue number a
//     commit already binds in its subject. steerpr never reads the plan — the
//     caller projects it into WaveBindings, exactly as partial.go takes the issue
//     graph as data (this leaf stays pure and stdlib-only, architest tier 1).
//   - The basis is stated on every unit, never inferred. Two grouping bases
//     coexisting silently is a legibility REGRESSION, not a feature: an operator
//     who cannot tell why a unit holds what it holds has been made worse off. That
//     is why Unit.GroupedBy has no omitempty.
//
// Nothing else changes: a wave unit bands by the same worst-member fold, orders
// by the same total order, and is addressed by the same unit key an ack, a pause,
// or a redirect already takes.

import (
	"fmt"
	"strings"
)

// Grouping bases. GroupedByLeaf is the default and the FALLBACK (the common
// case: a commit with no wave binding); GroupedByWave is the regrouping this
// file adds.
const (
	GroupedByLeaf = "leaf"
	GroupedByWave = "wave"
)

// waveKeyPrefix opens every wave unit key. The colon is load-bearing: the ship
// stamp grammar is `(fak [a-z0-9][a-z0-9-]*)`, which cannot produce a colon, so
// no real leaf can ever collide with a wave key — a lane that happened to be
// named `wave-0` would otherwise silently absorb wave 0's commits.
const waveKeyPrefix = "wave:"

// WaveKey renders a cohort wave's stable unit key from its plan index. It is the
// string an operator passes to `fak steer ack|pause|redirect`, and the only place
// the wave key format is spelled.
func WaveKey(index int) string { return fmt.Sprintf("%s%d", waveKeyPrefix, index) }

// IsWaveKey reports whether a unit key names a cohort wave rather than a leaf.
func IsWaveKey(key string) bool { return strings.HasPrefix(key, waveKeyPrefix) }

// WaveBinding is ONE cohort wave's declared membership, projected by the caller
// from an existing `fak issue cohort` plan: the wave's own index plus the issue
// refs of its members. It is data, not a planner — steerpr neither builds waves
// nor re-partitions them, so there is exactly one wave planner in the repo and
// this overlay is a view over it.
//
// Issues carries "#123" or "123"; anything that is not a number is ignored
// rather than guessed at.
type WaveBinding struct {
	Index  int      `json:"index"`
	Issues []string `json:"issues"`
}

// WaveIndex folds bindings into the issue-ref -> wave-key lookup FoldUnitsByWave
// consumes. A ref claimed by two waves stays with the FIRST binding that claimed
// it: a later wave cannot silently steal a commit out of an earlier wave's unit,
// so the mapping is a function of the plan's own order and not of map iteration.
//
// An empty result (no bindings, or none carrying issue numbers) is the normal
// case and means "no wave grouping is available" — the fold then degrades to pure
// leaf grouping rather than to an empty view.
func WaveIndex(bindings []WaveBinding) map[string]string {
	out := map[string]string{}
	for _, b := range bindings {
		key := WaveKey(b.Index)
		for _, raw := range b.Issues {
			ref, ok := normalizeIssueRef(raw)
			if !ok {
				continue
			}
			if _, taken := out[ref]; taken {
				continue
			}
			out[ref] = key
		}
	}
	return out
}

// normalizeIssueRef canonicalizes "123" / "#123" / " #123 " to "#123", the form
// ParseLog already puts in Commit.Resolves. A non-numeric ref is refused rather
// than normalized to something that would match nothing quietly.
func normalizeIssueRef(raw string) (string, bool) {
	s := strings.TrimSpace(raw)
	s = strings.TrimPrefix(s, "#")
	if s == "" {
		return "", false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return "", false
		}
	}
	return "#" + s, true
}

// FoldUnitsByWave is the one fold behind both grouping bases.
//
// A commit lands in a WAVE unit when one of its subject-bound issue refs names a
// wave in waveOf; otherwise it lands in its LEAF unit. Body mentions deliberately
// do not bind: Commit.Resolves is the closure-grade set (the ref the commit says
// it resolves), and a passing mention of a wave member's issue number in a commit
// body is not a statement that this commit is part of that wave. Grouping on a
// mention would pull unrelated work into an operator's decision unit.
//
// Unstamped commits stay unstamped even when they are wave-bound. The
// (units, unstamped) partition is what makes every landed commit visible, and
// `prplan --check` gates on that debt pile; letting a wave binding absorb an
// unstamped commit would quietly shrink the legibility debt without anyone
// stamping anything.
//
// waveOf nil or empty yields exactly the leaf fold, which is why FoldUnits can be
// this function with no bindings.
func FoldUnitsByWave(commits []Commit, waveOf map[string]string) (units []Unit, unstamped []Commit) {
	byKey := map[string]*Unit{}
	var order []string
	for _, c := range commits {
		c.Band = commitBand(c)
		if c.Leaf == "" {
			unstamped = append(unstamped, c)
			continue
		}
		key, basis := c.Leaf, GroupedByLeaf
		if wave, ok := waveFor(c, waveOf); ok {
			key, basis = wave, GroupedByWave
		}
		unit, ok := byKey[key]
		if !ok {
			unit = &Unit{Leaf: key, GroupedBy: basis, Types: map[string]int{}}
			byKey[key] = unit
			order = append(order, key)
		}
		unit.Commits = append(unit.Commits, c)
		if basis == GroupedByWave {
			unit.Leaves = MergeRefs(unit.Leaves, []string{c.Leaf})
		}
		if c.Type != "" {
			unit.Types[c.Type]++
		}
		unit.Resolves = MergeRefs(unit.Resolves, c.Resolves)
		unit.Mentions = MergeRefs(unit.Mentions, c.Mentions)
		unit.Files = MergeRefs(unit.Files, c.Files)
	}
	units = make([]Unit, 0, len(byKey))
	for _, key := range order {
		unit := byKey[key]
		// git log yields newest-first; a PR body reads oldest-first.
		for i, j := 0, len(unit.Commits)-1; i < j; i, j = i+1, j-1 {
			unit.Commits[i], unit.Commits[j] = unit.Commits[j], unit.Commits[i]
		}
		// A body mention that some commit subject-binds is already a closure.
		unit.Mentions = SubtractRefs(unit.Mentions, unit.Resolves)
		// The worst-member rule is the SAME rule for both bases — a wave unit is
		// as pessimistic as a leaf unit, so regrouping can never clear a band.
		unit.Band = FoldBand(unit.Commits)
		unit.Title = UnitTitle(*unit)
		units = append(units, *unit)
	}
	sortBiggestFirst(units)
	return units, unstamped
}

// waveFor resolves the wave a commit belongs to, if any. Resolves is sorted by
// ParseLog, so a commit binding two issues in two different waves resolves to the
// same wave on every run — deterministic, and reported rather than duplicated:
// one commit is a member of exactly one unit.
func waveFor(c Commit, waveOf map[string]string) (string, bool) {
	if len(waveOf) == 0 {
		return "", false
	}
	for _, ref := range c.Resolves {
		if wave, ok := waveOf[ref]; ok {
			return wave, true
		}
	}
	return "", false
}

// WaveUnits returns the units grouped by wave, in the order given. Naming the
// set lets a caller post "3 of 11 units are wave units" up front, so an operator
// scanning the overlay sees that two bases are in play before reading any single
// unit rather than discovering it halfway down.
func WaveUnits(units []Unit) []Unit {
	var out []Unit
	for _, u := range units {
		if u.GroupedBy == GroupedByWave {
			out = append(out, u)
		}
	}
	return out
}

// GroupingBasis renders the unit's basis for an operator line. A unit whose
// basis is somehow unset reads "unknown" rather than defaulting to leaf: an
// un-stated basis is exactly the confusion this issue refuses, so it is surfaced
// instead of being papered over with the common case.
func GroupingBasis(u Unit) string {
	switch u.GroupedBy {
	case GroupedByWave, GroupedByLeaf:
		return u.GroupedBy
	default:
		return "unknown"
	}
}
