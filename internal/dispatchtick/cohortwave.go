package dispatchtick

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
)

// This file bridges the creation-time cohort wave plan (internal/issuecohort)
// into the dispatch-time wave planner. issuecohort partitions a batch of
// candidate issues into concurrency-safe waves at CREATION time using a
// disjoint-tree rule whose overlap semantics mirror this package's
// duplicatePathOverlaps. Historically the dispatcher discarded that plan and
// recomputed the same partition from the open-issue set at launch. Seeding the
// dispatch waves from the cohort plan lets the two agree by construction — and
// CohortWaveConflicts re-checks the seeded partition under THIS package's own
// overlap rule so a parity test can prove the creation-time and dispatch-time
// rules agree on a fixture rather than assuming it.

// CohortWaveSeed is dispatchtick's projection of ONE issuecohort creation-time
// wave: the member keys that are safe to dispatch concurrently, plus the lease
// region/lanes to hand `dos arbitrate` for the whole wave. Seeding these lets the
// dispatch planner reuse the creation-time disjoint-tree partition instead of
// recomputing it from open issues at launch.
type CohortWaveSeed struct {
	Index       int
	Keys        []string
	LeaseRegion []string
	LeaseLanes  []string
}

// SeedWavesFromCohort adapts an issuecohort.Plan's wave assignments into dispatch
// wave seeds, preserving the cohort's concurrency-safe grouping and its lease
// region/lanes. Seeds come back in cohort wave order; blank member keys are
// dropped; empty in, empty out.
func SeedWavesFromCohort(plan issuecohort.Plan) []CohortWaveSeed {
	seeds := make([]CohortWaveSeed, 0, len(plan.Waves))
	for _, w := range plan.Waves {
		seed := CohortWaveSeed{
			Index:       w.Index,
			LeaseRegion: append([]string(nil), w.LeaseRegion...),
			LeaseLanes:  append([]string(nil), w.LeaseLanes...),
		}
		for _, m := range w.Members {
			if k := strings.TrimSpace(m.Key); k != "" {
				seed.Keys = append(seed.Keys, k)
			}
		}
		seeds = append(seeds, seed)
	}
	return seeds
}

// CohortWaveConflict is one member pair that issuecohort co-scheduled in a wave
// but that dispatchtick's OWN overlap rule would treat as a file-tree collision —
// a place where the creation-time and dispatch-time rules DISAGREE. A cohort plan
// with no conflicts is safe to seed directly, with no re-partition.
type CohortWaveConflict struct {
	WaveIndex int
	A         string
	B         string
}

// CohortWaveConflicts re-checks every intra-wave member pair of a cohort plan
// under dispatchtick's own overlap rule (duplicatePathOverlaps for path-scoped
// members, same-lane for whole-lane takers) and returns the pairs it would
// consider colliding. An empty result proves the two overlap rules agree on this
// plan: the cohort's disjoint-tree partition seeds the dispatch waves as-is.
func CohortWaveConflicts(plan issuecohort.Plan) []CohortWaveConflict {
	var conflicts []CohortWaveConflict
	for _, w := range plan.Waves {
		for i := 0; i < len(w.Members); i++ {
			for j := i + 1; j < len(w.Members); j++ {
				if cohortMembersCollide(w.Members[i], w.Members[j]) {
					conflicts = append(conflicts, CohortWaveConflict{
						WaveIndex: w.Index,
						A:         w.Members[i].Key,
						B:         w.Members[j].Key,
					})
				}
			}
		}
	}
	return conflicts
}

// cohortMembersCollide applies dispatchtick's file-tree overlap rule to two cohort
// wave members, mirroring issuecohort.collide but routed through this package's
// duplicatePathOverlaps: two path-scoped members collide when any pair of their
// normalized paths overlaps; a member that names no paths takes its whole lane,
// so two such members collide when they share a non-empty lane.
func cohortMembersCollide(a, b issuecohort.WaveMember) bool {
	ap := normalizeRepoPaths(a.Paths)
	bp := normalizeRepoPaths(b.Paths)
	if len(ap) > 0 && len(bp) > 0 {
		for _, x := range ap {
			for _, y := range bp {
				if duplicatePathOverlaps(x, y) {
					return true
				}
			}
		}
		return false
	}
	return a.Lane != "" && a.Lane == b.Lane
}
