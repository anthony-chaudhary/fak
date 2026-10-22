package model

import (
	"os"
	"strings"
	"sync"
	"sync/atomic"
)

// v41CohortExpertUnionEnv selects the decode layer cohort's cross-lane expert
// union: "1" always unions, "0" keeps every lane on its own fused routed path,
// and "auto" unions only when lanes share at least one top-k of flat rows.
// The default stays "1" until a hardware ABBA measures auto's token time.
const v41CohortExpertUnionEnv = "FAK_V41_COHORT_EXPERT_UNION"

type v41CohortExpertUnionMode uint8

const (
	v41CohortExpertUnionOn v41CohortExpertUnionMode = iota
	v41CohortExpertUnionOff
	v41CohortExpertUnionAuto
)

var v41CohortExpertUnionSetting = sync.OnceValue(func() v41CohortExpertUnionMode {
	return parseV41CohortExpertUnionMode(os.Getenv(v41CohortExpertUnionEnv))
})

func parseV41CohortExpertUnionMode(value string) v41CohortExpertUnionMode {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "0":
		return v41CohortExpertUnionOff
	case "auto":
		return v41CohortExpertUnionAuto
	}
	return v41CohortExpertUnionOn
}

func (mode v41CohortExpertUnionMode) admitsSharedRows(shared int) bool {
	return mode != v41CohortExpertUnionAuto || shared >= V41RouterTopK
}

// V41CohortExpertUnionCounters is the process-wide source for
// fak_v41_cohort_expert_union_total{outcome} and the shared-rows observation.
// SharedRows[n] counts built unions whose lanes shared n flat rows.
// DeclinedCapacity counts unions over the row cap; DeclinedAdmission counts
// any routed-cohort lease decline (expert ring fit, backend or weight
// capability, routing payload), so it is not an expert-ring sizing signal alone.
type V41CohortExpertUnionCounters struct {
	Selected          uint64                            `json:"selected"`
	DeclinedOverlap   uint64                            `json:"declined_overlap"`
	DeclinedCapacity  uint64                            `json:"declined_capacity"`
	DeclinedAdmission uint64                            `json:"declined_admission"`
	SharedRows        [v41LayerCohortMaxRows + 1]uint64 `json:"shared_rows"`
}

var v41CohortExpertUnionStats struct {
	selected, declinedOverlap, declinedCapacity, declinedAdmission atomic.Uint64
	sharedRows                                                     [v41LayerCohortMaxRows + 1]atomic.Uint64
}

func observeV41CohortSharedRows(shared int) {
	if shared >= 0 && shared < len(v41CohortExpertUnionStats.sharedRows) {
		v41CohortExpertUnionStats.sharedRows[shared].Add(1)
	}
}

// V41CohortExpertUnionSnapshot reads the cumulative cohort expert-union outcomes.
func V41CohortExpertUnionSnapshot() V41CohortExpertUnionCounters {
	s := &v41CohortExpertUnionStats
	out := V41CohortExpertUnionCounters{
		Selected:          s.selected.Load(),
		DeclinedOverlap:   s.declinedOverlap.Load(),
		DeclinedCapacity:  s.declinedCapacity.Load(),
		DeclinedAdmission: s.declinedAdmission.Load(),
	}
	for i := range s.sharedRows {
		out.SharedRows[i] = s.sharedRows[i].Load()
	}
	return out
}
