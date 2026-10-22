package model

import (
	"fmt"
	"math"
	"sort"
)

// ExpertRouteOverlapSchema names the cross-stream overlap report.
const ExpertRouteOverlapSchema = "fak.expert-route.overlap/v1"

// ExpertRouteBalanceStat is per-step expert-parallel balance over routed steps
// (one router decision per layer per step). MoeLB is DeepInfer's measured
// moe_lb: sum over steps of the busiest rank's picks divided by the sum of the
// mean rank's picks, so 1.0 is perfect and R is all picks on one rank.
type ExpertRouteBalanceStat struct {
	Steps         int     `json:"steps"`
	MoeLB         float64 `json:"moe_lb"`
	MeanStepRatio float64 `json:"mean_step_ratio"`
	P99StepRatio  float64 `json:"p99_step_ratio"`
}

// ExpertRouteRankBalance compares placements of one layer's experts across
// Ranks contiguous bands. LoadBanded is fitted per layer on the first half of
// that layer's steps and scored on the second half (held-out), next to
// CountEven on the same steps. LoadBandedInSample is fitted and scored on every
// step, which is what an in-sample predicted balance reports. Floor is the best
// any assignment of the observed picks could reach: ceil(picks/R) per step.
type ExpertRouteRankBalance struct {
	Ranks              int                    `json:"ranks"`
	FitSteps           int                    `json:"fit_steps"`
	CountEven          ExpertRouteBalanceStat `json:"count_even"`
	LoadBanded         ExpertRouteBalanceStat `json:"load_banded"`
	Floor              ExpertRouteBalanceStat `json:"floor"`
	CountEvenAll       ExpertRouteBalanceStat `json:"count_even_all"`
	LoadBandedInSample ExpertRouteBalanceStat `json:"load_banded_in_sample"`
}

// ExpertRouteStreamOverlap reports how much expert weight traffic concurrent
// streams would share if their i-th routed steps ran in the same batch step.
// Streams are aligned per layer by routed-step order; steps past the shortest
// stream for a layer are not counted.
type ExpertRouteStreamOverlap struct {
	Schema         string                   `json:"schema"`
	Phase          string                   `json:"phase,omitempty"`
	Streams        int                      `json:"streams"`
	Layers         int                      `json:"layers"`
	Steps          int                      `json:"steps"`
	Picks          int                      `json:"picks"`
	UniqueReads    int                      `json:"unique_reads"`
	SharedFraction float64                  `json:"shared_fraction"`
	MeanUnion      float64                  `json:"mean_union_per_step"`
	RankBalance    []ExpertRouteRankBalance `json:"rank_balance,omitempty"`
}

func expertRouteExperts(t ExpertRouteTrace, routes []ExpertRouteEvent) int {
	n := t.Header.Experts
	for _, ev := range routes {
		for _, e := range ev.Experts {
			if e+1 > n {
				n = e + 1
			}
		}
	}
	return n
}

func routeLayerSteps(routes []ExpertRouteEvent) map[int][][]int {
	byLayer := map[int][][]int{}
	for _, ev := range routes {
		byLayer[ev.Layer] = append(byLayer[ev.Layer], dedupeInts(ev.Experts))
	}
	return byLayer
}

func expertBandOwner(p TPPlan) []int {
	owner := make([]int, p.Dim)
	for r, s := range p.Shards {
		for e := s.Lo; e < s.Hi; e++ {
			owner[e] = r
		}
	}
	return owner
}

type balanceAccum struct {
	maxSum, meanSum float64
	ratios          []float64
}

func (a *balanceAccum) add(maxLoad, picks, ranks int) {
	if picks == 0 {
		return
	}
	mean := float64(picks) / float64(ranks)
	a.maxSum += float64(maxLoad)
	a.meanSum += mean
	a.ratios = append(a.ratios, float64(maxLoad)/mean)
}

func (a *balanceAccum) stat() ExpertRouteBalanceStat {
	st := ExpertRouteBalanceStat{Steps: len(a.ratios)}
	if st.Steps == 0 {
		return st
	}
	st.MoeLB = a.maxSum / a.meanSum
	sum := 0.0
	for _, r := range a.ratios {
		sum += r
	}
	st.MeanStepRatio = sum / float64(st.Steps)
	sorted := append([]float64(nil), a.ratios...)
	sort.Float64s(sorted)
	st.P99StepRatio = sorted[int(math.Ceil(0.99*float64(len(sorted))))-1]
	return st
}

func stepMaxRankLoad(step []int, owner []int, loads []int) int {
	for i := range loads {
		loads[i] = 0
	}
	maxLoad := 0
	for _, e := range step {
		r := owner[e]
		loads[r]++
		if loads[r] > maxLoad {
			maxLoad = loads[r]
		}
	}
	return maxLoad
}

func stepLoadHistogram(steps [][]int, n int) []int {
	load := make([]int, n)
	for _, step := range steps {
		for _, e := range step {
			load[e]++
		}
	}
	return load
}

// expertRouteRankBalance scores count-even and load-banded placements per
// routed step. A step's picks may repeat an expert when several streams share
// it: each repeat is one more token routed to that expert's rank.
func expertRouteRankBalance(byLayer map[int][][]int, n int, ranks []int) ([]ExpertRouteRankBalance, error) {
	layers := make([]int, 0, len(byLayer))
	for l := range byLayer {
		layers = append(layers, l)
	}
	sort.Ints(layers)
	out := make([]ExpertRouteRankBalance, 0, len(ranks))
	for _, r := range ranks {
		if r < 1 || r > n {
			return nil, fmt.Errorf("%w: ranks=%d outside [1,%d experts]", ErrExpertRouteTrace, r, n)
		}
		countPlan, err := NewTPPlan(n, r)
		if err != nil {
			return nil, err
		}
		countOwner := expertBandOwner(countPlan)
		loads := make([]int, r)
		var count, banded, floor, countAll, inSample balanceAccum
		row := ExpertRouteRankBalance{Ranks: r}
		for _, l := range layers {
			steps := byLayer[l]
			split := len(steps) / 2
			row.FitSteps += split
			fitted, err := LoadBalancedExpertBands(stepLoadHistogram(steps[:split], n), r)
			if err != nil {
				return nil, err
			}
			allPlan, err := LoadBalancedExpertBands(stepLoadHistogram(steps, n), r)
			if err != nil {
				return nil, err
			}
			fitOwner, allOwner := expertBandOwner(fitted), expertBandOwner(allPlan)
			for i, step := range steps {
				picks := len(step)
				countAll.add(stepMaxRankLoad(step, countOwner, loads), picks, r)
				inSample.add(stepMaxRankLoad(step, allOwner, loads), picks, r)
				if i < split {
					continue
				}
				count.add(stepMaxRankLoad(step, countOwner, loads), picks, r)
				banded.add(stepMaxRankLoad(step, fitOwner, loads), picks, r)
				floor.add((picks+r-1)/r, picks, r)
			}
		}
		row.CountEven, row.LoadBanded, row.Floor = count.stat(), banded.stat(), floor.stat()
		row.CountEvenAll, row.LoadBandedInSample = countAll.stat(), inSample.stat()
		out = append(out, row)
	}
	return out, nil
}

// AnalyzeExpertRouteStreams aligns separately recorded streams by routed-step
// order per layer and reports the shared fraction of expert reads plus the rank
// balance of the merged batch steps. It is pure.
func AnalyzeExpertRouteStreams(traces []ExpertRouteTrace, opts ExpertRouteAnalyzeOptions) (ExpertRouteStreamOverlap, error) {
	rep := ExpertRouteStreamOverlap{Schema: ExpertRouteOverlapSchema, Phase: opts.Phase, Streams: len(traces)}
	if len(traces) < 2 {
		return rep, fmt.Errorf("%w: overlap needs at least 2 streams, got %d", ErrExpertRouteTrace, len(traces))
	}
	perStream := make([]map[int][][]int, len(traces))
	n := 0
	for i, t := range traces {
		routes, _ := t.selectRoutes(opts)
		if len(routes) == 0 {
			return rep, fmt.Errorf("%w: stream %d has no routes after phase filter %q", ErrExpertRouteTrace, i, opts.Phase)
		}
		perStream[i] = routeLayerSteps(routes)
		if e := expertRouteExperts(t, routes); e > n {
			n = e
		}
	}
	merged := map[int][][]int{}
	for l, first := range perStream[0] {
		steps := len(first)
		for _, s := range perStream[1:] {
			if len(s[l]) < steps {
				steps = len(s[l])
			}
		}
		for i := 0; i < steps; i++ {
			var step []int
			seen := map[int]bool{}
			for _, s := range perStream {
				for _, e := range s[l][i] {
					step = append(step, e)
					seen[e] = true
				}
			}
			merged[l] = append(merged[l], step)
			rep.Picks += len(step)
			rep.UniqueReads += len(seen)
			rep.Steps++
		}
	}
	if rep.Steps == 0 {
		return rep, fmt.Errorf("%w: streams share no aligned layer steps", ErrExpertRouteTrace)
	}
	rep.Layers = len(merged)
	rep.SharedFraction = 1 - float64(rep.UniqueReads)/float64(rep.Picks)
	rep.MeanUnion = float64(rep.UniqueReads) / float64(rep.Steps)
	if len(opts.Ranks) > 0 {
		rb, err := expertRouteRankBalance(merged, n, opts.Ranks)
		if err != nil {
			return rep, err
		}
		rep.RankBalance = rb
	}
	return rep, nil
}
