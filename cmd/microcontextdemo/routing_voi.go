package main

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"
)

const routingVOISchema = "fak-microcontext-routing-voi/1"

type routingVOIReport struct {
	Schema       string              `json:"schema"`
	Mode         string              `json:"mode"`
	Seed         int64               `json:"seed"`
	Trials       int                 `json:"trials"`
	RecordsTrial int                 `json:"records_per_trial"`
	Calibration  routingCalibration  `json:"calibration"`
	Mixtures     []routingMixtureRun `json:"mixtures"`
	Findings     []string            `json:"findings"`
	Limits       []string            `json:"limits"`
}

type routingCalibration struct {
	Source        string              `json:"source"`
	Utility       string              `json:"utility"`
	QualityReward float64             `json:"quality_reward"`
	Stages        map[string]stageCal `json:"stages"`
}

type stageCal struct {
	LatencyMS float64 `json:"latency_ms"`
	CostUnits float64 `json:"cost_units"`
}

type routingMixtureRun struct {
	Name            string              `json:"name"`
	Weights         map[string]int      `json:"weights_percent"`
	Policies        []routingPolicyStat `json:"policies"`
	Winner          string              `json:"winner"`
	AdaptiveRegret  float64             `json:"adaptive_regret_vs_oracle"`
	AdaptiveQuality float64             `json:"adaptive_quality"`
}

type routingPolicyStat struct {
	Policy          string         `json:"policy"`
	Quality         float64        `json:"quality"`
	MeanLatencyMS   float64        `json:"mean_latency_ms"`
	P95LatencyMS    float64        `json:"p95_latency_ms"`
	MeanCostUnits   float64        `json:"mean_cost_units"`
	MeanUtility     float64        `json:"mean_utility"`
	Wrong           int            `json:"wrong"`
	Abstained       int            `json:"abstained"`
	StageCalls      map[string]int `json:"stage_calls"`
	CancelledStages int            `json:"cancelled_stages"`
}

type routingOutcome struct {
	correct, abstain bool
	latency, cost    float64
	calls            map[string]int
	cancelled        int
}

func defaultRoutingCalibration() routingCalibration {
	return routingCalibration{
		Source:        "controlled deterministic stage fixture; latency and cost are calibrated workload units, not provider billing",
		Utility:       "100*correct - 100*wrong - 15*abstain - latency_ms - 8*cost_units",
		QualityReward: 100,
		Stages: map[string]stageCal{
			"exact_filter":     {LatencyMS: 0.05, CostUnits: 0.001},
			"selector":         {LatencyMS: 0.80, CostUnits: 0.040},
			"expensive_filter": {LatencyMS: 1.50, CostUnits: 0.050},
			"model":            {LatencyMS: 8.00, CostUnits: 0.800},
			"tool":             {LatencyMS: 5.00, CostUnits: 0.300},
		},
	}
}

var routingMixtures = []struct {
	name string
	w    map[string]int
}{
	{"exact-heavy", map[string]int{"exact": 98, "semantic": 1, "fresh": 1, "boundary": 0}},
	{"semantic-heavy", map[string]int{"exact": 10, "semantic": 75, "fresh": 5, "boundary": 10}},
	{"fresh-tool-heavy", map[string]int{"exact": 10, "semantic": 5, "fresh": 75, "boundary": 10}},
	{"mixed", map[string]int{"exact": 25, "semantic": 30, "fresh": 30, "boundary": 15}},
	{"boundary-heavy", map[string]int{"exact": 10, "semantic": 15, "fresh": 15, "boundary": 60}},
}

func stablePercent(seed int64, trial, record int, salt string) int {
	b := make([]byte, 16)
	binary.LittleEndian.PutUint64(b, uint64(seed))
	binary.LittleEndian.PutUint32(b[8:], uint32(trial))
	binary.LittleEndian.PutUint32(b[12:], uint32(record))
	h := sha256.Sum256(append(b, salt...))
	return int(binary.LittleEndian.Uint32(h[:4]) % 100)
}

func pickKind(weights map[string]int, n int) string {
	for _, k := range []string{"exact", "semantic", "fresh", "boundary"} {
		n -= weights[k]
		if n < 0 {
			return k
		}
	}
	return "boundary"
}

func addStage(o *routingOutcome, c routingCalibration, stage string) {
	x := c.Stages[stage]
	o.latency += x.LatencyMS
	o.cost += x.CostUnits
	o.calls[stage]++
}

func runRoutingPolicy(policy, kind string, seed int64, trial, record int, c routingCalibration) routingOutcome {
	if policy == "oracle" {
		// Hindsight chooses the highest-utility admissible route per record. It is a
		// counterfactual lower bound on regret, not an implementable online policy.
		choices := []routingOutcome{runRoutingPolicy("always-model", kind, seed, trial, record, c), runRoutingPolicy("always-filter-tool", kind, seed, trial, record, c), runRoutingPolicy("adaptive", kind, seed, trial, record, c)}
		sort.Slice(choices, func(i, j int) bool { return outcomeUtility(choices[i]) > outcomeUtility(choices[j]) })
		return choices[0]
	}
	o := routingOutcome{calls: map[string]int{}}
	addStage(&o, c, "exact_filter")
	if kind == "exact" {
		o.correct = true
		if policy == "adaptive" || policy == "oracle" {
			addStage(&o, c, "selector")
		}
		o.cancelled = 3
		return o
	}

	success := func(stage string, threshold int) bool {
		return stablePercent(seed, trial, record, policy+stage) < threshold
	}
	switch policy {
	case "always-model":
		addStage(&o, c, "model")
		threshold := map[string]int{"semantic": 96, "fresh": 52, "boundary": 72}[kind]
		o.correct = success("model", threshold)
		o.cancelled = 2
	case "always-filter-tool":
		addStage(&o, c, "expensive_filter")
		addStage(&o, c, "tool")
		threshold := map[string]int{"semantic": 68, "fresh": 99, "boundary": 88}[kind]
		o.correct = success("tool", threshold)
		o.cancelled = 1
	case "adaptive":
		addStage(&o, c, "selector")
		switch kind {
		case "semantic":
			addStage(&o, c, "model")
			o.correct = success("model", 96)
			o.cancelled = 2
		case "fresh":
			addStage(&o, c, "tool")
			o.correct = success("tool", 98)
			o.cancelled = 2
		case "boundary":
			addStage(&o, c, "expensive_filter")
			if stablePercent(seed, trial, record, "boundary-route") < 50 {
				addStage(&o, c, "model")
				o.correct = success("model", 94)
			} else {
				addStage(&o, c, "tool")
				o.correct = success("tool", 96)
			}
			o.cancelled = 1
		}
	default:
		o.abstain = true
	}
	if !o.correct && stablePercent(seed, trial, record, policy+"abstain") < 20 {
		o.abstain = true
	}
	return o
}

func outcomeUtility(o routingOutcome) float64 {
	q := -100.0
	if o.correct {
		q = 100
	} else if o.abstain {
		q = -15
	}
	return q - o.latency - 8*o.cost
}

func percentile95(xs []float64) float64 {
	sort.Float64s(xs)
	if len(xs) == 0 {
		return 0
	}
	return xs[(95*len(xs)-1)/100]
}

func runRoutingVOI(output string, seed int64, trials, records int) error {
	if output == "" || trials < 2 || records < 20 {
		return fmt.Errorf("output, trials >=2, and records >=20 required")
	}
	c := defaultRoutingCalibration()
	rep := routingVOIReport{Schema: routingVOISchema, Mode: "controlled executable value-of-information routing experiment", Seed: seed, Trials: trials, RecordsTrial: records, Calibration: c}
	for _, mix := range routingMixtures {
		m := routingMixtureRun{Name: mix.name, Weights: mix.w}
		policies := []string{"always-model", "always-filter-tool", "adaptive", "oracle"}
		for _, policy := range policies {
			var lat []float64
			var totalCost, totalUtility float64
			var correct, wrong, abstain, cancelled int
			calls := map[string]int{}
			for t := 0; t < trials; t++ {
				for i := 0; i < records; i++ {
					kind := pickKind(mix.w, stablePercent(seed, t, i, "kind"))
					o := runRoutingPolicy(policy, kind, seed, t, i, c)
					lat = append(lat, o.latency)
					totalCost += o.cost
					totalUtility += outcomeUtility(o)
					cancelled += o.cancelled
					if o.correct {
						correct++
					} else if o.abstain {
						abstain++
					} else {
						wrong++
					}
					for k, v := range o.calls {
						calls[k] += v
					}
				}
			}
			n := float64(trials * records)
			sumLat := 0.0
			for _, v := range lat {
				sumLat += v
			}
			s := routingPolicyStat{Policy: policy, Quality: float64(correct) / n, MeanLatencyMS: sumLat / n, P95LatencyMS: percentile95(lat), MeanCostUnits: totalCost / n, MeanUtility: totalUtility / n, Wrong: wrong, Abstained: abstain, StageCalls: calls, CancelledStages: cancelled}
			m.Policies = append(m.Policies, s)
		}
		m.Winner = m.Policies[0].Policy
		best := m.Policies[0].MeanUtility
		var adaptive, oracle routingPolicyStat
		for _, p := range m.Policies {
			if p.Policy != "oracle" && p.MeanUtility > best {
				best = p.MeanUtility
				m.Winner = p.Policy
			}
			if p.Policy == "adaptive" {
				adaptive = p
			}
			if p.Policy == "oracle" {
				oracle = p
			}
		}
		m.AdaptiveRegret = oracle.MeanUtility - adaptive.MeanUtility
		m.AdaptiveQuality = adaptive.Quality
		rep.Mixtures = append(rep.Mixtures, m)
	}
	rep.Findings = []string{"Exact-heavy inputs reward deterministic early stop; adaptive routing is not universally best.", "Semantic-heavy and fresh-evidence-heavy mixtures expose opposite static-policy advantages.", "Mixed and boundary-heavy workloads test whether selector cost is repaid by avoiding wrong expensive stages.", "Cancellation counts unopened expensive stages, making negative work observable rather than inferred."}
	rep.Limits = []string{"Stage latency and cost are controlled calibrated units, not live provider tokens, dollars, or billing.", "Fixture outcomes test routing policy and regret accounting, not general model intelligence.", "Live endpoint measurement remains #6110; a non-zero independently adjudicated semantic corpus remains #6124."}
	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(output, b, 0644)
}

func verifyRoutingVOI(path string) error {
	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var r routingVOIReport
	if err = json.Unmarshal(b, &r); err != nil {
		return err
	}
	if r.Schema != routingVOISchema || r.Trials < 2 || r.RecordsTrial < 20 || len(r.Mixtures) != 5 {
		return fmt.Errorf("invalid routing VOI envelope")
	}
	seenAdaptiveWin := false
	seenStaticWin := false
	for _, m := range r.Mixtures {
		if len(m.Policies) != 4 {
			return fmt.Errorf("%s: policy matrix incomplete", m.Name)
		}
		seen := map[string]bool{}
		for _, p := range m.Policies {
			seen[p.Policy] = true
			if p.Quality < 0 || p.Quality > 1 || p.MeanLatencyMS <= 0 || p.MeanCostUnits <= 0 {
				return fmt.Errorf("%s/%s: invalid metrics", m.Name, p.Policy)
			}
		}
		for _, p := range []string{"always-model", "always-filter-tool", "adaptive", "oracle"} {
			if !seen[p] {
				return fmt.Errorf("%s: missing %s", m.Name, p)
			}
		}
		if m.Winner == "adaptive" {
			seenAdaptiveWin = true
		} else {
			seenStaticWin = true
		}
		if m.AdaptiveRegret < 0 {
			return fmt.Errorf("%s: negative regret", m.Name)
		}
	}
	if !seenAdaptiveWin || !seenStaticWin {
		return fmt.Errorf("experiment lacks a crossover boundary")
	}
	if len(r.Limits) < 3 {
		return fmt.Errorf("claim boundary incomplete")
	}
	return nil
}

var _ = time.Now
