package webbench

// Geometry is the per-web-task agent-workload shape the cost arms replay.
// It models a multi-turn web automation session: a shared cacheable prefix
// of P tokens (system prompt + browser tool schemas + task description), then T
// navigation turns, each performing an action (decoding A assistant tokens) and
// ingesting the resulting DOM/page state (R tokens).
//
// This is the web-agent analogue of sessionbench's (P,T,D,R) and swebench's
// geometry, but tailored to web workloads where "result tokens" are DOM state
// rather than file reads.
type Geometry struct {
	TaskID     string `json:"task_id"`
	Prefix     int    `json:"prefix"`    // P: system + browser tools + task description (cacheable)
	Turns      int    `json:"turns"`     // T: navigation/action rounds to complete
	Action     int    `json:"action"`    // A: assistant tokens per turn (action + reasoning)
	DOMState   int    `json:"dom_state"` // R: ingested page state tokens after each action
	Difficulty string `json:"difficulty,omitempty"`
	Source     string `json:"source"` // "trajectory" | "difficulty" | "default"
}

// GeometryModel maps web task instances to Geometry. The defaults are designed
// for a browser agent (e.g., Browser Use, WebVoyager class) and can be
// overridden per run. Turn counts vary by difficulty like swebench.
type GeometryModel struct {
	BasePrefix        int            // system prompt + browser tool schemas, shared across tasks
	TaskDesc          int            // average task description tokens (added to base prefix)
	Action            int            // assistant tokens per turn (action selection + reasoning)
	DOMState          int            // average DOM state tokens ingested per turn
	TurnsByDifficulty map[string]int // difficulty -> expected navigation turns
	DefaultTurns      int            // fallback when difficulty unknown
	Trajectories      map[string]int // optional real turn counts from recorded runs
}

// DefaultGeometryModel returns standard knobs for a browser agent.
// BasePrefix ~3K accounts for a browser agent's system prompt + action schemas.
// Action ~150 tokens per navigation action (selector choice + reasoning).
// DOMState ~2K tokens for typical page state after an action.
func DefaultGeometryModel() GeometryModel {
	return GeometryModel{
		BasePrefix: 3000,
		TaskDesc:   400,
		Action:     150,
		DOMState:   2000,
		TurnsByDifficulty: map[string]int{
			"easy":   5,
			"medium": 12,
			"hard":   25,
			"expert": 45,
		},
		DefaultTurns: 15,
	}
}

// Derive builds Geometry for one web task instance. The prefix includes the
// base preamble plus the task description tokens (both cacheable across turns).
// Turn count comes from trajectory (if recorded), difficulty bucket, or default.
func (gm GeometryModel) Derive(in Instance) Geometry {
	descTokens := EstimateTokens(in.Description) + EstimateTokens(in.Instructions)
	if descTokens == 0 {
		descTokens = gm.TaskDesc
	}

	g := Geometry{
		TaskID:     in.TaskID,
		Prefix:     gm.BasePrefix + descTokens,
		Action:     gm.Action,
		DOMState:   gm.DOMState,
		Difficulty: in.Difficulty,
	}

	// Determine turn count with honest provenance.
	switch {
	case gm.Trajectories != nil && gm.Trajectories[in.TaskID] > 0:
		g.Turns = gm.Trajectories[in.TaskID]
		g.Source = "trajectory"
	case in.Difficulty != "" && gm.TurnsByDifficulty[in.Difficulty] > 0:
		g.Turns = gm.TurnsByDifficulty[in.Difficulty]
		g.Source = "difficulty"
	case len(in.Actions) > 0:
		// Use action count from the instance as a proxy for turns.
		g.Turns = len(in.Actions)
		g.Source = "actions"
	default:
		g.Turns = gm.DefaultTurns
		g.Source = "default"
	}

	return g
}

// DeriveAll maps a whole dataset to geometries, preserving order.
func (gm GeometryModel) DeriveAll(d *Dataset) []Geometry {
	if d == nil {
		return nil
	}
	out := make([]Geometry, 0, d.Len())
	for _, in := range d.Instances {
		out = append(out, gm.Derive(in))
	}
	return out
}

// Cost arms for web tasks (parallel to swebench):
// A: naive re-prefill every turn (no KV persistence)
// B: per-agent KV (each agent reuses its own prefix, no cross-agent sharing)
// C: fak fused (shared prefix across all agents, cross-worker reuse)

// ArmCost is the prefill-token cost for one geometry under a given policy.
type ArmCost struct {
	Workers int   `json:"workers"`
	ANaive  int64 `json:"a_naive"` // re-prefill full context every turn, every worker
	BAgent  int64 `json:"b_agent"` // per-agent KV, no cross-worker sharing
	CFak    int64 `json:"c_fak"`   // fak fused, shared prefix
}

// ComputeArms calculates the three cost arms for a geometry at given worker count.
func (g Geometry) ComputeArms(workers int) ArmCost {
	// A: naive - every worker re-prefills the full growing context each turn.
	// Context = Prefix + (i * (Action + DOMState)) for turn i.
	// Sum over T turns: Σ (Prefix + i*(Action+DOMState)) for i=0..T-1
	// = T*Prefix + (Action+DOMState) * T*(T-1)/2
	growth := int64(g.Action + g.DOMState)
	perWorkerNaive := int64(g.Turns)*int64(g.Prefix) + growth*int64(g.Turns*(g.Turns-1)/2)
	a := int64(workers) * perWorkerNaive

	// B: per-agent KV - each worker reuses its own prefix.
	// First turn prefill = Prefix; subsequent turns only prefill new tokens.
	// Per worker: Prefix + (Action+DOMState) * (T-1)
	perWorkerAgent := int64(g.Prefix) + growth*int64(g.Turns-1)
	b := int64(workers) * perWorkerAgent

	// C: fak fused - shared prefix across all workers (cross-worker reuse).
	// First worker pays Prefix once; all workers reuse it for first turn.
	// Subsequent turns: each worker prefill only its Action + DOMState delta.
	// Total = Prefix + workers * (Action+DOMState) * (T-1)
	c := int64(g.Prefix) + int64(workers)*growth*int64(g.Turns-1)

	return ArmCost{
		Workers: workers,
		ANaive:  a,
		BAgent:  b,
		CFak:    c,
	}
}

// Summary is the aggregated view of cost arms across a dataset.
type Summary struct {
	Instances       int            `json:"instances"`
	TotalTurns      int            `json:"total_turns"`
	TurnsMin        int            `json:"turns_min"`
	TurnsMedian     int            `json:"turns_median"`
	TurnsMax        int            `json:"turns_max"`
	DifficultyDist  map[string]int `json:"difficulty_dist"`
	CategoryDist    map[string]int `json:"category_dist"`
	GeometrySources map[string]int `json:"geometry_sources"`
	Prefill         []PrefillRow   `json:"prefill"` // A/B/C by worker count
}

// PrefillRow is one row of the prefill work-elimination table.
type PrefillRow struct {
	Workers int     `json:"workers"`
	ANaive  int64   `json:"a_naive"`
	BAgent  int64   `json:"b_agent"`
	CFak    int64   `json:"c_fak"`
	AOverC  float64 `json:"a_over_c"` // net work-elimination vs naive
	BOverC  float64 `json:"b_over_c"` // cross-worker prefix reuse
	AOverB  float64 `json:"a_over_b"` // turn-tax (re-prefill vs KV persistence)
}

// Describe computes the cost-elimination summary for a dataset.
func Describe(d *Dataset, gm GeometryModel, workers []int) Summary {
	geoms := gm.DeriveAll(d)

	// Aggregate stats.
	var totalTurns int
	minTurns, maxTurns := -1, -1
	turnsSlice := make([]int, 0, len(geoms))
	geomSources := make(map[string]int)
	diffDist := make(map[string]int)
	catDist := make(map[string]int)

	for _, g := range geoms {
		totalTurns += g.Turns
		turnsSlice = append(turnsSlice, g.Turns)
		if minTurns < 0 || g.Turns < minTurns {
			minTurns = g.Turns
		}
		if g.Turns > maxTurns {
			maxTurns = g.Turns
		}
		geomSources[g.Source]++
		if g.Difficulty != "" {
			diffDist[g.Difficulty]++
		}
	}
	// Category stats from instances.
	for _, in := range d.Instances {
		cat := in.Category
		if cat == "" {
			cat = "uncategorized"
		}
		catDist[cat]++
	}

	// Median turns.
	medianTurns := percentile(turnsSlice, 50)

	// Compute prefill rows for each worker count.
	prefillRows := make([]PrefillRow, 0, len(workers))
	for _, w := range workers {
		var sumA, sumB, sumC int64
		for _, g := range geoms {
			cost := g.ComputeArms(w)
			sumA += cost.ANaive
			sumB += cost.BAgent
			sumC += cost.CFak
		}
		row := PrefillRow{
			Workers: w,
			ANaive:  sumA,
			BAgent:  sumB,
			CFak:    sumC,
		}
		if row.CFak > 0 {
			row.AOverC = float64(row.ANaive) / float64(row.CFak)
			row.BOverC = float64(row.BAgent) / float64(row.CFak)
		}
		if row.BAgent > 0 {
			row.AOverB = float64(row.ANaive) / float64(row.BAgent)
		}
		prefillRows = append(prefillRows, row)
	}

	return Summary{
		Instances:       d.Len(),
		TotalTurns:      totalTurns,
		TurnsMin:        minTurns,
		TurnsMedian:     medianTurns,
		TurnsMax:        maxTurns,
		DifficultyDist:  diffDist,
		CategoryDist:    catDist,
		GeometrySources: geomSources,
		Prefill:         prefillRows,
	}
}

// percentile computes the p-th percentile of a sorted slice.
func percentile(data []int, p int) int {
	if len(data) == 0 {
		return 0
	}
	sorted := make([]int, len(data))
	copy(sorted, data)
	// Simple sort - fine for benchmark sizes.
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	idx := len(sorted) * p / 100
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
