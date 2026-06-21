package swebench

// Geometry is the per-instance agent-workload shape the harness-cost arms replay:
// a shared, cacheable prefix of P tokens (system prompt + tool schemas + the
// problem statement), then T model turns, each decoding D assistant tokens and
// then ingesting R tool-result tokens (file reads, test output). It is the
// SWE-bench-shaped analogue of sessionbench's synthetic (P,T,D,R) — but driven by
// REAL instance signal (problem-statement size + the official difficulty bucket,
// or a recorded trajectory's true turn count) instead of LCG noise. That is what
// makes the cost comparison "SWE-bench Verified" rather than a synthetic stand-in.
type Geometry struct {
	InstanceID    string `json:"instance_id"`
	Prefix        int    `json:"prefix"`         // P: initial context (system+tools+problem), cacheable across turns
	Turns         int    `json:"turns"`          // T: model round-trips to solve
	Decode        int    `json:"decode"`         // D: assistant tokens decoded per turn
	Result        int    `json:"result"`         // R: tool-result tokens ingested between turns
	ProblemTokens int    `json:"problem_tokens"` // the part of P that is this instance's problem statement
	Difficulty    string `json:"difficulty,omitempty"`
	Source        string `json:"source"` // "trajectory" | "difficulty" | "default" — honest provenance of T
}

// GeometryModel maps instances to Geometry. The defaults below are the knobs a
// run can override; T is the only field with a difficulty-derived default, and
// every Geometry records (in Source) whether T came from a real trajectory, the
// difficulty bucket, or the flat default — so a report never silently passes a
// bucket estimate off as a measured trajectory.
type GeometryModel struct {
	BasePrefix        int            // system prompt + tool schemas, shared across all instances (cacheable)
	Decode            int            // assistant tokens per turn
	Result            int            // tool-result tokens ingested per turn
	TurnsByDifficulty map[string]int // difficulty bucket -> expected turns
	DefaultTurns      int            // fallback when difficulty is unknown
	// Trajectories optionally supplies a real per-instance turn count (from a
	// recorded agent run / bench replay-data); when present it overrides the
	// difficulty estimate and sets Source="trajectory".
	Trajectories map[string]int
}

// DefaultGeometryModel returns the standard knobs. The turn counts are honest
// order-of-magnitude estimates for a coding agent (mini-swe-agent class) by SWE-
// bench's official difficulty buckets — harder issues take more round-trips —
// and are clearly labeled difficulty-derived in any output. BasePrefix ~2.5K
// approximates a coding agent's system prompt + bash/edit tool schemas; Decode
// and Result are per-turn assistant output and ingested file/test bytes.
func DefaultGeometryModel() GeometryModel {
	return GeometryModel{
		BasePrefix: 2500,
		Decode:     200,
		Result:     400,
		TurnsByDifficulty: map[string]int{
			"<15min":    12,
			"15min-1hr": 22,
			"1-4hr":     38,
			">4hr":      60,
		},
		DefaultTurns: 20,
	}
}

// Derive builds the Geometry for one instance. The prefix is the base preamble
// plus the problem statement's tokens (the problem is part of the persistent,
// cacheable initial context the whole session carries — exactly the context the
// naive arm re-prefills every turn and the fak arm reuses). When the instance
// carries no problem statement (bench's ID-list / difficulty files), only the
// base prefix is used and ProblemTokens is 0.
func (gm GeometryModel) Derive(in Instance) Geometry {
	problem := EstimateTokens(in.ProblemStatement)
	g := Geometry{
		InstanceID:    in.InstanceID,
		Prefix:        gm.BasePrefix + problem,
		Decode:        gm.Decode,
		Result:        gm.Result,
		ProblemTokens: problem,
		Difficulty:    in.Difficulty,
	}
	switch {
	case gm.Trajectories != nil && gm.Trajectories[in.InstanceID] > 0:
		g.Turns = gm.Trajectories[in.InstanceID]
		g.Source = "trajectory"
	case in.Difficulty != "" && gm.TurnsByDifficulty[in.Difficulty] > 0:
		g.Turns = gm.TurnsByDifficulty[in.Difficulty]
		g.Source = "difficulty"
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

// MaxContext is the largest context length this geometry reaches — the initial
// prefix plus T turns of (decode + result) growth. The cost arms need this to
// size the prefill-cost samples that span the whole session.
func (g Geometry) MaxContext() int { return g.Prefix + g.Turns*(g.Decode+g.Result) }
