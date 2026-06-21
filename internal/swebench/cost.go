package swebench

import "sort"

// This file holds the CONTENTION-FREE half of the harness-cost comparison: the
// exact prefill-token work each arm processes, derived purely from the session
// structure (P,T,D,R) and worker count C. It is timing-free arithmetic — the
// deterministic floor under the measured wall-clock ratios — so it runs on this
// Mac with NO model, NO GPU, and cannot drift with machine load. It mirrors
// sessionbench's prefillTokens formula exactly (the established fak value-stack
// methodology), applied per SWE-bench instance instead of one synthetic cell.
// The LIVE timed arms (real prefill/decode wall-clock) live alongside and need
// the model engine; this part is the honest headline available offline.
//
// The three arms (same as the value stack):
//
//	A naive-stateless — re-prefill the WHOLE context every turn, ×C workers.
//	    quadratic in T: C · Σ_{t=0..T-1}(P + t·(D+R))
//	B per-agent-KV    — prefix prefilled once per worker + incremental result
//	    ingestion:        C · (P + (T-1)·R)
//	C fak fused        — prefix prefilled ONCE total (shared across workers) +
//	    incremental:           P + C·(T-1)·R
//
// A→B is the turn-tax (KV persistence vs re-prefill, shows even at C=1).
// B→C is cross-worker prefix reuse (the value stack; only bites at C>1 — exactly
// bench's "mini-workers-sweep" where C workers share the system+tool preamble).

// PrefillTokens returns the exact prefill-token counts (a,b,c) one instance's
// geometry implies for C workers. Pure arithmetic; no timing.
func PrefillTokens(g Geometry, workers int) (a, b, c int) {
	if workers < 1 {
		workers = 1
	}
	P, T, D, R := g.Prefix, g.Turns, g.Decode, g.Result
	for t := 0; t < T; t++ {
		a += P + t*(D+R)
	}
	a *= workers
	inc := 0
	if T > 1 {
		inc = (T - 1) * R
	}
	b = workers * (P + inc)
	c = P + workers*inc
	return
}

// PrefillAgg is the aggregate prefill-token work-elimination across a set of
// instances at a fixed worker count — the headline value-stack number for the
// SWE-bench Verified set, deterministic and machine-load-independent.
type PrefillAgg struct {
	Workers   int   `json:"workers"`
	Instances int   `json:"instances"`
	A         int64 `json:"a_naive_prefill_tokens"`     // re-prefill whole context every turn ×C
	B         int64 `json:"b_per_agent_prefill_tokens"` // prefix ×C + incremental
	C         int64 `json:"c_fak_prefill_tokens"`       // prefix once + incremental
	// All three ratios below are fak-vs-fak ablation arms, NOT an external comparator:
	// a tuned SGLang/llama server (seq_cp/kv_unified) also reuses a shared prefix.
	AOverC float64 `json:"a_over_c"`          // fak naive-re-prefill arm (A) vs fak-fused arm (C) — timing-free
	BOverC float64 `json:"b_over_c"`          // fak per-agent-KV arm (B) vs fak-fused shared-prefix arm (C) — the value-stack lever
	AOverB float64 `json:"a_over_b_turn_tax"` // turn-tax: re-prefill (A) vs KV persistence (B), worker-independent
}

// AggregatePrefill sums the exact prefill-token work across geoms at C workers.
func AggregatePrefill(geoms []Geometry, workers int) PrefillAgg {
	agg := PrefillAgg{Workers: workers, Instances: len(geoms)}
	for _, g := range geoms {
		a, b, c := PrefillTokens(g, workers)
		agg.A += int64(a)
		agg.B += int64(b)
		agg.C += int64(c)
	}
	if agg.C > 0 {
		agg.AOverC = float64(agg.A) / float64(agg.C)
		agg.BOverC = float64(agg.B) / float64(agg.C)
	}
	if agg.B > 0 {
		agg.AOverB = float64(agg.A) / float64(agg.B)
	}
	return agg
}

// Summary describes a SWE-bench Verified dataset + its derived geometry: the
// difficulty distribution, geometry-source provenance (how many instances used a
// real trajectory vs a difficulty-bucket estimate vs the flat default), turn
// statistics, and the deterministic prefill-token work-elimination at a sweep of
// worker counts. This is what `fak swebench describe` reports — a real,
// offline-computable preview of the value stack on the actual instance set.
type Summary struct {
	Instances       int            `json:"instances"`
	DifficultyDist  map[string]int `json:"difficulty_distribution"`
	GeometrySources map[string]int `json:"geometry_sources"`
	TurnsMin        int            `json:"turns_min"`
	TurnsMedian     int            `json:"turns_median"`
	TurnsMax        int            `json:"turns_max"`
	TotalTurns      int64          `json:"total_turns"`
	Prefill         []PrefillAgg   `json:"prefill_work_elimination"` // one entry per worker count in the sweep
}

// Describe builds the Summary for a dataset under a geometry model, computing the
// prefill work-elimination at each worker count in workerSweep (defaults to
// {1,2,4,8} when empty).
func Describe(d *Dataset, gm GeometryModel, workerSweep []int) Summary {
	if len(workerSweep) == 0 {
		workerSweep = []int{1, 2, 4, 8}
	}
	if d == nil {
		d = NewDataset(nil)
	}
	geoms := gm.DeriveAll(d)
	s := Summary{
		Instances:       len(geoms),
		DifficultyDist:  map[string]int{},
		GeometrySources: map[string]int{},
	}
	turns := make([]int, 0, len(geoms))
	for _, g := range geoms {
		diff := g.Difficulty
		if diff == "" {
			diff = "unknown"
		}
		s.DifficultyDist[diff]++
		s.GeometrySources[g.Source]++
		turns = append(turns, g.Turns)
		s.TotalTurns += int64(g.Turns)
	}
	if len(turns) > 0 {
		sort.Ints(turns)
		s.TurnsMin = turns[0]
		s.TurnsMax = turns[len(turns)-1]
		s.TurnsMedian = turns[len(turns)/2]
	}
	for _, w := range workerSweep {
		s.Prefill = append(s.Prefill, AggregatePrefill(geoms, w))
	}
	return s
}
