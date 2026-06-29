// bench.go is the measured-numbers half of polymodelbench — the runnable artifact
// for issue #535 ("bench(polymodel): measured numbers for the poly-model lane").
//
// The plan (docs/serving/polymodel-prefill-share-plan.md §7 item 6) requires that
// every poly-model speedup claim be gated on a MEASURED run, not a closed-form
// assumption. The three measured axes it names are:
//
//  1. SPECULATION — E (real tokens per target verify) vs draft cost. The acceptance
//     rate is MEASURED by running a real draft/target pair through specDecodeModel
//     (real Session.Step, real AcceptGreedy, real bit-exact KVCache.Evict rollback);
//     E is then the closed-form EffectiveTokensPerVerify evaluated AT THAT measured
//     acceptance. The run is gated lossless (spec == greedy) or the report is invalid.
//  2. DECODE LANE — utilization of the single serial decode lane: decode tokens
//     emitted, decode steps, tokens-per-step, and MaxConcurrentDecode (the
//     load-bearing ==1 invariant), measured off a real Schedule plan + the real
//     per-model weight footprints via DecodeBandwidthBytes.
//  3. RESIDENCY — multi-model hit-rate: a request stream with a hot working set
//     and a cold tail driven against a real Pool under a tight budget; every request
//     is a real Pool.Admit, classified hit (model already warm → Touch path) vs cold
//     (load + LRU eviction). The hit-rate, eviction count, and pinned-survival are
//     the measured output of the Pool's actual LRU behavior.
//
// Honest fence: these are DETERMINISTIC SYNTHETIC-WORKLOAD measurements on the real
// polymodel bookkeeping primitives. There is no GPU here, so no tokens/sec-on-hardware
// claim is made — only the acceptance rate, lane throughput, and hit-rate that the
// policy/accounting core actually produces, which is what every speedup claim reduces
// to. Reproduce: `polymodelbench -bench` (prints) or `-bench -out report.json`.
package main

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/benchcli"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// BenchReport is the reproducible artifact emitted by -bench / -out. Every numeric
// field is MEASURED by driving the real polymodel primitives over deterministic
// synthetic workloads; none is a target, an assumption, or a hardware tokens/sec.
type BenchReport struct {
	Issue       string         `json:"issue"`
	Speculation SpecBench      `json:"speculation"`
	DecodeLane  LaneBench      `json:"decode_lane"`
	Residency   ResidencyBench `json:"residency"`
	HonestFence string         `json:"honest_fence"`
}

// SpecBench is the E-vs-draft-cost axis, reported as the MEASURED SPREAD of acceptance
// across two real regimes, not a single curated point: the realistic real-draft regime
// (a co-resident draft model proposes) brackets E from above; the adversarial regime
// (a proposer that forces rejections) brackets E from below AND proves the bit-exact
// Evict rollback path actually ran (EvictedKVAdv > 0, else the witness is vacuous).
// E is the closed-form EffectiveTokensPerVerify evaluated at each MEASURED acceptance.
type SpecBench struct {
	DraftK          int        `json:"draft_k"`
	BaselineENoSpec float64    `json:"baseline_e_no_spec"` // plain greedy: 1 real token / target forward
	RealDraft       specRegime `json:"real_draft"`         // realistic upper bound on E
	Adversarial     specRegime `json:"adversarial"`        // rollback-stress lower bound on E
	Lossless        bool       `json:"lossless"`           // BOTH regimes == greedy (the gate)
}

// specRegime is one measured speculative-decode regime: its acceptance rate, the
// closed-form E at that rate, and the draft tokens spent per real token emitted.
type specRegime struct {
	AcceptanceRate   float64 `json:"acceptance_rate"` // MEASURED: accepted/drafted
	EffectiveE       float64 `json:"effective_e"`     // E(K, measured acc)
	DraftedTokens    int     `json:"drafted_tokens"`
	AcceptedTokens   int     `json:"accepted_tokens"`
	EvictedKVSpans   int     `json:"evicted_kv_spans"`          // bit-exact Evict rollbacks performed
	DraftCostPerReal float64 `json:"draft_cost_per_real_token"` // drafted / real emitted
}

// LaneBench is the serial decode-lane utilization axis, measured off a real Schedule.
type LaneBench struct {
	ModelsOnLane  int     `json:"models_on_lane"`
	Quantum       int     `json:"quantum"`
	PrefillTokens int     `json:"prefill_tokens"`
	DecodeTokens  int     `json:"decode_tokens"`
	DecodeSteps   int     `json:"decode_steps"`
	MaxConcurrent int     `json:"max_concurrent_decode"` // invariant: 1
	TokensPerStep float64 `json:"tokens_per_step"`       // MEASURED lane throughput
	DecodeHBMKB   int64   `json:"decode_hbm_traffic_kb"` // MEASURED via DecodeBandwidthBytes
}

// ResidencyBench is the multi-model hit-rate axis, measured off a real Pool workload.
type ResidencyBench struct {
	DistinctModels int     `json:"distinct_models"`
	BudgetKB       int64   `json:"budget_kb"`
	Requests       int     `json:"requests"`
	ReAdmitHits    int     `json:"readmit_hits"` // requests that found the model already warm
	ColdAdmits     int     `json:"cold_admits"`  // requests that loaded a cold model
	Evictions      int     `json:"evictions"`
	HitRate        float64 `json:"hit_rate"` // MEASURED: hits / (hits + cold)
	FinalWarm      int     `json:"final_warm"`
	PinnedSurvived bool    `json:"pinned_survived"`
}

// benchHarness drives the three measured workloads and returns the report. It
// re-asserts the correctness invariants (losslessness, MaxConcurrentDecode==1,
// pinned-survives, budget-never-exceeded) ON THE MEASURED RUNS and lowers *ok if any
// fails — a measured number on a broken core is worthless, so the report is only as
// good as the gate. quiet suppresses the per-axis log lines (used under -selfcheck).
func benchHarness(quiet bool, ok *bool) BenchReport {
	spec := measureSpec(quiet, ok)
	lane := measureLane(quiet, ok)
	res := measureResidency(quiet, ok)
	r := BenchReport{
		Issue:       "bench(polymodel): measured numbers for the poly-model lane (#535)",
		Speculation: spec,
		DecodeLane:  lane,
		Residency:   res,
		HonestFence: "Deterministic synthetic-workload measurements on the real polymodel " +
			"primitives (Pool, Schedule, AcceptGreedy, KVCache.Evict). No GPU, so no " +
			"tokens/sec-on-hardware claim — only the acceptance rate, serial-lane " +
			"throughput, and residency hit-rate the policy core actually produces. " +
			"Reproduce: polymodelbench -bench [-out FILE].",
	}
	logf(quiet, "")
	logf(quiet, "== #535 bench harness: measured poly-model numbers ==")
	logf(quiet, "  speculation : E(K=%d) spread over measured acceptance — real-draft acc=%.3f E=%.2f (draft-cost/real=%.2f), adversarial acc=%.3f E=%.2f (%d KV rollbacks); baseline E=%.0f; lossless=%v",
		spec.DraftK, spec.RealDraft.AcceptanceRate, spec.RealDraft.EffectiveE, spec.RealDraft.DraftCostPerReal, spec.Adversarial.AcceptanceRate, spec.Adversarial.EffectiveE, spec.Adversarial.EvictedKVSpans, spec.BaselineENoSpec, spec.Lossless)
	logf(quiet, "  decode lane : %d models, %d decode steps, %.2f tokens/step, max-concurrent-decode=%d, HBM=%dKB",
		lane.ModelsOnLane, lane.DecodeSteps, lane.TokensPerStep, lane.MaxConcurrent, lane.DecodeHBMKB)
	logf(quiet, "  residency  : %d reqs -> %d hits / %d cold, hit-rate=%.3f, %d evictions, final-warm=%d, pinned-survived=%v",
		res.Requests, res.ReAdmitHits, res.ColdAdmits, res.HitRate, res.Evictions, res.FinalWarm, res.PinnedSurvived)
	return r
}

// measureSpec runs BOTH speculative-decode regimes against the same target and measures
// each one's acceptance rate, E, and draft cost. Losslessness (spec == greedy on the
// same target) is the correctness gate for BOTH regimes, and the adversarial regime
// must actually roll back KV spans (EvictedKVAdv > 0) or the rollback-stress witness is
// vacuous — either failure lowers *ok and invalidates the report.
func measureSpec(quiet bool, ok *bool) SpecBench {
	const N, K = 24, 4
	target := model.NewSynthetic(cfg(64, 4, 4, 2, 16, 128))
	prompt := bytesToIDs([]byte("speculative decoding is lossless when verified greedily"))

	want := greedyDecode(target, prompt, N)
	draft := model.NewSynthetic(cfg(32, 2, 2, 1, 16, 64)) // cheaper, different weights
	gotA, draftedA, acceptedA, evictedA := specDecodeModel(target, draft, prompt, N, K)
	adversary := func(round, j, last int) int { return (round*13 + j*7 + 1) % 256 }
	gotB, draftedB, acceptedB, evictedB := specDecodeProposer(target, prompt, N, K, adversary)

	lossless := losslessEqual(gotA, want, N) && losslessEqual(gotB, want, N)
	if !lossless {
		fmt.Fprintln(os.Stderr, "  BENCH GATE FAILED: speculative decode != greedy on the measured run (bit-exact KV rollback regressed) — report invalid")
		*ok = false
	}
	if evictedB == 0 {
		fmt.Fprintln(os.Stderr, "  BENCH GATE FAILED: adversarial regime caused 0 KV rollbacks — the Evict path was never exercised (vacuous witness)")
		*ok = false
	}

	return SpecBench{
		DraftK:          K,
		BaselineENoSpec: 1.0,
		RealDraft:       makeSpecRegime(draftedA, acceptedA, evictedA, K, gotA),
		Adversarial:     makeSpecRegime(draftedB, acceptedB, evictedB, K, gotB),
		Lossless:        lossless,
	}
}

// losslessEqual is the spec == greedy check over the first n tokens (the correctness
// gate for a measured speculative-decode regime).
func losslessEqual(got, want []int, n int) bool {
	if len(got) < n || len(want) < n {
		return false
	}
	for i := 0; i < n; i++ {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// makeSpecRegime folds a regime's measured raw counts into its acceptance rate,
// closed-form E (at the regime's per-round draft length k), and draft-cost-per-real.
func makeSpecRegime(drafted, accepted, evicted, k int, out []int) specRegime {
	r := specRegime{
		DraftedTokens:  drafted,
		AcceptedTokens: accepted,
		EvictedKVSpans: evicted,
		AcceptanceRate: rate(accepted, drafted),
	}
	r.EffectiveE = polymodel.EffectiveTokensPerVerify(k, r.AcceptanceRate)
	if len(out) > 0 {
		r.DraftCostPerReal = float64(drafted) / float64(len(out))
	}
	return r
}

// measureLane measures the serial decode-lane plan over the same 3-model workload the
// decodeOne witness proves correct, so the bench number corresponds 1:1 to the witness.
// tokens/step is the real lane throughput; MaxConcurrentDecode==1 is the invariant.
func measureLane(quiet bool, ok *bool) LaneBench {
	names := []string{"a", "b", "c"}
	cfgs := map[string]model.Config{
		"a": cfg(48, 3, 3, 1, 16, 96),
		"b": cfg(32, 2, 2, 1, 16, 64),
		"c": cfg(64, 4, 4, 2, 16, 128),
	}
	prompt := bytesToIDs([]byte("the cache is the lever"))
	weights := map[polymodel.ModelID]int64{}
	var reqs []polymodel.Request
	for i, n := range names {
		m := model.NewSynthetic(cfgs[n])
		m.NewSession().Prefill(prompt) // prefill-warm, matching the witness
		id := polymodel.ModelID(n)
		weights[id] = estimateBytes(cfgs[n])
		reqs = append(reqs, polymodel.Request{Model: id, Prefill: len(prompt), Decode: 4, Priority: 3 - i, Seq: uint64(i)})
	}
	const quantum = 2
	steps, st := polymodel.Schedule(reqs, quantum)
	if st.MaxConcurrentDecode != 1 {
		fmt.Fprintf(os.Stderr, "  BENCH GATE FAILED: MaxConcurrentDecode=%d, want 1\n", st.MaxConcurrentDecode)
		*ok = false
	}
	bw := polymodel.DecodeBandwidthBytes(steps, weights)
	tps := 0.0
	if st.DecodeSteps > 0 {
		tps = float64(st.DecodeTokens) / float64(st.DecodeSteps)
	}
	return LaneBench{
		ModelsOnLane:  len(names),
		Quantum:       quantum,
		PrefillTokens: st.PrefillTokens,
		DecodeTokens:  st.DecodeTokens,
		DecodeSteps:   st.DecodeSteps,
		MaxConcurrent: st.MaxConcurrentDecode,
		TokensPerStep: tps,
		DecodeHBMKB:   bw / 1024,
	}
}

// measureResidency drives a real Pool through a request stream with a hot working set
// (m0..m3, touched frequently) and a cold tail (m4..m9, touched rarely) under a budget
// that forces eviction. Every request is a real Pool.Admit; a hit (model already warm)
// takes the Touch path with no eviction, a cold miss loads the model and evicts the
// LRU victim. The hit-rate, eviction count, and pinned-survival are the Pool's actual
// measured output — no assumption about working-set size is baked in.
func measureResidency(quiet bool, ok *bool) ResidencyBench {
	specs := modelZoo(10)
	var total int64
	for _, s := range specs {
		total += s.bytes
	}
	budget := total * 50 / 100 // ~half fit → the hot set mostly stays warm, the cold tail evicts
	pinned := smallest(specs)
	pool := polymodel.NewPool(budget)
	byName := map[string]modelSpec{}
	for _, s := range specs {
		byName[s.name] = s
	}
	// Cold-load seed: admit every model once (many evict immediately). This is the
	// initial residency state, not part of the hit-rate measurement.
	for _, s := range specs {
		pool.Admit(polymodel.Model{
			ID: polymodel.ModelID(s.name), Family: "synthetic",
			WeightBytes: s.bytes, Pinned: s.name == pinned,
		})
	}

	// Deterministic measured workload: 10 batches of (4 hot + 1 cold).
	hot := []string{"m0", "m1", "m2", "m3"}
	coldTail := []string{"m4", "m5", "m6", "m7", "m8", "m9"}
	var stream []string
	for b := 0; b < 10; b++ {
		for j := 0; j < 4; j++ {
			stream = append(stream, hot[(b+j)%4])
		}
		stream = append(stream, coldTail[b%len(coldTail)])
	}

	hits, cold, evictions := 0, 0, 0
	for _, name := range stream {
		spec := byName[name]
		id := polymodel.ModelID(name)
		if pool.Has(id) {
			hits++ // model already warm → Admit takes the Touch path
		} else {
			cold++ // not resident → Admit loads + LRU-evicts
		}
		evicted, err := pool.Admit(polymodel.Model{
			ID: id, Family: "synthetic", WeightBytes: spec.bytes, Pinned: name == pinned,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "  BENCH GATE FAILED: admit %s: %v\n", name, err)
			*ok = false
			continue
		}
		evictions += len(evicted)
		if pool.Used() > pool.Budget() {
			fmt.Fprintf(os.Stderr, "  BENCH GATE FAILED: used %d > budget %d after %s\n", pool.Used(), pool.Budget(), name)
			*ok = false
		}
	}
	pinnedSurvived := pool.Has(polymodel.ModelID(pinned))
	if !pinnedSurvived {
		fmt.Fprintf(os.Stderr, "  BENCH GATE FAILED: pinned model %s was evicted\n", pinned)
		*ok = false
	}
	return ResidencyBench{
		DistinctModels: len(specs),
		BudgetKB:       budget / 1024,
		Requests:       len(stream),
		ReAdmitHits:    hits,
		ColdAdmits:     cold,
		Evictions:      evictions,
		HitRate:        rate(hits, hits+cold),
		FinalWarm:      pool.Len(),
		PinnedSurvived: pinnedSurvived,
	}
}

// writeJSON writes the report as indented JSON — the reproducible artifact for -out.
func writeJSON(path string, r BenchReport) error {
	b, err := benchcli.MarshalReport(r)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}
