package turnbench

// stochastic.go — the DISTRIBUTION grounding of the turn-tax headline.
//
// turnbench.Run prices a single frozen trace: the turns_saved it reports is a
// POINT estimate of one hand-authored error mix. The load-bearing assumption of
// the benchmark — "a SOTA agent loop pays +1 model turn per recoverable error"
// — is still a MODELED baseline, and a single point can be cherry-picked. This
// file turns that point into a DISTRIBUTION: it takes a clean BASE workload that
// by construction saves NOTHING (every call is a first-occurrence allow+engine
// round-trip), then injects the four happy-path error classes at configurable
// per-call RATES over many SEEDED trials, and reports p10/p50/p90 of turns_saved.
//
// What is grounded vs. what is modeled (the honest line, unchanged from Run).
// Each individual trial is still scored by the REAL kernel: Perturb only chooses
// WHICH calls to add (a stochastic model of how often a frontier agent aliases /
// re-reads / makes an elidable call), and turnbench.Run replays the resulting
// trace through k.Syscall and reads the kernel's own counters. So the per-trial
// turns_saved is a real kernel event count; the DISTRIBUTION over trials reflects
// the modeled error-rate prior. Both halves stay separable.
//
// Scope (deliberately narrow). Only the four HAPPY-PATH turn-tax classes are in
// the distribution — grammar repair, tier-2 dedup, tier-1 pure, tier-3 static.
// The safety floor (quarantine / deny) is a completion-integrity delta, not a
// turn count, and the IFC sink-gate makes `fetch_policy` order-dependent (it is
// an egress sink once the session is tainted), so poison/destructive calls are
// kept OUT of the stochastic base entirely. See the package doc's two-axes note.
//
// Determinism. Everything uses math/rand seeded once; per-trial seeds are derived
// deterministically from the run seed, so the same (base, rates, trials, seed)
// always yields the identical distribution. This is a normal Go program — a fixed
// seed is the right tool and makes the test stable.

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// RateProfile is the per-eligible-call injection probability for each happy-path
// error class. Each is an independent Bernoulli draw against an eligible base
// call, so a profile is a compact model of "how error-prone is this agent loop".
type RateProfile struct {
	Version    string  `json:"version,omitempty"`
	Name       string  `json:"name,omitempty"`
	AliasRate  float64 `json:"alias_rate"`  // P(inject an aliased convert_currency -> grammar TRANSFORM)
	DupRate    float64 `json:"dup_rate"`    // P(inject a duplicate of an earlier read -> tier-2 dedup hit)
	PureRate   float64 `json:"pure_rate"`   // P(inject a pure calculate call -> tier-1 local serve)
	StaticRate float64 `json:"static_rate"` // P(inject a static list_all_airports call -> tier-3 local serve)
}

// Named profiles: a low / mid / high error-mix ladder. Monotone in every rate so
// the distribution's median is monotone in the profile (the test asserts this).
var (
	// ProfileZero injects nothing — the anti-inflation control (p50 must be 0).
	ProfileZero = RateProfile{Version: BenchmarkConceptVersion, Name: "zero", AliasRate: 0, DupRate: 0, PureRate: 0, StaticRate: 0}
	// ProfileLow is a clean, careful agent: rare aliasing, occasional elidable call.
	ProfileLow = RateProfile{Version: BenchmarkConceptVersion, Name: "low", AliasRate: 0.10, DupRate: 0.10, PureRate: 0.10, StaticRate: 0.05}
	// ProfileMid is a typical SOTA loop.
	ProfileMid = RateProfile{Version: BenchmarkConceptVersion, Name: "mid", AliasRate: 0.30, DupRate: 0.30, PureRate: 0.25, StaticRate: 0.20}
	// ProfileHigh is an error-prone / under-grounded loop.
	ProfileHigh = RateProfile{Version: BenchmarkConceptVersion, Name: "high", AliasRate: 0.55, DupRate: 0.55, PureRate: 0.50, StaticRate: 0.40}
)

// DefaultProfiles is the ladder the stochastic artifact reports, low -> high.
func DefaultProfiles() []RateProfile { return []RateProfile{ProfileLow, ProfileMid, ProfileHigh} }

// BaseTrace is the clean stochastic base workload: every call is a FIRST
// occurrence allow+engine round-trip — no alias, no duplicate, no pure/static
// local serve, no poison, no destructive deny — so by construction it saves
// NOTHING on its own (a zero-rate perturbation has p50 == 0). It also seeds the
// reads that a later duplicate can hit in tier-2: each get_user_details /
// search_direct_flight here is a COLD first occurrence within the trace, so a
// duplicate injected by Perturb into the SAME trace is a genuine tier-2 hit
// (replay bumps the vDSO world per Run, so the first occurrence is always cold).
//
// 14 calls: 6 distinct cold reads + 6 first-occurrence converts (non-aliased,
// canonical args -> plain engine pass) + 2 destructive writes (book_flight). The
// reads are the dedup-eligible anchors; the converts and books are filler that
// stays a "pass" so the base headline is exactly 0.
func BaseTrace() *Trace {
	mk := func(tool string, args map[string]any, ro bool) Call {
		b, _ := json.Marshal(args)
		meta := map[string]string{"readOnlyHint": "false", "idempotentHint": "false", "destructive": "true"}
		if ro {
			meta = map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}
		}
		return Call{Tool: tool, Args: json.RawMessage(b), Meta: meta, Class: "pass"}
	}
	calls := []Call{
		mk("get_user_details", map[string]any{"user_id": "mia_li_3668"}, true),
		mk("search_direct_flight", map[string]any{"origin": "SFO", "destination": "JFK", "date": "2026-07-01"}, true),
		mk("get_user_details", map[string]any{"user_id": "raj_patel_9001"}, true),
		mk("search_direct_flight", map[string]any{"origin": "LAX", "destination": "ORD", "date": "2026-07-02"}, true),
		mk("get_user_details", map[string]any{"user_id": "ana_gomez_5520"}, true),
		mk("search_direct_flight", map[string]any{"origin": "BOS", "destination": "SEA", "date": "2026-07-03"}, true),
		// Non-aliased converts: canonical from_currency/to_currency -> the grammar rung
		// finds nothing to repair, so these are plain engine passes (save nothing). They
		// keep the base ~14 calls without seeding any savings.
		mk("convert_currency", map[string]any{"from_currency": "USD", "to_currency": "EUR", "amount": 100}, true),
		mk("convert_currency", map[string]any{"from_currency": "USD", "to_currency": "GBP", "amount": 200}, true),
		mk("convert_currency", map[string]any{"from_currency": "EUR", "to_currency": "USD", "amount": 300}, true),
		mk("convert_currency", map[string]any{"from_currency": "GBP", "to_currency": "USD", "amount": 400}, true),
		mk("convert_currency", map[string]any{"from_currency": "USD", "to_currency": "JPY", "amount": 500}, true),
		mk("convert_currency", map[string]any{"from_currency": "EUR", "to_currency": "USD", "amount": 600}, true),
		mk("book_flight", map[string]any{"user_id": "mia_li_3668", "flight_id": "UA123"}, false),
		mk("book_flight", map[string]any{"user_id": "raj_patel_9001", "flight_id": "DL456"}, false),
	}
	return &Trace{SliceID: "turntax-stochastic-base", Calls: calls}
}

// readAnchors returns the distinct (tool,args) first-occurrence READS in the base
// — the only calls a later duplicate can hit in tier-2. A duplicate of a write or
// of a pure/static tool is not a dedup event, so only reads are eligible anchors.
func readAnchors(base *Trace) []Call {
	var out []Call
	for _, c := range base.Calls {
		if c.Tool == "get_user_details" || c.Tool == "search_direct_flight" {
			out = append(out, c)
		}
	}
	return out
}

// aliasPair is one alias spelling the grammar rung repairs (from/to & source/target).
type aliasPair struct{ a, b string }

var aliasPairs = []aliasPair{{"from", "to"}, {"source", "target"}}

// Perturb produces a NEW trace from base by injecting the four happy-path error
// classes per the given rates, deterministically driven by rng. The base is never
// mutated. For every eligible base call we make four independent Bernoulli draws:
//
//   - AliasRate  -> append an aliased convert_currency (from/to or source/target)
//     after the call: the grammar rung repairs it in-syscall (a TRANSFORM the
//     baseline would error on and re-prompt for).
//   - DupRate    -> append a duplicate of an EARLIER read anchor (verbatim args):
//     a tier-2 content-cache hit (the cold first occurrence is already in this
//     same trace, so the duplicate is a genuine hit after replay's world bump).
//   - PureRate   -> append a pure calculate{a,b} call: a tier-1 local serve.
//   - StaticRate -> append a static list_all_airports call: a tier-3 local serve.
//
// The injected calls are interleaved right after the call they were drawn for, so
// the duplicate of an anchor always lands AFTER that anchor's cold occurrence.
func Perturb(base *Trace, rates RateProfile, rng *rand.Rand) *Trace {
	anchors := readAnchors(base)
	out := make([]Call, 0, len(base.Calls)*2)

	for i, c := range base.Calls {
		out = append(out, c) // the original base call, unchanged

		// An aliased convert_currency: pick an alias spelling deterministically.
		if rng.Float64() < rates.AliasRate {
			ap := aliasPairs[rng.Intn(len(aliasPairs))]
			amt := 50 + rng.Intn(900)
			args, _ := json.Marshal(map[string]any{ap.a: "USD", ap.b: "EUR", "amount": amt})
			out = append(out, Call{
				Tool: "convert_currency", Args: json.RawMessage(args),
				Meta:  map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
				Class: "grammar", Note: fmt.Sprintf("injected alias %s/%s", ap.a, ap.b),
			})
		}

		// A duplicate of an earlier read anchor -> tier-2 dedup hit. Only fire once at
		// least one anchor has appeared at or before this index (it always has by the
		// time we reach a non-first read, and we additionally guard on len>0).
		if len(anchors) > 0 && rng.Float64() < rates.DupRate {
			// choose among anchors whose cold occurrence is at index <= i so the dup
			// lands strictly after a cold first occurrence in the same trace.
			eligible := anchorsBefore(base, anchors, i)
			if len(eligible) > 0 {
				dup := eligible[rng.Intn(len(eligible))]
				out = append(out, Call{
					Tool: dup.Tool, Args: dup.Args, Meta: dup.Meta,
					Class: "dedup", Note: "injected duplicate read (tier-2 hit)",
				})
			}
		}

		// A pure calculate{a,b} -> tier-1 local serve.
		if rng.Float64() < rates.PureRate {
			a, b := rng.Intn(1000), rng.Intn(1000)
			args, _ := json.Marshal(map[string]any{"a": a, "b": b})
			out = append(out, Call{
				Tool: "calculate", Args: json.RawMessage(args),
				Meta:  map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
				Class: "pure", Note: "injected pure calculate (tier-1)",
			})
		}

		// A static list_all_airports -> tier-3 local serve.
		if rng.Float64() < rates.StaticRate {
			out = append(out, Call{
				Tool: "list_all_airports", Args: json.RawMessage(`{}`),
				Meta:  map[string]string{"readOnlyHint": "true", "idempotentHint": "true"},
				Class: "static", Note: "injected static list_all_airports (tier-3)",
			})
		}
	}
	return &Trace{SliceID: base.SliceID + "-perturbed", Calls: out}
}

// anchorsBefore returns the read anchors whose COLD first occurrence in base is at
// an index <= idx, so a duplicate injected at idx lands after a cold occurrence.
func anchorsBefore(base *Trace, anchors []Call, idx int) []Call {
	// map each anchor to the index of its first occurrence in base.
	firstIdx := func(a Call) int {
		for j, c := range base.Calls {
			if c.Tool == a.Tool && string(c.Args) == string(a.Args) {
				return j
			}
		}
		return len(base.Calls)
	}
	var out []Call
	for _, a := range anchors {
		if firstIdx(a) <= idx {
			out = append(out, a)
		}
	}
	return out
}

// Dist is the order-statistic summary of one metric across the trials.
type Dist struct {
	Min  int     `json:"min"`
	P10  int     `json:"p10"`
	P50  int     `json:"p50"`
	P90  int     `json:"p90"`
	Max  int     `json:"max"`
	Mean float64 `json:"mean"`
}

// distOf computes the order statistics over a slice of samples (samples are not
// required to be pre-sorted). p10/p50/p90 use nearest-rank on the sorted samples.
func distOf(samples []int) Dist {
	if len(samples) == 0 {
		return Dist{}
	}
	s := append([]int(nil), samples...)
	sort.Ints(s)
	pick := func(p float64) int {
		// nearest-rank: index = ceil(p*n)-1, clamped.
		rank := int(p*float64(len(s)) + 0.999999)
		if rank < 1 {
			rank = 1
		}
		if rank > len(s) {
			rank = len(s)
		}
		return s[rank-1]
	}
	sum := 0
	for _, v := range s {
		sum += v
	}
	return Dist{
		Min:  s[0],
		P10:  pick(0.10),
		P50:  pick(0.50),
		P90:  pick(0.90),
		Max:  s[len(s)-1],
		Mean: float64(sum) / float64(len(s)),
	}
}

// ProfileResult is the distribution for one rate profile.
type ProfileResult struct {
	Profile     RateProfile `json:"profile"`
	Trials      int         `json:"trials"`
	TurnsSaved  Dist        `json:"turns_saved"`
	Forced      Dist        `json:"forced"`    // grammar + tier-2 dedup
	Elision     Dist        `json:"elision"`   // tier-1 pure + tier-3 static
	AvgCalls    float64     `json:"avg_calls"` // mean perturbed-trace length (provenance)
	SampleTurns []int       `json:"sample_turns_saved"`
}

// StochasticReport is the full distribution artifact: the base workload identity,
// the cost model, and one ProfileResult per rate profile. By construction a
// zero-rate profile yields TurnsSaved.P50 == 0 (anti-inflation), and a strictly
// higher-rate profile yields a higher median (monotonicity) — both asserted by
// the test.
type StochasticReport struct {
	AppVersion  string          `json:"app_version"`
	SliceID     string          `json:"slice_id"`
	BaseHash    string          `json:"base_workload_hash"`
	BaseCalls   int             `json:"base_calls"`
	Seed        int64           `json:"seed"`
	Trials      int             `json:"trials"`
	Cost        CostModel       `json:"cost_model"`
	GeneratedBy string          `json:"generated_by"`
	Profiles    []ProfileResult `json:"profiles"`
}

// RunStochastic replays `trials` perturbed traces of base through the REAL kernel
// (turnbench.Run, which is self-isolating per call) and summarizes the turns_saved
// distribution for the single given rate profile. The RNG is seeded ONCE from seed
// and a fresh per-trial RNG is derived deterministically (seed + trial index), so
// the whole result is reproducible. forced/elision are the same honest split Run
// reports (Forced = grammar+dedup, Elision = pure+static).
func RunStochastic(ctx context.Context, base *Trace, rates RateProfile, cm CostModel, trials int, seed int64) ProfileResult {
	rates = withRateProfileVersion(rates)
	if trials <= 0 {
		trials = 1
	}
	turns := make([]int, 0, trials)
	forced := make([]int, 0, trials)
	elision := make([]int, 0, trials)
	totalCalls := 0

	// Install the kernel world ONCE (idempotent); scoreTrace, unlike Run, does not
	// call Configure per trial, so each trial is a single calibration-free replay.
	agent.Configure()

	// Derive a deterministic per-trial seed stream from the root RNG so the trial
	// order is fixed and independent of how many draws each Perturb consumes.
	seeds := deriveSeeds(seed, trials)

	for i := 0; i < trials; i++ {
		trng := rand.New(rand.NewSource(seeds[i]))
		pt := Perturb(base, rates, trng)
		totalCalls += len(pt.Calls)
		ts, fc, el, err := scoreTrace(ctx, pt)
		if err != nil {
			// A replay error is fatal to the trial's meaning; record a zero so the
			// distribution is still well-formed but the caller can see avg_calls move.
			turns = append(turns, 0)
			forced = append(forced, 0)
			elision = append(elision, 0)
			continue
		}
		turns = append(turns, ts)
		forced = append(forced, fc)
		elision = append(elision, el)
	}

	pr := ProfileResult{
		Profile:    rates,
		Trials:     trials,
		TurnsSaved: distOf(turns),
		Forced:     distOf(forced),
		Elision:    distOf(elision),
		AvgCalls:   float64(totalCalls) / float64(trials),
	}
	// Keep a small, deterministic sample of the raw turns_saved for the artifact so
	// a reader can see the spread, not just the order statistics (capped at 50).
	n := len(turns)
	if n > 50 {
		n = 50
	}
	pr.SampleTurns = append([]int(nil), turns[:n]...)
	return pr
}

// RunStochasticAll runs RunStochastic for each profile against a SHARED base and
// cost model under one run seed (each profile gets a deterministically-derived
// sub-seed), returning the full artifact. This is what the test marshals to JSON.
func RunStochasticAll(ctx context.Context, base *Trace, profiles []RateProfile, cm CostModel, trials int, seed int64) StochasticReport {
	cm = withCostModelVersion(cm)
	rep := StochasticReport{
		AppVersion:  appversion.Current(),
		SliceID:     base.SliceID,
		BaseHash:    base.WorkloadHash(),
		BaseCalls:   len(base.Calls),
		Seed:        seed,
		Trials:      trials,
		Cost:        cm,
		GeneratedBy: "fak/internal/turnbench (stochastic)",
	}
	// Derive a per-profile sub-seed deterministically so profile order is stable and
	// adding a profile does not perturb the earlier profiles' streams.
	root := rand.New(rand.NewSource(seed))
	for _, p := range profiles {
		sub := root.Int63()
		rep.Profiles = append(rep.Profiles, RunStochastic(ctx, base, p, cm, trials, sub))
	}
	return rep
}

func withRateProfileVersion(p RateProfile) RateProfile {
	if p.Version == "" {
		p.Version = BenchmarkConceptVersion
	}
	return p
}

// JSON renders the stochastic report.
func (r *StochasticReport) JSON() []byte { return marshalArtifact(r) }

// -----------------------------------------------------------------------------
// break-even.go (kept in this file): the hit-rate -> turns-saved -> amortization
// curve. The stochastic ladder above answers "how many turns at THIS error mix";
// this answers the question TURN-TAX-RESULTS.md §3.1 raises and never charts:
// where does the per-session win sit at the REAL-WORLD addressable rate (~0.7%),
// and — given the §3.1 regime build costs — how many sessions amortize it.
//
// The honest line is unchanged. Each point's turns-saved is a REAL-kernel
// stochastic replay (RunStochastic), not a closed form: a single scalar hit-rate
// h drives all four happy-path class rates uniformly, so h is "the fraction of
// eligible calls that are turn-tax-addressable" — exactly the §3.1 vDSO-purity
// axis. The per-turn PRICE and the build-cost amortization are a transparent cost
// model (the same one Run uses); the measured and modeled halves stay separable.
// -----------------------------------------------------------------------------

// BuildRegime is one §3.1 ownership regime priced for amortization: a marginal
// build cost in dollars and how it is reached. Provider-ships is ~$0 (break-even
// on session one); the self-host fork is the honest ~$2.8M/3yr total, NOT the
// $200k knob. An API consumer has NO efficiency axis at all (the turn-saving is
// N/A — they get only the §1 safety floor), so it carries no break-even row.
type BuildRegime struct {
	Name      string  `json:"name"`
	BuildCost float64 `json:"build_cost_usd"` // marginal $ to make the in-process win available
	Note      string  `json:"note"`
}

// DefaultBuildRegimes are the two §3.1 regimes that HAVE an efficiency axis.
func DefaultBuildRegimes() []BuildRegime {
	return []BuildRegime{
		{Name: "provider-ships", BuildCost: 0, Note: "provider exposes in-tensor tool calling — ~$0 marginal, break-even on session one"},
		{Name: "self-host-fork", BuildCost: 2_800_000, Note: "divergent KV-splice fork (~3 eng-yr) + format re-train + fork-maintenance — the honest ~$2.8M/3yr total, not the $200k knob"},
	}
}

// BreakEvenRegime is one regime's amortization at a given hit-rate point.
// SessionsToBreakEven: ceil(build_cost / $saved-per-session). 0 == already
// positive at session one (a ~$0 build cost). NeverAmortizes (-1) is the sentinel
// for "a positive build cost over a zero per-session saving" (e.g. the h=0
// control). The sentinel keeps the artifact valid JSON — json.Marshal rejects
// +Inf — and is unambiguous since a real session count is always >= 0.
type BreakEvenRegime struct {
	Name                string  `json:"name"`
	BuildCost           float64 `json:"build_cost_usd"`
	SessionsToBreakEven float64 `json:"sessions_to_break_even"`
	Note                string  `json:"note"`
}

// NeverAmortizes is the SessionsToBreakEven sentinel for "a positive build cost
// can never be recovered from a zero per-session saving".
const NeverAmortizes = -1.0

// BreakEvenPoint is one hit-rate row: the measured turns-saved distribution at
// rate h over an N-call session, the priced per-session net, and the per-regime
// amortization. h==0 is the anti-inflation control (every field must be 0).
type BreakEvenPoint struct {
	HitRate             float64           `json:"hit_rate"`               // addressable fraction (all four class rates = h)
	TurnsSaved          Dist              `json:"turns_saved"`            // measured distribution over trials (real kernel)
	MeanTurnsSaved      float64           `json:"mean_turns_saved"`       // the expectation the amortization uses
	AvgCalls            float64           `json:"avg_calls"`              // mean perturbed session length (provenance)
	TokensSavedMean     float64           `json:"tokens_saved_mean"`      // per session, at the cost model's prompt+completion size
	DollarsSavedMean    float64           `json:"dollars_saved_mean"`     // per session
	LatencySavedSecMean float64           `json:"latency_saved_sec_mean"` // per session
	Regimes             []BreakEvenRegime `json:"regimes"`                // amortization per §3.1 build regime
}

// BreakEvenReport is the full curve artifact.
type BreakEvenReport struct {
	AppVersion       string           `json:"app_version"`
	SliceID          string           `json:"slice_id"`
	BaseHash         string           `json:"base_workload_hash"`
	BaseCalls        int              `json:"base_calls"`
	Seed             int64            `json:"seed"`
	Trials           int              `json:"trials"`
	RealWorldHitRate float64          `json:"real_world_hit_rate"` // the §3.1 measured tau2-airline vDSO purity (~0.007)
	Cost             CostModel        `json:"cost_model"`
	Regimes          []BuildRegime    `json:"build_regimes"`
	GeneratedBy      string           `json:"generated_by"`
	Points           []BreakEvenPoint `json:"points"`
}

// JSON renders the break-even report. It panics if the report cannot be marshaled
// — a non-finite float would make the artifact silently empty, which once shipped
// a zero-byte file; the sentinel (NeverAmortizes) exists precisely so this never
// happens, and the panic is the belt to that suspenders.
func (r *BreakEvenReport) JSON() []byte {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		panic("turnbench: break-even report not marshalable (non-finite float?): " + err.Error())
	}
	return append(b, '\n')
}

// DefaultHitRateGrid spans the §3.1 real-world rate (~0.7%) up through the
// cache-favorable slice, with 0 as the anti-inflation control. The log-ish spacing
// keeps the low end (where the real answer lives) legible.
func DefaultHitRateGrid() []float64 {
	return []float64{0, 0.007, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50}
}

// realWorldHitRate is the §3.1 measured addressable vDSO purity on real
// tau2-airline (~0.7%) — the number the curve exists to give an honest home.
const realWorldHitRate = 0.007

// uniformProfile builds a RateProfile whose four class rates are all h, so a
// single scalar sweeps "fraction of eligible calls that are turn-tax-addressable".
func uniformProfile(h float64) RateProfile {
	return RateProfile{
		Version: BenchmarkConceptVersion, Name: "h",
		AliasRate: h, DupRate: h, PureRate: h, StaticRate: h,
	}
}

// RunBreakEvenSweep replays the clean base at each hit-rate in grid (real-kernel
// stochastic, `trials` seeded trials per point) and prices the per-session net +
// per-regime amortization. Determinism mirrors RunStochasticAll: a per-point
// sub-seed is derived from the run seed so adding a grid point never perturbs the
// earlier points' streams. A nil/empty grid uses DefaultHitRateGrid; nil regimes
// use DefaultBuildRegimes.
func RunBreakEvenSweep(ctx context.Context, base *Trace, grid []float64, regimes []BuildRegime, cm CostModel, trials int, seed int64) BreakEvenReport {
	cm = withCostModelVersion(cm)
	if len(grid) == 0 {
		grid = DefaultHitRateGrid()
	}
	if len(regimes) == 0 {
		regimes = DefaultBuildRegimes()
	}
	rep := BreakEvenReport{
		AppVersion:       appversion.Current(),
		SliceID:          base.SliceID,
		BaseHash:         base.WorkloadHash(),
		BaseCalls:        len(base.Calls),
		Seed:             seed,
		Trials:           trials,
		RealWorldHitRate: realWorldHitRate,
		Cost:             cm,
		Regimes:          regimes,
		GeneratedBy:      "fak/internal/turnbench (break-even)",
	}
	root := rand.New(rand.NewSource(seed))
	for _, h := range grid {
		sub := root.Int63()
		pr := RunStochastic(ctx, base, uniformProfile(h), cm, trials, sub)
		mean := pr.TurnsSaved.Mean
		tokens := mean * float64(cm.tokensPerTurn())
		dollars := mean * cm.dollarsPerTurn()
		latSec := mean * cm.ModelTurnLatencyMs / 1000.0

		pt := BreakEvenPoint{
			HitRate:             h,
			TurnsSaved:          pr.TurnsSaved,
			MeanTurnsSaved:      mean,
			AvgCalls:            pr.AvgCalls,
			TokensSavedMean:     tokens,
			DollarsSavedMean:    dollars,
			LatencySavedSecMean: latSec,
		}
		for _, rg := range regimes {
			var sessions float64
			switch {
			case rg.BuildCost <= 0:
				sessions = 0 // already positive at session one
			case dollars <= 0:
				sessions = NeverAmortizes // a zero-rate point never amortizes a positive build cost
			default:
				sessions = math.Ceil(rg.BuildCost / dollars)
			}
			pt.Regimes = append(pt.Regimes, BreakEvenRegime{
				Name: rg.Name, BuildCost: rg.BuildCost, SessionsToBreakEven: sessions, Note: rg.Note,
			})
		}
		rep.Points = append(rep.Points, pt)
	}
	return rep
}
