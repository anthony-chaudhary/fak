package frontierswe

// This file is the deterministic, model-free time-to-solution (TTS) floor for
// FrontierSWE — the value-thesis keystone of epic #1706. It is the FrontierSWE-
// regime analogue of internal/swebench's A/B/C prefill-work model
// (swebench.DefaultGeometryModel / AggregatePrefill), but shaped for FrontierSWE's
// regime instead of SWE-bench's: a single long-horizon trajectory of THOUSANDS of
// turns whose resident context grows to hundreds of thousands of tokens, rather
// than a short fix replayed across a worker sweep.
//
// Why a separate model. A FrontierSWE trial is a ~20-hour wall-clock budget
// (task.toml [agent] timeout_sec = 72000.0) dominated by the agent loop, and the
// agent loop's dominant cost is RE-PREFILLING a growing context every turn: at
// turn k a naive harness re-ingests the codebase + running diff + accumulated
// tool output, work that grows ~linearly in k, so its integral over T turns is
// quadratic in T. That quadratic re-prefill integral is exactly the N-axis work
// fak eliminates with KV persistence + cross-turn prefix reuse. This model is the
// number that says "of the budget, X% is re-prefill work the value stack removes,
// so projected TTS at reuse rate r is a known function of (T_raw, turns, context
// growth, r)".
//
// The three work arms (same house A/B/C shape as swebench, single-agent so there
// is no worker multiplier C — the cross-axis lever here is the CROSS-TURN reuse
// rate r, not cross-worker prefix sharing):
//
//	A naive re-prefill-every-turn — the FrontierSWE-harness default: re-ingest the
//	    WHOLE resident context every turn.  Σ_{t=0..T-1}(P + t·(D+R)) — quadratic in T.
//	B tuned per-agent KV — prefix prefilled once, then only the incremental tool
//	    result ingested each turn:  P + (T-1)·R — linear in T.
//	C fak cross-turn reuse at reuse rate r — interpolates between A (r=0, nothing
//	    reused, fall back to re-prefill) and B (r=1, every reusable prefill turn is
//	    served from the persistent KV / radix prefix):  C(r) = A − r·(A − B).
//
// Derived ratios (the headline value numbers, all deterministic):
//
//	A/C(r) net prefill work-elimination — how much total re-prefill work the value
//	    stack removes at reuse rate r (=1 at r=0, rises to A/B at r=1).
//	A/B   the turn-tax — the structural cost of re-prefilling vs persisting KV,
//	    independent of r (the ceiling A/C(r) climbs toward).
//	TTSRatio(r) projected time-to-solution ratio T_fak/T_raw — the fraction of the
//	    raw budget fak's trajectory takes, under the floor assumption that
//	    prefill work maps linearly to wall-clock.  = C(r)/A = 1 − r·(1 − B/A).
//
// WITNESSED vs PROJECTION (the honesty boundary this model is required to state).
//   - WITNESSED-by-construction: the work integrals A and B, and therefore A/B and
//     the SHAPE C(r) = A − r·(A − B). These are exact arithmetic over the declared
//     (P,T,D,R) geometry — no model, no GPU, no timing; they cannot drift with
//     machine load and are reproduced bit-for-bit by the test.
//   - PROJECTION: the REALIZED reuse rate r an actual fak run achieves, and the
//     assumption that prefill work maps linearly to wall-clock. r is a free dial
//     here; pinning it to a measured value (and validating the work→time map) is a
//     LIVE measurement deferred to C8/C14. Until then TTSRatio(r) is a floor/curve,
//     NOT a measurement, and is labeled as such wherever it surfaces.

// TaskGeometry is one FrontierSWE task's per-turn token geometry — the long-horizon
// analogue of swebench.Geometry. Prefix is the initial resident context (system
// prompt + tool schemas + the repo/instruction snapshot the trajectory carries);
// each of Turns round-trips decodes Decode assistant tokens and ingests Result
// tool-result tokens, so the resident context grows by (Decode+Result) per turn.
type TaskGeometry struct {
	Name   string `json:"name"`   // task name (e.g. "git-to-zig"); informational
	Prefix int    `json:"prefix"` // P: initial resident context, persistent across turns
	Turns  int    `json:"turns"`  // T: model round-trips over the long-horizon trajectory
	Decode int    `json:"decode"` // D: assistant tokens decoded per turn
	Result int    `json:"result"` // R: tool-result tokens ingested per turn
}

// MaxContext is the largest resident context the trajectory reaches — the initial
// prefix plus T turns of (decode + result) growth. For a long-horizon FrontierSWE
// task this is the hundreds-of-thousands-of-tokens figure the re-prefill integral
// is taken over.
func (g TaskGeometry) MaxContext() int { return g.Prefix + g.Turns*(g.Decode+g.Result) }

// TTSModel maps a FrontierSWE task's geometry to the A/B/C re-prefill work arms and
// the reuse-rate-parameterised TTS curve. It carries no per-task default turn table
// (unlike swebench.GeometryModel) because FrontierSWE turn counts come from the
// task's own long-horizon trajectory, supplied directly on TaskGeometry; the model
// is the deterministic arithmetic over that geometry. It is offline and model-free,
// exactly like swebench.DefaultGeometryModel().
type TTSModel struct{}

// DefaultTTSModel returns the standard model. It holds no knobs today — the
// arithmetic is fixed and the only free dial is the reuse rate r passed to Derive /
// TTSRatio — but the constructor mirrors swebench.DefaultGeometryModel() so the two
// value-stack floors are discovered and used the same way.
func DefaultTTSModel() TTSModel { return TTSModel{} }

// WorkArms is the deterministic re-prefill work decomposition for one task geometry
// at a given cross-turn reuse rate. A, B, and C are token-work integrals (the
// WITNESSED part); the ratios are the headline value numbers derived from them.
type WorkArms struct {
	Name      string  `json:"name"`
	Reuse     float64 `json:"reuse_rate"`             // r in [0,1] used for the C arm and TTS ratio (PROJECTION dial)
	A         int64   `json:"a_naive_prefill_tokens"` // re-prefill whole context every turn (quadratic in T)
	B         int64   `json:"b_per_agent_kv_tokens"`  // prefix once + incremental result ingest (linear in T)
	C         float64 `json:"c_fak_reuse_tokens"`     // C(r) = A − r·(A − B): fak cross-turn reuse at rate r
	AOverC    float64 `json:"a_over_c"`               // net prefill work-elimination at reuse rate r
	AOverB    float64 `json:"a_over_b_turn_tax"`      // turn-tax: re-prefill vs KV persistence, r-independent
	TTSRatio  float64 `json:"tts_ratio"`              // projected T_fak/T_raw = C(r)/A (floor; PROJECTION)
	MaxTokens int     `json:"max_context_tokens"`     // resident context at the end of the trajectory
}

// PrefillWork returns the exact naive (A) and per-agent-KV (B) prefill-token
// integrals for one task geometry. Pure arithmetic, no timing, no r — these are the
// WITNESSED-by-construction floor the reuse curve is built on.
//
//	A = Σ_{t=0..T-1}(P + t·(D+R))   — naive re-prefill every turn, quadratic in T
//	B = P + (T-1)·R                 — prefix once + incremental result ingest, linear in T
func PrefillWork(g TaskGeometry) (a, b int64) {
	P, T, D, R := int64(g.Prefix), int64(g.Turns), int64(g.Decode), int64(g.Result)
	if T < 0 {
		T = 0
	}
	for t := int64(0); t < T; t++ {
		a += P + t*(D+R)
	}
	inc := int64(0)
	if T > 1 {
		inc = (T - 1) * R
	}
	b = P + inc
	return
}

// Derive computes the A/B/C work arms and derived ratios for one task at cross-turn
// reuse rate r. r is clamped to [0,1]: at r=0 fak reuses nothing and the C arm
// collapses to the naive A integral (TTS ratio 1.0 — no speedup); at r=1 every
// reusable re-prefill turn is served from the persistent KV / radix prefix and C
// reaches the per-agent-KV floor B (TTS ratio B/A — the full turn-tax removed).
// Intermediate r linearly interpolates: C(r) = A − r·(A − B).
func (TTSModel) Derive(g TaskGeometry, r float64) WorkArms {
	if r < 0 {
		r = 0
	}
	if r > 1 {
		r = 1
	}
	a, b := PrefillWork(g)
	w := WorkArms{
		Name:      g.Name,
		Reuse:     r,
		A:         a,
		B:         b,
		MaxTokens: g.MaxContext(),
	}
	// C(r) = A − r·(A − B): A is the ceiling (no reuse), B is the floor (full reuse).
	w.C = float64(a) - r*float64(a-b)
	if w.C > 0 {
		w.AOverC = float64(a) / w.C
		w.TTSRatio = w.C / float64(a)
	}
	if b > 0 {
		w.AOverB = float64(a) / float64(b)
	}
	return w
}

// TTSRatio is the projected time-to-solution ratio T_fak/T_raw at reuse rate r — the
// fraction of the raw 20-hour budget fak's trajectory is projected to take, under
// the floor assumption that prefill work maps linearly to wall-clock. It is a closed
// form of Derive's TTSRatio field:
//
//	TTSRatio(r) = C(r)/A = 1 − r·(1 − B/A)
//
// monotone decreasing in r from 1.0 (r=0, no reuse, no speedup) down to B/A (r=1,
// the full turn-tax removed). This is the PROJECTION curve; the realized r is the
// C8/C14 measurement.
func TTSRatio(g TaskGeometry, r float64) float64 {
	return DefaultTTSModel().Derive(g, r).TTSRatio
}
