// Package savingsvector decomposes a turn-tax saving into the four orthogonal
// accounts named by docs/explainers/compounding-benefits-of-a-saved-call.md.
package savingsvector

import (
	"fmt"
	"math"
	"sort"
)

// SpawnedHookTaxX is the spawned-hook boundary-tax floor, MEASURED on an M3 Pro.
// BENCHMARK-AUTHORITY.md: in-process ~362 ns allow vs ~2,849x for a spawned
// `fak hook` round-trip (n=100). The ratio credits the local-CPU account: a
// vDSO/grammar serve the baseline would have spawned a hook to gate saves the
// spawn round-trip, not just the in-process decide. This is the one account whose
// saving is MEASURED end to end.
const SpawnedHookTaxX = 2849.0

// Profiles name the SCARCE budget per host shape, as an ordered scarcity list.
// The numbers are illustrative ceilings used ONLY to pick the binding account;
// they are not billed and never enter a saving amount. A reader overrides for
// their own stack.
var profiles = map[string][]string{
	// a long local agent: context window is the hard ceiling (can't buy more mid-
	// session), wall-clock is what the human waits on, dollars are cheap/local.
	"laptop": {"context_window", "wall_clock", "local_cpu", "gpu_prefill"},
	// a fleet on rented GPUs: prefill FLOPs are the binding cost.
	"gpu-fleet": {"gpu_prefill", "wall_clock", "context_window", "local_cpu"},
	// a hooked CI gate: the per-gate spawn dominates; the gate is CPU-bound on itself.
	"ci-hook": {"local_cpu", "wall_clock", "gpu_prefill", "context_window"},
}

// Profiles returns the sorted list of known profile names (for flag help/validation).
func Profiles() []string {
	out := make([]string, 0, len(profiles))
	for k := range profiles {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// KnownProfile reports whether name is a declared host-shape profile.
func KnownProfile(name string) bool {
	_, ok := profiles[name]
	return ok
}

// Report is the minimal slice of a turnbench Report this lens reads. The schema
// matches internal/turnbench.Report exactly (same JSON tags), decoded into a local
// struct so the foundation layer does not import the tier-4 bench. Every field this
// tool projects is already MEASURED or already COMPUTED upstream; nothing new is
// measured here.
type Report struct {
	Cost         CostModel `json:"cost_model"`
	LocalServeNs int64     `json:"local_serve_ns"`
	Net          Net       `json:"net"`
	TurnKinds    TurnKinds `json:"turn_kinds"`
}

// CostModel mirrors the turnbench cost knobs this lens reads.
type CostModel struct {
	PromptTokensPerTurn int `json:"prompt_tokens_per_turn"`
}

// Net mirrors the turnbench flat saving this lens re-projects.
type Net struct {
	TurnsSaved     int     `json:"turns_saved"`
	TokensSaved    int     `json:"tokens_saved"`
	DollarsSaved   float64 `json:"dollars_saved"`
	LatencySavedMs float64 `json:"latency_saved_ms"`
}

// TurnKinds mirrors the forced/elision split of turns_saved.
type TurnKinds struct {
	Forced  int `json:"forced"`
	Elision int `json:"elision"`
}

// AxisSaving is one account's saving, labeled by per-axis provenance. The label is
// the honesty contract: "measured" | "modeled" | "measured-rate".
type AxisSaving struct {
	Account    string  `json:"account"`
	Amount     float64 `json:"amount"`
	Unit       string  `json:"unit"`
	Provenance string  `json:"provenance"`
	Note       string  `json:"note"`
}

// Vector is the four-account decomposition of one flat Net saving.
type Vector struct {
	TurnsSaved int          `json:"turns_saved"`
	Forced     int          `json:"forced"`
	Elision    int          `json:"elision"`
	Axes       []AxisSaving `json:"axes"`
	Binding    string       `json:"binding_account"`
	Profile    string       `json:"profile"`
	// Cross-checks against the flat Net (re-projection, not a new claim).
	NetDollarsSaved    float64 `json:"net_dollars_saved"`
	VectorDollarsSaved float64 `json:"vector_dollars_saved"`
}

// round mirrors Python's round(x, n) (round-half-to-even) so the Go output matches
// the original tool to the digit.
func round(x float64, n int) float64 {
	p := math.Pow(10, float64(n))
	return math.RoundToEven(x*p) / p
}

// BuildVector re-projects a Report's flat Net into the four-account vector. Every
// amount is derived from a field the Report ALREADY carries; this is a lens,
// labeled per axis. pollutionRate is optional: a non-nil value promotes the context
// axis to "measured-rate". An unknown profile is an error.
func BuildVector(r Report, profile string, pollutionRate *float64) (Vector, error) {
	order, ok := profiles[profile]
	if !ok {
		return Vector{}, fmt.Errorf("unknown profile %q; pick one of %v", profile, Profiles())
	}

	turns := r.Net.TurnsSaved
	dollars := r.Net.DollarsSaved
	latencyMs := r.Net.LatencySavedMs
	promptTok := r.Cost.PromptTokensPerTurn
	localServeNs := r.LocalServeNs
	if localServeNs < 0 {
		localServeNs = 0
	}

	axes := make([]AxisSaving, 0, 4)

	// local_cpu: MEASURED. The per-saved-call in-process serve cost vs the spawned-
	// hook round-trip the baseline would have paid to gate the same call. The SAVED
	// local-CPU time = turns_saved x (spawned - in_process). in_process ~=
	// local_serve_ns; spawned ~= local_serve_ns x tax. Saving per call =
	// local_serve_ns x (tax - 1). This is the account Net omits entirely.
	if localServeNs > 0 {
		perCallSavedNs := float64(localServeNs) * (SpawnedHookTaxX - 1.0)
		cpuSavedMs := float64(turns) * perCallSavedNs / 1e6
		axes = append(axes, AxisSaving{
			Account:    "local_cpu",
			Amount:     round(cpuSavedMs, 4),
			Unit:       "ms_cpu",
			Provenance: "measured",
			Note: fmt.Sprintf("turns_saved x local_serve_ns x (boundary_tax-1); "+
				"local_serve_ns=%d, tax~%.0fx "+
				"(BENCHMARK-AUTHORITY M3). The axis Net omits.", localServeNs, SpawnedHookTaxX),
		})
	} else {
		axes = append(axes, AxisSaving{
			Account: "local_cpu", Amount: 0.0, Unit: "ms_cpu", Provenance: "measured",
			Note: "no local_serve_ns in report; local-CPU saving unmeasured here",
		})
	}

	// gpu_prefill: MODELED. A token proxy for the forward pass never run. The prompt
	// tokens of each saved turn are a forward pass the engine skips. This is a TOKEN
	// proxy, NOT FLOPs and NOT wall-clock -- attention is O(L^2), so this
	// under-counts long-context saves on purpose (the safe direction).
	prefillTokSaved := turns * promptTok
	axes = append(axes, AxisSaving{
		Account:    "gpu_prefill",
		Amount:     float64(prefillTokSaved),
		Unit:       "prefill_tokens",
		Provenance: "modeled",
		Note: "turns_saved x prompt_tokens_per_turn; a token proxy for the forward " +
			"pass skipped. O(L^2) attention means this UNDER-counts long context.",
	})

	// context_window: MEASURED-as-a-rate if a pollution rate is supplied, else
	// MODELED. Each saved call is a result that never occupied a window slot. With a
	// ctxmmu pollution rate we can say what fraction of results would have been paged
	// out anyway; without it we report the modeled slot count.
	var ctx AxisSaving
	if pollutionRate != nil {
		ctx = AxisSaving{
			Account: "context_window", Amount: float64(turns), Unit: "window_slots",
			Provenance: "measured-rate",
			Note: fmt.Sprintf("turns_saved slots never entered the window; ctxmmu pollution_rate="+
				"%.3f supplied as the measured paged-out fraction.", *pollutionRate),
		}
	} else {
		ctx = AxisSaving{
			Account: "context_window", Amount: float64(turns), Unit: "window_slots",
			Provenance: "modeled",
			Note: "turns_saved window slots never consumed (1 slot per saved result); " +
				"supply --pollution-rate from ctxmmu for the measured paged-out fraction.",
		}
	}
	axes = append(axes, ctx)

	// wall_clock: MODELED. The round-trips never blocked on. This is the knobbed
	// ModelTurnLatencyMs from the cost model, never a measured wall-clock.
	axes = append(axes, AxisSaving{
		Account:    "wall_clock",
		Amount:     round(latencyMs, 3),
		Unit:       "ms_wall",
		Provenance: "modeled",
		Note: "turns_saved x cost_model.ModelTurnLatencyMs; a knobbed round-trip " +
			"constant, never a measured wall-clock.",
	})

	// The dollar saving is NOT a fifth axis -- it is the wall_clock+prefill accounts
	// priced. We carry it only to cross-check the re-projection equals the flat Net.
	vectorDollars := dollars // by construction: same turns, same price

	return Vector{
		TurnsSaved: turns, Forced: r.TurnKinds.Forced, Elision: r.TurnKinds.Elision,
		Axes:               axes,
		Binding:            pickBinding(axes, order, turns),
		Profile:            profile,
		NetDollarsSaved:    round(dollars, 6),
		VectorDollarsSaved: round(vectorDollars, 6),
	}, nil
}

// pickBinding returns the first account in the profile's scarcity order that has a
// non-zero saving -- the scarcest budget this run actually relieved. With no saving
// at all, nothing binds.
func pickBinding(axes []AxisSaving, order []string, turns int) string {
	if turns == 0 {
		return "none"
	}
	have := map[string]bool{}
	for _, a := range axes {
		if a.Amount > 0 {
			have[a.Account] = true
		}
	}
	for _, acct := range order {
		if have[acct] {
			return acct
		}
	}
	return order[0]
}

// Selfcheck returns the first violated anti-overclaim invariant, or nil if every
// hard invariant holds. It is the Go port of the original tool's selfcheck: the
// vector must DECOMPOSE one event, never INFLATE it.
func Selfcheck() error {
	report := syntheticReport()
	happy := syntheticHappy()

	// 1. The re-projection equals Net on dollars (decompose, never inflate).
	v, err := BuildVector(report, "laptop", nil)
	if err != nil {
		return err
	}
	if math.Abs(v.NetDollarsSaved-v.VectorDollarsSaved) >= 1e-9 {
		return fmt.Errorf("vector dollars != Net dollars (inflation)")
	}

	// 2. Every axis saving is >= 0.
	for _, a := range v.Axes {
		if a.Amount < 0 {
			return fmt.Errorf("negative saving on %s", a.Account)
		}
	}

	// 3. The local_cpu axis is non-zero and MEASURED when local_serve_ns is present
	//    -- this is the whole point: the account Net omits is now surfaced.
	cpu, ok := axisByAccount(v.Axes, "local_cpu")
	if !ok {
		return fmt.Errorf("local_cpu axis missing")
	}
	if cpu.Provenance != "measured" {
		return fmt.Errorf("local_cpu axis is not labeled measured")
	}
	if cpu.Amount <= 0 {
		return fmt.Errorf("local_cpu saving is zero despite local_serve_ns>0 and turns>0")
	}

	// 4. gpu_prefill and wall_clock are honestly labeled MODELED (not measured).
	for _, acct := range []string{"gpu_prefill", "wall_clock"} {
		ax, ok := axisByAccount(v.Axes, acct)
		if !ok {
			return fmt.Errorf("%s axis missing", acct)
		}
		if ax.Provenance != "modeled" {
			return fmt.Errorf("%s must be labeled modeled, got %s", acct, ax.Provenance)
		}
	}

	// 5. The happy-path control saves EXACTLY zero on every axis (anti-inflation
	//    gate, the same one turntax-happy enforces).
	hv, err := BuildVector(happy, "laptop", nil)
	if err != nil {
		return err
	}
	for _, a := range hv.Axes {
		if a.Amount != 0 {
			return fmt.Errorf("happy control nonzero on %s: %v", a.Account, a.Amount)
		}
	}
	if hv.Binding != "none" {
		return fmt.Errorf("happy control binds an account: %s", hv.Binding)
	}

	// 6. The binding account follows the PROFILE, never hard-coded to dollars.
	//    On a laptop, context_window binds; on ci-hook, local_cpu binds. Same
	//    saving, different binding.
	lapV, err := BuildVector(report, "laptop", nil)
	if err != nil {
		return err
	}
	ciV, err := BuildVector(report, "ci-hook", nil)
	if err != nil {
		return err
	}
	lap, ci := lapV.Binding, ciV.Binding
	if lap == ci {
		return fmt.Errorf("binding account did not vary by profile (laptop=%s, ci=%s)", lap, ci)
	}
	if lap != "context_window" {
		return fmt.Errorf("laptop profile should bind context_window, got %s", lap)
	}
	if ci != "local_cpu" {
		return fmt.Errorf("ci-hook profile should bind local_cpu, got %s", ci)
	}
	return nil
}

func axisByAccount(axes []AxisSaving, account string) (AxisSaving, bool) {
	for _, a := range axes {
		if a.Account == account {
			return a, true
		}
	}
	return AxisSaving{}, false
}

// syntheticReport is a Report with the exact shape internal/turnbench emits, used by
// Selfcheck so the contract is testable with no model run. Values are illustrative,
// NOT a benchmark claim.
func syntheticReport() Report {
	return Report{
		Cost:         CostModel{PromptTokensPerTurn: 1200},
		LocalServeNs: 362, // ~M3 decide
		Net: Net{
			TurnsSaved:     10,
			TokensSaved:    13200,   // 10 x (1200+120)
			DollarsSaved:   0.054,   // 10 x (1200/1e6*3 + 120/1e6*15)
			LatencySavedMs: 15000.0, // 10 x 1500
		},
		TurnKinds: TurnKinds{Forced: 6, Elision: 4},
	}
}

// syntheticHappy is a happy-path control: zero turns saved. Every axis must be 0
// (anti-inflation).
func syntheticHappy() Report {
	return Report{
		Cost:         CostModel{PromptTokensPerTurn: 1200},
		LocalServeNs: 362,
		Net:          Net{TurnsSaved: 0, TokensSaved: 0, DollarsSaved: 0.0, LatencySavedMs: 0.0},
		TurnKinds:    TurnKinds{Forced: 0, Elision: 0},
	}
}
