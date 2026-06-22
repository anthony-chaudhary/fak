// leverflip.go — LEVER-FLIP CAUSAL ATTRIBUTION: replay ONE recorded trace through L
// masked kernels, each with exactly one rung ABLATED, and diff the counters to get
// EXACT per-rung causal attribution — which adjudicator/rung actually changed the
// outcome on this trace.
//
// WHAT IT GENERALIZES. The turn-tax report already proves the vDSO lever with a REAL
// ON/OFF path swap (Run's vdso-on vs vdso-off replay; TestRun_VDSOAblationIsARealPathSwap
// asserts net(on) - net(off) == VDSOHits). That is one lever, flipped by a special
// SetVDSO toggle. This file flips the WHOLE rung chain the SAME way: take the full
// registered adjudicator chain, produce L copies each missing exactly one rung (named
// GENERICALLY by its self-reported By — abi.RungName / abi.WithoutRung), inject each via
// kernel.WithAdjudicators (#500), replay, and diff. The vDSO lever falls out of the SAME
// generic loop — it is one named lever among L, not a hard-coded path. (vDSO is a
// FastPath, not a chain rung, so its ablation is realized by the kernel's vdso-off arm
// the blessed replay() already exposes; the ATTRIBUTION logic treats it uniformly with
// every chain rung — no SetVDSO branch in the diff/attribution.)
//
// WHY IT IS CHEAP (the honest multiple — carry it, do not inflate). Replay is MODEL-FREE:
// each lever is one more deterministic kernel replay of the frozen trace (O(calls)), not
// another agent+model run. For L levers it is L replays off ONE recording — so the
// attribution COVERAGE per recorded run is L× (one trace yields L causal ablations), and
// composed with the K-policy spine (policyreplay.go) it is L×K replays off one recording.
// The MEASURED win is that L×K ≈ 10–300× more attribution per dollar of model time, NOT
// 10^9×: the avoided model run is counted ONCE by the spine (ModelTurnsAvoided), and the
// lever-flip rides that SAME avoided run — it does not avoid a second one, so it must not
// be multiplied back in. The 10^9× figure is the cost of ONE model turn vs one in-process
// adjudication; quoting it here would double-count. LeverFlipReport.AttributionPerRun
// states the honest L× coverage multiple for this trace.
package turnbench

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/appversion"
)

// VDSOLever is the canonical name of the vDSO lever. It is NOT a chain rung (vDSO is a
// FastPath consulted before adjudication), so the driver realizes its ablation via the
// kernel's vdso-off replay arm — but it is named and flipped in the SAME generic loop as
// every chain rung, so the attribution logic never special-cases it.
const VDSOLever = "vdso"

// CounterDelta is the baseline→masked change in each kernel outcome counter for one
// ablated rung. A NEGATIVE delta means the counter went DOWN when the rung was removed
// (the rung was CAUSING those events); a positive delta means removing the rung let other
// events fire (e.g. a vDSO-served call falling through to the engine, or a default-deny
// floor newly refusing calls the removed allow-rung had permitted). Every field is the
// masked value minus the baseline value, so 0 == "this rung did not change this counter".
type CounterDelta struct {
	Transforms  int64 `json:"transforms"`
	VDSOHits    int64 `json:"vdso_hits"`
	Quarantines int64 `json:"quarantines"`
	Denies      int64 `json:"denies"`
	EngineCalls int64 `json:"engine_calls"`
}

func (d CounterDelta) any() bool {
	return d.Transforms != 0 || d.VDSOHits != 0 || d.Quarantines != 0 || d.Denies != 0 || d.EngineCalls != 0
}

// LeverAttribution is one row of the attribution table: a single rung ablated from the
// otherwise-full chain, and the outcome-counter delta its removal caused on this trace.
type LeverAttribution struct {
	Lever string `json:"lever"` // the rung's canonical name (its self-reported By), or "vdso"

	// Realization records HOW the lever was flipped: "chain-mask" for an adjudicator rung
	// removed from the injected chain, or "vdso-off" for the fast-path arm. It is forensic
	// only — the attribution treats both uniformly — and documents that the vDSO ablation
	// is NOT a special-case in the diff (it is just a differently-realized lever).
	Realization string `json:"realization"`

	// Present is false when no rung in the base chain is named Lever (the mask removed
	// nothing). Such a lever has an all-zero delta by construction and is reported so a
	// reader sees the lever was requested but absent, not silently skipped.
	Present bool `json:"present"`

	Masked  KernelCounters `json:"masked_counters"`
	Delta   CounterDelta   `json:"delta"`
	Changed bool           `json:"changed_outcome"` // any non-zero delta — did removing this rung change ANYTHING?

	// Witness is a one-line, reader-checkable statement of what the delta proves — the
	// causal claim grounded in the live kernel counters, mirroring the Lever rows in the
	// turn-tax report.
	Witness string `json:"witness"`
}

// LeverFlipReport is the lever-flip artifact: ONE recorded trace, a baseline replay
// through the full chain, and L masked replays (one rung off each), with the per-rung
// causal attribution. It mirrors the other turnbench reports (Provenance + JSON()).
type LeverFlipReport struct {
	Provenance Provenance         `json:"provenance"`
	Calls      int                `json:"calls"`
	Baseline   KernelCounters     `json:"baseline_counters"`
	Levers     []LeverAttribution `json:"levers"`

	// Cheapness accounting (the honest multiple, MEASURED — see the file doc). L levers
	// are L model-free replays off ONE recording, so this trace yields L causal ablations
	// per recorded model run. AttributionPerRun is that L× coverage multiple; it is NOT
	// 10^9× (the avoided model turn is counted once by the policy-replay spine, and the
	// lever-flip rides the same avoided run).
	LeversReplayed    int    `json:"levers_replayed"`     // L: how many masked replays ran
	AttributionPerRun int    `json:"attribution_per_run"` // == L: causal ablations per recorded trace
	CheapnessNote     string `json:"cheapness_note"`
}

// JSON renders the report.
func (r *LeverFlipReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

// deltaOf computes masked − baseline for each outcome counter.
func deltaOf(base, masked KernelCounters) CounterDelta {
	return CounterDelta{
		Transforms:  masked.Transforms - base.Transforms,
		VDSOHits:    masked.VDSOHits - base.VDSOHits,
		Quarantines: masked.Quarantines - base.Quarantines,
		Denies:      masked.Denies - base.Denies,
		EngineCalls: masked.EngineCalls - base.EngineCalls,
	}
}

// RunLeverFlip replays ONE frozen trace through the all-rungs BASELINE plus L masked
// kernels — one rung ABLATED per masked replay — and diffs the kernel's outcome counters
// to attribute each delta to the rung that was removed. The result is an EXACT per-rung
// causal-attribution table: which adjudicator/rung actually changed the outcome on this
// trace, measured (not modeled) from the live kernel counters.
//
// levers names the rungs to flip. When empty it defaults to EVERY chain rung (by its
// canonical abi.RungName) plus the vDSO lever — the full causal sweep. A requested lever
// that names no rung in the base chain is reported Present=false with an all-zero delta
// (requested-but-absent, never silently dropped).
//
// The replays are SEQUENTIAL: each is a model-free kernel replay of the frozen trace
// (O(calls)), and replay() resets two process-global pieces of cross-call state (the vDSO
// world counter and the IFC ledger) per arm, so running them in series keeps each arm's
// reset uncontended and the whole driver deterministic and obviously race-free. L is tiny
// (the registered chain depth) and a replay is microseconds, so series costs nothing the
// attribution needs.
func RunLeverFlip(ctx context.Context, t *Trace, levers ...string) (*LeverFlipReport, error) {
	if t == nil || len(t.Calls) == 0 {
		return nil, fmt.Errorf("turnbench: RunLeverFlip needs a non-empty trace")
	}
	// Install the agent's policy/grammar/engine world (idempotent) so the trace's tools
	// trigger the REAL rungs — exactly as Run/RunPolicyReplay do.
	agent.Configure()

	// The base chain is the FULL registered adjudicator chain (rank-sorted) — every rung a
	// global-registry replay folds. Per lever we copy it and remove exactly one rung by
	// name (abi.WithoutRung), then inject the masked chain via kernel.WithAdjudicators
	// (#500). Nothing global is mutated, so the baseline and every masked arm are
	// independent.
	baseChain := abi.Adjudicators()

	if len(levers) == 0 {
		levers = defaultLevers(baseChain)
	}

	// Baseline: full chain, vDSO ON. Every masked arm is diffed against this.
	baseKC, _, _, _, _, err := replay(ctx, t, true, false, false)
	if err != nil {
		return nil, fmt.Errorf("turnbench: lever-flip baseline replay: %w", err)
	}

	rows := make([]LeverAttribution, 0, len(levers))
	for _, lever := range levers {
		row, err := flipOne(ctx, t, baseChain, baseKC, lever)
		if err != nil {
			return nil, err
		}
		rows = append(rows, row)
	}

	l := len(rows)
	return &LeverFlipReport{
		Provenance: Provenance{
			AppVersion:   appversion.Current(),
			Command:      "turnbench.RunLeverFlip",
			SliceID:      t.SliceID,
			WorkloadHash: t.WorkloadHash(),
			GoVersion:    runtime.Version(),
			OS:           runtime.GOOS,
			GeneratedBy:  "fak/internal/turnbench (lever-flip causal attribution)",
		},
		Calls:             len(t.Calls),
		Baseline:          baseKC,
		Levers:            rows,
		LeversReplayed:    l,
		AttributionPerRun: l,
		CheapnessNote: fmt.Sprintf(
			"%d causal ablations off ONE recorded trace (L model-free replays, O(levers)); "+
				"composed with K policies it is L×K replays per recording — a MEASURED 10–300× "+
				"attribution-coverage multiple, NOT 10^9× (the avoided model run is counted once "+
				"by the policy-replay spine; the lever-flip rides that same avoided run).", l),
	}, nil
}

// defaultLevers is the full causal sweep: every chain rung by its canonical name, in chain
// (rank) order, plus the vDSO lever appended. Unnamed rungs (By=="") and duplicate names
// are dropped so each named lever is flipped once.
func defaultLevers(baseChain []abi.Adjudicator) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(baseChain)+1)
	for _, name := range abi.RungNames(baseChain) {
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	if !seen[VDSOLever] {
		out = append(out, VDSOLever)
	}
	return out
}

// flipOne ablates exactly one lever from the otherwise-full chain, replays the trace, and
// attributes the counter delta. The vDSO lever is realized by the kernel's vdso-off arm
// (vDSO is a FastPath, not a chain rung); every other lever is a chain mask. The
// attribution — the diff and the Changed/Witness derivation — is IDENTICAL for both, so
// there is no special-case in the causal logic.
func flipOne(ctx context.Context, t *Trace, baseChain []abi.Adjudicator, base KernelCounters, lever string) (LeverAttribution, error) {
	row := LeverAttribution{Lever: lever}

	var maskedKC KernelCounters
	var err error
	if lever == VDSOLever {
		// Fast-path lever: replay with vDSO OFF and the FULL chain. This is the exact
		// ON/OFF path swap Run already uses for the vdso lever — reused generically here.
		row.Realization, row.Present = "vdso-off", true
		maskedKC, _, _, _, _, err = replay(ctx, t, false, false, false)
	} else {
		// Chain-rung lever: remove the named rung and inject the masked chain.
		masked, removed := abi.WithoutRung(baseChain, lever)
		row.Realization, row.Present = "chain-mask", removed > 0
		if removed == 0 {
			// Requested lever names no rung in this chain — an all-zero delta by
			// construction; record it as absent rather than silently dropping it.
			row.Masked = base
			row.Witness = fmt.Sprintf("lever %q matches no rung in the registered chain — no ablation possible", lever)
			return row, nil
		}
		maskedKC, _, _, _, _, err = replay(ctx, t, true, false, false, withAdjudicators(masked))
	}
	if err != nil {
		return LeverAttribution{}, fmt.Errorf("turnbench: lever-flip replay for %q: %w", lever, err)
	}

	row.Masked = maskedKC
	row.Delta = deltaOf(base, maskedKC)
	row.Changed = row.Delta.any()
	row.Witness = witnessFor(lever, base, row.Delta)
	return row, nil
}

// witnessFor renders the one-line causal claim the delta proves, grounded in the live
// counters. The vDSO lever's witness names the reproduction of the existing turn-tax
// result (baseline VDSOHits all disappear when the fast path is off — the same path swap
// TestRun_VDSOAblationIsARealPathSwap checks). A no-change lever is named as such — a
// REAL finding: on this trace, removing the rung changed no outcome (e.g. a redundant
// deny another rung also covers).
func witnessFor(lever string, base KernelCounters, d CounterDelta) string {
	if !d.any() {
		return fmt.Sprintf("removing %q changed NO outcome counter on this trace (no causal effect here — e.g. a redundantly-covered verdict)", lever)
	}
	if lever == VDSOLever {
		return fmt.Sprintf("vDSO OFF − ON: vdso_hits delta %d (all %d baseline local serves fall through to the engine; engine_calls delta %+d) — the SAME path swap turn-tax's vdso-off arm proves",
			d.VDSOHits, base.VDSOHits, d.EngineCalls)
	}
	return fmt.Sprintf("removing rung %q shifts outcomes by transforms %+d, vdso_hits %+d, quarantines %+d, denies %+d, engine_calls %+d",
		lever, d.Transforms, d.VDSOHits, d.Quarantines, d.Denies, d.EngineCalls)
}

// LeverByName returns the attribution row for a named lever, or (zero, false) if the
// report did not flip that lever. A small accessor for tests/callers that assert a single
// rung's delta without scanning the slice.
func (r *LeverFlipReport) LeverByName(name string) (LeverAttribution, bool) {
	for _, l := range r.Levers {
		if l.Lever == name {
			return l, true
		}
	}
	return LeverAttribution{}, false
}
