package main

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// writeDispatchModelSidecar records the pinned model id next to a worker log as a plain-text
// .model sidecar (Layer 5b), so the witness sweep can scrape it back and a model-accounting
// run can attribute each slot's spend. A blank model (the seat-default floor) writes NOTHING,
// so an unconfigured fleet's runs dir stays byte-identical to before the seam. Fail-open.
func writeDispatchModelSidecar(logPath, model string) {
	model = strings.TrimSpace(model)
	if strings.TrimSpace(logPath) == "" || model == "" {
		return
	}
	stem := strings.TrimSuffix(logPath, filepath.Ext(logPath))
	_ = os.WriteFile(stem+dispatchtick.ModelSidecarSuffix, []byte(model), 0o644)
}

// applyModelDowngrade is Layer-2 in-tick re-dispatch: when the target issue's last finished
// slot (in this tick's witness records) exited CLAIM_NO_COMMIT with a model-switchable reason,
// it overrides the resolved model with the NEXT downgrade-chain rung so the re-dispatch does
// not re-storm the same walled model. fired is false when no switchable slot names the target
// or its ladder is exhausted — the resolved policy then stands unchanged.
func applyModelDowngrade(backend string, target int, records []dispatchtick.WitnessRecord) (workerModelPolicy, bool) {
	chain := workerDowngradeChain(backend)
	next, ok := dispatchtick.ModelDowngradeReDispatch(records, chain)[target]
	if !ok {
		return workerModelPolicy{}, false
	}
	return workerModelPolicy{Model: next, Chain: dropModel(chain, next), Source: modelSourceDowngrade}, true
}

// applyPlacementGate is the PREVENTIVE placement check (#3521). Before launch it tests the
// resolved placement's (work-shape × model-reliability) pairing and, when that pairing is
// known-bad, re-routes onto a model whose restart profile can hold the shape — so a cheap
// model is never dropped into a churning slot it cannot survive and then blamed for the
// outcome. This runs BEFORE the reactive Layer-2 downgrade, which never fires on the
// witnessed failure (a context-balloon/restart-limit starvation is not a model-switchable
// wall like usage_cap / model_unknown / rate_limit).
//
// Only the AUTOMATIC pins are gated — modelSourceTier, modelSourceWorkClass, modelSourceRung
// and modelSourceEscalated. An explicit --worker-model pin, a lane_models pin, and the
// benchmark gate are deliberate operator intent and always win, matching the resolver's stated
// precedence doctrine; the gate exists to protect the worker from a TABLE's choice, not to
// override a human's. The placement ladder (#5416 track E) belongs on the gated side for the
// same reason the tier table does, and more so: it picks a rung from a corpus of past outcomes,
// which says a model CAN do this class of work and says nothing about whether it can hold
// this issue's work SHAPE. Those are different questions, and this is the one that asks the
// second.
//
// An ESCALATED pin (track D) is gated for that same reason and is the case where it matters
// most: the item has already failed once, so dropping the rung it just paid for into a shape
// that model cannot hold spends the budget on a second starvation. The debit is NOT refunded
// when this fires — the ledger records what was authorised, and a refund path is a way to
// drain a budget in a loop.
//
// fired is false for every other source, an unpinned seat default, a surgical shape, and an
// unknown shape (an untagged or contradictorily-tagged issue) — so a default fleet tick is
// byte-identical to before this seam.
//
// The tier's reasoning posture (Effort/Ultracode) is CARRIED across the re-route rather than
// stripped with the model — the same rule Layer-2's downgrade follows, since a placement wall
// is model-scoped, not reasoning-scoped. So the witnessed fable+ultracode-on-churning-hard
// placement becomes opus+ultracode, not a bare opus. The chain drops the safe model AND keeps
// the refused one dropped, so a later downgrade never re-offers the model that cannot hold it.
func applyPlacementGate(p workerModelPolicy, shape dispatchtick.WorkShape) (workerModelPolicy, bool) {
	switch p.Source {
	case modelSourceTier, modelSourceWorkClass, modelSourceRung, modelSourceEscalated:
	default:
		return p, false
	}
	v := dispatchtick.AssessPlacement(shape, p.Model)
	if v.OK {
		return p, false
	}
	p.Model = v.SafeModel
	p.Chain = dropModel(p.Chain, v.SafeModel)
	p.Source = modelSourcePlacement
	p.PlacementReason = v.Reason
	return p, true
}

// dispatchTickPolicy loads the operator account policy (accounts_policy.json) for a workspace
// so the resolver can read a lane_models pin. Fail-open: a missing/malformed policy yields
// DefaultPolicy (no lane pins), exactly the pre-seam behavior. Kept here (not in the tick) so
// the tick file needs no new import.
func dispatchTickPolicy(root string) fleetaccounts.Policy {
	paths := fleetaccounts.ResolvePaths(filepath.Join(root, "tools"))
	return fleetaccounts.LoadPolicy(paths)
}

// dispatchWorkerModelMap renders a resolved worker-model decision for the tick payload — the
// pinned model, its provenance source, and the downgrade chain a Layer-2 switch would try.
func dispatchWorkerModelMap(p workerModelPolicy) map[string]any {
	out := map[string]any{"model": p.Model, "source": p.Source}
	if len(p.Chain) > 0 {
		out["downgrade_chain"] = append([]string(nil), p.Chain...)
	}
	// Surface the tier launch knobs only when set, so a non-tier decision's payload is
	// byte-identical to before the seam.
	if strings.TrimSpace(p.Effort) != "" {
		out["effort"] = p.Effort
	}
	if p.Ultracode {
		out["ultracode"] = true
	}
	// The preventive placement gate's typed, closed reason — surfaced only when it fired, so
	// an ungated decision's payload is byte-identical to before the seam.
	if r := strings.TrimSpace(p.PlacementReason); r != "" {
		out["placement_reason"] = r
	}
	return out
}

// Model-switching Layer 3: the worker-model RESOLVER. It decides which model a dispatch
// resolution worker starts on, from three operator-controlled sources (an explicit pin, a
// per-lane lane_models pin, the benchmark gate) plus the ordered downgrade chain a Layer-2
// model switch re-dispatches onto. Pure: config in, decision out — the tick shell reads the
// policy off disk and hands it here, so the resolution unit-tests without any I/O.

// workerModelSource names where a resolved worker model came from — surfaced in the tick
// payload and the .model witness so a model-accounting run can attribute each worker's spend.
// dispatchEngineForWorkClass resolves the worker engine before account/model routing.
// The table is intentionally small: throughput-shaped grind classes use codex;
// correctness-shaped rigor classes use Claude. An explicit operator backend pin
// always wins, and an unknown/untagged class preserves current byte-for-byte.
func dispatchEngineForWorkClass(current, workClass, explicit string) string {
	if pin := strings.TrimSpace(strings.ToLower(explicit)); pin != "" {
		return pin
	}
	class := strings.NewReplacer("-", "_", " ", "_").Replace(strings.TrimSpace(strings.ToLower(workClass)))
	grind := map[string]bool{
		"gardening": true, "grind": true, "mechanical": true, "mechanical_grind": true,
		"hygiene": true, "doc_sync": true, "log_sweep": true,
	}
	rigor := map[string]bool{
		"engineering": true, "rigor": true, "audit": true, "verify": true,
		"security": true, "security_audit": true, "design": true, "benchmark_claims": true,
	}
	switch {
	case grind[class]:
		return "codex"
	case rigor[class]:
		return "claude"
	default:
		return strings.TrimSpace(strings.ToLower(current))
	}
}

const (
	// modelSourceSeatDefault is the DEFAULT claude posture: the model is blank, so the
	// worker starts on the seat's own saved default and degrades through --fallback-model.
	// A normal unattended fleet tick keeps this — the seam changes nothing until asked.
	modelSourceSeatDefault = "seat-default"
	// modelSourceExplicit is an operator --worker-model pin (highest precedence).
	modelSourceExplicit = "explicit"
	// modelSourceLane is a per-lane lane_models pin.
	modelSourceLane = "lane"
	// modelSourceProfile is the account routing-profile model, pinned under the benchmark gate.
	modelSourceProfile = "profile"
	// modelSourceDefault is the fleet default model, pinned under the benchmark gate when the
	// account itself names none.
	modelSourceDefault = "default"
	// modelSourceAccount is the opencode/codex seat model pinned via -m, exactly as before.
	modelSourceAccount = "account"
	// modelSourceDowngrade is a Layer-2 in-tick re-dispatch: the target's last slot walled
	// on a model-switchable reason, so it advances one rung down the downgrade ladder.
	modelSourceDowngrade = "model-downgrade"
	// modelSourceTier is the opt-in per-issue tier launch profile (FLEET_TIER_LAUNCH): the
	// target's trusted tier labels resolve to a {model, effort, ultracode} profile. It sits
	// BELOW the operator pins and the benchmark gate (all of which still win) and ABOVE the
	// seat default, so an untagged issue or a fleet with the knob off is unchanged.
	modelSourceTier = "tier"
	// modelSourceWorkClass is a LOW-precedence work-class model default: a work-class
	// dispatcher (e.g. `fak garden dispatch`, project-management / repo-maintenance work)
	// pins the cheap model for its whole class. It sits BELOW the per-issue tier profile
	// (so an explicit tier/pm or tier/T0 label still decides — hard planning still escalates
	// to opus) and ABOVE the seat default, so a normal tick that names no work-class model
	// is unchanged.
	modelSourceWorkClass = "work-class"
	// modelSourcePlacement is the PREVENTIVE placement gate (#3521): the resolved AUTOMATIC
	// placement paired a work SHAPE with a model whose restart/reliability profile cannot
	// hold it, so the worker is re-routed onto a model that can — before launch, rather than
	// after a CLAIM_NO_COMMIT wall the reactive downgrade chain never sees (a restart-amnesia
	// starvation is not one of the model-switchable walls).
	modelSourcePlacement = "placement-gate"
)

// workerModelPolicy is the resolved launch decision for one dispatch worker: the Model to pin
// (empty => seat default + fallback chain), the ordered downgrade Chain a Layer-2 model switch
// tries after it, the Source that decided it, and — for the per-issue tier profile — the
// Claude reasoning Effort and Ultracode workflow-mode knobs carried into WorkerLaunch. Effort
// and Ultracode are empty/false for every non-tier source, so nothing changes unless the
// tier launch is on and the issue is tagged.
type workerModelPolicy struct {
	Model     string
	Chain     []string
	Source    string
	Effort    string
	Ultracode bool
	// PlacementReason is the closed reason token (dispatchtick.PlacementShapeMismatch) set
	// only when the preventive placement gate re-routed this placement; empty otherwise, so
	// an ungated decision's payload is byte-identical to before the seam.
	PlacementReason string
}

// pinned reports whether the resolver un-blanked the model to an exact id — i.e. this worker
// starts on a KNOWN model rather than the seat default. Layer-2 re-dispatch and the .model
// witness only act on a pinned worker.
func (p workerModelPolicy) pinned() bool { return strings.TrimSpace(p.Model) != "" }

// resolveWorkerModelPolicy decides which model a dispatch resolution worker starts on.
//
// opencode/codex pin the seat's own model via -m exactly as before (Source=account). For the
// claude backend the model is BLANK by default (Source=seat-default) — the worker starts on the
// seat's saved default and degrades through the --fallback-model chain — so a normal unattended
// fleet tick is byte-identical to before this seam. The claude model is UN-BLANKED (pinned to an
// exact --model) only when an operator asks, in precedence order:
//
//  1. an explicit --worker-model pin (Source=explicit);
//  2. a per-lane lane_models pin from the account policy (Source=lane);
//  3. the benchmark gate (benchGate) — a model-accounting run that pins the account's profile
//     model, else the fleet default, so the run measures a KNOWN model instead of one the
//     fallback chain silently switched under it;
//  4. the per-issue tier launch profile (tierProfile) — the opt-in FLEET_TIER_LAUNCH map from
//     the target's trusted tier labels to a {model, effort, ultracode} launch profile. The
//     shell resolves it (flag + labels + table) and hands it in; nil (knob off, untagged, or
//     non-claude) leaves the seat default in place. It carries the ONLY effort/ultracode a
//     worker is launched with.
//  5. the work-class model default (workClassModel) — a whole-class cheap-model pin set by a
//     work-class dispatcher (e.g. `fak garden dispatch` pins fable for project-management /
//     maintenance work). Lowest-precedence un-blanking: it applies only when no operator pin,
//     bench gate, or per-issue tier profile spoke, so an explicit tier label on a garden issue
//     still escalates and an ordinary tick that names no work-class model is unchanged.
//
// The downgrade Chain is the same --fallback-model chain the worker command already carries,
// with the primary dropped so a Layer-2 switch never re-dispatches onto the walled model.
func resolveWorkerModelPolicy(backend, lane, explicit string, account dispatchtick.Account, pol fleetaccounts.Policy, benchGate bool, tierProfile *dispatchtick.LaunchProfile, workClassModel string) workerModelPolicy {
	if backend == "opencode" || backend == "codex" {
		return workerModelPolicy{Model: strings.TrimSpace(account.Model), Source: modelSourceAccount}
	}
	chain := workerDowngradeChain(backend)
	pin := func(model, source string) workerModelPolicy {
		return workerModelPolicy{Model: model, Chain: dropModel(chain, model), Source: source}
	}
	if m := strings.TrimSpace(explicit); m != "" {
		return pin(m, modelSourceExplicit)
	}
	if m := pol.LaneModel(lane); m != "" {
		return pin(m, modelSourceLane)
	}
	if benchGate {
		if m := strings.TrimSpace(account.Model); m != "" {
			return pin(m, modelSourceProfile)
		}
		return pin(defaultLaunchModel, modelSourceDefault)
	}
	// Tier launch is a low-precedence UN-BLANKING: it pins a model (and carries the worker's
	// effort/ultracode) only when no operator pin or bench gate already spoke, so an explicit
	// intent always wins and an untagged issue falls through to the work-class default or seat.
	if tierProfile != nil {
		return workerModelPolicy{
			Model:     strings.TrimSpace(tierProfile.Model),
			Chain:     dropModel(chain, tierProfile.Model),
			Source:    modelSourceTier,
			Effort:    tierProfile.Effort,
			Ultracode: tierProfile.Ultracode,
		}
	}
	// Work-class default is the LOWEST-precedence un-blanking: a whole-class cheap-model pin
	// (garden/PM dispatch -> fable) that applies only when nothing more specific spoke, so a
	// per-issue tier label above still escalates and a normal tick stays on the seat default.
	if m := strings.TrimSpace(workClassModel); m != "" {
		return pin(m, modelSourceWorkClass)
	}
	return workerModelPolicy{Model: "", Chain: chain, Source: modelSourceSeatDefault}
}

// workerDowngradeChain is the ordered, de-duplicated model chain a worker degrades through —
// the same comma-separated list dispatchWorkerFallbackModel hands `claude -p --fallback-model`,
// so the --fallback-model flag and a Layer-2 re-dispatch draw from ONE source. Empty for
// non-claude backends and when fallback is disabled.
func workerDowngradeChain(backend string) []string {
	return splitModelChain(dispatchWorkerFallbackModel(backend))
}

// splitModelChain parses a comma-separated model chain, dropping blanks and case-insensitive
// duplicates while preserving order.
func splitModelChain(raw string) []string {
	var out []string
	seen := map[string]bool{}
	for _, m := range strings.Split(raw, ",") {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		key := strings.ToLower(m)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, m)
	}
	return out
}

// dropModel returns chain without the given model (case-insensitive) — so a downgrade chain
// never re-offers the model already tried.
func dropModel(chain []string, model string) []string {
	model = strings.ToLower(strings.TrimSpace(model))
	if model == "" {
		return append([]string(nil), chain...)
	}
	out := make([]string, 0, len(chain))
	for _, m := range chain {
		if strings.ToLower(strings.TrimSpace(m)) == model {
			continue
		}
		out = append(out, m)
	}
	return out
}
