package ablate

// Cold-tool-defer ON/OFF A/B arm (#3628). Parent epic #3569 (cache-verify: independent
// trust-but-verify LOOPS for managed cache). Sibling to compaction_ab.go (#2805),
// anchor_ab.go (#2809), and context_extension_ab.go (#2808): each replays ONE identical
// workload under a feature ON vs OFF and reports the delta the lever's stated default-on
// gate requires. This one does it for the #3232 cold-tool deferral, whose stated default-on
// gate — a 'token-delta x held-accuracy' A/B — was DECLARED but never evidenced, so defer
// stays default-off on faith. The unit tests over the transform prove fak AUTHORED it, not
// that the provider actually reduced resident context or that accuracy held; this arm is the
// evidence half.
//
// THE TWO ARMS over ONE WorkloadHash. Both arms price the SAME frozen tool-heavy request
// body — its sha256 is the WorkloadHash they bind to, the identical-workload anchor the
// N-arm Report.Validate enforces. defer_OFF is the verbatim body (every tool def resident);
// defer_ON is the PRODUCTION #3232 transform applied to it (gateway.DeferColdToolsAB: cold
// defs carry defer_loading, one tool_search_tool injected). The arm reads TWO signals off
// the pair:
//
//   - PROVIDER RESIDENT-TOKEN DELTA: the house tokenizer (mcpfootprint.Price, the ONE
//     estimator the #3532 scorecard uses) over the PROVIDER-RESIDENT tool slice
//     (gateway.ResidentToolDefs — the defs Anthropic keeps in context), OFF (all defs) minus
//     ON (hot core + tool_search_tool). Positive = tokens the provider no longer holds
//     resident when defer is armed, faulted in on demand. This is the provider-side payoff
//     the issue's out-of-scope contrasts against request BYTES, which GROW under defer.
//   - HELD-ACCURACY: the structural capability-retention proxy — the fraction of tools
//     advertised OFF that stay REACHABLE ON (resident directly, or deferred but still present
//     in tools[] and discoverable via the injected tool_search_tool). Deferral must not COST
//     the agent a tool it had; a DROPPED def is an accuracy regression this catches by name.
//     For the correct transform this is 1.0 — held.
//
// HONESTY (the load-bearing nuance the issue names). The resident-token delta is ESTIMATED
// (house tokenizer on the resident slice), NOT the provider's OBSERVED usage; and
// held-accuracy is STRUCTURAL reachability, NOT a live task-success rate. The truly OBSERVED
// provider resident-token reduction and the end-to-end task accuracy come from a live run's
// usage relay + fak_gateway_tool_defer_* /metrics (#3233/#3536) — those are the PROMOTION
// EVIDENCE before defer flips default-on (#1844/#3232). The INVALIDATING ASSUMPTION named on
// the report Caveat: structural reachability holding does not prove a live agent still
// SELECTS a deferred tool as readily as a resident one; a live selection regression would
// need the live signal, not this proxy.
//
// Generation posture (deterministic $0, no model, no GPU): it replays a frozen body through
// the production transform and reads counters off it; it does not run a live session. The
// rendered report feeds `fak cachevalue status --ablation-report`.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/mcpfootprint"
)

// DeferArm names the two arms of the cold-tool-defer split, for labeling.
const (
	DeferArmOff = "defer_off"
	DeferArmOn  = "defer_on"
	// DeferABArmID is the sweep-row arm id for the ON-vs-OFF provider resident-token delta.
	// OFF is named first as the control (all defs resident); a positive delta reads as "arming
	// defer removed this many resident tokens", mirroring the sibling arms' treatment framing.
	DeferABArmID = "defer_on_vs_off"
)

// DeferABReport is the ONE row the cold-tool-defer ON/OFF arm reports over a single
// WorkloadHash: the provider resident-token delta (OFF−ON) and the held-accuracy figure,
// with the per-arm resident tool counts/tokens and the cold-def count the transform
// witnessed. DroppedTools names any tool advertised OFF that became unreachable ON — empty
// for the correct transform (held-accuracy 1.0), non-empty only on a real capability
// regression (a def dropped rather than deferred).
type DeferABReport struct {
	ArmID                    string   `json:"arm_id"`
	WorkloadHash             string   `json:"workload_hash"`
	ColdDeferred             int      `json:"cold_deferred"`
	OffResidentTools         int      `json:"off_resident_tools"`
	OnResidentTools          int      `json:"on_resident_tools"`
	OffResidentTokens        int      `json:"off_resident_tokens"`
	OnResidentTokens         int      `json:"on_resident_tokens"`
	ResidentTokenDelta       int      `json:"resident_token_delta"`
	HeldToolCount            int      `json:"held_tool_count"`
	TotalToolCount           int      `json:"total_tool_count"`
	HeldAccuracy             float64  `json:"held_accuracy"`
	DroppedTools             []string `json:"dropped_tools,omitempty"`
	CachePrefixByteIdentical bool     `json:"cache_prefix_byte_identical"`
	Caveat                   string   `json:"caveat,omitempty"`
}

// DeferABSweep builds the defer ON/OFF report over ONE frozen tool-heavy request body. It
// runs the PRODUCTION #3232 transform (gateway.DeferColdToolsAB), prices the provider-
// resident tool slice on each arm with the house estimator, and folds the held-accuracy
// reachability witness. It fails closed on an empty body and on a stand-down (the lever did
// not fire) — a fabricated zero-delta row would read as a measured no-op.
func DeferABSweep(body []byte) (DeferABReport, error) {
	if len(body) == 0 {
		return DeferABReport{}, errors.New("ablate: defer A/B needs a non-empty tool-heavy request body")
	}
	arms := gateway.DeferColdToolsAB(body)
	if !arms.Changed {
		return DeferABReport{}, fmt.Errorf("ablate: cold-tool deferral did not fire (%s); no defer A/B over this workload", arms.Reason)
	}

	off := gateway.ResidentToolDefs(arms.Ablated)
	on := gateway.ResidentToolDefs(arms.Armed)
	offTokens := mcpfootprint.Price(off).Tools.Tokens
	onTokens := mcpfootprint.Price(on).Tools.Tokens

	held, total, dropped := heldAccuracy(off, arms.Armed)
	acc := 1.0
	if total > 0 {
		acc = float64(held) / float64(total)
	}

	sum := sha256.Sum256(arms.Ablated)
	return DeferABReport{
		ArmID:                    DeferABArmID,
		WorkloadHash:             hex.EncodeToString(sum[:]),
		ColdDeferred:             arms.ColdCount,
		OffResidentTools:         len(off),
		OnResidentTools:          len(on),
		OffResidentTokens:        offTokens,
		OnResidentTokens:         onTokens,
		ResidentTokenDelta:       offTokens - onTokens,
		HeldToolCount:            held,
		TotalToolCount:           total,
		HeldAccuracy:             acc,
		DroppedTools:             dropped,
		CachePrefixByteIdentical: gateway.NonToolsByteIdentical(arms.Ablated, arms.Armed),
		Caveat:                   deferABCaveat(),
	}, nil
}

// DeferABOverCanonical runs the defer ON/OFF arm over the canonical Claude-Code-shaped
// tool-heavy body (gateway.CanonicalDeferABBody — a hot core + the real MCP registry as the
// cold tail), the frozen trace the #3628 witness replays. It is the default artifact a CLI
// emits and the arm the package test binds.
func DeferABOverCanonical() (DeferABReport, error) {
	return DeferABSweep(gateway.CanonicalDeferABBody())
}

// heldAccuracy witnesses that deferral did not COST the agent a tool: every tool advertised
// in the OFF arm must still be ADVERTISED in the ARMED body (resident directly, or deferred
// but present in tools[] and discoverable via tool_search_tool). It decodes the armed body's
// FULL tool set — deferred defs stay in tools[], so the decode includes them — and checks
// each OFF name against it, returning the held count, the total, and the sorted names of any
// tool that went missing. A missing name is a real regression (a dropped def, not a deferred
// one), which is the falsifiable half of the held-accuracy figure.
func heldAccuracy(off []agent.ToolDef, armed []byte) (held, total int, dropped []string) {
	advertised := map[string]bool{}
	if req, err := agent.DecodeAnthropicMessagesRequest(armed); err == nil {
		for _, t := range req.Tools {
			advertised[t.Function.Name] = true
		}
	}
	for _, d := range off {
		total++
		if advertised[d.Function.Name] {
			held++
		} else {
			dropped = append(dropped, d.Function.Name)
		}
	}
	sort.Strings(dropped)
	return held, total, dropped
}

// AccuracyHeld reports whether deferral preserved every tool's reachability — true iff no
// tool advertised OFF went missing ON (held-accuracy == 1.0). This is the "held" the
// default-on gate reads: the token delta only counts if accuracy held.
func (r DeferABReport) AccuracyHeld() bool {
	return r.TotalToolCount > 0 && r.HeldToolCount == r.TotalToolCount
}

// SweepRow renders the human one-liner: the provider resident-token delta and the
// held-accuracy figure over the bound workload — the two signals the #3232 default-on gate
// is declared to need.
func (r DeferABReport) SweepRow() string {
	return fmt.Sprintf("%s: provider resident tokens %d→%d (Δ%+d est. faulted-in on demand) · held-accuracy %.3f (%d/%d tools reachable) · %d cold defs deferred over workload %s",
		r.ArmID, r.OffResidentTokens, r.OnResidentTokens, r.ResidentTokenDelta,
		r.HeldAccuracy, r.HeldToolCount, r.TotalToolCount, r.ColdDeferred, shortHash(r.WorkloadHash))
}

// JSON renders the report as canonical indented JSON terminated by a newline — the rendered
// artifact witness the done-condition names (feeds `fak cachevalue status --ablation-report`).
func (r DeferABReport) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return append(b, '\n')
}

func shortHash(h string) string {
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

func deferABCaveat() string {
	return "resident-token delta is ESTIMATED (the house tokenizer on the provider-resident tool slice: non-deferred defs OFF vs ON), NOT the provider's OBSERVED usage; defer_loading GROWS request bytes, so this is a provider-side RESIDENT reduction, never a byte shrink. Held-accuracy is the STRUCTURAL capability-retention proxy — the fraction of pre-defer tools still reachable armed (resident, or deferred+discoverable via tool_search_tool) — not a live task-success rate. Promotion evidence before defer flips default-on (#1844/#3232): a live run's usage relay + fak_gateway_tool_defer_* /metrics (#3233/#3536) for the OBSERVED resident delta, and a live task arm for real accuracy. Invalidating assumption: structural reachability holding does not prove a live agent SELECTS a deferred tool as readily as a resident one."
}
