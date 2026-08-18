package trajhook

// ctxhealth.go — issue #3098: the context-health VERDICT scorer. The closed
// vocabulary HEALTHY / STALL / DRIFT / DETOUR_OVERRUN already exists
// (internal/trajctl/curve.go, golden-tested) and this seam already invites
// "register your own, no core edit" — so a confusion/thrash verdict reuses
// both instead of inventing new signals or a new subsystem. The scorer folds
// the deterministic behavioral fields the corpus rows already carry (verdicts,
// args digests, queries, turn counts) into EXACTLY ONE verdict Finding per
// trace, labeled with the EXISTING trajctl Signal strings — never a new one.
// As the #3097 BehaviorLens fields and the #3096 span refcount reach the
// corpus rows, they fold into the same verdict here.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// defaultContextHealthLoopMin is the identical-call count at which a trace
// reads as STALL. The healthy 2-3 re-asks of iterative work stay silent; a
// fourth identical call is a loop, not iteration.
const defaultContextHealthLoopMin = 4

// ContextHealthConfig pins the deterministic thresholds the verdict fold
// reads. The zero value selects the reference defaults, so
// ContextHealth(ContextHealthConfig{}) is the reference configuration.
type ContextHealthConfig struct {
	// DenyRate is the DENY/QUARANTINE fraction at which a trace reads as
	// DRIFT — the high_deny_rate fold reused verbatim: a trace the kernel
	// kept refusing is a confused or adversarial agent loop, i.e. alignment
	// falling. Zero uses 0.5 (the Default registry's rate).
	DenyRate float64
	// DenyMinTurns is the minimum trace length before the deny fold may fire
	// (mirrors DenyRate's guard). Zero uses 2.
	DenyMinTurns int
	// LoopMin is the identical non-refused call count (same tool + same args
	// digest, or same tool + same query when no digest is stamped) at which a
	// trace reads as STALL — busy but not moving, the corpus-level analogue
	// of the BehaviorLens success-loop detector. Zero uses
	// defaultContextHealthLoopMin.
	LoopMin int
	// TurnBudget, when positive, is the declared per-trace turn budget; a
	// trace that runs past it reads as DETOUR_OVERRUN. It mirrors trajctl's
	// declared-budget rule (Objective.Budget.Turns): no declared budget, no
	// overrun. Zero disables the check.
	TurnBudget int
}

// ContextHealth returns a CorpusScorer that emits one context-health verdict
// per trace in the closed trajctl vocabulary. Unlike the anomaly scorers it
// ALWAYS emits — HEALTHY is a verdict, not a notable finding, so its Score is
// 0 and it sorts last in Run's worst-first output. Precedence mirrors
// trajctl's classify priority: the most actionable, most specific signal wins
// (DETOUR_OVERRUN > DRIFT > STALL > HEALTHY). The fold is pure and
// deterministic: the same corpus always yields the same verdicts.
func ContextHealth(cfg ContextHealthConfig) CorpusScorer {
	denyRate := cfg.DenyRate
	if denyRate <= 0 {
		denyRate = 0.5
	}
	denyMinTurns := cfg.DenyMinTurns
	if denyMinTurns <= 0 {
		denyMinTurns = 2
	}
	loopMin := cfg.LoopMin
	if loopMin <= 0 {
		loopMin = defaultContextHealthLoopMin
	}
	return func(corpus []trajectory.Turn) []Finding {
		// Reuse high_deny_rate verbatim: its trace-level findings ARE the
		// DRIFT evidence, thresholds and all.
		drift := map[string]Finding{}
		for _, f := range DenyRate(denyRate, denyMinTurns)(corpus) {
			drift[f.TraceID] = f
		}
		type traceAcc struct {
			turns     int
			loops     map[string]int
			maxLoop   int
			loopTool  string
			loopQuery string
		}
		byTrace := map[string]*traceAcc{}
		order := []string{}
		for _, t := range corpus {
			a := traceAccumulator(byTrace, &order, t, func() *traceAcc { return &traceAcc{loops: map[string]int{}} })
			a.turns++
			if t.Verdict == "DENY" || t.Verdict == "QUARANTINE" {
				continue // refused turns are the deny fold's evidence, not loop activity
			}
			key := contextHealthLoopKey(t)
			if key == "" {
				continue
			}
			a.loops[key]++
			if a.loops[key] > a.maxLoop {
				a.maxLoop = a.loops[key]
				a.loopTool, a.loopQuery = t.Tool, t.Query
			}
		}
		out := make([]Finding, 0, len(order))
		for _, id := range order {
			a := byTrace[id]
			f := Finding{TraceID: id}
			deny, drifting := drift[id]
			switch {
			case cfg.TurnBudget > 0 && a.turns > cfg.TurnBudget:
				f.Label = string(trajctl.SignalDetourOverrun)
				f.Score = float64(a.turns - cfg.TurnBudget)
				f.Reason = fmt.Sprintf("trace ran %d turns past its declared %d-turn budget",
					a.turns-cfg.TurnBudget, cfg.TurnBudget)
			case drifting:
				f.Label = string(trajctl.SignalDrift)
				f.Score = deny.Score
				f.Reason = deny.Reason + " — a confused or adversarial loop, alignment falling"
			case a.maxLoop >= loopMin:
				f.Label = string(trajctl.SignalStall)
				f.Score = float64(a.maxLoop)
				f.Reason = fmt.Sprintf("busy but not moving: %d identical %s calls (%s)",
					a.maxLoop, a.loopTool, truncate(a.loopQuery, 60))
			default:
				f.Label = string(trajctl.SignalHealthy)
				f.Reason = fmt.Sprintf("no stall/drift/overrun evidence over %d turns", a.turns)
			}
			out = append(out, f)
		}
		return out
	}
}

// contextHealthLoopKey is a turn's identical-call identity for the STALL fold:
// the stamped args digest when present (content identity of the call), else
// the verbatim query. A turn with neither carries no loop evidence.
func contextHealthLoopKey(t trajectory.Turn) string {
	if t.ArgsDigest != "" {
		return t.Tool + "\x1f" + t.ArgsDigest
	}
	if t.Query != "" {
		return t.Tool + "\x1f" + t.Query
	}
	return ""
}
