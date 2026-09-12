package gateway

import (
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/turncost"
)

// newTurnCostRecord opens a per-turn cost record for a served turn, or returns nil
// when the surface is disabled. A nil record makes every timing hook a no-op, so the
// request path stays byte-for-byte historical when FAK_TURN_COST is off.
func newTurnCostRecord(streaming bool) *turncost.TurnCostRecord {
	if (!streaming && !turncost.Enabled()) || (streaming && !turncost.EnabledStreaming()) {
		return nil
	}
	return &turncost.TurnCostRecord{Streaming: streaming}
}

// observeTurnCost folds a completed turn's record into the gateway's turn-cost
// collector. Nil-safe at every step: a disabled surface, nil collector, or nil
// record is a no-op. Fail-open — it never perturbs the request path.
func (s *Server) observeTurnCost(rec *turncost.TurnCostRecord) {
	if s == nil || rec == nil {
		return
	}
	s.turnCost.Observe(rec)
}

// writeTurnCostMetrics folds the fak_turn_cost_* family into the gateway /metrics
// exposition. A nil collector emits nothing, so a construction path that never
// armed the surface leaves the family absent.
func (s *Server) writeTurnCostMetrics(b *strings.Builder) {
	if s == nil || s.turnCost == nil {
		return
	}
	b.WriteString(s.turnCost.Prometheus())
}

// timePhase runs fn and accumulates its wall time into rec under phase p. It is a
// best-effort timing helper: a nil record just executes fn. Negative durations are
// dropped by TurnCostRecord.Add, so a coarse clock can never fabricate a sample.
func timePhase(rec *turncost.TurnCostRecord, p turncost.Phase, fn func()) {
	if rec == nil {
		fn()
		return
	}
	start := time.Now()
	fn()
	rec.Add(p, time.Since(start))
}

// recordBufferedTurnCost fills the prefill/decode phases of a buffered turn from the
// native inference receipt when the engine measured them. Without a native receipt,
// the direct planner call is measured as proxy-hop wall time; backend phases stay absent.
func (s *Server) recordBufferedTurnCost(turn servedSessionTurn, comp *agent.Completion, began time.Time) {
	rec := turn.turnCost
	if rec == nil || comp == nil {
		return
	}
	rec.Model = comp.Model
	if ni := comp.NativeInference; ni != nil {
		if ni.PrefillSeconds > 0 {
			rec.SetPhase(turncost.PhasePrefill, time.Duration(ni.PrefillSeconds*float64(time.Second)))
		}
		if ni.DecodeSeconds > 0 {
			rec.SetPhase(turncost.PhaseDecode, time.Duration(ni.DecodeSeconds*float64(time.Second)))
		}
		return
	}
	rec.Add(turncost.PhaseProxyHop, time.Since(began))
}

// finishTurnCost stamps the identity/total fields and folds the record into the
// gateway collector. A nil record is a no-op.
func (s *Server) finishTurnCost(turn servedSessionTurn, trace, model string, began time.Time) {
	rec := turn.turnCost
	if rec == nil {
		return
	}
	if rec.Model != "" {
		model = rec.Model
	}
	stampTurnCost(rec, trace, model, began)
	s.observeTurnCost(rec)
}

// stampTurnCost prepares the receipt snapshot before serialization. Metrics are folded
// after ledger completion and include the later response emission cost.
func stampTurnCost(rec *turncost.TurnCostRecord, trace, model string, began time.Time) {
	if rec == nil {
		return
	}
	rec.Trace = trace
	rec.Model = model
	rec.TotalSeconds = time.Since(began).Seconds()
}
