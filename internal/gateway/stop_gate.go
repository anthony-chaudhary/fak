package gateway

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	// ReasonStopUnwitnessed is outside the frozen core reason range and is
	// registered additively by the gateway consumer.
	ReasonStopUnwitnessed     abi.ReasonCode = 1070
	ReasonStopUnwitnessedName                = "STOP_UNWITNESSED"
)

func init() {
	abi.RegisterReason(ReasonStopUnwitnessed, ReasonStopUnwitnessedName)
}

// recordStopGateHold writes the failed evidence check through the same emitter
// path as every other durable gateway decision. The witness is bounded to the
// declarative name supplied by the trusted host.
func (s *Server) recordStopGateHold(trace, witness string) {
	call := &abi.ToolCall{Tool: "session.stop", TraceID: trace}
	verdict := &abi.Verdict{
		Kind:    abi.VerdictDeny,
		Reason:  ReasonStopUnwitnessed,
		By:      "gateway.stop_gate",
		Payload: abi.WitnessPayload{Claim: witness},
	}
	ev := abi.Event{Kind: abi.EvDeny, Call: call, Verdict: verdict}
	for _, emitter := range abi.EmittersFor(abi.EvDeny) {
		emitter.Emit(ev)
	}
}
