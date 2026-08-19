package privacy

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
	"testing"
	"time"
)

func TestCanonicalTrajectoryEventAcrossSinks(t *testing.T) {
	event, _ := trajectory.NewRuntimeEvent("e", "s", "turn", "trace", 1, time.Now(), trajectory.RuntimeError, trajectory.RuntimeSource{Rung: "loop", Component: "loop", Instance: "test", Runtime: "fak"}, json.RawMessage(`{"message":"safe","secret":"hidden"}`))
	payload, _ := json.Marshal(event)
	p := DefaultPolicy()
	p.LocalOnly = false
	p.Export = SinkPolicy{Enabled: true, RedactFields: []string{"secret"}}
	p.Telemetry = SinkPolicy{Enabled: false}
	local, _ := p.Evaluate(SinkLog, payload, time.Now())
	exported, _ := p.Evaluate(SinkExport, payload, time.Now())
	telemetry, _ := p.Evaluate(SinkTelemetry, payload, time.Now())
	if len(local.Payload) == 0 || exported.Receipt.Action != ActionRedact || telemetry.Receipt.Action != ActionDeny {
		t.Fatalf("local=%+v export=%+v telemetry=%+v", local, exported, telemetry)
	}
}
