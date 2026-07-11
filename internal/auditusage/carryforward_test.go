package auditusage

// carryforward_test.go — after a gatewayusageledger.Cut (#3490), the rollup's
// session total must stay TRUE: a carryforward row expands back to the row
// count it folded, and never leaks into the oldest-vs-newest trend.

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func TestFoldGatewayExpandsCarryforwardRows(t *testing.T) {
	rows := []gatewayusageledger.Row{
		{Schema: gatewayusageledger.Schema, Kind: gatewayusageledger.KindCarryforward,
			SessionType: "serve", UnixMillis: 500,
			Carryforward: &gatewayusageledger.Carryforward{FoldedKind: "exit", FoldedRows: 7, FirstUnixMillis: 100, LastUnixMillis: 500}},
		{Schema: gatewayusageledger.Schema, Kind: "exit", SessionType: "serve", UnixMillis: 1000,
			Counters: gatewayusageledger.Counters{InputTokens: 10}},
		{Schema: gatewayusageledger.Schema, Kind: "exit", SessionType: "serve", UnixMillis: 2000,
			Counters: gatewayusageledger.Counters{InputTokens: 30}},
	}
	g := foldGateway(rows, time.Time{})
	if g.Sessions != 9 { // 7 folded + 2 real
		t.Fatalf("Sessions must stay true across a cut: want 9, got %d", g.Sessions)
	}
	if g.Trend == nil {
		t.Fatalf("two real rows must still trend")
	}
	if g.Trend.First.Kind == gatewayusageledger.KindCarryforward || g.Trend.DeltaInputTokens != 20 {
		t.Fatalf("carryforward leaked into the trend: %+v", g.Trend)
	}
}
