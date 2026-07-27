package auditusage

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func gwRow(ms int64, c gatewayusageledger.Counters) gatewayusageledger.Row {
	return gatewayusageledger.Row{Schema: gatewayusageledger.Schema, UnixMillis: ms, SessionType: "serve", Counters: c}
}

// Every gateway-usage ledger in existence today was written before the split
// existed, so this is the case the rollup will actually be in for a while. It must
// come back as a REFUSAL carrying a reason, not as a share of zero: a reader
// deciding whether the self-hosting build is worth continuing would read 0.0% as
// "it serves nothing" when the truth is "no row was asked".
func TestAGatewayCorpusWithNoSplitRefusesToReportAShare(t *testing.T) {
	rep := Fold(Input{
		Now: time.Unix(9000, 0),
		GatewayUsage: GatewayUsageInput{Path: "/x/gateway-usage.jsonl", Present: true, Rows: []gatewayusageledger.Row{
			gwRow(1000*1000, gatewayusageledger.Counters{OutputTokens: 4000}),
			gwRow(2000*1000, gatewayusageledger.Counters{OutputTokens: 6000}),
		}},
	})
	sh := rep.Gateway.SelfHosted
	if sh == nil {
		t.Fatal("rows were folded, so the rollup must state why it cannot answer")
	}
	if sh.OutputShare != nil {
		t.Errorf("OutputShare = %v, want nil — nothing was classified", *sh.OutputShare)
	}
	if sh.Reason != string(gatewayusageledger.ShareNotInstrumented) {
		t.Errorf("Reason = %q, want %q", sh.Reason, gatewayusageledger.ShareNotInstrumented)
	}
	// The unattributed volume is still reported, because "10000 tokens nobody
	// classified" is the size of the blind spot and the reason to go fix it.
	if sh.OutputTokens != 10000 {
		t.Errorf("OutputTokens = %d, want 10000", sh.OutputTokens)
	}
	if sh.ClassifiedOutputFraction != 0 {
		t.Errorf("ClassifiedOutputFraction = %v, want 0", sh.ClassifiedOutputFraction)
	}

	// And the refusal must survive the wire: omitempty on OutputShare is what
	// carries it, so a JSON consumer must find no share key at all rather than a
	// zero it will happily chart.
	b, err := json.Marshal(sh)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "output_share") {
		t.Errorf("declined share serialized a share key: %s", b)
	}
	if !strings.Contains(string(b), `"reason":"not_instrumented"`) {
		t.Errorf("want the reason on the wire: %s", b)
	}
}

// The answerable case, and the qualifier that has to travel with it.
func TestAnInstrumentedCorpusReportsAShareWithItsCoverage(t *testing.T) {
	rep := Fold(Input{
		Now: time.Unix(9000, 0),
		GatewayUsage: GatewayUsageInput{Path: "/x/gateway-usage.jsonl", Present: true, Rows: []gatewayusageledger.Row{
			// 600 local + 200 vendor classified, of 2000 output tokens served.
			gwRow(1000*1000, gatewayusageledger.Counters{
				OutputTokens:           1200,
				SelfHostedTurns:        3,
				SelfHostedOutputTokens: 600,
				VendorTurns:            1,
				VendorOutputTokens:     200,
			}),
			// A row from the same era that classified nothing: it drags coverage
			// down honestly instead of vanishing from the denominator.
			gwRow(2000*1000, gatewayusageledger.Counters{OutputTokens: 800}),
		}},
	})
	sh := rep.Gateway.SelfHosted
	if sh == nil || sh.OutputShare == nil {
		t.Fatalf("want a computed share, got %+v", sh)
	}
	if got := *sh.OutputShare; got != 0.75 {
		t.Errorf("OutputShare = %v, want 0.75 (600 of 800 CLASSIFIED, not of 2000 served)", got)
	}
	if got := sh.ClassifiedOutputFraction; got != 0.4 {
		t.Errorf("ClassifiedOutputFraction = %v, want 0.4 (800 of 2000)", got)
	}
	if sh.SelfHostedTurns != 3 || sh.VendorTurns != 1 {
		t.Errorf("turns = %d local / %d vendor, want 3/1", sh.SelfHostedTurns, sh.VendorTurns)
	}
	if sh.Reason != "" {
		t.Errorf("Reason = %q, want empty when a share was computed", sh.Reason)
	}
}

// An all-vendor corpus DID measure, and its zero is earned. It must be a real 0.0
// share rather than the same refusal an unmeasured corpus gets — the two are the
// opposite finding about a fleet.
func TestAnAllVendorCorpusReportsAnEarnedZero(t *testing.T) {
	rep := Fold(Input{
		Now: time.Unix(9000, 0),
		GatewayUsage: GatewayUsageInput{Present: true, Rows: []gatewayusageledger.Row{
			gwRow(1000*1000, gatewayusageledger.Counters{OutputTokens: 500, VendorTurns: 4, VendorOutputTokens: 500}),
		}},
	})
	sh := rep.Gateway.SelfHosted
	if sh == nil || sh.OutputShare == nil {
		t.Fatalf("a measured corpus must produce a share, got %+v", sh)
	}
	if *sh.OutputShare != 0 {
		t.Errorf("OutputShare = %v, want an earned 0", *sh.OutputShare)
	}
	if sh.Reason != "" {
		t.Errorf("an earned zero must carry no refusal reason, got %q", sh.Reason)
	}
	b, _ := json.Marshal(sh)
	if strings.Contains(string(b), "output_share") == false {
		t.Errorf("an earned zero must reach the wire as a share: %s", b)
	}
}

// The --since window and the rollup must agree: a share computed over rows the
// operator excluded is a different question than the one they asked.
func TestTheSelfHostedShareHonoursTheSinceWindow(t *testing.T) {
	rep := Fold(Input{
		Now:   time.Unix(9000, 0),
		Since: time.Unix(2000, 0),
		GatewayUsage: GatewayUsageInput{Present: true, Rows: []gatewayusageledger.Row{
			// Excluded: an all-local era before the window.
			gwRow(1000*1000, gatewayusageledger.Counters{OutputTokens: 900, SelfHostedTurns: 9, SelfHostedOutputTokens: 900}),
			gwRow(3000*1000, gatewayusageledger.Counters{OutputTokens: 100, VendorTurns: 1, VendorOutputTokens: 100}),
		}},
	})
	sh := rep.Gateway.SelfHosted
	if sh == nil || sh.OutputShare == nil {
		t.Fatalf("want a share over the window, got %+v", sh)
	}
	if *sh.OutputShare != 0 || sh.SelfHostedOutputTokens != 0 {
		t.Errorf("pre-window local volume leaked into the share: %+v", sh)
	}
	if sh.OutputTokens != 100 {
		t.Errorf("OutputTokens = %d, want 100 — only the windowed row", sh.OutputTokens)
	}
}

// No rows, no question. Declining to answer is a claim about instrumentation, and
// there is nothing to make that claim about when nothing was folded.
func TestAnEmptyGatewayWindowProducesNoSelfHostedRollup(t *testing.T) {
	rep := Fold(Input{Now: time.Unix(9000, 0)})
	if rep.Gateway.SelfHosted != nil {
		t.Errorf("want nil rollup for an empty corpus, got %+v", rep.Gateway.SelfHosted)
	}
	b, err := json.Marshal(rep.Gateway)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "self_hosted") {
		t.Errorf("empty gateway rollup serialized a self-hosted section: %s", b)
	}
}
