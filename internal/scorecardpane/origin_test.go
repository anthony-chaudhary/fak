package scorecardpane

import "testing"

func TestOriginHooksExposeFiveCheapGoBackedCards(t *testing.T) {
	want := map[string]ProducerOrigin{
		"tokendefaults": {Command: "token-defaults-scorecard", File: "cmd/fak/token_defaults.go"},
		"guard_rsi":     {Command: "guard-rsi-scorecard", File: "cmd/fak/guardrsi.go"},
		"conceptusage":  {Command: "concept-usage-score", File: "cmd/fak/conceptusagescore.go"},
		"negframe":      {Command: "score negframe", File: "cmd/fak/negframescore.go"},
		"negation_tax":  {Command: "score negation-tax", File: "cmd/fak/negationtaxscore.go"},
	}
	seen := 0
	for _, card := range Cards {
		origin, ok := want[card.Key]
		if !ok {
			continue
		}
		seen++
		if card.Origin == nil || *card.Origin != origin {
			t.Errorf("%s origin = %#v, want %#v", card.Key, card.Origin, origin)
		}
		if card.Cmd == "" || !goBackedKey(card.Key) {
			t.Errorf("%s origin hook is not Go-backed", card.Key)
		}
		metric := MetricFromPayload(card, map[string]any{card.Debt: float64(0), "ok": true, "verdict": "OK"}, "")
		if metric.Origin == nil || *metric.Origin != origin {
			t.Errorf("%s metric lost origin: %#v", card.Key, metric.Origin)
		}
	}
	if seen != len(want) {
		t.Fatalf("saw %d origin hooks, want %d", seen, len(want))
	}
}

func TestOriginMetadataAbsentForAggregateOnlyCard(t *testing.T) {
	metric := MetricFromPayload(Card{Key: "aggregate", Debt: "debt", Label: "aggregate"}, map[string]any{"debt": float64(0)}, "")
	if metric.Origin != nil {
		t.Fatalf("aggregate-only card unexpectedly has origin: %#v", metric.Origin)
	}
}
