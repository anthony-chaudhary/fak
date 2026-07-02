package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestSessionEnvelopeFlagFormDispatchesOffline(t *testing.T) {
	var out, errb bytes.Buffer
	code := runSession(&out, &errb, []string{"envelope", "--tokens", "50000", "--wall-clock", "10m", "--turns", "25", "--spend", "$5", "--throughput", "20", "--json"})
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got struct {
		Envelope struct {
			Tokens          int     `json:"tokens"`
			SpendCapCents   int64   `json:"spend_cap_cents"`
			ThroughputFloor float64 `json:"throughput_floor"`
		} `json:"envelope"`
		Budget struct {
			TurnsLeft  int `json:"turns_left"`
			TokensLeft int `json:"tokens_left"`
		} `json:"budget"`
		TimeBudget struct {
			LimitNanos int64 `json:"limit_nanos"`
		} `json:"time_budget"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("json: %v\n%s", err, out.String())
	}
	if got.Envelope.Tokens != 50000 || got.Budget.TokensLeft != 50000 || got.Budget.TurnsLeft != 25 {
		t.Fatalf("parsed token/turn budget = %+v / %+v", got.Envelope, got.Budget)
	}
	if got.Envelope.SpendCapCents != 500 || got.Envelope.ThroughputFloor != 20 || got.TimeBudget.LimitNanos == 0 {
		t.Fatalf("parsed envelope axes = %+v time=%+v", got.Envelope, got.TimeBudget)
	}
}

func TestSessionEnvelopeLiveFormStillRequiresGatewayShape(t *testing.T) {
	var out, errb bytes.Buffer
	code := runSession(&out, &errb, []string{"envelope", "sess-1", "tokens=10", "--inspect-only"})
	if code != 0 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", code, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "tokens=10") {
		t.Fatalf("live inspect output missing parsed spec:\n%s", out.String())
	}
}
