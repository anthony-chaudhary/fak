package main

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// TestOpenCodeRunReceiptSessionJoinOmitEmptyAbsentWhenZero pins the backward-compat
// half of #1559: a receipt with no session join must carry NONE of the four additive
// fields, so fak-opencode-run/1 stays byte-equivalent to the pre-#1559 shape.
func TestOpenCodeRunReceiptSessionJoinOmitEmptyAbsentWhenZero(t *testing.T) {
	r := OpenCodeRunReceipt{
		Schema:     cronOpenCodeRunSchema,
		RunID:      "run-1559-zero",
		SessionID:  "ses_1559_zero",
		ExitCode:   0,
		Outcome:    "succeeded",
		WitnessRef: nil,
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{"tokens_total", "cost_usd", "model_id", "provider_id"} {
		if strings.Contains(string(b), `"`+key+`"`) {
			t.Errorf("zero-valued %q must be omitted (omitempty), got: %s", key, string(b))
		}
	}
}

// TestOpenCodeRunReceiptSessionJoinPresentWhenSet pins the set half: when the join
// resolves, all four fields marshal under their json names.
func TestOpenCodeRunReceiptSessionJoinPresentWhenSet(t *testing.T) {
	r := OpenCodeRunReceipt{
		Schema:      cronOpenCodeRunSchema,
		RunID:       "run-1559-set",
		SessionID:   "ses_1559_set",
		ExitCode:    0,
		Outcome:     "succeeded",
		WitnessRef:  nil,
		TokensTotal: 1234,
		CostUSD:     0.0567,
		ModelID:     "deepseek-v4.1-flash",
		ProviderID:  "deepseek",
	}
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if got := m["tokens_total"]; got != float64(1234) {
		t.Errorf("tokens_total = %v, want 1234", got)
	}
	if got := m["cost_usd"]; got != 0.0567 {
		t.Errorf("cost_usd = %v, want 0.0567", got)
	}
	if got := m["model_id"]; got != "deepseek-v4.1-flash" {
		t.Errorf("model_id = %v", got)
	}
	if got := m["provider_id"]; got != "deepseek" {
		t.Errorf("provider_id = %v", got)
	}
}

// TestCronPopulateReceiptSessionJoinFoldsTokensFromUsageLedger is the join witness: a
// receipt whose SessionID has a gateway-usage row gets its total token volume folded in
// from the PUBLIC gatewayusageledger path, with the newest row winning.
func TestCronPopulateReceiptSessionJoinFoldsTokensFromUsageLedger(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gateway-usage.jsonl")
	rows := []gatewayusageledger.Row{
		{
			Schema:     gatewayusageledger.Schema,
			Kind:       "exit",
			SessionID:  "ses_1559_join",
			PID:        1,
			UnixMillis: 1000,
			Counters: gatewayusageledger.Counters{
				InputTokens: 100, OutputTokens: 10, CachedPromptTokens: 5, CacheCreationTokens: 2,
			},
		},
		{
			Schema:     gatewayusageledger.Schema,
			Kind:       "exit",
			SessionID:  "ses_1559_join",
			PID:        1,
			UnixMillis: 2000,
			Counters: gatewayusageledger.Counters{
				InputTokens: 300, OutputTokens: 40, CachedPromptTokens: 50, CacheCreationTokens: 10,
			},
		},
		{
			Schema:     gatewayusageledger.Schema,
			Kind:       "exit",
			SessionID:  "ses_other",
			PID:        1,
			UnixMillis: 3000,
			Counters:   gatewayusageledger.Counters{InputTokens: 999999},
		},
	}
	for _, r := range rows {
		if err := gatewayusageledger.Append(path, r); err != nil {
			t.Fatalf("append row: %v", err)
		}
	}

	t.Setenv("FAK_OPENCODE_USAGE_LEDGER", path)

	r := OpenCodeRunReceipt{Schema: cronOpenCodeRunSchema, RunID: "run-1559-join", SessionID: "ses_1559_join"}
	cronPopulateReceiptSessionJoin(&r)

	const want = int64(300 + 40 + 50 + 10)
	if r.TokensTotal != want {
		t.Errorf("TokensTotal = %d, want %d (newest join row)", r.TokensTotal, want)
	}
	// No priced/model source in the public path yet: those stay zero and are omitted.
	if r.CostUSD != 0 || r.ModelID != "" || r.ProviderID != "" {
		t.Errorf("unmeasured axes must stay zero: cost=%v model=%q provider=%q", r.CostUSD, r.ModelID, r.ProviderID)
	}
}

// TestCronPopulateReceiptSessionJoinIsBestEffortOnMissingLedger pins that a missing
// ledger, an empty session id, or an unknown session never populates and never panics.
func TestCronPopulateReceiptSessionJoinIsBestEffortOnMissingLedger(t *testing.T) {
	t.Setenv("FAK_OPENCODE_USAGE_LEDGER", filepath.Join(t.TempDir(), "does-not-exist.jsonl"))

	unknown := OpenCodeRunReceipt{SessionID: "ses_not_joined"}
	cronPopulateReceiptSessionJoin(&unknown)
	if unknown.TokensTotal != 0 {
		t.Errorf("unknown session must not join, got TokensTotal=%d", unknown.TokensTotal)
	}

	empty := OpenCodeRunReceipt{SessionID: ""}
	cronPopulateReceiptSessionJoin(&empty)
	if empty.TokensTotal != 0 {
		t.Errorf("empty session id must not join, got TokensTotal=%d", empty.TokensTotal)
	}

	cronPopulateReceiptSessionJoin(nil)
}
