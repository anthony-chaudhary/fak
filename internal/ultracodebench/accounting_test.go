package ultracodebench

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestAccountingReceiptCarriesSixIndependentAxes(t *testing.T) {
	receipt := knownAccounting(10, 2, 8, 1, 12, .05, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
	raw, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeAccounting(raw)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.InputTokens.Authority != AuthorityProviderUsage || decoded.BilledTokens.Authority != AuthorityProviderBilling || decoded.SpendUSD.Authority != AuthorityProviderBilling {
		t.Fatalf("receipt conflated usage and billing authority: %+v", decoded)
	}
	if decoded.InputTokens.Coverage != 1 || decoded.CacheWriteTokens.Value == nil || *decoded.CacheWriteTokens.Value != 1 {
		t.Fatalf("receipt lost per-axis coverage/value: %+v", decoded)
	}
}

func TestAccountingReceiptRejectsUnredactedOrMalformedFields(t *testing.T) {
	raw := `{"schema":"fak.ultracode.accounting.v1","raw_prompt":"secret"}`
	if _, err := DecodeAccounting([]byte(raw)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unredacted extension accepted: %v", err)
	}
}

func TestAccountingRefusalsNameRecovery(t *testing.T) {
	base := knownAccounting(10, 5, 2, 1, 16, 0.01, "sha256:"+strings.Repeat("a", 64))
	cases := []struct {
		name string
		edit func(*AccountingReceipt)
		want string
	}{
		{name: "invalid-schema", edit: func(r *AccountingReceipt) { r.Schema = "hostile/99" }, want: "schema must be"},
		{name: "negative-token-count", edit: func(r *AccountingReceipt) { r.InputTokens.Value = int64Ptr(-1) }, want: "input_tokens value cannot be negative"},
		{name: "invalid-coverage", edit: func(r *AccountingReceipt) { r.SpendUSD.Coverage = 1.1 }, want: "spend_usd coverage must be in [0,1]"},
		{name: "raw-provenance", edit: func(r *AccountingReceipt) { r.SpendUSD.ArtifactDigest = "secret bearer token" }, want: "spend_usd artifact_digest must be a sha256 digest"},
		{name: "missing-reason", edit: func(r *AccountingReceipt) {
			r.SpendUSD = SpendAccounting{Availability: AccountingUnavailable, Authority: AuthorityUnreported}
		}, want: "spend_usd unavailable value requires zero coverage, no value, and a reason"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receipt := base
			tc.edit(&receipt)
			err := receipt.Validate()
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want recovery cue %q", err, tc.want)
			}
		})
	}
}

func int64Ptr(value int64) *int64 { return &value }
