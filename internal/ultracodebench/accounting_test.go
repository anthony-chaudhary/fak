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

func TestAccountingDeterminism(t *testing.T) {
	receipt := knownAccounting(101, 23, 89, 7, 131, 0.42, "sha256:"+strings.Repeat("b", 64))
	first, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		next, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		if string(next) != string(first) {
			t.Fatalf("marshal %d differs:\nfirst=%s\nnext=%s", i, first, next)
		}
		decoded, err := DecodeAccounting(next)
		if err != nil {
			t.Fatal(err)
		}
		roundTrip, err := json.Marshal(decoded)
		if err != nil {
			t.Fatal(err)
		}
		if string(roundTrip) != string(first) {
			t.Fatalf("round trip %d differs:\nfirst=%s\nnext=%s", i, first, roundTrip)
		}
	}
}

func TestAccountingEdgeAdversarialDecode(t *testing.T) {
	valid, err := json.Marshal(knownAccounting(1, 1, 0, 0, 2, 0.01, "sha256:"+strings.Repeat("c", 64)))
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		data []byte
		want string
	}{
		{name: "empty", data: nil, want: "decode accounting receipt"},
		{name: "malformed", data: []byte(`{"schema":`), want: "decode accounting receipt"},
		{name: "trailing-value", data: append(append([]byte{}, valid...), []byte(` {}`)...), want: "trailing JSON value"},
		{name: "hostile-unknown-field", data: []byte(`{"schema":"fak.ultracode.accounting.v1","raw_prompt":"secret"}`), want: "unknown field"},
		{name: "oversized-number", data: []byte(`{"schema":"fak.ultracode.accounting.v1","input_tokens":{"availability":"available","authority":"provider_usage","coverage":1,"value":999999999999999999999999999999}}`), want: "cannot unmarshal number"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := DecodeAccounting(tc.data)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error = %v, want refusal %q", err, tc.want)
			}
		})
	}
}
