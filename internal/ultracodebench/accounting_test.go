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
