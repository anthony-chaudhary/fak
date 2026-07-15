package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func promotionEvent(tool string, kind abi.VerdictKind, reason abi.ReasonCode, posture string) abi.Event {
	v := &abi.Verdict{Kind: kind, Reason: reason, Meta: map[string]string{}}
	if posture != "" {
		v.Meta["posture"] = posture
	}
	return abi.Event{Kind: abi.EvDecide, Call: &abi.ToolCall{Tool: tool}, Verdict: v}
}

func TestGuardPromotionLedgerAndRunnableOffer(t *testing.T) {
	ledger := configureGuardPromotionLedger(map[string]bool{"custom_tool": true}, 2)
	if ledger == nil {
		t.Fatal("complain set did not register ledger")
	}
	ledger.Emit(promotionEvent("custom_tool", abi.VerdictAllow, 0, "admit_and_log"))
	ledger.Emit(promotionEvent("custom_tool", abi.VerdictAllow, 0, "admit_and_log"))
	var out bytes.Buffer
	if got := renderGuardPromotionOffers(&out); got != 1 {
		t.Fatalf("offers=%d output=%s", got, out.String())
	}
	if !strings.Contains(out.String(), "fak guard allow custom_tool") {
		t.Fatalf("output lacks runnable command: %s", out.String())
	}
}

func TestGuardPromotionHardRefusalSuppressesOffer(t *testing.T) {
	ledger := configureGuardPromotionLedger(map[string]bool{"custom_tool": true}, 2)
	ledger.Emit(promotionEvent("custom_tool", abi.VerdictAllow, 0, "admit_and_log"))
	ledger.Emit(promotionEvent("custom_tool", abi.VerdictAllow, 0, "admit_and_log"))
	ledger.Emit(promotionEvent("custom_tool", abi.VerdictDeny, abi.ReasonPolicyBlock, ""))
	var out bytes.Buffer
	if got := renderGuardPromotionOffers(&out); got != 0 || out.Len() != 0 {
		t.Fatalf("hard refusal offer=%d output=%s", got, out.String())
	}
}

func TestGuardPromotionEmptyComplainSetHasNoOutputDelta(t *testing.T) {
	if ledger := configureGuardPromotionLedger(nil, 1); ledger != nil {
		t.Fatal("empty set registered ledger")
	}
	var out bytes.Buffer
	if got := renderGuardPromotionOffers(&out); got != 0 || out.Len() != 0 {
		t.Fatalf("offers=%d output=%s", got, out.String())
	}
}
