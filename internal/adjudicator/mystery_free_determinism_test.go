package adjudicator

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type mysteryFreeAdjustmentRun struct {
	Before []abi.Verdict `json:"before"`
	After  abi.Verdict   `json:"after"`
}

func TestMysteryFreeAdjustmentDeterminism(t *testing.T) {
	first := runMysteryFreeAdjustment(t)
	second := runMysteryFreeAdjustment(t)

	if !reflect.DeepEqual(first, second) {
		t.Fatalf("identical policy labs differ:\nfirst:  %#v\nsecond: %#v", first, second)
	}

	firstBytes, err := json.Marshal(first)
	if err != nil {
		t.Fatalf("marshal first policy lab: %v", err)
	}
	secondBytes, err := json.Marshal(second)
	if err != nil {
		t.Fatalf("marshal second policy lab: %v", err)
	}
	if !bytes.Equal(firstBytes, secondBytes) {
		t.Fatalf("identical policy labs are not byte-identical:\nfirst:  %s\nsecond: %s", firstBytes, secondBytes)
	}
}

func runMysteryFreeAdjustment(t *testing.T) mysteryFreeAdjustmentRun {
	t.Helper()

	policy := mysteryFreePolicy(false)
	monitor := New(policy)
	ctx := context.Background()

	run := mysteryFreeAdjustmentRun{
		Before: []abi.Verdict{
			monitor.Adjudicate(ctx, inlineCall("refund_payment", `{}`)),
			monitor.Adjudicate(ctx, inlineCall("search_kb", `{}`)),
			monitor.Adjudicate(ctx, inlineCall("mystery_action", `{}`)),
		},
	}

	monitor.SetPolicy(mysteryFreePolicy(true))
	run.After = monitor.Adjudicate(ctx, inlineCall("mystery_action", `{}`))

	want := []struct {
		verdict abi.Verdict
		kind    abi.VerdictKind
		reason  abi.ReasonCode
	}{
		{run.Before[0], abi.VerdictDeny, abi.ReasonPolicyBlock},
		{run.Before[1], abi.VerdictAllow, abi.ReasonNone},
		{run.Before[2], abi.VerdictDeny, abi.ReasonDefaultDeny},
		{run.After, abi.VerdictAllow, abi.ReasonNone},
	}
	for i, check := range want {
		if check.verdict.Kind != check.kind || check.verdict.Reason != check.reason {
			t.Fatalf("policy lab verdict %d = %v/%s, want %v/%s", i,
				check.verdict.Kind, abi.ReasonName(check.verdict.Reason),
				check.kind, abi.ReasonName(check.reason))
		}
	}

	return run
}

func mysteryFreePolicy(adjusted bool) Policy {
	allow := map[string]bool{}
	if adjusted {
		allow["mystery_action"] = true
	}
	return Policy{
		Allow:       allow,
		AllowPrefix: []string{"search_"},
		Deny: map[string]abi.ReasonCode{
			"refund_payment": abi.ReasonPolicyBlock,
		},
	}
}
