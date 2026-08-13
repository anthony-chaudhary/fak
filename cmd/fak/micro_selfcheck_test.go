package main

import (
	"context"
	"testing"
	"time"
)

func TestMicroSelfcheckCapturesKernelValueChain(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	receipt, err := runMicroSelfcheck(ctx)
	if err != nil {
		t.Fatalf("runMicroSelfcheck: %v; receipt=%+v", err, receipt)
	}
	if receipt.Schema != "fak-micro-selfcheck/2" || receipt.ParentTaskID == "" || len(receipt.Children) != 2 {
		t.Fatalf("missing parent receipt: %+v", receipt)
	}
	for _, child := range receipt.Children {
		if child.LeaseID == "" || child.SessionID == "" || child.State != "stopped" || !child.Witnessed || child.EffectDigest == "" {
			t.Fatalf("incomplete child receipt: %+v", child)
		}
	}
	if receipt.Verdict != "PASS" || receipt.Done != 2 || receipt.HTTPCount != 2 || receipt.StoppedCount != 2 {
		t.Fatalf("incomplete receipt: %+v", receipt)
	}
	if receipt.ProviderTokens <= 0 {
		t.Fatalf("provider usage was not observed: %+v", receipt)
	}
	if !receipt.Offline {
		t.Fatalf("selfcheck did not identify itself as offline: %+v", receipt)
	}
}
