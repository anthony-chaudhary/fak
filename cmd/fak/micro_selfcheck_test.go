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
