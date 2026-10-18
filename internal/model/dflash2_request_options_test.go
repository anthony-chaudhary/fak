package model

import (
	"context"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestDFlash2TypedPanelBudgets(t *testing.T) {
	for _, tc := range []struct {
		mode  DFlash2TargetPanel
		width int
	}{
		{DFlash2TargetSerial, 8}, {DFlash2TargetPanel4, 4}, {DFlash2TargetPanel8, 8},
	} {
		o := Qwen38DFlash2RequestOptions{TargetPanel: tc.mode}
		for requested := 1; requested <= 8; requested++ {
			got, err := o.blockBudget(requested)
			if err != nil || got != min(requested, tc.width) {
				t.Fatalf("mode %d budget %d: %d %v", tc.mode, requested, got, err)
			}
			if got-1 > tc.width-1 {
				t.Fatal("anchor was omitted from panel width")
			}
		}
	}
	for _, n := range []int{0, 9} {
		if _, err := (Qwen38DFlash2RequestOptions{}).blockBudget(n); err == nil {
			t.Fatal("invalid budget admitted")
		}
	}
	if _, err := (Qwen38DFlash2RequestOptions{TargetPanel: 255}).blockBudget(1); err == nil {
		t.Fatal("unknown panel admitted")
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2TypedDefaultReceipt(t *testing.T) {
	q, plain := dflash2RequestFixture(t, true)
	prompt := []int{3, 7, 11}
	if err := q.Bootstrap(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	want := dflash2RequestPlain(prompt, plain)
	maximum := 4
	round, timing, err := q.PrepareTimed(context.Background(), maximum)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = round.Abort() })
	if !timing.Complete || !timing.ProposalRan || timing.ProposedDraftTokens != maximum-1 ||
		timing.Verification.Rows != maximum || timing.Verification.ConfiguredPanelWidth != 0 ||
		timing.Verification.DecodeSteps != maximum || timing.Verification.Operations != 0 ||
		timing.Verification.OneOperation || timing.Verification.Path != targetVerificationDecodePath {
		t.Fatalf("actual serial receipt: %+v", timing)
	}
	ids, err := round.Commit(context.Background(), len(round.Tokens()))
	if err != nil || len(ids) != maximum {
		t.Fatalf("commit %v %v", ids, err)
	}
	for _, id := range ids {
		if id != argmaxF32(want) {
			t.Fatal("token mismatch")
		}
		want = append([]float32(nil), plain.Step(id)...)
	}
	dflash2RequestEqual(t, q, plain, want)
	round, timing, err = q.PrepareTimed(context.Background(), 1)
	if err != nil {
		t.Fatal(err)
	}
	if timing.ProposalRan || timing.ProposedDraftTokens != 0 || timing.Verification.Rows != 1 {
		t.Fatalf("anchor-only %+v", timing)
	}
	if err = round.Abort(); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime medium est=2s lane=default
func TestDFlash2TypedPanelRefusesWithoutFallback(t *testing.T) {
	q, plain := dflash2RequestFixture(t, true)
	prompt := []int{3, 7}
	if err := q.Bootstrap(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	want := dflash2RequestPlain(prompt, plain)
	q.options = Qwen38DFlash2RequestOptions{TargetPanel: DFlash2TargetPanel4}
	round, timing, err := q.PrepareTimed(context.Background(), 1)
	if err == nil || round != nil || timing.Complete {
		t.Fatal("missing native panel silently used serial")
	}
	dflash2RequestEqual(t, q, plain, want)
	if q.pending != nil {
		t.Fatal("failed selection left a pending round")
	}
	if _, err := NewQwen38DFlash2RequestWithOptions(context.Background(), nil, "/must-not-open", Qwen38DFlash2RequestOptions{TargetPanel: 255}); err == nil {
		t.Fatal("unknown policy reached constructor")
	}
}
