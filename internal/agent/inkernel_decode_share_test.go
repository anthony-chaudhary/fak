package agent

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestDecodeShareDebtOwedAndZeroKeepsWholePromptOrder pins the wall-time split
// adapted from TensorFold into the in-kernel coalescer: a prompt step taking t
// ns while decoders are live owes decode time t*(1-p)/p (p floored at
// inKernelDecodeShareMinPrompt), decode rounds pay the debt down, and a decode
// share outside (0,1) — including 0 — keeps whole-prompt arrival ordering. The
// optional row budget caps the rows a visit fills beside live decoders.
// Source: ashhart/TensorFold zig/src/core/lane_host_fill.zig @9e8c368b0b4c2c7892a0c9f0f302d7c164016209
// plus docs/api.md:236 (Apache-2.0, ADAPT; see THIRD_PARTY_NOTICES.md).
//
// fak-test:runtime fast est=100ms lane=default
func TestDecodeShareDebtOwedAndZeroKeepsWholePromptOrder(t *testing.T) {
	t.Run("debt owed and paid down", func(t *testing.T) {
		half := newInKernelDecodeShare(0.5, 0) // p = 0.5: a step of t owes t
		half.stepped(1000, true)
		if half.debtNs != 1000 {
			t.Fatalf("p=0.5 debt=%d, want 1000", half.debtNs)
		}
		half.stepped(1000, false) // No live decoders: nothing owed.
		if half.debtNs != 1000 {
			t.Fatalf("idle step changed debt to %d, want 1000", half.debtNs)
		}
		half.rounded(400)
		half.rounded(800)
		if half.debtNs != 0 {
			t.Fatalf("rounds left debt %d, want 0", half.debtNs)
		}

		quarter := newInKernelDecodeShare(0.75, 0) // p = 0.25: a step of t owes 3t
		quarter.stepped(1000, true)
		if quarter.debtNs != 3000 {
			t.Fatalf("p=0.25 debt=%d, want 3000", quarter.debtNs)
		}

		floored := newInKernelDecodeShare(0.999, 0) // p floors at 0.01: a step of t owes 99t
		if floored.promptShare != inKernelDecodeShareMinPrompt {
			t.Fatalf("floored prompt share=%v, want %v", floored.promptShare, inKernelDecodeShareMinPrompt)
		}
		floored.stepped(1000, true)
		if floored.debtNs != 99000 {
			t.Fatalf("floored debt=%d, want 99000", floored.debtNs)
		}

		if floored.due(true) || !half.due(true) {
			t.Fatalf("due(decoding) floored=%v half=%v, want false/true", floored.due(true), half.due(true))
		}
		if !floored.due(false) {
			t.Fatal("owed debt still blocks when nothing is decoding")
		}
	})

	t.Run("zero share keeps whole-prompt order", func(t *testing.T) {
		for _, fraction := range []float64{0, -0.5, 1, 2, math.NaN(), math.Inf(1)} {
			share := newInKernelDecodeShare(fraction, 0)
			if share.promptShare != 1 {
				t.Fatalf("decode share %v gave prompt share %v, want 1", fraction, share.promptShare)
			}
			share.stepped(1_000_000, true)
			if share.debtNs != 0 {
				t.Fatalf("decode share %v owed debt %d, want 0", fraction, share.debtNs)
			}
			if !share.due(true) {
				t.Fatalf("decode share %v deferred a prompt step with no debt", fraction)
			}
		}
	})

	t.Run("debt caps at the 2s window instead of unbounded", func(t *testing.T) {
		share := newInKernelDecodeShare(0.99, 0)
		share.stepped(math.MaxInt64, true)
		if share.debtNs != inKernelDecodeShareDebtCapNs {
			t.Fatalf("max step left debt %d, want capped %d", share.debtNs, inKernelDecodeShareDebtCapNs)
		}
		share.stepped(math.MaxInt64, true)
		if share.debtNs != inKernelDecodeShareDebtCapNs {
			t.Fatalf("debt=%d, want capped %d", share.debtNs, inKernelDecodeShareDebtCapNs)
		}
		share.rounded(1000)
		if share.debtNs != inKernelDecodeShareDebtCapNs-1000 {
			t.Fatal("rounded did not reduce the capped debt")
		}
	})

	t.Run("visit limit caps rows beside decoders", func(t *testing.T) {
		cases := []struct {
			budget, planned, want int
		}{
			{0, 5, 5},  // No budget: uncapped.
			{2, 5, 2},  // Budget below the plan.
			{8, 5, 5},  // Budget above the plan.
			{-1, 5, 5}, // Negative parses as no budget.
			{0, 0, 0},  // Empty visit stays empty.
		}
		for _, tc := range cases {
			got := newInKernelDecodeShare(0.5, tc.budget).visitLimit(tc.planned)
			if got != tc.want {
				t.Errorf("budget=%d planned=%d visitLimit=%d, want %d", tc.budget, tc.planned, got, tc.want)
			}
		}
	})

	t.Run("env reader falls back on bad text", func(t *testing.T) {
		t.Setenv(inKernelDecodeShareEnv, "junk")
		t.Setenv(inKernelPromptRowBudgetEnv, "junk")
		share := inKernelDecodeShareFromEnv()
		if share.promptShare != 1 || share.rowBudget != 0 {
			t.Fatalf("bad env gave %+v, want default p=1 no budget", share)
		}
		t.Setenv(inKernelDecodeShareEnv, "0.5")
		t.Setenv(inKernelPromptRowBudgetEnv, "3")
		share = inKernelDecodeShareFromEnv()
		if share.promptShare != 0.5 || share.rowBudget != 3 {
			t.Fatalf("env gave %+v, want p=0.5 budget=3", share)
		}
		t.Setenv(inKernelPromptRowBudgetEnv, "0")
		if share := inKernelDecodeShareFromEnv(); share.rowBudget != 0 {
			t.Fatalf("zero budget parsed as %d, want none", share.rowBudget)
		}
	})

	v41ShareEpisode := func(t *testing.T) (batches [][]uint64) {
		t.Helper()
		planner := &InKernelPlanner{m: model.NewSynthetic(tinyCfg())}
		requestA, _, _, cancelA := v41LateFillRequest(6)
		requestB, _, _, cancelB := v41LateFillRequest(2)
		requestC, _, _, cancelC := v41LateFillRequest(2)
		defer func() {
			cancelA()
			cancelB()
			cancelC()
			v41LateFillClearQueue(planner)
		}()
		v41LateFillQueue(planner, requestB, requestC)
		// batches is written only by the cohort goroutine and read after its
		// done channel closes, so no lock is needed.
		operations := inKernelV41DecodeOps{
			batch: func(input []model.V41BatchLane) (model.V41BatchResult, error) {
				batches = append(batches, v41LateFillIDs(input))
				return v41LateFillSelected(input), nil
			},
			serial: func(*model.Session, int) ([]float32, error) {
				return nil, errors.New("unexpected serial fallback")
			},
			wait: func(context.Context) error { return nil },
		}
		done := make(chan struct{})
		go func() {
			planner.runV41DecodeCohort([]*inKernelCoalesceRequest{requestA}, func(ctx context.Context, lanes []*decodeLane, _ *model.Model) (inKernelV41DecodeReceipt, error) {
				return inKernelDecodeV41LanesWithOps(ctx, lanes, operations)
			})
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			cancelA()
			cancelB()
			cancelC()
			select {
			case <-done:
			case <-time.After(2 * time.Second):
				t.Fatal("cohort failed to join after cancellation")
			}
			t.Fatal("cohort exceeded deadline")
		}
		for _, request := range []*inKernelCoalesceRequest{requestA, requestB, requestC} {
			select {
			case <-request.receiptReady:
			default:
				t.Fatalf("request lacks terminal receipt")
			}
			if request.closes.Load() != 1 {
				t.Fatalf("request closes=%d, want 1", request.closes.Load())
			}
		}
		return batches
	}

	t.Run("v41 seam default keeps arrival order", func(t *testing.T) {
		t.Setenv(inKernelDecodeShareEnv, "")
		t.Setenv(inKernelPromptRowBudgetEnv, "")
		got := v41ShareEpisode(t)
		// A runs six rounds; B and C are admitted together at the first
		// boundary (whole-prompt arrival order, nothing owed at default p=1)
		// and finish their two-token budgets on the round after admission.
		want := [][]uint64{{1}, {1, 2, 3}, {1}, {1}, {1}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("default batches=%v, want %v (whole-prompt arrival order: B and C admitted together)", got, want)
		}
	})

	t.Run("v41 seam row budget caps rows per visit", func(t *testing.T) {
		t.Setenv(inKernelDecodeShareEnv, "")
		t.Setenv(inKernelPromptRowBudgetEnv, "1")
		got := v41ShareEpisode(t)
		// One row per visit: B fills the first boundary, C the second, in
		// arrival order.
		want := [][]uint64{{1}, {1, 2}, {1, 3}, {1}, {1}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("budget-1 batches=%v, want %v (one row filled beside decoders per visit)", got, want)
		}
	})
}
