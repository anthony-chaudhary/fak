package ratelimit

import (
	"context"
	"math"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Three bounded in-memory counter sequences; runtime estimate is unmeasured.
// fak-test:runtime fast est=1ms lane=default
func TestCostOverflowCannotReopenBudget(t *testing.T) {
	t.Parallel()
	charge := func(r *Limiter, cost int64) abi.Verdict {
		return r.Adjudicate(context.Background(), &abi.ToolCall{
			TraceID: "overflow",
			Tool:    "tool",
			Meta:    map[string]string{"fak.ratelimit.cost": strconv.FormatInt(cost, 10)},
		})
	}
	t.Run("configured cost cap", func(t *testing.T) {
		r := New()
		r.SetLimit(Limit{MaxCost: 10}, KeyPerTrace)
		mustDefer(t, charge(r, 1), "initial charge")
		for i := 0; i < 2; i++ {
			v := charge(r, math.MaxInt64)
			mustRateLimited(t, v, "overflowing cost")
			if v.Meta["cap"] != "max_cost" || v.Meta["limit"] != "10" {
				t.Fatalf("cost refusal evidence changed: %+v", v)
			}
		}
		mustDefer(t, charge(r, 9), "denied probes must leave remaining budget")
		mustDefer(t, charge(r, 0), "zero cost fits exactly at cap")
		mustRateLimited(t, charge(r, 1), "exhausted cap")
		if admits, denies, dropped := r.Stats(); admits != 3 || denies != 3 || dropped != 0 {
			t.Fatalf("admission accounting changed: admits=%d denies=%d dropped=%d", admits, denies, dropped)
		}
	})
	for _, tc := range []struct {
		name     string
		extra    int64
		exceeded bool
	}{
		{"exact range before tightening", 0, false},
		{"above range before tightening", 1, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := New()
			r.SetLimit(Limit{MaxCalls: 4}, KeyPerTrace)
			mustDefer(t, charge(r, math.MaxInt64), "unlimited-cost first charge")
			mustDefer(t, charge(r, tc.extra), "unlimited cost remains unlimited")
			mustDefer(t, charge(r, 0), "zero-cost admission must preserve accrued usage")
			r.SetLimit(Limit{MaxCost: math.MaxInt64}, KeyPerTrace)
			v := charge(r, 0)
			if tc.exceeded {
				mustRateLimited(t, v, "tightening must retain above-range cost")
				if v.Meta["cap"] != "max_cost" || v.Meta["limit"] != strconv.FormatInt(math.MaxInt64, 10) {
					t.Fatalf("tightened cost refusal evidence changed: %+v", v)
				}
			} else {
				mustDefer(t, v, "exact range is not overflow")
			}
			r.Reset("trace:overflow")
			r.SetLimit(Limit{MaxCost: 1}, KeyPerTrace)
			mustDefer(t, charge(r, 1), "reset must clear prior cost and overflow")
		})
	}
}
