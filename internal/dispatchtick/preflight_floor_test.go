package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetcap"
)

// TestEvaluatePreflightForecastFloorRaisesReactiveTick is the #3368 two-timescale
// witness. A SLOW predictive loop (the Little's-law forecast in fleetcap) emits a
// workerFloor that leads the FAST reactive demand (the kernel/lease target) by one
// tick. It proves the two properties the issue's first checkable step names: the
// rising forecast raises the floor one tick BEFORE reactive demand catches up, and
// the reactive tick never dips below that floor -- capacity is pre-warmed for the
// ramp instead of tracking the lagging reactive signal down.
func TestEvaluatePreflightForecastFloorRaisesReactiveTick(t *testing.T) {
	const sessionMin = 10.0 // W in Little's law: a 10-minute median session.

	ticks := []struct {
		name          string
		forecastRate  float64 // the SLOW loop's target issue-rate at this tick
		reactiveLease int     // the FAST kernel/lease target -- lags the forecast by one tick
		wantFloor     int     // fleetcap.RequiredWorkers(forecastRate, sessionMin)
		wantCap       int     // effective cap WITH the forecast floor
		wantReactive  int     // effective cap WITHOUT the floor (the dip the floor prevents)
		wantLimiting  string
	}{
		// t0: the forecast already sees the ramp; reactive demand is still flat. The
		// floor pre-warms capacity to 5 a full tick before the reactive tick moves.
		{"pre-warm: forecast leads a flat reactive tick", 30, 1, 5, 5, 1, "floor"},
		// t1..t2: reactive demand catches each PRIOR tick's forecast, but the forecast
		// keeps leading, so the floor stays one step ahead and keeps lifting the cap.
		{"reactive catches t0 forecast; floor still leads", 60, 5, 10, 10, 5, "floor"},
		{"reactive catches t1 forecast; floor still leads", 90, 10, 15, 15, 10, "floor"},
		// t3: demand has fully caught the (now steady) forecast -- floor and reactive
		// agree, so the reactive lease is the honest limiter, not the floor.
		{"steady: reactive equals the forecast", 90, 15, 15, 15, 15, "lease"},
	}

	for _, tc := range ticks {
		t.Run(tc.name, func(t *testing.T) {
			// The slow predictive loop's forecast -> the worker floor it emits.
			floor := fleetcap.RequiredWorkers(tc.forecastRate, sessionMin)
			if floor != tc.wantFloor {
				t.Fatalf("forecast floor = %d, want %d (rate=%g, session=%g)", floor, tc.wantFloor, tc.forecastRate, sessionMin)
			}

			base := preflightInput()
			base.MaxWorkers = FallbackMaxWorkers
			base.Resources = roomyResources() // host cap 32 -- well above every floor here
			base.Kernel = KernelCheck{Alive: IntPtr(0), Target: IntPtr(tc.reactiveLease)}

			// The reactive tick clamped UP to the forecast floor.
			withFloor := base
			withFloor.WorkerFloor = floor
			got := EvaluatePreflight(withFloor)

			// The same reactive tick WITHOUT the forecast floor -- the value that would
			// track the lagging reactive demand down.
			reactiveOnly := EvaluatePreflight(base)

			if got.Cap != tc.wantCap {
				t.Fatalf("cap with floor = %d, want %d", got.Cap, tc.wantCap)
			}
			if reactiveOnly.Cap != tc.wantReactive {
				t.Fatalf("cap without floor = %d, want %d", reactiveOnly.Cap, tc.wantReactive)
			}
			// The load-bearing property: the reactive tick never dips below the floor.
			if got.Cap < floor {
				t.Fatalf("reactive cap %d dipped below forecast floor %d", got.Cap, floor)
			}
			// Where the forecast leads, the floor genuinely pre-warmed capacity above
			// what the lagging reactive signal alone would have allowed.
			if tc.wantCap > tc.wantReactive && !(got.Cap > reactiveOnly.Cap) {
				t.Fatalf("floor did not pre-warm: with=%d without=%d", got.Cap, reactiveOnly.Cap)
			}
			if got.CapTerms.WorkerFloor != floor {
				t.Fatalf("cap_terms worker_floor = %d, want %d", got.CapTerms.WorkerFloor, floor)
			}
			if got.CapTerms.Limiting != tc.wantLimiting {
				t.Fatalf("limiting = %q, want %q; terms=%#v", got.CapTerms.Limiting, tc.wantLimiting, got.CapTerms)
			}
			if fl := got.Map()["cap_terms"].(map[string]any)["worker_floor"]; fl != floor {
				t.Fatalf("cap_terms map worker_floor = %v, want %d", fl, floor)
			}
		})
	}
}

// TestEvaluatePreflightForecastFloorBoundedByHardCeiling proves the forecast floor
// can override a soft reactive dip but NEVER overbooks the box or the seat pool: it
// is clamped to the hard physical/config ceiling (host cap, seat inventory, config
// max), so a forecast asking for more workers than the host or seats can hold is
// capped at what is physically runnable -- pre-warming is not overbooking (#3368).
func TestEvaluatePreflightForecastFloorBoundedByHardCeiling(t *testing.T) {
	t.Run("host cap bounds the floor", func(t *testing.T) {
		in := preflightInput()
		in.MaxWorkers = 5
		in.Resources = HostResources{Cores: IntPtr(4), FreeRAMMB: IntPtr(3000), TotalThreads: IntPtr(1000)} // host cap 2
		in.Kernel = KernelCheck{Alive: IntPtr(0), Target: IntPtr(1)}
		in.WorkerFloor = 15 // a forecast far above what this box can run
		got := EvaluatePreflight(in)
		if got.HostCap == nil || *got.HostCap != 2 {
			t.Fatalf("host cap = %v, want 2", got.HostCap)
		}
		if got.Cap != 2 {
			t.Fatalf("cap = %d, want 2 (floor clamped to host cap, never overbooking)", got.Cap)
		}
		if got.CapTerms.WorkerFloor != 2 {
			t.Fatalf("cap_terms worker_floor = %d, want 2 (the ceiling-bounded floor)", got.CapTerms.WorkerFloor)
		}
	})

	t.Run("seat inventory bounds the floor", func(t *testing.T) {
		in := preflightInput()
		in.MaxWorkers = 20
		in.Resources = roomyResources() // host cap 32
		in.Kernel = KernelCheck{Alive: IntPtr(0), Target: IntPtr(1)}
		in.Seat = SeatCheck{Total: IntPtr(3), Free: IntPtr(3), Leased: IntPtr(0)}
		in.WorkerFloor = 15 // more than the seat pool holds
		got := EvaluatePreflight(in)
		if got.Cap != 3 {
			t.Fatalf("cap = %d, want 3 (floor clamped to seat inventory, never overbooking)", got.Cap)
		}
		if got.CapTerms.WorkerFloor != 3 {
			t.Fatalf("cap_terms worker_floor = %d, want 3 (bounded by the seat pool)", got.CapTerms.WorkerFloor)
		}
	})

	t.Run("zero floor is byte-identical to before the term existed", func(t *testing.T) {
		in := preflightInput()
		in.MaxWorkers = FallbackMaxWorkers
		in.Resources = roomyResources()
		in.Kernel = KernelCheck{Alive: IntPtr(0), Target: IntPtr(3)}
		got := EvaluatePreflight(in) // WorkerFloor unset (0)
		if got.CapTerms.WorkerFloor != 0 || got.CapTerms.Limiting != "lease" || got.Cap != 3 {
			t.Fatalf("zero-floor preflight = %#v, want floor 0 / lease / cap 3", got.CapTerms)
		}
	})
}
