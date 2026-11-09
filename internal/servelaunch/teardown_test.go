package servelaunch

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"
)

// stopHarness wires a StopConfig to hand-driven time (the fakeClock /
// fakeTicker seams from readiness_test.go) with recording signaler, exit, and
// port-kill seams so no test touches a real process.
type stopHarness struct {
	clock   *fakeClock
	ticker  *fakeTicker
	signals []Signal
	killed  []int // ports passed to PortKill, in order
	cfg     StopConfig
}

func newStopHarness(t *testing.T, steps []Step, exited func() bool) *stopHarness {
	t.Helper()
	clock := &fakeClock{now: time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)}
	ticker := newFakeTicker(256)
	h := &stopHarness{clock: clock, ticker: ticker}
	h.cfg = StopConfig{
		Steps:        steps,
		Port:         30000,
		PollInterval: time.Second,
		Signal:       func(sig Signal) error { h.signals = append(h.signals, sig); return nil },
		Exited:       exited,
		PortKill: func(port int, _ Signal) ([]int, error) {
			h.killed = append(h.killed, port)
			return []int{4242}, nil
		},
		Clock:     clock,
		NewTicker: func(time.Duration) Ticker { return ticker },
	}
	return h
}

// TestStopLadderEscalatesInOrder proves the core contract: a target that
// survives SIGTERM gets escalated to SIGKILL, the confirmed exit stops the
// ladder before the safety net, and delivery order matches rung order.
// fak-test:runtime fast est=10ms lane=default
func TestStopLadderEscalatesInOrder(t *testing.T) {
	state := 0 // 0 alive, 1 term delivered, 2 kill delivered
	h := newStopHarness(t, DefaultSteps(), func() bool { return state >= 2 })
	h.cfg.Signal = func(sig Signal) error {
		h.signals = append(h.signals, sig)
		if sig == SignalTerm {
			state = 1
		} else {
			state = 2
		}
		return nil
	}
	// DefaultSteps gives SIGTERM the full 30s default grace: prime one tick
	// per second of fake time so the poll rides the whole window.
	h.ticker.prime(h.clock, 31, time.Second)

	res, err := StopLadder(context.Background(), h.cfg)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if res.StoppedBy != "sigkill" {
		t.Errorf("StoppedBy = %q, want sigkill", res.StoppedBy)
	}
	if want := []string{"sigterm", "sigkill"}; !reflect.DeepEqual(res.RungsFired, want) {
		t.Errorf("RungsFired = %v, want %v", res.RungsFired, want)
	}
	if want := []Signal{SignalTerm, SignalKill}; !reflect.DeepEqual(h.signals, want) {
		t.Errorf("signals delivered = %v, want %v (rung order)", h.signals, want)
	}
	if len(h.killed) != 0 {
		t.Errorf("PortKill called with ports %v, want it never reached", h.killed)
	}
	if res.Elapsed != DefaultTermGrace {
		t.Errorf("Elapsed = %s, want %s (full SIGTERM grace, tick granularity)", res.Elapsed, DefaultTermGrace)
	}
}

// TestStopLadderStopsWithoutEscalatingWhenExitConfirmed proves escalation
// stops on a confirmed exit: the target dies inside the SIGTERM window, so
// SIGKILL and the safety net must never fire, and the credit goes to the
// rung whose effect was observed.
// fak-test:runtime fast est=10ms lane=default
func TestStopLadderStopsWithoutEscalatingWhenExitConfirmed(t *testing.T) {
	var h *stopHarness
	h = newStopHarness(t, DefaultSteps(), func() bool { return len(h.signals) >= 1 })
	h.ticker.prime(h.clock, 4, time.Second)

	res, err := StopLadder(context.Background(), h.cfg)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if res.StoppedBy != "sigterm" {
		t.Errorf("StoppedBy = %q, want sigterm", res.StoppedBy)
	}
	if want := []string{"sigterm"}; !reflect.DeepEqual(res.RungsFired, want) {
		t.Errorf("RungsFired = %v, want %v (no escalation after confirmed exit)", res.RungsFired, want)
	}
	if want := []Signal{SignalTerm}; !reflect.DeepEqual(h.signals, want) {
		t.Errorf("signals delivered = %v, want only %v", h.signals, want)
	}
	if len(h.killed) != 0 {
		t.Errorf("PortKill called with ports %v, want it never reached", h.killed)
	}
}

// TestStopLadderIdempotentWhenAlreadyStopped proves the idempotent no-op: an
// already-gone target returns the typed ErrAlreadyStopped sentinel with no
// rung fired, branchable via errors.Is.
// fak-test:runtime fast est=10ms lane=default
func TestStopLadderIdempotentWhenAlreadyStopped(t *testing.T) {
	h := newStopHarness(t, DefaultSteps(), func() bool { return true })

	res, err := StopLadder(context.Background(), h.cfg)
	if !errors.Is(err, ErrAlreadyStopped) {
		t.Fatalf("err = %v, want errors.Is(err, ErrAlreadyStopped)", err)
	}
	if res == nil || res.StoppedBy != "" {
		t.Errorf("res = %+v, want empty result on the idempotent no-op", res)
	}
	if len(res.RungsFired) != 0 || len(h.signals) != 0 || len(h.killed) != 0 {
		t.Errorf("rungs=%v signals=%v killed=%v, want nothing delivered", res.RungsFired, h.signals, h.killed)
	}
}

// TestStopLadderPortKillSafetyNetLast proves the safety-net rung: a target
// that ignores both signal rungs gets discovered by port and signalled, with
// the discovered pids recorded and the credit going to the port-kill rung.
// fak-test:runtime fast est=10ms lane=default
func TestStopLadderPortKillSafetyNetLast(t *testing.T) {
	steps := []Step{
		{Name: "sigterm", Kind: RungSignal, Signal: SignalTerm, WaitForExit: time.Second},
		{Name: "sigkill", Kind: RungSignal, Signal: SignalKill, WaitForExit: time.Second},
		{Name: "port-kill", Kind: RungPortKill, Signal: SignalKill, WaitForExit: time.Second},
	}
	var h *stopHarness
	h = newStopHarness(t, steps, func() bool { return len(h.killed) > 0 })
	h.ticker.prime(h.clock, 6, time.Second)

	res, err := StopLadder(context.Background(), h.cfg)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if res.StoppedBy != "port-kill" {
		t.Errorf("StoppedBy = %q, want port-kill", res.StoppedBy)
	}
	if want := []string{"sigterm", "sigkill", "port-kill"}; !reflect.DeepEqual(res.RungsFired, want) {
		t.Errorf("RungsFired = %v, want %v", res.RungsFired, want)
	}
	if want := []int{30000}; !reflect.DeepEqual(h.killed, want) {
		t.Errorf("PortKill called for ports %v, want %v", h.killed, want)
	}
	if want := []int{4242}; !reflect.DeepEqual(res.PortKilledPIDs, want) {
		t.Errorf("PortKilledPIDs = %v, want %v", res.PortKilledPIDs, want)
	}
}

// TestStopLadderNotStoppedWhenNothingConfirms proves the fail-closed end of
// the ladder: every rung fires, no exit is ever confirmed, and the result is
// the typed ErrNotStopped sentinel, not a silent success.
// fak-test:runtime fast est=10ms lane=default
func TestStopLadderNotStoppedWhenNothingConfirms(t *testing.T) {
	steps := []Step{
		{Name: "sigterm", Kind: RungSignal, Signal: SignalTerm, WaitForExit: time.Second},
		{Name: "sigkill", Kind: RungSignal, Signal: SignalKill, WaitForExit: time.Second},
		{Name: "port-kill", Kind: RungPortKill, Signal: SignalKill, WaitForExit: time.Second},
	}
	h := newStopHarness(t, steps, func() bool { return false })
	h.cfg.PortKill = func(port int, _ Signal) ([]int, error) {
		h.killed = append(h.killed, port)
		return nil, nil
	}
	h.ticker.prime(h.clock, 6, time.Second)

	res, err := StopLadder(context.Background(), h.cfg)
	if !errors.Is(err, ErrNotStopped) {
		t.Fatalf("err = %v, want errors.Is(err, ErrNotStopped)", err)
	}
	if res.StoppedBy != "" {
		t.Errorf("StoppedBy = %q, want empty (never confirmed)", res.StoppedBy)
	}
	if want := []string{"sigterm", "sigkill", "port-kill"}; !reflect.DeepEqual(res.RungsFired, want) {
		t.Errorf("RungsFired = %v, want %v (full ladder ran)", res.RungsFired, want)
	}
}

// TestStopLadderFollowsDataNotLiterals proves the ladder order and windows
// come from the Step data: a custom plan that fires SIGKILL first is honored
// verbatim, so the runner cannot be accused of hardcoding term-before-kill.
// fak-test:runtime fast est=10ms lane=default
func TestStopLadderFollowsDataNotLiterals(t *testing.T) {
	state := 0
	steps := []Step{
		{Name: "force-first", Kind: RungSignal, Signal: SignalKill, WaitForExit: time.Second},
		{Name: "gentle-second", Kind: RungSignal, Signal: SignalTerm, WaitForExit: time.Second},
	}
	h := newStopHarness(t, steps, func() bool { return state >= 2 })
	h.cfg.Signal = func(sig Signal) error {
		h.signals = append(h.signals, sig)
		state++
		return nil
	}
	h.ticker.prime(h.clock, 3, time.Second)

	res, err := StopLadder(context.Background(), h.cfg)
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if want := []Signal{SignalKill, SignalTerm}; !reflect.DeepEqual(h.signals, want) {
		t.Errorf("signals delivered = %v, want %v (data order, not literals)", h.signals, want)
	}
	if res.StoppedBy != "gentle-second" {
		t.Errorf("StoppedBy = %q, want gentle-second", res.StoppedBy)
	}
}

// TestStopLadderRefusesMissingSeams proves fail-closed config validation:
// each missing seam is refused up front with an error that matches none of
// the ladder sentinels.
// fak-test:runtime fast est=1ms lane=default
func TestStopLadderRefusesMissingSeams(t *testing.T) {
	cases := []struct {
		name string
		mut  func(*StopConfig)
	}{
		{"empty ladder", func(c *StopConfig) { c.Steps = nil }},
		{"missing Exited probe", func(c *StopConfig) { c.Exited = nil }},
		{"signal rung without Signal func", func(c *StopConfig) { c.Signal = nil }},
		{"port-kill rung without PortKill func", func(c *StopConfig) { c.PortKill = nil }},
		{"port-kill rung without positive Port", func(c *StopConfig) { c.Port = 0 }},
		{"unknown rung kind", func(c *StopConfig) { c.Steps[0].Kind = RungKind(99) }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := newStopHarness(t, DefaultSteps(), func() bool { return false })
			tc.mut(&h.cfg)
			_, err := StopLadder(context.Background(), h.cfg)
			if err == nil {
				t.Fatal("expected a config refusal")
			}
			for _, s := range []error{ErrAlreadyStopped, ErrNotStopped} {
				if errors.Is(err, s) {
					t.Fatalf("config refusal matched ladder sentinel %v", s)
				}
			}
		})
	}
}

// TestRollbackStopsInReverseStartOrder asserts the rollback contract: units
// are stopped last-started-first.
// fak-test:runtime fast est=1ms lane=default
func TestRollbackStopsInReverseStartOrder(t *testing.T) {
	var order []string
	units := make([]Unit, 0, 3)
	for _, name := range []string{"a", "b", "c"} {
		name := name
		units = append(units, Unit{Name: name, Stop: func(context.Context) error {
			order = append(order, name)
			return nil
		}})
	}

	if err := Rollback(context.Background(), units); err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if want := []string{"c", "b", "a"}; !reflect.DeepEqual(order, want) {
		t.Errorf("stop order = %v, want %v (reverse start order)", order, want)
	}
}

// TestRollbackContinuesAfterFailure proves one wedged unit cannot strand the
// earlier ones: the failing unit's sentinel stays reachable via errors.Is
// while the remaining units still get stopped in reverse order.
// fak-test:runtime fast est=1ms lane=default
func TestRollbackContinuesAfterFailure(t *testing.T) {
	wedged := errors.New("unit b wedged")
	var order []string
	stops := map[string]func(context.Context) error{
		"a": func(context.Context) error { order = append(order, "a"); return nil },
		"b": func(context.Context) error { order = append(order, "b"); return fmt.Errorf("wrap: %w", wedged) },
		"c": func(context.Context) error { order = append(order, "c"); return nil },
	}
	var units []Unit
	for _, name := range []string{"a", "b", "c"} {
		units = append(units, Unit{Name: name, Stop: stops[name]})
	}

	err := Rollback(context.Background(), units)
	if !errors.Is(err, wedged) {
		t.Fatalf("err = %v, want errors.Is(err, %v)", err, wedged)
	}
	if want := []string{"c", "b", "a"}; !reflect.DeepEqual(order, want) {
		t.Errorf("stop order = %v, want %v (failure did not stop the rest)", order, want)
	}
}

// TestRollbackIgnoresAlreadyStopped proves idempotence composes through
// rollback: a unit that reports ErrAlreadyStopped is not a rollback failure.
// fak-test:runtime fast est=1ms lane=default
func TestRollbackIgnoresAlreadyStopped(t *testing.T) {
	var order []string
	units := []Unit{
		{Name: "a", Stop: func(context.Context) error {
			order = append(order, "a")
			return nil
		}},
		{Name: "b", Stop: func(context.Context) error {
			order = append(order, "b")
			return ErrAlreadyStopped
		}},
		{Name: "c", Stop: func(context.Context) error {
			order = append(order, "c")
			return nil
		}},
	}

	if err := Rollback(context.Background(), units); err != nil {
		t.Fatalf("err = %v, want nil (already-stopped is not a failure)", err)
	}
	if want := []string{"c", "b", "a"}; !reflect.DeepEqual(order, want) {
		t.Errorf("stop order = %v, want %v", order, want)
	}
}
