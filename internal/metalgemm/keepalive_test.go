package metalgemm

import (
	"sync"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestKeepAlivePowerPolicy(t *testing.T) {
	for _, tc := range []struct {
		name, policy string
		power        KeepAlivePowerState
		want         bool
	}{
		{name: "default off"},
		{name: "explicit off", policy: "off"},
		{name: "unknown policy fails closed", policy: "maybe"},
		{name: "explicit on", policy: "on", want: true},
		{name: "auto unknown power", policy: "auto"},
		{name: "auto AC", policy: "auto", power: KeepAlivePowerState{Known: true}, want: true},
		{name: "auto battery", policy: "auto", power: KeepAlivePowerState{Known: true, OnBattery: true}},
		{name: "auto low power AC", policy: "auto", power: KeepAlivePowerState{Known: true, LowPowerMode: true}},
		{name: "auto low power battery", policy: "auto", power: KeepAlivePowerState{Known: true, OnBattery: true, LowPowerMode: true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := KeepAliveEnabled(tc.policy, tc.power); got != tc.want {
				t.Fatalf("policy %q with power %+v enabled=%v, want %v", tc.policy, tc.power, got, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestKeepAliveLeaseLifecycle(t *testing.T) {
	t.Run("concurrent duplicate releases remain balanced", func(t *testing.T) {
		starts, stops := 0, 0
		c := newKeepAliveController(func() bool { starts++; return true }, func() { stops++ })
		const holders = 16
		leases := make([]func(), holders)
		for i := range leases {
			leases[i] = c.begin()
		}
		var done sync.WaitGroup
		for _, release := range leases {
			done.Add(1)
			go func() {
				defer done.Done()
				release()
				release()
			}()
		}
		done.Wait()
		if starts != 1 || stops != 1 {
			t.Fatalf("concurrent releases: starts=%d stops=%d, want 1/1", starts, stops)
		}
	})
	t.Run("nested leases stop only after final release", func(t *testing.T) {
		starts, stops := 0, 0
		c := newKeepAliveController(func() bool { starts++; return true }, func() { stops++ })
		outer := c.begin()
		inner := c.begin()
		if starts != 1 || stops != 0 {
			t.Fatalf("nested acquisition: starts=%d stops=%d, want 1/0", starts, stops)
		}
		outer()
		outer()
		if stops != 0 {
			t.Fatal("duplicate outer release stopped a live inner lease")
		}
		inner()
		inner()
		if starts != 1 || stops != 1 {
			t.Fatalf("final release: starts=%d stops=%d, want 1/1", starts, stops)
		}
		next := c.begin()
		next()
		if starts != 2 || stops != 2 {
			t.Fatalf("reuse: starts=%d stops=%d, want 2/2", starts, stops)
		}
	})
	t.Run("failed start leaves no holder or stop obligation", func(t *testing.T) {
		starts, stops := 0, 0
		c := newKeepAliveController(func() bool { starts++; return starts > 1 }, func() { stops++ })
		failed := c.begin()
		retry := c.begin()
		failed()
		failed()
		if starts != 2 || stops != 0 {
			t.Fatalf("retry after failed start: starts=%d stops=%d, want 2/0", starts, stops)
		}
		retry()
		if stops != 1 {
			t.Fatalf("successful retry released: stops=%d, want 1", stops)
		}
	})
	t.Run("deferred cleanup survives panic", func(t *testing.T) {
		stops := 0
		c := newKeepAliveController(func() bool { return true }, func() { stops++ })
		func() {
			defer func() { _ = recover() }()
			defer c.begin()()
			panic("forward failed")
		}()
		if stops != 1 {
			t.Fatalf("panic cleanup: stops=%d, want 1", stops)
		}
	})
}
