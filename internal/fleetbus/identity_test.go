package fleetbus

import (
	"testing"
	"time"
)

func TestFleetBusIdentitySurvivesRestart(t *testing.T) {
	t0 := time.Date(2026, 8, 11, 17, 0, 0, 0, time.UTC)
	root := t.TempDir()
	bus, err := OpenDir(root)
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}

	first := ResolveServeIdentity(ServeIdentityRequest{
		Machine: "box-a",
		Addr:    "127.0.0.1:8080",
		PID:     1000,
	})
	restarted := ResolveServeIdentity(ServeIdentityRequest{
		Machine: "box-a",
		Addr:    "127.0.0.1:8080",
		PID:     2000,
	})
	if first.ID != restarted.ID {
		t.Fatalf("same serve transport changed identity across restart: first=%q restarted=%q", first.ID, restarted.ID)
	}
	if !first.RestartStable || first.Source != IdentityConfiguredAddress {
		t.Fatalf("fixed-address identity = %+v, want configured-address restart stability", first)
	}

	inst1 := identityTestInstance(t, first, "box-a", 1000, t0)
	if err := bus.Announce(inst1); err != nil {
		t.Fatalf("announce boot 1: %v", err)
	}
	directive := identityTestDirective(t, t0)
	if err := bus.Publish(directive); err != nil {
		t.Fatalf("Publish: %v", err)
	}

	applied := 0
	applier := ApplierFunc(func(Directive) Outcome {
		applied++
		return OutcomeApplied("test apply", 1)
	})
	rep1, err := Drain(bus, inst1, applier, t0.Add(time.Second))
	if err != nil {
		t.Fatalf("drain boot 1: %v", err)
	}
	if rep1.Applied != 1 || rep1.AlreadyDone != 0 {
		t.Fatalf("boot 1 drain = %+v, want one apply", rep1)
	}

	// The old presence row is stale by the time the process comes back. Announce
	// must replace it under the SAME identity rather than growing the roster with a
	// PID-shaped sibling.
	restartAt := t0.Add(6 * time.Hour)
	inst2 := identityTestInstance(t, restarted, "box-a", 2000, restartAt)
	if err := bus.Announce(inst2); err != nil {
		t.Fatalf("announce boot 2: %v", err)
	}
	roster, err := bus.Instances(restartAt, DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances after restart: %v", err)
	}
	if len(roster) != 1 || roster[0].ID != first.ID || roster[0].PID != 2000 {
		t.Fatalf("restart roster = %+v, want one refreshed row for %q at pid 2000", roster, first.ID)
	}
	if roster[0].Addr != "127.0.0.1:8080" {
		t.Fatalf("restart roster addr = %q, want configured listen address", roster[0].Addr)
	}

	rep2, err := Drain(bus, inst2, applier, restartAt.Add(time.Second))
	if err != nil {
		t.Fatalf("drain boot 2: %v", err)
	}
	if rep2.AlreadyDone != 1 || rep2.Applied != 0 {
		t.Fatalf("boot 2 drain = %+v, want already_done=1 applied=0", rep2)
	}
	if applied != 1 {
		t.Fatalf("applier invoked %d times across one restart, want exactly once", applied)
	}
	acks, err := bus.Acks(directive.ID)
	if err != nil {
		t.Fatalf("Acks: %v", err)
	}
	if len(acks) != 1 {
		t.Fatalf("ack log has %d rows, want one applied ack across restart: %+v", len(acks), acks)
	}
}

func TestFleetBusIdentityIsUniqueForSimultaneousAddresses(t *testing.T) {
	t0 := time.Date(2026, 8, 11, 17, 0, 0, 0, time.UTC)
	bus, err := OpenDir(t.TempDir())
	if err != nil {
		t.Fatalf("OpenDir: %v", err)
	}
	left := ResolveServeIdentity(ServeIdentityRequest{
		Machine: "box-a",
		Addr:    "127.0.0.1:8080",
		PID:     1000,
	})
	right := ResolveServeIdentity(ServeIdentityRequest{
		Machine: "box-a",
		Addr:    "127.0.0.1:8081",
		PID:     2000,
	})
	if left.ID == right.ID {
		t.Fatalf("distinct simultaneous listen addresses collapsed onto %q", left.ID)
	}

	instLeft := identityTestInstance(t, left, "box-a", 1000, t0)
	instRight := identityTestInstance(t, right, "box-a", 2000, t0)
	for _, inst := range []Instance{instLeft, instRight} {
		if err := bus.Announce(inst); err != nil {
			t.Fatalf("Announce(%s): %v", inst.ID, err)
		}
	}
	roster, err := bus.Instances(t0, DefaultInstanceTTL)
	if err != nil {
		t.Fatalf("Instances: %v", err)
	}
	if len(roster) != 2 {
		t.Fatalf("simultaneous roster = %+v, want two independently addressable serves", roster)
	}

	directive := identityTestDirective(t, t0)
	directive = directive.WithTargets(PublishTargets(directive.Selector, roster))
	if err := bus.Publish(directive); err != nil {
		t.Fatalf("Publish: %v", err)
	}
	leftApplied, rightApplied := 0, 0
	repLeft, err := Drain(bus, instLeft, ApplierFunc(func(Directive) Outcome {
		leftApplied++
		return OutcomeApplied("left apply", 1)
	}), t0.Add(time.Second))
	if err != nil {
		t.Fatalf("left drain: %v", err)
	}
	repRight, err := Drain(bus, instRight, ApplierFunc(func(Directive) Outcome {
		rightApplied++
		return OutcomeApplied("right apply", 1)
	}), t0.Add(time.Second))
	if err != nil {
		t.Fatalf("right drain: %v", err)
	}
	if repLeft.Applied != 1 || repRight.Applied != 1 || leftApplied != 1 || rightApplied != 1 {
		t.Fatalf("simultaneous drains left=%+v right=%+v calls=%d/%d, want one apply each",
			repLeft, repRight, leftApplied, rightApplied)
	}
	acks, err := bus.Acks(directive.ID)
	if err != nil {
		t.Fatalf("Acks: %v", err)
	}
	if len(acks) != 2 {
		t.Fatalf("ack log has %d rows, want one per simultaneous serve: %+v", len(acks), acks)
	}
}

func TestFleetBusIdentityPreservesExplicitConfiguredName(t *testing.T) {
	const configured = "operator-chosen.serve_7"
	for _, req := range []ServeIdentityRequest{
		{ExplicitID: configured, Machine: "box-a", Addr: "127.0.0.1:8080", PID: 1000},
		{ExplicitID: configured, Machine: "box-a", Addr: "127.0.0.1:9090", PID: 2000},
	} {
		got := ResolveServeIdentity(req)
		if got.ID != configured {
			t.Fatalf("explicit id = %q, want byte-preserved %q", got.ID, configured)
		}
		if got.Source != IdentityExplicit || !got.RestartStable {
			t.Fatalf("explicit identity = %+v, want explicit restart-stable override", got)
		}
	}
}

func TestFleetBusIdentityNamesUnstableTransportFallback(t *testing.T) {
	cases := []struct {
		name string
		req  ServeIdentityRequest
		addr string
	}{
		{
			name: "stdio",
			req:  ServeIdentityRequest{Machine: "box-a", Addr: "127.0.0.1:8080", PID: 1000, Stdio: true},
			addr: "stdio",
		},
		{
			name: "dynamic port",
			req:  ServeIdentityRequest{Machine: "box-a", Addr: "127.0.0.1:0", PID: 1000},
			addr: "127.0.0.1:0",
		},
		{
			name: "missing address",
			req:  ServeIdentityRequest{Machine: "box-a", PID: 1000},
			addr: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			first := ResolveServeIdentity(tc.req)
			restartedReq := tc.req
			restartedReq.PID = 2000
			restarted := ResolveServeIdentity(restartedReq)
			if first.ID == restarted.ID {
				t.Fatalf("fallback transport unexpectedly reused %q across different pids", first.ID)
			}
			if first.RestartStable || first.Source != IdentityProcessFallback {
				t.Fatalf("fallback identity = %+v, want named process-local fallback", first)
			}
			if first.Addr != tc.addr {
				t.Fatalf("fallback addr = %q, want %q", first.Addr, tc.addr)
			}
			if !ValidToken(first.ID) {
				t.Fatalf("fallback id %q is not a valid bus token", first.ID)
			}
		})
	}
}

func identityTestInstance(t *testing.T, identity ServeIdentity, machine string, pid int, now time.Time) Instance {
	t.Helper()
	inst, refusal := NewInstance(identity.ID, machine, "serve", pid, identity.Addr, []Op{"terminate"}, now)
	if refusal != nil {
		t.Fatalf("NewInstance(%q): %v", identity.ID, refusal)
	}
	return inst
}

func identityTestDirective(t *testing.T, now time.Time) Directive {
	t.Helper()
	d, refusal := NewDirective("identity-test", "terminate", "", Selector{All: true}, 0, "restart witness", now)
	if refusal != nil {
		t.Fatalf("NewDirective: %v", refusal)
	}
	return d
}
