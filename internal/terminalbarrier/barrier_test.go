package terminalbarrier

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetbus"
	"github.com/anthony-chaudhary/fak/internal/lifecycleadapter"
)

type fakeActuator struct {
	stops, restores     int
	stopErr, restoreErr error
}

func (a *fakeActuator) StopHost(context.Context) error    { a.stops++; return a.stopErr }
func (a *fakeActuator) RestoreHost(context.Context) error { a.restores++; return a.restoreErr }
func mixedForest() Forest {
	return Forest{ID: "forest-terminal", Authority: "guard:root", Generation: 1, Members: []Member{{ID: "root", Generation: 1, Active: true, Adapter: lifecycleadapter.NativeFAK()}, {ID: "codex", Generation: 1, Active: true, Adapter: lifecycleadapter.Codex()}, {ID: "claude", Generation: 1, Active: true, Adapter: lifecycleadapter.Claude()}}}
}
func TestCapturedMixedForestPauseReplaceRestoreReady(t *testing.T) {
	now := time.Date(2026, 8, 11, 23, 30, 0, 0, time.UTC)
	root := t.TempDir()
	act := &fakeActuator{}
	bus := fleetbus.LifecycleDirBus{Root: root}
	c := Coordinator{Bus: bus, Actuator: act, Now: func() time.Time { return now }}
	r := c.Replace(context.Background(), true, mixedForest(), "tx-pressure-1", time.Now().UTC().Add(time.Minute))
	if r.Verdict != "READY" || act.stops != 1 || act.restores != 1 || len(r.Acks) != 3 {
		t.Fatalf("report=%+v actuator=%+v", r, act)
	}
	// Every active member received a durable prepare AND pause request before any
	// host action: the requests are on the bus, not merely in the coordinator.
	reqs, err := os.ReadDir(filepath.Join(root, "lifecycle", "requests", "tx-pressure-1"))
	if err != nil || len(reqs) != 3 {
		t.Fatalf("prepare/pause requests=%v err=%v", reqs, err)
	}
	var published fleetbus.LifecycleRequest
	raw, err := os.ReadFile(filepath.Join(root, "lifecycle", "requests", "tx-pressure-1", "codex.json"))
	if err != nil || json.Unmarshal(raw, &published) != nil || published.Action != fleetbus.LifecyclePause {
		t.Fatalf("published=%+v raw=%s err=%v", published, raw, err)
	}
	// Quiescence is proven by the durable acknowledgements read back off the bus.
	acks, err := bus.ReadAcks("tx-pressure-1")
	if err != nil || len(acks) != 3 {
		t.Fatalf("read-back acks=%+v err=%v", acks, err)
	}
	for _, a := range acks {
		if a.State != fleetbus.AckCompleted || a.ReadbackRef == "" {
			t.Fatalf("ack=%+v", a)
		}
	}
	if acks[1].MemberID != "codex" || acks[2].MemberID != "root" || acks[2].CheckpointRef == "" {
		t.Fatalf("checkpoint evidence missing: %+v", acks)
	}
	captured, _ := json.MarshalIndent(r, "", "  ")
	if !strings.Contains(string(captured), "tx-pressure-1") || !strings.Contains(string(captured), "readback") {
		t.Fatalf("captured report=%s", captured)
	}
}

func TestMonitorLineNamesTransactionAndReadback(t *testing.T) {
	now := time.Now().UTC()
	deadline := now.Add(time.Minute)
	c := Coordinator{Bus: fleetbus.LifecycleDirBus{Root: t.TempDir()}, Actuator: &fakeActuator{}, Now: func() time.Time { return now }}
	line := c.Replace(context.Background(), true, mixedForest(), "tx-pressure-2", deadline).MonitorLine()
	for _, want := range []string{"READY", "transaction=tx-pressure-2", "forest=forest-terminal", "stops=1", "readback=", "claude:readiness"} {
		if !strings.Contains(line, want) {
			t.Fatalf("monitor line %q missing %q", line, want)
		}
	}
	abstain := Coordinator{Bus: fleetbus.LifecycleDirBus{Root: t.TempDir()}, Actuator: &fakeActuator{}, Now: func() time.Time { return now }}
	f := mixedForest()
	f.Members[1].Adapter = lifecycleadapter.Unknown("missing")
	line = abstain.Replace(context.Background(), true, f, "tx-pressure-3", deadline).MonitorLine()
	if !strings.Contains(line, "ABSTAIN") || !strings.Contains(line, "transaction=tx-pressure-3") || !strings.Contains(line, "stops=0") || !strings.Contains(line, "readback=none") {
		t.Fatalf("abstain monitor line %q", line)
	}
}
func TestBarrierFailuresMakeZeroStopCalls(t *testing.T) {
	now := time.Now().UTC()
	cases := map[string]func(*Coordinator, *Forest, time.Time){
		"missing_ack": func(_ *Coordinator, f *Forest, _ time.Time) {
			f.Members[1].Adapter = lifecycleadapter.Unknown("missing")
		},
		"refused": func(_ *Coordinator, f *Forest, _ time.Time) {
			f.Members[1].Adapter = lifecycleadapter.Custom(lifecycleadapter.CapabilityDocument{Protocol: lifecycleadapter.ProtocolVersion, AdapterKind: "refuse", Operations: []lifecycleadapter.Operation{lifecycleadapter.Prepare, lifecycleadapter.Pause}}, func(context.Context, lifecycleadapter.Request) lifecycleadapter.Result {
				return lifecycleadapter.Result{State: lifecycleadapter.ResultRefused, Reason: "operator refused"}
			})
		},
		"late": func(_ *Coordinator, _ *Forest, deadline time.Time) { _ = deadline },
		"dynamic_child": func(c *Coordinator, f *Forest, _ time.Time) {
			next := *f
			next.Members = append(append([]Member(nil), f.Members...), Member{ID: "new-child", Generation: 1, Active: true, Adapter: lifecycleadapter.Claude()})
			c.Discover = func() Forest { return next }
		},
		"unknown": func(_ *Coordinator, f *Forest, _ time.Time) {
			f.Members[2].Adapter = lifecycleadapter.Unknown("custom")
		},
		"checkpoint_failure": func(_ *Coordinator, f *Forest, _ time.Time) {
			f.Members[0].Adapter = lifecycleadapter.Custom(lifecycleadapter.CapabilityDocument{Protocol: lifecycleadapter.ProtocolVersion, AdapterKind: "fak", Operations: []lifecycleadapter.Operation{lifecycleadapter.Prepare, lifecycleadapter.Pause, lifecycleadapter.Checkpoint}, ApplicationCheckpoint: true}, func(_ context.Context, r lifecycleadapter.Request) lifecycleadapter.Result {
				if r.Operation == lifecycleadapter.Checkpoint {
					return lifecycleadapter.Result{State: lifecycleadapter.ResultFailed, Reason: "disk full"}
				}
				return lifecycleadapter.Result{State: lifecycleadapter.ResultCompleted, ReadbackRef: "ok"}
			})
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			f := mixedForest()
			act := &fakeActuator{}
			c := Coordinator{Bus: fleetbus.LifecycleDirBus{Root: t.TempDir()}, Actuator: act, Now: func() time.Time { return now }}
			deadline := now.Add(time.Minute)
			if name == "late" {
				deadline = now.Add(-time.Second)
			}
			mutate(&c, &f, deadline)
			r := c.Replace(context.Background(), true, f, "tx-"+name, deadline)
			if r.Verdict != "ABSTAIN" || act.stops != 0 {
				t.Fatalf("report=%+v stop=%d", r, act.stops)
			}
		})
	}
}
func TestForestUnderHostAbstainsOnUnmanagedDescendant(t *testing.T) {
	now := time.Now().UTC()
	managed := ForestUnderHost(7, 1, []ManagedProcess{{MemberID: "fak-info-9", Image: "fak.exe"}, {MemberID: "codex-14", Image: "Codex.exe"}})
	if managed.ID != "terminal-host-7" || managed.Authority != "terminal-relief:7" || len(managed.Members) != 2 {
		t.Fatalf("forest=%+v", managed)
	}
	act := &fakeActuator{}
	c := Coordinator{Bus: fleetbus.LifecycleDirBus{Root: t.TempDir()}, Actuator: act, Now: func() time.Time { return now }}
	if r := c.Replace(context.Background(), true, managed, "tx-managed", now.Add(time.Minute)); r.Verdict != "READY" || act.stops != 1 {
		t.Fatalf("managed report=%+v stops=%d", r, act.stops)
	}
	unmanaged := ForestUnderHost(7, 1, []ManagedProcess{{MemberID: "vim-8", Image: "vim.exe"}})
	blocked := &fakeActuator{}
	r := Coordinator{Bus: fleetbus.LifecycleDirBus{Root: t.TempDir()}, Actuator: blocked, Now: func() time.Time { return now }}.Replace(context.Background(), true, unmanaged, "tx-unmanaged", now.Add(time.Minute))
	if r.Verdict != "ABSTAIN" || blocked.stops != 0 {
		t.Fatalf("unmanaged report=%+v stops=%d", r, blocked.stops)
	}
}

func TestBelowThresholdEmitsNoLifecycleTraffic(t *testing.T) {
	root := t.TempDir()
	act := &fakeActuator{}
	r := Coordinator{Bus: fleetbus.LifecycleDirBus{Root: root}, Actuator: act}.Replace(context.Background(), false, mixedForest(), "tx-none", time.Now().Add(time.Minute))
	if r.Verdict != "BELOW_THRESHOLD" || act.stops != 0 {
		t.Fatalf("report=%+v", r)
	}
	if _, err := os.Stat(filepath.Join(root, "lifecycle")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("lifecycle traffic exists: %v", err)
	}
}
