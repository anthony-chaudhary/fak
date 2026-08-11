package fleetbus

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLifecycleDirBusBroadcastAndDurableAckFold(t *testing.T) {
	root := t.TempDir()
	bus := LifecycleDirBus{Root: root}
	now := time.Date(2026, 8, 11, 23, 0, 0, 0, time.UTC)
	req := LifecycleRequest{Schema: LifecycleSchema, TransactionID: "tx-1", ForestID: "forest-1", Generation: 3, Action: LifecyclePause, Deadline: now.Add(time.Minute), Capability: "forest.pause", IdempotencyKey: "idem-1", Authority: "guard:root"}
	members := []string{"codex", "claude", "helper"}
	if err := bus.Broadcast(req, members); err != nil {
		t.Fatal(err)
	}
	for _, m := range members {
		if _, err := os.Stat(filepath.Join(root, "lifecycle", "requests", "tx-1", m+".json")); err != nil {
			t.Fatalf("request %s not durable: %v", m, err)
		}
		if err := bus.WriteAck(LifecycleAck{Schema: LifecycleSchema, TransactionID: "tx-1", ForestID: "forest-1", MemberID: m, Generation: 3, State: AckCompleted, ReadbackRef: "witness:" + m, At: now}); err != nil {
			t.Fatal(err)
		}
	}
	// A fresh controller instance reads independently durable ACKs.
	got, err := (LifecycleDirBus{Root: root}).ReadAcks("tx-1")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || got[0].MemberID != "claude" || got[2].ReadbackRef != "witness:helper" {
		t.Fatalf("acks=%+v", got)
	}
}

func TestLifecycleGateFailsClosedAndRedeliveryIsIdempotent(t *testing.T) {
	now := time.Now().UTC()
	base := LifecycleRequest{Schema: LifecycleSchema, TransactionID: "tx", ForestID: "f", Generation: 2, Action: LifecycleCheckpoint, Deadline: now.Add(time.Minute), Capability: "checkpoint", IdempotencyKey: "k", Authority: "a"}
	g := LifecycleGate{Authority: "a", Capability: "checkpoint", Generation: 2}
	if err := g.Validate(base, now); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(base, now); err != nil {
		t.Fatalf("redelivery: %v", err)
	}
	cases := map[string]LifecycleRequest{"expired": base, "generation": base, "authority": base, "capability": base, "malformed": base, "replay": base}
	x := cases["expired"]
	x.Deadline = now.Add(-time.Second)
	cases["expired"] = x
	x = cases["generation"]
	x.Generation = 3
	cases["generation"] = x
	x = cases["authority"]
	x.Authority = "forged"
	cases["authority"] = x
	x = cases["capability"]
	x.Capability = "stop"
	cases["capability"] = x
	x = cases["malformed"]
	x.TransactionID = ""
	cases["malformed"] = x
	x = cases["replay"]
	x.Action = LifecycleStop
	cases["replay"] = x
	for name, req := range cases {
		if err := g.Validate(req, now); err == nil {
			t.Errorf("%s accepted", name)
		}
	}
}

func TestLifecycleBroadcastNeverSilentlyDropsMember(t *testing.T) {
	bus := LifecycleDirBus{Root: t.TempDir()}
	req := LifecycleRequest{Schema: LifecycleSchema, TransactionID: "tx", ForestID: "f", Generation: 1, Action: LifecycleStatus, Deadline: time.Now().Add(time.Minute), Capability: "status", IdempotencyKey: "k", Authority: "a"}
	if err := bus.Broadcast(req, []string{"ok", ""}); err == nil {
		t.Fatal("partial malformed broadcast reported success")
	}
}
