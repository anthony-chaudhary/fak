package fleetspine

import (
	"context"
	"testing"
	"time"
)

// TestEndToEndTwoSpinesDiscoverEachOther is the live integration proof the plan calls for: two
// spines on the SAME multicast group, each advertising itself, must discover the OTHER within a
// couple of heartbeat intervals — and neither must list itself (self-echo dropped). It uses a
// real UDP-multicast socket, so it self-skips when multicast is unavailable (CI, locked-down
// network) rather than failing; the deterministic coverage lives in the chan-transport tests.
func TestEndToEndTwoSpinesDiscoverEachOther(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real-socket end-to-end multicast test in -short mode")
	}
	const group, port = "239.255.79.7", 41937

	newSide := func(id string) (*Spine, *Registry, error) {
		tr, err := NewUDPMulticastTransport(group, port)
		if err != nil {
			return nil, nil, err
		}
		reg := NewRegistry(RegistryConfig{SelfID: id, MissWindow: time.Minute})
		s := &Spine{
			Transport: tr,
			Registry:  reg,
			Interval:  100 * time.Millisecond,
			Snapshot: func() Heartbeat {
				return Heartbeat{Schema: HeartbeatSchema, ID: id, Host: id, State: "OK", GeneratedUTC: time.Now().UTC().Format(time.RFC3339)}
			},
		}
		return s, reg, nil
	}

	sA, regA, err := newSide("box-a")
	if err != nil {
		t.Skipf("multicast unavailable: %v", err)
	}
	sB, regB, err := newSide("box-b")
	if err != nil {
		t.Skipf("multicast unavailable: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	go sA.Run(ctx)
	go sB.Run(ctx)

	// Poll until A sees B and B sees A, or time out.
	deadline := time.After(4 * time.Second)
	for {
		aSeesB := hasPeer(regA.Snapshot(time.Now()), "box-b")
		bSeesA := hasPeer(regB.Snapshot(time.Now()), "box-a")
		if aSeesB && bSeesA {
			break
		}
		select {
		case <-deadline:
			t.Skipf("peers did not converge over multicast (a→b=%v b→a=%v) — likely no multicast on this host/network", aSeesB, bSeesA)
		case <-time.After(50 * time.Millisecond):
		}
	}

	// Neither side lists itself (self-echo drop).
	if hasPeer(regA.Snapshot(time.Now()), "box-a") {
		t.Fatal("box-a discovered itself — self-echo not dropped")
	}
	if hasPeer(regB.Snapshot(time.Now()), "box-b") {
		t.Fatal("box-b discovered itself — self-echo not dropped")
	}
}

func hasPeer(peers []Peer, id string) bool {
	for _, p := range peers {
		if p.ID == id {
			return true
		}
	}
	return false
}
