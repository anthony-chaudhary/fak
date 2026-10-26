package remotemaint

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

// fak-test:runtime fast est=0.1s lane=default
func TestReservationGrantAndReceiptRemainFinite(t *testing.T) {
	now := time.Now().UTC()
	g := validTestGrant(now)
	g.Operation, g.Target, g.HostKey, g.BootID, g.PayloadSHA256, g.PriorStateSHA256 = OperationMaintainLeaseAcquire, "", "", "", "", ""
	g.LeasePins, g.Units, g.PathPrefixes = nil, nil, nil
	root := t.TempDir()
	g.Paths = []string{filepath.Join(root, "leases.json")}
	g.Reservation = &TrustedReservation{WorkspaceRoot: root, NodeIDs: []string{"node-one"}, Kind: "maintain", TTLSeconds: 120, CatalogGeneration: "catalog-one"}
	if err := validateGrant(g, now); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Grant){
		func(g *Grant) { g.Target = "invented.invalid" },
		func(g *Grant) { g.LeasePins = []LeasePin{{NodeID: "node-one"}} },
		func(g *Grant) { g.Reservation.TTLSeconds = 86401 },
		func(g *Grant) { g.Reservation.NodeIDs = []string{"node-one", "node-one"} },
		func(g *Grant) { g.Reservation.Kind = "serve" },
		func(g *Grant) { g.Reservation.CatalogGeneration = "" },
	} {
		candidate := cloneGrant(g)
		mutate(&candidate)
		if err := validateGrant(candidate, now); err == nil {
			t.Fatal("widened reservation admitted")
		}
	}
	a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) {
		r.ReservationLeases = []LeasePin{{NodeID: "node-one", ID: "native-pin", Epoch: 1, ExpiresAt: r.EndedAt.Add(120 * time.Second)}}
	}}
	p, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
	if err != nil {
		t.Fatal(err)
	}
	// The authority and caller cannot mutate the prepared node set via aliases.
	copy := p.Grant()
	copy.Reservation.NodeIDs[0] = "changed"
	if p.Grant().Reservation.NodeIDs[0] != "node-one" {
		t.Fatal("prepared scope aliases caller")
	}
	if r, err := p.Execute(context.Background()); err != nil || len(r.ReservationLeases) != 1 {
		t.Fatalf("receipt=%+v error=%v", r, err)
	}
	a.receiptMutator = func(r *Receipt) {
		r.ReservationLeases = []LeasePin{{NodeID: "wrong-node", ID: "native-pin", Epoch: 1, ExpiresAt: r.EndedAt.Add(120 * time.Second)}}
	}
	p, err = Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
	if err != nil {
		t.Fatal(err)
	}
	if r, err := p.Execute(context.Background()); !errors.Is(err, ErrUnknownEffect) || r.Verified || len(r.ReservationLeases) != 0 {
		t.Fatalf("mismatched post-effect receipt=%+v err=%v", r, err)
	}
	systemd := validTestGrant(now)
	receipt := receiptFor(systemd, now)
	receipt.ReservationLeases = []LeasePin{{NodeID: "node-one"}}
	if err := validateReceipt(receipt, systemd, now); err == nil {
		t.Fatal("systemd accepted reservation leases")
	}
}

func reservationPinsGrant(t *testing.T) Grant {
	t.Helper()
	g := validTestGrant(time.Now().UTC())
	g.Operation, g.Target, g.HostKey, g.BootID, g.PayloadSHA256, g.PriorStateSHA256 = OperationMaintainLeaseAcquire, "", "", "", "", ""
	g.LeasePins, g.Units, g.PathPrefixes = nil, nil, nil
	root := t.TempDir()
	g.Paths = []string{filepath.Join(root, "leases.json")}
	g.Reservation = &TrustedReservation{WorkspaceRoot: root, NodeIDs: []string{"node-one", "node-two"}, Kind: "maintain", TTLSeconds: 120, CatalogGeneration: "catalog-one", ExpectedStoreEpoch: 8}
	return g
}

// fak-test:runtime fast est=0.1s lane=default
func TestReservationReceiptRequiresNativeEpochsAndUniqueIDs(t *testing.T) {
	for _, mode := range []string{"valid", "duplicate-ID", "stale-epoch", "epoch-gap"} {
		t.Run(mode, func(t *testing.T) {
			g := reservationPinsGrant(t)
			a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) {
				r.ReservationLeases = []LeasePin{{NodeID: "node-one", ID: "native-one", Epoch: 9, ExpiresAt: r.EndedAt.Add(120 * time.Second)}, {NodeID: "node-two", ID: "native-two", Epoch: 10, ExpiresAt: r.EndedAt.Add(120 * time.Second)}}
				switch mode {
				case "duplicate-ID":
					r.ReservationLeases[1].ID = r.ReservationLeases[0].ID
				case "stale-epoch":
					r.ReservationLeases[0].Epoch = 8
				case "epoch-gap":
					r.ReservationLeases[1].Epoch = 11
				}
			}}
			p, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
			if err != nil {
				t.Fatal(err)
			}
			r, err := p.Execute(context.Background())
			if len(a.executeCalls) != 1 {
				t.Fatal("effect count differs from one")
			}
			if mode == "valid" {
				if err != nil || !r.Verified || len(r.ReservationLeases) != 2 {
					t.Fatalf("valid receipt=%+v err=%v", r, err)
				}
				return
			}
			if !errors.Is(err, ErrUnknownEffect) || r.Outcome != OutcomeUnknownEffect || r.Verified || len(r.ReservationLeases) != 0 {
				t.Fatalf("invalid native pins exposed: receipt=%+v err=%v", r, err)
			}
		})
	}
	t.Run("overflow-grant-is-pre-effect", func(t *testing.T) {
		g := reservationPinsGrant(t)
		g.Reservation.ExpectedStoreEpoch = ^uint64(0)
		a := &fakeAuthority{grant: g}
		if _, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest}); !errors.Is(err, ErrScope) {
			t.Errorf("overflow grant accepted: %v", err)
		}
		if len(a.executeCalls) != 0 {
			t.Fatal("overflow grant attempted execution")
		}
		now := time.Now().UTC()
		r := receiptFor(g, now)
		r.ReservationLeases = []LeasePin{{NodeID: "node-one", ID: "one", Epoch: 1, ExpiresAt: now.Add(120 * time.Second)}, {NodeID: "node-two", ID: "two", Epoch: 2, ExpiresAt: now.Add(120 * time.Second)}}
		if err := validateReceipt(r, g, now); err == nil {
			t.Fatal("overflow receipt accepted")
		}
	})
}

// fak-test:runtime fast est=0.1s lane=default
func TestReservationReleaseRequiresExactOwnedPins(t *testing.T) {
	newGrant := func() Grant {
		g := reservationPinsGrant(t)
		g.Operation = OperationMaintainLeaseRelease
		g.Reservation.TTLSeconds = 0
		g.Reservation.ExpectedStoreEpoch = 10
		g.LeasePins = []LeasePin{{NodeID: "node-one", ID: "native-one", Epoch: 9, ExpiresAt: g.ExpiresAt.Add(time.Minute)}, {NodeID: "node-two", ID: "native-two", Epoch: 10, ExpiresAt: g.ExpiresAt.Add(time.Minute)}}
		return g
	}
	t.Run("exact-release-receipt", func(t *testing.T) {
		g := newGrant()
		a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) { r.ReservationLeases = append([]LeasePin(nil), g.LeasePins...) }}
		p, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
		if err != nil {
			t.Fatal(err)
		}
		r, err := p.Execute(context.Background())
		if err != nil || !r.Verified || r.Outcome != OutcomeApplied || len(r.ReservationLeases) != 2 || len(a.executeCalls) != 1 {
			t.Fatalf("release=%+v err=%v calls=%d", r, err, len(a.executeCalls))
		}
		for i, pin := range r.ReservationLeases {
			if pin != p.Grant().LeasePins[i] {
				t.Fatal("release altered a bound native pin")
			}
		}
	})
	t.Run("expired-exact-pins", func(t *testing.T) {
		g := newGrant()
		for i := range g.LeasePins {
			g.LeasePins[i].ExpiresAt = time.Now().Add(-time.Hour).UTC()
		}
		a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) { r.ReservationLeases = append([]LeasePin(nil), g.LeasePins...) }}
		p, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
		if err != nil {
			t.Fatal(err)
		}
		r, err := p.Execute(context.Background())
		if err != nil || !r.Verified || r.Outcome != OutcomeApplied || len(r.ReservationLeases) != 2 || len(a.executeCalls) != 1 {
			t.Fatalf("expired exact release=%+v err=%v executions=%d", r, err, len(a.executeCalls))
		}
		for i, pin := range r.ReservationLeases {
			if pin != p.Grant().LeasePins[i] {
				t.Fatal("expired cleanup changed exact pin identity")
			}
		}
	})
	for _, test := range []struct {
		name   string
		mutate func(*Grant)
	}{
		{"positive-TTL", func(g *Grant) { g.Reservation.TTLSeconds = 1 }},
		{"missing-pins", func(g *Grant) { g.LeasePins = nil }},
		{"duplicate-ID", func(g *Grant) { g.LeasePins[1].ID = g.LeasePins[0].ID }},
		{"zero-epoch", func(g *Grant) { g.LeasePins[0].Epoch = 0 }},
		{"future-epoch", func(g *Grant) { g.LeasePins[1].Epoch = 11 }},
		{"wrong-node", func(g *Grant) { g.LeasePins[0].NodeID = "node-foreign" }},
		{"zero-expiry", func(g *Grant) { g.LeasePins[0].ExpiresAt = time.Time{} }},
		{"SSH-scope", func(g *Grant) { g.Target = "foreign.invalid" }},
	} {
		t.Run("pre-effect-"+test.name, func(t *testing.T) {
			g := newGrant()
			test.mutate(&g)
			a := &fakeAuthority{grant: g}
			p, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
			if err == nil || p != nil || len(a.executeCalls) != 0 {
				t.Fatalf("invalid release admitted: prepared=%v err=%v executions=%d", p != nil, err, len(a.executeCalls))
			}
		})
	}
	for _, mode := range []string{"ID", "epoch", "expiry", "node", "missing", "Readback", "unchanged-outcome"} {
		t.Run("post-effect-"+mode, func(t *testing.T) {
			g := newGrant()
			a := &fakeAuthority{grant: g, receiptMutator: func(r *Receipt) {
				r.ReservationLeases = append([]LeasePin(nil), g.LeasePins...)
				switch mode {
				case "ID":
					r.ReservationLeases[0].ID = "foreign"
				case "epoch":
					r.ReservationLeases[0].Epoch++
				case "expiry":
					r.ReservationLeases[0].ExpiresAt = r.ReservationLeases[0].ExpiresAt.Add(time.Second)
				case "node":
					r.ReservationLeases[0].NodeID = "foreign"
				case "missing":
					r.ReservationLeases = nil
				case "Readback":
					r.Readback = &SystemdProfileReadback{}
				case "unchanged-outcome":
					r.Outcome = OutcomeUnchanged
				}
			}}
			p, err := Prepare(context.Background(), a, Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest})
			if err != nil {
				t.Fatal(err)
			}
			r, err := p.Execute(context.Background())
			if !errors.Is(err, ErrUnknownEffect) || r.Outcome != OutcomeUnknownEffect || r.Verified || len(r.ReservationLeases) != 0 || r.Readback != nil || len(a.executeCalls) != 1 {
				t.Fatalf("unbound release exposed: receipt=%+v err=%v executions=%d", r, err, len(a.executeCalls))
			}
		})
	}
}
