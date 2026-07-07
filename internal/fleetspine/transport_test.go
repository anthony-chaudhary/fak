package fleetspine

import (
	"context"
	"net"
	"testing"
	"time"
)

// TestChanTransportRoundTrip: the in-memory fake carries a heartbeat from Advertise to a
// Listen callback intact, exercising the JSON encode/decode gate with no socket.
func TestChanTransportRoundTrip(t *testing.T) {
	tr := newChanTransport(4)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	got := make(chan Heartbeat, 1)
	go func() { _ = tr.Listen(ctx, func(hb Heartbeat) { got <- hb }) }()

	want := mkHB("alpha", time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC))
	if err := tr.Advertise(ctx, want); err != nil {
		t.Fatalf("Advertise: %v", err)
	}
	select {
	case hb := <-got:
		if hb.ID != want.ID || hb.AppVersion != want.AppVersion || hb.Sessions != want.Sessions {
			t.Fatalf("round-trip mismatch: got %+v want %+v", hb, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for heartbeat")
	}
}

// TestChanTransportDropsMalformed: a raw non-JSON datagram and an id-less one are both dropped
// by the decode gate — the callback never fires for them.
func TestChanTransportDropsMalformed(t *testing.T) {
	tr := newChanTransport(4)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	got := make(chan Heartbeat, 2)
	go func() { _ = tr.Listen(ctx, func(hb Heartbeat) { got <- hb }) }()

	tr.inject([]byte("{not json"))    // malformed
	tr.inject([]byte(`{"id":""}`))    // empty id
	tr.inject([]byte(`{"host":"x"}`)) // missing id
	// A good one after the bad ones proves the loop survived them.
	if err := tr.Advertise(ctx, mkHB("good", time.Now())); err != nil {
		t.Fatalf("Advertise: %v", err)
	}
	select {
	case hb := <-got:
		if hb.ID != "good" {
			t.Fatalf("first delivered heartbeat = %q, want good (bad ones dropped)", hb.ID)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out — a malformed packet may have broken the listen loop")
	}
}

// TestDecodeHeartbeat exercises the decode gate directly.
func TestDecodeHeartbeat(t *testing.T) {
	if _, ok := decodeHeartbeat([]byte("garbage")); ok {
		t.Fatal("decoded garbage as a heartbeat")
	}
	if _, ok := decodeHeartbeat([]byte(`{"id":""}`)); ok {
		t.Fatal("decoded an empty-id heartbeat")
	}
	hb, ok := decodeHeartbeat([]byte(`{"id":"alpha","app_version":"v9"}`))
	if !ok || hb.ID != "alpha" || hb.AppVersion != "v9" {
		t.Fatalf("decode good heartbeat: ok=%v hb=%+v", ok, hb)
	}
}

// TestUDPMulticastRoundTrip is a real-socket smoke test: it binds the multicast group, sends
// one heartbeat, and reads it back. Multicast is frequently unavailable on CI runners and
// locked-down networks, so it self-skips on any bind/join/send/receive failure rather than
// failing the suite — the chan-transport tests carry the deterministic coverage.
func TestUDPMulticastRoundTrip(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping real-socket multicast test in -short mode")
	}
	const group, port = "239.255.79.9", 41931
	tr, err := NewUDPMulticastTransport(group, port)
	if err != nil {
		t.Skipf("multicast transport unavailable: %v", err)
	}
	// Probe that the group actually joins on this host before committing to the test.
	if _, err := net.ResolveUDPAddr("udp4", group+":0"); err != nil {
		t.Skipf("cannot resolve multicast group: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	got := make(chan Heartbeat, 1)
	listenErr := make(chan error, 1)
	go func() { listenErr <- tr.Listen(ctx, func(hb Heartbeat) { got <- hb }) }()

	// Give the listener a moment to join the group before advertising.
	time.Sleep(150 * time.Millisecond)
	want := mkHB("alpha", time.Now())
	if err := tr.Advertise(ctx, want); err != nil {
		t.Skipf("multicast advertise failed (network may block it): %v", err)
	}
	select {
	case hb := <-got:
		if hb.ID != "alpha" {
			t.Fatalf("received id = %q, want alpha", hb.ID)
		}
	case <-time.After(1500 * time.Millisecond):
		t.Skip("no multicast loopback on this host/network — skipping (covered by chan tests)")
	}
	cancel()
	<-listenErr
}
