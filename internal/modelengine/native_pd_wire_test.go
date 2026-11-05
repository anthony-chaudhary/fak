package modelengine

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// pdWireTestRequest builds the closed-contract request shape both wire tests
// admit under: a trusted fleet-scope call with an explicit prompt.
func pdWireTestRequest(prompt []int) NativePDRequest {
	return NativePDRequest{
		Call:             inlineCall("issue28_pd", `{"prompt":"alpha"}`),
		Prompt:           prompt,
		ModelID:          "synthetic-test",
		TokenizerID:      "byte",
		Lease:            "lease-wire",
		Taint:            abi.TaintTrusted,
		Scope:            abi.ScopeFleet,
		AdmissionVerdict: cachemeta.AdmissionAllow,
		AdmittedBy:       "admission-test",
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireTwoProcessHandoffMatchesSerialGreedy(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1) // one nominal decode worker; unused locally
	defer prefillCluster.Close()
	decodeCluster := NewNativePDCluster(m, 2)
	defer decodeCluster.Close()

	server := NewPDWireServer(decodeCluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	defer server.Close()

	client := NewPDWireClient(prefillCluster, server.Addr().String(), time.Second)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5, 17, 19}

	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	if len(frame) == 0 || len(logits) != m.Cfg.VocabSize {
		t.Fatalf("prefill half produced frame=%d bytes, logits=%d, want non-empty frame and %d logits", len(frame), len(logits), m.Cfg.VocabSize)
	}

	res, err := client.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt})
	if err != nil {
		t.Fatalf("Ship: %v", err)
	}
	want := serialGreedyTokens(m, prompt)
	if !reflect.DeepEqual(res.Tokens, want) {
		t.Fatalf("wire handoff tokens changed:\n got %v\nwant %v", res.Tokens, want)
	}
	if res.DecodeWorker == "" || res.SpanDigest == "" || res.BytesMoved <= 0 {
		t.Fatalf("ship result incomplete: %+v", res)
	}

	decode, ok := decodeCluster.DecodeWorker(res.DecodeWorker)
	if !ok {
		t.Fatalf("decode worker %q not found on decode cluster", res.DecodeWorker)
	}
	stats := decode.Stats()
	if stats.DecodeImports != 1 || stats.SchedulerAdmits != 1 || stats.RePrefillOnDecode != 0 {
		t.Fatalf("decode stats = %+v, want one import/admit and no decode-side prefill", stats)
	}
	if metrics := decodeCluster.Metrics(); !strings.Contains(metrics, `fak_native_pd_transfer_backend_total{backend="tcp"} 1`) {
		t.Fatalf("metrics missing tcp transfer observation:\n%s", metrics)
	}
	if entry, ok := decode.TransferEntry(res.SpanDigest); !ok || entry.Security.Taint != abi.TaintTrusted || entry.Residency.Lease != "lease-wire" {
		t.Fatalf("transfer entry for digest %q not recorded with trust metadata: ok=%v entry=%+v", res.SpanDigest, ok, entry)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireIdentityMismatchRefusesAtHeader(t *testing.T) {
	cfgA := SyntheticConfig()
	cfgB := SyntheticConfig()
	cfgB.HiddenSize = 32 // same vocabulary, different checkpoint geometry
	prem := model.NewSynthetic(cfgA)
	postm := model.NewSynthetic(cfgB)

	prefillCluster := NewNativePDCluster(prem, 1)
	defer prefillCluster.Close()
	decodeCluster := NewNativePDCluster(postm, 1)
	defer decodeCluster.Close()

	server := NewPDWireServer(decodeCluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	defer server.Close()

	client := NewPDWireClient(prefillCluster, server.Addr().String(), time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11}

	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	if _, err := client.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt}); err == nil {
		t.Fatal("Ship across mismatched configurations succeeded, want identity refusal")
	} else if !errors.Is(err, ErrPDWireIdentity) {
		t.Fatalf("Ship error = %v, want ErrPDWireIdentity", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireMalformedHeaderGetsNegativeAckProtocolCode(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 1)
	defer cluster.Close()

	server := NewPDWireServer(cluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	defer server.Close()

	cases := []struct {
		name   string
		header []byte
	}{
		{name: "garbage JSON", header: []byte("not-json")},
		{name: "wrong version", header: mustJSON(t, pdWireHeader{Protocol: "fak-pd-wire/v0"})},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conn, err := net.Dial("tcp", server.Addr().String())
			if err != nil {
				t.Fatalf("Dial: %v", err)
			}
			defer conn.Close()
			if err := pdWireWriteFrame(conn, tc.header, pdWireMaxHeaderBytes); err != nil {
				t.Fatalf("write header frame: %v", err)
			}
			raw, err := pdWireReadFrame(conn, pdWireMaxReplyBytes)
			if err != nil {
				t.Fatalf("read ack frame: %v", err)
			}
			var ack pdWireAck
			if err := json.Unmarshal(raw, &ack); err != nil {
				t.Fatalf("unmarshal ack: %v", err)
			}
			if ack.OK || ack.Code != pdWireCodeProtocol || ack.Error == "" {
				t.Fatalf("ack = %+v, want negative ack with code %q and detail", ack, pdWireCodeProtocol)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireCanceledShipSurfacesContextError(t *testing.T) {
	// A silent peer: accepts the handoff, swallows the header, and never
	// acknowledges, so Ship blocks on the ack read until the caller's context
	// loses. Cancellation must surface as context.Canceled, not a transport
	// fault.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen: %v", err)
	}
	defer ln.Close()
	headerSeen := make(chan struct{})
	go func() {
		conn, aerr := ln.Accept()
		if aerr != nil {
			close(headerSeen)
			return
		}
		defer conn.Close()
		if _, rerr := pdWireReadFrame(conn, pdWireMaxHeaderBytes); rerr != nil {
			close(headerSeen)
			return
		}
		close(headerSeen)
		// Sit on the socket: no ack ever comes.
		select {}
	}()

	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 1)
	defer cluster.Close()
	client := NewPDWireClient(cluster, ln.Addr().String(), time.Second)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := client.Ship(ctx, PDShipRequest{KVFrame: []byte{1}, Logits: make([]float32, m.Cfg.VocabSize), Prompt: []int{3}})
		done <- err
	}()
	<-headerSeen
	cancel()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Ship after cancel succeeded, want an error")
		}
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Ship error = %v, want context.Canceled", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Ship did not wake after context cancellation")
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireSessionHookDecoratesBothRoles(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	var prefillHooks, decodeHooks atomic.Int64
	prefillCluster := NewNativePDClusterWithResidency(m, 1, nil, WithNativePDSessionHook(func(*model.Session) {
		prefillHooks.Add(1)
	}))
	defer prefillCluster.Close()
	decodeCluster := NewNativePDClusterWithResidency(m, 1, nil, WithNativePDSessionHook(func(*model.Session) {
		decodeHooks.Add(1)
	}))
	defer decodeCluster.Close()

	server := NewPDWireServer(decodeCluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	defer server.Close()
	client := NewPDWireClient(prefillCluster, server.Addr().String(), time.Second)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	if prefillHooks.Load() < 1 {
		t.Fatalf("prefill session hook fired %d times, want >= 1", prefillHooks.Load())
	}
	if _, err := client.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt}); err != nil {
		t.Fatalf("Ship: %v", err)
	}
	if decodeHooks.Load() < 1 {
		t.Fatalf("decode lane session hook fired %d times, want >= 1", decodeHooks.Load())
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return raw
}
