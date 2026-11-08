package modelengine

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=2s lane=default
func TestPDWireStreamedHandoffDeliversTokensIncrementally(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
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

	var mu sync.Mutex
	var streamed []int
	sink := func(tok int) error {
		mu.Lock()
		streamed = append(streamed, tok)
		mu.Unlock()
		return nil
	}
	res, err := client.ShipStream(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt}, sink)
	if err != nil {
		t.Fatalf("ShipStream: %v", err)
	}
	want := serialGreedyTokens(m, prompt)
	if !reflect.DeepEqual(res.Tokens, want) {
		t.Fatalf("streamed result tokens changed:\n got %v\nwant %v", res.Tokens, want)
	}
	mu.Lock()
	got := append([]int(nil), streamed...)
	mu.Unlock()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("sink saw %v out of order or incomplete, want every generated token in order:\n%v", got, want)
	}
	if res.DecodeWorker == "" || res.SpanDigest == "" || res.BytesMoved <= 0 {
		t.Fatalf("streamed ship result incomplete: %+v", res)
	}
	decode, ok := decodeCluster.DecodeWorker(res.DecodeWorker)
	if !ok {
		t.Fatalf("decode worker %q not found on decode cluster", res.DecodeWorker)
	}
	if stats := decode.Stats(); stats.DecodeImports != 1 || stats.RePrefillOnDecode != 0 {
		t.Fatalf("decode stats = %+v, want one import/admit and no decode-side prefill", stats)
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireStreamNilSinkRefuses(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 1)
	defer cluster.Close()
	client := NewPDWireClient(cluster, "127.0.0.1:1", time.Second)
	if _, err := client.ShipStream(context.Background(), PDShipRequest{}, nil); err == nil {
		t.Fatal("ShipStream with nil sink succeeded, want a refusal")
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireStreamConsumerStopSurfacesClosedSentinel(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	decodeCluster := NewNativePDCluster(m, 1)
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
	prompt := []int{3, 7, 11}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	seen := 0
	sink := func(int) error {
		seen++
		return errors.New("front door response writer failed")
	}
	_, err = client.ShipStream(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt}, sink)
	if !errors.Is(err, ErrPDStreamConsumerStopped) {
		t.Fatalf("ShipStream error = %v, want errors.Is(_, ErrPDStreamConsumerStopped)", err)
	}
	if seen == 0 {
		t.Fatal("sink never fired, want the abort to happen after the first streamed token")
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireLegacyShipAgainstStreamingServerKeepsSingleReply(t *testing.T) {
	// An older-style client (plain Ship, stream_tokens unset) must still pair
	// with the current server: the historical single-reply exchange survives.
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	decodeCluster := NewNativePDCluster(m, 1)
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
	prompt := []int{3, 7, 11}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}
	res, err := client.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt})
	if err != nil {
		t.Fatalf("Ship: %v", err)
	}
	if want := serialGreedyTokens(m, prompt); !reflect.DeepEqual(res.Tokens, want) {
		t.Fatalf("legacy ship tokens changed:\n got %v\nwant %v", res.Tokens, want)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireGenBudgetOutsideBoundRefuses(t *testing.T) {
	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 1)
	defer cluster.Close()
	client := NewPDWireClient(cluster, "127.0.0.1:1", time.Second)
	req := PDShipRequest{KVFrame: []byte{1}, Logits: make([]float32, m.Cfg.VocabSize), Prompt: []int{3}}
	for _, budget := range []int{-1, PDWireMaxPromptTokens + 1} {
		req.GenBudget = budget
		if _, err := client.Ship(context.Background(), req); !errors.Is(err, ErrPDWireProtocol) {
			t.Fatalf("Ship(GenBudget=%d) err = %v, want errors.Is(_, ErrPDWireProtocol)", budget, err)
		}
	}
}

// fak-test:runtime fast est=2s lane=default
func TestPDWireGenBudgetRetiresImportedLaneEarly(t *testing.T) {
	// max_tokens rides the wire as the imported lane's retirement budget: the
	// lane stops at the request cap instead of the engine-wide genTokens, and
	// a zero budget keeps the historical full-length generation.
	m := model.NewSynthetic(SyntheticConfig())
	prefillCluster := NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	decodeCluster := NewNativePDCluster(m, 1)
	defer decodeCluster.Close()

	server := NewPDWireServer(decodeCluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	defer server.Close()

	client := NewPDWireClient(prefillCluster, server.Addr().String(), time.Second)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	prompt := []int{3, 7, 11, 5}
	frame, logits, err := prefillCluster.PrefillForWire(ctx, pdWireTestRequest(prompt))
	if err != nil {
		t.Fatalf("PrefillForWire: %v", err)
	}

	budgeted, err := client.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt, GenBudget: 3})
	if err != nil {
		t.Fatalf("Ship(GenBudget=3): %v", err)
	}
	if len(budgeted.Tokens) != 3 {
		t.Fatalf("budgeted lane generated %d tokens, want exactly 3", len(budgeted.Tokens))
	}
	full, err := client.Ship(ctx, PDShipRequest{KVFrame: frame, Logits: logits, Prompt: prompt})
	if err != nil {
		t.Fatalf("Ship(default): %v", err)
	}
	if len(full.Tokens) != genTokens {
		t.Fatalf("default lane generated %d tokens, want the engine-wide %d", len(full.Tokens), genTokens)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPDWireGenBudgetHeaderRefusedAtAck(t *testing.T) {
	// The decode role validates the budget itself at the header: a producer
	// declaring an out-of-bound gen budget gets a negative ack with the closed
	// protocol code before any payload byte moves.
	m := model.NewSynthetic(SyntheticConfig())
	cluster := NewNativePDCluster(m, 1)
	defer cluster.Close()
	server := NewPDWireServer(cluster)
	if err := server.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("Listen: %v", err)
	}
	go func() { _ = server.Serve() }()
	defer server.Close()

	prompt := []int{3, 7}
	header := pdWireHeader{
		Protocol:     PDWireProtocol,
		ConfigSHA256: pdWireConfigDigest(m.Cfg),
		VocabSize:    m.Cfg.VocabSize,
		PromptIDs:    prompt,
		PromptSHA256: pdWirePromptDigest(prompt),
		KVFrameBytes: 4,
		LogitsBytes:  4 * m.Cfg.VocabSize,
		GenBudget:    PDWireMaxPromptTokens + 1,
	}
	conn, err := net.Dial("tcp", server.Addr().String())
	if err != nil {
		t.Fatalf("Dial: %v", err)
	}
	defer conn.Close()
	if err := pdWireWriteFrame(conn, mustJSON(t, header), pdWireMaxHeaderBytes); err != nil {
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
	if ack.OK || ack.Code != pdWireCodeProtocol {
		t.Fatalf("ack = %+v, want negative ack with code %q", ack, pdWireCodeProtocol)
	}
}
