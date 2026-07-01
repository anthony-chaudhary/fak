package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestProxyResultQuarantineJoinsOriginCallSeq(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, allowAllAdj{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	j := journal.OpenMemory()
	abi.RegisterEmitter(j)

	srv, err := New(Config{EngineID: "test", Model: "test", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	const (
		trace = "trace-proxy-1958"
		id    = "call_1958"
		tool  = "fetch_url"
		args  = `{"url":"https://example.test"}`
	)
	kept, _, dropped, _, _ := srv.adjudicateProposedServed(context.Background(), []agent.ToolCall{{
		ID:       id,
		Type:     "function",
		Function: agent.Func{Name: tool, Arguments: args},
	}}, trace)
	if dropped != 0 || len(kept) != 1 {
		t.Fatalf("proposed call = kept %d dropped %d, want kept 1 dropped 0", len(kept), dropped)
	}

	messages := inboundTurn(id, tool, args, `{"page":"api_key=sk-abcdef0123456789abcdef0123"}`)
	admissions, err := srv.admitInboundResults(context.Background(), messages, nil, trace)
	if err != nil {
		t.Fatalf("admitInboundResults: %v", err)
	}
	if len(admissions) != 1 || admissions[0].Verdict.Kind != "QUARANTINE" {
		t.Fatalf("result admissions = %+v, want one QUARANTINE", admissions)
	}

	var decideSeq, quarantineSeq uint64
	var quarantineWitness string
	for _, row := range j.Recent(0) {
		if row.TraceID != trace || row.Tool != tool {
			continue
		}
		switch row.Kind {
		case "DECIDE":
			decideSeq = row.CallSeq
		case "QUARANTINE":
			quarantineSeq = row.CallSeq
			quarantineWitness = row.Witness
		}
	}
	if decideSeq == 0 {
		t.Fatalf("proxy DECIDE row did not carry a call_seq")
	}
	if quarantineSeq != decideSeq {
		t.Fatalf("proxy QUARANTINE call_seq = %d, want DECIDE call_seq %d", quarantineSeq, decideSeq)
	}
	if quarantineWitness == "" {
		t.Fatalf("proxy QUARANTINE row did not carry a forensic witness")
	}
}
