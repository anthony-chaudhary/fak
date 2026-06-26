package main

import (
	"context"
	"testing"
)

func TestServedReadProofCrossesHTTPAndShowsTier2Hits(t *testing.T) {
	const calls = 4
	proof, err := buildServedReadProof(context.Background(), calls)
	if err != nil {
		t.Fatal(err)
	}
	if proof.Schema != "fak.tokendemo.served-read-cache.v1" {
		t.Fatalf("schema = %q, want fak.tokendemo.served-read-cache.v1", proof.Schema)
	}
	if proof.Surface != "http+mcp" || len(proof.Endpoints) != 2 {
		t.Fatalf("surface/endpoints = %s/%v, want http+mcp with two endpoints", proof.Surface, proof.Endpoints)
	}
	total := calls * 2
	if proof.CallsPerSurface != calls || proof.Calls != total {
		t.Fatalf("calls per surface/total = %d/%d, want %d/%d", proof.CallsPerSurface, proof.Calls, calls, total)
	}
	if proof.RawEngineCalls != int64(total) {
		t.Fatalf("raw engine calls = %d, want %d", proof.RawEngineCalls, total)
	}
	if proof.FakEngineCalls != 1 {
		t.Fatalf("served engine calls = %d, want 1", proof.FakEngineCalls)
	}
	if proof.VDSOHits != int64(total-1) || proof.RoundtripsCollapsed != int64(total-1) {
		t.Fatalf("vdso hits/roundtrips = %d/%d, want %d/%d",
			proof.VDSOHits, proof.RoundtripsCollapsed, total-1, total-1)
	}
	if len(proof.CallsDetail) != total {
		t.Fatalf("call rows = %d, want %d", len(proof.CallsDetail), total)
	}
	first := proof.CallsDetail[0]
	if first.Surface != "http" || !first.EngineRanFak || first.Engine != "fakread" || first.ServedBy == "vdso" {
		t.Fatalf("first served call should reach fakread, got %+v", first)
	}
	for _, row := range proof.CallsDetail[1:] {
		if row.ServedBy != "vdso" || row.Tier != "2" || row.EngineRanFak {
			t.Fatalf("repeat call should be served_by=vdso tier=2 with no engine: %+v", row)
		}
		if !row.ResponseMetaOK {
			t.Fatalf("repeat call did not carry accepted response metadata: %+v", row)
		}
	}
	sawMCP := false
	for _, row := range proof.CallsDetail {
		if row.Surface == "mcp" {
			sawMCP = true
			if row.ServedBy != "vdso" || row.Tier != "2" {
				t.Fatalf("MCP row should reuse the warmed tier-2 entry: %+v", row)
			}
		}
	}
	if !sawMCP {
		t.Fatal("proof did not exercise the MCP surface")
	}

	ev := proof.GatewayMetricEvidence
	if ev.KernelSubmits != int64(total) || ev.KernelEngineCalls != 1 || ev.KernelVDSOHits != int64(total-1) {
		t.Fatalf("kernel metrics = submits %d engine %d vdso %d, want %d/1/%d",
			ev.KernelSubmits, ev.KernelEngineCalls, ev.KernelVDSOHits, total, total-1)
	}
	if ev.GatewayHTTPPostSyscall200 != calls {
		t.Fatalf("HTTP syscall metric = %d, want %d", ev.GatewayHTTPPostSyscall200, calls)
	}
	if ev.GatewayMCPPost200 != calls {
		t.Fatalf("MCP metric = %d, want %d", ev.GatewayMCPPost200, calls)
	}
	if ev.GatewaySyscallAllowEngine != 1 || ev.GatewaySyscallAllowVDSO != int64(total-1) {
		t.Fatalf("operation metrics engine/vdso = %d/%d, want 1/%d",
			ev.GatewaySyscallAllowEngine, ev.GatewaySyscallAllowVDSO, total-1)
	}
}

func TestServedReadProofRejectsTooFewCalls(t *testing.T) {
	if _, err := buildServedReadProof(context.Background(), 1); err == nil {
		t.Fatal("calls=1 should not satisfy same-read served cache proof")
	}
}
