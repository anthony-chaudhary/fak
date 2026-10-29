package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

type fakeServeLedgerBackend struct {
	compute.Backend
	snap compute.DeviceAllocSnapshot
}

func (f fakeServeLedgerBackend) DeviceAllocLedger() compute.DeviceAllocSnapshot { return f.snap }

func TestAttachServePlanLedgerDiffLoadPhase(t *testing.T) {
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Bytes: 16 << 30, Detail: "gguf-q4k-load"},
		{Class: compute.MemoryKVCache, Bytes: 1 << 30, Detail: "kv context"},
	}
	be := fakeServeLedgerBackend{snap: compute.DeviceAllocSnapshot{
		Cells: []compute.DeviceAllocCell{
			{Class: compute.AllocWeights, Kind: compute.AllocDeviceLocal, LiveBytes: 16 << 30},
			{Class: compute.AllocKVCache, Kind: compute.AllocDeviceLocal, LiveBytes: 1 << 30},
			{Class: compute.AllocQ4CoeffCache, Kind: compute.AllocDeviceLocal, LiveBytes: 2 << 30},
		},
		MemInfoKnown: true, UsedBytes: 23 << 30, UnaccountedBytes: 4 << 30,
	}}
	p := &gateway.ModelLoadProfile{}
	attachServePlanLedgerDiff(p, plan, be)
	d := p.PlanLedgerDiff
	if d == nil || d.Phase != "load" {
		t.Fatalf("diff = %+v", d)
	}
	if d.DeltaBytes != 2<<30 || len(d.TopPositive) != 1 || d.TopPositive[0].Class != compute.AllocQ4CoeffCache.String() {
		t.Fatalf("delta=%d top=%+v", d.DeltaBytes, d.TopPositive)
	}
	if len(p.Messages) != 1 || p.Messages[0].Kind != "plan-ledger-diff" || !strings.HasPrefix(p.Messages[0].Text, "fak_mem_plan_diff phase=load ") {
		t.Fatalf("messages = %+v", p.Messages)
	}
}

func TestAttachServePlanLedgerDiffNoLedgerIsNoop(t *testing.T) {
	p := &gateway.ModelLoadProfile{}
	attachServePlanLedgerDiff(p, compute.MemoryPlan{{Class: compute.MemoryWeights, Bytes: 1}}, nil)
	if p.PlanLedgerDiff != nil || len(p.Messages) != 0 {
		t.Fatalf("no-ledger backend attached %+v", p)
	}
}
