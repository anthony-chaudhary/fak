package compute

import (
	"strings"
	"testing"
)

func fakeLedgerSnapshot() DeviceAllocSnapshot {
	return DeviceAllocSnapshot{
		Cells: []DeviceAllocCell{
			{Class: AllocWeights, Kind: AllocDeviceLocal, LiveBytes: 1000},
			{Class: AllocQ4CoeffCache, Kind: AllocDeviceLocal, LiveBytes: 500},
			{Class: AllocKVCache, Kind: AllocDeviceLocal, LiveBytes: 180},
			{Class: AllocIdlePool, Kind: AllocDeviceLocal, LiveBytes: 300},
			{Class: AllocGraphPool, Kind: AllocManaged, LiveBytes: 40},
			{Class: AllocActivationsScratch, Kind: AllocPinnedHost, LiveBytes: 9999},
		},
		MemInfoKnown:     true,
		UsedBytes:        2400,
		UnaccountedBytes: 380,
	}
}

func fakePlannerPlan() MemoryPlan {
	return MemoryPlan{
		{Class: MemoryWeights, Bytes: 900, Detail: "gguf-q4k-load"},
		{Class: MemoryKVCache, Bytes: 200, Detail: "kv prefill window", Scope: MemoryScopeDevice},
		{Class: MemoryScratchpad, Bytes: 50, Detail: "decode logits argmax"},
		{Class: MemoryWeights, Bytes: 7777, Detail: "host offload", Scope: MemoryScopeHost},
	}
}

func TestDiffMemoryPlanAgainstLedgerClassesAndSums(t *testing.T) {
	d := DiffMemoryPlanAgainstLedger("load", fakePlannerPlan(), fakeLedgerSnapshot())
	want := map[string][2]int64{
		"weights":        {900, 1000},
		"q4_coeff_cache": {0, 500},
		"kv_cache":       {200, 180},
		"sampling":       {50, 0},
		"graph_pool":     {0, 40},
		"idle_pool":      {0, 300},
	}
	if len(d.Rows) != len(want) {
		t.Fatalf("rows = %d, want %d: %+v", len(d.Rows), len(want), d.Rows)
	}
	var sumPred, sumAct, sumDelta int64
	for _, r := range d.Rows {
		w, ok := want[r.Class]
		if !ok {
			t.Fatalf("unexpected class %q", r.Class)
		}
		if r.PredictedBytes != w[0] || r.ActualBytes != w[1] || r.DeltaBytes != w[1]-w[0] {
			t.Fatalf("class %s = %d/%d/%d, want %d/%d", r.Class, r.PredictedBytes, r.ActualBytes, r.DeltaBytes, w[0], w[1])
		}
		sumPred += r.PredictedBytes
		sumAct += r.ActualBytes
		sumDelta += r.DeltaBytes
	}
	if d.PredictedBytes != sumPred || d.ActualBytes != sumAct || d.DeltaBytes != sumDelta {
		t.Fatalf("totals %d/%d/%d do not conserve row sums %d/%d/%d", d.PredictedBytes, d.ActualBytes, d.DeltaBytes, sumPred, sumAct, sumDelta)
	}
	if d.PredictedBytes != 1150 || d.ActualBytes != 2020 {
		t.Fatalf("host-scope demand or pinned-host cell leaked: predicted=%d actual=%d", d.PredictedBytes, d.ActualBytes)
	}
	if d.UnaccountedBytes != 380 || !d.MemInfoKnown || d.Phase != "load" {
		t.Fatalf("snapshot passthrough lost: %+v", d)
	}
	gotTop := make([]string, 0, len(d.TopPositive))
	for _, r := range d.TopPositive {
		gotTop = append(gotTop, r.Class)
	}
	if strings.Join(gotTop, ",") != "q4_coeff_cache,idle_pool,weights" {
		t.Fatalf("top positive = %v", gotTop)
	}
}

func TestDiffMemoryPlanAgainstLedgerEmptyIsZero(t *testing.T) {
	d := DiffMemoryPlanAgainstLedger("first_prefill", nil, DeviceAllocSnapshot{})
	if len(d.Rows) != 0 || len(d.TopPositive) != 0 || d.PredictedBytes != 0 || d.ActualBytes != 0 {
		t.Fatalf("empty diff = %+v", d)
	}
	if !strings.HasPrefix(d.LogLine(), "fak_mem_plan_diff phase=first_prefill ") || !strings.Contains(d.LogLine(), " top=none") {
		t.Fatalf("empty log line = %q", d.LogLine())
	}
}

func TestMemoryPlanLedgerDiffLogLineKeys(t *testing.T) {
	line := DiffMemoryPlanAgainstLedger("load", fakePlannerPlan(), fakeLedgerSnapshot()).LogLine()
	fields := strings.Fields(line)
	if fields[0] != "fak_mem_plan_diff" {
		t.Fatalf("prefix = %q", fields[0])
	}
	got := map[string]string{}
	var order []string
	for _, f := range fields[1:] {
		k, v, ok := strings.Cut(f, "=")
		if !ok {
			t.Fatalf("field %q not key=value", f)
		}
		got[k] = v
		order = append(order, k)
	}
	head := []string{"phase", "predicted", "actual", "delta", "unaccounted", "mem_info_known", "top"}
	for i, k := range head {
		if order[i] != k {
			t.Fatalf("key order = %v, want prefix %v", order, head)
		}
	}
	checks := map[string]string{
		"phase": "load", "predicted": "1150", "actual": "2020", "delta": "870", "unaccounted": "380",
		"top": "q4_coeff_cache:500,idle_pool:300,weights:100", "weights": "900/1000/100", "kv_cache": "200/180/-20",
	}
	for k, v := range checks {
		if got[k] != v {
			t.Fatalf("%s = %q, want %q (line %q)", k, got[k], v, line)
		}
	}
}

type fakeLedgerBackend struct {
	Backend
	snap DeviceAllocSnapshot
}

func (f fakeLedgerBackend) DeviceAllocLedger() DeviceAllocSnapshot { return f.snap }

func TestDeviceAllocLedgerOfAndClone(t *testing.T) {
	if _, ok := DeviceAllocLedgerOf(nil); ok {
		t.Fatal("nil backend reported a ledger")
	}
	snap, ok := DeviceAllocLedgerOf(fakeLedgerBackend{snap: fakeLedgerSnapshot()})
	if !ok || snap.UnaccountedBytes != 380 {
		t.Fatalf("ledger read = %v %+v", ok, snap)
	}
	d := DiffMemoryPlanAgainstLedger("load", fakePlannerPlan(), snap)
	c := d.Clone()
	c.Rows[0].ActualBytes = -1
	c.TopPositive[0].DeltaBytes = -1
	if d.Rows[0].ActualBytes == -1 || d.TopPositive[0].DeltaBytes == -1 {
		t.Fatal("Clone aliases row slices")
	}
	var nilDiff *MemoryPlanLedgerDiff
	if nilDiff.Clone() != nil {
		t.Fatal("nil Clone != nil")
	}
}
