package main

import (
	"encoding/json"
	"os"
	"os/exec"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// fak-test:runtime fast est=25ms lane=default
func TestServeSizingRingUsesLiveHeaderBudgetWithoutPayload(t *testing.T) {
	path := writeServeStreamedSynthGGUF(t, "ring-plan.gguf")
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	if err := os.Truncate(path, ws.File.TensorDataOffset); err != nil {
		t.Fatal(err)
	}
	be := sizingNamedBackend{serveCapBackend{total: 256 << 10, free: 256 << 10, known: true, uploadDtype: true}}
	host := serveExpertRingHost{Total: 32 << 30, Avail: 20 << 30, Known: true}
	size := serveExpertRingSize{Mode: serveExpertRingAuto}
	r := buildServeSizingRing(ws, be, path, 1, size, host, true, nil)
	if r.RingBytes <= 0 {
		t.Fatalf("no numeric ring: fit=%s reason=%s", r.Fit, r.Reason)
	}
	if r.Fit != "PLANNED" {
		t.Fatalf("known header projection fit=%s reason=%s", r.Fit, r.Reason)
	}
	fit := serveDeviceFitBudget(be)
	snapshot := serveSizingSnapshot{Backend: be, deviceTotal: 256 << 10, deviceFree: 256 << 10, deviceKnown: true, host: host}
	ring, selected, err := serveActivatedExpertRingPlacementSized(ws, snapshot, true, serveRingFullPlanTooBig(), r.ResolvedContextTokens, fit, size, host)
	if err != nil || !selected {
		t.Fatalf("live planner: selected=%t err=%v", selected, err)
	}
	if r.RingBytes != ring.RingBytes || r.NonRingDeviceBytes != ring.Sizing.NonRingDevice || r.HostBoundBytes != ring.Sizing.HostBound || r.GuardTrimBytes != ring.Sizing.GuardTrimBytes {
		t.Fatal("projection differs from live closed budget")
	}
	var expected []serveSizingDemandRow
	for _, d := range ring.Plan {
		expected = append(expected, serveSizingDemandRow{Class: string(d.Class), Bytes: d.Bytes, Detail: d.Detail, Scope: string(d.ScopeOrDefault()), DType: d.DType})
	}
	if !reflect.DeepEqual(r.Demands, expected) {
		t.Fatal("projected plan rows differ from live plan")
	}
	if r.CombinedFit != "UNKNOWN" || r.TargetCgroup != "UNKNOWN" || r.Runtime != "UNRUN" {
		t.Fatal("header plan overclaims target or runtime")
	}
	refused := buildServeSizingRing(ws, be, path, 1, serveExpertRingSize{Mode: serveExpertRingExplicit, Bytes: 1 << 30}, host, true, nil)
	if refused.Fit != "REFUSED" {
		t.Fatalf("explicit oversized ring fit=%s", refused.Fit)
	}
	unknown := buildServeSizingRing(ws, be, path, 4096, size, host, false, nil)
	if unknown.Fit != "UNKNOWN" || unknown.RequestedContextTokens != 4096 || unknown.ResolvedContextTokens != 0 {
		t.Fatal("missing peak estimate silently admitted")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestServeSizingRingFlagWinsBeforePlanExit(t *testing.T) {
	if path := os.Getenv("FAK_TEST_RING_PLAN_HEADER"); path != "" {
		fs, sf := newServeFlagSet()
		if err := fs.Parse([]string{"--plan-json", "--backend", "cpu", "--gguf", path, "--expert-ring-bytes", "invalid"}); err != nil {
			t.Fatal(err)
		}
		runServeSizingJSON(sf)
		os.Exit(0)
	}
	fs, sf := newServeFlagSet()
	t.Setenv(FakExpertRingBytesEnv, "auto")
	if err := fs.Parse([]string{"--plan-json", "--expert-ring-bytes", "invalid"}); err != nil {
		t.Fatal(err)
	}
	r := buildServeSizingRingForFlags(nil, nil, sf)
	if r.Fit != "REFUSED" {
		t.Fatal("plan ignored explicit invalid ring flag")
	}
	if os.Getenv(FakExpertRingBytesEnv) != "auto" {
		t.Fatal("read-only plan mutated environment")
	}
	path := writeServeStreamedSynthGGUF(t, "entrypoint-ring-plan.gguf")
	cmd := exec.Command(os.Args[0], "-test.run=^TestServeSizingRingFlagWinsBeforePlanExit$")
	cmd.Env = append(os.Environ(), "FAK_TEST_RING_PLAN_HEADER="+path)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("plan entrypoint: %v", err)
	}
	var art serveSizingArtifact
	if err := json.Unmarshal(out, &art); err != nil {
		t.Fatal(err)
	}
	if art.ActivatedExpertRing == nil || art.ActivatedExpertRing.Fit != "REFUSED" {
		t.Fatal("public plan entrypoint did not project explicit ring flag")
	}
}
