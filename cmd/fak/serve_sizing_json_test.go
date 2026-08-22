// Tests for the `fak serve --plan-json` dry-run artifact (#4361): the emitted
// sizing numbers must be the SAME classed demands the live serve arm's fit check
// admits against, versioned, with concrete scopes, tier rollups, per-pool usable
// bytes, and would-be refusals downgraded to warnings instead of lost.
package main

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// sizingNamedBackend gives the serveCapBackend capacity stub the Name() the
// artifact's pool row reads (the embedded nil compute.Backend would panic).
type sizingNamedBackend struct{ serveCapBackend }

func (sizingNamedBackend) Name() string { return "testdev" }

func TestServeSizingArtifactCPUArm(t *testing.T) {
	t.Setenv("FAK_Q4K", "")
	ws := serveSynthConfiguredWeightSource(t)
	art, err := buildServeSizingArtifact(ws, nil, false, 16, "synthetic.gguf", 4321)
	if err != nil {
		t.Fatalf("buildServeSizingArtifact: %v", err)
	}
	if art.Version != serveSizingVersion {
		t.Fatalf("version = %q, want %q", art.Version, serveSizingVersion)
	}
	if art.Model != "synthetic.gguf" || art.Arm != "cpu-lean-q8" || art.ContextBudgetTokens != 16 {
		t.Fatalf("header fields wrong: %+v", art)
	}
	if len(art.Demands) == 0 {
		t.Fatal("expected classed demands from the header estimate, got none")
	}
	var grand int64
	for _, d := range art.Demands {
		if d.Scope != string(compute.MemoryScopeDevice) && d.Scope != string(compute.MemoryScopeHost) {
			t.Fatalf("demand %q scope %q is not concrete", d.Detail, d.Scope)
		}
		grand += d.Bytes
	}
	// Pure-CPU serve: every demand is anonymous host RAM, so the ram tier is the
	// grand total (the RefuseMemoryPlanIfTooBigForHost ceiling) and vram is zero.
	if art.Tiers.RAMBytes != grand || art.Tiers.VRAMBytes != 0 || art.Tiers.DiskBytes != 4321 {
		t.Fatalf("tiers = %+v, want ram=%d vram=0 disk=4321", art.Tiers, grand)
	}
	if art.Warnings == nil {
		t.Fatal("warnings must be an array, never nil")
	}
	if len(art.Pools) != 1 || art.Pools[0].Pool != "host" {
		t.Fatalf("CPU arm pools = %+v, want exactly the host pool", art.Pools)
	}
}

func TestServeSizingArtifactDeviceArmRefusalBecomesWarning(t *testing.T) {
	t.Setenv("FAK_Q4K", "")
	ws := serveSynthConfiguredWeightSource(t)
	// A known device pool far too small for the weights: the live arm would refuse
	// with a typed FitError; the dry-run must emit anyway and carry the refusal.
	be := sizingNamedBackend{serveCapBackend{total: 1 << 19, free: 1 << 19, known: true, uploadDtype: true}}
	art, err := buildServeSizingArtifact(ws, be, false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatalf("buildServeSizingArtifact: %v", err)
	}
	if art.Arm != "device-resident-q4k" {
		t.Fatalf("arm = %q, want device-resident-q4k", art.Arm)
	}
	found := false
	for _, w := range art.Warnings {
		if strings.Contains(w, "device fit would refuse") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected a device-fit refusal warning, got %v", art.Warnings)
	}
	if art.Tiers.VRAMBytes <= 0 {
		t.Fatalf("device arm vram tier = %d, want > 0", art.Tiers.VRAMBytes)
	}
	if len(art.Pools) != 2 || art.Pools[0].Pool != "device" || art.Pools[0].Backend != "testdev" {
		t.Fatalf("pools = %+v, want device pool first with backend name", art.Pools)
	}
	wantUsable := compute.BudgetAfterHeadroom(1<<19, serveGGUFDeviceHeadroom)
	if art.Pools[0].UsableBytes != wantUsable {
		t.Fatalf("device usable_bytes = %d, want the BudgetAfterHeadroom number %d", art.Pools[0].UsableBytes, wantUsable)
	}
}

func TestServeSizingArtifactJSONShape(t *testing.T) {
	t.Setenv("FAK_Q4K", "")
	ws := serveSynthConfiguredWeightSource(t)
	art, err := buildServeSizingArtifact(ws, nil, false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatalf("buildServeSizingArtifact: %v", err)
	}
	out, err := json.Marshal(art)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, key := range []string{`"version"`, `"model"`, `"arm"`, `"demands"`, `"tiers"`, `"pools"`, `"usable_bytes"`, `"warnings"`} {
		if !strings.Contains(string(out), key) {
			t.Fatalf("artifact JSON missing %s: %s", key, out)
		}
	}
	if strings.Contains(string(out), `"warnings":null`) {
		t.Fatalf("warnings marshaled to null: %s", out)
	}
}

func TestServeDeviceResidentQ4KDefaultsOnWithRollback(t *testing.T) {
	quantized := serveCapBackend{Backend: compute.Default(), uploadDtype: true}

	t.Setenv("FAK_Q4K", "")
	if !serveDeviceResidentQ4K(quantized) {
		t.Fatal("quantized device backend should default to resident Q4_K")
	}
	if got := serveSizingArm(quantized, false); got != "device-resident-q4k" {
		t.Fatalf("default sizing arm = %q, want device-resident-q4k", got)
	}

	t.Setenv("FAK_Q4K", "0")
	if serveDeviceResidentQ4K(quantized) {
		t.Fatal("FAK_Q4K=0 should roll back to legacy Q8 staging")
	}
	if got := serveSizingArm(quantized, false); got != "device-lean-q8" {
		t.Fatalf("rollback sizing arm = %q, want device-lean-q8", got)
	}

	plain := serveCapBackend{Backend: compute.Default()}
	t.Setenv("FAK_Q4K", "")
	if serveDeviceResidentQ4K(plain) {
		t.Fatal("backend without quantized upload cannot select resident device Q4_K")
	}
}
