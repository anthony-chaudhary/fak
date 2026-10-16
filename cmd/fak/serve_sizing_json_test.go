// Tests for the `fak serve --plan-json` dry-run artifact (#4361): the emitted
// sizing numbers must be the SAME classed demands the live serve arm's fit check
// admits against, versioned, with concrete scopes, tier rollups, per-pool usable
// bytes, and would-be refusals downgraded to warnings instead of lost.
package main

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// sizingNamedBackend gives the serveCapBackend capacity stub the Name() the
// artifact's pool row reads (the embedded nil compute.Backend would panic).
type sizingNamedBackend struct{ serveCapBackend }

func (sizingNamedBackend) Name() string { return "testdev" }

func TestServeSizingArtifactCPUArm(t *testing.T) {
	origMetalAvailable := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = origMetalAvailable })
	serveMetalAvailable = func() bool { return false }
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

// fak-test:runtime fast est=1s
func TestServeSizingMetalQ8StartupMatchesNativeWeightsPlan(t *testing.T) {
	original := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = original })
	t.Setenv("FAK_Q4K", "0")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_TEST_STARTUP_PEAK_BYTES", "")
	t.Setenv("FAK_TEST_STEADY_BYTES", "")
	t.Setenv("FAK_ADMISSION_STATE_ENVELOPE", "")
	path := writeServePackedEmbeddingFixture(t, "startup-q8.gguf", false)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	serveMetalAvailable = func() bool { return true }
	art, err := buildServeSizingArtifact(ws, nil, false, 16, path, 0)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := json.Marshal(art)
	if err != nil {
		t.Fatal(err)
	}
	var decoded serveSizingArtifact
	if err := json.Unmarshal(wire, &decoded); err != nil {
		t.Fatal(err)
	}
	native, err := estimateMetalWeightsMemoryBounds(path)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Arm != "device-lean-q8" || decoded.Startup == nil {
		t.Fatalf("selected Q8 JSON has no startup projection: %s", wire)
	}
	if decoded.Startup.Scope != "metal-q8-weights-only" || decoded.Startup.SteadyBytes != native.SteadyBytes || decoded.Startup.PeakBytes != native.StartupPeakBytes || decoded.Startup.PeakBytes <= decoded.Startup.SteadyBytes {
		t.Fatalf("JSON startup=%+v native weights plan=%+v", decoded.Startup, native)
	}
	// A CPU Q8 selection retains exactly the same steady demands and totals,
	// but must not acquire a Metal startup estimate.
	serveMetalAvailable = func() bool { return false }
	cpu, err := buildServeSizingArtifact(ws, nil, false, 16, path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if cpu.Startup != nil || !reflect.DeepEqual(cpu.Tiers, art.Tiers) || !reflect.DeepEqual(cpu.Demands, art.Demands) {
		t.Fatalf("startup projection changed steady sizing: CPU=%+v Metal=%+v", cpu, art)
	}
	for _, stream := range []string{"FAK_STREAM_Q4K", "FAK_METAL_STREAM_Q4K"} {
		t.Run(stream, func(t *testing.T) {
			t.Setenv(stream, "1")
			serveMetalAvailable = func() bool { return true }
			streamed, err := buildServeSizingArtifact(ws, nil, false, 16, path, 0)
			if err != nil {
				t.Fatal(err)
			}
			if streamed.Startup != nil {
				t.Fatal("separate streamed reservation acquired nonstreamed Q8 estimate")
			}
		})
	}
	serveMetalAvailable = func() bool { return true }
	t.Setenv("FAK_Q4K", "1")
	resident, err := buildServeSizingArtifact(ws, nil, false, 16, path, 0)
	if err != nil {
		t.Fatal(err)
	}
	if resident.Arm != "device-resident-q4k" || resident.Startup != nil {
		t.Fatal("resident selection acquired a legacy Q8 startup estimate")
	}
}

// fak-test:runtime fast est=1s
func TestServeSizingMetalQ8StartupBudgetWarnings(t *testing.T) {
	const steady int64 = 1024
	peak, _ := metalServeStartupPeakBytes(serveLoadArmQuantProfileQ8, steady, 0, false)
	for _, tc := range []struct {
		name       string
		total      int64
		known      bool
		fit        serveFitBudget
		wantBudget bool
		wantTotal  bool
	}{
		{"fits-exactly", peak, true, serveFitBudget{Base: peak}, false, false},
		{"host-total-short", peak - 1, true, serveFitBudget{Base: peak}, false, true},
		{"usable-budget-short", peak, true, serveFitBudget{Base: peak - 1}, true, false},
		{"headroom-budget-short", peak, true, serveFitBudget{Base: peak, Headroom: 0.1}, true, false},
		{"unknown-host", 0, false, serveFitBudget{}, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			startup, warnings := serveSizingMetalQ8Startup(steady, tc.total, tc.known, tc.fit)
			if startup == nil || startup.PeakBytes != peak || startup.SteadyBytes != steady {
				t.Fatalf("startup=%+v warnings=%v", startup, warnings)
			}
			text := strings.Join(warnings, "\n")
			if strings.Contains(text, "host capacity") != tc.wantTotal || strings.Contains(text, "usable host budget") != tc.wantBudget {
				t.Fatalf("unexpected startup warnings: %v", warnings)
			}
		})
	}
	if startup, warnings := serveSizingMetalQ8Startup(0, 0, false, serveFitBudget{}); startup != nil || len(warnings) == 0 {
		t.Fatal("unavailable startup estimate must not appear as zero-cost success")
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
	host := art.Pools[1]
	var wantHostUsable int64
	if host.CapacityKnown && host.TotalBytes > 0 {
		wantHostUsable = compute.BudgetAfterHeadroom(host.FreeBytes, serveGGUFHostHeadroom)
	}
	if host.UsableBytes != wantHostUsable {
		t.Fatalf("host usable_bytes = %d, want host-reported budget %d (device usable=%d)", host.UsableBytes, wantHostUsable, art.Pools[0].UsableBytes)
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

// Software storage and JSON-path witness; Metal availability selects the route
// without executing GPU kernels or a full inference workload.
func TestServeSizingJSONPreservesPackedEmbeddingPath(t *testing.T) {
	original := serveMetalAvailable
	serveMetalAvailable = func() bool { return true }
	t.Cleanup(func() { serveMetalAvailable = original })
	t.Setenv("FAK_Q4K", "1")
	t.Setenv("FAK_STREAM_Q4K", "")
	t.Setenv("FAK_METAL_STREAM_Q4K", "")
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	for _, exact := range []bool{true, false} {
		name := "ordinary-name"
		if exact {
			name = "qualified-name-size"
		}
		t.Run(name, func(t *testing.T) {
			base := "ordinary-q2.gguf"
			if exact {
				base = "Qwen3.8-27B-UD-Q2_K_XL.gguf"
			}
			path := writeServePackedQ2EmbeddingFixture(t, base, exact)
			ws, err := ggufload.OpenWeights(path)
			if err != nil {
				t.Fatal(err)
			}
			defer ws.Close()
			opts := serveResidentQ4KLoadOptions(nil, path, true, ggufload.ClassifyTensorQuant(ws.File.Tensors))
			model, err := ggufload.LoadModelQ4KProfileOptions(path, nil, opts...)
			if err != nil {
				t.Fatalf("actual software load: %v", err)
			}
			defer model.CloseWeights()
			const projections int64 = 2*256*144 + 256*84 + 3*210
			want := projections + 3*256*4
			if exact {
				want = projections + 3*84
			}
			if got := model.ResidentReport().TotalResidentBytes; got != want {
				t.Fatalf("actual=%d independent=%d", got, want)
			}
			if (model.Q2KEmbedding != nil) != exact {
				t.Fatal("actual packed embedding route did not follow path qualification")
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			art, err := buildServeSizingArtifact(ws, nil, false, 16, path, info.Size())
			if err != nil {
				t.Fatal(err)
			}
			wire, err := json.Marshal(art)
			if err != nil {
				t.Fatal(err)
			}
			var decoded serveSizingArtifact
			if err := json.Unmarshal(wire, &decoded); err != nil {
				t.Fatal(err)
			}
			var weights int64
			for _, demand := range decoded.Demands {
				if demand.Class == string(compute.MemoryWeights) {
					weights += demand.Bytes
				}
			}
			t.Logf("actual selected storage=%d JSON weights=%d exact=%v", want, weights, exact)
			if decoded.Model != path || decoded.Arm != "device-resident-q4k" || weights != want {
				t.Fatalf("JSON dropped selected path storage: model=%q arm=%q weights=%d actual=%d", decoded.Model, decoded.Arm, weights, want)
			}
		})
	}
}
