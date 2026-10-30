package main

import (
	"encoding/json"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func serveFitPlanTestEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{"FAK_Q4K", "FAK_STREAM_Q4K", "FAK_METAL_STREAM_Q4K", "FAK_UP_KV_PRECISION", "FAK_HYBRID_KV"} {
		t.Setenv(k, "")
	}
	orig := serveMetalAvailable
	t.Cleanup(func() { serveMetalAvailable = orig })
	serveMetalAvailable = func() bool { return false }
}

func serveFitPlanUnifiedBackend() compute.Backend {
	return sizingNamedBackend{serveCapBackend{
		total: 96 << 30, free: 96 << 30, known: true, uploadDtype: true,
		tier:      "integrated:AMD Radeon 8060S Graphics (gfx1151)",
		hostTotal: 128 << 30, hostFree: 64 << 30, hostKnown: true,
	}}
}

func serveFitPlanCurrent(t *testing.T, rep *fitplan.Report) fitplan.Prediction {
	t.Helper()
	var cur []fitplan.Prediction
	for _, pr := range rep.Candidates {
		if pr.Current {
			cur = append(cur, pr)
		}
	}
	if len(cur) != 1 {
		t.Fatalf("%d current candidates, want exactly 1", len(cur))
	}
	return cur[0]
}

func serveFitPlanOmits(list []fitplan.Omission, o fitplan.Omission) bool {
	for _, x := range list {
		if x == o {
			return true
		}
	}
	return false
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanCurrentIsTheResolvedBoot(t *testing.T) {
	serveFitPlanTestEnv(t)
	art, err := buildServeSizingArtifact(serveSynthConfiguredWeightSource(t), nil, false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil {
		t.Fatal("fit_plan missing from the artifact")
	}
	cur := serveFitPlanCurrent(t, rep)
	if cur.ContextTokens != art.ResolvedNativeContextTokens || rep.Pins.ContextTokens != 16 {
		t.Fatalf("current ctx %d pin %d, resolved %d", cur.ContextTokens, rep.Pins.ContextTokens, art.ResolvedNativeContextTokens)
	}
	host := art.Pools[len(art.Pools)-1]
	if rep.Evidence.Topology != fitplan.TopologyHostOnly || rep.Evidence.PoolUsableBytes != host.UsableBytes {
		t.Fatalf("evidence %+v, host pool %+v", rep.Evidence, host)
	}
	if !serveFitPlanOmits(cur.Omitted, fitplan.OmitPeakBandwidthUnknown) {
		t.Fatalf("current omitted %v, want peak_bandwidth_unknown", cur.Omitted)
	}
	if (rep.Status == fitplan.StatusUnknown) != !host.CapacityKnown {
		t.Fatalf("status %s with host capacity_known=%v", rep.Status, host.CapacityKnown)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanUnifiedPoolIsHostWindow(t *testing.T) {
	serveFitPlanTestEnv(t)
	art, err := buildServeSizingArtifact(serveSynthConfiguredWeightSource(t), serveFitPlanUnifiedBackend(), false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil {
		t.Fatal("fit_plan missing from the artifact")
	}
	if rep.Evidence.Topology != fitplan.TopologyUnified || rep.Evidence.PoolUsableBytes != compute.BudgetAfterHeadroom(64<<30, serveGGUFHostHeadroom) {
		t.Fatalf("evidence = %+v", rep.Evidence)
	}
	if want, _ := fitplan.DevicePeaks("gfx1151"); rep.Evidence.Peaks != want {
		t.Fatalf("peaks = %+v, want %+v", rep.Evidence.Peaks, want)
	}
	c := rep.Chosen
	if rep.Status != fitplan.StatusFits || c == nil || !c.Current || !(c.DecodeTokSPerSeq > 0) || c.Bound != fitplan.BoundMemory {
		t.Fatalf("status %s chosen %+v", rep.Status, c)
	}
	if r := c.DecodeTokSPerSeq / c.DecodeTokSCeiling; math.Abs(r-0.55) > 1e-9*0.55 {
		t.Fatalf("per-seq/ceiling = %v, want 0.55", r)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanDeviceOversizeIsTiered(t *testing.T) {
	serveFitPlanTestEnv(t)
	be := sizingNamedBackend{serveCapBackend{total: 1 << 19, free: 1 << 19, known: true, uploadDtype: true}}
	art, err := buildServeSizingArtifact(serveSynthConfiguredWeightSource(t), be, false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil || rep.Status != fitplan.StatusTiered || rep.Refusal != nil {
		t.Fatalf("fit plan = %+v, want a tiered placement and no refusal", rep)
	}
	if pl := rep.Placement; pl == nil || len(pl.Rows) == 0 || pl.Rows[0].Segment != fitplan.SegmentDenseCore || pl.Rows[0].Pool != fitplan.PlacementPoolDevice {
		t.Fatalf("placement %+v, want the dense core first on device", rep.Placement)
	}
}

// serveFitPlanLongContextSource is serveSynthConfiguredWeightSource with a 1M-token window, so the
// auto-sizer resolves a context strictly inside it.
func serveFitPlanLongContextSource(t *testing.T) *ggufload.WeightSource {
	t.Helper()
	ws := serveSynthConfiguredWeightSource(t)
	ws.File.Metadata["qwen2.context_length"] = ggufload.Value{Type: ggufload.TypeUint64, Value: uint64(1 << 20)}
	return ws
}

func serveFitPlanSmallUMABackend() sizingNamedBackend {
	return sizingNamedBackend{serveCapBackend{
		total: 96 << 30, free: 96 << 30, known: true, uploadDtype: true,
		tier:      "integrated:AMD Radeon 8060S Graphics (gfx1151)",
		hostTotal: 128 << 30, hostFree: 64 << 20, hostKnown: true,
	}}
}

// serveDriftHostBackend reports 1 MiB less free host memory on every probe, as MemAvailable does.
type serveDriftHostBackend struct {
	sizingNamedBackend
	probes *int64
}

func (b serveDriftHostBackend) HostMemory() (int64, int64, bool) {
	*b.probes++
	return b.hostTotal, b.hostFree - *b.probes<<20, b.hostKnown
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanUMAPricesTheSizedBudget(t *testing.T) {
	serveFitPlanTestEnv(t)
	be := serveDriftHostBackend{serveFitPlanSmallUMABackend(), new(int64)}
	art, err := buildServeSizingArtifact(serveFitPlanLongContextSource(t), be, false, 0, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil || art.NativeContextSource != "auto" || art.ResolvedNativeContextTokens >= 1<<20 {
		t.Fatalf("precondition: source %s resolved %d fit plan %+v", art.NativeContextSource, art.ResolvedNativeContextTokens, rep)
	}
	cur := serveFitPlanCurrent(t, rep)
	if cur.ContextTokens != art.ResolvedNativeContextTokens || cur.Fit != compute.FitOK.String() {
		t.Fatalf("current %+v, want the resolved %d context fitting", cur, art.ResolvedNativeContextTokens)
	}
	if rep.Status != fitplan.StatusFits || rep.Chosen == nil || !rep.Chosen.Current || rep.MaxContextTokens < art.ResolvedNativeContextTokens {
		t.Fatalf("status %s chosen %+v max ctx %d, resolved %d", rep.Status, rep.Chosen, rep.MaxContextTokens, art.ResolvedNativeContextTokens)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanPinsTheRequestNotTheClamp(t *testing.T) {
	serveFitPlanTestEnv(t)
	const requested = 1 << 20
	art, err := buildServeSizingArtifact(serveFitPlanLongContextSource(t), serveFitPlanSmallUMABackend(), false, requested, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil || art.NativeContextSource != "explicit" || art.ResolvedNativeContextTokens >= requested {
		t.Fatalf("precondition: source %s resolved %d fit plan %+v", art.NativeContextSource, art.ResolvedNativeContextTokens, rep)
	}
	if rep.Pins.ContextTokens != requested || rep.Status != fitplan.StatusRefused || !errors.Is(rep.Refusal, fitplan.ErrContextExceedsPool) {
		t.Fatalf("pins %+v status %s refusal %+v, want the %d request refused", rep.Pins, rep.Status, rep.Refusal, requested)
	}
	if rep.Refusal.MaxContextTokens < art.ResolvedNativeContextTokens {
		t.Fatalf("refusal max ctx %d below the clamped boot context %d", rep.Refusal.MaxContextTokens, art.ResolvedNativeContextTokens)
	}
	if cur := serveFitPlanCurrent(t, rep); cur.ContextTokens != art.ResolvedNativeContextTokens || cur.Fit != compute.FitOK.String() {
		t.Fatalf("current %+v, want the clamped boot context fitting", cur)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanDiscreteHostExpertsUseHostPool(t *testing.T) {
	serveFitPlanTestEnv(t)
	ws := serveSynthOffloadWeightSource(t)
	for i := range ws.File.Tensors {
		if ws.File.Tensors[i].Name == "blk.0.ffn_gate_exps.weight" {
			ws.File.Tensors[i].Dims = []uint64{64, 1 << 20, 1 << 20} // 256 TiB of f32 routed experts
		}
	}
	be := sizingNamedBackend{serveCapBackend{total: 24 << 30, free: 24 << 30, known: true, uploadDtype: true}}
	art, err := buildServeSizingArtifact(ws, be, true, 0, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil || rep.Evidence.Topology != fitplan.TopologyDiscrete {
		t.Fatalf("fit plan = %+v", rep)
	}
	host := art.Pools[len(art.Pools)-1]
	if rep.Evidence.HostPoolUsableBytes != host.UsableBytes {
		t.Fatalf("host pool %d, artifact reports %d", rep.Evidence.HostPoolUsableBytes, host.UsableBytes)
	}
	if !host.CapacityKnown {
		if rep.Status != fitplan.StatusUnknown || !serveFitPlanOmits(rep.Omitted, fitplan.OmitHostPoolCapacityUnknown) {
			t.Fatalf("status %s omitted %v, want unknown host pool", rep.Status, rep.Omitted)
		}
		return
	}
	if rep.Status != fitplan.StatusTiered || rep.Refusal != nil || rep.Placement == nil {
		t.Fatalf("status %s refusal %+v, want a tiered placement for the host overflow", rep.Status, rep.Refusal)
	}
	var nvme bool
	for _, r := range rep.Placement.Rows {
		nvme = nvme || r.Pool == fitplan.PlacementPoolNVMe
	}
	if !nvme {
		t.Fatalf("rows %+v, want the overflow spilled to nvme_mmap", rep.Placement.Rows)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanHonorsKVPin(t *testing.T) {
	serveFitPlanTestEnv(t)
	t.Setenv("FAK_UP_KV_PRECISION", "q8")
	art, err := buildServeSizingArtifact(serveSynthConfiguredWeightSource(t), serveFitPlanUnifiedBackend(), false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	rep := art.FitPlan
	if rep == nil || rep.Pins.KVPrecision != "q8" || rep.Chosen == nil {
		t.Fatalf("fit plan = %+v", rep)
	}
	if cur := serveFitPlanCurrent(t, rep); cur.KVPrecision != "q8" || rep.Chosen.KVPrecision != "q8" {
		t.Fatalf("current %s chosen %s, want q8", cur.KVPrecision, rep.Chosen.KVPrecision)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestServeSizingFitPlanJSONIsAdditive(t *testing.T) {
	serveFitPlanTestEnv(t)
	art, err := buildServeSizingArtifact(serveSynthConfiguredWeightSource(t), nil, false, 16, "synthetic.gguf", 0)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(art)
	if err != nil {
		t.Fatal(err)
	}
	var keys map[string]json.RawMessage
	if err := json.Unmarshal(raw, &keys); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"version", "model", "arm", "native_context_tokens", "context_budget_tokens",
		"model_declared_context_tokens", "resolved_native_context_tokens", "native_context_source",
		"demands", "tiers", "pools", "warnings", "fit_plan"} {
		if _, ok := keys[k]; !ok {
			t.Fatalf("artifact key %q missing", k)
		}
	}
	var version string
	if err := json.Unmarshal(keys["version"], &version); err != nil || version != serveSizingVersion {
		t.Fatalf("version = %q (%v), want %q", version, err, serveSizingVersion)
	}
	var decoded serveSizingArtifact
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(decoded.FitPlan, art.FitPlan) {
		t.Fatalf("fit_plan does not round-trip:\n%+v\n%+v", decoded.FitPlan, art.FitPlan)
	}
}
