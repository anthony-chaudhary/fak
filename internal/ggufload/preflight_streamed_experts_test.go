package ggufload

import (
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func streamedPreflightFields(in *PreflightInput, enabled bool, ring, dense int64) {
	v := reflect.ValueOf(in).Elem()
	for name, value := range map[string]any{"StreamedExperts": enabled, "StreamedExpertDeviceRingBytes": ring, "StreamedDenseWorkingSetBytes": dense} {
		field := v.FieldByName(name)
		if field.IsValid() {
			field.Set(reflect.ValueOf(value))
		}
	}
}

func streamedPreflightFixture(t *testing.T) (*WeightSource, *countingReaderAt) {
	t.Helper()
	blob := glmMoeDsaFullGGUFWithTypes(256, 8, 32, 32, 16, 16, 16, 2, 2, 16, 3, 256, 256, func(name string) TensorType {
		switch {
		case name == "output.weight", strings.HasSuffix(name, "ffn_gate_exps.weight"), strings.HasSuffix(name, "ffn_up_exps.weight"):
			return TensorQ2_K
		case strings.HasSuffix(name, "ffn_down_exps.weight"), strings.HasSuffix(name, "ffn_gate_shexp.weight"):
			return TensorQ3_K
		case strings.HasSuffix(name, "ffn_up_shexp.weight"):
			return TensorQ4_K
		default:
			return TensorF32
		}
	})
	path := filepath.Join(t.TempDir(), "mixed-streamed.gguf")
	if err := os.WriteFile(path, blob, 0o600); err != nil {
		t.Fatal(err)
	}
	f, header, size, err := openAndRead(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = f.Close() })
	r := &countingReaderAt{r: f}
	ws, err := NewWeightSource(header, r, size)
	if err != nil {
		t.Fatal(err)
	}
	// Exercise the real per-tensor split reader routing with complete backing ranges.
	ws.readerFor, ws.sizeFor = make([]io.ReaderAt, len(header.Tensors)), make([]int64, len(header.Tensors))
	for i := range ws.readerFor {
		ws.readerFor[i], ws.sizeFor[i] = r, size
	}
	return ws, r
}

func streamedPreflightInput(ws *WeightSource, ring, dense int64) PreflightInput {
	in := PreflightInput{Source: ws, VulkanMixedQ4K: true, Backend: dualCapacityBackend{
		capBackend: capBackend{total: 64 << 20, free: 64 << 20, known: true},
		hostTotal:  64 << 20, hostFree: 64 << 20, hostKnown: true, name: "vulkan", tier: "discrete:test",
	}}
	streamedPreflightFields(&in, true, ring, dense)
	return in
}

// fak-test:runtime fast est=500ms lane=default
func TestPreflightStreamedExpertsSplitLoaderContract(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "1")
	ws, reader := streamedPreflightFixture(t)
	const ring, dense = int64(1 << 20), int64(1 << 20)
	in := streamedPreflightInput(ws, ring, dense)
	pf := BuildModelPreflight(in)
	if pf.Verdict != PreflightReady || pf.FitState != FitOK {
		t.Fatalf("mixed split streamed source verdict=%s fit=%s: %s", pf.Verdict, pf.FitState, pf.Reason)
	}
	if reader.reads.Load() != 0 {
		t.Fatal("preflight performed payload IO")
	}
	legacy := in
	streamedPreflightFields(&legacy, false, 0, 0)
	if old := BuildModelPreflight(legacy); old.Verdict != PreflightRefuseHeader {
		t.Fatalf("explicit streamed support changed eager split behavior: %s", old.Verdict)
	}
	opts := []Q4KLoadOption{WithDenseKQuantResident(false), WithDenseQ2KResident(true), WithStreamedExperts(0), WithStreamedExpertDeviceRing(ring), WithStreamedDenseQ4KWorkingSet(dense)}
	peak, err := ws.EstimateStreamedExpertHostLoadPeak(opts...)
	if err != nil {
		t.Fatal(err)
	}
	if pf.EstHostResidentBytes < peak.SteadyAnonBytes || pf.EstHostResidentBytes+pf.EstLoadStagingBytes < peak.PeakAnonBytes {
		t.Fatalf("preflight undercharges existing host peak: resident=%d stage=%d steady=%d peak=%d", pf.EstHostResidentBytes, pf.EstLoadStagingBytes, peak.SteadyAnonBytes, peak.PeakAnonBytes)
	}
	m, err := ws.QuantModelQ4KProfileOptions(nil, opts...)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if !m.HasKQuant("lm_head.weight") || !m.HasQ8("model.layers.0.mlp.shared_experts.gate_proj.weight") || !m.HasQ4K("model.layers.0.mlp.shared_experts.up_proj.weight") {
		t.Fatal("mixed loader did not preserve packed Q2/Q4 and convert dense Q3 to Q8")
	}
	stats := m.ExpertCheckpointStats()
	if !stats.Enabled || stats.BudgetBytes != 0 || stats.ResidentBytes != 0 || stats.ResidentCount != 0 || stats.Tensors != 9 {
		t.Fatalf("streamed tier does not retain zero host experts: %+v", stats)
	}
	actual := m.ResidentReport()
	if actual.F32Bytes == 0 || actual.Q8Bytes == 0 || pf.EstHostResidentBytes < actual.Q8Bytes+actual.F32Bytes {
		t.Fatalf("host charge %d misses logical Q8/F32 storage %d/%d", pf.EstHostResidentBytes, actual.Q8Bytes, actual.F32Bytes)
	}
	var logicalQ8 int64
	for _, row := range peak.Rows {
		logicalQ8 += row.Q8Bytes
	}
	if actual.Q8Bytes != logicalQ8 {
		t.Fatalf("loader logical Q8 bytes=%d header logical=%d", actual.Q8Bytes, logicalQ8)
	}
	// Header allocation contract: page-round each owner and reserve one alignment page.
	// ResidentReport supplies logical lengths, not a measurement of allocated backing bytes.
	page := int64(os.Getpagesize())
	allocation := func(n int64) int64 { return ((n+page-1)/page + 1) * page }
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	var q8Owners, ownerLogical int64
	owners := 0
	for _, info := range ws.File.Tensors {
		canon, mapped := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !mapped || !m.HasQ8(canon) {
			continue
		}
		if len(info.Dims) != 2 || info.Dims[0]%32 != 0 {
			t.Fatalf("fixture Q8 owner %s lacks ordinary block-aligned matrix dimensions", canon)
		}
		codes := int64(info.Dims[0] * info.Dims[1])
		scales := codes / 32 * 4
		ownerLogical += codes + scales
		q8Owners += allocation(codes) + allocation(scales)
		owners++
	}
	if owners != actual.Q8Tensors || ownerLogical != logicalQ8 {
		t.Fatalf("header owner coverage %d/%d logical=%d/%d", owners, actual.Q8Tensors, ownerLogical, logicalQ8)
	}
	if want := peak.SteadyAnonBytes + q8Owners - logicalQ8; pf.EstHostResidentBytes != want {
		t.Fatalf("header host allocation contract=%d want %d", pf.EstHostResidentBytes, want)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPreflightStreamedExpertsRefusesUnsizedFusedQKV(t *testing.T) {
	ws, reader := streamedPreflightFixture(t)
	ws.File.Tensors = append(ws.File.Tensors, TensorInfo{
		Name: "blk.0.attn_qkv.weight", Dims: []uint64{256, 768}, Type: TensorQ3_K,
	})
	pf := BuildModelPreflight(streamedPreflightInput(ws, 1<<20, 1<<20))
	if pf.Verdict != PreflightRefuseHeader {
		t.Fatalf("unqualified fused child-allocation layout admitted: verdict=%s", pf.Verdict)
	}
	if reader.reads.Load() != 0 {
		t.Fatal("fused allocation refusal read tensor payloads")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPreflightStreamedExpertsRingUsesTierFloor(t *testing.T) {
	ws, reader := streamedPreflightFixture(t)
	shards, err := ws.FusedExpertTensors()
	if err != nil {
		t.Fatal(err)
	}
	tier, err := buildExpertCheckpointTier(shards, 0)
	if err != nil {
		t.Fatal(err)
	}
	var floor *model.ExpertCheckpointDeviceBudgetError
	if err := tier.SetDeviceRingBudget(1); !errors.As(err, &floor) || floor.Bytes <= 1 {
		t.Fatalf("fixture failed to expose actual complete-expert floor: %v", err)
	}
	for _, ring := range []int64{0, -1, floor.Bytes - 1, floor.Bytes, math.MaxInt64} {
		pf := BuildModelPreflight(streamedPreflightInput(ws, ring, 1<<20))
		if ring == floor.Bytes {
			if pf.Verdict != PreflightReady {
				t.Fatalf("exact runtime expert floor refused: %s", pf.Reason)
			}
		} else if !pf.Refused() {
			t.Fatalf("invalid ring %d admitted", ring)
		}
	}
	for _, dense := range []int64{0, -1} {
		if pf := BuildModelPreflight(streamedPreflightInput(ws, floor.Bytes, dense)); !pf.Refused() {
			t.Fatalf("unbounded dense working set %d admitted", dense)
		}
	}
	if reader.reads.Load() != 0 {
		t.Fatal("floor admission performed payload IO")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPreflightStreamedExpertsIndependentAndUnifiedCapacity(t *testing.T) {
	ws, _ := streamedPreflightFixture(t)
	in := streamedPreflightInput(ws, 1<<20, 1<<20)
	p := BuildModelPreflight(in)
	if p.Verdict != PreflightReady {
		t.Fatalf("capacity probe refused: %s", p.Reason)
	}
	host := p.EstHostResidentBytes + p.EstLoadStagingBytes
	device := p.EstDeviceResidentBytes
	if host <= 0 || device <= 0 || p.EstLoadBytes < host+device {
		t.Fatal("preflight omitted separate capacity demands")
	}
	be := in.Backend.(dualCapacityBackend)
	for _, scope := range []compute.MemoryScope{compute.MemoryScopeDevice, compute.MemoryScopeHost} {
		limited := be
		if scope == compute.MemoryScopeDevice {
			limited.free = device - 1
		} else {
			limited.hostFree = host - 1
		}
		in.Backend = limited
		pf := BuildModelPreflight(in)
		if pf.Verdict != PreflightRefuseTooBig || pf.FitScope != string(scope) {
			t.Fatalf("limited %s verdict=%s scope=%s", scope, pf.Verdict, pf.FitScope)
		}
	}
	shared := max(host, device) + min(host, device)/2
	be.free, be.hostFree, be.total, be.hostTotal = shared, shared, shared, shared
	in.Backend = be
	if pf := BuildModelPreflight(in); pf.Verdict != PreflightReady {
		t.Fatalf("separate pools large enough individually refused: %s", pf.Reason)
	}
	be.tier = "integrated:test"
	in.Backend = be
	if pf := BuildModelPreflight(in); pf.Verdict != PreflightRefuseTooBig || pf.FitScope != string(compute.MemoryScopeHost) {
		t.Fatalf("combined UMA demand not refused: verdict=%s scope=%s", pf.Verdict, pf.FitScope)
	}
}

// fak-test:runtime fast est=500ms lane=default
func TestPreflightStreamedExpertsPackedQ2EmbeddingIsHostOnly(t *testing.T) {
	ws, err := OpenWeights(writeDeepSeek41ExpertHALFile(t))
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	ws.File.Tensors = append(ws.File.Tensors, TensorInfo{Name: "blk.0.attn_q_a_norm.weight", Dims: []uint64{uint64(cfg.QLoraRank)}, Type: TensorF32})
	raw, _, err := ws.TensorBytes("token_embd.weight")
	if err != nil {
		t.Fatal(err)
	}
	f32Bytes := int64(len(raw))
	for i := range ws.File.Tensors {
		if ws.File.Tensors[i].Name == "token_embd.weight" {
			ws.File.Tensors[i].Type = TensorQ2_K
		}
	}
	in := streamedPreflightInput(ws, 1<<20, 1<<20)
	base := BuildModelPreflight(in)
	in.ResidentQ2KEmbedding = true
	packed := BuildModelPreflight(in)
	if base.Verdict != PreflightReady || packed.Verdict != PreflightReady {
		t.Fatalf("embedding source refused: base=%s packed=%s", base.Reason, packed.Reason)
	}
	page := int64(os.Getpagesize())
	f32Allocation := ((f32Bytes+page-1)/page + 1) * page
	if reduction := base.EstDeviceResidentBytes - packed.EstDeviceResidentBytes; reduction != f32Allocation {
		t.Fatalf("host-only packed embedding removed %d device bytes, want whole F32 allocation contract %d", reduction, f32Allocation)
	}
	if packed.EstHostResidentBytes <= 0 || packed.EstHostResidentBytes >= base.EstHostResidentBytes {
		t.Fatalf("packed embedding host charge=%d baseline=%d", packed.EstHostResidentBytes, base.EstHostResidentBytes)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestPreflightStreamedExpertsRejectsInvalidRegimes(t *testing.T) {
	ws, reader := streamedPreflightFixture(t)
	for _, tc := range []struct {
		name   string
		mutate func(*PreflightInput)
	}{
		{"cpu", func(in *PreflightInput) { be := in.Backend.(dualCapacityBackend); be.name = "cpu-ref"; in.Backend = be }},
		{"missing_mixed", func(in *PreflightInput) { in.VulkanMixedQ4K = false }},
		{"lean", func(in *PreflightInput) { in.Lean = true }},
		{"offload", func(in *PreflightInput) { in.OffloadExperts = true }},
		{"single_residency", func(in *PreflightInput) { in.SingleResidencyQ4K = true }},
		{"inactive_bounds", func(in *PreflightInput) { streamedPreflightFields(in, false, 1<<20, 1<<20) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			in := streamedPreflightInput(ws, 1<<20, 1<<20)
			tc.mutate(&in)
			if pf := BuildModelPreflight(in); !pf.Refused() {
				t.Fatalf("invalid %s regime admitted: %s", tc.name, pf.Verdict)
			}
		})
	}
	if reader.reads.Load() != 0 {
		t.Fatal("invalid regime checks read tensor payloads")
	}
}
