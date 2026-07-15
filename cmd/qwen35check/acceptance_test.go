package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type acceptanceMarkerBackend struct {
	compute.Backend
	name, path string
}

func (b *acceptanceMarkerBackend) Name() string          { return b.name }
func (b *acceptanceMarkerBackend) Qwen35GDNPath() string { return b.path }
func (b *acceptanceMarkerBackend) Qwen35GDNDecode(normalizedInput, inProjQKV, inProjZ, inProjB, inProjA, conv1D, aLog, dtBias, norm, outProj, convState, recurrentState compute.Tensor, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int, rmsNormEpsilon float32) (compute.Tensor, compute.Tensor, compute.Tensor, error) {
	return compute.Tensor{}, convState, recurrentState, nil
}

func validAcceptanceConfig() model.Config {
	layers := make([]string, 64)
	for i := range layers {
		layers[i] = "linear_attention"
		if (i+1)%4 == 0 {
			layers[i] = "full_attention"
		}
	}
	return model.Config{ModelType: "qwen35", HiddenSize: 5120, NumLayers: 64, NumHeads: 24, NumKVHeads: 4, HeadDim: 256, IntermediateSize: 17408, VocabSize: 248320, FullAttentionInterval: 4, LinearConvKernelDim: 4, LinearNumKeyHeads: 16, LinearKeyHeadDim: 128, LinearNumValueHeads: 48, LinearValueHeadDim: 128, PartialRotaryFactor: .25, RopeTheta: 10000000, AttnOutputGate: true, NormGain1p: true, QKNorm: true, TieWordEmbeddings: false, RMSNormEps: 1e-6, LayerTypes: layers}
}

func sampleAcceptanceManifest(t *testing.T) acceptanceManifest {
	t.Helper()
	a := make([]float32, 248320)
	b := make([]float32, 248320)
	a[1], a[2] = 2, 3
	b[0], b[1], b[2] = -2, 1, .5
	m, err := buildAcceptanceManifest([]int{1, 2}, []int{3, 4}, [][]float32{a, b})
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func TestAcceptanceManifestDeterministicRoundTrip(t *testing.T) {
	m1 := sampleAcceptanceManifest(t)
	m2 := sampleAcceptanceManifest(t)
	b1, _ := json.MarshalIndent(m1, "", "  ")
	b2, _ := json.MarshalIndent(m2, "", "  ")
	if string(b1) != string(b2) {
		t.Fatal("manifest is not deterministic")
	}
	p := filepath.Join(t.TempDir(), "reference.json")
	if err := writeAcceptanceManifest(p, m1); err != nil {
		t.Fatal(err)
	}
	got, err := readAcceptanceManifest(p)
	if err != nil {
		t.Fatal(err)
	}
	if got.IntegritySHA256 != m1.IntegritySHA256 || len(got.Steps) != 2 {
		t.Fatalf("round trip mismatch: %#v", got)
	}
}

func TestAcceptanceManifestRefusesCorruptionSchemaAndIdentity(t *testing.T) {
	base := sampleAcceptanceManifest(t)
	tests := []struct {
		name   string
		mutate func(*acceptanceManifest)
	}{
		{"logits", func(m *acceptanceManifest) { m.Steps[0].LogitsF32[0]++ }},
		{"schema", func(m *acceptanceManifest) { m.Schema = "wrong" }},
		{"path", func(m *acceptanceManifest) { m.RequiredCUDAPath = "cpu/fallback" }},
		{"model", func(m *acceptanceManifest) { m.Model.SHA256 = "sha256:" + strings.Repeat("0", 64) }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := base
			m.Steps = append([]acceptanceReferenceStep(nil), base.Steps...)
			m.Steps[0].LogitsF32 = append([]float32(nil), base.Steps[0].LogitsF32...)
			tt.mutate(&m)
			b, _ := json.Marshal(m)
			if _, err := parseAcceptanceManifest(b); err == nil {
				t.Fatal("accepted malformed/corrupt manifest")
			}
		})
	}
}

func TestAcceptanceComparisonThresholdsAndFiniteChecks(t *testing.T) {
	ref := sampleAcceptanceManifest(t)
	good := [][]float32{append([]float32(nil), ref.Steps[0].LogitsF32...), append([]float32(nil), ref.Steps[1].LogitsF32...)}
	report, err := compareAcceptance(ref, good, []int64{1_000_000, 2_000_000})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Passed || report.Decode.TokensPerSec <= 0 {
		t.Fatalf("good comparison failed: %#v", report)
	}
	bad := [][]float32{append([]float32(nil), good[0]...), append([]float32(nil), good[1]...)}
	bad[1][2] = float32(math.NaN())
	if _, err = compareAcceptance(ref, bad, []int64{1, 1}); err == nil {
		t.Fatal("accepted non-finite logits")
	}
	argmax := [][]float32{append([]float32(nil), good[0]...), append([]float32(nil), good[1]...)}
	argmax[0][0] = 4
	if _, err = compareAcceptance(ref, argmax, []int64{1, 1}); err == nil {
		t.Fatal("accepted argmax/cosine mismatch")
	}
}

func TestAcceptanceCUDARequiredRefusesFallbackAndWrongPath(t *testing.T) {
	cfg := validAcceptanceConfig()
	cpu := compute.Default()
	if err := validateAcceptanceBackendConfig(cfg, "cuda", cpu); err == nil {
		t.Fatal("accepted CPU fallback")
	}
	wrong := &acceptanceMarkerBackend{Backend: cpu, name: "cuda", path: "cuda/wrong"}
	if err := validateAcceptanceBackendConfig(cfg, "cuda", wrong); err == nil {
		t.Fatal("accepted wrong CUDA path")
	}
	right := &acceptanceMarkerBackend{Backend: cpu, name: "cuda", path: model.Qwen35GDNCUDAPath}
	if err := validateAcceptanceBackendConfig(cfg, "cuda", right); err != nil {
		t.Fatalf("refused structural required path: %v", err)
	}
}

func TestAcceptanceExactModelFileFailsClosedBeforeLoad(t *testing.T) {
	p := filepath.Join(t.TempDir(), acceptanceCheckpointFile)
	if err := os.WriteFile(p, []byte("not the exact checkpoint"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := verifyAcceptanceModelFile(p, expectedAcceptanceModel()); err == nil || !strings.Contains(err.Error(), "size") {
		t.Fatalf("wanted pre-load size refusal, got %v", err)
	}
}
