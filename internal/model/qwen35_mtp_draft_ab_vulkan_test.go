//go:build vulkan && (windows || linux) && cgo

package model_test

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

const qwen35MTPDraftABSchema = "fak/qwen35-mtp-draft-ab/v1"

const qwen35MTPDraftABTestCommand = "go test -buildvcs=true -count=1 -v -tags vulkan -run ^TestQwen35MTPDraftRealArtifactAB$ ./internal/model"

type qwen35MTPDraftABManifest struct {
	Schema         string                         `json:"schema"`
	Source         string                         `json:"source"`
	Revision       string                         `json:"revision"`
	CodeRevision   string                         `json:"code_revision"`
	GGUF           string                         `json:"gguf"`
	GGUFSHA256     string                         `json:"gguf_sha256"`
	Inputs         string                         `json:"inputs"`
	InputsSHA256   string                         `json:"inputs_sha256"`
	SystemState    string                         `json:"system_state"`
	SystemSHA256   string                         `json:"system_state_sha256"`
	ProducedInputs string                         `json:"produced_inputs"`
	Samples        int                            `json:"samples"`
	MaxP50NS       int64                          `json:"max_device_p50_ns"`
	MaxP90NS       int64                          `json:"max_device_p90_ns"`
	ExpectedSystem compute.BackendRuntimeIdentity `json:"expected_system"`
}

type qwen35MTPDraftABInputs struct {
	Schema          string `json:"schema"`
	Source          string `json:"source"`
	Revision        string `json:"revision"`
	GGUFSHA256      string `json:"gguf_sha256"`
	Producer        string `json:"producer"`
	ProducerCommand string `json:"producer_command"`
	Tokens          []int  `json:"tokens"`
}

type qwen35MTPDraftABSystemState struct {
	Schema       string             `json:"schema"`
	Timestamp    string             `json:"timestamp"`
	CodeRevision string             `json:"code_revision"`
	GGUFSHA256   string             `json:"gguf_sha256"`
	Governor     string             `json:"governor"`
	ClocksMHz    map[string]float64 `json:"clocks_mhz"`
	ThermalsC    map[string]float64 `json:"thermals_c"`
}

type qwen35MTPDraftABInputRow struct {
	Position    int       `json:"position"`
	Token       int       `json:"token"`
	PriorHidden []float32 `json:"prior_hidden"`
}

type qwen35MTPDraftABSample struct {
	Sample            int                                 `json:"sample"`
	Position          int                                 `json:"position"`
	Order             string                              `json:"order"`
	CPUNS             int64                               `json:"cpu_ns"`
	DeviceNS          int64                               `json:"device_ns"`
	CPUArgmax         int                                 `json:"cpu_argmax"`
	DeviceArgmax      int                                 `json:"device_argmax"`
	Cosine            float64                             `json:"cosine"`
	MaxRelativeDelta  float64                             `json:"max_relative_delta"`
	DeviceObservation compute.BackendExecutionObservation `json:"device_observation"`
}

type qwen35MTPDraftABCI struct {
	LowNS  int64 `json:"low_ns"`
	HighNS int64 `json:"high_ns"`
}

type qwen35MTPDraftABSummary struct {
	Schema                   string                         `json:"schema"`
	Source                   string                         `json:"source"`
	Revision                 string                         `json:"revision"`
	CodeRevision             string                         `json:"code_revision"`
	GGUFSHA256               string                         `json:"gguf_sha256"`
	InputsSHA256             string                         `json:"inputs_sha256"`
	ProducedInputsSHA256     string                         `json:"produced_inputs_sha256"`
	ProducerExecutableSHA256 string                         `json:"producer_executable_sha256"`
	SystemStateSHA256        string                         `json:"system_state_sha256"`
	RecordedAt               string                         `json:"recorded_at"`
	SystemStateAt            string                         `json:"system_state_at"`
	System                   compute.BackendRuntimeIdentity `json:"system"`
	Samples                  []qwen35MTPDraftABSample       `json:"samples"`
	CPUP50NS                 int64                          `json:"cpu_p50_ns"`
	CPUP90NS                 int64                          `json:"cpu_p90_ns"`
	CPUP50CI95               qwen35MTPDraftABCI             `json:"cpu_p50_ci95_ns"`
	CPUP90CI95               qwen35MTPDraftABCI             `json:"cpu_p90_ci95_ns"`
	DeviceP50NS              int64                          `json:"device_p50_ns"`
	DeviceP90NS              int64                          `json:"device_p90_ns"`
	DeviceP50CI95            qwen35MTPDraftABCI             `json:"device_p50_ci95_ns"`
	DeviceP90CI95            qwen35MTPDraftABCI             `json:"device_p90_ci95_ns"`
	P50ImprovementNS         int64                          `json:"p50_improvement_ns"`
	P90ImprovementNS         int64                          `json:"p90_improvement_ns"`
	P50ImprovementPercent    float64                        `json:"p50_improvement_percent"`
	P90ImprovementPercent    float64                        `json:"p90_improvement_percent"`
	ForwardReceipt           model.Qwen35MTPForwardReceipt  `json:"forward_receipt"`
	Qualified                bool                           `json:"qualified"`
	Qualification            string                         `json:"qualification"`
}

type qwen35MTPDraftABProducedInputs struct {
	Schema                   string                     `json:"schema"`
	Timestamp                string                     `json:"timestamp"`
	CodeRevision             string                     `json:"code_revision"`
	GGUFSHA256               string                     `json:"gguf_sha256"`
	TokenInputsSHA256        string                     `json:"token_inputs_sha256"`
	ProducedInputsSHA256     string                     `json:"produced_inputs_sha256"`
	ProducerExecutableSHA256 string                     `json:"producer_executable_sha256"`
	Rows                     []qwen35MTPDraftABInputRow `json:"rows"`
}

// TestQwen35MTPDraftRealArtifactAB is the opt-in physical draft-head witness.
// It compares the retained checkpoint's CPU and Vulkan MTP primitive using fixed,
// hashed target-hidden rows and checkpoint-derived token embeddings. It does not
// claim full speculative-decode acceptance because the target device API does not
// currently export its raw pre-final-normalization hidden state.
func TestQwen35MTPDraftRealArtifactAB(t *testing.T) {
	manifestPath := os.Getenv("FAK_MTP_DRAFT_AB_MANIFEST")
	if manifestPath == "" {
		t.Skip("set FAK_MTP_DRAFT_AB_MANIFEST to a reviewed real-artifact manifest")
	}
	candidate := readQwen35MTPDraftABManifest(t, manifestPath)
	revision, modified := buildProvenance()
	if modified || revision == "unknown" || revision != candidate.CodeRevision {
		t.Fatalf("source provenance revision=%q modified=%t, want clean %q", revision, modified, candidate.CodeRevision)
	}
	assertFileSHA256(t, candidate.GGUF, candidate.GGUFSHA256)
	assertFileSHA256(t, candidate.Inputs, candidate.InputsSHA256)
	assertFileSHA256(t, candidate.SystemState, candidate.SystemSHA256)
	inputs := readQwen35MTPDraftABInputs(t, candidate, manifestPath)
	systemState := readQwen35MTPDraftABSystemState(t, candidate)

	backend, ok := compute.Lookup("vulkan")
	if !ok {
		t.Fatal("Vulkan backend is required for physical MTP draft qualification")
	}
	q6, ok := backend.(interface{ SupportsQ6KMatMul() bool })
	if !ok || !q6.SupportsQ6KMatMul() {
		t.Fatal("Vulkan backend lacks the native packed Q6_K projection required by the retained artifact")
	}

	originalRetain := model.RetainMTP
	model.RetainMTP = true
	defer func() { model.RetainMTP = originalRetain }()
	m, err := ggufload.LoadModelQ4KProfileOptions(candidate.GGUF, nil)
	if err != nil {
		t.Fatalf("load retained Qwen3.8 GGUF: %v", err)
	}
	defer func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("close retained model weights: %v", err)
		}
	}()
	if !m.Cfg.IsQwen35Hybrid() || !m.Cfg.HasMTPHead() {
		t.Fatalf("artifact is not a Qwen3.8 hybrid checkpoint with MTP: %+v", m.Cfg)
	}
	assertQwen35MTPDraftABQ6(t, m)
	validateQwen35MTPDraftABTokens(t, inputs.Tokens, candidate.Samples, m.Cfg)
	priorHidden, err := model.Qwen35MTPTestPriorHiddenRows(m, inputs.Tokens, filepath.Join(t.TempDir(), "native-target-hidden"))
	if err != nil {
		t.Fatalf("produce native target-hidden rows: %v", err)
	}
	rows := make([]qwen35MTPDraftABInputRow, len(inputs.Tokens))
	for pos, token := range inputs.Tokens {
		rows[pos] = qwen35MTPDraftABInputRow{Position: pos, Token: token, PriorHidden: priorHidden[pos]}
	}
	validateQwen35MTPDraftABRows(t, rows, candidate.Samples, m.Cfg)
	producedInputsSHA256 := qwen35MTPDraftABHashRows(rows)
	producerExecutableSHA256 := qwen35MTPDraftABExecutableSHA256(t)
	writeQwen35MTPDraftABProducedInputs(t, candidate, rows, producedInputsSHA256, producerExecutableSHA256)

	embeddings := m.NewSession()
	defer embeddings.Close()
	cpuForward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("construct retained CPU MTP oracle: %v", err)
	}
	defer cpuForward.Close()
	deviceForward, err := m.NewQwen35MTPForwardWithBackend(backend)
	if err != nil {
		t.Fatalf("construct retained Vulkan MTP draft: %v", err)
	}
	defer deviceForward.Close()

	run := func(row qwen35MTPDraftABInputRow, sample int, cpuFirst bool) qwen35MTPDraftABSample {
		embedding, err := embeddings.TokenEmbedding(row.Token)
		if err != nil {
			t.Fatalf("embedding token %d: %v", row.Token, err)
		}
		result := qwen35MTPDraftABSample{Sample: sample, Position: row.Position, Order: "cpu,vulkan"}
		var cpuLogits, deviceLogits []float32
		cpu := func() {
			started := time.Now()
			cpuLogits, err = cpuForward.Forward(row.Position, row.PriorHidden, embedding)
			result.CPUNS = time.Since(started).Nanoseconds()
			if err != nil {
				t.Fatalf("CPU MTP position %d: %v", row.Position, err)
			}
		}
		device := func() {
			window, available, err := compute.BeginBackendExecutionObservation(backend)
			if err != nil || !available {
				t.Fatalf("begin Vulkan observation position %d: available=%t err=%v", row.Position, available, err)
			}
			started := time.Now()
			deviceLogits, err = deviceForward.Forward(row.Position, row.PriorHidden, embedding)
			result.DeviceNS = time.Since(started).Nanoseconds()
			if err != nil {
				t.Fatalf("Vulkan MTP position %d: %v", row.Position, err)
			}
			result.DeviceObservation, err = window.End()
			if err != nil {
				t.Fatalf("end Vulkan observation position %d: %v", row.Position, err)
			}
			if result.DeviceObservation.Identity != candidate.ExpectedSystem {
				t.Fatalf("system provenance=%+v, want %+v", result.DeviceObservation.Identity, candidate.ExpectedSystem)
			}
		}
		if cpuFirst {
			cpu()
			device()
		} else {
			result.Order = "vulkan,cpu"
			device()
			cpu()
		}
		if !qwen35MTPDraftABFinite(cpuLogits) || !qwen35MTPDraftABFinite(deviceLogits) {
			t.Fatalf("position %d produced non-finite CPU or Vulkan logits", row.Position)
		}
		result.CPUArgmax, result.DeviceArgmax = qwen35MTPDraftABArgmax(cpuLogits), qwen35MTPDraftABArgmax(deviceLogits)
		result.Cosine, result.MaxRelativeDelta = qwen35MTPDraftABAccuracy(cpuLogits, deviceLogits)
		if math.IsNaN(result.Cosine) || math.IsInf(result.Cosine, 0) || math.IsNaN(result.MaxRelativeDelta) || math.IsInf(result.MaxRelativeDelta, 0) ||
			result.CPUArgmax != result.DeviceArgmax || result.Cosine < 0.99999 || result.MaxRelativeDelta > 2e-5 {
			t.Fatalf("position %d parity argmax=%d/%d cosine=%.9f relative=%g", row.Position, result.CPUArgmax, result.DeviceArgmax, result.Cosine, result.MaxRelativeDelta)
		}
		if !result.DeviceObservation.TransferCountersObserved || result.DeviceObservation.Counters.ComputeDispatches == 0 {
			t.Fatalf("position %d did not prove observed Vulkan execution: %+v", row.Position, result.DeviceObservation)
		}
		return result
	}

	_ = run(rows[0], -1, true) // unreported residency/pipeline warm-up
	samples := make([]qwen35MTPDraftABSample, candidate.Samples)
	for i := range samples {
		samples[i] = run(rows[i+1], i, i%2 == 0)
	}
	receipt := deviceForward.Receipt()
	if receipt.Path != compute.Qwen35MTPDraftPath || receipt.Backend != backend.Name() || receipt.Positions != uint64(candidate.Samples+1) {
		t.Fatalf("incomplete resident MTP receipt: %+v", receipt)
	}
	if receipt.IntermediateH2DBytes != 0 || receipt.IntermediateD2HBytes != 0 {
		t.Fatalf("resident MTP performed intermediate host transfers: %+v", receipt)
	}

	summary := summarizeQwen35MTPDraftAB(candidate, samples, receipt, producedInputsSHA256, producerExecutableSHA256, systemState)
	encoded, err := json.Marshal(summary)
	if err != nil {
		t.Fatal(err)
	}
	t.Log(string(encoded))
	if !summary.Qualified {
		t.Fatal(summary.Qualification)
	}
}

func readQwen35MTPDraftABManifest(t *testing.T, path string) qwen35MTPDraftABManifest {
	t.Helper()
	if !filepath.IsAbs(path) {
		t.Fatal("FAK_MTP_DRAFT_AB_MANIFEST must be absolute")
	}
	var candidate qwen35MTPDraftABManifest
	decodeStrictJSON(t, path, &candidate)
	if candidate.Schema != qwen35MTPDraftABSchema || !strings.Contains(strings.ToLower(candidate.Source), "qwen3.8") || len(candidate.Revision) < 12 || len(candidate.CodeRevision) < 12 {
		t.Fatal("manifest requires the Qwen3.8 schema, source, immutable artifact revision, and code revision")
	}
	if !filepath.IsAbs(candidate.GGUF) || !filepath.IsAbs(candidate.Inputs) || !filepath.IsAbs(candidate.SystemState) || !filepath.IsAbs(candidate.ProducedInputs) || candidate.Samples < 5 || candidate.MaxP50NS <= 0 || candidate.MaxP90NS <= 0 {
		t.Fatal("manifest requires absolute artifact/input/system-state paths, samples >= 5, and positive absolute P50/P90 ceilings")
	}
	if candidate.ExpectedSystem.Backend != "vulkan" || candidate.ExpectedSystem.Device == "" || candidate.ExpectedSystem.Driver == "" || candidate.ExpectedSystem.Runtime == "" {
		t.Fatal("manifest requires exact Vulkan device, driver, and runtime provenance")
	}
	decodeSHA256(t, candidate.GGUFSHA256)
	decodeSHA256(t, candidate.InputsSHA256)
	decodeSHA256(t, candidate.SystemSHA256)
	return candidate
}

func readQwen35MTPDraftABInputs(t *testing.T, candidate qwen35MTPDraftABManifest, manifestPath string) qwen35MTPDraftABInputs {
	t.Helper()
	var inputs qwen35MTPDraftABInputs
	decodeStrictJSON(t, candidate.Inputs, &inputs)
	if inputs.Schema != qwen35MTPDraftABSchema+"/inputs" || inputs.Source != candidate.Source || inputs.Revision != candidate.Revision || !strings.EqualFold(inputs.GGUFSHA256, candidate.GGUFSHA256) {
		t.Fatal("input provenance does not match the checkpoint manifest")
	}
	wantCommand := fmt.Sprintf("FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 FAK_MTP_DRAFT_AB_MANIFEST=%s %s", manifestPath, qwen35MTPDraftABTestCommand)
	if inputs.Producer != "fak-native-cpu-target-hidden" || inputs.ProducerCommand != wantCommand {
		t.Fatalf("target-hidden producer=%q command=%q, want fak-native producer and %q", inputs.Producer, inputs.ProducerCommand, wantCommand)
	}
	return inputs
}

func readQwen35MTPDraftABSystemState(t *testing.T, candidate qwen35MTPDraftABManifest) qwen35MTPDraftABSystemState {
	t.Helper()
	var state qwen35MTPDraftABSystemState
	decodeStrictJSON(t, candidate.SystemState, &state)
	if state.Schema != qwen35MTPDraftABSchema+"/system-state" || state.CodeRevision != candidate.CodeRevision || !strings.EqualFold(state.GGUFSHA256, candidate.GGUFSHA256) {
		t.Fatal("system-state provenance does not match the code and checkpoint manifest")
	}
	if _, err := time.Parse(time.RFC3339Nano, state.Timestamp); err != nil {
		t.Fatalf("system-state timestamp %q is not RFC3339: %v", state.Timestamp, err)
	}
	if strings.TrimSpace(state.Governor) == "" || len(state.ClocksMHz) == 0 || len(state.ThermalsC) == 0 {
		t.Fatal("system-state receipt requires observed governor, clocks_mhz, and thermals_c")
	}
	for name, value := range state.ClocksMHz {
		if strings.TrimSpace(name) == "" || value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("invalid observed clock %q=%v MHz", name, value)
		}
	}
	for name, value := range state.ThermalsC {
		if strings.TrimSpace(name) == "" || math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("invalid observed thermal %q=%v C", name, value)
		}
	}
	return state
}

func validateQwen35MTPDraftABTokens(t *testing.T, tokens []int, samples int, cfg model.Config) {
	t.Helper()
	if len(tokens) != samples+1 {
		t.Fatalf("input tokens=%d, want one warm-up + %d measured", len(tokens), samples)
	}
	for pos, token := range tokens {
		if token < 0 || token >= cfg.VocabSize {
			t.Fatalf("input token at position %d is %d, want [0,%d)", pos, token, cfg.VocabSize)
		}
	}
}

func validateQwen35MTPDraftABRows(t *testing.T, rows []qwen35MTPDraftABInputRow, samples int, cfg model.Config) {
	t.Helper()
	if len(rows) != samples+1 {
		t.Fatalf("input rows=%d, want one warm-up + %d measured", len(rows), samples)
	}
	for i, row := range rows {
		if row.Position != i || row.Token < 0 || row.Token >= cfg.VocabSize || len(row.PriorHidden) != cfg.HiddenSize {
			t.Fatalf("input row %d has position/token/hidden %d/%d/%d, want %d/[0,%d)/%d", i, row.Position, row.Token, len(row.PriorHidden), i, cfg.VocabSize, cfg.HiddenSize)
		}
		var squares float64
		positive, negative := false, false
		for _, value := range row.PriorHidden {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("input row %d contains non-finite hidden value", i)
			}
			squares += float64(value) * float64(value)
			positive = positive || value > 0
			negative = negative || value < 0
		}
		rms := math.Sqrt(squares / float64(len(row.PriorHidden)))
		if i == 0 {
			if squares != 0 {
				t.Fatalf("input row 0 prior hidden is non-zero; MTP bootstrap requires exact zero")
			}
			continue
		}
		if !positive || !negative || rms < 1e-3 || rms > 1e3 {
			t.Fatalf("input row %d lacks a bounded mixed-sign target hidden (rms=%g)", i, rms)
		}
	}
}

func assertQwen35MTPDraftABQ6(t *testing.T, m *model.Model) {
	t.Helper()
	checks := []struct {
		name    string
		out, in int
	}{
		{name: "mtp.layers.0.self_attn.v_proj.weight", out: m.Cfg.NumKVHeads * m.Cfg.HeadDim, in: m.Cfg.HiddenSize},
		{name: "mtp.layers.0.mlp.down_proj.weight", out: m.Cfg.HiddenSize, in: m.Cfg.IntermediateSize},
		{name: "lm_head.weight", out: m.Cfg.VocabSize, in: m.Cfg.HiddenSize},
	}
	for _, check := range checks {
		raw, ok := m.KQuantRaw(check.name)
		want := check.out * (check.in / 256) * 210
		if !ok || check.in%256 != 0 || len(raw) != want {
			t.Fatalf("%s is not retained packed Q6_K: present=%t bytes=%d want=%d shape=[%d,%d]", check.name, ok, len(raw), want, check.out, check.in)
		}
	}
}

func summarizeQwen35MTPDraftAB(candidate qwen35MTPDraftABManifest, samples []qwen35MTPDraftABSample, receipt model.Qwen35MTPForwardReceipt, producedInputsSHA256, producerExecutableSHA256 string, systemState qwen35MTPDraftABSystemState) qwen35MTPDraftABSummary {
	cpu, device := make([]int64, len(samples)), make([]int64, len(samples))
	for i := range samples {
		cpu[i], device[i] = samples[i].CPUNS, samples[i].DeviceNS
	}
	summary := qwen35MTPDraftABSummary{
		Schema: qwen35MTPDraftABSchema, Source: candidate.Source, Revision: candidate.Revision,
		CodeRevision: candidate.CodeRevision, GGUFSHA256: strings.ToLower(candidate.GGUFSHA256), InputsSHA256: strings.ToLower(candidate.InputsSHA256),
		ProducedInputsSHA256: producedInputsSHA256, ProducerExecutableSHA256: producerExecutableSHA256,
		SystemStateSHA256: strings.ToLower(candidate.SystemSHA256), RecordedAt: time.Now().UTC().Format(time.RFC3339Nano), SystemStateAt: systemState.Timestamp,
		System: candidate.ExpectedSystem, Samples: samples, CPUP50NS: qwen35MTPDraftABPercentile(cpu, .50), CPUP90NS: qwen35MTPDraftABPercentile(cpu, .90),
		DeviceP50NS: qwen35MTPDraftABPercentile(device, .50), DeviceP90NS: qwen35MTPDraftABPercentile(device, .90), ForwardReceipt: receipt,
	}
	summary.CPUP50CI95 = qwen35MTPDraftABBootstrap(cpu, .50, "cpu-p50")
	summary.CPUP90CI95 = qwen35MTPDraftABBootstrap(cpu, .90, "cpu-p90")
	summary.DeviceP50CI95 = qwen35MTPDraftABBootstrap(device, .50, "p50")
	summary.DeviceP90CI95 = qwen35MTPDraftABBootstrap(device, .90, "p90")
	summary.P50ImprovementNS = summary.CPUP50NS - summary.DeviceP50NS
	summary.P90ImprovementNS = summary.CPUP90NS - summary.DeviceP90NS
	if summary.CPUP50NS > 0 {
		summary.P50ImprovementPercent = float64(summary.P50ImprovementNS) * 100 / float64(summary.CPUP50NS)
	}
	if summary.CPUP90NS > 0 {
		summary.P90ImprovementPercent = float64(summary.P90ImprovementNS) * 100 / float64(summary.CPUP90NS)
	}
	medianImproved := summary.DeviceP50NS*100 <= summary.CPUP50NS*95
	summary.Qualified = summary.DeviceP50CI95.HighNS <= candidate.MaxP50NS && summary.DeviceP90CI95.HighNS <= candidate.MaxP90NS && medianImproved
	summary.Qualification = fmt.Sprintf("device CI95 upper P50/P90=%d/%d ns; reviewed ceilings=%d/%d ns; matched P50 improvement=%.3f%% (required >=5%%)", summary.DeviceP50CI95.HighNS, summary.DeviceP90CI95.HighNS, candidate.MaxP50NS, candidate.MaxP90NS, summary.P50ImprovementPercent)
	return summary
}

func qwen35MTPDraftABAccuracy(cpu, device []float32) (cosine, maxRelative float64) {
	if len(cpu) != len(device) || len(cpu) == 0 {
		return 0, math.Inf(1)
	}
	var dot, na, nb, maxDelta, maxRef float64
	for i := range cpu {
		a, b := float64(cpu[i]), float64(device[i])
		dot, na, nb = dot+a*b, na+a*a, nb+b*b
		maxDelta = math.Max(maxDelta, math.Abs(a-b))
		maxRef = math.Max(maxRef, math.Abs(a))
	}
	return dot / math.Sqrt(na*nb), maxDelta / math.Max(1, maxRef)
}

func qwen35MTPDraftABFinite(values []float32) bool {
	if len(values) == 0 {
		return false
	}
	for _, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return false
		}
	}
	return true
}

func qwen35MTPDraftABHashRows(rows []qwen35MTPDraftABInputRow) string {
	h := sha256.New()
	var word [8]byte
	for _, row := range rows {
		binary.LittleEndian.PutUint64(word[:], uint64(row.Position))
		_, _ = h.Write(word[:])
		binary.LittleEndian.PutUint64(word[:], uint64(row.Token))
		_, _ = h.Write(word[:])
		for _, value := range row.PriorHidden {
			binary.LittleEndian.PutUint32(word[:4], math.Float32bits(value))
			_, _ = h.Write(word[:4])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func qwen35MTPDraftABExecutableSHA256(t *testing.T) string {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return fileSHA256Hex(t, executable)
}

func writeQwen35MTPDraftABProducedInputs(t *testing.T, candidate qwen35MTPDraftABManifest, rows []qwen35MTPDraftABInputRow, producedInputsSHA256, producerExecutableSHA256 string) {
	t.Helper()
	receipt := qwen35MTPDraftABProducedInputs{
		Schema: qwen35MTPDraftABSchema + "/produced-inputs", Timestamp: time.Now().UTC().Format(time.RFC3339Nano),
		CodeRevision: candidate.CodeRevision, GGUFSHA256: strings.ToLower(candidate.GGUFSHA256), TokenInputsSHA256: strings.ToLower(candidate.InputsSHA256),
		ProducedInputsSHA256: producedInputsSHA256, ProducerExecutableSHA256: producerExecutableSHA256, Rows: rows,
	}
	encoded, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(candidate.ProducedInputs), 0o755); err != nil {
		t.Fatal(err)
	}
	temporary := candidate.ProducedInputs + ".tmp"
	if err := os.WriteFile(temporary, append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(temporary, candidate.ProducedInputs); err != nil {
		t.Fatal(err)
	}
}

func qwen35MTPDraftABArgmax(values []float32) int {
	best := 0
	for i := 1; i < len(values); i++ {
		if values[i] > values[best] {
			best = i
		}
	}
	return best
}

func qwen35MTPDraftABBootstrap(values []int64, quantile float64, label string) qwen35MTPDraftABCI {
	h := fnv.New64a()
	_, _ = h.Write([]byte(qwen35MTPDraftABSchema + label))
	rng := rand.New(rand.NewSource(int64(h.Sum64() & uint64(^uint64(0)>>1))))
	const replicates = 4096
	distribution, resample := make([]int64, replicates), make([]int64, len(values))
	for i := range distribution {
		for j := range resample {
			resample[j] = values[rng.Intn(len(values))]
		}
		distribution[i] = qwen35MTPDraftABPercentile(resample, quantile)
	}
	return qwen35MTPDraftABCI{LowNS: qwen35MTPDraftABPercentile(distribution, .025), HighNS: qwen35MTPDraftABPercentile(distribution, .975)}
}

func qwen35MTPDraftABPercentile(values []int64, quantile float64) int64 {
	ordered := append([]int64(nil), values...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })
	return ordered[int(float64(len(ordered)-1)*quantile+.5)]
}

func decodeStrictJSON(t *testing.T, path string, target any) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	decoder := json.NewDecoder(f)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
}

func assertFileSHA256(t *testing.T, path, expected string) {
	t.Helper()
	want := decodeSHA256(t, expected)
	got := fileSHA256Hex(t, path)
	if !strings.EqualFold(got, hex.EncodeToString(want)) {
		t.Fatalf("SHA256 mismatch for %s", path)
	}
}

func fileSHA256Hex(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h.Sum(nil))
}

func decodeSHA256(t *testing.T, value string) []byte {
	t.Helper()
	decoded, err := hex.DecodeString(value)
	if err != nil || len(decoded) != sha256.Size {
		t.Fatalf("malformed SHA256 %q", value)
	}
	return decoded
}
