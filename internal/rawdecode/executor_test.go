package rawdecode

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type fakeSession struct {
	outputs [][]float32
	prefill int
	steps   int
	closed  int
}

func (s *fakeSession) Prefill([]int) []float32 {
	s.prefill++
	return append([]float32(nil), s.outputs[0]...)
}
func (s *fakeSession) Step(int) []float32 {
	s.steps++
	return append([]float32(nil), s.outputs[s.steps]...)
}
func (s *fakeSession) Close() { s.closed++ }

type fakeLoadedModel struct {
	candidateCalls int
	cpuCalls       int
	closeCalls     int
	candidate      *fakeSession
	cpu            *fakeSession
}

type rawObservedBackend struct {
	compute.Backend
	name      string
	snapshots []compute.BackendExecutionSnapshot
	calls     int
}

func (b *rawObservedBackend) Name() string                    { return b.name }
func (b *rawObservedBackend) Tier() string                    { return "observed" }
func (b *rawObservedBackend) Class() compute.CorrectnessClass { return compute.Approx }
func (b *rawObservedBackend) Caps() compute.Caps              { return compute.Caps{DeviceMemory: true} }
func (b *rawObservedBackend) BackendExecutionSnapshot() (compute.BackendExecutionSnapshot, error) {
	b.calls++
	return b.snapshots[b.calls-1], nil
}

type rawUnsupportedBackend struct {
	compute.Backend
	name string
}

func (b rawUnsupportedBackend) Name() string                    { return b.name }
func (b rawUnsupportedBackend) Tier() string                    { return "unsupported" }
func (b rawUnsupportedBackend) Class() compute.CorrectnessClass { return compute.Approx }
func (b rawUnsupportedBackend) Caps() compute.Caps              { return compute.Caps{} }

func (m *fakeLoadedModel) Config() model.Config { return model.Config{VocabSize: 3, EOSTokenID: -1} }
func (m *fakeLoadedModel) IsEOS(int) bool       { return false }
func (m *fakeLoadedModel) CloseWeights() error  { m.closeCalls++; return nil }
func (m *fakeLoadedModel) NewCandidateSession(compute.Backend, Request) (session, error) {
	m.candidateCalls++
	return m.candidate, nil
}
func (m *fakeLoadedModel) NewCPUSession(Request) session {
	m.cpuCalls++
	return m.cpu
}

func fakeClock() func() time.Time {
	now := time.Unix(100, 0)
	return func() time.Time {
		now = now.Add(time.Millisecond)
		return now
	}
}

type rawDecodeGGUFTensor struct {
	name string
	dims []uint64
	typ  ggufload.TensorType
}

func rawDecodeGGUFFixture(t *testing.T, tensors []rawDecodeGGUFTensor) []byte {
	t.Helper()
	var b bytes.Buffer
	b.WriteString(ggufload.Magic)
	write := func(value any) {
		t.Helper()
		if err := binary.Write(&b, binary.LittleEndian, value); err != nil {
			t.Fatalf("write GGUF fixture: %v", err)
		}
	}
	write(uint32(ggufload.Version))
	write(uint64(len(tensors)))
	write(uint64(0))
	for _, tensor := range tensors {
		write(uint64(len(tensor.name)))
		b.WriteString(tensor.name)
		write(uint32(len(tensor.dims)))
		for _, dim := range tensor.dims {
			write(dim)
		}
		write(uint32(tensor.typ))
		write(uint64(0))
	}
	return b.Bytes()
}

func TestRawDecodeExecutorCarriesParsedGGUFProvenance(t *testing.T) {
	tensors := []rawDecodeGGUFTensor{
		{name: "blk.0.ffn_gate.weight", dims: []uint64{256, 256}, typ: ggufload.TensorQ4_K},
		{name: "blk.0.ffn_up.weight", dims: []uint64{256, 256}, typ: ggufload.TensorQ6_K},
		{name: "output.weight", dims: []uint64{32, 32}, typ: ggufload.TensorQ8_0},
	}
	original := rawDecodeGGUFFixture(t, tensors)
	openBytes := func(data []byte) func(string) (io.ReadCloser, error) {
		return func(string) (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(data)), nil
		}
	}
	observed, err := inspectGGUFArtifact("fixtures/qwen.gguf", openBytes(original))
	if err != nil {
		t.Fatalf("inspect GGUF artifact: %v", err)
	}
	wantDigest := fmt.Sprintf("%x", sha256.Sum256(original))
	if observed.SHA256 != wantDigest || observed.Path != "fixtures/qwen.gguf" || observed.Quantization != "Q4_K_M" || !strings.HasPrefix(observed.TensorInventorySHA256, "sha256:") {
		t.Fatalf("parsed artifact observation mismatch: %+v", observed)
	}
	if observed.TokenizerSHA256 != "" || observed.TemplateSHA256 != "" {
		t.Fatalf("fixture without tokenizer metadata invented identity: %+v", observed)
	}

	reordered := rawDecodeGGUFFixture(t, []rawDecodeGGUFTensor{tensors[2], tensors[0], tensors[1]})
	reorderedObserved, err := inspectGGUFArtifact("reordered.gguf", openBytes(reordered))
	if err != nil {
		t.Fatalf("inspect reordered GGUF artifact: %v", err)
	}
	if reorderedObserved.TensorInventorySHA256 != observed.TensorInventorySHA256 {
		t.Fatalf("tensor directory order changed canonical digest: %q != %q", reorderedObserved.TensorInventorySHA256, observed.TensorInventorySHA256)
	}
	mutated := append([]rawDecodeGGUFTensor(nil), tensors...)
	mutated[0].dims = []uint64{256, 512}
	mutatedObserved, err := inspectGGUFArtifact("mutated.gguf", openBytes(rawDecodeGGUFFixture(t, mutated)))
	if err != nil {
		t.Fatalf("inspect mutated GGUF artifact: %v", err)
	}
	if mutatedObserved.TensorInventorySHA256 == observed.TensorInventorySHA256 {
		t.Fatal("tensor dimension change did not change canonical digest")
	}

	m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 1, 5}}}}
	d := dependencies{
		openArtifact:    openBytes(original),
		inspectArtifact: inspectGGUFArtifact,
		loadModel: func(context.Context, Request) (loadedModel, string, error) {
			return m, "qwen.gguf [gguf-q4k]", nil
		},
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			return nil, BackendObservation{Selected: "legacy"}, nil
		},
		now: fakeClock(),
	}
	execution, err := d.execute(context.Background(), Request{
		ArtifactPath: "fixtures/qwen.gguf", ExpectedArtifactSHA256: wantDigest,
		PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1,
	})
	if err != nil {
		t.Fatalf("execute parsed GGUF artifact: %v", err)
	}
	if execution.ArtifactPath != observed.Path || execution.ArtifactSHA256 != observed.SHA256 ||
		execution.TensorInventorySHA256 != observed.TensorInventorySHA256 || execution.Quantization != observed.Quantization ||
		execution.TokenizerSHA256 != "" || execution.TemplateSHA256 != "" ||
		execution.ModelName != "qwen.gguf [gguf-q4k]" {
		t.Fatalf("execution lost parsed GGUF provenance: %+v", execution)
	}
}

func TestRawDecodeExecutorCarriesEmbeddedTokenizerTemplateIdentity(t *testing.T) {
	compressed, err := os.ReadFile(filepath.Join("..", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	closeErr := zr.Close()
	if err != nil {
		t.Fatal(err)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	openBytes := func(string) (io.ReadCloser, error) {
		return io.NopCloser(bytes.NewReader(raw)), nil
	}
	observed, err := inspectGGUFArtifact("fixture.gguf", openBytes)
	if err != nil {
		t.Fatal(err)
	}
	if observed.TokenizerSHA256 != "839c662c4a47759df9150bd939b382767b1e36bc2372740ad6d02495a63fa5a0" || observed.TemplateSHA256 != "87049d017c4eee304541572ddfee78756784389fac4808d02039215f062d31d1" {
		t.Fatalf("embedded identity mismatch: %+v", observed)
	}

	m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 1, 5}}}}
	d := dependencies{
		openArtifact: openBytes,
		inspectArtifact: func(string, func(string) (io.ReadCloser, error)) (artifactObservation, error) {
			return observed, nil
		},
		loadModel: func(context.Context, Request) (loadedModel, string, error) {
			return m, "qwen.gguf [gguf-q4k]", nil
		},
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			return nil, BackendObservation{Selected: "legacy"}, nil
		},
		now: fakeClock(),
	}
	execution, err := d.execute(context.Background(), Request{
		ArtifactPath: "fixture.gguf", ExpectedArtifactSHA256: observed.SHA256,
		PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if execution.TokenizerSHA256 != observed.TokenizerSHA256 || execution.TemplateSHA256 != observed.TemplateSHA256 {
		t.Fatalf("execution lost paired tokenizer/template identity: %+v", execution)
	}
}

func TestRawDecodeExecutorRejectsArtifactMismatchBeforeExecution(t *testing.T) {
	loadCalls, backendCalls := 0, 0
	d := dependencies{
		openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader("actual artifact")), nil },
		loadModel: func(context.Context, Request) (loadedModel, string, error) {
			loadCalls++
			return nil, "", fmt.Errorf("must not load")
		},
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			backendCalls++
			return nil, BackendObservation{}, fmt.Errorf("must not resolve")
		},
		now: fakeClock(),
	}
	req := Request{ArtifactPath: "model.gguf", ExpectedArtifactSHA256: strings.Repeat("0", 64), PromptTokenIDs: []int{1}, ContextLimit: 4, GeneratedTokenLimit: 1, Repetitions: 1}
	if _, err := d.execute(context.Background(), req); err == nil || !strings.Contains(err.Error(), "artifact SHA-256 mismatch") {
		t.Fatalf("expected digest mismatch, got %v", err)
	}
	if loadCalls != 0 || backendCalls != 0 {
		t.Fatalf("mismatch crossed fail-closed boundary: load=%d backend=%d", loadCalls, backendCalls)
	}
}

func TestRawDecodeExecutorObservesRealSessionCallsOnly(t *testing.T) {
	artifact := "device-free fake artifact bytes"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	outputs := [][]float32{{0, 1, 5}, {0, 4, 1}, {3, 2, 1}}
	m := &fakeLoadedModel{
		candidate: &fakeSession{outputs: outputs},
		cpu:       &fakeSession{outputs: outputs},
	}
	loadCalls, backendCalls := 0, 0
	d := dependencies{
		openArtifact: func(path string) (io.ReadCloser, error) {
			if path != "selected.gguf" {
				t.Fatalf("unexpected artifact path %q", path)
			}
			return io.NopCloser(strings.NewReader(artifact)), nil
		},
		loadModel: func(_ context.Context, req Request) (loadedModel, string, error) {
			loadCalls++
			return m, "observed-model [gguf]", nil
		},
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			backendCalls++
			return nil, BackendObservation{Selected: "legacy"}, nil
		},
		now: fakeClock(),
	}
	req := Request{
		ArtifactPath: "selected.gguf", ExpectedArtifactSHA256: digest,
		PromptTokenIDs: []int{0, 1}, ContextLimit: 8, GeneratedTokenLimit: 3,
		Repetitions: 1, VerifyCPU: true,
	}
	execution, err := d.execute(context.Background(), req)
	if err != nil {
		t.Fatalf("execute: %v", err)
	}
	if loadCalls != 1 || backendCalls != 1 || m.candidateCalls != 1 || m.cpuCalls != 1 || m.closeCalls != 1 {
		t.Fatalf("seam calls load=%d backend=%d candidate=%d cpu=%d close=%d", loadCalls, backendCalls, m.candidateCalls, m.cpuCalls, m.closeCalls)
	}
	if got := execution.Runs[0].GeneratedTokens; fmt.Sprint(got) != "[2 1 0]" {
		t.Fatalf("generated tokens %v did not come from session logits", got)
	}
	if execution.CPUModelParity == nil || !*execution.CPUModelParity {
		t.Fatalf("CPU parity was not observed: %v", execution.CPUModelParity)
	}
	if execution.ArtifactSHA256 != digest || execution.Runs[0].PrefillDuration <= 0 || execution.Runs[0].DecodeDuration <= 0 {
		t.Fatalf("missing observed digest/timings: %+v", execution)
	}
	if execution.Backend.Selected != "legacy" || execution.ModelName != "observed-model [gguf]" {
		t.Fatalf("resolved identity mismatch: backend=%q model=%q", execution.Backend.Selected, execution.ModelName)
	}
}

func TestRawDecodeExecutorRejectsModelSelectorBeforeSession(t *testing.T) {
	artifact := "selector-bound artifact"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{1}}}}
	d := dependencies{
		openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(artifact)), nil },
		loadModel: func(context.Context, Request) (loadedModel, string, error) {
			return m, "loaded-model [gguf]", nil
		},
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			return nil, BackendObservation{Selected: "legacy"}, nil
		},
		now: fakeClock(),
	}
	req := Request{
		ArtifactPath: "selected.gguf", ExpectedArtifactSHA256: digest,
		ModelName: "different-model", PromptTokenIDs: []int{0}, ContextLimit: 2,
		GeneratedTokenLimit: 1, Repetitions: 1,
	}
	if _, err := d.execute(context.Background(), req); err == nil || !strings.Contains(err.Error(), "model selector") {
		t.Fatalf("expected model selector mismatch, got %v", err)
	}
	if m.candidateCalls != 0 {
		t.Fatalf("model mismatch created %d candidate sessions", m.candidateCalls)
	}
}

func TestRawDecodeBackendObservationIsPerRunAndBackendOwned(t *testing.T) {
	artifact := "observed backend artifact"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	identity := compute.BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "vulkan-1.3.0"}
	before := compute.BackendExecutionSnapshot{Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 100, H2DBytes: 1000}}
	after := before
	after.Counters.ComputeDispatches = 107
	after.Counters.H2DBytes = 1064
	backend := &rawObservedBackend{name: "vulkan", snapshots: []compute.BackendExecutionSnapshot{before, after}}
	m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 2, 1}}}}
	d := dependencies{
		openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(artifact)), nil },
		loadModel:    func(context.Context, Request) (loadedModel, string, error) { return m, "observed-model", nil },
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			return backend, BackendObservation{Selected: backend.Name()}, nil
		},
		now: fakeClock(),
	}
	req := Request{ArtifactPath: "model.gguf", ExpectedArtifactSHA256: digest, ModelName: "observed-model", BackendName: "caller-selector", PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}
	exec, err := d.execute(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	observed := exec.Runs[0].BackendExecution
	if observed == nil || observed.Identity != identity || observed.Counters.ComputeDispatches != 7 || observed.Counters.H2DBytes != 64 {
		t.Fatalf("backend-owned per-run observation = %+v", observed)
	}
	if observed.Counters.ComputeDispatches == after.Counters.ComputeDispatches || observed.Identity.Backend == req.BackendName {
		t.Fatalf("caller label or prior cumulative counters leaked into observation: %+v", observed)
	}
}

func TestRawDecodeBackendObservationRefusesForgedLabelMissingIdentityAndReset(t *testing.T) {
	artifact := "refusal artifact"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	request := Request{ArtifactPath: "model.gguf", ExpectedArtifactSHA256: digest, ModelName: "observed-model", BackendName: "forged", PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}
	newModel := func() *fakeLoadedModel {
		return &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 2, 1}}}}
	}
	baseDeps := func(m *fakeLoadedModel, backend compute.Backend, label string) dependencies {
		return dependencies{
			openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(artifact)), nil },
			loadModel:    func(context.Context, Request) (loadedModel, string, error) { return m, "observed-model", nil },
			resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
				return backend, BackendObservation{Selected: label}, nil
			},
			now: fakeClock(),
		}
	}

	t.Run("forged resolver label", func(t *testing.T) {
		m := newModel()
		backend := rawUnsupportedBackend{name: "vulkan"}
		if got, err := baseDeps(m, backend, "forged").execute(context.Background(), request); err == nil || len(got.Runs) != 0 || got.ArtifactSHA256 != "" || m.candidateCalls != 0 {
			t.Fatalf("got=%+v err=%v candidate_calls=%d", got, err, m.candidateCalls)
		}
	})

	t.Run("missing backend identity", func(t *testing.T) {
		m := newModel()
		backend := &rawObservedBackend{name: "vulkan", snapshots: []compute.BackendExecutionSnapshot{{Identity: compute.BackendRuntimeIdentity{Backend: "vulkan"}}}}
		if got, err := baseDeps(m, backend, "vulkan").execute(context.Background(), request); err == nil || len(got.Runs) != 0 || got.ArtifactSHA256 != "" || m.candidateCalls != 0 {
			t.Fatalf("got=%+v err=%v candidate_calls=%d", got, err, m.candidateCalls)
		}
	})

	t.Run("counter reset", func(t *testing.T) {
		m := newModel()
		identity := compute.BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "runtime"}
		backend := &rawObservedBackend{name: "vulkan", snapshots: []compute.BackendExecutionSnapshot{
			{Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 2}},
			{Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 1}},
		}}
		if got, err := baseDeps(m, backend, "vulkan").execute(context.Background(), request); err == nil || len(got.Runs) != 0 || got.ArtifactSHA256 != "" || !strings.Contains(err.Error(), "reset") {
			t.Fatalf("got=%+v err=%v", got, err)
		}
	})
}

func TestRawDecodeUnsupportedBackendObservationRemainsUnavailable(t *testing.T) {
	artifact := "unsupported observation artifact"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 2, 1}}}}
	backend := rawUnsupportedBackend{name: "accelerator"}
	d := dependencies{
		openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(artifact)), nil },
		loadModel:    func(context.Context, Request) (loadedModel, string, error) { return m, "observed-model", nil },
		resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
			return backend, BackendObservation{Selected: backend.Name()}, nil
		},
		now: fakeClock(),
	}
	req := Request{ArtifactPath: "model.gguf", ExpectedArtifactSHA256: digest, ModelName: "observed-model", BackendName: "accelerator", PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}
	exec, err := d.execute(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if exec.Runs[0].BackendExecution != nil {
		t.Fatalf("unsupported backend was zero-filled as available: %+v", exec.Runs[0].BackendExecution)
	}
}
