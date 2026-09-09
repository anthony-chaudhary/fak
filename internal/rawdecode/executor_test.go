package rawdecode

import (
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
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
	candidateHook  func()
	config         *model.Config
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
	if b.calls > len(b.snapshots) {
		return compute.BackendExecutionSnapshot{}, fmt.Errorf("snapshot %d unavailable", b.calls)
	}
	return b.snapshots[b.calls-1], nil
}

func (b *rawObservedBackend) BeginBackendExecutionWindow() (compute.BackendExecutionWindow, error) {
	before, available, err := compute.CaptureBackendExecutionSnapshot(b)
	if err != nil {
		return nil, err
	}
	if !available {
		return nil, fmt.Errorf("opening snapshot unavailable")
	}
	return &rawObservedWindow{backend: b, before: before}, nil
}

type rawObservedWindow struct {
	backend *rawObservedBackend
	before  compute.BackendExecutionSnapshot
}

func (w *rawObservedWindow) End() (compute.BackendExecutionObservation, error) {
	after, available, err := compute.CaptureBackendExecutionSnapshot(w.backend)
	if err != nil {
		return compute.BackendExecutionObservation{}, err
	}
	if !available {
		return compute.BackendExecutionObservation{}, fmt.Errorf("closing snapshot unavailable")
	}
	peak := max(w.before.DeviceAllocationLiveBytes, after.DeviceAllocationLiveBytes)
	return compute.BackendExecutionWindowDelta(w.before, after, after.DeviceAllocationLiveBytes, peak)
}

type rawSnapshotOnlyBackend struct {
	rawUnsupportedBackend
	snapshot compute.BackendExecutionSnapshot
	calls    int
}

func (b *rawSnapshotOnlyBackend) BackendExecutionSnapshot() (compute.BackendExecutionSnapshot, error) {
	b.calls++
	return b.snapshot, nil
}

type rawUnsupportedBackend struct {
	compute.Backend
	name string
}

func (b rawUnsupportedBackend) Name() string                    { return b.name }
func (b rawUnsupportedBackend) Tier() string                    { return "unsupported" }
func (b rawUnsupportedBackend) Class() compute.CorrectnessClass { return compute.Approx }
func (b rawUnsupportedBackend) Caps() compute.Caps              { return compute.Caps{} }

type rawWindowBackend struct {
	rawUnsupportedBackend
	observations []compute.BackendExecutionObservation
	events       *[]string
	beginCalls   int
	endCalls     int
	beginErr     error
	endErr       error
	nilWindow    bool
}

func (b *rawWindowBackend) BeginBackendExecutionWindow() (compute.BackendExecutionWindow, error) {
	if b.events != nil {
		*b.events = append(*b.events, "begin")
	}
	index := b.beginCalls
	b.beginCalls++
	if b.beginErr != nil {
		return nil, b.beginErr
	}
	if b.nilWindow {
		return nil, nil
	}
	observation := compute.BackendExecutionObservation{}
	if index < len(b.observations) {
		observation = b.observations[index]
	}
	return &rawWindow{backend: b, observation: observation}, nil
}

type rawWindow struct {
	backend     *rawWindowBackend
	observation compute.BackendExecutionObservation
	ended       bool
}

func (w *rawWindow) End() (compute.BackendExecutionObservation, error) {
	if w.ended {
		return compute.BackendExecutionObservation{}, fmt.Errorf("window already ended")
	}
	w.ended = true
	w.backend.endCalls++
	if w.backend.events != nil {
		*w.backend.events = append(*w.backend.events, "end")
	}
	if w.backend.endErr != nil {
		return compute.BackendExecutionObservation{}, w.backend.endErr
	}
	return w.observation, nil
}

func (m *fakeLoadedModel) Config() model.Config {
	if m.config != nil {
		return *m.config
	}
	return model.Config{VocabSize: 3, EOSTokenID: -1}
}
func (m *fakeLoadedModel) IsEOS(int) bool      { return false }
func (m *fakeLoadedModel) CloseWeights() error { m.closeCalls++; return nil }
func (m *fakeLoadedModel) NewCandidateSession(compute.Backend, Request) (session, error) {
	m.candidateCalls++
	if m.candidateHook != nil {
		m.candidateHook()
	}
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

func TestRawDecodeOutputTextIsArtifactOwnedAndRunnerSealed(t *testing.T) {
	compressed, err := os.ReadFile(filepath.Join("..", "ggufload", "testdata", "qwen38_ud_q2kxl_header.gguf.gz"))
	if err != nil {
		t.Fatal(err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(io.LimitReader(zr, 16<<20))
	if err != nil {
		t.Fatal(err)
	}
	if err := zr.Close(); err != nil {
		t.Fatal(err)
	}
	openBytes := func(string) (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(raw)), nil }
	artifact, err := inspectGGUFArtifact("fixture.gguf", openBytes)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ggufload.Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	metadata, ok := parsed.GGMLTokenizer()
	if !ok {
		t.Fatal("fixture has no parsed GGML tokenizer")
	}
	expectedTokenizer, err := tokenizer.FromGGML(metadata.Tokens, metadata.Merges, metadata.TokenTypes, metadata.Pre)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := expectedTokenizer.Decode([]int{0})
	if err != nil || expected == "" {
		t.Fatalf("fixture token 0 decode = %q, %v", expected, err)
	}

	run := func(observed artifactObservation, tokenID, vocabSize, repetitions int) (Execution, error) {
		logits := make([]float32, vocabSize)
		logits[tokenID] = 1
		config := model.Config{VocabSize: vocabSize, EOSTokenID: -1}
		m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{logits}}, config: &config}
		d := dependencies{
			openArtifact: openBytes,
			inspectArtifact: func(string, func(string) (io.ReadCloser, error)) (artifactObservation, error) {
				return observed, nil
			},
			loadModel: func(context.Context, Request) (loadedModel, string, error) { return m, "fixture", nil },
			resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
				return nil, BackendObservation{Selected: "legacy"}, nil
			},
			now: fakeClock(),
		}
		return d.execute(context.Background(), Request{
			ArtifactPath: observed.Path, ExpectedArtifactSHA256: observed.SHA256,
			PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: repetitions,
		})
	}

	execution, err := run(artifact, 0, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	for repetition := range execution.Runs {
		observation, ok := execution.OutputTextObservation(repetition)
		if !ok {
			t.Fatalf("output text observation %d unavailable", repetition)
		}
		if text, ok := observation.Text(); !ok || text != expected {
			t.Fatalf("output text %d = %q, %v; want %q", repetition, text, ok, expected)
		}
	}

	mutatedTokens := execution
	mutatedTokens.Runs = slices.Clone(execution.Runs)
	mutatedTokens.Runs[0].GeneratedTokens = slices.Clone(execution.Runs[0].GeneratedTokens)
	mutatedTokens.Runs[0].GeneratedTokens[0]++
	if _, ok := mutatedTokens.OutputTextObservation(0); ok {
		t.Fatal("public token mutation retained output-text authority")
	}
	permuted := execution
	permuted.Runs = slices.Clone(execution.Runs)
	permuted.Runs[0], permuted.Runs[1] = permuted.Runs[1], permuted.Runs[0]
	if _, ok := permuted.OutputTextObservation(0); ok {
		t.Fatal("cross-run permutation retained output-text authority")
	}
	other, err := run(artifact, 0, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	transplanted := execution
	transplanted.Runs = slices.Clone(execution.Runs)
	transplanted.Runs[0] = other.Runs[0]
	if _, ok := transplanted.OutputTextObservation(0); ok {
		t.Fatal("cross-execution transplant retained output-text authority")
	}
	for name, mutate := range map[string]func(*Execution){
		"request":  func(got *Execution) { got.ContextLimit++ },
		"artifact": func(got *Execution) { got.ArtifactSHA256 = strings.Repeat("0", sha256.Size*2) },
		"model":    func(got *Execution) { got.ModelName = "other" },
	} {
		t.Run("binding "+name, func(t *testing.T) {
			drifted := execution
			mutate(&drifted)
			if _, ok := drifted.OutputTextObservation(0); ok {
				t.Fatal("execution binding drift retained output-text authority")
			}
		})
	}

	missing := artifact
	missing.outputTokenizer = nil
	if got, err := run(missing, 0, 3, 1); err != nil {
		t.Fatal(err)
	} else if observation, ok := got.OutputTextObservation(0); ok || !reflect.DeepEqual(observation, OutputTextObservation{}) {
		t.Fatalf("missing tokenizer metadata gained output-text authority: %+v, %v", observation, ok)
	}
	cloneParsed := func() *ggufload.File {
		cloned := *parsed
		cloned.Metadata = make(map[string]ggufload.Value, len(parsed.Metadata))
		for key, value := range parsed.Metadata {
			cloned.Metadata[key] = value
		}
		return &cloned
	}
	for _, key := range []string{"tokenizer.ggml.pre", "tokenizer.ggml.token_type"} {
		t.Run("optional metadata absent "+key, func(t *testing.T) {
			withoutOptional := cloneParsed()
			delete(withoutOptional.Metadata, key)
			observed := artifact
			observed.outputTokenizer = buildArtifactOutputTokenizerFromGGUF(withoutOptional)
			got, err := run(observed, 0, 3, 1)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := got.OutputTextObservation(0); !ok {
				t.Fatalf("valid absence of %s lost output-text authority", key)
			}
		})
	}
	for name, malformedValue := range map[string]struct {
		key   string
		value ggufload.Value
	}{
		"pre wrong type": {
			key: "tokenizer.ggml.pre", value: ggufload.Value{Type: ggufload.TypeUint32, Value: uint32(1)},
		},
		"token type wrong type": {
			key: "tokenizer.ggml.token_type", value: ggufload.Value{Type: ggufload.TypeString, Value: "wrong"},
		},
		"token type wrong element type": {
			key: "tokenizer.ggml.token_type", value: ggufload.Value{Type: ggufload.TypeArray, Value: []ggufload.Value{{Type: ggufload.TypeString, Value: "wrong"}}},
		},
	} {
		t.Run("malformed metadata "+name, func(t *testing.T) {
			malformedParsed := cloneParsed()
			malformedParsed.Metadata[malformedValue.key] = malformedValue.value
			observed := artifact
			observed.outputTokenizer = buildArtifactOutputTokenizerFromGGUF(malformedParsed)
			if observed.outputTokenizer != nil {
				t.Fatal("malformed present tokenizer metadata built a decoder")
			}
			got, err := run(observed, 0, 3, 1)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := got.OutputTextObservation(0); ok {
				t.Fatal("malformed present tokenizer metadata gained output-text authority")
			}
		})
	}
	malformed := artifact
	malformed.outputTokenizer = buildArtifactOutputTokenizer(&ggufload.GGMLTokenizer{
		Tokens: []string{"a", "b"}, Merges: []string{"malformed"},
	})
	if malformed.outputTokenizer != nil {
		t.Fatal("malformed tokenizer metadata built a decoder")
	}
	if got, err := run(malformed, 0, 3, 1); err != nil {
		t.Fatal(err)
	} else if _, ok := got.OutputTextObservation(0); ok {
		t.Fatal("malformed tokenizer metadata gained output-text authority")
	}
	emptyVocabulary := artifact
	emptyVocabulary.outputTokenizer = buildArtifactOutputTokenizer(&ggufload.GGMLTokenizer{
		Tokens: []string{"a", "", "b"}, Merges: []string{"a b"},
	})
	if emptyVocabulary.outputTokenizer != nil {
		t.Fatal("empty vocabulary entry synthesized an authoritative decoder")
	}
	if got, err := run(emptyVocabulary, 0, 3, 1); err != nil {
		t.Fatal(err)
	} else if _, ok := got.OutputTextObservation(0); ok {
		t.Fatal("empty vocabulary entry gained output-text authority")
	}

	tiny := artifact
	tiny.outputTokenizer = buildArtifactOutputTokenizer(&ggufload.GGMLTokenizer{
		Tokens: []string{"a", "b"}, Merges: []string{"a b"},
	})
	if got, err := run(tiny, 2, 3, 1); err != nil {
		t.Fatal(err)
	} else if _, ok := got.OutputTextObservation(0); ok {
		t.Fatal("out-of-range accepted token ID gained output-text authority")
	}
	if text, ok := decodeArtifactOutputText(tiny.outputTokenizer, []int{-1}); ok || text != "" {
		t.Fatalf("invalid token ID decoded as %q, %v", text, ok)
	}
	if text, ok := decodeArtifactOutputText(tiny.outputTokenizer, nil); ok || text != "" {
		t.Fatalf("empty output decoded as %q, %v", text, ok)
	}

	invalidUTF8 := artifact
	invalidUTF8.outputTokenizer = buildArtifactOutputTokenizer(&ggufload.GGMLTokenizer{
		Tokens: []string{"a", "b", "ab", "ÿ"}, Merges: []string{"a b"},
	})
	if got, err := run(invalidUTF8, 3, 4, 1); err != nil {
		t.Fatal(err)
	} else if _, ok := got.OutputTextObservation(0); ok {
		t.Fatal("invalid UTF-8 output gained output-text authority")
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

func rawDecodeProductionArtifactFixture(t *testing.T) []byte {
	t.Helper()
	// One real decoder block: the swap changes weight bytes while preserving
	// every model/config/tokenizer field, so metadata equality cannot detect it.
	var fixture bytes.Buffer
	write := func(value any) {
		t.Helper()
		if err := binary.Write(&fixture, binary.LittleEndian, value); err != nil {
			t.Fatal(err)
		}
	}
	writeString := func(value string) { write(uint64(len(value))); fixture.WriteString(value) }
	fixture.WriteString(ggufload.Magic)
	write(uint32(ggufload.Version))
	tensors := []rawDecodeGGUFTensor{
		{name: "token_embd.weight", dims: []uint64{32, 3}},
		{name: "output_norm.weight", dims: []uint64{32}},
		{name: "output.weight", dims: []uint64{32, 3}},
		{name: "blk.0.attn_norm.weight", dims: []uint64{32}},
		{name: "blk.0.attn_q.weight", dims: []uint64{32, 32}},
		{name: "blk.0.attn_k.weight", dims: []uint64{32, 32}},
		{name: "blk.0.attn_v.weight", dims: []uint64{32, 32}},
		{name: "blk.0.attn_output.weight", dims: []uint64{32, 32}},
		{name: "blk.0.ffn_norm.weight", dims: []uint64{32}},
		{name: "blk.0.ffn_gate.weight", dims: []uint64{32, 32}},
		{name: "blk.0.ffn_up.weight", dims: []uint64{32, 32}},
		{name: "blk.0.ffn_down.weight", dims: []uint64{32, 32}},
	}
	write(uint64(len(tensors)))
	write(uint64(10))
	for _, item := range []struct{ key, value string }{
		{"general.architecture", "qwen2"}, {"tokenizer.ggml.pre", "qwen2"},
	} {
		writeString(item.key)
		write(uint32(ggufload.TypeString))
		writeString(item.value)
	}
	for _, item := range []struct {
		key   string
		value uint32
	}{
		{"qwen2.embedding_length", 32}, {"qwen2.block_count", 1},
		{"qwen2.attention.head_count", 1}, {"qwen2.feed_forward_length", 32},
	} {
		writeString(item.key)
		write(uint32(ggufload.TypeUint32))
		write(item.value)
	}
	for _, item := range []struct {
		key    string
		values []string
	}{
		{"tokenizer.ggml.tokens", []string{"a", "b", "ab"}},
		{"tokenizer.ggml.merges", []string{"a b"}},
	} {
		writeString(item.key)
		write(uint32(ggufload.TypeArray))
		write(uint32(ggufload.TypeString))
		write(uint64(len(item.values)))
		for _, value := range item.values {
			writeString(value)
		}
	}
	for _, item := range []struct {
		key   string
		value float32
	}{
		{"qwen2.attention.layer_norm_rms_epsilon", 1e-5}, {"qwen2.rope.freq_base", 10000},
	} {
		writeString(item.key)
		write(uint32(ggufload.TypeFloat32))
		write(item.value)
	}
	var offset uint64
	for _, tensor := range tensors {
		writeString(tensor.name)
		write(uint32(len(tensor.dims)))
		elements := uint64(1)
		for _, dim := range tensor.dims {
			write(dim)
			elements *= dim
		}
		write(uint32(ggufload.TensorF32))
		write(offset)
		offset += elements * 4
	}
	for fixture.Len()%32 != 0 {
		fixture.WriteByte(0)
	}
	for _, tensor := range tensors {
		elements := uint64(1)
		for _, dim := range tensor.dims {
			elements *= dim
		}
		for i := uint64(0); i < elements; i++ {
			value := float32(0)
			if tensor.name == "token_embd.weight" || strings.Contains(tensor.name, "norm.weight") {
				value = 1
			}
			if tensor.name == "output.weight" {
				value = float32(i / 32)
			}
			write(value)
		}
	}
	return fixture.Bytes()
}

func TestRawDecodeProductionLoadRejectsArtifactSwap(t *testing.T) {
	original := rawDecodeProductionArtifactFixture(t)
	digest := fmt.Sprintf("%x", sha256.Sum256(original))
	t.Run("split requires authenticated siblings", func(t *testing.T) {
		parsed, err := ggufload.Read(bytes.NewReader(original))
		if err != nil {
			t.Fatal(err)
		}
		entry := binary.LittleEndian.AppendUint64(nil, uint64(len("split.count")))
		entry = append(entry, "split.count"...)
		entry = binary.LittleEndian.AppendUint32(entry, uint32(ggufload.TypeUint32))
		entry = binary.LittleEndian.AppendUint32(entry, 2)
		split := slices.Clone(original[:24])
		binary.LittleEndian.PutUint64(split[16:24], binary.LittleEndian.Uint64(split[16:24])+1)
		split = append(split, entry...)
		split = append(split, original[24:parsed.TensorDataOffset]...)
		for len(split)%32 != 0 {
			split = append(split, 0)
		}
		split = append(split, original[parsed.TensorDataOffset:]...)
		path := filepath.Join(t.TempDir(), "model-00001-of-00002.gguf")
		if err := os.WriteFile(path, split, 0o600); err != nil {
			t.Fatal(err)
		}
		loaded, _, err := loadProductionModel(context.Background(), Request{
			ArtifactPath: path, ExpectedArtifactSHA256: fmt.Sprintf("%x", sha256.Sum256(split)),
		})
		if err == nil || loaded != nil || !strings.Contains(err.Error(), "digest-bound shard manifest") {
			t.Fatalf("unverified sibling shard admitted: model=%v err=%v", loaded, err)
		}
	})
	for _, mmap := range []string{"0", "1"} {
		for _, changed := range []bool{false, true} {
			t.Run(fmt.Sprintf("mmap_%s_changed_%t", mmap, changed), func(t *testing.T) {
				t.Setenv("FAK_GGUF_MMAP", mmap)
				path := filepath.Join(t.TempDir(), "model.gguf")
				replacement := slices.Clone(original)
				if changed {
					binary.LittleEndian.PutUint32(replacement[len(replacement)-4:], math.Float32bits(1))
				}
				if err := os.WriteFile(path, original, 0o600); err != nil {
					t.Fatal(err)
				}
				d := defaultDependencies()
				d.inspectArtifact = func(path string, open func(string) (io.ReadCloser, error)) (artifactObservation, error) {
					observed, err := inspectGGUFArtifact(path, open)
					if err != nil {
						return observed, err
					}
					if err := os.Rename(path, path+".inspected"); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, replacement, 0o600); err != nil {
						t.Fatal(err)
					}
					return observed, nil
				}
				got, err := d.execute(context.Background(), Request{
					ArtifactPath: path, ExpectedArtifactSHA256: digest,
					PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1,
				})
				if changed {
					if err == nil || !strings.Contains(err.Error(), "SHA-256 mismatch") || len(got.Runs) != 0 {
						t.Fatalf("different loaded weight bytes gained authority: runs=%d err=%v", len(got.Runs), err)
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				observed, ok := got.OutputTextObservation(0)
				if text, textOK := observed.Text(); !ok || !textOK || text != "ab" {
					t.Fatalf("identical replacement lost real decoded output: text=%q available=%t/%t", text, ok, textOK)
				}
			})
		}
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

func TestRawDecodeSelectedTokenLogprobNumerics(t *testing.T) {
	base := []float32{10000, 9999, -10000}
	shifted := []float32{11024, 11023, -8976}
	baseToken, _, _, baseLogprob, err := greedySelection(base)
	if err != nil {
		t.Fatal(err)
	}
	shiftedToken, _, _, shiftedLogprob, err := greedySelection(shifted)
	if err != nil {
		t.Fatal(err)
	}
	want := -math.Log(1 + math.Exp(-1) + math.Exp(-20000))
	if baseToken != 0 || shiftedToken != 0 || math.Abs(baseLogprob-want) > 1e-12 {
		t.Fatalf("huge-logit selection = token %d/%d logprob %.17g, want token 0/0 logprob %.17g", baseToken, shiftedToken, baseLogprob, want)
	}
	if math.Abs(baseLogprob-shiftedLogprob) > 1e-12 {
		t.Fatalf("common shift changed selected-token logprob: base=%.17g shifted=%.17g", baseLogprob, shiftedLogprob)
	}
	token, top1, top2, singletonLogprob, err := greedySelection([]float32{10000})
	if err != nil {
		t.Fatal(err)
	}
	if token != 0 || top1 != 10000 || top2 != -float32(math.MaxFloat32) || singletonLogprob != 0 || math.Signbit(singletonLogprob) {
		t.Fatalf("singleton selection = (%d, %v, %v, %v signbit=%v), want normalized +0", token, top1, top2, singletonLogprob, math.Signbit(singletonLogprob))
	}
}

func selectedTokenLogprobBindingFixture(t *testing.T) (Execution, compute.Qwen38VulkanRawDecodeResult) {
	t.Helper()
	outputs := [][]float32{{10000, 9999, -10000}, {0, 4, 1}, {3, 2, 1}}
	m := &fakeLoadedModel{candidate: &fakeSession{outputs: outputs}}
	req := Request{PromptTokenIDs: []int{1}, ContextLimit: 8, GeneratedTokenLimit: len(outputs), Repetitions: 1}
	execution, err := executeLoaded(context.Background(), req, m, nil, nil, fakeClock(), time.Since)
	if err != nil {
		t.Fatalf("execute loaded: %v", err)
	}
	ignoreEOS, eosStopped := false, false
	raw := compute.Qwen38VulkanRawDecodeResult{
		PromptTokenIDs:      []int32{1},
		GeneratedTokenLimit: 3,
		OutputTokenIDs:      []int32{0, 1, 0},
		Runs: []compute.Qwen38VulkanDecodeRun{{
			Repetition:            1,
			GeneratedTokenLimit:   3,
			ActualGeneratedTokens: 3,
			IgnoreEOS:             &ignoreEOS,
			EOSStopped:            &eosStopped,
			OutputTokenIDs:        []int32{0, 1, 0},
		}},
	}
	return execution, raw
}

func TestBindQwen38VulkanSelectedTokenLogprobsV3IncludesPrefillAndClones(t *testing.T) {
	execution, raw := selectedTokenLogprobBindingFixture(t)
	bound, err := BindQwen38VulkanSelectedTokenLogprobsV3(execution, raw)
	if err != nil {
		t.Fatal(err)
	}
	logprobs := bound.Runs[0].SelectedTokenLogprobs
	if len(logprobs) != len(execution.Runs[0].GeneratedTokens) || len(logprobs) != 3 {
		t.Fatalf("bound logprobs = %v, want one per output token", logprobs)
	}
	wantPrefill := -math.Log(1 + math.Exp(-1) + math.Exp(-20000))
	if math.Abs(logprobs[0]-wantPrefill) > 1e-12 {
		t.Fatalf("prefill-selected logprob = %.17g, want %.17g", logprobs[0], wantPrefill)
	}
	for i, logprob := range logprobs {
		if math.IsNaN(logprob) || math.IsInf(logprob, 0) || logprob > 0 {
			t.Fatalf("selected-token logprob[%d] = %v, want finite and non-positive", i, logprob)
		}
	}
	if bound.Runs[0].SelectedTokenLogprobsSHA256 != "" || raw.Runs[0].SelectedTokenLogprobs != nil {
		t.Fatalf("binder derived digest or mutated caller raw result: bound=%+v raw=%+v", bound.Runs[0], raw.Runs[0])
	}
	bound.Runs[0].SelectedTokenLogprobs[0] = 0
	rebound, err := BindQwen38VulkanSelectedTokenLogprobsV3(execution, raw)
	if err != nil || rebound.Runs[0].SelectedTokenLogprobs[0] != wantPrefill {
		t.Fatalf("bound slice aliased sealed evidence: rebound=%v err=%v", rebound.Runs[0].SelectedTokenLogprobs, err)
	}
}

func TestBindQwen38VulkanSelectedTokenLogprobsV3RefusesMutations(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Execution, *compute.Qwen38VulkanRawDecodeResult)
	}{
		{name: "caller execution", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) {
			*e = Execution{PromptTokenIDs: []int{1}, GeneratedLimit: 3}
		}},
		{name: "execution prompt", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.PromptTokenIDs[0] = 2 }},
		{name: "execution limit", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.GeneratedLimit++ }},
		{name: "execution generated token", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].GeneratedTokens[0] = 2 }},
		{name: "execution prefill token", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].PrefillOutputID = 2 }},
		{name: "execution step token", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].Steps[0].TokenID = 2 }},
		{name: "execution step order", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].Steps[0].Step = 1 }},
		{name: "execution step logit", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].Steps[0].Top1++ }},
		{name: "execution step margin", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].Steps[0].Margin++ }},
		{name: "execution step tokens", mutate: func(e *Execution, _ *compute.Qwen38VulkanRawDecodeResult) { e.Runs[0].StepTokens[0] = 2 }},
		{name: "raw prompt", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.PromptTokenIDs[0] = 2 }},
		{name: "raw limit", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.GeneratedTokenLimit++ }},
		{name: "raw output", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.OutputTokenIDs[0] = 2 }},
		{name: "raw repetitions", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.Runs = nil }},
		{name: "raw reported repetitions", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.ReportedRuns = 2 }},
		{name: "raw repetition index", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.Runs[0].Repetition = 2 }},
		{name: "raw run limit", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.Runs[0].GeneratedTokenLimit++ }},
		{name: "raw actual count", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.Runs[0].ActualGeneratedTokens-- }},
		{name: "raw run output", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) { r.Runs[0].OutputTokenIDs[0] = 2 }},
		{name: "caller logprobs", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) {
			r.Runs[0].SelectedTokenLogprobs = []float64{-1}
		}},
		{name: "caller digest", mutate: func(_ *Execution, r *compute.Qwen38VulkanRawDecodeResult) {
			r.Runs[0].SelectedTokenLogprobsSHA256 = "sha256:caller"
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			execution, raw := selectedTokenLogprobBindingFixture(t)
			tt.mutate(&execution, &raw)
			got, err := BindQwen38VulkanSelectedTokenLogprobsV3(execution, raw)
			if err == nil {
				t.Fatal("mutation was accepted")
			}
			if !reflect.DeepEqual(got, compute.Qwen38VulkanRawDecodeResult{}) {
				t.Fatalf("error returned non-zeroable raw result: %+v", got)
			}
		})
	}
}

func TestBindQwen38VulkanSelectedTokenLogprobsV3RefusesPartialCPUVerificationRun(t *testing.T) {
	m := &fakeLoadedModel{
		candidate: &fakeSession{outputs: [][]float32{{0, 3, 1}}},
		cpu:       &fakeSession{outputs: [][]float32{{4, 0, 1}}},
	}
	req := Request{PromptTokenIDs: []int{1}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 2, VerifyCPU: true}
	execution, executeErr := executeLoaded(context.Background(), req, m, nil, nil, fakeClock(), time.Since)
	if executeErr == nil || len(execution.Runs) != 1 {
		t.Fatalf("partial CPU-verification execution = runs %d err %v, want one of two runs plus divergence", len(execution.Runs), executeErr)
	}
	raw := compute.Qwen38VulkanRawDecodeResult{
		PromptTokenIDs:      []int32{1},
		GeneratedTokenLimit: 1,
		OutputTokenIDs:      []int32{1},
		Runs: []compute.Qwen38VulkanDecodeRun{{
			Repetition:            1,
			GeneratedTokenLimit:   1,
			ActualGeneratedTokens: 1,
			OutputTokenIDs:        []int32{1},
		}},
	}
	got, err := BindQwen38VulkanSelectedTokenLogprobsV3(execution, raw)
	if err == nil || !reflect.DeepEqual(got, compute.Qwen38VulkanRawDecodeResult{}) {
		t.Fatalf("partial repetition set bound: got=%+v err=%v", got, err)
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
	before := compute.BackendExecutionSnapshot{
		Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 100, H2DBytes: 1000, H2DCount: 10},
		TransferCountersObserved: true, DeviceAllocationObserved: true, DeviceAllocationLiveBytes: 128,
	}
	after := before
	after.Counters.ComputeDispatches = 107
	after.Counters.H2DBytes = 1064
	after.Counters.H2DCount = 11
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

func TestRawDecodeTimingObservationIsRunnerSealedAndResourceWindowBound(t *testing.T) {
	artifact := "timing observation artifact"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	identity := compute.BackendRuntimeIdentity{Backend: "vulkan", Device: "device", Driver: "driver", Runtime: "runtime"}
	complete := compute.BackendExecutionObservation{
		Identity:                  identity,
		Counters:                  compute.BackendCounterSnapshot{ComputeDispatches: 1, H2DBytes: 64, H2DCount: 1},
		DeviceMemoryTotalBytes:    1024,
		DeviceMemoryFreeBytes:     512,
		DeviceMemoryObserved:      true,
		TransferCountersObserved:  true,
		DeviceAllocationLiveBytes: 128,
		DeviceAllocationPeakBytes: 256,
		DeviceAllocationObserved:  true,
	}
	request := Request{
		ArtifactPath: "model.gguf", ExpectedArtifactSHA256: digest, ModelName: "timed-model",
		BackendName: "vulkan", PromptTokenIDs: []int{0}, ContextLimit: 3,
		GeneratedTokenLimit: 1, Repetitions: 2,
	}
	newModelConfig := func() model.Config {
		truncate := true
		return model.Config{
			VocabSize: 3, EOSTokenID: -1,
			LongRope: &model.RopeScaling{Factor: 2},
			RopeParameters: model.RopeParameters{
				"full_attention": model.RopeScaling{Truncate: &truncate},
			},
		}
	}
	newExecutionWithConfig := func(backend compute.Backend, now func() time.Time, candidateHook func(), config *model.Config) (Execution, error) {
		m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 2, 1}}}, config: config}
		m.candidateHook = candidateHook
		d := dependencies{
			openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(artifact)), nil },
			inspectArtifact: func(path string, _ func(string) (io.ReadCloser, error)) (artifactObservation, error) {
				return artifactObservation{Path: path, SHA256: digest, TensorInventorySHA256: "tensor", TokenizerSHA256: "tokenizer", TemplateSHA256: "template", Quantization: "F32"}, nil
			},
			loadModel: func(context.Context, Request) (loadedModel, string, error) { return m, "timed-model", nil },
			resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
				return backend, BackendObservation{Selected: backend.Name()}, nil
			},
			now: now,
		}
		return d.execute(context.Background(), request)
	}
	newExecution := func(backend compute.Backend, now func() time.Time, candidateHook func()) (Execution, error) {
		config := newModelConfig()
		return newExecutionWithConfig(backend, now, candidateHook, &config)
	}

	events := []string{}
	backend := &rawWindowBackend{
		rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"},
		observations:          []compute.BackendExecutionObservation{complete, complete},
		events:                &events,
	}
	execution, err := newExecution(backend, fakeClock(), func() { events = append(events, "candidate") })
	if err != nil {
		t.Fatal(err)
	}
	if backend.beginCalls != 2 || backend.endCalls != 2 || fmt.Sprint(events) != "[begin candidate end begin candidate end]" {
		t.Fatalf("resource windows begin=%d end=%d events=%v", backend.beginCalls, backend.endCalls, events)
	}
	hookConfig := newModelConfig()
	hookConfig.EnableResidualHook = true
	hookModel := &model.Model{Cfg: hookConfig}
	hookModel.SetResidualHook(func(int, []float32) {})
	hookBackend := &rawWindowBackend{
		rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"},
		observations:          []compute.BackendExecutionObservation{complete, complete},
	}
	hookExecution, err := newExecutionWithConfig(hookBackend, fakeClock(), nil, &hookModel.Cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := hookExecution.GenerationObservation(0); ok {
		t.Fatal("enabled residual hook gained generation authority")
	}
	if _, ok := hookExecution.TimingObservation(0); ok {
		t.Fatal("enabled residual hook gained timing authority")
	}
	for rep := range execution.Runs {
		observation, ok := execution.TimingObservation(rep)
		if !ok {
			t.Fatalf("sealed timing observation %d is unavailable", rep)
		}
		boundaries := observation.boundaries
		for i, boundary := range boundaries {
			want := time.Unix(100, 0).Add(time.Duration(3+rep*6+i) * time.Millisecond)
			if boundary != want {
				t.Fatalf("timing boundary %d = %v, want %v", i, boundary, want)
			}
			if i > 0 && boundary.Before(boundaries[i-1]) {
				t.Fatalf("timing boundary %d regressed: %v before %v", i, boundary, boundaries[i-1])
			}
		}
		for name, duration := range map[string]time.Duration{
			"setup": observation.SessionSetupDuration(), "prefill": observation.PrefillDuration(),
			"sample": observation.FirstSampleDuration(), "decode": observation.DecodeDuration(),
			"teardown": observation.TeardownDuration(),
		} {
			if duration != time.Millisecond {
				t.Fatalf("%s duration = %v, want 1ms", name, duration)
			}
		}
		if got, ok := observation.BackendExecution(); !ok || got != complete {
			t.Fatalf("backend observation = %+v, %v", got, ok)
		}
	}

	for name, mutate := range map[string]func(*Run){
		"setup":    func(run *Run) { run.SessionSetupDuration++ },
		"prefill":  func(run *Run) { run.PrefillDuration++ },
		"sample":   func(run *Run) { run.FirstSampleDuration++ },
		"decode":   func(run *Run) { run.DecodeDuration++ },
		"teardown": func(run *Run) { run.TeardownDuration++ },
	} {
		t.Run("duration "+name, func(t *testing.T) {
			mutated := execution
			mutated.Runs = slices.Clone(execution.Runs)
			mutate(&mutated.Runs[0])
			if _, ok := mutated.TimingObservation(0); ok {
				t.Fatal("public duration mutation retained timing authority")
			}
		})
	}
	for name, mutate := range map[string]func(*compute.BackendExecutionObservation){
		"identity backend":      func(got *compute.BackendExecutionObservation) { got.Identity.Backend = "other" },
		"identity device":       func(got *compute.BackendExecutionObservation) { got.Identity.Device = "other" },
		"identity driver":       func(got *compute.BackendExecutionObservation) { got.Identity.Driver = "other" },
		"identity runtime":      func(got *compute.BackendExecutionObservation) { got.Identity.Runtime = "other" },
		"counter":               func(got *compute.BackendExecutionObservation) { got.Counters.ComputeDispatches++ },
		"transfer availability": func(got *compute.BackendExecutionObservation) { got.TransferCountersObserved = false },
		"memory availability":   func(got *compute.BackendExecutionObservation) { got.DeviceMemoryObserved = false },
		"memory total":          func(got *compute.BackendExecutionObservation) { got.DeviceMemoryTotalBytes++ },
		"memory free":           func(got *compute.BackendExecutionObservation) { got.DeviceMemoryFreeBytes++ },
		"allocation availability": func(got *compute.BackendExecutionObservation) {
			got.DeviceAllocationObserved = false
		},
		"allocation live": func(got *compute.BackendExecutionObservation) { got.DeviceAllocationLiveBytes++ },
		"allocation peak": func(got *compute.BackendExecutionObservation) { got.DeviceAllocationPeakBytes++ },
	} {
		t.Run("backend "+name, func(t *testing.T) {
			mutated := execution
			mutated.Runs = slices.Clone(execution.Runs)
			forged := *mutated.Runs[0].BackendExecution
			mutate(&forged)
			mutated.Runs[0].BackendExecution = &forged
			if _, ok := mutated.TimingObservation(0); ok {
				t.Fatal("public backend mutation retained timing authority")
			}
		})
	}
	permuted := execution
	permuted.Runs = slices.Clone(execution.Runs)
	permuted.Runs[0], permuted.Runs[1] = permuted.Runs[1], permuted.Runs[0]
	if _, ok := permuted.TimingObservation(0); ok {
		t.Fatal("permuted repetition retained timing authority")
	}
	otherBackend := &rawWindowBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{complete, complete}}
	other, err := newExecution(otherBackend, fakeClock(), nil)
	if err != nil {
		t.Fatal(err)
	}
	transplanted := execution
	transplanted.Runs = slices.Clone(execution.Runs)
	transplanted.Runs[0] = other.Runs[0]
	if _, ok := transplanted.TimingObservation(0); ok {
		t.Fatal("cross-execution transplant retained timing authority")
	}
	bindingMutations := map[string]func(*Execution){
		"prompt": func(got *Execution) {
			got.PromptTokenIDs = slices.Clone(got.PromptTokenIDs)
			got.PromptTokenIDs[0]++
		},
		"context limit":    func(got *Execution) { got.ContextLimit++ },
		"generated limit":  func(got *Execution) { got.GeneratedLimit++ },
		"ignore EOS":       func(got *Execution) { got.IgnoreEOS = !got.IgnoreEOS },
		"artifact path":    func(got *Execution) { got.ArtifactPath += ".other" },
		"artifact digest":  func(got *Execution) { got.ArtifactSHA256 = "other" },
		"tensor inventory": func(got *Execution) { got.TensorInventorySHA256 = "other" },
		"tokenizer":        func(got *Execution) { got.TokenizerSHA256 = "other" },
		"template":         func(got *Execution) { got.TemplateSHA256 = "other" },
		"quantization":     func(got *Execution) { got.Quantization = "other" },
		"model name":       func(got *Execution) { got.ModelName = "different-model" },
		"model config":     func(got *Execution) { got.ModelConfig.VocabSize++ },
		"backend selection": func(got *Execution) {
			got.Backend.Selected = "other"
		},
	}
	for name, mutate := range bindingMutations {
		t.Run("binding "+name, func(t *testing.T) {
			drifted := execution
			mutate(&drifted)
			if _, ok := drifted.TimingObservation(0); ok {
				t.Fatal("execution binding drift retained timing authority")
			}
		})
	}
	assertNestedMutationRejected := func(t *testing.T, mutated Execution) {
		t.Helper()
		if _, ok := mutated.GenerationObservation(0); ok {
			t.Fatal("nested model configuration mutation retained generation authority")
		}
		if _, ok := mutated.TimingObservation(0); ok {
			t.Fatal("nested model configuration mutation retained timing authority")
		}
	}
	t.Run("nested binding longrope factor", func(t *testing.T) {
		backend := &rawWindowBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{complete, complete}}
		mutated, err := newExecution(backend, fakeClock(), nil)
		if err != nil {
			t.Fatal(err)
		}
		longRope := mutated.ModelConfig.LongRope
		mutated.ModelConfig.LongRope.Factor++
		if mutated.ModelConfig.LongRope != longRope {
			t.Fatal("LongRope mutation replaced rather than mutated the existing pointee")
		}
		assertNestedMutationRejected(t, mutated)
	})
	t.Run("nested binding rope parameter truncate", func(t *testing.T) {
		backend := &rawWindowBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{complete, complete}}
		mutated, err := newExecution(backend, fakeClock(), nil)
		if err != nil {
			t.Fatal(err)
		}
		rope := mutated.ModelConfig.RopeParameters["full_attention"]
		truncate := rope.Truncate
		*rope.Truncate = !*rope.Truncate
		if mutated.ModelConfig.RopeParameters["full_attention"].Truncate != truncate {
			t.Fatal("RopeParameters truncate mutation replaced rather than mutated the existing pointee")
		}
		assertNestedMutationRejected(t, mutated)
	})
	for name, mutate := range map[string]func(*Run){
		"generated tokens": func(run *Run) {
			run.GeneratedTokens = slices.Clone(run.GeneratedTokens)
			run.GeneratedTokens[0]++
		},
		"prefill output": func(run *Run) { run.PrefillOutputID++ },
		"steps": func(run *Run) {
			run.Steps = slices.Clone(run.Steps)
			run.Steps[0].TokenID++
		},
		"EOS state": func(run *Run) { run.EOSStopped = !run.EOSStopped },
	} {
		t.Run("generation "+name, func(t *testing.T) {
			mutated := execution
			mutated.Runs = slices.Clone(execution.Runs)
			mutate(&mutated.Runs[0])
			if _, ok := mutated.TimingObservation(0); ok {
				t.Fatal("generation alias mutation retained timing authority")
			}
		})
	}
	cardinality := execution
	cardinality.Runs = slices.Clone(execution.Runs[:1])
	if _, ok := cardinality.TimingObservation(0); ok {
		t.Fatal("repetition cardinality mismatch retained timing authority")
	}
	constructed := execution
	constructed.Runs = slices.Clone(execution.Runs)
	constructed.Runs[0] = Run{
		SessionSetupDuration: time.Millisecond, PrefillDuration: time.Millisecond,
		FirstSampleDuration: time.Millisecond, DecodeDuration: time.Millisecond,
		TeardownDuration: time.Millisecond, BackendExecution: &complete,
	}
	if _, ok := constructed.TimingObservation(0); ok {
		t.Fatal("caller-constructed run gained timing authority")
	}

	malformedObservations := map[string]func() compute.BackendExecutionObservation{
		"identity": func() compute.BackendExecutionObservation {
			got := complete
			got.Identity.Device = ""
			return got
		},
		"wrong backend": func() compute.BackendExecutionObservation {
			got := complete
			got.Identity.Backend = "other"
			return got
		},
		"transfer unavailable": func() compute.BackendExecutionObservation {
			got := complete
			got.TransferCountersObserved = false
			return got
		},
		"transfer pair": func() compute.BackendExecutionObservation {
			got := complete
			got.Counters.H2DCount = 0
			return got
		},
		"allocation unavailable": func() compute.BackendExecutionObservation {
			got := complete
			got.DeviceAllocationObserved = false
			return got
		},
		"allocation peak": func() compute.BackendExecutionObservation {
			got := complete
			got.DeviceAllocationPeakBytes = got.DeviceAllocationLiveBytes - 1
			return got
		},
		"memory total": func() compute.BackendExecutionObservation {
			got := complete
			got.DeviceMemoryTotalBytes = 0
			return got
		},
		"memory free": func() compute.BackendExecutionObservation {
			got := complete
			got.DeviceMemoryFreeBytes = got.DeviceMemoryTotalBytes + 1
			return got
		},
	}
	for name, makeObservation := range malformedObservations {
		t.Run("malformed "+name, func(t *testing.T) {
			backend := &rawWindowBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{makeObservation()}}
			got, err := newExecution(backend, fakeClock(), nil)
			if err == nil || len(got.Runs) != 0 || backend.beginCalls != 1 || backend.endCalls != 1 {
				t.Fatalf("malformed observation published a repetition: got=%+v err=%v begin=%d end=%d", got, err, backend.beginCalls, backend.endCalls)
			}
		})
	}

	for name, backend := range map[string]*rawWindowBackend{
		"begin error": {rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, beginErr: fmt.Errorf("begin failed")},
		"nil window":  {rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, nilWindow: true},
		"end error":   {rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{complete}, endErr: fmt.Errorf("end failed")},
	} {
		t.Run(name, func(t *testing.T) {
			events := []string{}
			backend.events = &events
			got, err := newExecution(backend, fakeClock(), func() { events = append(events, "candidate") })
			if err == nil || len(got.Runs) != 0 || backend.beginCalls != 1 {
				t.Fatalf("window failure published a repetition: got=%+v err=%v begin=%d end=%d events=%v", got, err, backend.beginCalls, backend.endCalls, events)
			}
			if name == "end error" && (backend.endCalls != 1 || fmt.Sprint(events) != "[begin candidate end]") {
				t.Fatalf("end failure lifecycle: begin=%d end=%d events=%v", backend.beginCalls, backend.endCalls, events)
			}
			if name != "end error" && (backend.endCalls != 0 || fmt.Sprint(events) != "[begin]") {
				t.Fatalf("begin failure lifecycle: begin=%d end=%d events=%v", backend.beginCalls, backend.endCalls, events)
			}
		})
	}

	unsupported := &rawUnsupportedBackend{name: "vulkan"}
	if got, err := newExecution(unsupported, fakeClock(), nil); err != nil || len(got.Runs) != request.Repetitions {
		t.Fatalf("unsupported observation changed execution: runs=%d err=%v", len(got.Runs), err)
	} else if _, ok := got.TimingObservation(0); ok || got.Runs[0].BackendExecution != nil {
		t.Fatal("unsupported backend gained timing/resource authority")
	}
	snapshotOnly := &rawSnapshotOnlyBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, snapshot: compute.BackendExecutionSnapshot{Identity: identity}}
	if got, err := newExecution(snapshotOnly, fakeClock(), nil); err != nil || len(got.Runs) != request.Repetitions || snapshotOnly.calls != 0 {
		t.Fatalf("snapshot fallback was used: runs=%d calls=%d err=%v", len(got.Runs), snapshotOnly.calls, err)
	} else if _, ok := got.TimingObservation(0); ok || got.Runs[0].BackendExecution != nil {
		t.Fatal("snapshot-only backend gained timing/resource authority")
	}

	for transition := 1; transition < 6; transition++ {
		t.Run(fmt.Sprintf("clock regression %d", transition), func(t *testing.T) {
			clockCalls := 0
			regressingClock := func() time.Time {
				clockCalls++
				if clockCalls == 3+transition {
					return time.Unix(100, 0).Add(time.Duration(clockCalls-2) * time.Millisecond)
				}
				return time.Unix(100, 0).Add(time.Duration(clockCalls) * time.Millisecond)
			}
			backend := &rawWindowBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{complete}}
			got, err := newExecution(backend, regressingClock, nil)
			if err == nil || len(got.Runs) != 0 || backend.beginCalls != 1 || backend.endCalls != 1 {
				t.Fatalf("clock regression published aliases: got=%+v err=%v begin=%d end=%d", got, err, backend.beginCalls, backend.endCalls)
			}
		})
	}

	errorBackend := &rawWindowBackend{rawUnsupportedBackend: rawUnsupportedBackend{name: "vulkan"}, observations: []compute.BackendExecutionObservation{complete}}
	errorModel := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{float32(math.NaN())}}}}
	_, executeErr := executeLoaded(context.Background(), Request{PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}, errorModel, errorBackend, nil, fakeClock(), time.Since)
	if executeErr == nil || errorBackend.beginCalls != 1 || errorBackend.endCalls != 1 {
		t.Fatalf("failed repetition did not close exactly one window: err=%v begin=%d end=%d", executeErr, errorBackend.beginCalls, errorBackend.endCalls)
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
			{Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 2}, TransferCountersObserved: true, DeviceAllocationObserved: true},
			{Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 1}, TransferCountersObserved: true, DeviceAllocationObserved: true},
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

func TestRawDecodeHostEnvironmentIsPerRunInjectedAndCrossBound(t *testing.T) {
	artifact := "host environment artifact"
	digest := fmt.Sprintf("%x", sha256.Sum256([]byte(artifact)))
	identity := compute.BackendRuntimeIdentity{
		Backend: "vulkan",
		Device:  "AMD Radeon 8060S Graphics",
		Driver:  "driver=mesa-26.1 vendor=0x1002 device=0x1586",
		Runtime: "vulkan-1.4.0",
	}
	host := compute.VulkanHostEnvironment{
		OS: "linux", Arch: "amd64", Kernel: "6.14.0", Device: identity.Device,
		VendorID: "0x1002", DeviceID: "0x1586", MesaDriver: "radv",
		MesaVersion: "Mesa 26.1.0", Firmware: "vbios-observed",
	}
	newExecution := func(observe func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error)) (Execution, error) {
		before := compute.BackendExecutionSnapshot{
			Identity: identity, Counters: compute.BackendCounterSnapshot{ComputeDispatches: 10},
			TransferCountersObserved: true, DeviceAllocationObserved: true,
		}
		after := before
		after.Counters.ComputeDispatches = 11
		backend := &rawObservedBackend{name: "vulkan", snapshots: []compute.BackendExecutionSnapshot{before, after}}
		m := &fakeLoadedModel{candidate: &fakeSession{outputs: [][]float32{{0, 2, 1}}}}
		d := dependencies{
			openArtifact: func(string) (io.ReadCloser, error) { return io.NopCloser(strings.NewReader(artifact)), nil },
			loadModel:    func(context.Context, Request) (loadedModel, string, error) { return m, "observed-model", nil },
			resolveBackend: func(Request) (compute.Backend, BackendObservation, error) {
				return backend, BackendObservation{Selected: backend.Name()}, nil
			},
			observeHost: observe,
			now:         fakeClock(),
		}
		req := Request{ArtifactPath: "model.gguf", ExpectedArtifactSHA256: digest, ModelName: "observed-model", BackendName: "vulkan", PromptTokenIDs: []int{0}, ContextLimit: 2, GeneratedTokenLimit: 1, Repetitions: 1}
		return d.execute(context.Background(), req)
	}

	observerCalls := 0
	exec, err := newExecution(func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error) {
		observerCalls++
		return host, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if observerCalls != 1 || len(exec.Runs) != 1 || exec.Runs[0].HostEnvironment == nil || *exec.Runs[0].HostEnvironment != host {
		t.Fatalf("per-run host observation calls=%d execution=%+v", observerCalls, exec)
	}

	for name, observe := range map[string]func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error){
		"missing": func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error) {
			return compute.VulkanHostEnvironment{}, fmt.Errorf("unavailable")
		},
		"wrong device": func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error) {
			candidate := host
			candidate.Device = "different"
			return candidate, nil
		},
		"wrong PCI": func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error) {
			candidate := host
			candidate.DeviceID = "0x9999"
			return candidate, nil
		},
		"not RADV": func(context.Context, compute.Backend) (compute.VulkanHostEnvironment, error) {
			candidate := host
			candidate.MesaDriver = "opaque-driver"
			return candidate, nil
		},
	} {
		t.Run(name, func(t *testing.T) {
			exec, err := newExecution(observe)
			if err != nil {
				t.Fatal(err)
			}
			if len(exec.Runs) != 1 || exec.Runs[0].HostEnvironment != nil {
				t.Fatalf("invalid host evidence was retained: %+v", exec.Runs)
			}
		})
	}
}
