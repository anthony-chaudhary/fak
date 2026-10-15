package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen4exp"
)

type traceStubFakeSource struct {
	cfg       model.Config
	configErr error
	tensors   map[string]ggufload.TensorInfo
	rowErr    map[string]error
	rowCalls  []string
	closes    int
	closeErr  error
}

func newTraceStubFakeSource() *traceStubFakeSource {
	return &traceStubFakeSource{
		cfg: model.Config{
			ModelType: "qwen4exp", NumLayers: qwen4exp.StubLayers,
			HiddenSize: qwen4exp.StubHiddenSize, VocabSize: qwen4exp.StubVocabSize,
			Qwen4Exp: &model.Qwen4ExpHParams{HCCount: qwen4exp.StubHCCount},
		},
		tensors: map[string]ggufload.TensorInfo{
			"token_embd.weight": {Name: "token_embd.weight", Dims: []uint64{qwen4exp.StubHiddenSize, qwen4exp.StubVocabSize}},
			"output.weight":     {Name: "output.weight", Dims: []uint64{qwen4exp.StubHiddenSize, qwen4exp.StubVocabSize}},
		},
		rowErr: map[string]error{},
	}
}

func (s *traceStubFakeSource) Config() (model.Config, error) { return s.cfg, s.configErr }
func (s *traceStubFakeSource) Tensor(name string) (ggufload.TensorInfo, bool) {
	info, ok := s.tensors[name]
	return info, ok
}
func (s *traceStubFakeSource) TensorRowF32(name string, row int) ([]float32, ggufload.TensorInfo, error) {
	s.rowCalls = append(s.rowCalls, fmt.Sprintf("%s:%d", name, row))
	info := s.tensors[name]
	if err := s.rowErr[name]; err != nil {
		return nil, info, err
	}
	values := make([]float32, qwen4exp.StubHiddenSize)
	for i := range values {
		values[i] = float32((row%17)+i) / qwen4exp.StubHiddenSize
	}
	return values, info, nil
}
func (s *traceStubFakeSource) Close() error {
	s.closes++
	return s.closeErr
}

func traceStubOpen(source *traceStubFakeSource, calls *int) func(string) (traceStubSource, error) {
	return func(string) (traceStubSource, error) {
		*calls++
		return source, nil
	}
}

// fak-test:runtime fast est=20ms
func TestModelTraceStubCommandEmitsBoundedDiagnosticArtifact(t *testing.T) {
	source := newTraceStubFakeSource()
	var stdout, stderr bytes.Buffer
	opens := 0
	code := runModelTraceStubWithSource(&stdout, &stderr, []string{"--gguf", "fixture.gguf", "--token-id", "7", "--token-id", "11"}, traceStubOpen(source, &opens))
	if code != 0 || stderr.Len() != 0 {
		t.Fatalf("exit=%d stderr=%q", code, stderr.String())
	}
	var receipt model.Qwen4ExpDiagnosticReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("decode receipt: %v\n%s", err, stdout.String())
	}
	if receipt.Mode != "diagnostic_stub" || receipt.Layers != 48 || receipt.Tokens != 2 {
		t.Fatalf("receipt identity=%+v", receipt)
	}
	identity := 0
	for _, event := range receipt.Artifact.Events {
		if event.Backend != "diagnostic-stub" || event.Device != "host" || event.Operation != "qwen4exp_diagnostic_stub" || event.Status != "stub_missing_native_layer_math" {
			t.Fatalf("event promoted beyond diagnostic stub: %+v", event)
		}
		if event.Phase == "residual_streams" {
			identity++
			if event.Candidate != "identity-replicated-hc4-missing-real-layer" {
				t.Fatalf("residual stage identity=%q", event.Candidate)
			}
		}
	}
	if identity != 2*48 || receipt.Artifact.Dropped != 0 || receipt.Artifact.DroppedSampleValues != 0 {
		t.Fatalf("identity stages=%d dropped=%d/%d", identity, receipt.Artifact.Dropped, receipt.Artifact.DroppedSampleValues)
	}
	outputRows := 0
	for _, call := range source.rowCalls {
		if strings.HasPrefix(call, "output.weight:") {
			outputRows++
		}
	}
	if opens != 1 || source.closes != 1 || len(source.rowCalls) != 2+2*32 || outputRows != 2*32 {
		t.Fatalf("opens=%d closes=%d row calls=%d output rows=%d", opens, source.closes, len(source.rowCalls), outputRows)
	}
}

// fak-test:runtime fast est=5ms
func TestModelTraceStubUsageRefusesBeforeOpeningSource(t *testing.T) {
	for _, args := range [][]string{
		{"--token-id", "1"},
		{"--gguf", "fixture.gguf"},
		{"--gguf", "fixture.gguf", "--token-id", "248320"},
		{"--gguf", "fixture.gguf", "--token-id", "1", "extra"},
	} {
		t.Run(strings.Join(args, "_"), func(t *testing.T) {
			opens := 0
			code := runModelTraceStubWithSource(io.Discard, io.Discard, args, func(string) (traceStubSource, error) {
				opens++
				return newTraceStubFakeSource(), nil
			})
			if code != 2 || opens != 0 {
				t.Fatalf("exit=%d opens=%d", code, opens)
			}
		})
	}
}

// fak-test:runtime fast est=5ms
func TestModelTraceStubRejectsArchitectureAndGeometryBeforeRows(t *testing.T) {
	for _, mutate := range []func(*traceStubFakeSource){
		func(s *traceStubFakeSource) { s.cfg.ModelType = "llama"; s.cfg.Qwen4Exp = nil },
		func(s *traceStubFakeSource) {
			info := s.tensors["output.weight"]
			info.Dims[0]--
			s.tensors["output.weight"] = info
		},
	} {
		source := newTraceStubFakeSource()
		mutate(source)
		var stdout bytes.Buffer
		opens := 0
		code := runModelTraceStubWithSource(&stdout, io.Discard, []string{"--gguf", "fixture.gguf", "--token-id", "1"}, traceStubOpen(source, &opens))
		if code != 1 || stdout.Len() != 0 || len(source.rowCalls) != 0 || source.closes != 1 {
			t.Fatalf("exit=%d stdout=%q rows=%v closes=%d", code, stdout.String(), source.rowCalls, source.closes)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestModelTraceStubSourceFailuresEmitNoReceipt(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*traceStubFakeSource)
	}{
		{name: "config read", mutate: func(s *traceStubFakeSource) { s.configErr = errors.New("config read failed") }},
		{name: "embedding row", mutate: func(s *traceStubFakeSource) { s.rowErr["token_embd.weight"] = errors.New("row read failed") }},
		{name: "output row", mutate: func(s *traceStubFakeSource) { s.rowErr["output.weight"] = errors.New("row read failed") }},
		{name: "source close", mutate: func(s *traceStubFakeSource) { s.closeErr = errors.New("close failed") }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			source := newTraceStubFakeSource()
			tc.mutate(source)
			var stdout, stderr bytes.Buffer
			opens := 0
			code := runModelTraceStubWithSource(&stdout, &stderr, []string{"--gguf", "fixture.gguf", "--token-id", "3"}, traceStubOpen(source, &opens))
			if code != 1 || stdout.Len() != 0 || stderr.Len() == 0 || opens != 1 || source.closes != 1 {
				t.Fatalf("exit=%d stdout=%q stderr=%q opens=%d closes=%d", code, stdout.String(), stderr.String(), opens, source.closes)
			}
		})
	}
}

// fak-test:runtime fast est=10ms
func TestModelTraceStubRefusesExistingOutput(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, []byte("owner-data"), 0o600); err != nil {
		t.Fatal(err)
	}
	source := newTraceStubFakeSource()
	var stdout bytes.Buffer
	opens := 0
	code := runModelTraceStubWithSource(&stdout, io.Discard, []string{"--gguf", "fixture.gguf", "--token-id", "5", "--out", path}, traceStubOpen(source, &opens))
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if code != 1 || stdout.Len() != 0 || string(got) != "owner-data" || source.closes != 1 {
		t.Fatalf("exit=%d stdout=%q file=%q closes=%d", code, stdout.String(), got, source.closes)
	}
}

type traceStubProbeWriter struct {
	err   error
	short bool
	calls int
}

func (w *traceStubProbeWriter) Write(p []byte) (int, error) {
	w.calls++
	if w.err != nil {
		return 0, w.err
	}
	if w.short && len(p) > 0 {
		return len(p) - 1, nil
	}
	return len(p), nil
}

// fak-test:runtime fast est=10ms
func TestModelTraceStubWriterFailurePropagates(t *testing.T) {
	for _, sink := range []*traceStubProbeWriter{
		{err: errors.New("sink failed")},
		{short: true},
	} {
		source := newTraceStubFakeSource()
		var stderr bytes.Buffer
		opens := 0
		code := runModelTraceStubWithSource(sink, &stderr, []string{"--gguf", "fixture.gguf", "--token-id", "9"}, traceStubOpen(source, &opens))
		if code != 1 || stderr.Len() == 0 || source.closes != 1 || sink.calls == 0 {
			t.Fatalf("exit=%d stderr=%q closes=%d sink calls=%d", code, stderr.String(), source.closes, sink.calls)
		}
	}
}
