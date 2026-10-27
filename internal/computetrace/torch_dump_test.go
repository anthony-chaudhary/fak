package computetrace

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const torchFixture = "testdata/sglang-dumper-tiny"

// fak-test:runtime fast est=500ms lane=default
func TestTorchDumpStreamsLargeAggregate(t *testing.T) {
	const layerCount = 10
	promptTokens := []int{1, 2}
	dir := t.TempDir()
	source := filepath.Join(torchFixture, "step=0___rank=0___dump_index=1___name=non_intrusive__model.layers.0.output.pt")
	expanded := torchRewriteFile(t, source, func(name string, b []byte) []byte {
		if strings.HasSuffix(name, "/data.pkl") {
			b = bytes.ReplaceAll(b, []byte{'K', 16}, []byte{'M', 0, 128})
			b = bytes.ReplaceAll(b, []byte{'K', 32}, []byte{'J', 0, 0, 1, 0})
		}
		if strings.Contains(name, "/data/") {
			b = make([]byte, 65536*4)
			for i := 0; i < 65536; i++ {
				binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(i)))
			}
		}
		return b
	})
	data, err := os.ReadFile(expanded)
	if err != nil {
		t.Fatal(err)
	}
	for rank := 0; rank < 2; rank++ {
		ids := fmt.Sprintf("step=0___rank=%d___dump_index=0___name=input_ids.pt", rank)
		raw, err := os.ReadFile(filepath.Join(torchFixture, ids))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, ids), raw, 0o600); err != nil {
			t.Fatal(err)
		}
		for layer := 0; layer < layerCount; layer++ {
			name := fmt.Sprintf("step=0___rank=%d___dump_index=%d___name=non_intrusive__model.layers.%d.output.pt", rank, layer+1, layer)
			if err = os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	result, err := ConvertTorchDump(dir, TorchDumpOptions{Engine: "sglang", Architecture: "LlamaForCausalLM", PromptTokens: promptTokens})
	if err != nil {
		t.Fatal(err)
	}
	expectedEvents := layerCount * len(promptTokens)
	if len(result.Artifact.Events) != expectedEvents || result.Artifact.RetainedSampleValues != uint64(expectedEvents)*LlamaCppActivationSamples {
		t.Fatalf("events=%d samples=%d", len(result.Artifact.Events), result.Artifact.RetainedSampleValues)
	}
	for _, event := range result.Artifact.Events {
		if len(event.Shapes) != 1 || len(event.Shapes[0]) != 1 || event.Shapes[0][0] != 32768 {
			t.Fatalf("large fixture width drifted: %v", event.Shapes)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestTorchDumpNonFiniteSamplesStayMarked(t *testing.T) {
	dir := torchCopyFixture(t)
	for rank := 0; rank < 2; rank++ {
		name := fmt.Sprintf("step=0___rank=%d___dump_index=1___name=non_intrusive__model.layers.0.output.pt", rank)
		path := filepath.Join(dir, name)
		variant := torchRewriteFile(t, path, func(name string, b []byte) []byte {
			if strings.Contains(name, "/data/") {
				binary.LittleEndian.PutUint32(b, 0x7fc00123)
			}
			return b
		})
		b, err := os.ReadFile(variant)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	result, err := ConvertTorchDump(dir, TorchDumpOptions{Engine: "sglang", Architecture: "LlamaForCausalLM", PromptTokens: []int{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Artifact.ClampNonFinite() != 1 {
		t.Fatal("expected one marked nonfinite sample")
	}
	for _, event := range result.Artifact.Events {
		if event.Layer == 0 && event.Phase == "residual_streams" && event.Token == 0 {
			if event.Status == "ok" {
				t.Fatalf("nonfinite sample lost its status: %v", event)
			}
		}
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestTorchDumpRoundTripAndStageCoverage(t *testing.T) {
	promptTokens := []int{1, 2}
	layers := []int{0, 1}
	stages := []string{"residual_streams", "attention_projected"}
	for _, engine := range []string{"sglang", "vllm"} {
		t.Run(engine, func(t *testing.T) {
			result, err := ConvertTorchDump(torchFixture, TorchDumpOptions{Engine: engine, Architecture: "LlamaForCausalLM", PromptTokens: promptTokens})
			if err != nil {
				t.Fatal(err)
			}
			if result.Step != 0 || len(result.Unmapped) != 0 {
				t.Fatalf("step=%d events=%d unmapped=%v", result.Step, len(result.Artifact.Events), result.Unmapped)
			}
			type eventKey struct {
				layer int
				phase string
				token int
			}
			expected := map[eventKey]bool{}
			for _, layer := range layers {
				for _, phase := range stages {
					for token := range promptTokens {
						expected[eventKey{layer, phase, token}] = true
					}
				}
			}
			for _, e := range result.Artifact.Events {
				key := eventKey{e.Layer, e.Phase, e.Token}
				if !expected[key] {
					t.Fatalf("unexpected or duplicate event: %+v", key)
				}
				delete(expected, key)
				if len(e.SampleValues) != LlamaCppActivationSamples {
					t.Fatalf("samples=%d", len(e.SampleValues))
				}
			}
			if len(expected) != 0 {
				t.Fatalf("missing fixture events: %v", expected)
			}
			ref := result.Artifact
			got := Artifact{Schema: Schema, InputTokens: ref.InputTokens}
			for _, e := range ref.Events {
				e.InputDigest = Digest("actual-input")
				e.WeightDigest = Digest("actual-weights")
				e.SampleValues = append([]float32(nil), e.SampleValues...)
				got.Events = append(got.Events, e)
			}
			if err = AdoptActivationDigests(&ref, got); err != nil {
				t.Fatal(err)
			}
			report := FirstDivergence(ref, got, testDivTol, ActivationTolerance{})
			if report.Verdict != TraceEqual || report.ParityLayers != 2 {
				t.Fatalf("equal report: %v", report)
			}
			perturb(&got, 1, "residual_streams", 1.5)
			report = FirstDivergence(ref, got, testDivTol, ActivationTolerance{})
			if report.Verdict != TraceDivergent || report.ParityLayers != 1 || report.FirstDivergent == nil || *report.FirstDivergent != (ActivationStage{Layer: 1, Stage: "residual_streams"}) {
				t.Fatalf("injected report: %v", report)
			}
		})
	}
	for _, engine := range []string{"sglang", "vllm"} {
		for _, arch := range []string{"DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"} {
			m, ok := TorchStageMapFor(engine, arch)
			if !ok {
				t.Fatal(engine, arch)
			}
			seen := map[string]bool{}
			for suffix := range m.Layered {
				stage, layer, reason := m.classify("non_intrusive__model.layers.7." + suffix)
				if layer != 7 || reason != "" {
					t.Fatal(stage, layer, reason)
				}
				seen[stage] = true
			}
			for _, stage := range []string{"q_latent", "kv_latent", "attention_projected", "moe_sum", "residual_streams"} {
				if !seen[stage] {
					t.Fatal(engine, arch, stage)
				}
			}
		}
	}
}

func torchCopyFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	entries, err := os.ReadDir(torchFixture)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".pt") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(torchFixture, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, e.Name()), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func torchEditStorage(t *testing.T, path string) {
	t.Helper()
	z, err := zip.OpenReader(path)
	if err != nil {
		t.Fatal(err)
	}
	var output strings.Builder
	w := zip.NewWriter(&output)
	for _, f := range z.File {
		r, err := f.Open()
		if err != nil {
			t.Fatal(err)
		}
		b, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(f.Name, "/data/") && len(b) > 0 {
			index := 0
			sampled := map[int]bool{}
			for _, n := range ActivationSampleIndices(16, 8) {
				sampled[n] = true
			}
			for sampled[index] {
				index++
			}
			b[index*4] ^= 1
		}
		o, err := w.Create(f.Name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = o.Write(b); err != nil {
			t.Fatal(err)
		}
	}
	z.Close()
	if err = w.Close(); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, []byte(output.String()), 0o600); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestTorchDumpClosedRefusals(t *testing.T) {
	base := TorchDumpOptions{Engine: "sglang", Architecture: "LlamaForCausalLM", PromptTokens: []int{1, 2}}
	t.Run("ambiguous", func(t *testing.T) {
		opts := base
		opts.PromptTokens = nil
		if _, err := ConvertTorchDump(torchFixture, opts); !errors.Is(err, ErrTorchDump) {
			t.Fatal(err)
		}
	})
	t.Run("no_match", func(t *testing.T) {
		opts := base
		opts.PromptTokens = []int{99}
		if _, err := ConvertTorchDump(torchFixture, opts); !errors.Is(err, ErrTorchDump) {
			t.Fatal(err)
		}
	})
	t.Run("explicit_step", func(t *testing.T) {
		step := 1
		opts := base
		opts.Step = &step
		opts.PromptTokens = nil
		result, err := ConvertTorchDump(torchFixture, opts)
		if err != nil || result.Step != 1 {
			t.Fatal(result.Step, err)
		}
	})
	t.Run("rank_disagree", func(t *testing.T) {
		dir := torchCopyFixture(t)
		paths, err := filepath.Glob(filepath.Join(dir, "step=0___rank=1*name=non_intrusive__model.layers.0.output.pt"))
		if err != nil || len(paths) != 1 {
			t.Fatal(paths, err)
		}
		torchEditStorage(t, paths[0])
		if _, err = ConvertTorchDump(dir, base); !errors.Is(err, ErrTorchRankDisagree) {
			t.Fatal(err)
		}
	})
	t.Run("missing_rank", func(t *testing.T) {
		dir := torchCopyFixture(t)
		paths, _ := filepath.Glob(filepath.Join(dir, "step=0___rank=1*name=non_intrusive__model.layers.0.output.pt"))
		if len(paths) != 1 {
			t.Fatal(paths)
		}
		if err := os.Remove(paths[0]); err != nil {
			t.Fatal(err)
		}
		if _, err := ConvertTorchDump(dir, base); !errors.Is(err, ErrTorchRankDisagree) {
			t.Fatal(err)
		}
	})
	t.Run("deferred_residual", func(t *testing.T) {
		dir := torchCopyFixture(t)
		paths, _ := filepath.Glob(filepath.Join(dir, "step=0___rank=0*name=non_intrusive__model.layers.0.self_attn.o_proj.output.pt"))
		if len(paths) != 1 {
			t.Fatal(paths)
		}
		b, err := os.ReadFile(paths[0])
		if err != nil {
			t.Fatal(err)
		}
		name := "step=0___rank=0___dump_index=99___name=non_intrusive__model.layers.0.output.1.pt"
		if err = os.WriteFile(filepath.Join(dir, name), b, 0o600); err != nil {
			t.Fatal(err)
		}
		opts := base
		opts.Architecture = "DeepseekV4ForCausalLM"
		if _, err = ConvertTorchDump(dir, opts); !errors.Is(err, ErrTorchDeferredResidual) {
			t.Fatal(err)
		}
	})
	t.Run("unsupported_layout", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "Pass00000.pt"), nil, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := ConvertTorchDump(dir, base); !errors.Is(err, ErrTorchUnsupportedLayout) {
			t.Fatal(err)
		}
	})
}
