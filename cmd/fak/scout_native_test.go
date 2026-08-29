package main

import (
	"context"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func TestNativeScoutPlannerUsesTypedConfigOverAmbientValues(t *testing.T) {
	t.Setenv("FAK_INKERNEL_QWEN_Q4K_PREFILL_CHUNK_TOKENS", "4096")
	t.Setenv("FAK_INKERNEL_QWEN35_METAL_GDN_SEQUENCE", "0")
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "0")
	want := agent.InKernelPlannerConfig{
		QwenQ4KPrefillChunkTokens: 1024,
		Qwen35MetalGDNSequence:    true,
		Q4KGateUpOutputSlab:       true,
	}
	p := newNativeScoutInKernelPlanner(model.NewSynthetic(model.Config{VocabSize: 8, HiddenSize: 4, NumLayers: 1}), nil, "typed-scout", false, nil, nativeControlConfig{Planner: want})
	if got := p.RuntimeConfig(); !reflect.DeepEqual(got, want) {
		t.Fatalf("scout planner config = %+v, want %+v", got, want)
	}
}

// stubScoutCompleter is a fixed-answer scoutCompleter: it returns a canned
// completion so the classify+parse wiring is provable with no weights on disk.
type stubScoutCompleter struct {
	content string
	err     error
	calls   int
}

func (s *stubScoutCompleter) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	s.calls++
	if s.err != nil {
		return nil, s.err
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: s.content}}, nil
}

// TestParseScoutLabelVocabulary pins the closed-vocabulary parse: valid answers map
// to their complexity, and an out-of-vocabulary or label-free answer is a fail-loud
// error (never a silent guess). The error rows are what let this test FAIL if the
// parser ever started guessing.
func TestParseScoutLabelVocabulary(t *testing.T) {
	cases := []struct {
		in      string
		want    modelroute.Complexity
		wantErr bool
	}{
		{"low", modelroute.ComplexityLow, false},
		{"medium", modelroute.ComplexityMedium, false},
		{"high", modelroute.ComplexityHigh, false},
		{"HIGH\n", modelroute.ComplexityHigh, false},
		{"the answer is medium", modelroute.ComplexityMedium, false},
		{`{"complexity":"low"}`, modelroute.ComplexityLow, false},
		{"```json\n{\"complexity\":\"high\"}\n```", modelroute.ComplexityHigh, false},
		{`{"complexity":"extreme"}`, "", true},
		{"banana", "", true},
		{"", "", true},
	}
	for _, tc := range cases {
		got, err := parseScoutLabel(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("parseScoutLabel(%q) = %+v, nil; want error", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseScoutLabel(%q) errored: %v", tc.in, err)
			continue
		}
		if got.Complexity != tc.want {
			t.Errorf("parseScoutLabel(%q) complexity = %q; want %q", tc.in, got.Complexity, tc.want)
		}
		if !got.Valid() {
			t.Errorf("parseScoutLabel(%q) = %+v; want Valid()", tc.in, got)
		}
	}
}

// TestClassifyWithNativeScoutStubParity proves the native classifier is exactly
// planner.Complete + parseScoutLabel: for a fixed model answer the classifier's
// ScoutLabel must equal what parseScoutLabel produces from the same text, and the
// model is called exactly once per classification. This is the fixed-stub parity
// guard the survey note's fence 2 calls for — it needs no weights.
func TestClassifyWithNativeScoutStubParity(t *testing.T) {
	for _, content := range []string{"high", "  HIGH\n", "The task is high.", "```json\n{\"complexity\":\"high\"}\n```"} {
		want, wErr := parseScoutLabel(content)
		if wErr != nil {
			t.Fatalf("parseScoutLabel(%q) errored: %v", content, wErr)
		}
		stub := &stubScoutCompleter{content: content}
		got, err := classifyWithNativeScout(context.Background(), stub, modelroute.Subject{Aspect: modelroute.AspectStep})
		if err != nil {
			t.Fatalf("classifyWithNativeScout(%q): %v", content, err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("classify(%q) = %+v; parseScoutLabel = %+v (classifier must wrap the parser)", content, got, want)
		}
		if got.Complexity != modelroute.ComplexityHigh {
			t.Fatalf("classify(%q) complexity = %q; want high", content, got.Complexity)
		}
		if stub.calls != 1 {
			t.Fatalf("classify(%q) made %d Complete calls; want exactly 1", content, stub.calls)
		}
	}
}

// scoutSmollm2LocalPath returns the local cache path for the embedded smollm2 GGUF,
// or "" when it is not present — checked WITHOUT any network call so the smoke test
// skips cleanly offline instead of triggering a pull.
func scoutSmollm2LocalPath() string {
	ref, _ := modelreg.Resolve("smollm2")
	ref = pathutil.ExpandTilde(ref)
	if !hfhub.IsURI(ref) {
		if fi, err := os.Stat(ref); err == nil && !fi.IsDir() {
			return ref
		}
		return ""
	}
	parsed, err := hfhub.ParseURI(ref)
	if err != nil || parsed.File == "" {
		return ""
	}
	p := hfhub.NewClient().CachePath(parsed)
	if fi, err := os.Stat(p); err == nil && !fi.IsDir() {
		return p
	}
	return ""
}

// TestBindNativeScoutSmollm2Smoke drives the real in-kernel binding end to end when
// the SmolLM2-135M weights are cached locally, and skips otherwise. When it runs it
// makes a REAL assertion: the scout must return a well-formed, in-vocabulary
// complexity for a concrete subject (never the empty ComplexityAny).
func TestBindNativeScoutSmollm2Smoke(t *testing.T) {
	if scoutSmollm2LocalPath() == "" {
		t.Skip("smollm2 weights not cached locally (run `fak run smollm2 hi` to fetch); skipping native scout smoke")
	}
	classify := bindNativeScout("smollm2", nativeControlConfig{})
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	subj := modelroute.Subject{
		Aspect: modelroute.AspectStep,
		Labels: map[string]string{"task": "rename a single local variable in one file"},
	}
	label, err := classify.Classify(ctx, subj)
	if err != nil {
		t.Fatalf("native scout classify: %v", err)
	}
	if !label.Valid() {
		t.Fatalf("scout returned an invalid label: %+v", label)
	}
	if label.Complexity == modelroute.ComplexityAny {
		t.Fatalf("scout returned empty complexity; want one of low/medium/high, got %+v", label)
	}
}
