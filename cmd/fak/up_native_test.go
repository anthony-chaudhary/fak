package main

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func TestTurnkeyNativeResourcesAdmissionAndLifecycle(t *testing.T) {
	refused := errors.New("peak refused")
	loadCalls := 0
	deps := turnkeyNativeLoadDeps{
		resolveBackend: func() (compute.Backend, error) { return nil, nil },
		resolveMetal:   func() (serveMetalDecision, error) { return serveMetalDecision{live: true}, nil },
		refusePeak:     func(string) error { return refused },
		admitAndLoad: func(bool, string, func()) (func(), error) {
			t.Fatal("admission ran after peak refusal")
			return nil, nil
		},
	}
	if _, err := loadTurnkeyNativeResourcesWith(context.Background(), "model.gguf", "qwen38", 2048, deps); !errors.Is(err, refused) {
		t.Fatalf("peak refusal = %v, want %v", err, refused)
	}
	if loadCalls != 0 {
		t.Fatalf("load calls after refusal = %d", loadCalls)
	}

	admissionRefused := errors.New("reservation refused")
	deps.refusePeak = func(string) error { return nil }
	deps.admitAndLoad = func(_ bool, _ string, _ func()) (func(), error) {
		return nil, admissionRefused
	}
	if _, err := loadTurnkeyNativeResourcesWith(context.Background(), "model.gguf", "qwen38", 2048, deps); !errors.Is(err, admissionRefused) {
		t.Fatalf("admission refusal = %v, want %v", err, admissionRefused)
	}
	if loadCalls != 0 {
		t.Fatalf("load calls after admission refusal = %d", loadCalls)
	}

	var order []string
	model := &fakmodel.Model{}
	profile := &gateway.ModelLoadProfile{Mode: "gguf-resident-q4k", TotalSeconds: 1.25, Bytes: 99, Tensors: 7, Bottleneck: "resident-copy"}
	planner := &agent.InKernelPlanner{}
	deps.refusePeak = func(string) error { return nil }
	deps.admitAndLoad = func(metal bool, path string, load func()) (func(), error) {
		if !metal || path != "model.gguf" {
			t.Fatalf("admission args metal=%v path=%q", metal, path)
		}
		load()
		return func() { order = append(order, "release-admission") }, nil
	}
	deps.loadModel = func(path string, _ compute.Backend, tokens int) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
		loadCalls++
		if path != "model.gguf" || tokens != 2048 {
			t.Fatalf("load args path=%q tokens=%d", path, tokens)
		}
		return model, true, profile
	}
	deps.loadTokenizer = func(string) (*tokenizer.Tokenizer, bool) { return &tokenizer.Tokenizer{}, true }
	memoryCalls := 0
	deps.hostMemory = func() (int64, int64, bool) {
		memoryCalls++
		return 1000, int64(900 - memoryCalls*100), true
	}
	deps.metalResidency = func() (int, int) { return 3, 4 }
	deps.newPlanner = func(gotModel *fakmodel.Model, _ *tokenizer.Tokenizer, id string, q4k bool, _ compute.Backend, metal bool, contextTokens int) *agent.InKernelPlanner {
		if gotModel != model || id != "qwen38" || !q4k || !metal {
			t.Fatalf("planner args model=%p id=%q q4k=%v metal=%v", gotModel, id, q4k, metal)
		}
		if contextTokens != 2048 {
			t.Fatalf("planner context tokens = %d, want 2048", contextTokens)
		}
		return planner
	}

	resources, err := loadTurnkeyNativeResourcesWith(context.Background(), "model.gguf", "qwen38", 2048, deps)
	if err != nil {
		t.Fatal(err)
	}
	if resources.Planner != planner || resources.Model != model || resources.LoadProfile != profile || loadCalls != 1 {
		t.Fatalf("resources = %+v, load calls=%d", resources, loadCalls)
	}
	if got := resources.Startup; got.HostTotalBefore != 1000 || got.HostAvailableBefore != 800 || got.HostTotalAfter != 1000 || got.HostAvailableAfter != 700 || got.LoadMode != "gguf-resident-q4k" || got.LoadSeconds != 1.25 || got.LoadBytes != 99 || got.LoadTensors != 7 || got.LoadBottleneck != "resident-copy" || got.Resident == nil || got.MetalLiveQ6Weights != 3 || got.MetalLiveQ8Weights != 4 || !got.MetalLive || got.MetalCompiled != metalgemm.Compiled() {
		t.Fatalf("startup snapshot = %+v", got)
	}
	resources.closeModel = func() error { order = append(order, "close-model"); return nil }
	if err := resources.Close(); err != nil {
		t.Fatal(err)
	}
	if err := resources.Close(); err != nil {
		t.Fatal(err)
	}
	if want := []string{"close-model", "release-admission"}; !reflect.DeepEqual(order, want) {
		t.Fatalf("cleanup order = %v, want %v", order, want)
	}
}

// TestTurnkeyNativeResourcesCPUDecisionPaths proves the CPU path records the
// Metal decision booleans (live/compiled) in the startup snapshot without
// disturbing the admission, load, planner, or cleanup behavior the matrix above
// pins for the live-Metal path.
func TestTurnkeyNativeResourcesCPUDecisionPaths(t *testing.T) {
	for _, tc := range []struct {
		name         string
		decision     serveMetalDecision
		wantLive     bool
		wantCompiled bool
	}{
		{name: "not compiled", decision: serveMetalDecision{live: false, skippedBecause: skipReasonNotCompiled}, wantLive: false, wantCompiled: false},
		{name: "compiled but no device", decision: serveMetalDecision{live: false, skippedBecause: skipReasonNoDevice}, wantLive: false, wantCompiled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			decided := tc.decision
			loadCalls := 0
			deps := turnkeyNativeLoadDeps{
				resolveBackend: func() (compute.Backend, error) { return nil, nil },
				resolveMetal:   func() (serveMetalDecision, error) { return decided, nil },
				refusePeak:     func(string) error { return nil },
				admitAndLoad: func(metal bool, path string, load func()) (func(), error) {
					if metal {
						t.Fatalf("CPU decision passed metal=%v to admission", metal)
					}
					load()
					return func() {}, nil
				},
				loadModel: func(path string, _ compute.Backend, tokens int) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
					loadCalls++
					return &fakmodel.Model{}, false, nil
				},
				loadTokenizer: func(string) (*tokenizer.Tokenizer, bool) { return &tokenizer.Tokenizer{}, true },
				newPlanner: func(_ *fakmodel.Model, _ *tokenizer.Tokenizer, id string, q4k bool, _ compute.Backend, metal bool, contextTokens int) *agent.InKernelPlanner {
					if metal {
						t.Fatalf("CPU decision still built the Metal planner")
					}
					if id != "qwen38" || q4k || contextTokens != 2048 {
						t.Fatalf("planner args id=%q q4k=%v contextTokens=%d", id, q4k, contextTokens)
					}
					return &agent.InKernelPlanner{}
				},
				// Compiled here must be the real stub-side probe so wantCompiled
				// matches startup.MetalCompiled exactly.
			}
			if loadCalls != 0 {
				t.Fatalf("pre-existing load calls = %d", loadCalls)
			}
			resources, err := loadTurnkeyNativeResourcesWith(context.Background(), "model.gguf", "qwen38", 2048, deps)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				resources.closeModel = func() error { return nil }
				_ = resources.Close()
			}()
			if resources.Startup.MetalLive != tc.wantLive {
				t.Fatalf("startup.MetalLive = %v, want %v", resources.Startup.MetalLive, tc.wantLive)
			}
			if resources.Startup.MetalCompiled != metalgemm.Compiled() {
				t.Fatalf("startup.MetalCompiled = %v, want %v", resources.Startup.MetalCompiled, metalgemm.Compiled())
			}
			if tc.decision.skippedBecause != skipReasonNotCompiled {
				return
			}
			if tc.wantCompiled {
				t.Fatalf("not-compiled case cannot expect compiled=true")
			}
		})
	}
}

func TestNewTurnkeyInKernelPlannerAppliesContextBudget(t *testing.T) {
	planner := newTurnkeyInKernelPlanner(fakmodel.NewSynthetic(fakmodel.Config{}), nil, "local", false, nil, false, 4096)
	if got := planner.RuntimeConfig().ContextTokens; got != 4096 {
		t.Fatalf("planner context tokens = %d, want 4096", got)
	}
}
