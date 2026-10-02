// Default suites: go test ./cmd/fak and go test ./... (also under -race).
// Focused suite: go test ./cmd/fak -run '^(TestDefaultTurnkeyNativeScavengeHookOptIn|TestTurnkeyNativeScavengeBeforeHostAfterAndPlanner|TestTurnkeyNativeScavengeSkippedOnStartupFailure)$' -count=1.
package main

import (
	"context"
	"errors"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=100ms lane=default
func TestDefaultTurnkeyNativeScavengeHookOptIn(t *testing.T) {
	const env = "FAK_UP_SCAVENGE_LOADER_HEAP"
	for _, tc := range []struct {
		name    string
		value   string
		unset   bool
		enabled bool
	}{
		{name: "unset default", unset: true},
		{name: "empty"},
		{name: "whitespace only", value: " \t\n"},
		{name: "zero", value: "0"},
		{name: "false", value: "false"},
		{name: "uppercase false", value: "FALSE"},
		{name: "invalid", value: "invalid"},
		{name: "other number", value: "2"},
		{name: "other truthy spelling", value: "yes"},
		{name: "one", value: "1", enabled: true},
		{name: "true", value: "true", enabled: true},
		{name: "uppercase true", value: "TRUE", enabled: true},
		{name: "mixed case true", value: "TrUe", enabled: true},
		{name: "trimmed one", value: " \t1\n ", enabled: true},
		{name: "trimmed mixed case true", value: " \tTrUe\n ", enabled: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Setenv owns restoration even for the genuinely absent-variable case.
			t.Setenv(env, tc.value)
			if tc.unset {
				if err := os.Unsetenv(env); err != nil {
					t.Fatalf("unset opt-in: %v", err)
				}
			}
			if enabled := defaultTurnkeyNativeLoadDeps().scavengeLoaderHeap != nil; enabled != tc.enabled {
				t.Fatalf("loader-heap scavenging configured=%v for %s=%q (unset=%v), want %v", enabled, env, tc.value, tc.unset, tc.enabled)
			}
		})
	}
}

// fak-test:runtime medium est=2s lane=default
func TestTurnkeyNativeScavengeBeforeHostAfterAndPlanner(t *testing.T) {
	for _, tc := range []struct {
		name    string
		q4k     bool
		metal   bool
		nilHook bool
	}{
		{name: "q4k Metal", q4k: true, metal: true},
		{name: "f32 CPU"},
		{name: "nil injected hook", q4k: true, metal: true, nilHook: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			deps := mtpStatusTestDeps()
			var events []string
			var loaded, tokenized, promoted bool
			var memoryCalls, scavenges, plannerCalls, statusCalls int
			available := int64(800)
			model := &fakmodel.Model{}
			tok := &tokenizer.Tokenizer{}
			planner := &agent.InKernelPlanner{}
			deps.resolveMetal = func() (serveMetalDecision, error) {
				return serveMetalDecision{live: tc.metal}, nil
			}
			deps.hostMemory = func() (int64, int64, bool) {
				memoryCalls++
				switch memoryCalls {
				case 1:
					if loaded {
						t.Fatalf("host-before sampled after model load: %v", events)
					}
					events = append(events, "host-before")
				case 2:
					wantScavenges := 1
					if tc.nilHook {
						wantScavenges = 0
					}
					if !loaded || !tokenized || scavenges != wantScavenges || plannerCalls != 0 || statusCalls != 0 {
						t.Fatalf("host-after must follow successful load and reclamation, before planner/MTP: %v", events)
					}
					events = append(events, "host-after")
				default:
					t.Fatalf("unexpected host-memory sample %d: %v", memoryCalls, events)
				}
				return 1000, available, true
			}
			deps.loadModel = func(string, compute.Backend, int, *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
				if memoryCalls != 1 {
					t.Fatalf("model load must follow host-before sample: %v", events)
				}
				loaded = true
				events = append(events, "load")
				return model, tc.q4k, nil
			}
			deps.loadTokenizer = func(string) (*tokenizer.Tokenizer, bool) {
				tokenized = true
				events = append(events, "tokenizer")
				return tok, true
			}
			deps.eagerMetalResidency = func(got *fakmodel.Model) error {
				if got != model || !loaded {
					t.Fatal("eager promotion must receive the loaded model")
				}
				promoted = true
				events = append(events, "eager")
				return nil
			}
			if !tc.nilHook {
				deps.scavengeLoaderHeap = func() {
					scavenges++
					if !loaded || !tokenized || (tc.metal && !promoted) || plannerCalls != 0 || memoryCalls != 1 {
						t.Fatalf("reclamation must follow model/tokenizer/eager promotion, before host-after and planner: %v", events)
					}
					events = append(events, "scavenge")
					available = 900
				}
			}
			deps.newPlanner = func(gotModel *fakmodel.Model, gotTok *tokenizer.Tokenizer, _ string, q4k bool, _ compute.Backend, metal bool, _ int) *agent.InKernelPlanner {
				plannerCalls++
				if memoryCalls != 2 || gotModel != model || gotTok != tok || q4k != tc.q4k || metal != tc.metal {
					t.Fatalf("planner must use loaded resources after host-after: %v", events)
				}
				events = append(events, "planner")
				return planner
			}
			deps.resolveMTPStatus = func(_ *turnkeyMTPQualificationResult, got *agent.InKernelPlanner) (bool, string) {
				statusCalls++
				if memoryCalls != 2 || plannerCalls != 1 || got != planner {
					t.Fatalf("MTP qualification must follow reclaimed host snapshot and planner: %v", events)
				}
				events = append(events, "mtp-status")
				return false, string(turnkeyMTPNoEligibleContext)
			}

			resources, err := loadTurnkeyNativeResourcesWith(context.Background(), "model.gguf", "qwen38", 2048, deps)
			if err != nil {
				t.Fatal(err)
			}
			resources.closeModel = func() error { return nil }
			t.Cleanup(func() { _ = resources.Close() })
			wantScavenges := 1
			if tc.nilHook {
				wantScavenges = 0
			}
			if scavenges != wantScavenges || memoryCalls != 2 || plannerCalls != 1 || statusCalls != 1 {
				t.Fatalf("startup counts scavenge=%d memory=%d planner=%d MTP=%d: %v", scavenges, memoryCalls, plannerCalls, statusCalls, events)
			}
			if got := resources.Startup; got.HostAvailableBefore != 800 || got.HostAvailableAfter != available || got.HostTotalBefore != 1000 || got.HostTotalAfter != 1000 {
				t.Fatalf("startup host-memory snapshot = %+v, want available before=800 after=%d", got, available)
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestTurnkeyNativeScavengeSkippedOnStartupFailure(t *testing.T) {
	refused := errors.New("startup admission refused")
	for _, tc := range []struct {
		name   string
		modify func(*turnkeyNativeLoadDeps)
	}{
		{"peak admission", func(deps *turnkeyNativeLoadDeps) {
			deps.refusePeak = func(string) error { return refused }
		}},
		{"load admission", func(deps *turnkeyNativeLoadDeps) {
			deps.admitAndLoad = func(bool, string, func(), *serveFitBudget) (func(), error) {
				return nil, refused
			}
		}},
		{"nil model", func(deps *turnkeyNativeLoadDeps) {
			deps.loadModel = func(string, compute.Backend, int, *serveFitBudget) (*fakmodel.Model, bool, *gateway.ModelLoadProfile) {
				return nil, true, nil
			}
		}},
		{"failed tokenizer", func(deps *turnkeyNativeLoadDeps) {
			deps.loadTokenizer = func(string) (*tokenizer.Tokenizer, bool) { return &tokenizer.Tokenizer{}, false }
		}},
		{"nil tokenizer", func(deps *turnkeyNativeLoadDeps) {
			deps.loadTokenizer = func(string) (*tokenizer.Tokenizer, bool) { return nil, true }
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			deps := mtpStatusTestDeps()
			deps.scavengeLoaderHeap = func() { t.Fatal("loader heap reclaimed on failed startup") }
			deps.newPlanner = func(*fakmodel.Model, *tokenizer.Tokenizer, string, bool, compute.Backend, bool, int) *agent.InKernelPlanner {
				t.Fatal("planner created on failed startup")
				return nil
			}
			tc.modify(&deps)
			resources, err := loadTurnkeyNativeResourcesWith(context.Background(), "model.gguf", "qwen38", 2048, deps)
			if resources != nil {
				resources.closeModel = func() error { return nil }
				_ = resources.Close()
			}
			if err == nil {
				t.Fatal("failed startup unexpectedly succeeded")
			}
		})
	}
}
