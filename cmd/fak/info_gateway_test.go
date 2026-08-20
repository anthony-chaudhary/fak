package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

func TestGGUFAlertFlowsIntoStartupMessage(t *testing.T) {
	profile := toGatewayLoadProfile(&ggufload.LoadProfile{Alerts: []ggufload.LoadAlert{{
		Kind: "memory-preflight", Level: "warning", Text: "NUMA node 1 is constrained",
	}}})
	if profile == nil || len(profile.Messages) != 1 {
		t.Fatalf("gateway model-load messages = %+v, want one retained GGUF alert", profile)
	}
	got := profile.Messages[0]
	if got.Source != "model-load" || got.Kind != "memory-preflight" || got.Level != "warning" || got.Text != "NUMA node 1 is constrained" {
		t.Fatalf("gateway startup message = %+v", got)
	}
}

// TestInfoGatewayViewCapturedRender is the visual witness for the durable startup
// surface: the bytes drawn after readiness contain the load profile and retained
// messages, while transient percentage churn has no place in the frame.
func TestInfoGatewayViewCapturedRender(t *testing.T) {
	v := guardInfoVars{Startup: &startupViewSnapshot{
		Status:             "ready",
		StartedAt:          "2026-08-20T12:00:00Z",
		ReadyAt:            "2026-08-20T12:00:01.5Z",
		TimeToReadySeconds: 1.5,
		UnaccountedSeconds: 0.1,
		Phases: []startupViewPhase{
			{Name: "policy-load", Seconds: 0.2, Provenance: "measured", Stage: "gateway-boot"},
			{Name: "model-load", Seconds: 1.2, Provenance: "measured", Stage: "gateway-boot"},
		},
		ModelLoad: &startupViewModelLoad{
			Source:       "C:/models/qwen.gguf",
			Mode:         "gguf-resident-q4k",
			TotalSeconds: 1.2,
			Bytes:        2_000_000,
			Tensors:      290,
			Bottleneck:   "resident-copy",
			Phases:       []startupViewModelLoadPhase{{Phase: "resident-copy", Seconds: 1.1, Bytes: 1_900_000, Tensors: 288}},
			LoadPaths:    []startupViewModelLoadPath{{QuantType: "Q4_K", Class: "expert", ResidentTensors: 256, ResidentBytes: 1_800_000}},
		},
		Messages: []gateway.StartupMessage{
			{Source: "model-load", Kind: "load-mode", Level: "info", Text: "direct-resident Q4_K"},
			{Source: "model-load", Kind: "memory-preflight", Level: "warning", Text: "NUMA node 1 is constrained"},
			{Source: "guard", Kind: "startup-report", Level: "info", Text: "hooks: installed\nauth: ready"},
		},
	}}

	captured := renderGuardInfoInteractiveBlock(infoViewState{active: viewStartup}, v, nil, 180, 0)
	for _, want := range []string{
		"«7 gateway»",
		"gateway startup: READY · ready in 1.5s · 100ms unaccounted",
		"model-load",
		"model load: gguf-resident-q4k · 1.2s · 290 tensors",
		"load path Q4_K",
		"info/model-load/load-mode: direct-resident Q4_K",
		"warning/model-load/memory-preflight: NUMA node 1 is constrained",
		"info/guard/startup-report: hooks: installed",
		"auth: ready",
	} {
		if !strings.Contains(captured, want) {
			t.Fatalf("captured Gateway frame missing %q:\n%s", want, captured)
		}
	}
	if strings.Contains(captured, "loading model 5%") {
		t.Fatalf("captured Gateway frame replayed transient percentage progress:\n%s", captured)
	}
}

func TestInfoGatewayViewOlderGateway(t *testing.T) {
	got := strings.Join(startupViewRows(guardInfoVars{}), "\n")
	if got != " gateway startup: not reported (older gateway)" {
		t.Fatalf("older-gateway fallback = %q", got)
	}
}
