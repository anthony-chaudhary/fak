package model

import (
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
)

func TestInkling2LoopRefuseBackendAdmitsCUDAAndMetal(t *testing.T) {
	admitted := []string{"cuda"}
	refused := map[string]string{"": "cpu", "cpu": "cpu", "vulkan": "vulkan", "rocm": "rocm", "Metal": "Metal"}
	// only a build that links the native target admits metal
	if inkling2MetalLinked {
		admitted = append(admitted, "metal")
	} else {
		refused["metal"] = "metal"
	}
	for _, backend := range admitted {
		if err := RefuseInkling2LoopBackend(backend); err != nil {
			t.Fatalf("%s refused: %v", backend, err)
		}
	}
	for backend, want := range refused {
		err := RefuseInkling2LoopBackend(backend)
		var refusal *UnsupportedInkling2LoopExecutionError
		if !errors.As(err, &refusal) || refusal.Backend != want {
			t.Fatalf("%q: refusal = %v, want typed refusal naming %q", backend, err, want)
		}
		msg := err.Error()
		if !strings.Contains(msg, "the "+want+" backend") || !strings.Contains(msg, "--backend cuda") || !strings.Contains(msg, "--metal") || strings.Contains(msg, "not implemented") {
			t.Fatalf("%q: stale or unhelpful refusal: %s", backend, msg)
		}
	}
}

func TestInkling2LoopGenericArchSupportStillRefuses(t *testing.T) {
	_, err := ClassifyForwardPath(Config{ModelType: inkling2LoopEnvelopeType}, nil)
	var refusal *UnsupportedInkling2LoopExecutionError
	if !errors.As(err, &refusal) || refusal.Backend != "" {
		t.Fatalf("generic forward = %v, want the backend-free refusal", err)
	}
	if msg := err.Error(); !strings.Contains(msg, "generic native forward") || strings.Contains(msg, "not implemented") {
		t.Fatalf("generic refusal message: %s", msg)
	}
}

// The safetensors path keeps refusing inkling2_loop, still as the typed
// execution refusal, but now names the GGUF aliases that do run.
func TestArchSupportInkling2SafetensorsRefusalPointsAtGGUFAliases(t *testing.T) {
	data, err := os.ReadFile(inkling2MiniConfigPath)
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatal(err)
	}
	_, err = ClassifyForwardPath(cfg, nil)
	var refusal *Inkling2SafetensorsRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("ClassifyForwardPath = %v, want Inkling2SafetensorsRefusal", err)
	}
	var unsupported *UnsupportedInkling2LoopExecutionError
	if !errors.As(err, &unsupported) {
		t.Fatalf("refusal %v does not unwrap to UnsupportedInkling2LoopExecutionError", err)
	}
	for _, alias := range Inkling2GGUFAliases {
		if !strings.Contains(err.Error(), "fak run "+alias) {
			t.Fatalf("refusal %q does not name alias %q", err, alias)
		}
	}
}

func TestArchSupportInkling2NonInklingConfigPasses(t *testing.T) {
	if err := refuseUnsupportedHybridArch(Config{ModelType: "llama"}, nil); err != nil {
		t.Fatalf("llama refused: %v", err)
	}
}
