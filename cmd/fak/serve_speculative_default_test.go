package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

func noSpeculativeEnv(string) string { return "" }

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeDefaultIsMTPAndRetainsHead(t *testing.T) {
	mode, err := resolveSpeculativeMode("mtp", noSpeculativeEnv)
	if err != nil || mode != serveSpeculativeMTP {
		t.Fatalf("default flag resolved (%q, %v), want mtp", mode, err)
	}
	if !serveRetainMTPHead(mode, false, "vulkan") {
		t.Fatal("default mtp selector on a resident backend must retain the MTP head")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeOptOutTokensDisable(t *testing.T) {
	for _, tok := range []string{"off", "none", "false", "disabled", "0", " OFF ", "Disabled"} {
		mode, err := resolveSpeculativeMode(tok, noSpeculativeEnv)
		if err != nil {
			t.Fatalf("opt-out %q refused: %v", tok, err)
		}
		if mode != "" {
			t.Fatalf("opt-out %q resolved %q, want disabled", tok, mode)
		}
		if serveRetainMTPHead(mode, true, "vulkan") {
			t.Fatalf("opt-out %q must not retain the MTP head", tok)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeEmptyLegacyDisableStillAccepted(t *testing.T) {
	mode, err := resolveSpeculativeMode("", noSpeculativeEnv)
	if err != nil || mode != "" {
		t.Fatalf("legacy empty flag resolved (%q, %v), want disabled", mode, err)
	}
	if serveRetainMTPHead(mode, true, "vulkan") {
		t.Fatal("legacy empty disable must not retain the MTP head")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeEnvOffOverridesMTPFlag(t *testing.T) {
	getenv := func(key string) string {
		if key == "FAK_SPECULATIVE" {
			return "off"
		}
		return ""
	}
	mode, err := resolveSpeculativeMode("mtp", getenv)
	if err != nil || mode != "" {
		t.Fatalf("FAK_SPECULATIVE=off over --speculative=mtp resolved (%q, %v), want disabled", mode, err)
	}
	if serveRetainMTPHead(mode, true, "vulkan") {
		t.Fatal("env opt-out must not retain the MTP head")
	}

	legacy := func(key string) string {
		if key == "SPECULATIVE" {
			return "none"
		}
		return ""
	}
	if mode, err := resolveSpeculativeMode("mtp", legacy); err != nil || mode != "" {
		t.Fatalf("SPECULATIVE=none over --speculative=mtp resolved (%q, %v), want disabled", mode, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeEnvMTPOverridesOffFlag(t *testing.T) {
	getenv := func(key string) string {
		if key == "FAK_SPECULATIVE" {
			return "mtp"
		}
		return ""
	}
	mode, err := resolveSpeculativeMode("off", getenv)
	if err != nil || mode != serveSpeculativeMTP {
		t.Fatalf("FAK_SPECULATIVE=mtp over --speculative=off resolved (%q, %v), want mtp", mode, err)
	}
	if !serveRetainMTPHead(mode, true, "vulkan") {
		t.Fatal("env mtp must retain the MTP head")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeUnknownStillRefuses(t *testing.T) {
	for _, bad := range []string{"eagle", "mtp2", "mt p", "offf", "on"} {
		_, err := resolveSpeculativeMode(bad, noSpeculativeEnv)
		if !errors.Is(err, ErrSpeculativeModeUnknown) {
			t.Fatalf("unknown %q: err=%v, want ErrSpeculativeModeUnknown", bad, err)
		}
	}
	_, err := resolveSpeculativeMode("mtp", func(key string) string {
		if key == "SPECULATIVE" {
			return "bogus"
		}
		return ""
	})
	if !errors.Is(err, ErrSpeculativeModeUnknown) {
		t.Fatalf("unknown env override: err=%v, want ErrSpeculativeModeUnknown", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeRetainMTPFollowsResolvedMode(t *testing.T) {
	prior := fakmodel.RetainMTP
	defer fakmodel.SetRetainMTP(prior)

	for _, tc := range []struct {
		mode string
		want bool
	}{
		{serveSpeculativeMTP, true},
		{serveSpeculativeNGram, false},
		{serveSpeculativeDFlash2, false},
		{"", false},
	} {
		fakmodel.SetRetainMTP(serveRetainMTPHead(tc.mode, false, "vulkan"))
		if fakmodel.RetainMTP != tc.want {
			t.Fatalf("mode %q: RetainMTP=%v, want %v", tc.mode, fakmodel.RetainMTP, tc.want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeRetainMTPRequiresConsumingBackend(t *testing.T) {
	for _, tc := range []struct {
		name    string
		metal   bool
		backend string
		want    bool
	}{
		{"metal", true, "", true},
		{"vulkan", false, "vulkan", true},
		{"cuda", false, "CUDA", true},
		{"cpu", false, "", false},
		{"other-device", false, "rocm", false},
	} {
		if got := serveRetainMTPHead(serveSpeculativeMTP, tc.metal, tc.backend); got != tc.want {
			t.Fatalf("%s: retain=%v, want %v", tc.name, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestServeSpeculativeFlagDefaultIsMTP(t *testing.T) {
	_, sf := newServeFlagSet()
	if got := *sf.speculative; got != serveSpeculativeMTP {
		t.Fatalf("--speculative default = %q, want mtp", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeDefaultRetainsOnlyConsumingFamily(t *testing.T) {
	if !serveRetainMTPForModel(serveSpeculativeMTP, false, "vulkan", "qwen35", false) {
		t.Fatal("default mtp on a resident backend with a qwen35 checkpoint must retain")
	}
	if !serveRetainMTPForModel(serveSpeculativeMTP, true, "", "qwen35moe", false) {
		t.Fatal("default mtp on Metal with a qwen35moe checkpoint must retain")
	}
	if serveRetainMTPForModel(serveSpeculativeMTP, false, "vulkan", "glm_moe_dsa", false) {
		t.Fatal("default GLM head has no consumer and must not be retained by default")
	}
	if serveRetainMTPForModel(serveSpeculativeMTP, false, "vulkan", "", false) {
		t.Fatal("unpeekable checkpoint must not retain by default")
	}
	if !serveRetainMTPForModel(serveSpeculativeMTP, false, "vulkan", "glm_moe_dsa", true) {
		t.Fatal("operator-forced mtp must retain regardless of family")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeMTPFallbackDecision(t *testing.T) {
	refused := &fakmodel.Qwen35MTPIncompleteError{Missing: []string{"mtp.fc.weight"}}
	estimate := errors.New("q4k estimate unsupported: retained MTP requires one shared-embedding layer")
	fit := errors.New("FitTooBig: plan exceeds device budget")
	for _, tc := range []struct {
		name     string
		err      error
		retained bool
		forced   bool
		want     bool
	}{
		{"default-retained-refused", refused, true, false, true},
		{"forced-stays-fail-loud", refused, true, true, false},
		{"not-retained", refused, false, false, false},
		{"load-ok", nil, true, false, false},
		{"retained-estimate-refused", estimate, true, false, true},
		{"fit-error-keeps-head", fit, true, false, false},
	} {
		if got := serveMTPRetentionFallback(tc.err, tc.retained, tc.forced); got != tc.want {
			t.Fatalf("%s: fallback=%v, want %v", tc.name, got, tc.want)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeMTPFallbackRetriesOnce(t *testing.T) {
	prior := fakmodel.RetainMTP
	defer fakmodel.SetRetainMTP(prior)

	fakmodel.SetRetainMTP(true)
	if !serveNoteMTPRetentionFallback(errors.New("mtp head refused"), noSpeculativeEnv) {
		t.Fatal("first refused default-retained load must arm the fallback")
	}
	if fakmodel.RetainMTP {
		t.Fatal("fallback must drop retention before the retry")
	}
	if serveNoteMTPRetentionFallback(errors.New("mtp head refused"), noSpeculativeEnv) {
		t.Fatal("second refusal with retention already dropped must not retry again")
	}

	fakmodel.SetRetainMTP(true)
	forcedEnv := func(key string) string {
		if key == "FAK_SPECULATIVE" {
			return "mtp"
		}
		return ""
	}
	if serveNoteMTPRetentionFallback(errors.New("mtp head refused"), forcedEnv) {
		t.Fatal("operator-forced mtp must not fall back")
	}
	if !fakmodel.RetainMTP {
		t.Fatal("operator-forced mtp must keep retention on refusal")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSpeculativeMTPPeekModelTypeReadsDirConfig(t *testing.T) {
	dir := t.TempDir()
	if got := serveMTPPeekModelType(dir); got != "" {
		t.Fatalf("dir without config.json peeked %q, want empty", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen35"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := serveMTPPeekModelType(dir); got != "qwen35" {
		t.Fatalf("dir config peeked %q, want qwen35", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := serveMTPPeekModelType(dir); !serveMTPArchConsumes(got) {
		t.Fatalf("HF qwen3_5 dir peeked %q, want a consuming family", got)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"llava","text_config":{"model_type":"qwen3_5_moe_text"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := serveMTPPeekModelType(dir); got != "qwen3_5_moe_text" {
		t.Fatalf("wrapped qwen3_5 text config peeked %q, want qwen3_5_moe_text", got)
	}
	if serveMTPArchConsumes("glm_moe_dsa") || serveMTPArchConsumes("llama") {
		t.Fatal("non-Qwen3.5 families must not consume the head")
	}
	if got := serveMTPPeekModelType(filepath.Join(dir, "missing.gguf")); got != "" {
		t.Fatalf("missing path peeked %q, want empty", got)
	}
}
