package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

func TestDefaultSyscallEngineSelectionIsMock(t *testing.T) {
	fs, flags := newServeFlagSet()
	if err := fs.Parse(nil); err != nil {
		t.Fatalf("parse serve defaults: %v", err)
	}
	if got := *flags.engineID; got != "mock" {
		t.Fatalf("serve default engine = %q, want mock", got)
	}

	contracts := map[string]string{
		"main.go":         `fs.String("engine", "mock",`,
		"guard.go":        `EngineID: "mock",`,
		"guard_replay.go": `EngineID:             "mock",`,
	}
	for path, want := range contracts {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if !strings.Contains(string(body), want) {
			t.Errorf("%s does not select the mock default", path)
		}
	}
}

func TestDefaultSyscallEngineAvoidsSyntheticGeneratedTokens(t *testing.T) {
	if got := abi.Engine("mock"); got != engine.MockEngine {
		t.Fatalf("default engine driver = %T, want registered mock driver", got)
	}

	ctx := context.Background()
	args, err := abi.ActiveResolver().Put(ctx, []byte(`{"path":"notes.txt"}`))
	if err != nil {
		t.Fatalf("put args: %v", err)
	}
	res, err := abi.Engine("mock").Complete(ctx, &abi.ToolCall{Tool: "read_file", Args: args})
	if err != nil {
		t.Fatalf("complete with default engine: %v", err)
	}
	payload, err := abi.ActiveResolver().Resolve(ctx, res.Payload)
	if err != nil {
		t.Fatalf("resolve default result: %v", err)
	}
	if strings.Contains(string(payload), "generated_tokens") {
		t.Fatalf("default result contains synthetic generated_tokens: %s", payload)
	}
	if res.Meta["engine"] != "mock" {
		t.Fatalf("default result engine = %q, want mock", res.Meta["engine"])
	}
}

func TestExplicitInkernelEngineRemainsFakNative(t *testing.T) {
	driver := abi.Engine(modelengine.EngineID)
	if driver == nil {
		t.Fatal("explicit inkernel engine is not registered")
	}
	if _, ok := driver.(*modelengine.Engine); !ok {
		t.Fatalf("inkernel driver = %T, want fak-native *modelengine.Engine", driver)
	}
	for _, cap := range driver.Caps() {
		if cap == "engine.inkernel" {
			return
		}
	}
	t.Fatalf("inkernel capabilities = %v, want engine.inkernel", driver.Caps())
}

func TestServeEngineResolution(t *testing.T) {
	// Case 1: When --gguf is passed without explicit --engine, the engine resolves to "inkernel".
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--gguf", "test.gguf"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	resolveServeEngine(sf, explicitFlagNames(fs), false)
	if got := *sf.engineID; got != "inkernel" {
		t.Fatalf("when --gguf passed without --engine: engine = %q, want %q", got, "inkernel")
	}

	// Case 2: When --gguf is passed WITH explicit --engine mock, the engine respects "mock".
	fs2, sf2 := newServeFlagSet()
	if err := fs2.Parse([]string{"--gguf", "test.gguf", "--engine", "mock"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	resolveServeEngine(sf2, explicitFlagNames(fs2), false)
	if got := *sf2.engineID; got != "mock" {
		t.Fatalf("when --gguf passed with explicit --engine mock: engine = %q, want %q", got, "mock")
	}

	// Case 3: When no --gguf and no --engine is passed, the engine defaults to "mock".
	fs3, sf3 := newServeFlagSet()
	if err := fs3.Parse([]string{}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	resolveServeEngine(sf3, explicitFlagNames(fs3), false)
	if got := *sf3.engineID; got != "mock" {
		t.Fatalf("when neither --gguf nor --engine passed: engine = %q, want %q", got, "mock")
	}

	// Case 4: When in-kernel model is loaded without explicit --engine, the engine resolves to "inkernel".
	fs4, sf4 := newServeFlagSet()
	if err := fs4.Parse([]string{}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	resolveServeEngine(sf4, explicitFlagNames(fs4), true)
	if got := *sf4.engineID; got != "inkernel" {
		t.Fatalf("when inKernelLoaded=true without --engine: engine = %q, want %q", got, "inkernel")
	}
}

func TestDefaultModelUpgradedToGemini38(t *testing.T) {
	// 1. chat default model
	_, cf := newChatFlagSet()
	if got := *cf.model; got != "gemini-3.8-flash" {
		t.Fatalf("chat default model = %q, want gemini-3.8-flash", got)
	}

	// 2. agent default model
	_, af := newAgentFlagSet()
	if got := *af.model; got != "gemini-3.8-flash" {
		t.Fatalf("agent default model = %q, want gemini-3.8-flash", got)
	}

	// 3. geminicache default model
	var buf bytes.Buffer
	rc := runGeminiCache(&buf, io.Discard, []string{"--prefix", "test", "--json"})
	if rc != 0 {
		t.Fatalf("runGeminiCache returned exit code %d", rc)
	}
	if !strings.Contains(buf.String(), "models/gemini-3.8-flash") {
		t.Fatalf("geminicache output did not contain models/gemini-3.8-flash: %s", buf.String())
	}
}
