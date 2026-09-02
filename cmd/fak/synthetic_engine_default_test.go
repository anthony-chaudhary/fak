package main

import (
	"context"
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
