package main

import (
	"bytes"
	"context"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"strconv"
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
		"main.go": `fs.String("engine", "mock",`,
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
	for _, path := range []string{"guard.go", "guard_replay.go"} {
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if !syscallEngineSourceSelectsMock(body) {
			t.Errorf("%s must select one explicit literal mock engine in gateway.New(gateway.Config{...})", path)
		}
	}
}

// Inspect the gateway constructor argument so formatting and comment decoys cannot
// establish the engine contract. Multiple constructors or EngineID fields are ambiguous.
func syscallEngineSourceSelectsMock(source []byte) bool {
	file, err := parser.ParseFile(token.NewFileSet(), "entrypoint.go", source, 0)
	if err != nil {
		return false
	}
	calls, valid := 0, true
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		constructor, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || constructor.Sel.Name != "New" {
			return true
		}
		pkg, ok := constructor.X.(*ast.Ident)
		if !ok || pkg.Name != "gateway" {
			return true
		}
		calls++
		if len(call.Args) != 1 {
			valid = false
			return true
		}
		config, ok := call.Args[0].(*ast.CompositeLit)
		if !ok {
			valid = false
			return true
		}
		typ, ok := config.Type.(*ast.SelectorExpr)
		if !ok || typ.Sel.Name != "Config" {
			valid = false
			return true
		}
		pkg, ok = typ.X.(*ast.Ident)
		if !ok || pkg.Name != "gateway" {
			valid = false
			return true
		}
		fields := 0
		for _, element := range config.Elts {
			field, ok := element.(*ast.KeyValueExpr)
			if !ok {
				valid = false
				continue
			}
			key, ok := field.Key.(*ast.Ident)
			if !ok || key.Name != "EngineID" {
				continue
			}
			fields++
			value, ok := field.Value.(*ast.BasicLit)
			if !ok || value.Kind != token.STRING {
				valid = false
				continue
			}
			engineID, err := strconv.Unquote(value.Value)
			valid = valid && err == nil && engineID == "mock"
		}
		valid = valid && fields == 1
		return true
	})
	return calls == 1 && valid
}

// fak-test:runtime fast est=10ms
func TestSyscallEngineSourceContract(t *testing.T) {
	for _, tc := range []struct {
		name, source string
		want         bool
	}{
		{"compact", `gateway.New(gateway.Config{EngineID:"mock"})`, true},
		{"aligned", `gateway.New(gateway.Config{EngineID:                     "mock"})`, true},
		{"tabbed", "gateway.New(gateway.Config{EngineID:\t\t\"mock\"})", true},
		{"inkernel", `gateway.New(gateway.Config{EngineID:"inkernel"})`, false},
		{"missing", `gateway.New(gateway.Config{})`, false},
		{"nonliteral", `gateway.New(gateway.Config{EngineID:engineID})`, false},
		{"comment-only", `gateway.New(gateway.Config{/* EngineID: "mock" */})`, false},
		{"unpassed-config", `cfg := gateway.Config{EngineID:"mock"}; gateway.New(cfg)`, false},
		{"duplicate-field", `gateway.New(gateway.Config{EngineID:"mock", EngineID:"mock"})`, false},
		{"multiple-constructors", `gateway.New(gateway.Config{EngineID:"mock"}); gateway.New(gateway.Config{EngineID:"mock"})`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			source := []byte("package main\nfunc fixture() { " + tc.source + " }")
			if got := syscallEngineSourceSelectsMock(source); got != tc.want {
				t.Fatalf("source contract = %v, want %v for %s", got, tc.want, tc.source)
			}
		})
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
