package main

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/deploymanifest"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/llamacppinterop"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

type fakeQwen38Process struct {
	baseURL string
	pid     int
	stopped bool
}

func (p *fakeQwen38Process) BaseURL() string { return p.baseURL }
func (p *fakeQwen38Process) PID() int        { return p.pid }
func (p *fakeQwen38Process) Stop() error {
	p.stopped = true
	return nil
}

func TestQwen38RuntimeOmissionStaysFakNativeAndObservable(t *testing.T) {
	fs, sf := newServeFlagSet()
	if got := *sf.qwen38Runtime; got != qwen38RuntimeNative {
		t.Fatalf("default runtime=%q, want %q", got, qwen38RuntimeNative)
	}
	if err := fs.Parse([]string{"--gguf", "missing.gguf"}); err != nil {
		t.Fatal(err)
	}
	rt := &serveRuntime{}
	if err := rt.maybeStartQwen38Delegation(sf); err != nil {
		t.Fatal(err)
	}
	if rt.llamaProcess != nil || *sf.baseURL != "" || *sf.ggufPath != "missing.gguf" {
		t.Fatalf("default delegated: process=%v base=%q gguf=%q", rt.llamaProcess, *sf.baseURL, *sf.ggufPath)
	}
	if !startupMessagesContain(rt.startupMessages, "execution_runtime=fak-native", "delegation=disabled") {
		t.Fatalf("native runtime identity missing: %+v", rt.startupMessages)
	}
}

func TestQwen38RuntimeIdentityAppearsInEffectiveConfig(t *testing.T) {
	fs, sf := newServeFlagSet()
	report := effectiveServeConfigWithQwen38Runtime(sf, deploymanifest.Manifest{}, false, explicitFlagNames(fs))
	if got := report.Values["qwen38_runtime"]; got.Value != qwen38RuntimeNative || got.Source != "built-in" {
		t.Fatalf("runtime=%+v", got)
	}
	if got := report.Values["qwen38_runtime_identity"]; got.Value != "fak-native" || got.Source != "built-in" {
		t.Fatalf("engine=%+v", got)
	}

	fs, sf = newServeFlagSet()
	if err := fs.Parse([]string{"--qwen38-runtime", "llama-mtp"}); err != nil {
		t.Fatal(err)
	}
	report = effectiveServeConfigWithQwen38Runtime(sf, deploymanifest.Manifest{}, false, explicitFlagNames(fs))
	if got := report.Values["qwen38_runtime_identity"]; got.Value != "llama.cpp" || got.Source != "flag" {
		t.Fatalf("explicit engine=%+v", got)
	}
}

func TestQwen38ExplicitLlamaMTPPreservesReferenceInterop(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--qwen38-runtime", "llama-mtp", "--gguf", "qwen38.gguf", "--llama-server", "llama-server-pinned"}); err != nil {
		t.Fatal(err)
	}
	proc := &fakeQwen38Process{baseURL: "http://127.0.0.1:18080/v1", pid: 4242}
	started := false
	rt := &serveRuntime{qwen38Deps: &qwen38RuntimeDependencies{
		inspect: func(path string) (quantmeta.Descriptor, string, error) {
			if path != "qwen38.gguf" {
				t.Fatalf("inspect path=%q", path)
			}
			return qwen38Descriptor(), "qwen35", nil
		},
		discover: func(_ context.Context, binary string) llamacppinterop.Result {
			if binary != "llama-server-pinned" {
				t.Fatalf("binary=%q", binary)
			}
			return llamacppinterop.Result{
				Outcome: llamacppinterop.OutcomeDelegate,
				Capability: llamacppinterop.Capability{
					Binary: "llama-server-pinned", Version: "0.0.6123", Commit: "8144f31",
					Server: true, DraftMTP: true, CUDA: true,
				},
			}
		},
		freePort: func() (int, error) { return 18080, nil },
		start: func(_ context.Context, argv []string, _ time.Duration) (qwen38ChildProcess, error) {
			started = true
			for _, want := range []string{"llama-server-pinned", "--spec-type", "draft-mtp", "--host", "127.0.0.1"} {
				if !containsQwenRuntimeString(argv, want) {
					t.Fatalf("delegation argv missing %q: %v", want, argv)
				}
			}
			return proc, nil
		},
	}}
	if err := rt.maybeStartQwen38Delegation(sf); err != nil {
		t.Fatal(err)
	}
	if !started || rt.llamaProcess != proc || *sf.baseURL != proc.baseURL || *sf.ggufPath != "" || *sf.model != "qwen38.gguf" {
		t.Fatalf("started=%v process=%v base=%q gguf=%q model=%q", started, rt.llamaProcess, *sf.baseURL, *sf.ggufPath, *sf.model)
	}
	if !startupMessagesContain(rt.startupMessages, "execution_runtime=llama.cpp", "delegation=explicit") {
		t.Fatalf("external runtime identity missing: %+v", rt.startupMessages)
	}
	rt.stopQwen38Delegation()
	if !proc.stopped || rt.llamaProcess != nil {
		t.Fatalf("delegated process was not stopped: stopped=%v process=%v", proc.stopped, rt.llamaProcess)
	}
}

func TestQwen38ExplicitLlamaMTPMissingCapabilityFailsClosed(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--qwen38-runtime", "llama-mtp", "--gguf", "qwen38.gguf"}); err != nil {
		t.Fatal(err)
	}
	started := false
	rt := &serveRuntime{qwen38Deps: &qwen38RuntimeDependencies{
		inspect: func(string) (quantmeta.Descriptor, string, error) {
			return qwen38Descriptor(), "qwen35", nil
		},
		discover: func(context.Context, string) llamacppinterop.Result {
			return llamacppinterop.Result{Outcome: llamacppinterop.OutcomeRefuse, Reason: "version probe failed: executable not found"}
		},
		start: func(context.Context, []string, time.Duration) (qwen38ChildProcess, error) {
			started = true
			return nil, errors.New("must not start")
		},
	}}
	err := rt.maybeStartQwen38Delegation(sf)
	if err == nil || !strings.Contains(err.Error(), "capability unavailable") || !strings.Contains(err.Error(), "executable not found") {
		t.Fatalf("err=%v", err)
	}
	if started || rt.llamaProcess != nil || *sf.ggufPath != "qwen38.gguf" || *sf.baseURL != "" {
		t.Fatalf("missing capability fell through: started=%v process=%v gguf=%q base=%q", started, rt.llamaProcess, *sf.ggufPath, *sf.baseURL)
	}
}

func TestQwen38ExplicitLlamaMTPRejectsIgnoredNativeConfiguration(t *testing.T) {
	for _, tc := range []struct {
		name string
		args []string
		want string
	}{
		{name: "missing gguf", args: []string{"--qwen38-runtime", "llama-mtp"}, want: "requires --gguf"},
		{name: "upstream", args: []string{"--qwen38-runtime", "llama-mtp", "--gguf", "m.gguf", "--base-url", "http://127.0.0.1:9000"}, want: "--base-url"},
		{name: "replica", args: []string{"--qwen38-runtime", "llama-mtp", "--gguf", "m.gguf", "--replica-base-url", "http://127.0.0.1:9001"}, want: "--replica-base-url"},
		{name: "backend", args: []string{"--qwen38-runtime", "llama-mtp", "--gguf", "m.gguf", "--backend", "cuda"}, want: "--backend"},
		{name: "cuda graph", args: []string{"--qwen38-runtime", "llama-mtp", "--gguf", "m.gguf", "--cuda-graph"}, want: "--cuda-graph"},
		{name: "metal", args: []string{"--qwen38-runtime", "llama-mtp", "--gguf", "m.gguf", "--metal"}, want: "--metal"},
		{name: "tokenizer", args: []string{"--qwen38-runtime", "llama-mtp", "--gguf", "m.gguf", "--tokenizer", "tokenizer.json"}, want: "--tokenizer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fs, sf := newServeFlagSet()
			if err := fs.Parse(tc.args); err != nil {
				t.Fatal(err)
			}
			err := (&serveRuntime{}).maybeStartQwen38Delegation(sf)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want diagnostic containing %q", err, tc.want)
			}
		})
	}
}

func TestQwen38RemovedAutoValueExplainsMigration(t *testing.T) {
	_, err := normalizeQwen38Runtime("auto")
	if err == nil || !strings.Contains(err.Error(), "could silently delegate") || !strings.Contains(err.Error(), "omit the flag") || !strings.Contains(err.Error(), "llama-mtp explicitly") {
		t.Fatalf("err=%v", err)
	}
	fs, _ := newServeFlagSet()
	usage := fs.Lookup("qwen38-runtime").Usage
	for _, want := range []string{"native (default)", "explicitly delegates", "benchmark/reference", "removed auto value is rejected"} {
		if !strings.Contains(usage, want) {
			t.Fatalf("help missing %q: %s", want, usage)
		}
	}
}

func qwen38Descriptor() quantmeta.Descriptor {
	return quantmeta.Descriptor{
		Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"},
		Extra:    map[string]json.RawMessage{"gguf_architecture": json.RawMessage(`"qwen35"`)},
	}
}

func startupMessagesContain(messages []gateway.StartupMessage, wants ...string) bool {
	for _, message := range messages {
		if message.Kind != "qwen38-runtime" {
			continue
		}
		matched := true
		for _, want := range wants {
			if !strings.Contains(message.Text, want) {
				matched = false
			}
		}
		if matched {
			return true
		}
	}
	return false
}

func containsQwenRuntimeString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
