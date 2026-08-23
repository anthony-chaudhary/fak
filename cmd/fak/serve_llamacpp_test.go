package main

import (
	"testing"
	"time"
)

func TestNormalizeQwen38Runtime(t *testing.T) {
	for _, tc := range []struct {
		in, want string
		bad      bool
	}{{"", qwen38RuntimeAuto, false}, {"AUTO", qwen38RuntimeAuto, false}, {"native", qwen38RuntimeNative, false}, {"llama-mtp", qwen38RuntimeLlamaMTP, false}, {"vllm", "", true}} {
		got, err := normalizeQwen38Runtime(tc.in)
		if (err != nil) != tc.bad || got != tc.want {
			t.Fatalf("%q got=%q err=%v", tc.in, got, err)
		}
	}
}

func TestQwen38DelegationNativeOverrideIsIdentity(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--qwen38-runtime", "native", "--gguf", "missing.gguf"}); err != nil {
		t.Fatal(err)
	}
	rt := &serveRuntime{}
	if err := rt.maybeStartQwen38Delegation(sf); err != nil {
		t.Fatal(err)
	}
	if rt.llamaProcess != nil || *sf.baseURL != "" || *sf.ggufPath != "missing.gguf" {
		t.Fatalf("process=%v base=%q gguf=%q", rt.llamaProcess, *sf.baseURL, *sf.ggufPath)
	}
}

func TestQwen38DelegationAutoUnavailableFallsBack(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--qwen38-runtime", "auto", "--gguf", "missing.gguf", "--llama-startup-timeout", time.Millisecond.String()}); err != nil {
		t.Fatal(err)
	}
	rt := &serveRuntime{}
	if err := rt.maybeStartQwen38Delegation(sf); err != nil {
		t.Fatal(err)
	}
	if rt.llamaProcess != nil || *sf.ggufPath != "missing.gguf" {
		t.Fatalf("process=%v gguf=%q", rt.llamaProcess, *sf.ggufPath)
	}
}

func TestQwen38DelegationRequiredUnavailableFails(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--qwen38-runtime", "llama-mtp", "--gguf", "missing.gguf"}); err != nil {
		t.Fatal(err)
	}
	if err := (&serveRuntime{}).maybeStartQwen38Delegation(sf); err == nil {
		t.Fatal("required delegation unexpectedly succeeded")
	}
}
