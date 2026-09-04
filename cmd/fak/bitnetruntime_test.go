package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestBitnetRuntimeContract(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runBitnetRuntime(&stdout, &stderr, []string{"--contract"})
	if code != 0 {
		t.Fatalf("expected 0, got %d: %s", code, stderr.String())
	}
	for _, want := range []string{`"contract_version": "bitnetruntime/v1"`, `"runtime_name": "bitnet.cpp"`, `"i2_s"`, `"tl1"`, `"tl2"`} {
		if !bytes.Contains(stdout.Bytes(), []byte(want)) {
			t.Errorf("output missing %s:\n%s", want, stdout.String())
		}
	}
}

func TestBitnetRuntimeInputDelegate(t *testing.T) {
	var stdout, stderr bytes.Buffer
	fixture := filepath.Join("..", "..", "internal", "bitnetruntime", "testdata", "delegate_darwin_arm64_tl1.input.json")
	code := runBitnetRuntime(&stdout, &stderr, []string{"--input", fixture})
	if code != 0 {
		t.Fatalf("expected 0, got %d: %s", code, stderr.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte(`"outcome": "delegate"`)) {
		t.Fatalf("output missing delegate outcome:\n%s", stdout.String())
	}
}

func TestMainDispatchIncludesBitnetRuntime(t *testing.T) {
	raw, err := os.ReadFile("main.go")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(raw, []byte(`case "bitnetruntime":`)) || !bytes.Contains(raw, []byte(`cmdBitnetRuntime(args)`)) {
		t.Fatal("bitnetruntime implementation exists but top-level fak dispatch is missing")
	}
}
