package serverlifecycle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"
)

func TestOperatingEnvelopeDefaults(t *testing.T) {
	if defaultReadinessTimeout < 300*time.Second {
		t.Fatalf("readiness deadline = %s, want at least 300s", defaultReadinessTimeout)
	}
	states := []State{StateConfigured, StateStarting, StateReady, StateStale, StateFailed, StateStopped}
	if len(states) < 6 {
		t.Fatalf("lifecycle states = %d, want at least 6", len(states))
	}
	seen := make(map[State]bool, len(states))
	for _, state := range states {
		if state == "" || seen[state] {
			t.Fatalf("state vocabulary is not distinct: %v", states)
		}
		seen[state] = true
	}
}

func TestInitDeterminism(t *testing.T) {
	root := t.TempDir()
	model := filepath.Join(root, "model.gguf")
	executable := filepath.Join(root, "llama-server")
	if err := os.WriteFile(model, []byte("model"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(executable, []byte("binary"), 0o700); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte("model"))
	makeOpts := func(dir string) InitOptions {
		return InitOptions{InstanceDirectory: dir, ServerName: "local", ModelPath: model, ArtifactSHA256: hex.EncodeToString(digest[:]), AdapterExecutable: executable, Port: 8080}
	}
	firstDir, secondDir := filepath.Join(root, "first"), filepath.Join(root, "second")
	first, err := Init(context.Background(), makeOpts(firstDir))
	if err != nil {
		t.Fatal(err)
	}
	second, err := Init(context.Background(), makeOpts(secondDir))
	if err != nil {
		t.Fatal(err)
	}
	first.ObservedAt, second.ObservedAt = "", ""
	first.InstanceDirectory, second.InstanceDirectory = "", ""
	first.SpecPath, second.SpecPath = "", ""
	first.ReceiptPath, second.ReceiptPath = "", ""
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("same input shape produced different results:\nfirst=%+v\nsecond=%+v", first, second)
	}
	for _, name := range []string{ConfigFilename} {
		a, err := os.ReadFile(filepath.Join(firstDir, name))
		if err != nil {
			t.Fatal(err)
		}
		b, err := os.ReadFile(filepath.Join(secondDir, name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(a, b) {
			t.Fatalf("%s differs across runs:\n%s\n%s", name, a, b)
		}
	}
}
