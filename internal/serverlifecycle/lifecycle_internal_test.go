package serverlifecycle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"reflect"
	"strings"
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

func TestEdgeAdversarialInputsFailClosed(t *testing.T) {
	t.Run("empty-required-path", func(t *testing.T) {
		_, err := cleanAbsolute("")
		if err == nil || !strings.Contains(err.Error(), "path is required") {
			t.Fatalf("error = %v, want actionable required-path refusal", err)
		}
	})

	t.Run("runtime-schema-and-executable", func(t *testing.T) {
		root := t.TempDir()
		cases := []struct {
			name string
			cfg  runtimeConfig
			want string
		}{
			{name: "unknown-schema", cfg: runtimeConfig{Schema: "hostile/99", AdapterExecutable: root}, want: "runtime schema must be"},
			{name: "empty-executable", cfg: runtimeConfig{Schema: configSchema}, want: "adapter executable"},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				err := validateRuntime(tc.cfg, filepath.Join(root, "model.gguf"))
				if err == nil || !strings.Contains(err.Error(), tc.want) {
					t.Fatalf("error = %v, want recovery cue %q", err, tc.want)
				}
			})
		}
	})

	t.Run("strict-json-rejects-trailing-values", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "state.json")
		if err := os.WriteFile(path, []byte(`{"schema":"fak-server-state/1"} {}`), 0o600); err != nil {
			t.Fatal(err)
		}
		var state stateRecord
		err := readStrict(path, &state)
		if err == nil || !strings.Contains(err.Error(), "multiple JSON values") {
			t.Fatalf("error = %v, want trailing-value refusal", err)
		}
	})

	t.Run("state-schema-is-versioned", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, StateFilename)
		if err := atomicWrite(path, mustJSON(t, stateRecord{Schema: "hostile/99"})); err != nil {
			t.Fatal(err)
		}
		_, err := readState(dir)
		if err == nil || !strings.Contains(err.Error(), "state schema must be") {
			t.Fatalf("error = %v, want schema recovery cue", err)
		}
	})
}

func mustJSON(t *testing.T, value any) []byte {
	t.Helper()
	data, err := marshalJSON(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestRefusalErrorsNameRecovery(t *testing.T) {
	root := t.TempDir()
	cases := []struct {
		name string
		run  func() error
		want []string
	}{
		{name: "status-uninitialized", run: func() error {
			_, err := Status(context.Background(), filepath.Join(root, "not-initialized"), Options{})
			return err
		}, want: []string{"read server spec", SpecFilename}},
		{name: "down-uninitialized", run: func() error {
			_, err := Down(context.Background(), filepath.Join(root, "not-initialized-down"), Options{})
			return err
		}, want: []string{"read server spec", SpecFilename}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.run()
			if err == nil {
				t.Fatal("expected refusal")
			}
			message := strings.ToLower(err.Error())
			for _, cue := range tc.want {
				if !strings.Contains(message, strings.ToLower(cue)) {
					t.Fatalf("error %q does not name recovery cue %q", err, cue)
				}
			}
		})
	}
}
