package serverlifecycle

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

func TestConsumeReadyBindsLiveIdentityWithoutMutation(t *testing.T) {
	dir, stateRaw, receiptRaw, want := writeConsumerFixture(t)
	binding, err := ConsumeReady(dir, want)
	if err != nil {
		t.Fatalf("ConsumeReady: %v", err)
	}
	if binding.ReceiptDigest != receiptDigest(receiptRaw) || !bytes.Equal(binding.ReceiptBytes, receiptRaw) {
		t.Fatalf("binding digest/bytes do not identify consumed receipt")
	}
	if binding.Receipt.Generation != want.Generation || binding.Receipt.Endpoint.BaseURL != want.BaseURL || binding.Receipt.ModelAlias != want.ModelAlias {
		t.Fatalf("binding identity = %+v", binding.Receipt)
	}
	assertConsumerFilesUnchanged(t, dir, stateRaw, receiptRaw)
	if _, ok := processIdentity(os.Getpid()); !ok {
		t.Fatal("consumer signalled or otherwise invalidated the live process")
	}
}

func TestConsumeReadyRefusesIdentityDrift(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*ReadyExpectation)
		want   string
	}{
		{"generation", func(w *ReadyExpectation) { w.Generation++ }, "generation"},
		{"minimum generation", func(w *ReadyExpectation) { w.Generation = 0; w.MinimumGeneration += 2 }, "at least"},
		{"process id", func(w *ReadyExpectation) { w.ProcessID++ }, "process id"},
		{"process start", func(w *ReadyExpectation) { w.ProcessStartIdentity += "-drift" }, "process start"},
		{"receipt digest", func(w *ReadyExpectation) { w.ReceiptDigest = "sha256:" + strings.Repeat("0", 64) }, "receipt digest"},
		{"protocol family", func(w *ReadyExpectation) { w.ProtocolFamily = "drift" }, "protocol family"},
		{"protocol revision", func(w *ReadyExpectation) { w.ProtocolRevision = "drift" }, "protocol revision"},
		{"capabilities", func(w *ReadyExpectation) { w.Capabilities = []string{"models"} }, "capabilities"},
		{"base URL", func(w *ReadyExpectation) { w.BaseURL = "http://127.0.0.1:2" }, "base URL"},
		{"model alias", func(w *ReadyExpectation) { w.ModelAlias = "drift" }, "model alias"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir, stateRaw, receiptRaw, want := writeConsumerFixture(t)
			test.mutate(&want)
			if _, err := ConsumeReady(dir, want); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err=%v, want substring %q", err, test.want)
			}
			assertConsumerFilesUnchanged(t, dir, stateRaw, receiptRaw)
		})
	}
}

func TestConsumeReadyRejectsMalformedAndStateReceiptDrift(t *testing.T) {
	t.Run("unknown receipt field", func(t *testing.T) {
		dir, stateRaw, receiptRaw, want := writeConsumerFixture(t)
		receiptRaw = bytes.TrimSpace(receiptRaw)
		receiptRaw = append(receiptRaw[:len(receiptRaw)-1], []byte(`,"unexpected":true}`)...)
		if err := os.WriteFile(filepath.Join(dir, ReceiptFilename), receiptRaw, 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := ConsumeReady(dir, want); err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("err=%v", err)
		}
		assertConsumerFilesUnchanged(t, dir, stateRaw, receiptRaw)
	})
	t.Run("state base URL differs", func(t *testing.T) {
		dir, _, receiptRaw, want := writeConsumerFixture(t)
		var state stateRecord
		readJSONTest(t, filepath.Join(dir, StateFilename), &state)
		state.BaseURL = "http://127.0.0.1:2"
		stateRaw := writeJSONBytesTest(t, filepath.Join(dir, StateFilename), state)
		if _, err := ConsumeReady(dir, want); err == nil || !strings.Contains(err.Error(), "does not match lifecycle state") {
			t.Fatalf("err=%v", err)
		}
		assertConsumerFilesUnchanged(t, dir, stateRaw, receiptRaw)
	})
}

func writeConsumerFixture(t *testing.T) (string, []byte, []byte, ReadyExpectation) {
	t.Helper()
	dir := t.TempDir()
	started, ok := processIdentity(os.Getpid())
	if !ok {
		t.Skip("current process start identity unavailable")
	}
	const (
		instance = "consumer-test"
		baseURL  = "http://127.0.0.1:18080"
		model    = "qwen3.8-27b-q4_k_m"
	)
	now := time.Now().UTC().Format(time.RFC3339Nano)
	state := stateRecord{Schema: stateSchema, State: StateReady, InstanceID: instance, Generation: 7, ProcessID: os.Getpid(), ProcessStartIdentity: started, BaseURL: baseURL, UpdatedAt: now}
	stateRaw := writeJSONBytesTest(t, filepath.Join(dir, StateFilename), state)
	digest := "sha256:" + strings.Repeat("a", 64)
	receipt := serverproduct.ServerReceipt{
		Schema: serverproduct.SchemaV1, State: serverproduct.ReceiptStateReady,
		Identity: serverproduct.ServerIdentity{ServerName: "consumer-server", InstanceID: instance}, SpecDigest: digest, Generation: 7, CreatedAt: now,
		Artifact: serverproduct.ArtifactIdentity{Reference: filepath.Join(dir, "model.gguf"), Digest: digest},
		Adapter:  serverproduct.AdapterIdentity{Name: "llama-server", Version: "test", ExecutableDigest: digest},
		Endpoint: serverproduct.LoopbackEndpoint{BaseURL: baseURL}, ModelAlias: model, Auth: serverproduct.AuthReference{Mode: serverproduct.AuthNone},
		Protocol:  serverproduct.ProtocolObservation{Family: serverproduct.ProtocolOpenAIHTTP, Revision: "v1", Capabilities: []string{"chat-completions", "models"}},
		Readiness: serverproduct.ReadinessEvidence{Probe: "models", ProbeDigest: digest, ObservedAt: now},
		Ownership: serverproduct.OwnershipReference{InstanceID: instance, ProcessID: os.Getpid(), ProcessStartIdentity: started},
		Provenance: serverproduct.ReceiptProvenance{
			Spec:     serverproduct.Provenance{Kind: serverproduct.ProvenanceAuthored, Source: "test"},
			Artifact: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Adapter: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"},
			Endpoint: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Readiness: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"}, Ownership: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "test"},
		},
	}
	receiptRaw := writeJSONBytesTest(t, filepath.Join(dir, ReceiptFilename), receipt)
	return dir, stateRaw, receiptRaw, ReadyExpectation{Generation: 7, MinimumGeneration: 7, ProcessID: os.Getpid(), ProcessStartIdentity: started, ReceiptDigest: receiptDigest(receiptRaw), ProtocolFamily: serverproduct.ProtocolOpenAIHTTP, ProtocolRevision: "v1", Capabilities: []string{"models", "chat-completions"}, BaseURL: baseURL, ModelAlias: model}
}

func writeJSONBytesTest(t *testing.T, path string, value any) []byte {
	t.Helper()
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	raw = append(raw, '\n')
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return raw
}

func readJSONTest(t *testing.T, path string, value any) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, value); err != nil {
		t.Fatal(err)
	}
}

func assertConsumerFilesUnchanged(t *testing.T, dir string, stateRaw, receiptRaw []byte) {
	t.Helper()
	for path, want := range map[string][]byte{filepath.Join(dir, StateFilename): stateRaw, filepath.Join(dir, ReceiptFilename): receiptRaw} {
		got, err := os.ReadFile(path)
		if err != nil || !bytes.Equal(got, want) {
			t.Fatalf("%s changed: err=%v", filepath.Base(path), err)
		}
	}
}
