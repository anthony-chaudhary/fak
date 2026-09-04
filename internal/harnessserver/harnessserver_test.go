package harnessserver_test

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessserver"
	"github.com/anthony-chaudhary/fak/internal/serverproduct"
)

func TestExternalReceiptImportIsImmutableAndLifecycleFree(t *testing.T) {
	root := t.TempDir()
	serverDir := filepath.Join(root, "server-product")
	harnessDir := filepath.Join(root, "harness-product")
	receiptPath := writeReceipt(t, serverDir, validReceipt(root, "local-code", 4, []string{"chat.completions", "models.list"}))
	lifecycleLog := filepath.Join(serverDir, "lifecycle.jsonl")
	if err := os.WriteFile(lifecycleLog, []byte("{\"operation\":\"up\",\"owner\":\"server-product\"}\n"), 0o444); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(receiptPath, 0o444); err != nil {
		t.Fatal(err)
	}
	receiptBefore := fileDigest(t, receiptPath)
	lifecycleBefore := fileDigest(t, lifecycleLog)

	binding, err := harnessserver.Import(harnessDir, receiptPath, requirements("local-code", "2026-02", 4))
	if err != nil {
		t.Fatal(err)
	}
	bindingPath := filepath.Join(harnessDir, harnessserver.BindingFileName)
	if _, err := harnessserver.WriteBinding(bindingPath, binding); err != nil {
		t.Fatal(err)
	}
	verified, err := harnessserver.VerifyFile(bindingPath)
	if err != nil {
		t.Fatal(err)
	}
	if verified.ModelAlias != "local-code" || verified.Generation != 4 || verified.BaseURL != "http://127.0.0.1:8080" {
		t.Fatalf("verified=%+v", verified)
	}
	if got := fileDigest(t, receiptPath); got != receiptBefore {
		t.Fatalf("receipt mutated: before=%s after=%s", receiptBefore, got)
	}
	if got := fileDigest(t, lifecycleLog); got != lifecycleBefore {
		t.Fatalf("lifecycle log mutated: before=%s after=%s", lifecycleBefore, got)
	}

	if err := os.Chmod(receiptPath, 0o600); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var tampered serverproduct.ServerReceipt
	if err := json.Unmarshal(raw, &tampered); err != nil {
		t.Fatal(err)
	}
	tampered.Protocol.Capabilities = []string{"sk-do-not-print-this-secret"}
	tamperedRaw, err := json.Marshal(tampered)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(receiptPath, tamperedRaw, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := harnessserver.VerifyFile(bindingPath); err == nil || !strings.Contains(err.Error(), "changed since immutable import") || strings.Contains(err.Error(), "sk-do-not-print-this-secret") {
		t.Fatalf("immutable verification error=%v", err)
	}
}

func TestImportAcceptsCompatibleReceiptEnvelope(t *testing.T) {
	for _, tc := range []struct {
		name       string
		model      string
		revision   string
		generation uint64
		caps       []string
	}{
		{name: "minimum chat", model: "code-a", revision: "2026-02", generation: 1, caps: []string{"chat.completions"}},
		{name: "capability superset", model: "code-b", revision: "2026-02", generation: 7, caps: []string{"chat.completions", "metrics", "models.list"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			receiptPath := writeReceipt(t, filepath.Join(root, "server"), validReceipt(root, tc.model, tc.generation, tc.caps))
			req := requirements(tc.model, tc.revision, tc.generation)
			req.RequiredCapabilities = []string{"chat.completions"}
			binding, err := harnessserver.Import(filepath.Join(root, "harness"), receiptPath, req)
			if err != nil {
				t.Fatal(err)
			}
			if binding.ReceiptGeneration != tc.generation {
				t.Fatalf("generation=%d", binding.ReceiptGeneration)
			}
		})
	}
}

func TestImportRejectsIncompatibleOrUnsafeReceiptsWithoutLeakingSecrets(t *testing.T) {
	secret := "sk-do-not-print-this-secret"
	for _, tc := range []struct {
		name      string
		mutate    func(*serverproduct.ServerReceipt)
		require   func(*harnessserver.Requirements)
		wantError string
	}{
		{name: "stale generation", require: func(req *harnessserver.Requirements) { req.MinimumGeneration = 2 }, wantError: "generation is stale"},
		{name: "wrong model", require: func(req *harnessserver.Requirements) { req.ModelAlias = "different-model" }, wantError: "model alias mismatch"},
		{name: "missing chat", mutate: func(receipt *serverproduct.ServerReceipt) { receipt.Protocol.Capabilities = []string{"models.list"} }, wantError: `missing required capability "chat.completions"`},
		{name: "wildcard endpoint", mutate: func(receipt *serverproduct.ServerReceipt) { receipt.Endpoint.BaseURL = "http://0.0.0.0:8080" }, wantError: "loopback"},
		{name: "missing readiness", mutate: func(receipt *serverproduct.ServerReceipt) { receipt.Readiness = serverproduct.ReadinessEvidence{} }, wantError: "readiness probe is required"},
		{name: "secret bearing", mutate: func(receipt *serverproduct.ServerReceipt) {
			receipt.Auth = serverproduct.AuthReference{Mode: serverproduct.AuthCredentialRef, CredentialRef: secret}
		}, wantError: "credential_ref must be an uppercase symbolic name"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			receipt := validReceipt(root, "local-code", 1, []string{"chat.completions", "models.list"})
			if tc.mutate != nil {
				tc.mutate(&receipt)
			}
			receiptPath := writeReceipt(t, filepath.Join(root, "server"), receipt)
			req := requirements("local-code", "2026-02", 1)
			if tc.require != nil {
				tc.require(&req)
			}
			_, err := harnessserver.Import(filepath.Join(root, "harness"), receiptPath, req)
			if err == nil || !strings.Contains(err.Error(), tc.wantError) {
				t.Fatalf("error=%v want=%q", err, tc.wantError)
			}
			if strings.Contains(err.Error(), secret) {
				t.Fatalf("diagnostic leaked secret: %v", err)
			}
		})
	}
}

func requirements(model, revision string, generation uint64) harnessserver.Requirements {
	return harnessserver.Requirements{
		ModelAlias:           model,
		ProtocolFamily:       serverproduct.ProtocolOpenAIHTTP,
		ProtocolRevision:     revision,
		RequiredCapabilities: []string{"chat.completions", "models.list"},
		MinimumGeneration:    generation,
	}
}

func validReceipt(root, model string, generation uint64, capabilities []string) serverproduct.ServerReceipt {
	return serverproduct.ServerReceipt{
		Schema:     serverproduct.SchemaV1,
		State:      serverproduct.ReceiptStateReady,
		Identity:   serverproduct.ServerIdentity{ServerName: "local-code", InstanceID: "local-code-0001"},
		SpecDigest: digestOf("server-spec"),
		Generation: generation,
		CreatedAt:  "2026-08-19T12:00:01Z",
		Artifact: serverproduct.ArtifactIdentity{
			Reference: filepath.Join(root, "model.gguf"),
			Digest:    digestOf("model"),
		},
		Adapter:    serverproduct.AdapterIdentity{Name: "llama-server", Version: "b6001", ExecutableDigest: digestOf("llama-server")},
		Endpoint:   serverproduct.LoopbackEndpoint{BaseURL: "http://127.0.0.1:8080"},
		ModelAlias: model,
		Auth:       serverproduct.AuthReference{Mode: serverproduct.AuthNone},
		Protocol: serverproduct.ProtocolObservation{
			Family: serverproduct.ProtocolOpenAIHTTP, Revision: "2026-02", Capabilities: capabilities,
		},
		Readiness: serverproduct.ReadinessEvidence{Probe: "GET_/health+POST_/v1/chat/completions", ProbeDigest: digestOf("probe"), ObservedAt: "2026-08-19T12:00:00Z"},
		Ownership: serverproduct.OwnershipReference{InstanceID: "local-code-0001", ProcessID: 4101, ProcessStartIdentity: "linux-startticks:100001"},
		Provenance: serverproduct.ReceiptProvenance{
			Spec:      serverproduct.Provenance{Kind: serverproduct.ProvenanceAuthored, Source: "server-spec"},
			Artifact:  serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "sha256-check"},
			Adapter:   serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "process-inspection"},
			Endpoint:  serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "listener-probe"},
			Readiness: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "http-probe"},
			Ownership: serverproduct.Provenance{Kind: serverproduct.ProvenanceObserved, Source: "process-start"},
		},
	}
}

func writeReceipt(t testing.TB, dir string, receipt serverproduct.ServerReceipt) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	raw, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "server-receipt.json")
	if err := os.WriteFile(path, append(raw, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func fileDigest(t testing.TB, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func digestOf(value string) string {
	sum := sha256.Sum256([]byte(value))
	return "sha256:" + hex.EncodeToString(sum[:])
}
