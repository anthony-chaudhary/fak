package harnessartifact

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func lifecycleFixture() ModelLifecycleReceipt {
	return ModelLifecycleReceipt{
		Declaration: LifecycleIdentity{ID: "decl-1"}, Artifact: LifecycleIdentity{ID: "gguf-1", Digest: strings.Repeat("a", 64)},
		Runtime: LifecycleIdentity{ID: "runtime-1"}, Admission: LifecycleIdentity{ID: "admission-1"},
		Process: LifecycleIdentity{ID: "pid-42"}, Readiness: LifecycleIdentity{ID: "probe-1"}, Stop: LifecycleIdentity{ID: "stop-1"},
		HealthURL: "http://user:supersecret@127.0.0.1:8080/health?token=supersecret", CompletionURL: "http://127.0.0.1:8080/v1/completions?key=supersecret", State: "ready",
	}
}

func TestModelLifecycleReceiptTamperAndRedaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := WriteModelLifecycleReceipt(path, lifecycleFixture()); err != nil {
		t.Fatal(err)
	}
	receipt, err := ReadModelLifecycleReceipt(path)
	if err != nil || receipt.Schema != ModelLifecycleReceiptSchema {
		t.Fatalf("read: %#v %v", receipt, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	raw[strings.Index(string(raw), "runtime-1")] = 'R'
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = ReadModelLifecycleReceipt(path)
	var diagnostic *LifecycleDiagnostic
	if !errors.As(err, &diagnostic) || diagnostic.Code != "LIFECYCLE_RECEIPT_TAMPERED" {
		t.Fatalf("got %v", err)
	}
	rendered := diagnostic.Error() + " " + RedactLifecycleText("failed "+lifecycleFixture().HealthURL+" and "+lifecycleFixture().CompletionURL)
	if strings.Contains(rendered, "supersecret") {
		t.Fatalf("secret leaked: %s", rendered)
	}
}

func TestCheckLifecycleDeclarationIsTypedStale(t *testing.T) {
	err := CheckLifecycleDeclaration(lifecycleFixture(), "decl-2")
	if LifecycleDiagnosticCode(err) != "LIFECYCLE_RECEIPT_STALE" {
		t.Fatalf("got %v", err)
	}
}
