package codexsession

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

var supportedAuthority = []string{
	"item/commandExecution/requestApproval",
	"item/fileChange/requestApproval",
}

func TestSupportedCompatibilityFixturesBindVersionAndSchema(t *testing.T) {
	for _, name := range []string{"minimum-supported.json", "current-supported.json"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join("testdata", "conformance", name)
			receipt, err := LoadCompatibilityReceipt(path)
			if err != nil {
				t.Fatal(err)
			}
			schema, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := CheckCompatibility(CompatibilityEnvelope{
				BinaryVersion: receipt.BinaryVersion, ProtocolVersion: receipt.ProtocolVersion,
				Schema: schema, AuthorityMethods: supportedAuthority,
			}, receipt, supportedAuthority); err != nil {
				t.Fatalf("supported fixture rejected: %v", err)
			}
		})
	}
}

func TestIncompatibleAuthorityRefusedBeforeToolExecution(t *testing.T) {
	path := filepath.Join("testdata", "conformance", "incompatible-authority.json")
	receipt, err := LoadCompatibilityReceipt(path)
	if err != nil {
		t.Fatal(err)
	}
	schema, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	toolEffects := 0
	envelope := CompatibilityEnvelope{
		BinaryVersion: receipt.BinaryVersion, ProtocolVersion: receipt.ProtocolVersion,
		Schema: schema, AuthorityMethods: []string{"item/commandExecution/requestAuthorization"},
	}
	adapter, newErr := New(Config{
		Command: "must-not-start", Version: receipt.BinaryVersion, RunID: "compatibility-test", Sink: func(harnesskit.Envelope) error { return nil },
		Compatibility: &envelope, TestedReceipt: &receipt, AuthorityMethods: supportedAuthority,
	})
	if newErr != nil {
		t.Fatal(newErr)
	}
	err = adapter.Run(context.Background(), "must not execute")
	if err == nil {
		toolEffects++
	}
	var compatibilityErr *CompatibilityError
	if !errors.As(err, &compatibilityErr) {
		t.Fatalf("got %T (%v), want *CompatibilityError", err, err)
	}
	if !strings.Contains(err.Error(), "authority methods changed") {
		t.Fatalf("diagnosis %q does not name authority drift", err)
	}
	if toolEffects != 0 {
		t.Fatalf("incompatible authority request caused %d tool effects", toolEffects)
	}
}

func TestCompatibilityReceiptRejectsFixtureDrift(t *testing.T) {
	original, err := os.ReadFile(filepath.Join("testdata", "conformance", "current-supported.json"))
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "current-supported.json")
	if err := os.WriteFile(path, append(original, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCompatibilityReceipt(path); err == nil {
		t.Fatal("modified tested fixture accepted")
	}
}

func TestCompatibilityReceiptRejectsSchemaDrift(t *testing.T) {
	path := filepath.Join("testdata", "conformance", "current-supported.json")
	receipt, err := LoadCompatibilityReceipt(path)
	if err != nil {
		t.Fatal(err)
	}
	err = CheckCompatibility(CompatibilityEnvelope{
		BinaryVersion: receipt.BinaryVersion, ProtocolVersion: receipt.ProtocolVersion,
		Schema: []byte(`{"changed":true}`), AuthorityMethods: supportedAuthority,
	}, receipt, supportedAuthority)
	if err == nil {
		t.Fatal("schema drift accepted")
	}
}
