package session

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type schemaCollisionMarker interface {
	error
	schemaCollision()
}

func TestFileStoreRefusesChildRegistrationSchemaWithoutChangingLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "child-registrations.jsonl")
	before := []byte(`{"schema":"fak-child-registration/1","at":"2026-08-13T12:00:00Z","record":{"schema":"fak-child-registration/1"}}` + "\n")
	if err := os.WriteFile(path, before, 0o600); err != nil {
		t.Fatal(err)
	}

	err := NewFileStore(path).Put(Descriptor{ID: "serve-must-not-overwrite"})
	if err == nil {
		t.Fatal("Put() adopted a child-registration ledger; want schema-collision refusal")
	}
	var collision schemaCollisionMarker
	if !errors.As(err, &collision) {
		t.Fatalf("Put() error %T %v, want typed schema collision", err, err)
	}
	if IsCorruptDescriptorFile(err) {
		t.Fatalf("schema collision reported as corruption %v; serve would quarantine the lineage ledger", err)
	}
	for _, want := range []string{"session registry schema collision", "fak-child-registration/1", "fak.session-descriptors.v1", "<UserConfigDir>/fak/session-registry.json", "<UserConfigDir>/fak/child-registrations.jsonl"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal message %q missing %q", err, want)
		}
	}
	after, readErr := os.ReadFile(path)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if !bytes.Equal(after, before) {
		t.Fatalf("schema-collision refusal changed lineage ledger\nbefore: %q\n after: %q", before, after)
	}
}

func TestFileStoreRefusesUnifiedSessionJournalSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-journal.jsonl")
	before := []byte(`{"schema":"fak.sessionjournal.v1","kind":"open","id":"guard-child"}` + "\n")
	if err := os.WriteFile(path, before, 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := NewFileStore(path).List()
	var collision schemaCollisionMarker
	if !errors.As(err, &collision) || !strings.Contains(err.Error(), "fak.sessionjournal.v1") {
		t.Fatalf("List() error = %T %v, want unified-lineage SchemaCollisionError", err, err)
	}
	if IsCorruptDescriptorFile(err) {
		t.Fatalf("unified lineage journal reported as corrupt: %v", err)
	}
}

// TestForwardCompatIncompatibleSchemaRefusesNotQuarantine is the #3424 contract:
// the durable session ledger carries a self-describing schema magic+version
// header, and an incompatible schema jump across a live-deployment upgrade seam
// must REFUSE loudly with a named reason — not be swept into quarantine, which
// would silently drop every live durable session across the version boundary.
//
// The distinction the serve layer keys on is IsCorruptDescriptorFile: a corrupt
// ledger is quarantined and the runtime restarts empty; an incompatible-but-
// intact ledger is NOT corrupt, so configureServeSessionDurability's
// `!IsCorruptDescriptorFile(err)` branch returns the error and refuses to start.
func TestForwardCompatIncompatibleSchemaRefusesNotQuarantine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	// A well-formed header (descriptorFileMagic) at a version this build does
	// not support: an incompatible forward jump, records intact.
	future := descriptorFileMagic + "v2"
	row := `{"version":"` + future + `","descriptors":[{"id":"live-session","trace":"t","run":1}]}`
	if err := os.WriteFile(path, []byte(row), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := NewFileStore(path).List()
	if err == nil {
		t.Fatal("List() accepted an incompatible-schema ledger; want a loud refusal")
	}
	if !IsIncompatibleSchema(err) {
		t.Fatalf("List() error %v is not an IncompatibleSchemaError", err)
	}
	// The load-bearing property: an incompatible jump must NOT look like
	// corruption, or the serve layer would quarantine it and drop live sessions.
	if IsCorruptDescriptorFile(err) {
		t.Fatalf("incompatible schema jump reported as corruption %v — would quarantine and silently drop live sessions", err)
	}

	msg := err.Error()
	for _, want := range []string{ledgerSchemaIncompatibleReason, future, descriptorFileVersion} {
		if !strings.Contains(msg, want) {
			t.Errorf("refusal message %q missing %q", msg, want)
		}
	}
}

// TestForwardCompatUnrecognizedVersionStaysCorruption pins the other side of the
// fork: a version with no recognizable schema magic is genuine corruption and
// stays on the quarantine path (RecoveryCauseVersion), unchanged by #3424.
func TestForwardCompatUnrecognizedVersionStaysCorruption(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	row := `{"version":"bogus.v9","descriptors":[{"id":"x","trace":"t","run":0}]}`
	if err := os.WriteFile(path, []byte(row), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := NewFileStore(path).List()
	if err == nil {
		t.Fatal("List() accepted a garbage-version ledger")
	}
	if IsIncompatibleSchema(err) {
		t.Fatalf("garbage version %v classified as an incompatible schema jump; want corruption", err)
	}
	if !IsCorruptDescriptorFile(err) {
		t.Fatalf("garbage version %v is not a corrupt-descriptor error", err)
	}
	if got := ClassifyRecoveryCause(err); got != RecoveryCauseVersion {
		t.Fatalf("ClassifyRecoveryCause = %q, want %q", got, RecoveryCauseVersion)
	}
}

// TestForwardCompatSupportedVersionLoads is the negative control: the current
// supported header loads normally, so the refusal fires only on a real
// incompatibility.
func TestForwardCompatSupportedVersionLoads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "registry.json")
	row := `{"version":"` + descriptorFileVersion + `","descriptors":[{"id":"live-session","trace":"t","run":1}]}`
	if err := os.WriteFile(path, []byte(row), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := NewFileStore(path).List()
	if err != nil {
		t.Fatalf("List() on supported version: %v", err)
	}
	if len(got) != 1 || got[0].ID != "live-session" {
		t.Fatalf("List() = %+v, want the one live-session descriptor", got)
	}
}
