package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestConfigureServeSessionDurabilityQuarantinesCorruptRegistry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-registry.json")
	corrupt := []byte{0, 0, 0, 0}
	if err := os.WriteFile(path, corrupt, 0o600); err != nil {
		t.Fatal(err)
	}

	var stderr bytes.Buffer
	if err := configureServeSessionDurability(session.NewTable(), path, &stderr); err != nil {
		t.Fatalf("configureServeSessionDurability() error = %v", err)
	}
	t.Cleanup(func() { serveSessionDurability = nil })

	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("corrupt registry still exists at active path: %v", err)
	}
	matches, err := filepath.Glob(path + ".corrupt-*")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 1 {
		t.Fatalf("quarantine matches = %v, want one", matches)
	}
	got, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, corrupt) {
		t.Fatalf("quarantined bytes = %v, want %v", got, corrupt)
	}
	if !strings.Contains(stderr.String(), "corrupt session registry quarantined") {
		t.Fatalf("stderr = %q, want quarantine warning", stderr.String())
	}
}

func TestConfigureServeSessionDurabilityKeepsReadFailureFatal(t *testing.T) {
	path := t.TempDir()
	err := configureServeSessionDurability(session.NewTable(), path, nil)
	if err == nil {
		t.Fatal("configureServeSessionDurability() unexpectedly accepted a registry directory")
	}
	if !strings.Contains(err.Error(), "read session descriptor file") {
		t.Fatalf("error = %v, want read failure", err)
	}
}
