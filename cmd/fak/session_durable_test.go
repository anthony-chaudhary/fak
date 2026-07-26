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

// TestConfigureServeSessionDurabilityMeasuresAndBoundsRepeatedRecovery is the
// startup-level #4658 witness: repeated corruption keeps startup alive, each
// recovery is counted with its cause class, and quarantine evidence stays
// bounded by the configured retention policy.
func TestConfigureServeSessionDurabilityMeasuresAndBoundsRepeatedRecovery(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session-registry.json")
	t.Setenv(sessionQuarantineRetentionEnv, "count=2")
	const rounds = 5
	for i := 0; i < rounds; i++ {
		if err := os.WriteFile(path, []byte("{broken-round"), 0o600); err != nil {
			t.Fatal(err)
		}
		var stderr bytes.Buffer
		if err := configureServeSessionDurability(session.NewTable(), path, &stderr); err != nil {
			t.Fatalf("round %d: configureServeSessionDurability() error = %v", i, err)
		}
		serveSessionDurability = nil
		if !strings.Contains(stderr.String(), "corrupt session registry quarantined") {
			t.Fatalf("round %d: stderr = %q, want quarantine warning", i, stderr.String())
		}
		if !strings.Contains(stderr.String(), "cause=decode") {
			t.Fatalf("round %d: stderr = %q, want normalized cause class", i, stderr.String())
		}
	}
	matches, err := filepath.Glob(path + ".corrupt-*")
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) > 2 {
		t.Fatalf("quarantine evidence unbounded: %v, want at most 2", matches)
	}
	stats, ok, err := session.ReadRecoveryStats(path)
	if err != nil || !ok {
		t.Fatalf("ReadRecoveryStats() ok=%v err=%v", ok, err)
	}
	if stats.Total != rounds {
		t.Fatalf("recoveries total = %d, want %d", stats.Total, rounds)
	}
	if stats.Causes["decode"] != rounds {
		t.Fatalf("decode cause count = %d, want %d", stats.Causes["decode"], rounds)
	}
}

func TestDoctorMCPSessionRegistryRecoveryStageIsReadOnly(t *testing.T) {
	registry := filepath.Join(t.TempDir(), "session-registry.json")
	t.Setenv(sessionRegistryEnv, registry)

	stage := sessionRegistryRecoveryStage()
	if stage.Name != "session_registry_recovery" || stage.Status != "pass" {
		t.Fatalf("stage = %+v, want passing no-recoveries stage", stage)
	}
	if _, err := os.Stat(session.RecoveryLedgerPath(registry)); !os.IsNotExist(err) {
		t.Fatalf("read-only diagnostic created ledger state: %v", err)
	}

	if err := os.WriteFile(registry, []byte("{broken"), 0o600); err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	if err := configureServeSessionDurability(session.NewTable(), registry, &stderr); err != nil {
		t.Fatalf("configureServeSessionDurability() error = %v", err)
	}
	serveSessionDurability = nil

	stage = sessionRegistryRecoveryStage()
	if stage.Status != "warn" {
		t.Fatalf("stage after recovery = %+v, want warn", stage)
	}
	for _, want := range []string{"recoveries_total=1", "evidence_current=1", "cause=decode", "last="} {
		if !strings.Contains(stage.Detail, want) {
			t.Fatalf("stage detail %q missing %q", stage.Detail, want)
		}
	}
	if strings.Contains(stage.Detail, "{broken") {
		t.Fatalf("stage detail leaks registry contents: %q", stage.Detail)
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
