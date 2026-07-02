package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// doctorTestRegistry writes a registry with an anchor seat, a creds-less seat, and a
// seat whose config dir does not exist, returning the registry path.
func doctorTestRegistry(t *testing.T, home string) string {
	t.Helper()
	anchor := mkHome(t, home, ".claude-anchor-seat", "anchor@example.test", true)
	needs := mkHome(t, home, ".claude-needs-seat", "needs@example.test", false)
	gone := filepath.Join(home, ".claude-gone-seat") // never created on disk
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"anchor-seat","dir":"` + jsonPath(anchor) + `"},` +
		`{"name":"needs-seat","dir":"` + jsonPath(needs) + `"},` +
		`{"name":"gone-seat","dir":"` + jsonPath(gone) + `"}` +
		`],"roles":{"active":"anchor-seat","anchor":"anchor-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	return regPath
}

func TestAccountsDoctorReportsClosedActions(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	regPath := doctorTestRegistry(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with actionable seats rc=%d, want 1; stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	got := out.String()
	for _, want := range []string{"fak.accounts.doctor.v1", "prune", "relogin",
		"CLAUDE_CONFIG_DIR=", "fak accounts remove --name gone-seat",
		"auto-fixable: 1", "doctor --write"} {
		if !strings.Contains(got, want) {
			t.Fatalf("doctor output missing %q:\n%s", want, got)
		}
	}
}

func TestAccountsDoctorWritePrunesMissingDir(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FAK_DOS_ROSTER", "")
	t.Setenv("FAK_JOB_ROSTER", "")
	home := t.TempDir()
	regPath := doctorTestRegistry(t, home)

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--write", "--registry", regPath, "--home", home})
	// The relogin seat still needs an operator, so the exit stays 1 — but the vanished
	// dir must have been tombstoned through the audited remove path.
	if rc != 1 {
		t.Fatalf("doctor --write rc=%d, want 1 (relogin remains); stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	if !strings.Contains(out.String(), "APPLIED") {
		t.Fatalf("doctor --write should report the applied repair:\n%s", out.String())
	}
	reg, err := accounts.LoadRegistry(regPath)
	if err != nil {
		t.Fatalf("registry after doctor --write does not load: %v", err)
	}
	found := false
	for _, h := range reg.Homes {
		if h.Name == "gone-seat" {
			found = true
			if h.Active() || h.RehomeTo != "anchor-seat" {
				t.Fatalf("gone-seat = status %q rehome %q, want tombstoned -> anchor-seat", h.Status, h.RehomeTo)
			}
		}
	}
	if !found {
		t.Fatalf("gone-seat missing from registry after doctor --write")
	}

	// A second doctor pass sees the tombstone as retired (action none): the repair
	// converges instead of re-firing.
	out.Reset()
	errb.Reset()
	if rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home}); rc != 1 {
		t.Fatalf("second doctor rc=%d, want 1 (relogin remains)", rc)
	}
	if strings.Contains(out.String(), "prune") {
		t.Fatalf("second doctor should not re-propose the applied prune:\n%s", out.String())
	}
}

func TestAccountsDoctorCleanRegistryExitsZero(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home}); rc != 0 {
		t.Fatalf("doctor on a clean registry rc=%d, want 0; out=%s stderr=%s", rc, out.String(), errb.String())
	}
	if !strings.Contains(out.String(), "actionable: 0") {
		t.Fatalf("clean doctor output:\n%s", out.String())
	}
}

func TestAccountsDoctorProbeLedgerOverlay(t *testing.T) {
	home := t.TempDir()
	ready := mkHome(t, home, ".claude-ready-seat", "ready@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"ready-seat","dir":"` + jsonPath(ready) + `"}` +
		`],"roles":{"active":"ready-seat","anchor":"ready-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	line := `{"ts":"` + time.Now().UTC().Format(time.RFC3339) + `","account":".claude-ready-seat","status":"LIMIT","reset":"3pm"}`
	if err := os.WriteFile(filepath.Join(rd, "probe_ledger.jsonl"), []byte(line+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with a fresh LIMIT probe rc=%d, want 1; out=%s", rc, out.String())
	}
	got := out.String()
	if !strings.Contains(got, "wait_reset") || !strings.Contains(got, "resets 3pm") {
		t.Fatalf("doctor should fold the fresh usage limit into wait_reset:\n%s", got)
	}
}

func TestAccountsDoctorIdentityMismatchRequiresRelogin(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	home := t.TempDir()
	wrong := mkHome(t, home, ".claude-gem8-seat", "day26@example.test", true)
	reg := `{"version":"fak-config-homes/v1","homes":[` +
		`{"name":"gem8-seat","dir":"` + jsonPath(wrong) + `","chrome_profile":"Profile 9"}` +
		`],"roles":{"active":"gem8-seat","anchor":"gem8-seat"}}`
	regPath := filepath.Join(home, "registry.json")
	if err := os.WriteFile(regPath, []byte(reg), 0o644); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	rc := runAccounts(&out, &errb, []string{"doctor", "--registry", regPath, "--home", home})
	if rc != 1 {
		t.Fatalf("doctor with identity mismatch rc=%d, want 1; stderr=%s\nout=%s", rc, errb.String(), out.String())
	}
	got := out.String()
	if !strings.Contains(got, "identity_mismatch") || !strings.Contains(got, "relogin") ||
		!strings.Contains(got, "CLAUDE_CONFIG_DIR=") {
		t.Fatalf("doctor should route identity mismatch to relogin:\n%s", got)
	}
}
