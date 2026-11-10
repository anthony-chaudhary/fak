package projectassets

// Regression tests for the anti-clobber plugin contract (fak-private#2960):
// a present dos-proof-guard.js that passes the structural admission check is
// preserved byte-for-byte by every sync/ensure path; only missing or
// structurally broken guards are (re)written from DefaultOpenCodePlugin.
// Drift from canonical is a reported condition (parity receipt, verify CLI),
// never a silent revert.

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// pluginFixture builds the standard asset fixture and rewrites its manifest to
// declare both .opencode/plugins assets, so the parity receipt carries a
// Plugins scope.
func pluginFixture(t *testing.T) string {
	t.Helper()
	r := fixture(t)
	m, err := Load(r)
	if err != nil {
		t.Fatal(err)
	}
	m.Plugins = []string{OpenCodePluginPath, openCodeGrepPath}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	write(t, r, ManifestPath, string(b))
	return r
}

// driftedGuard is DefaultOpenCodePlugin plus one trailing comment line: it
// passes every structural admission pattern and emits no strict= kwarg, so it
// verifies as valid while remaining non-canonical.
func driftedGuard() string {
	return DefaultOpenCodePlugin + "\n// locally evolved: operator-added trailing comment\n"
}

// T1: Build(write) preserves a hand-authored (drifting but structurally valid)
// guard byte-for-byte and reports it as "drifted" in the receipt.
func TestBuildWritePreservesHandAuthoredGuard(t *testing.T) {
	root := pluginFixture(t)
	drifted := driftedGuard()
	write(t, root, OpenCodePluginPath, drifted)

	receipt, err := Build(root, true)
	if err != nil {
		t.Fatalf("Build(write) failed: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(OpenCodePluginPath)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b, []byte(drifted)) {
		t.Fatalf("hand-authored guard was reverted to canonical; anti-clobber violated (got %d bytes, want %d)", len(b), len(drifted))
	}
	var found bool
	for _, p := range receipt.Plugins {
		if p.Path == OpenCodePluginPath {
			found = true
			if p.Status != "drifted" {
				t.Fatalf("guard plugin receipt status = %q, want \"drifted\": %+v", p.Status, p)
			}
		}
	}
	if !found {
		t.Fatalf("receipt.Plugins missing %s: %+v", OpenCodePluginPath, receipt.Plugins)
	}
}

// T2: Build(write) repairs a structurally broken guard from the canonical asset.
func TestBuildWriteRepairsBrokenGuard(t *testing.T) {
	root := pluginFixture(t)
	// Fails the structural check: no leaseref, no hooks, nothing.
	write(t, root, OpenCodePluginPath, "export default function() {}\n")

	if _, err := Build(root, true); err != nil {
		t.Fatalf("Build(write) failed: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(OpenCodePluginPath)))
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != DefaultOpenCodePlugin {
		t.Fatalf("broken guard was not repaired to canonical DefaultOpenCodePlugin (got %d bytes, want %d)", len(b), len(DefaultOpenCodePlugin))
	}
}

// T3: parity reports a drifting-but-valid plugin as a non-zero gap with status
// "drifted"; the clean case (canonical guard + canonical grep) is zero-gap with
// both receipts "ok".
func TestParityReportsDriftedPluginNonZeroGap(t *testing.T) {
	root := pluginFixture(t)
	if _, err := Build(root, true); err != nil {
		t.Fatalf("initial sync failed: %v", err)
	}
	write(t, root, openCodeGrepPath, defaultOpenCodeGrep)
	write(t, root, OpenCodePluginPath, driftedGuard())

	receipt, err := Build(root, false)
	if err != nil {
		t.Fatalf("Build(parity) failed: %v", err)
	}
	if receipt.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=false with a drifted plugin, got true: %+v", receipt.Plugins)
	}
	var guard, grep PluginReceipt
	for _, p := range receipt.Plugins {
		switch p.Path {
		case OpenCodePluginPath:
			guard = p
		case openCodeGrepPath:
			grep = p
		}
	}
	if guard.Status != "drifted" {
		t.Fatalf("guard plugin receipt = %+v, want status \"drifted\"", guard)
	}
	if grep.Status != "ok" {
		t.Fatalf("grep plugin receipt = %+v, want status \"ok\"", grep)
	}

	// Clean case: canonical guard + canonical grep -> zero gaps, both ok.
	// Seed both canonical assets up front: Build(write) only runs the OpenCode
	// sync when an .opencode dir or plugin asset already exists.
	cleanRoot := pluginFixture(t)
	write(t, cleanRoot, OpenCodePluginPath, DefaultOpenCodePlugin)
	write(t, cleanRoot, openCodeGrepPath, defaultOpenCodeGrep)
	if _, err := Build(cleanRoot, true); err != nil {
		t.Fatalf("clean initial sync failed: %v", err)
	}
	clean, err := Build(cleanRoot, false)
	if err != nil {
		t.Fatalf("clean parity failed: %v", err)
	}
	if !clean.ZeroUnexplainedGaps {
		t.Fatalf("expected ZeroUnexplainedGaps=true on canonical plugins, got false: %+v", clean.Plugins)
	}
	for _, p := range clean.Plugins {
		if p.Status != "ok" {
			t.Fatalf("clean plugin receipt not ok: %+v", p)
		}
	}
}

// T4: background EnsureSync cannot revert a drifting-but-valid guard; drift
// alone does not arm a plugin sync and the call stays error-free.
func TestEnsureSyncDoesNotRevertDriftingGuard(t *testing.T) {
	root := pluginFixture(t)
	if _, err := Build(root, true); err != nil {
		t.Fatalf("initial sync failed: %v", err)
	}
	drifted := driftedGuard()
	write(t, root, OpenCodePluginPath, drifted)

	if _, _, err := EnsureSync(root); err != nil {
		t.Fatalf("EnsureSync failed on drifting-but-valid guard: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(OpenCodePluginPath)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b, []byte(drifted)) {
		t.Fatalf("background EnsureSync reverted the drifting guard (got %d bytes, want %d)", len(b), len(drifted))
	}
}

// T5: syncOpenCodeGrep installs fak-grep.js only where absent; a present
// divergent copy survives SyncOpenCodePlugin byte-for-byte and VerifyOpenCodePlugin
// reports it through the drift sentinel.
func TestSyncOpenCodeGrepNeverClobbersPresentPlugin(t *testing.T) {
	root := pluginFixture(t)
	write(t, root, OpenCodePluginPath, DefaultOpenCodePlugin)
	divergent := []byte(defaultOpenCodeGrep)
	divergent[0] ^= 1 // "import" -> "jmport": one byte off canonical
	write(t, root, openCodeGrepPath, string(divergent))

	if err := SyncOpenCodePlugin(root); err != nil {
		t.Fatalf("SyncOpenCodePlugin failed: %v", err)
	}
	b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(openCodeGrepPath)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b, divergent) {
		t.Fatalf("present fak-grep.js was clobbered by sync (got %d bytes, want %d)", len(b), len(divergent))
	}
	if err := VerifyOpenCodePlugin(root); !errors.Is(err, ErrOpenCodePluginDrifted) {
		t.Fatalf("VerifyOpenCodePlugin = %v, want wrapped ErrOpenCodePluginDrifted", err)
	}

	// Install case: an absent grep is written from the canonical asset.
	if err := os.Remove(filepath.Join(root, filepath.FromSlash(openCodeGrepPath))); err != nil {
		t.Fatal(err)
	}
	if err := SyncOpenCodePlugin(root); err != nil {
		t.Fatalf("SyncOpenCodePlugin install failed: %v", err)
	}
	installed, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(openCodeGrepPath)))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(installed, []byte(defaultOpenCodeGrep)) {
		t.Fatalf("reinstalled fak-grep.js differs from canonical defaultOpenCodeGrep (got %d bytes, want %d)", len(installed), len(defaultOpenCodeGrep))
	}
}

// T6: VerifyOpenCodePlugin is clean on canonical assets, wraps the drift
// sentinel on a drifting-but-valid guard, and reports a plain (non-drift)
// error when the guard file is missing.
func TestVerifyOpenCodePluginDetectsDrift(t *testing.T) {
	root := pluginFixture(t)
	write(t, root, OpenCodePluginPath, DefaultOpenCodePlugin)
	write(t, root, openCodeGrepPath, defaultOpenCodeGrep)
	if err := VerifyOpenCodePlugin(root); err != nil {
		t.Fatalf("canonical guard+grep must verify clean, got: %v", err)
	}

	write(t, root, OpenCodePluginPath, driftedGuard())
	err := VerifyOpenCodePlugin(root)
	if err == nil {
		t.Fatal("expected drift error for non-canonical guard, got nil")
	}
	if !errors.Is(err, ErrOpenCodePluginDrifted) {
		t.Fatalf("drift error = %v, want wrapped ErrOpenCodePluginDrifted", err)
	}

	if err := os.Remove(filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))); err != nil {
		t.Fatal(err)
	}
	err = VerifyOpenCodePlugin(root)
	if err == nil {
		t.Fatal("expected error for missing guard, got nil")
	}
	if errors.Is(err, ErrOpenCodePluginDrifted) {
		t.Fatalf("missing guard must not report the drift sentinel: %v", err)
	}
}

// T7 (contract #3): EnsureOpenCodePlugin tolerates drift-class errors in both
// autoSync modes and leaves the local copy untouched.
func TestEnsureOpenCodePluginToleratesDrift(t *testing.T) {
	root := pluginFixture(t)
	drifted := driftedGuard()
	write(t, root, OpenCodePluginPath, drifted)
	write(t, root, openCodeGrepPath, defaultOpenCodeGrep)

	for _, autoSync := range []bool{false, true} {
		if err := EnsureOpenCodePlugin(root, autoSync); err != nil {
			t.Fatalf("EnsureOpenCodePlugin(autoSync=%v) = %v, want nil (drift tolerated)", autoSync, err)
		}
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(OpenCodePluginPath)))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b, []byte(drifted)) {
			t.Fatalf("EnsureOpenCodePlugin(autoSync=%v) touched the drifting guard (got %d bytes, want %d)", autoSync, len(b), len(drifted))
		}
	}
}
