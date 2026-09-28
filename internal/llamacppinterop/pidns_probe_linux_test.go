//go:build linux

package llamacppinterop

import (
	"fmt"
	"os/exec"
	"slices"
	"strings"
	"testing"
)

// strixWitnessNamespaceFlags are the unshare(1) flags the Strix authority
// witnesses re-exec under: a private user+pid namespace with its own /proc, so
// the authority's global /proc ownership scan sees only the witness's process
// tree rather than the whole host. The namespace is test isolation; it is not
// what the witnesses assert.
var strixWitnessNamespaceFlags = []string{"--user", "--map-current-user", "--pid", "--fork", "--mount-proc"}

// strixWitnessNamespaceArgv returns the unshare(1) argv that runs argv inside
// the witness namespace.
func strixWitnessNamespaceArgv(argv ...string) []string {
	return append(slices.Clone(strixWitnessNamespaceFlags), argv...)
}

// probeStrixWitnessNamespace runs the exact witness namespace flags around
// true(1) and returns nil only when unshare could build that namespace here.
// Any failure means the capability is unavailable: the probe does nothing else.
func probeStrixWitnessNamespace(unshare string) error {
	truePath, err := exec.LookPath("true")
	if err != nil {
		return fmt.Errorf("locate true(1) for the namespace probe: %w", err)
	}
	output, err := exec.Command(unshare, strixWitnessNamespaceArgv(truePath)...).CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s %s: %w: %s", unshare, strings.Join(strixWitnessNamespaceFlags, " "), err, strings.TrimSpace(string(output)))
	}
	return nil
}

// requireStrixWitnessNamespace skips the calling witness when this host cannot
// create an unprivileged user+pid namespace. GitHub's ubuntu-latest runners
// (Ubuntu 24.04, kernel.apparmor_restrict_unprivileged_userns=1) allow the
// unshare(CLONE_NEWUSER) but deny the follow-up uid_map write, so unshare exits
// with "write failed /proc/self/uid_map: Operation not permitted" before the
// witness body runs. Hosts that grant the namespace (root, or unrestricted
// kernels) still run the full witness.
func requireStrixWitnessNamespace(t *testing.T) {
	t.Helper()
	unshare, err := exec.LookPath("unshare")
	if err != nil {
		t.Skipf("Strix authority witness needs unshare(1) for its private pid namespace: %v", err)
	}
	if err := probeStrixWitnessNamespace(unshare); err != nil {
		t.Skipf("Strix authority witness needs an unprivileged user+pid namespace to isolate /proc, unavailable on this host: %v", err)
	}
}

// TestStrixWitnessNamespaceProbeReportsUnshareOutcome pins the probe's
// decision on hosts where the real capability cannot be toggled: a failing
// unshare stand-in must report the namespace unavailable (the skip path), and
// a succeeding one must report it available (the witness runs).
func TestStrixWitnessNamespaceProbeReportsUnshareOutcome(t *testing.T) {
	falsePath, err := exec.LookPath("false")
	if err != nil {
		t.Fatalf("locate false(1): %v", err)
	}
	if err := probeStrixWitnessNamespace(falsePath); err == nil {
		t.Fatal("probe with a failing unshare reported the namespace available")
	} else if !strings.Contains(err.Error(), "--user --map-current-user --pid --fork --mount-proc") {
		t.Fatalf("probe error does not name the witness flags: %v", err)
	}
	truePath, err := exec.LookPath("true")
	if err != nil {
		t.Fatalf("locate true(1): %v", err)
	}
	if err := probeStrixWitnessNamespace(truePath); err != nil {
		t.Fatalf("probe with a succeeding unshare reported the namespace unavailable: %v", err)
	}
}
