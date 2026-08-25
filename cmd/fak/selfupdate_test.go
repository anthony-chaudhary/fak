package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
	"github.com/anthony-chaudhary/fak/internal/versionskew"
)

const selfUpdateProbeHelperEnv = "GO_WANT_SELFUPDATE_PROBE_HELPER"
const selfUpdateProbeHelperRev = "1234567890abcdef1234567890abcdef12345678"

func init() {
	if os.Getenv(selfUpdateProbeHelperEnv) != "1" {
		return
	}
	_, _ = os.Stdout.WriteString("fak test helper\nbuild: " + selfUpdateProbeHelperRev + "\n")
	os.Exit(0)
}

// TestSelfUpdateShouldBuild pins the proceed decision, and in particular the case binstamp
// alone gets WRONG: a clean local binary that is AHEAD of origin/main. Under the old
// `verdict == binstamp.Stale` rule that case (rev differs => Stale) rebuilt origin/main OVER
// the newer binary; keying SELF mode off versionskew.Skewed makes Ahead a no-op. This is the
// "previously-collapsed case now drives a distinct decision" the wiring exists to produce.
func TestSelfUpdateShouldBuild(t *testing.T) {
	cases := []struct {
		name  string
		force bool
		fleet bool
		bin   binstamp.Freshness
		skew  versionskew.Verdict
		want  bool
	}{
		// SELF mode: ONLY a provably-behind skew rebuilds.
		{"self behind rebuilds", false, false, binstamp.Stale, versionskew.Skewed, true},
		{"self ahead does NOT rebuild (the fix)", false, false, binstamp.Stale, versionskew.Ahead, false},
		{"self diverged does NOT rebuild", false, false, binstamp.Stale, versionskew.Diverged, false},
		{"self fresh no-op", false, false, binstamp.Fresh, versionskew.Fresh, false},
		{"self dirty no-op", false, false, binstamp.Unknown, versionskew.Dirty, false},
		{"self unstamped no-op", false, false, binstamp.Unknown, versionskew.Unstamped, false},
		{"self unknown no-op", false, false, binstamp.Unknown, versionskew.Unknown, false},
		{"self force overrides a fresh binary", true, false, binstamp.Fresh, versionskew.Fresh, true},
		// FLEET mode: rebuild unless binstamp proves Fresh — regardless of the skew token.
		{"fleet not-fresh rebuilds", false, true, binstamp.Unknown, versionskew.Unknown, true},
		{"fleet behind rebuilds", false, true, binstamp.Stale, versionskew.Skewed, true},
		{"fleet fresh no-op", false, true, binstamp.Fresh, versionskew.Fresh, false},
		{"fleet fresh + force rebuilds", true, true, binstamp.Fresh, versionskew.Fresh, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := selfUpdateShouldBuild(c.force, c.fleet, c.bin, c.skew); got != c.want {
				t.Fatalf("selfUpdateShouldBuild(force=%v fleet=%v bin=%v skew=%v) = %v, want %v",
					c.force, c.fleet, c.bin, c.skew, got, c.want)
			}
		})
	}
}

// TestSelfUpdateSiblingsIncludesInTreeFleetBinary pins the fix for the stale-fleet-binary lag:
// `self-update --target X` converged X and nothing else, while every dispatcher-launched worker
// runs `<root>/tools/.bin/fak[.exe] guard -- claude …` — the path
// tools/dispatch_worker.py resolve_fak_bin prefers AHEAD of PATH. Because that in-tree file
// existed, PATH was never consulted, so the fleet ran a binary no updater targeted and the tick
// still exited 0. The sibling set must therefore contain the in-tree fleet binary.
func TestSelfUpdateSiblingsIncludesInTreeFleetBinary(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "tools", ".bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fleetBin := filepath.Join(binDir, "fak"+exeSuffix())
	if err := os.WriteFile(fleetBin, []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}
	target := filepath.Join(root, "installed"+exeSuffix())
	if err := os.WriteFile(target, []byte("fresh"), 0o755); err != nil {
		t.Fatal(err)
	}

	got := selfUpdateSiblings(root, target)
	found := false
	for _, p := range got {
		if strings.EqualFold(p, fleetBin) {
			found = true
		}
		if strings.EqualFold(p, target) {
			t.Errorf("sibling set must not repeat the primary --target %q: %v", target, got)
		}
	}
	if !found {
		t.Errorf("selfUpdateSiblings(%q, %q) = %v; want it to include the in-tree fleet binary %q",
			root, target, got, fleetBin)
	}
}

// TestSelfUpdateSiblingsSkipsMissingPaths — we converge binaries that already exist; a path that
// is absent is not an install location we should create. With no tools/.bin on disk the only
// sibling is the running test binary itself, never a phantom <root>/tools/.bin entry.
func TestSelfUpdateSiblingsSkipsMissingPaths(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "installed"+exeSuffix())
	if err := os.WriteFile(target, []byte("fresh"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range selfUpdateSiblings(root, target) {
		if strings.Contains(strings.ToLower(p), filepath.Join("tools", ".bin")) {
			t.Errorf("selfUpdateSiblings returned a non-existent path %q", p)
		}
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			t.Errorf("selfUpdateSiblings returned %q which is not an existing file", p)
		}
	}
}

func TestSelfUpdateFakDevNeedsConvergeTriggersWhenPrimaryIsCurrent(t *testing.T) {
	head := strings.Repeat("a", 40)
	probe := func(path string) (string, bool, bool) {
		if strings.Contains(path, "stale") {
			return strings.Repeat("b", 40), false, true
		}
		return head, false, true
	}
	if !selfUpdateFakDevNeedsConverge([]string{"fak-dev-stale"}, head, probe) {
		t.Fatal("stale fak-dev must force an update even when fak itself is current")
	}
	if selfUpdateFakDevNeedsConverge([]string{"fak-dev-current"}, head, probe) {
		t.Fatal("current fak-dev should not force a rebuild")
	}
}

// TestSelfUpdateFakDevTargetsFindsOnlyInstalledCompanions proves product-only hosts stay
// product-only while a side-by-side developer install joins the same convergence cycle.
func TestSelfUpdateFakDevTargetsFindsOnlyInstalledCompanions(t *testing.T) {
	root := t.TempDir()
	binDir := filepath.Join(root, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	fakPath := filepath.Join(binDir, "fak"+exeSuffix())
	if err := os.WriteFile(fakPath, []byte("fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	devPath := filepath.Join(binDir, "fak-dev"+exeSuffix())
	for _, got := range selfUpdateFakDevTargets(root, fakPath) {
		if strings.EqualFold(got, devPath) {
			t.Fatalf("missing companion should not be created: %v", got)
		}
	}
	if err := os.WriteFile(devPath, []byte("stale"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := selfUpdateFakDevTargets(root, fakPath)
	found := false
	for _, candidate := range got {
		if strings.EqualFold(candidate, devPath) {
			found = true
		}
	}
	if !found {
		t.Fatalf("targets=%v; want it to include %s", got, devPath)
	}
}

// TestSelfUpdateProbeReadsOwnPathAfterSwap reproduces the Windows post-swap audit bug. The
// running process still has its old embedded stamp, but invoking its path starts the new bytes.
// The census must read the deployed path or it reports a successful update as divergent.
func TestSelfUpdateProbeReadsOwnPathAfterSwap(t *testing.T) {
	t.Setenv(selfUpdateProbeHelperEnv, "1")
	revision, dirty, attested := selfUpdateProbe(os.Args[0])
	if !attested || dirty || revision != selfUpdateProbeHelperRev {
		t.Fatalf("selfUpdateProbe(own path) = (%q, dirty=%v, attested=%v), want deployed helper stamp %q",
			revision, dirty, attested, selfUpdateProbeHelperRev)
	}
}

// convergeSiblings — the old "is the INVOKER provably fresh?" guard on the sibling swap — is
// gone (#6508). It both over- and under-shot: it re-swapped siblings that were already current
// and never looked at the PATH / Go-bin copies at all. The decision is now per-copy, from the
// role census, and is pinned by TestNeedsConvergeDemandsProofOfFreshness in
// internal/selfinstall — a package whose tests actually build and run.

// TestSelfUpdateSkipOutcome pins the closed outcome vocabulary against the message switch it
// mirrors. The scheduler sees only an exit code, and rc=0 is identical for "installed",
// "already current", "busy" and "--check"; the named outcome is what re-couples the success
// code to whether an update actually happened.
func TestSelfUpdateSkipOutcome(t *testing.T) {
	cases := []struct {
		fleet bool
		skew  versionskew.Verdict
		want  selfUpdateOutcome
	}{
		{true, versionskew.Fresh, outcomeTargetCurrent},
		{true, versionskew.Unknown, outcomeTargetCurrent},
		{false, versionskew.Fresh, outcomeSelfFresh},
		{false, versionskew.Ahead, outcomeSelfAhead},
		{false, versionskew.Dirty, outcomeSelfLocal},
		{false, versionskew.Unstamped, outcomeSelfLocal},
		{false, versionskew.Diverged, outcomeSelfLocal},
		{false, versionskew.Unknown, outcomeSelfUnknown},
	}
	for _, c := range cases {
		if got := selfUpdateSkipOutcome(c.fleet, c.skew); got != c.want {
			t.Errorf("selfUpdateSkipOutcome(fleet=%v, %v) = %q; want %q", c.fleet, c.skew, got, c.want)
		}
	}
}

func TestSelfUpdateReceiptPostures(t *testing.T) {
	oldCorrelation := selfUpdateCorrelationID
	selfUpdateCorrelationID = func() string { return "corr-123" }
	defer func() { selfUpdateCorrelationID = oldCorrelation }()
	selfUpdateReceiptOldRevision = "oldrev"
	selfUpdateReceiptNewRevision = "newrev"
	selfUpdateReceiptTargets = []selfUpdateReceiptTarget{{Role: "primary", Path: filepath.Clean("bin/fak")}}

	cases := []struct {
		name    string
		outcome selfUpdateOutcome
		status  string
		restart bool
		roll    string
	}{
		{"current", outcomeSelfFresh, "current", false, "not_attempted"},
		{"updated", outcomeInstalled, "updated", false, "not_attempted"},
		{"rolled_back", outcomeRolledBack, "rolled_back", false, "succeeded"},
		{"rollback_failed", outcomeRollbackFailed, "rollback_failed", false, "failed"},
		{"busy", outcomeBusy, "busy", false, "not_attempted"},
		{"restart_required", selfUpdateOutcome("restart_required"), "restart_required", true, "not_attempted"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			receipt := newSelfUpdateReceipt(tc.outcome, "bin/fak", "rollback detail")
			encoded, err := json.Marshal(receipt)
			if err != nil {
				t.Fatal(err)
			}
			var parsed map[string]any
			if err := json.Unmarshal(encoded, &parsed); err != nil {
				t.Fatalf("receipt is not JSON: %v\n%s", err, encoded)
			}
			if receipt.Schema != selfUpdateReceiptSchema || receipt.SchemaVersion != 1 || receipt.CorrelationID != "corr-123" {
				t.Fatalf("unstable envelope: %+v", receipt)
			}
			if receipt.Status != tc.status || receipt.RestartRequired != tc.restart || receipt.RollbackStatus != tc.roll {
				t.Fatalf("posture = %+v", receipt)
			}
			if receipt.OldRevision == nil || *receipt.OldRevision != "oldrev" || receipt.NewRevision == nil || *receipt.NewRevision != "newrev" {
				t.Fatalf("revision fields = %+v", receipt)
			}
			if receipt.NextCommand == "" || len(receipt.Targets) != 1 {
				t.Fatalf("action/targets missing: %+v", receipt)
			}
		})
	}
}

func TestEmitSelfUpdateJSONIsOneObjectWithoutProse(t *testing.T) {
	oldCorrelation := selfUpdateCorrelationID
	oldProgress, oldJSON := selfUpdateProgress, selfUpdateJSON
	selfUpdateCorrelationID = func() string { return "corr-123" }
	t.Cleanup(func() {
		selfUpdateCorrelationID = oldCorrelation
		selfUpdateProgress, selfUpdateJSON = oldProgress, oldJSON
	})

	var stdout, stderr strings.Builder
	selfUpdateJSON = &stdout
	selfUpdateProgress = &stderr
	fmt.Fprintln(selfUpdateProgress, "self-update: checking transaction")
	emitSelfUpdateOutcome(outcomeBusy, "bin/fak", "single-flight lock held")
	if strings.Count(stdout.String(), "\n") != 1 {
		t.Fatalf("stdout must contain exactly one JSON line: %q", stdout.String())
	}
	decoder := json.NewDecoder(strings.NewReader(stdout.String()))
	var receipt selfUpdateReceipt
	if err := decoder.Decode(&receipt); err != nil {
		t.Fatalf("stdout contains prose or invalid JSON: %v: %q", err, stdout.String())
	}
	var extra any
	if err := decoder.Decode(&extra); err != io.EOF {
		t.Fatalf("stdout contains more than one object: %v: %q", err, stdout.String())
	}
	if receipt.Status != "busy" {
		t.Fatalf("status = %q", receipt.Status)
	}
	if got := stderr.String(); !strings.Contains(got, "checking transaction") || strings.Contains(stdout.String(), "checking transaction") {
		t.Fatalf("progress routing: stdout=%q stderr=%q", stdout.String(), got)
	}
}
