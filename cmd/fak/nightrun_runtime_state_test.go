package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/fleettrend"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/nightrun"
)

// TestLiveNightrunTicksKeepTrackedDocsClean is the captured behavior witness for
// #3209. It starts with published docs snapshots in a clean git repository, runs
// one tick through the five Go writer seams that used to target those snapshots,
// and proves git still sees no docs/nightrun change. A path-prefix assertion alone
// would miss a writer accidentally retaining a hard-coded docs default.
func TestLiveNightrunTicksKeepTrackedDocsClean(t *testing.T) {
	root := t.TempDir()
	runGit := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return string(out)
	}

	if err := os.WriteFile(filepath.Join(root, ".gitignore"), []byte("/.fak/\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	trackedDir := filepath.Join(root, "docs", "nightrun")
	if err := os.MkdirAll(trackedDir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{
		"collected.jsonl",
		"cache-savings.jsonl",
		"gateway-usage.jsonl",
		"harness-resources.jsonl",
		"fleet-status-history.jsonl",
	} {
		if err := os.WriteFile(filepath.Join(trackedDir, name), []byte("published snapshot\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	runGit("init", "-q")
	runGit("add", ".gitignore", "docs/nightrun")
	runGit("-c", "user.name=fak test", "-c", "user.email=fak@example.test", "commit", "-q", "-m", "seed")

	defaults := []string{
		nightrun.DefaultLedgerRel,
		cachevaluereport.DefaultSavingsLedgerRel,
		gatewayusageledger.DefaultLedgerRel,
		harnessres.DefaultLedgerRel,
		fleettrend.DefaultLedger,
	}
	for _, rel := range defaults {
		if !strings.HasPrefix(filepath.ToSlash(rel), ".fak/nightrun/") {
			t.Fatalf("live ledger default %q escapes ignored runtime state", rel)
		}
	}
	runtimeDir := filepath.Join(root, ".fak", "nightrun")
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 7, 11, 12, 0, 0, 0, time.UTC)
	if err := nightrun.AppendLedgerFile(filepath.Join(root, nightrun.DefaultLedgerRel), nightrun.CollectRow{
		Schema: nightrun.CollectSchema, Date: "2026-07-11", TaskID: "clean-tick", Outcome: string(nightrun.OutcomeCollected), GeneratedAt: now.Format(time.RFC3339),
	}); err != nil {
		t.Fatal(err)
	}
	if err := cachevaluereport.AppendSavings(filepath.Join(root, cachevaluereport.DefaultSavingsLedgerRel), cachevaluereport.SavingsRow{
		Schema: cachevaluereport.SavingsLedgerSchema, Date: "2026-07-11", GeneratedAt: now.Format(time.RFC3339),
	}); err != nil {
		t.Fatal(err)
	}
	if err := gatewayusageledger.Append(filepath.Join(root, gatewayusageledger.DefaultLedgerRel), gatewayusageledger.NewRow(
		"exit", "guard", "test", "clean-tick", 0, nil, gatewayusageledger.Counters{}, now,
	)); err != nil {
		t.Fatal(err)
	}
	if err := appendHarnessResourcesTo(filepath.Join(root, harnessres.DefaultLedgerRel), "guard", "test", "clean-tick", harnessres.Snapshot{}, now); err != nil {
		t.Fatal(err)
	}
	if _, err := fleettrend.Append(filepath.Join(root, fleettrend.DefaultLedger), map[string]float64{"sessions": 1}, now.Format(time.RFC3339), fleettrend.DefaultCap); err != nil {
		t.Fatal(err)
	}

	if got := strings.TrimSpace(runGit("status", "--porcelain", "--", "docs/nightrun")); got != "" {
		t.Fatalf("background writer tick dirtied tracked docs/nightrun:\n%s", got)
	}
}
