package ops

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const fakeRunawayPID = 424242

// hostFixture is a stand-in host outside the repo root: a Go cache over the
// high watermark, an aged temp dump, and a census holding one reapable PID.
type hostFixture struct {
	cacheRoot     string
	cacheEntries  []string
	tempRoot      string
	tempDump      string
	census        func() ([]procguard.Proc, string)
	killed        []int
	recordingKill func(int) (bool, string)
}

func newHostFixture(t *testing.T) *hostFixture {
	t.Helper()
	outside := t.TempDir()
	f := &hostFixture{
		cacheRoot: filepath.Join(outside, "go-build"),
		tempRoot:  filepath.Join(outside, "temp", "opencode"),
	}
	old := time.Now().Add(-30 * 24 * time.Hour)
	if err := os.MkdirAll(filepath.Join(f.cacheRoot, "ab"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, n := range []string{"sentinel-a", "sentinel-d"} {
		p := filepath.Join(f.cacheRoot, "ab", n)
		if err := os.WriteFile(p, []byte(strings.Repeat("x", 4096)), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, old, old); err != nil {
			t.Fatal(err)
		}
		f.cacheEntries = append(f.cacheEntries, p)
	}
	f.tempDump = filepath.Join(f.tempRoot, "operator-dump")
	if err := os.MkdirAll(f.tempDump, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(f.tempDump, old, old); err != nil {
		t.Fatal(err)
	}
	threads := 5000
	f.census = func() ([]procguard.Proc, string) {
		return []procguard.Proc{{PID: fakeRunawayPID, Name: "fak-test-runaway", Threads: &threads}}, ""
	}
	f.recordingKill = func(pid int) (bool, string) {
		f.killed = append(f.killed, pid)
		return true, "recorded only"
	}
	return f
}

func (f *hostFixture) config() Config {
	cfg := DefaultConfig()
	cfg.GoCacheHighBytes = 1
	cfg.GoCacheLowBytes = 1
	cfg.GoCacheMinAge = time.Hour
	cfg.MaxThreads = 100
	cfg.OrphanReapEnabled = true
	return cfg
}

func (f *hostFixture) reach() HostReach {
	return HostReach{
		GoCacheRoot:      func() string { return f.cacheRoot },
		TempRoots:        func() []string { return []string{f.tempRoot} },
		CollectProcesses: f.census,
		EnactProcessReap: true,
	}
}

func phaseByName(t *testing.T, phases []PhaseLoad, name string) PhaseLoad {
	t.Helper()
	for _, p := range phases {
		if p.Phase == name {
			return p
		}
	}
	t.Fatalf("phase %q not recorded in %+v", name, phases)
	return PhaseLoad{}
}

// TestTickDoesNotLeaveItsRoot pins that an engine built with NewEngine over a
// temp root cannot reach the host. The environment is pointed at the fixture
// so a regression back to env resolution is caught here instead of pruning
// the operator's real cache.
//
// fak-test:runtime small est=3s lane=default
func TestTickDoesNotLeaveItsRoot(t *testing.T) {
	repo := setupTestAssetRepo(t)
	f := newHostFixture(t)
	t.Setenv("GOCACHE", f.cacheRoot)
	t.Setenv("TEMP", filepath.Dir(f.tempRoot))

	engine, err := NewEngine(repo, f.config())
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	if engine.Process.Collect != nil || engine.Process.EnactAuthorized ||
		engine.Storage.GoCacheRoot != nil || engine.Storage.TempRoots != nil {
		t.Fatal("NewEngine granted host reach")
	}

	dry, err := engine.Storage.ReclaimCascade(context.Background(), true)
	if err != nil {
		t.Fatalf("dry-run cascade: %v", err)
	}
	if dry.Tier2Bytes != 0 || dry.Tier3Bytes != 0 {
		t.Fatalf("dry run read host paths: tier2=%d tier3=%d actions=%q", dry.Tier2Bytes, dry.Tier3Bytes, dry.Actions)
	}

	// A library caller that hands over a census without kill authority still
	// must not reap.
	engine.Process.Collect = f.census
	engine.Process.Killer = f.recordingKill
	if err := engine.Tick(context.Background(), true); err != nil {
		t.Fatalf("Tick: %v", err)
	}

	if len(f.killed) != 0 {
		t.Errorf("killer called for %v from a confined tick", f.killed)
	}
	for _, p := range f.cacheEntries {
		if _, err := os.Stat(p); err != nil {
			t.Errorf("out-of-root cache entry %s touched: %v", p, err)
		}
	}
	if _, err := os.Stat(filepath.Join(f.cacheRoot, ".fak-gocache-lifecycle.lock")); !os.IsNotExist(err) {
		t.Errorf("confined tick wrote a lifecycle lock into the out-of-root cache: %v", err)
	}
	if _, err := os.Stat(f.tempDump); err != nil {
		t.Errorf("out-of-root temp dump %s pruned: %v", f.tempDump, err)
	}

	cycles, err := engine.LoadLedger.Read()
	if err != nil || len(cycles) != 1 {
		t.Fatalf("load ledger: cycles=%d err=%v", len(cycles), err)
	}
	storage := phaseByName(t, cycles[0].Phases, "storage_reclaim")
	for _, want := range []ScopeSkip{SkipGoCacheUnauthorized, SkipHostTempUnauthorized} {
		if !slices.Contains(storage.Skipped, want) {
			t.Errorf("storage_reclaim skipped = %v, want %s", storage.Skipped, want)
		}
	}
	proc := phaseByName(t, cycles[0].Phases, "process_runaways")
	if !slices.Contains(proc.Skipped, SkipProcessReapUnauthorized) {
		t.Errorf("process_runaways skipped = %v, want %s", proc.Skipped, SkipProcessReapUnauthorized)
	}
}

// TestNewEngineSkipsProcessCensus pins that a confined engine does not even
// scan the host process table, and says so.
//
// fak-test:runtime small est=1s lane=default
func TestNewEngineSkipsProcessCensus(t *testing.T) {
	pm := NewProcessManager(DefaultConfig())
	pm.Killer = func(pid int) (bool, string) {
		t.Fatalf("killer called for %d", pid)
		return false, ""
	}
	res, err := pm.SweepProcessRunaways(context.Background(), false)
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if !slices.Equal(res.Skipped, []ScopeSkip{SkipProcessCensusUnauthorized}) {
		t.Fatalf("skipped = %v, want [%s]", res.Skipped, SkipProcessCensusUnauthorized)
	}
}

// TestHostReachGrantsInjectedSeamsOnly is the positive control: with reach
// granted, the engine acts on exactly the injected cache, temp root and census.
//
// fak-test:runtime small est=3s lane=default
func TestHostReachGrantsInjectedSeamsOnly(t *testing.T) {
	repo := setupTestAssetRepo(t)
	f := newHostFixture(t)
	t.Setenv("GOCACHE", "off")

	engine, err := NewEngineWithHostReach(repo, f.config(), f.reach())
	if err != nil {
		t.Fatalf("NewEngineWithHostReach: %v", err)
	}
	engine.Process.Killer = f.recordingKill

	dry, err := engine.Storage.ReclaimCascade(context.Background(), true)
	if err != nil {
		t.Fatalf("dry-run cascade: %v", err)
	}
	if dry.Tier2Bytes == 0 {
		t.Errorf("granted cache root was not consulted: actions=%q", dry.Actions)
	}
	if len(dry.Skipped) != 0 {
		t.Errorf("granted engine recorded skips %v", dry.Skipped)
	}

	if err := engine.Tick(context.Background(), true); err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if !slices.Equal(f.killed, []int{fakeRunawayPID}) {
		t.Errorf("killed = %v, want [%d]", f.killed, fakeRunawayPID)
	}
	if _, err := os.Stat(f.tempDump); !os.IsNotExist(err) {
		t.Errorf("granted temp root dump %s not pruned: %v", f.tempDump, err)
	}
	cycles, err := engine.LoadLedger.Read()
	if err != nil || len(cycles) != 1 {
		t.Fatalf("load ledger: cycles=%d err=%v", len(cycles), err)
	}
	for _, p := range cycles[0].Phases {
		if len(p.Skipped) != 0 {
			t.Errorf("phase %s recorded skips %v under granted reach", p.Phase, p.Skipped)
		}
	}
}
