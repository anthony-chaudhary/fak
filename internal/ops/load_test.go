package ops

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

// fixedClock returns a clock the test advances explicitly, so the tick's
// due-phase logic (and therefore "was the expensive tier considered?") is
// deterministic with no sleeping.
func fixedClock(start time.Time) (func() time.Time, func(time.Duration)) {
	cur := start
	return func() time.Time { return cur }, func(d time.Duration) { cur = cur.Add(d) }
}

// hermeticOpsConfig returns a Config and environment for a test engine whose
// Tick cannot touch anything outside its temp root.
//
// A plain os.Setenv-or-not is NOT enough: Tick's storage tier resolves the Go
// build cache from the environment (#GOCACHE) and its process tier can ENACT a
// kill, so an unscoped Tick in a unit test prunes the operator's real build
// cache and can terminate unrelated host processes on a shared machine. See
// issue #14485 for the root-scoping gap this works around.
func hermeticOpsConfig(t *testing.T, repoRoot string) Config {
	t.Helper()
	// GOCACHE=off is the supported seam that makes GoCacheRootFromEnv return ""
	// (internal/treedoctor), so reclaimTier2 returns before it walks anything.
	t.Setenv("GOCACHE", "off")
	// Scopes tier 3's <TEMP>/opencode prune to a temp tree.
	t.Setenv("TEMP", t.TempDir())

	cfg := DefaultConfig()
	// Never let a unit test enact a kill on the shared host.
	cfg.OrphanReapEnabled = false
	return cfg
}

// newHermeticTestEngine builds an engine whose Tick is confined to repoRoot.
func newHermeticTestEngine(t *testing.T, repoRoot string) *Engine {
	t.Helper()
	e, err := NewEngine(repoRoot, hermeticOpsConfig(t, repoRoot))
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	return e
}

// TestLoadCycleRecordsEveryTick is the spine witness for #14474.
//
// The estimate is dominated by filesystem scanning in the storage/scratch
// tiers and is deliberately pessimistic: on a quiet host it runs in a few
// seconds, but on a contended shared host the same work has been observed at
// 27s. That variance is itself evidence for the root-scoping defect in #14484.
//
// The defect it pins: before this change a tick that mutated nothing wrote no
// ledger row at all, so `fak ops status`'s recent_events count could not tell a
// healthy quiet daemon from a dead one. The assertion is that EVERY tick —
// productive or idle — is accounted as one fak-ops-load/1 row, that the idle
// tick's own CPU is measured rather than assumed free, and that a phase that
// was not due is still recorded as considered-but-skipped instead of vanishing.
//
// fak-test:runtime medium est=10s lane=default
func TestLoadCycleRecordsEveryTick(t *testing.T) {
	repo := setupTestAssetRepo(t)
	engine := newHermeticTestEngine(t, repo)

	// Deterministic CPU reader: 40ms on the first reading, 65ms on the second,
	// so the second (idle) tick has a measured, non-zero self cost.
	var cpu time.Duration = 40 * time.Millisecond
	engine.SetSelfCostFunc(func() (time.Duration, bool) { return cpu, true })

	clock, advance := fixedClock(time.Date(2026, 11, 17, 12, 0, 0, 0, time.UTC))
	engine.SetClock(clock)

	ctx := context.Background()

	// --- Tick 1: productive. The asset adapters do not exist yet, so the asset
	// sync phase mutates state on this repo (see TestEngineTickAssetSync).
	if err := engine.Tick(ctx, true); err != nil {
		t.Fatalf("tick 1: %v", err)
	}

	// --- Tick 2: idle. All tiers are forced, but every phase now finds nothing
	// to do, and no fak-ops-event/1 row is appended for it.
	cpu = 65 * time.Millisecond
	advance(2 * time.Minute)
	if err := engine.Tick(ctx, true); err != nil {
		t.Fatalf("tick 2: %v", err)
	}

	cycles, err := engine.LoadLedger.Read()
	if err != nil {
		t.Fatalf("read load ledger: %v", err)
	}
	if len(cycles) != 2 {
		t.Fatalf("expected one accounted cycle per tick (2), got %d", len(cycles))
	}

	first, second := cycles[0], cycles[1]

	// Both rows are the accounted schema, with monotonic ordinals.
	for i, c := range cycles {
		if c.Schema != LoadSchemaV1 {
			t.Errorf("cycle %d: schema = %q, want %q", i, c.Schema, LoadSchemaV1)
		}
		if c.Tick != uint64(i+1) {
			t.Errorf("cycle %d: tick = %d, want %d", i, c.Tick, i+1)
		}
		if err := c.Validate(); err != nil {
			t.Errorf("cycle %d failed its own contract: %v", i, err)
		}
	}

	// The productive tick is classified PRODUCTIVE and names its mutations.
	if first.Verdict != LoadProductive {
		t.Errorf("tick 1 verdict = %q, want %q", first.Verdict, LoadProductive)
	}
	if first.Mutations == 0 {
		t.Errorf("tick 1: expected at least one mutation, got 0")
	}

	// The idle tick is a first-class accounted row, not an absent one.
	if second.Verdict != LoadIdle {
		t.Errorf("tick 2 verdict = %q, want %q (an idle tick is still a fact)", second.Verdict, LoadIdle)
	}

	// The idle tick's own load is MEASURED, not assumed free: the CPU reader
	// advanced 25ms between the two ticks, so the second row must carry it.
	if !second.HaveCPU {
		t.Fatalf("tick 2: expected a measured CPU axis (HaveCPU), got unmeasured")
	}
	if !second.HaveSelfCPU || second.SelfCPUMillis <= 0 {
		t.Fatalf("tick 2: expected non-zero measured self CPU, got have=%v ms=%d",
			second.HaveSelfCPU, second.SelfCPUMillis)
	}
	if got := second.SelfCPUMillis; got != 25 {
		t.Errorf("tick 2 self CPU = %dms, want 25ms (delta between the two readings)", got)
	}

	// The phases that were considered are recorded even when they were not due,
	// so "the expensive tier was skipped" is visible rather than absent.
	if len(second.Phases) == 0 {
		t.Fatalf("tick 2: expected the considered phases to be recorded, got none")
	}
	names := map[string]PhaseLoad{}
	for _, p := range second.Phases {
		names[p.Phase] = p
	}
	for _, want := range []string{"locks_worktrees", "process_runaways", "asset_sync", "storage_reclaim"} {
		if _, ok := names[want]; !ok {
			t.Errorf("tick 2: phase %q was not accounted at all", want)
		}
	}

	// A phase that was considered but mutated nothing must be distinguishable
	// from one that did: idle-with-cost is the exact signal the pause policy
	// (#14477) consumes.
	for _, p := range second.Phases {
		if p.Mutated {
			t.Errorf("tick 2: phase %q reported a mutation on a tick with none", p.Phase)
		}
	}

	// The tick ledger must NOT have gained rows for the idle tick: the load
	// ledger is a separate schema in a separate file precisely so the existing
	// fak-ops-event/1 readers (fak ops log / status recent_events) are untouched.
	events, err := engine.Ledger.QueryEvents(24 * time.Hour)
	if err != nil {
		t.Fatalf("query event ledger: %v", err)
	}
	for _, ev := range events {
		if ev.ActionType == "" {
			t.Fatalf("a load row leaked into the event ledger as an empty ActionType")
		}
	}

	// The two schemas live in different files; assert that on disk, since
	// co-locating them would let json.Decoder silently decode a load row as an
	// Event with an empty ActionType.
	if _, err := os.Stat(filepath.Join(repo, filepath.FromSlash(DefaultLoadLedgerRel))); err != nil {
		t.Fatalf("load ledger not written where declared: %v", err)
	}
	if DefaultLoadLedgerPath(repo) == DefaultLedgerPath(repo) {
		t.Fatalf("load ledger and event ledger must not share a path")
	}
}

// TestLoadCycleContinuesTickOrdinalAcrossRestart pins that the ordinal is
// derived from the ledger, so a restarted daemon does not re-issue tick 1 and
// make a gap look like an unrecorded tick.
func TestLoadCycleContinuesTickOrdinalAcrossRestart(t *testing.T) {
	repo := setupTestAssetRepo(t)
	ctx := context.Background()

	e1 := newHermeticTestEngine(t, repo)
	e1.SetSelfCostFunc(func() (time.Duration, bool) { return 10 * time.Millisecond, true })
	if err := e1.Tick(ctx, true); err != nil {
		t.Fatalf("tick: %v", err)
	}

	// A fresh engine over the same root stands in for a restarted daemon.
	e2 := newHermeticTestEngine(t, repo)
	e2.SetSelfCostFunc(func() (time.Duration, bool) { return 20 * time.Millisecond, true })
	if err := e2.Tick(ctx, true); err != nil {
		t.Fatalf("tick after restart: %v", err)
	}

	cycles, err := e2.LoadLedger.Read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(cycles) != 2 {
		t.Fatalf("expected 2 cycles, got %d", len(cycles))
	}
	if cycles[1].Tick != 2 {
		t.Errorf("tick after restart = %d, want 2 (ordinal must continue, not reset)", cycles[1].Tick)
	}
}

// TestLoadCycleUnmeasuredCPUIsNotFree pins the presence bit: a host whose CPU
// reader is unavailable must record an UNMEASURED axis rather than a zero cost,
// which would make an expensive tick look free to the pause policy.
func TestLoadCycleUnmeasuredCPUIsNotFree(t *testing.T) {
	repo := t.TempDir()
	engine := newHermeticTestEngine(t, repo)
	engine.SetSelfCostFunc(func() (time.Duration, bool) { return 0, false })

	if err := engine.Tick(context.Background(), true); err != nil {
		t.Fatalf("tick: %v", err)
	}
	cycles, err := engine.LoadLedger.Read()
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(cycles) != 1 {
		t.Fatalf("expected 1 cycle, got %d", len(cycles))
	}
	if cycles[0].HaveCPU || cycles[0].HaveSelfCPU {
		t.Errorf("unreadable CPU axis must not be reported as measured: %+v", cycles[0])
	}
	if cycles[0].SelfCPUMillis != 0 {
		t.Errorf("unreadable CPU axis must report 0 ms, got %d", cycles[0].SelfCPUMillis)
	}
}

// TestLoadLedgerRefusesOutOfVocabularyVerdict pins that the verdict is a closed
// set, so a typo cannot invent a new state that no policy understands.
func TestLoadLedgerRefusesOutOfVocabularyVerdict(t *testing.T) {
	led, err := OpenLoadLedger(filepath.Join(t.TempDir(), "ops-load.jsonl"))
	if err != nil {
		t.Fatalf("OpenLoadLedger: %v", err)
	}
	bad := LoadCycle{Schema: LoadSchemaV1, Tick: 1, Verdict: LoadVerdict("SLEEPY")}
	if err := led.Record(bad); err == nil {
		t.Fatalf("expected an out-of-vocabulary verdict to be refused")
	}
	good := LoadCycle{Schema: LoadSchemaV1, Tick: 1, Verdict: LoadIdle}
	if err := led.Record(good); err != nil {
		t.Fatalf("expected a valid cycle to record: %v", err)
	}
	if got, err := led.Count(); err != nil || got != 1 {
		t.Fatalf("Count = %d, err = %v; want 1, nil", got, err)
	}
}

type fakeHostSampler struct {
	sample harnessres.HostSample
	err    error
	calls  int
}

func (f *fakeHostSampler) SampleHost() (harnessres.HostSample, error) {
	f.calls++
	return f.sample, f.err
}

// TestHostSampleSeam pins the injectable host-load sampler seam (#14475):
// MEASURED carries RAM/PSI through into fak-ops-load/1, UNKNOWN and ERROR are
// distinguishable from MEASURED and from each other, each tick samples once,
// and the real harnessres reader does not panic on the host OS.
func TestHostSampleSeam(t *testing.T) {
	repo := t.TempDir()
	engine := newHermeticTestEngine(t, repo)
	engine.SetSelfCostFunc(func() (time.Duration, bool) { return 10 * time.Millisecond, true })

	clock, advance := fixedClock(time.Date(2026, 11, 27, 8, 0, 0, 0, time.UTC))
	engine.SetClock(clock)
	ctx := context.Background()

	// 1. MEASURED: fake RAM/PSI values propagate into fak-ops-load/1, sampled once.
	measured := &fakeHostSampler{
		sample: harnessres.HostSample{
			TotalRAMBytes: 64 << 30,
			AvailRAMBytes: 16 << 30,
			HaveRAM:       true,
			PSIAvg10:      4.5,
			PSIAvg60:      2.0,
			PSIAvg300:     1.0,
			PSISomeTotal:  150 * time.Millisecond,
			PSIFullTotal:  25 * time.Millisecond,
			HavePSI:       true,
		},
	}
	engine.SetHostSampler(measured)
	if err := engine.Tick(ctx, false); err != nil {
		t.Fatalf("tick 1 (MEASURED): %v", err)
	}
	if measured.calls != 1 {
		t.Fatalf("expected 1 host sample per tick, got %d", measured.calls)
	}

	// 2. UNKNOWN: HaveRAM=false without an error is UNKNOWN, not MEASURED or ERROR.
	advance(2 * time.Minute)
	unknown := &fakeHostSampler{
		sample: harnessres.HostSample{HaveRAM: false},
	}
	engine.SetHostSampler(unknown)
	if err := engine.Tick(ctx, false); err != nil {
		t.Fatalf("tick 2 (UNKNOWN): %v", err)
	}
	if unknown.calls != 1 {
		t.Fatalf("expected 1 host sample per tick, got %d", unknown.calls)
	}

	// 3. ERROR: reader error is ERROR, distinct from UNKNOWN and MEASURED.
	advance(2 * time.Minute)
	errored := &fakeHostSampler{
		err: errors.New("procfs unavailable"),
	}
	engine.SetHostSampler(errored)
	if err := engine.Tick(ctx, false); err != nil {
		t.Fatalf("tick 3 (ERROR): %v", err)
	}
	if errored.calls != 1 {
		t.Fatalf("expected 1 host sample per tick, got %d", errored.calls)
	}

	cycles, err := engine.LoadLedger.Read()
	if err != nil {
		t.Fatalf("read load ledger: %v", err)
	}
	if len(cycles) != 3 {
		t.Fatalf("expected 3 cycles, got %d", len(cycles))
	}

	h0 := cycles[0].Host
	if h0 == nil || h0.State != HostSampleMeasured {
		t.Fatalf("cycle 0 host = %+v, want state %q", h0, HostSampleMeasured)
	}
	if !h0.HaveRAM || h0.TotalRAMBytes != 64<<30 || h0.AvailRAMBytes != 16<<30 {
		t.Errorf("cycle 0 RAM mismatch: %+v", h0)
	}
	if !h0.HavePSI || h0.PSIAvg10 != 4.5 || h0.PSIAvg60 != 2.0 || h0.PSIAvg300 != 1.0 || h0.PSISomeTotalMS != 150 || h0.PSIFullTotalMS != 25 {
		t.Errorf("cycle 0 PSI mismatch: %+v", h0)
	}

	h1 := cycles[1].Host
	if h1 == nil || h1.State != HostSampleUnknown {
		t.Fatalf("cycle 1 host = %+v, want state %q", h1, HostSampleUnknown)
	}
	if h1.Error != "" {
		t.Errorf("cycle 1 UNKNOWN must not carry error string, got %q", h1.Error)
	}

	h2 := cycles[2].Host
	if h2 == nil || h2.State != HostSampleError {
		t.Fatalf("cycle 2 host = %+v, want state %q", h2, HostSampleError)
	}
	if h2.Error == "" {
		t.Errorf("cycle 2 ERROR must carry diagnostic error")
	}

	// Out-of-vocabulary host state is rejected at write time.
	bad := LoadCycle{
		Schema:  LoadSchemaV1,
		Tick:    4,
		Verdict: LoadIdle,
		Host:    &HostLoad{State: HostSampleState("STRUGGLING")},
	}
	if err := engine.LoadLedger.Record(bad); err == nil {
		t.Fatalf("expected out-of-vocabulary host state to be refused")
	}

	// 4. Real harnessres smoke call must not panic on the host OS.
	realSample, realErr := (HarnessHostSampler{}).SampleHost()
	if realErr != nil {
		t.Fatalf("HarnessHostSampler.SampleHost returned unexpected error: %v", realErr)
	}
	_ = realSample
}
