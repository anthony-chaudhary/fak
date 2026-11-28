package main

// fak-test:runtime medium est=10s lane=default

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/ops"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// e2eSampler is the injected host sampler (#14482): CI never reads the real host.
type e2eSampler struct {
	s   harnessres.HostSample
	err error
}

func (f *e2eSampler) SampleHost() (harnessres.HostSample, error) { return f.s, f.err }

func e2eHost(usedFrac float64) *e2eSampler {
	const total = uint64(16 << 30)
	return &e2eSampler{s: harnessres.HostSample{
		TotalRAMBytes: total, HaveRAM: true,
		AvailRAMBytes: uint64(float64(total) * (1 - usedFrac)),
	}}
}

// e2eMeter charges a deterministic self-CPU cost per tick: heavy when the pause
// gate let deferrable junkwork RUN, light when it SKIPped. Tick only runs the
// deferrable asset_sync phase when the same directive says RUN, so the model is
// tied to the real gate decision rather than asserted separately.
type e2eMeter struct {
	inner ops.PauseEvaluator
	total time.Duration
	ran   int
	skip  int
}

func (m *e2eMeter) Decide(c ops.LoadCycle, p ops.PauseState, cfg ops.Config) ops.PauseDirective {
	d := m.inner.Decide(c, p, cfg)
	if d.JunkworkAction == ops.ActionSkip {
		m.total += 5 * time.Millisecond
		m.skip++
	} else {
		m.total += 120 * time.Millisecond
		m.ran++
	}
	return d
}
func (m *e2eMeter) cost() (time.Duration, bool) { return m.total, true }

type e2eRig struct {
	root    string
	engine  *ops.Engine
	now     time.Time
	reaped  []int
	meter   *e2eMeter
	sampler *e2eSampler
}

var e2eBase = time.Date(2026, 11, 27, 9, 0, 0, 0, time.UTC)

func newE2ERig(t *testing.T, root string, cfg ops.Config, s *e2eSampler, at time.Time) *e2eRig {
	t.Helper()
	r := &e2eRig{root: root, now: at, sampler: s}
	// A real, parity-clean project-assets tree so the deferrable asset_sync phase
	// runs its real path to success (and is idempotent after the first sync).
	seedProjectAssets(t, root)
	procs := func() ([]procguard.Proc, string) { return nil, "" }
	e, err := ops.NewEngineWithHostReach(root, cfg, ops.HostReach{CollectProcesses: procs, EnactProcessReap: true})
	if err != nil {
		t.Fatal(err)
	}
	r.engine = e
	e.SetHostSampler(s)
	e.SetClock(func() time.Time { return r.now })
	e.Workspace.ProcessAlive = func(int) bool { return false }
	e.Process.Killer = func(pid int) (bool, string) { r.reaped = append(r.reaped, pid); return true, "killed" }
	r.meter = &e2eMeter{inner: ops.DefaultPauseEvaluator{}}
	e.SetPauseEvaluator(r.meter)
	e.SetSelfCostFunc(r.meter.cost)
	return r
}

// plant creates a genuinely stale lock and a genuinely runaway (over MaxThreads) census row, the only process class the ops loop reaps.
func (r *e2eRig) plant(t *testing.T) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(r.root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(r.root, ".git", "fak-commit.lock"), []byte("9999999"), 0o644); err != nil {
		t.Fatal(err)
	}
	threads := 50000
	r.engine.Process.Collect = func() ([]procguard.Proc, string) {
		return []procguard.Proc{{PID: 4242, Name: "runaway", Threads: &threads}}, ""
	}
}

func (r *e2eRig) tick(t *testing.T, force bool, advance time.Duration) ops.LoadCycle {
	t.Helper()
	r.now = r.now.Add(advance)
	if err := r.engine.Tick(context.Background(), force); err != nil {
		t.Fatalf("tick: %v", err)
	}
	rows, err := r.engine.LoadLedger.Read()
	if err != nil || len(rows) == 0 {
		t.Fatalf("load ledger: %v rows=%d", err, len(rows))
	}
	return rows[len(rows)-1]
}

func phaseOf(c ops.LoadCycle, name string) ops.PhaseLoad {
	for _, p := range c.Phases {
		if p.Phase == name {
			return p
		}
	}
	return ops.PhaseLoad{}
}

func pressureCfg() ops.Config {
	cfg := ops.DefaultConfig()
	cfg.ReactivePauseEnabled = true
	cfg.ScratchTTL = time.Hour
	return cfg
}

func readPause(t *testing.T, root string) ops.PauseState {
	t.Helper()
	st, err := ops.OpenPauseStore(ops.DefaultPauseStatePath(root))
	if err != nil {
		t.Fatal(err)
	}
	return st.ReadFailSafe()
}

func TestOpsDaemonLoadAccountingE2E(t *testing.T) {
	t.Run("relief_never_gated_and_pause_durable_across_restart", func(t *testing.T) {
		root := t.TempDir()
		host := e2eHost(0.10)
		r := newE2ERig(t, root, pressureCfg(), host, e2eBase)

		// Calm host: junkwork runs, stale lock + orphan swept -> PRODUCTIVE.
		r.plant(t)
		c1 := r.tick(t, true, 0)
		if c1.Verdict != ops.LoadProductive || !phaseOf(c1, "asset_sync").Ran {
			t.Fatalf("calm tick: verdict=%s asset_ran=%v", c1.Verdict, phaseOf(c1, "asset_sync").Ran)
		}
		// Calm, nothing to do -> IDLE.
		r.engine.Process.Collect = func() ([]procguard.Proc, string) { return nil, "" } // census is clean now
		c2 := r.tick(t, true, time.Minute)
		if c2.Verdict != ops.LoadIdle {
			t.Fatalf("idle tick verdict=%s", c2.Verdict)
		}

		// (1) Heavy pressure: a NEW stale lock + orphan must still be swept while junkwork is skipped.
		host.s = e2eHost(0.97).s
		r.reaped = nil
		r.plant(t)
		c3 := r.tick(t, true, time.Minute)
		if !r.engine.Paused() || c3.PauseAction != string(ops.ActionSkip) {
			t.Fatalf("expected pause under pressure: paused=%v action=%s", r.engine.Paused(), c3.PauseAction)
		}
		if phaseOf(c3, "asset_sync").Ran {
			t.Fatal("deferrable asset_sync ran under pressure")
		}
		if _, err := os.Stat(filepath.Join(root, ".git", "fak-commit.lock")); !os.IsNotExist(err) {
			t.Fatal("stale lock was NOT swept under pressure: relief was gated")
		}
		if len(r.reaped) != 1 || r.reaped[0] != 4242 {
			t.Fatalf("orphan was NOT reaped under pressure: %v", r.reaped)
		}
		if !phaseOf(c3, "locks_worktrees").Mutated || !phaseOf(c3, "process_runaways").Mutated {
			t.Fatalf("relief phases not recorded as mutating: %+v", c3.Phases)
		}

		// Paused, relief ran with nothing to sweep -> PRESSURE_RELIEF_ONLY.
		r.engine.Process.Collect = func() ([]procguard.Proc, string) { return nil, "" } // census is clean again
		c4 := r.tick(t, true, time.Second)
		if c4.Verdict != ops.LoadPressureReliefOnly {
			t.Fatalf("paused relief tick verdict=%s", c4.Verdict)
		}
		// Paused, nothing due, probe deferred -> SKIPPED.
		c5 := r.tick(t, false, time.Second)
		if c5.Verdict != ops.LoadSkipped {
			t.Fatalf("paused idle tick verdict=%s", c5.Verdict)
		}

		// (2) Restart: a completely fresh engine over the same root honors the pause on its FIRST tick.
		if st := readPause(t, root); !st.Paused {
			t.Fatal("pause record was not durable on disk")
		}
		r2 := newE2ERig(t, root, pressureCfg(), host, r.now)
		if !r2.engine.Paused() {
			t.Fatal("fresh engine did not re-read the durable pause")
		}
		c6 := r2.tick(t, true, time.Second)
		if c6.PauseAction != string(ops.ActionSkip) || phaseOf(c6, "asset_sync").Ran {
			t.Fatalf("first tick after restart did deferred work: action=%s asset_ran=%v", c6.PauseAction, phaseOf(c6, "asset_sync").Ran)
		}
		if c6.Tick != c5.Tick+1 {
			t.Fatalf("tick ordinal not continued across restart: %d after %d", c6.Tick, c5.Tick)
		}

		// (3) Paused window costs materially less self CPU than a running window.
		running := newE2ERig(t, t.TempDir(), pressureCfg(), e2eHost(0.10), e2eBase)
		for i := 0; i < 5; i++ {
			running.tick(t, true, time.Minute)
		}
		paused := newE2ERig(t, t.TempDir(), pressureCfg(), e2eHost(0.97), e2eBase)
		for i := 0; i < 5; i++ {
			paused.tick(t, true, time.Minute)
		}
		sum := func(rg *e2eRig) (s int64) {
			rows, _ := rg.engine.LoadLedger.Read()
			for _, row := range rows {
				s += row.SelfCPUMillis
			}
			return
		}
		rs, ps := sum(running), sum(paused)
		if ps*3 > rs {
			t.Fatalf("paused window self_cpu_ms=%d not materially below running=%d", ps, rs)
		}

		// Capture the honest fixture rows (set FAK_WRITE_OPS_LOAD_FIXTURE=1 to refresh).
		if os.Getenv("FAK_WRITE_OPS_LOAD_FIXTURE") != "" {
			var buf bytes.Buffer
			for _, c := range []ops.LoadCycle{c1, c2, c3, c4, c5} {
				b, _ := json.Marshal(c)
				buf.Write(append(b, '\n'))
			}
			if err := os.WriteFile(filepath.Join("testdata", "ops_load_e2e_ledger.jsonl"), buf.Bytes(), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	})

	t.Run("unmeasured_host_never_pauses", func(t *testing.T) {
		for name, s := range map[string]*e2eSampler{
			"unknown": {s: harnessres.HostSample{}},
			"error":   {err: errors.New("sensor offline")},
		} {
			r := newE2ERig(t, t.TempDir(), pressureCfg(), s, e2eBase)
			for i := 0; i < 3; i++ {
				c := r.tick(t, true, time.Minute)
				if r.engine.Paused() || readPause(t, r.root).Paused {
					t.Fatalf("%s: entered pause on unmeasured host", name)
				}
				if c.Host == nil || (c.Host.State != ops.HostSampleUnknown && c.Host.State != ops.HostSampleError) {
					t.Fatalf("%s: host state not recorded as unmeasured: %+v", name, c.Host)
				}
				// OBSERVED seam (reported on #14482, not asserted): the row verdict is
				// IDLE/PRODUCTIVE and ClassifyLoadCycle reads UNKNOWN_HOST_STATE as a
				// deferral (PRESSURE_RELIEF), so no row is ever verdict=UNKNOWN.
				t.Logf("%s tick %d: row verdict=%s classified=%s host=%s", name, i, c.Verdict, ops.ClassifyLoadCycle(c, ops.PauseState{}), c.Host.State)
				if !phaseOf(c, "asset_sync").Ran {
					t.Fatalf("%s: ticks stopped running work", name)
				}
			}
		}
	})

	t.Run("never_wedged_force_resume_at_max_pause", func(t *testing.T) {
		cfg := pressureCfg()
		cfg.MaxPauseDuration = 10 * time.Minute
		r := newE2ERig(t, t.TempDir(), cfg, e2eHost(0.99), e2eBase) // trigger can never clear
		r.tick(t, true, 0)
		if !r.engine.Paused() {
			t.Fatal("expected pause")
		}
		c := r.tick(t, true, 11*time.Minute)
		if c.ReasonClass != ops.PauseReasonMaxPauseReached {
			t.Fatalf("force-resume not recorded: reason=%s", c.ReasonClass)
		}
		if !phaseOf(c, "asset_sync").Ran {
			t.Fatal("deferred work did not run at the MaxPause hand-off")
		}
		st := readPause(t, r.root)
		if st.Paused || st.ResumeReasonClass != ops.PauseReasonMaxPauseReached {
			t.Fatalf("durable record not force-resumed: %+v", st)
		}
	})

	t.Run("default_posture_inert", func(t *testing.T) {
		// Same workload, heavy-pressure vs errored sampler, reactive pause OFF (default).
		run := func(s *e2eSampler) (ops.LoadCycle, []ops.Event, *e2eRig) {
			cfg := ops.DefaultConfig()
			cfg.ScratchTTL = time.Hour
			r := newE2ERig(t, t.TempDir(), cfg, s, e2eBase)
			r.plant(t)
			c := r.tick(t, true, 0)
			evs, _ := r.engine.Ledger.QueryEvents(24 * 365 * time.Hour)
			return c, evs, r
		}
		ca, ea, ra := run(e2eHost(0.99))
		cb, eb, _ := run(&e2eSampler{err: errors.New("x")})
		if ra.engine.Paused() || readPause(t, ra.root).Paused {
			t.Fatal("default config paused")
		}
		if ca.PauseAction != string(ops.ActionRun) || !phaseOf(ca, "asset_sync").Ran || ca.Verdict != cb.Verdict {
			t.Fatalf("default posture changed behavior: %+v vs %+v", ca, cb)
		}
		if len(ea) != len(eb) || len(ea) == 0 {
			t.Fatalf("event ledger differs: %d vs %d", len(ea), len(eb))
		}

		// The shipped single-tick daemon path: real entrypoint, real host sampler, no daemon left running.
		prev := opsHostReach
		opsHostReach = func() ops.HostReach { return ops.HostReach{} }
		defer func() { opsHostReach = prev }()
		root := t.TempDir()
		var out, errb bytes.Buffer
		if rc := runOpsDaemon(&out, &errb, root, ops.DefaultConfig(), []string{"--once"}); rc != 0 {
			t.Fatalf("daemon --once rc=%d %s", rc, errb.String())
		}
		ll, _ := ops.OpenLoadLedger(ops.DefaultLoadLedgerPath(root))
		rows, err := ll.Read()
		if err != nil || len(rows) != 1 || rows[0].Validate() != nil || readPause(t, root).Paused {
			t.Fatalf("daemon --once load row/pause: err=%v rows=%d", err, len(rows))
		}
	})

	t.Run("committed_fixture_is_valid", func(t *testing.T) {
		f, err := os.ReadFile(filepath.Join("testdata", "ops_load_e2e_ledger.jsonl"))
		if err != nil {
			t.Fatal(err)
		}
		seen := map[ops.LoadVerdict]bool{}
		for _, line := range bytes.Split(bytes.TrimSpace(f), []byte("\n")) {
			var c ops.LoadCycle
			if err := json.Unmarshal(line, &c); err != nil || c.Validate() != nil || c.Schema != ops.LoadSchemaV1 {
				t.Fatalf("bad fixture row: %v %s", err, line)
			}
			seen[c.Verdict] = true
		}
		for _, v := range []ops.LoadVerdict{ops.LoadProductive, ops.LoadIdle, ops.LoadSkipped, ops.LoadPressureReliefOnly} {
			if !seen[v] {
				t.Fatalf("fixture missing verdict %s", v)
			}
		}
	})
}

func seedProjectAssets(t *testing.T, root string) {
	t.Helper()
	m := projectassets.Manifest{
		Schema:      "fak-project-assets/1",
		Skills:      projectassets.SkillPolicy{CanonicalRoot: ".claude/skills", CodexRoot: ".agents/skills", Include: []string{"SKILL.md"}},
		Memories:    projectassets.Policy{CanonicalRoot: ".claude/memory", Include: []string{"*.md"}, Exclude: []projectassets.Exclusion{{Pattern: "MEMORY.md", Reason: "index"}}, StartupCommand: "fak memory recall --intent <task> --json"},
		GoalPrompts: projectassets.Policy{CanonicalRoot: ".claude/goal-prompts", Include: []string{"template.md"}, Exclude: []projectassets.Exclusion{{Pattern: "resolve-[0-9]*.md", Reason: "run fuel"}}},
		Harnesses:   map[string]projectassets.Harness{"codex": {}, "fak-native": {}, "opencode": {}},
	}
	b, _ := json.Marshal(m)
	files := map[string]string{
		projectassets.ManifestPath:            string(b),
		".claude/skills/verify/SKILL.md":      "---\nname: verify\ndescription: Verify.\n---\nbody\n",
		".claude/memory/MEMORY.md":            "index\n",
		".claude/memory/durable.md":           "memory\n",
		".claude/goal-prompts/template.md":    "template\n",
		".claude/goal-prompts/resolve-123.md": "fuel\n",
	}
	for rel, body := range files {
		abs := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
