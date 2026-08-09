package main

// #3405 host-probe shell reuse spine — witnessed through the REAL probe seam.
//
// These tests never touch PowerShell. They swap the two OS boundaries of the
// seam (dispatchHostProbeSpawner, dispatchHostProbeOneShot) and then call the
// production probe functions themselves — dispatchScanProcessesWindows,
// dispatchScanWorkerProcessRowsWindows, dispatchScanCodexProcessRowsWindows —
// so what is under test is the wiring a tick actually executes, not the rack in
// isolation. The measurement is process creations: three probes cost three
// spawns without the spine and one with it.

import (
	"errors"
	"io"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// dispatchHostProbeShellReuseAtInit snapshots the seam's value at package
// initialization, BEFORE any test has run. Package-level variables initialize in
// dependency order, so this is the genuine declared default of
// dispatchHostProbeShellReuse and not whatever a neighbouring test last wrote --
// which is the only way to pin a package-var default that every other test in
// this file deliberately flips.
var dispatchHostProbeShellReuseAtInit = dispatchHostProbeShellReuse

const (
	probeProcJSON   = `[{"pid":11,"name":"claude","threads":7,"handles":300,"ws_mb":512}]`
	probeWorkerJSON = `[{"pid":21,"ppid":2,"name":"claude.exe","cmdline":"claude -p resolve GitHub issue #3405"}]`
	probeCodexJSON  = `[{"pid":31,"ppid":3,"name":"codex.exe","cmdline":"codex exec"}]`
)

// fakeProbeShell is one warm shell: it answers a task from the script text, and
// counts every task it served so the test can prove reuse rather than infer it.
type fakeProbeShell struct {
	mu      sync.Mutex
	tasks   []string
	healthy bool
	closes  int
	failNth int // 1-based task index that breaks the shell (0 = never)
}

func (s *fakeProbeShell) Healthy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.healthy
}

func (s *fakeProbeShell) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closes++
	s.healthy = false
	return nil
}

func (s *fakeProbeShell) RunTask(task string) ([]byte, error) {
	s.mu.Lock()
	s.tasks = append(s.tasks, task)
	n := len(s.tasks)
	if s.failNth == n {
		s.healthy = false
		s.mu.Unlock()
		return nil, errors.New("fake probe shell broke mid-task")
	}
	s.mu.Unlock()
	switch {
	case strings.Contains(task, "Get-Process"):
		return []byte(probeProcJSON), nil
	case strings.Contains(task, "claude.exe"):
		return []byte(probeWorkerJSON), nil
	case strings.Contains(task, "codex.exe"):
		return []byte(probeCodexJSON), nil
	}
	return nil, errors.New("unexpected probe script: " + task)
}

func (s *fakeProbeShell) taskCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.tasks)
}

func (s *fakeProbeShell) closeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closes
}

// probeSeamHarness swaps both OS boundaries of the seam and restores them (and
// the process-wide rack) when the test ends, so nothing leaks between tests and
// no console is ever created.
type probeSeamHarness struct {
	mu       sync.Mutex
	spawns   int
	oneShots []string
	shells   []*fakeProbeShell
	failNth  int
	spawnErr error
}

func newProbeSeamHarness(t *testing.T) *probeSeamHarness {
	t.Helper()
	h := &probeSeamHarness{}
	prevSpawner, prevOneShot := dispatchHostProbeSpawner, dispatchHostProbeOneShot
	prevReuse := dispatchHostProbeShellEnabled()
	dispatchCloseHostProbeShells()
	// Start every harness test from the DISARMED posture rather than from whatever the
	// previous test left behind. The seam is process-wide, so once the tick's own arming
	// witness runs on Windows it stays on, and a test that deliberately does not arm --
	// the BEFORE number the spine is measured against -- would silently inherit a warm
	// shell it never asked for. Saving prevReuse only restores the neighbour's posture on
	// the way out; it is this line that makes the way IN hermetic.
	setDispatchHostProbeShellReuse(false)
	dispatchHostProbeSpawner = h.spawn
	dispatchHostProbeOneShot = h.oneShot
	t.Cleanup(func() {
		dispatchCloseHostProbeShells()
		dispatchHostProbeSpawner = prevSpawner
		dispatchHostProbeOneShot = prevOneShot
		setDispatchHostProbeShellReuse(prevReuse)
	})
	return h
}

// armReuseSpine switches the spine on for one test; the harness cleanup puts the
// seam back the way it found it.
func (h *probeSeamHarness) armReuseSpine() { setDispatchHostProbeShellReuse(true) }

func (h *probeSeamHarness) spawn(string) (dispatchtick.WarmShell, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.spawnErr != nil {
		return nil, h.spawnErr
	}
	h.spawns++
	s := &fakeProbeShell{healthy: true, failNth: h.failNth}
	h.shells = append(h.shells, s)
	return s, nil
}

func (h *probeSeamHarness) oneShot(script string, _ time.Duration, _ bool) ([]byte, error) {
	h.mu.Lock()
	h.oneShots = append(h.oneShots, script)
	h.mu.Unlock()
	switch {
	case strings.Contains(script, "Get-Process"):
		return []byte(probeProcJSON), nil
	case strings.Contains(script, "claude.exe"):
		return []byte(probeWorkerJSON), nil
	case strings.Contains(script, "codex.exe"):
		return []byte(probeCodexJSON), nil
	}
	return nil, errors.New("unexpected one-shot probe script: " + script)
}

func (h *probeSeamHarness) counts() (spawns, oneShots int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.spawns, len(h.oneShots)
}

// shellAt fails the test rather than panicking when the seam spawned no shell,
// so a regression that bypasses the rack reports a readable assertion.
func (h *probeSeamHarness) shellAt(t *testing.T, i int) *fakeProbeShell {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	if i >= len(h.shells) {
		t.Fatalf("no warm shell #%d was spawned (the probes never reached the rack)", i)
	}
	return h.shells[i]
}

// runThreeHostProbes drives three DIFFERENT production probe functions and
// asserts each decoded the payload the shell returned. Decoding is the proof
// the reuse path carries real probe output, not an empty success.
func runThreeHostProbes(t *testing.T) {
	t.Helper()
	procs, err := dispatchScanProcessesWindows()
	if err != nil {
		t.Fatalf("dispatchScanProcessesWindows: %v", err)
	}
	if len(procs) != 1 || procs[0].PID != 11 || procs[0].Name != "claude" {
		t.Fatalf("process scan decoded %+v, want one row pid=11 name=claude", procs)
	}
	workers, err := dispatchScanWorkerProcessRowsWindows()
	if err != nil {
		t.Fatalf("dispatchScanWorkerProcessRowsWindows: %v", err)
	}
	if len(workers) != 1 || workers[0].PID != 21 {
		t.Fatalf("worker rows decoded %+v, want one row pid=21", workers)
	}
	codex, err := dispatchScanCodexProcessRowsWindows()
	if err != nil {
		t.Fatalf("dispatchScanCodexProcessRowsWindows: %v", err)
	}
	if len(codex) != 1 || codex[0].PID != 31 {
		t.Fatalf("codex rows decoded %+v, want one row pid=31", codex)
	}
}

// TestHostProbesSpawnOnePerProbeWithoutTheReuseSpine pins the BEFORE number the
// spine is measured against: with the knob off, each of the three probes is its
// own process (and console) creation, and the rack is never touched.
func TestHostProbesSpawnOnePerProbeWithoutTheReuseSpine(t *testing.T) {
	h := newProbeSeamHarness(t)
	// deliberately NOT armed: this is the default posture.

	runThreeHostProbes(t)

	spawns, oneShots := h.counts()
	if oneShots != 3 {
		t.Fatalf("one-shot spawns = %d, want 3 (one process per probe)", oneShots)
	}
	if spawns != 0 {
		t.Fatalf("warm shell spawns = %d, want 0 (spine is off)", spawns)
	}
}

// TestHostProbesReuseOneWarmShellAcrossTasks is the #3405 spine witness: three
// tasks, driven through three real probe functions, ride ONE shell. Process
// creations for the same work drop 3 -> 1.
func TestHostProbesReuseOneWarmShellAcrossTasks(t *testing.T) {
	h := newProbeSeamHarness(t)
	h.armReuseSpine()

	runThreeHostProbes(t)

	spawns, oneShots := h.counts()
	if spawns != 1 {
		t.Fatalf("warm shell spawns = %d, want 1 (one shell serves every probe)", spawns)
	}
	if oneShots != 0 {
		t.Fatalf("one-shot spawns = %d, want 0 (every probe rode the warm shell)", oneShots)
	}
	if got := h.shellAt(t, 0).taskCount(); got != 3 {
		t.Fatalf("warm shell ran %d tasks, want 3 (reuse across >= 2 tasks)", got)
	}

	st := dispatchHostProbeRackStats()
	if st.ColdSpawns != 1 || st.WarmReuses != 2 || st.TasksRun != 3 {
		t.Fatalf("rack stats = %+v, want ColdSpawns=1 WarmReuses=2 TasksRun=3", st)
	}
	// The measured churn reduction: 3 probes used to cost 3 spawns, now 1.
	if got := st.SpawnsAvoided(); got != 2 {
		t.Fatalf("spawns avoided = %d, want 2 (3 probes - 1 shell)", got)
	}
}

// TestHostProbeShellIsClosedOnTeardown witnesses the teardown half: the warm
// shell the spine opened is closed exactly once when the spine is torn down,
// and the rack retains nothing afterwards. (This witnesses the shell's Close
// contract, not a live conhost/OpenConsole count.)
func TestHostProbeShellIsClosedOnTeardown(t *testing.T) {
	h := newProbeSeamHarness(t)
	h.armReuseSpine()

	runThreeHostProbes(t)
	shell := h.shellAt(t, 0)
	if shell.closeCount() != 0 {
		t.Fatalf("shell closed %d times while still in service, want 0", shell.closeCount())
	}

	dispatchCloseHostProbeShells()
	if got := shell.closeCount(); got != 1 {
		t.Fatalf("shell Close called %d times on teardown, want exactly 1", got)
	}
	if st := dispatchHostProbeRackStats(); st != (dispatchtick.ShellRackStats{}) {
		t.Fatalf("rack survived teardown with stats %+v, want a dropped rack", st)
	}
}

// TestHostProbeBrokenShellIsRetiredAndReplaced proves the reuse is
// health-checked at the real seam: a shell that dies mid-probe is retired, the
// probe still returns its answer (via the one-shot fallback), and the NEXT
// probe gets a fresh shell instead of the corpse.
func TestHostProbeBrokenShellIsRetiredAndReplaced(t *testing.T) {
	h := newProbeSeamHarness(t)
	h.armReuseSpine()
	h.failNth = 2 // every shell breaks on its second task

	runThreeHostProbes(t)

	spawns, oneShots := h.counts()
	if spawns != 2 {
		t.Fatalf("warm shell spawns = %d, want 2 (broken shell retired and replaced)", spawns)
	}
	if oneShots != 1 {
		t.Fatalf("one-shot fallbacks = %d, want 1 (the failed probe still answered)", oneShots)
	}
	if got := h.shellAt(t, 0).closeCount(); got != 1 {
		t.Fatalf("broken shell closed %d times, want 1", got)
	}
	if st := dispatchHostProbeRackStats(); st.UnhealthyRetired != 1 {
		t.Fatalf("rack stats = %+v, want UnhealthyRetired=1", st)
	}
}

// TestHostProbeFallsBackWhenTheShellCannotSpawn proves the spine is fail-open
// at the real seam: if no shell can be created the probes still answer, on the
// historical one-shot path.
func TestHostProbeFallsBackWhenTheShellCannotSpawn(t *testing.T) {
	h := newProbeSeamHarness(t)
	h.armReuseSpine()
	h.spawnErr = errors.New("no shell available")

	runThreeHostProbes(t)

	spawns, oneShots := h.counts()
	if spawns != 0 || oneShots != 3 {
		t.Fatalf("spawns=%d one-shots=%d, want 0/3 (fail-open to the one-shot path)", spawns, oneShots)
	}
}

// TestHostProbeShellReuseIsOffByDefault pins the default posture: the spine is a
// declared config seam (CONFIG_NOT_ENV -- no env read), it starts disarmed, and
// arming it is what flips dispatchRunHostProbe off the one-shot path.
func TestHostProbeShellReuseIsOffByDefault(t *testing.T) {
	prev := dispatchHostProbeShellEnabled()
	t.Cleanup(func() { setDispatchHostProbeShellReuse(prev) })

	// The PACKAGE default, not just the setter round-trip: every caller that shares
	// dispatchRunHostProbe without declaring anything -- another verb, a helper, a
	// test -- must keep the historical one-shot spawn. Only a caller that arms the
	// seam (the dispatch tick, below) moves off it.
	if dispatchHostProbeShellReuseAtInit {
		t.Fatalf("the process-wide seam initialized armed: an undeclared caller would " +
			"silently route its probes through a warm console it never asked for")
	}

	setDispatchHostProbeShellReuse(false)
	if dispatchHostProbeShellEnabled() {
		t.Fatalf("a disarmed seam must leave the reuse spine off")
	}
	setDispatchHostProbeShellReuse(true)
	if !dispatchHostProbeShellEnabled() {
		t.Fatalf("an armed seam must turn the reuse spine on")
	}
	setDispatchHostProbeShellReuse(false)
	if dispatchHostProbeShellEnabled() {
		t.Fatalf("disarming must put the reuse spine back off")
	}
}

// TestHostProbeSeamReadsNoEnvironmentKnob is the CONFIG_NOT_ENV guard-rail for
// this seam: the reuse spine must stay a declared setting, so re-introducing an
// os.Getenv/os.LookupEnv arming read here reds this test instead of reding
// internal/envconfiglint's tree ratchet one commit later.
func TestHostProbeSeamReadsNoEnvironmentKnob(t *testing.T) {
	src, err := os.ReadFile("dispatch_tick_preflight_host.go")
	if err != nil {
		t.Fatalf("read the seam file: %v", err)
	}
	for _, banned := range []string{"os.Getenv(", "os.LookupEnv("} {
		if strings.Contains(string(src), banned) {
			t.Fatalf("the host-probe seam calls %s: behavioral settings belong on the config "+
				"surface, not the environment (internal/envconfiglint CONFIG_NOT_ENV)", banned)
		}
	}
}

// boolRef is the declaration helper: a *bool field distinguishes "the caller
// declared this" from "the caller declared nothing", and a test needs both.
func boolRef(v bool) *bool { return &v }

// TestHostProbeShellReuseDefaultsPerGOOS pins WHERE the spine is on. Windows is
// the only GOOS whose host probes go through PowerShell, so it is the only one
// that pays a ConPTY/conhost per probe and the only one with churn to collapse;
// arming it elsewhere would add a moving part to a `ps` one-shot for no dividend.
func TestHostProbeShellReuseDefaultsPerGOOS(t *testing.T) {
	for _, tc := range []struct {
		goos string
		want bool
	}{
		{"windows", true},
		{"linux", false},
		{"darwin", false},
		{"freebsd", false},
	} {
		if got := dispatchHostProbeShellReuseForOS(tc.goos); got != tc.want {
			t.Errorf("default on %s = %v, want %v", tc.goos, got, tc.want)
		}
	}
	// The host default this box actually gets is that same rule applied to runtime.GOOS,
	// so the flag's printed default and the tick's resolved posture cannot drift apart.
	if got, want := dispatchHostProbeShellReuseDefault(), runtime.GOOS == "windows"; got != want {
		t.Errorf("host default = %v on GOOS=%s, want %v", got, runtime.GOOS, want)
	}
}

// TestHostProbeShellReuseResolvesTheDeclaredSetting pins the tri-state: an
// undeclared tick (the wave/sweep/garden callers, which build the options struct
// directly and fill no flag) takes the host default, and an explicit declaration
// wins in BOTH directions -- which is what makes --host-probe-shell-reuse=false a
// real off switch and not just an absent opt-in.
func TestHostProbeShellReuseResolvesTheDeclaredSetting(t *testing.T) {
	hostDefault := dispatchHostProbeShellReuseDefault()
	for _, tc := range []struct {
		name     string
		declared *bool
		want     bool
	}{
		{"undeclared takes the host default", nil, hostDefault},
		{"declared on", boolRef(true), true},
		{"declared off", boolRef(false), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := dispatchResolveHostProbeShellReuse(tc.declared); got != tc.want {
				t.Fatalf("resolved = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestDispatchTickArmsHostProbeShellReuse is the ANTI-DARKNESS witness for #3405:
// the spine was landed, tested, and reachable by nobody, so the fleet got zero
// spawn reduction from green code. Here the tick's own arming call is driven and
// the seam is read back -- on Windows a bare tick arms the spine, everywhere else
// it stays on the one-shot path, and an explicit declaration overrides both.
func TestDispatchTickArmsHostProbeShellReuse(t *testing.T) {
	for _, tc := range []struct {
		name     string
		declared *bool
		want     bool
	}{
		{"a bare tick follows the host", nil, runtime.GOOS == "windows"},
		{"an operator can arm it anywhere", boolRef(true), true},
		{"an operator can disarm it on any host", boolRef(false), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prev := dispatchHostProbeShellEnabled()
			t.Cleanup(func() { setDispatchHostProbeShellReuse(prev) })
			// Start from the opposite posture so a no-op arm cannot pass by accident.
			setDispatchHostProbeShellReuse(!tc.want)

			closeShells := dispatchArmHostProbeShellReuse(tc.declared)
			if closeShells == nil {
				t.Fatal("arming handed back no teardown: the tick has nothing to defer")
			}
			if got := dispatchHostProbeShellEnabled(); got != tc.want {
				t.Fatalf("after arming, the spine is %v, want %v", got, tc.want)
			}
			closeShells()
		})
	}
}

// TestDispatchTickArmingTearsDownTheWarmShell witnesses the other half of the arm:
// the teardown the tick defers really closes the console the spine opened, so a
// warm shell cannot outlive the tick that opened it (the leak the whole rack would
// otherwise trade the spawn saving for).
func TestDispatchTickArmingTearsDownTheWarmShell(t *testing.T) {
	h := newProbeSeamHarness(t)

	closeShells := dispatchArmHostProbeShellReuse(boolRef(true))
	runThreeHostProbes(t)

	shell := h.shellAt(t, 0)
	if got := shell.taskCount(); got != 3 {
		t.Fatalf("warm shell ran %d tasks, want 3 (the arm did not route the probes)", got)
	}
	if got := shell.closeCount(); got != 0 {
		t.Fatalf("shell closed %d times while the tick was still running, want 0", got)
	}

	closeShells()

	if got := shell.closeCount(); got != 1 {
		t.Fatalf("the deferred teardown closed the shell %d times, want exactly 1", got)
	}
	if st := dispatchHostProbeRackStats(); st != (dispatchtick.ShellRackStats{}) {
		t.Fatalf("the rack survived the tick with stats %+v, want it dropped", st)
	}
}

// TestHostProbeShellReuseIsOnTheTickConfigSurface pins the DECLARATION channel:
// the setting must be a named flag `--help` prints, not an environment read
// (internal/envconfiglint CONFIG_NOT_ENV), and a bare tick must resolve to the
// host default rather than to Go's zero value.
func TestHostProbeShellReuseIsOnTheTickConfigSurface(t *testing.T) {
	for _, tc := range []struct {
		name string
		argv []string
		want bool
	}{
		{"a bare tick", nil, dispatchHostProbeShellReuseDefault()},
		{"declared off", []string{"--host-probe-shell-reuse=false"}, false},
		{"declared on", []string{"--host-probe-shell-reuse=true"}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			argv := append([]string{"--workspace", t.TempDir()}, tc.argv...)
			opts, _, code := parseDispatchTickFlags(io.Discard, argv)
			if code != 0 {
				t.Fatalf("parse failed with code %d", code)
			}
			if opts.HostProbeShellReuse == nil {
				t.Fatal("the CLI left the setting undeclared: the flag never reached the options")
			}
			if got := dispatchResolveHostProbeShellReuse(opts.HostProbeShellReuse); got != tc.want {
				t.Fatalf("the tick resolved %v, want %v", got, tc.want)
			}
		})
	}
}

// TestDispatchTickWiresTheHostProbeSpine is the regression pin against the exact
// staleness this ticket exists to fix: the spine going DARK again. A green rack
// with no caller looks shipped in the commit log and changes nothing on the fleet,
// and no behavioral test can catch that -- the seam simply never runs. So assert
// the wiring itself: the tick both ARMS the spine and DEFERS the teardown it was
// handed. Dropping either line reds here instead of quietly costing the fleet a
// console per probe again.
func TestDispatchTickWiresTheHostProbeSpine(t *testing.T) {
	src, err := os.ReadFile("dispatch_tick.go")
	if err != nil {
		t.Fatalf("read the tick: %v", err)
	}
	for _, want := range []string{
		"dispatchArmHostProbeShellReuse(opts.HostProbeShellReuse)",
		"defer closeHostProbeShells()",
	} {
		if !strings.Contains(string(src), want) {
			t.Fatalf("the dispatch tick no longer contains %q: the #3405 reuse spine is "+
				"landed but unreachable, so every Windows probe is back to its own "+
				"process and its own ConPTY/conhost", want)
		}
	}
}
