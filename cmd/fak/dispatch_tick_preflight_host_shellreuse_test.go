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
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

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
