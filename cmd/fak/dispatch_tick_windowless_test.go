//go:build windows

package main

// #3597 acceptance witness for the dispatched-worker launch seam.
//
// Why DETACHED_PROCESS and not CREATE_NO_WINDOW: CREATE_NO_WINDOW suppresses the console's
// WINDOW but still ALLOCATES the console, and on modern Windows every console is hosted by
// its own conhost.exe/OpenConsole.exe process -- so a CREATE_NO_WINDOW worker is invisible
// to an operator yet still pays the full per-console price #2340 measured (87 panes ->
// 2,829 threads / 54k handles / 2 GB). DETACHED_PROCESS declines the console outright.
//
// At the seam, non-Codex workers still use DETACHED_PROCESS. Codex is the deliberate
// exception: a detached Codex has no console for its PowerShell, Node, and stdio MCP
// descendants to inherit, so those console-subsystem children allocate visible desktop
// windows. It receives one CREATE_NO_WINDOW console instead; the second live witness below
// proves the representative descendants inherit it without gaining a window handle.
//
// The remaining acceptance boxes are covered elsewhere and deliberately not duplicated here:
// "the windowless flags are applied only on the headless branch" is guard_child.go's gate,
// pinned by guard_child_windowless_test.go; the flag algebra itself is pinned by
// internal/windowgate/exec_windows_test.go.
//
// What THIS file supplies is the box no unit test can: a LIVE launch through the real
// spawnDispatchIssueWorker seam showing (1) the worker holds NO console at all, (2) a
// before/after process-snapshot diff finds ZERO new conhost/OpenConsole processes in its
// subtree, (3) its stdout AND stderr still land in the transcript, and (4) the monitor
// still classifies it as live.
//
// Windows-only by construction: conhost/OpenConsole is a Windows-only cost, and
// configureDispatchSpawn is a no-op window-mode-wise on every other GOOS (it sets Setsid),
// so the same assertions off Windows would be vacuous rather than reassuring.

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

const (
	dispatchWindowlessStdoutMarker  = "fak-3597-worker-stdout"
	dispatchWindowlessStderrMarker  = "fak-3597-worker-stderr"
	dispatchWindowlessConsoleMarker = "fak-3597-worker-console="
	dispatchWindowlessHelperEnv     = "FAK_DISPATCH_WINDOWLESS_HELPER"
	dispatchWindowlessReleaseEnv    = "FAK_DISPATCH_WINDOWLESS_RELEASE"
	dispatchCodexConsoleRoleEnv     = "FAK_DISPATCH_CODEX_CONSOLE_ROLE"
	dispatchCodexConsoleLabelEnv    = "FAK_DISPATCH_CODEX_CONSOLE_LABEL"
	dispatchCodexConsoleBinDirEnv   = "FAK_DISPATCH_CODEX_CONSOLE_BIN_DIR"
	dispatchCodexConsoleMarker      = "fak-8252-console"
)

// dispatchHelperConsoleProcessCount returns how many processes share the CALLING
// process's console, via GetConsoleProcessList. It is 0 exactly when the caller owns
// no console — the direct measure of the cost #3597 removes.
//
// GetConsoleWindow cannot substitute: CREATE_NO_WINDOW allocates a console whose WINDOW
// is suppressed, so it reads NULL for both flags even though only DETACHED_PROCESS
// actually declined the console (and its conhost.exe host process).
func dispatchHelperConsoleProcessCount() uint32 {
	buf := make([]uint32, 64)
	r, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("GetConsoleProcessList").Call(
		uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return uint32(r)
}

func dispatchHelperConsoleWindow() uintptr {
	r, _, _ := syscall.NewLazyDLL("kernel32.dll").NewProc("GetConsoleWindow").Call()
	return r
}

// dispatchProcRow is one row of the no-spawn process census below.
type dispatchProcRow struct {
	PID  uint32
	PPID uint32
	Name string // lower-cased, ".exe" trimmed -- the vocabulary procguard's name sets use
}

// dispatchProcessCensus snapshots every live process via the Toolhelp32 API. It must not
// itself spawn anything: procguard.CollectProcesses shells out to PowerShell, and a
// PowerShell spawn is exactly the console-allocating event this test is trying to count,
// so using it would poison the before/after diff it is supposed to measure.
func dispatchProcessCensus(t *testing.T) []dispatchProcRow {
	t.Helper()
	h, err := syscall.CreateToolhelp32Snapshot(syscall.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		t.Fatalf("CreateToolhelp32Snapshot: %v", err)
	}
	defer syscall.CloseHandle(h)

	var e syscall.ProcessEntry32
	e.Size = uint32(unsafe.Sizeof(e))
	if err := syscall.Process32First(h, &e); err != nil {
		t.Fatalf("Process32First: %v", err)
	}
	var rows []dispatchProcRow
	for {
		name := strings.ToLower(syscall.UTF16ToString(e.ExeFile[:]))
		rows = append(rows, dispatchProcRow{
			PID:  e.ProcessID,
			PPID: e.ParentProcessID,
			Name: strings.TrimSuffix(name, ".exe"),
		})
		if err := syscall.Process32Next(h, &e); err != nil {
			break // ERROR_NO_MORE_FILES ends the walk
		}
	}
	return rows
}

// dispatchConsoleHostPIDs projects the census onto procguard's own console-host vocabulary
// (conhost / openconsole), so this test and the procguard classifier can never disagree
// about what counts as a console pane.
func dispatchConsoleHostPIDs(rows []dispatchProcRow) map[uint32]dispatchProcRow {
	out := map[uint32]dispatchProcRow{}
	for _, r := range rows {
		if procguard.DefaultConsoleHostChildNames[r.Name] {
			out[r.PID] = r
		}
	}
	return out
}

// dispatchSubtreePIDs returns root plus every descendant pid in rows.
func dispatchSubtreePIDs(rows []dispatchProcRow, root uint32) map[uint32]bool {
	children := map[uint32][]uint32{}
	for _, r := range rows {
		children[r.PPID] = append(children[r.PPID], r.PID)
	}
	seen := map[uint32]bool{root: true}
	queue := []uint32{root}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, kid := range children[cur] {
			if kid == cur || seen[kid] {
				continue // self-parent (pid 0) and cycles cannot wedge the walk
			}
			seen[kid] = true
			queue = append(queue, kid)
		}
	}
	return seen
}

func dispatchWindowlessWaitFor(t *testing.T, what string, timeout time.Duration, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", timeout, what)
}

// TestDispatchWorkerLaunchAllocatesNoConsolePane is the live/integration half of #3597's
// first acceptance box: launch a real worker through the real spawnDispatchIssueWorker seam
// and diff a before/after process census while that worker is provably still alive.
//
// The subtree census alone would have a blind spot: a console child can INHERIT its
// parent's console instead of allocating a fresh one, so on a host where this test runs
// attached to a console the census could read clean even with the detach dropped. The
// console-attachment assertion closes that hole — it asks the worker itself whether it owns
// a console, which is true or false independent of what the runner inherited. The two
// assertions together are the regression detector; the census alone is not.
func TestDispatchWorkerLaunchAllocatesNoConsolePane(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	runsDir := t.TempDir()
	release := filepath.Join(t.TempDir(), "release")

	env := envMap(os.Environ())
	env[dispatchWindowlessHelperEnv] = "1"
	env[dispatchWindowlessReleaseEnv] = release

	before := dispatchConsoleHostPIDs(dispatchProcessCensus(t))

	// probeS=0 keeps the spawn non-blocking and leaves the child running (a positive probe
	// would Wait on it), which is what lets the census below observe a LIVE worker subtree.
	spawned, err := spawnDispatchIssueWorker(
		[]string{exe, "-test.run=TestDispatchWindowlessSpawnHelper"},
		env,
		t.TempDir(),
		runsDir,
		3597,
		"tools",
		"claude",
		"resolve-tools",
		[]string{"tools/**"},
		dispatchtick.Account{},
		nil,
		"",
		"",
		0,
	)
	if err != nil {
		t.Fatalf("spawnDispatchIssueWorker: %v", err)
	}
	t.Cleanup(func() {
		_ = os.WriteFile(release, []byte("release"), 0o644)
		if p, findErr := os.FindProcess(spawned.PID); findErr == nil {
			dispatchWindowlessWaitFor(t, "worker exit", 10*time.Second, func() bool {
				return !dispatchPIDAlive(spawned.PID)
			})
			_ = p.Release()
		}
	})

	// Acceptance: "the worker's stdout/stderr are still captured to its transcript".
	// Waiting on the stdout marker doubles as the barrier that proves the worker really ran.
	transcript := func() string {
		b, _ := os.ReadFile(spawned.Log)
		return string(b)
	}
	dispatchWindowlessWaitFor(t, "worker stdout in the transcript", 60*time.Second, func() bool {
		return strings.Contains(transcript(), dispatchWindowlessStdoutMarker)
	})
	dispatchWindowlessWaitFor(t, "worker stderr in the transcript", 30*time.Second, func() bool {
		return strings.Contains(transcript(), dispatchWindowlessStderrMarker)
	})

	// Acceptance: the dispatched worker owns NO console. This is the assertion that
	// discriminates DETACHED_PROCESS from CREATE_NO_WINDOW — the latter would report a
	// non-zero count here while still looking "windowless" to an operator.
	dispatchWindowlessWaitFor(t, "worker console-attachment report", 30*time.Second, func() bool {
		return strings.Contains(transcript(), dispatchWindowlessConsoleMarker)
	})
	consoleReport := transcript()
	idx := strings.Index(consoleReport, dispatchWindowlessConsoleMarker)
	field := strings.TrimSpace(consoleReport[idx+len(dispatchWindowlessConsoleMarker):])
	if nl := strings.IndexAny(field, "\r\n"); nl >= 0 {
		field = field[:nl]
	}
	if field != "0" {
		t.Fatalf("dispatched worker reported %q processes attached to its console, want 0 (#3597): "+
			"it still owns a console and therefore a conhost/OpenConsole host process", field)
	}

	if !dispatchPIDAlive(spawned.PID) {
		t.Fatalf("worker pid %d exited before the console census; cannot witness its subtree", spawned.PID)
	}

	// Acceptance: "creates ZERO new conhost/OpenConsole processes". Scoped to the worker's
	// own subtree so a SIBLING fleet session spawning a pane on this shared box cannot make
	// the assertion flap -- an unfiltered global diff is not stable on a live multi-worker host.
	rows := dispatchProcessCensus(t)
	subtree := dispatchSubtreePIDs(rows, uint32(spawned.PID))
	var fresh []string
	for pid, row := range dispatchConsoleHostPIDs(rows) {
		if _, existed := before[pid]; existed {
			continue // pre-existing pane, not attributable to this launch
		}
		if subtree[pid] || subtree[row.PPID] {
			fresh = append(fresh, fmt.Sprintf("%s(pid=%d ppid=%d)", row.Name, row.PID, row.PPID))
		}
	}
	// Log the diff even on success: #3597's acceptance box asks for a before/after console
	// count, so the passing run should PRINT the numbers it witnessed rather than only
	// asserting on them silently.
	t.Logf("#3597 console-host census: host-wide before=%d after=%d; fresh under worker pid %d=%v; "+
		"processes attached to the worker's own console=%s",
		len(before), len(dispatchConsoleHostPIDs(rows)), spawned.PID, fresh, field)
	if len(fresh) > 0 {
		t.Fatalf("dispatched worker pid %d allocated %d new console host process(es): %s; want zero (#3597)",
			spawned.PID, len(fresh), strings.Join(fresh, ", "))
	}

	// Acceptance: "`fak info`/monitor still classifies the windowless worker (transcript-driven
	// liveness unaffected)". liveResolutionIssueDetails is the projection the tick picker and
	// the monitor read: a windowless worker must still register as a live scope on its lane.
	details := liveResolutionIssueDetails(runsDir)
	scope, ok := details[3597]
	if !ok {
		t.Fatalf("liveResolutionIssueDetails(%q) = %#v, want a live scope for issue 3597 (#3597)", runsDir, details)
	}
	if scope.PID != spawned.PID {
		t.Fatalf("live scope PID = %d, want the spawned worker pid %d", scope.PID, spawned.PID)
	}
	if scope.Lane != "tools" {
		t.Fatalf("live scope Lane = %q, want %q (read from the transcript's spawn header)", scope.Lane, "tools")
	}
	if len(scope.Tree) == 0 {
		t.Fatalf("live scope Tree is empty, want the lease tree sidecar to be classified")
	}
}

// TestDispatchCodexWorkerDescendantsStayOffDesktop is #8513's captured visual and
// lifecycle witness. The root crosses the real dispatch spawn seam with backend=codex,
// then starts console-subsystem children named for the three process families observed
// on the desktop. Every process reports its actual console attachment and window handle;
// the job-backed hidden tree must stay invisible and leave no process or console host.
func TestDispatchCodexWorkerDescendantsStayOffDesktop(t *testing.T) {
	switch os.Getenv(dispatchCodexConsoleRoleEnv) {
	case "root":
		runDispatchCodexConsoleRoot(t)
		os.Exit(0)
	case "child":
		label := os.Getenv(dispatchCodexConsoleLabelEnv)
		fmt.Fprintf(os.Stdout, "%s label=%s console=%d hwnd=%d\n",
			dispatchCodexConsoleMarker, label, dispatchHelperConsoleProcessCount(), dispatchHelperConsoleWindow())
		_ = os.Stdout.Sync()
		waitForDispatchCodexConsoleRelease()
		os.Exit(0)
	}

	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	runsDir := t.TempDir()
	release := filepath.Join(t.TempDir(), "release")
	binDir := t.TempDir()
	env := envMap(os.Environ())
	env[dispatchCodexConsoleRoleEnv] = "root"
	env[dispatchCodexConsoleBinDirEnv] = binDir
	env[dispatchWindowlessReleaseEnv] = release

	before := dispatchConsoleHostPIDs(dispatchProcessCensus(t))
	spawned, err := spawnDispatchIssueWorker(
		[]string{exe, "-test.run=^TestDispatchCodexWorkerDescendantsStayOffDesktop$"},
		env,
		t.TempDir(),
		runsDir,
		8252,
		"cmd",
		"codex",
		"issue-8252",
		[]string{"cmd/fak/**"},
		dispatchtick.Account{},
		nil,
		"",
		"",
		0,
	)
	if err != nil {
		t.Fatalf("spawn Codex console witness: %v", err)
	}
	cleaned := false
	var witnessedTree map[uint32]bool
	t.Cleanup(func() {
		_ = os.WriteFile(release, []byte("release"), 0o644)
		if !cleaned {
			for pid := range witnessedTree {
				if dispatchPIDAlive(int(pid)) {
					_, _ = procguard.KillPID(int(pid))
				}
			}
		}
		for pid := range witnessedTree {
			pid := int(pid)
			dispatchWindowlessWaitFor(t, fmt.Sprintf("Codex cleanup pid %d exit", pid), 15*time.Second, func() bool {
				return !dispatchPIDAlive(pid)
			})
		}
	})

	transcript := func() string {
		b, _ := os.ReadFile(spawned.Log)
		return string(b)
	}
	wantLabels := []string{"codex-root", "pwsh", "node", "fak-mcp"}
	dispatchWindowlessWaitFor(t, "all descendant console reports", 30*time.Second, func() bool {
		body := transcript()
		for _, label := range wantLabels {
			if !strings.Contains(body, dispatchCodexConsoleMarker+" label="+label+" ") {
				return false
			}
		}
		return true
	})

	for _, line := range strings.Split(transcript(), "\n") {
		if !strings.HasPrefix(line, dispatchCodexConsoleMarker+" ") {
			continue
		}
		fields := map[string]string{}
		for _, field := range strings.Fields(line) {
			if pair := strings.SplitN(field, "=", 2); len(pair) == 2 {
				fields[pair[0]] = pair[1]
			}
		}
		consoleCount, _ := strconv.ParseUint(fields["console"], 10, 32)
		windowHandle, _ := strconv.ParseUint(fields["hwnd"], 10, 64)
		if consoleCount == 0 {
			t.Errorf("%s has no inherited hidden console; its own console children can reach the desktop", fields["label"])
		}
		if windowHandle != 0 {
			t.Errorf("%s console window handle = %d, want 0 (hidden)", fields["label"], windowHandle)
		}
	}

	rows := dispatchProcessCensus(t)
	subtree := dispatchSubtreePIDs(rows, uint32(spawned.PID))
	witnessedTree = subtree
	var fresh []string
	freshPIDs := map[uint32]bool{}
	for pid, row := range dispatchConsoleHostPIDs(rows) {
		if _, existed := before[pid]; existed {
			continue
		}
		if subtree[pid] || subtree[row.PPID] {
			fresh = append(fresh, fmt.Sprintf("%s(pid=%d ppid=%d)", row.Name, row.PID, row.PPID))
			freshPIDs[pid] = true
		}
	}
	t.Logf("#8513 live hidden-tree witness: descendants=%v fresh console hosts under Codex pid %d=%v",
		wantLabels[1:], spawned.PID, fresh)
	if len(fresh) > 1 {
		t.Fatalf("Codex descendants allocated %d console hosts, want at most one inherited hidden host: %v", len(fresh), fresh)
	}

	if err := os.WriteFile(release, []byte("release"), 0o644); err != nil {
		t.Fatalf("release Codex root: %v", err)
	}
	for pid := range subtree {
		pid := int(pid)
		dispatchWindowlessWaitFor(t, fmt.Sprintf("Codex tree pid %d exit", pid), 15*time.Second, func() bool {
			return !dispatchPIDAlive(pid)
		})
	}
	cleaned = true
	afterHosts := dispatchConsoleHostPIDs(dispatchProcessCensus(t))
	var survivingHosts []string
	for pid := range freshPIDs {
		if row, ok := afterHosts[pid]; ok {
			survivingHosts = append(survivingHosts, fmt.Sprintf("%s(pid=%d ppid=%d)", row.Name, row.PID, row.PPID))
		}
	}
	t.Logf("#8513 reaped-tree witness: fresh console hosts after reap=%v", survivingHosts)
	if len(survivingHosts) > 0 {
		t.Fatalf("Codex cleanup left %d fresh console host process(es): %v", len(survivingHosts), survivingHosts)
	}
	for _, label := range wantLabels[1:] {
		path := filepath.Join(binDir, label+".exe")
		if err := os.Remove(path); err != nil {
			t.Fatalf("remove reaped %s witness %q: %v", label, path, err)
		}
	}
}

func runDispatchCodexConsoleRoot(t *testing.T) {
	binDir := os.Getenv(dispatchCodexConsoleBinDirEnv)
	if binDir == "" {
		fmt.Fprintln(os.Stderr, "missing Codex console witness bin dir")
		os.Exit(2)
	}
	fmt.Fprintf(os.Stdout, "%s label=codex-root console=%d hwnd=%d\n",
		dispatchCodexConsoleMarker, dispatchHelperConsoleProcessCount(), dispatchHelperConsoleWindow())
	_ = os.Stdout.Sync()

	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	for _, label := range []string{"pwsh", "node", "fak-mcp"} {
		path := filepath.Join(binDir, label+".exe")
		if err := os.Link(exe, path); err != nil {
			fmt.Fprintf(os.Stderr, "link %s witness: %v\n", label, err)
			os.Exit(2)
		}
		childEnv := envMap(os.Environ())
		childEnv[dispatchCodexConsoleRoleEnv] = "child"
		childEnv[dispatchCodexConsoleLabelEnv] = label
		cmd := exec.Command(path, "-test.run=^TestDispatchCodexWorkerDescendantsStayOffDesktop$")
		cmd.Env = envSliceFromMap(childEnv)
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		if err := cmd.Start(); err != nil {
			fmt.Fprintf(os.Stderr, "start %s witness: %v\n", label, err)
			os.Exit(2)
		}
		_ = cmd.Process.Release()
	}
	waitForDispatchCodexConsoleRelease()
}

func waitForDispatchCodexConsoleRelease() {
	release := os.Getenv(dispatchWindowlessReleaseEnv)
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		if release != "" {
			if _, err := os.Stat(release); err == nil {
				return
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
}

// TestDispatchWindowlessSpawnHelper is the child process TestDispatchWorkerLaunchAllocatesNoConsolePane
// launches through the real seam. It writes one marker to each stream, then blocks until the
// parent drops the release file so the parent can census a LIVE worker subtree. It exits on its
// own after the ceiling below even if the parent dies, so a crashed test never strands it.
func TestDispatchWindowlessSpawnHelper(t *testing.T) {
	if os.Getenv(dispatchWindowlessHelperEnv) != "1" {
		t.Skip("child-process helper; runs only when the parent test spawns it")
	}
	fmt.Fprintln(os.Stdout, dispatchWindowlessStdoutMarker)
	fmt.Fprintln(os.Stderr, dispatchWindowlessStderrMarker)
	// Report our OWN console attachment through the transcript. This is what makes the
	// parent's assertion immune to the console-inheritance blind spot: a process either
	// has a console (and thus a conhost host process) or it does not, regardless of what
	// the test runner inherited.
	fmt.Fprintf(os.Stdout, "%s%d\n", dispatchWindowlessConsoleMarker, dispatchHelperConsoleProcessCount())
	_ = os.Stdout.Sync()
	_ = os.Stderr.Sync()

	release := os.Getenv(dispatchWindowlessReleaseEnv)
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		if release != "" {
			if _, err := os.Stat(release); err == nil {
				break
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	os.Exit(0)
}

// TestDispatchWorkerOutlivesLauncherScope is the #9064 regression witness. The
// dispatch tick is a short-lived launcher, so the worker must remain alive after
// spawnDispatchIssueWorker returns; a kill-on-close Job Object owned by the tick
// violated that contract and reaped every guarded worker before its first turn.
func TestDispatchWorkerOutlivesLauncherScope(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	runsDir := t.TempDir()
	release := filepath.Join(t.TempDir(), "release")
	env := envMap(os.Environ())
	env[dispatchWindowlessHelperEnv] = "1"
	env[dispatchWindowlessReleaseEnv] = release

	spawned, err := spawnDispatchIssueWorker(
		[]string{exe, "-test.run=TestDispatchWindowlessSpawnHelper"}, env,
		t.TempDir(), runsDir, 9064, "dispatch", "codex", "resolve-dispatch-9064",
		[]string{"cmd/fak/dispatch_tick_worker.go"}, dispatchtick.Account{}, nil, "", "", 0,
	)
	if err != nil {
		t.Fatalf("spawnDispatchIssueWorker: %v", err)
	}
	t.Cleanup(func() {
		_ = os.WriteFile(release, []byte("release"), 0o644)
		dispatchWindowlessWaitFor(t, "worker exit", 10*time.Second, func() bool {
			return !dispatchPIDAlive(spawned.PID)
		})
	})

	dispatchWindowlessWaitFor(t, "worker marker after launcher return", 30*time.Second, func() bool {
		b, _ := os.ReadFile(spawned.Log)
		return strings.Contains(string(b), dispatchWindowlessStdoutMarker)
	})
	if !dispatchPIDAlive(spawned.PID) {
		t.Fatalf("worker pid %d exited when launcher scope returned", spawned.PID)
	}
}
