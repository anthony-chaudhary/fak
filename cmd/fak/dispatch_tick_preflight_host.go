package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/committedbuildwitness"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/trunkbuildprobe"
)

// ---- #3405 host-probe shell reuse spine ---------------------------------- //
//
// Every Windows host probe in this file (process scan, free RAM, worker rows,
// codex rows) used to be its OWN `powershell -Command <script>` spawn. On
// Windows each of those creates a ConPTY with its own conhost/OpenConsole
// process plus pipes, so console cost scales linearly with how many probes the
// fleet runs -- the churn #3153 diagnoses, for which "no registry knob is a
// substitute". The probes are short, independent, and identical in shape, which
// is exactly what a warm shell can serve: routing them through
// dispatchtick.ShellRack collapses a tick's N probe spawns into ONE shell that
// runs N tasks.
//
// Guardrails, in order of what can go wrong:
//   - Fail-open. A spawn error, a protocol error, or a task timeout falls back
//     to the one-shot path, so the reuse spine can never wedge a tick; the rack
//     retires the broken shell rather than handing it to the next probe.
//   - Bounded. Capacity 1 and a max-idle retirement, so at most one extra
//     console exists, and the tick closes it on the way out (the deferred
//     teardown evaluateDispatchTick arms it with) so no console outlives the
//     tick that opened it.
//   - Reversible without a build. The spine is ON by default where the ConPTY
//     cost exists -- Windows -- and off on every other GOOS, where the probes
//     never reach PowerShell at all; `fak dispatch tick
//     --host-probe-shell-reuse=false` puts every probe back on the historical
//     one-shot path byte for byte.

const (
	// dispatchHostProbeKey is the single reuse domain: all host probes are the
	// same shape (a short PowerShell script against this host), so one key.
	dispatchHostProbeKey = "host-probe/powershell"
	// dispatchHostProbeSentinel terminates one task's output on the wire.
	dispatchHostProbeSentinel = "@@fak-host-probe-done@@"
	// dispatchHostProbeMaxIdle retires a warm probe shell that no tick has used
	// recently, so an idle fleet holds no console open.
	dispatchHostProbeMaxIdle = 5 * time.Minute
	// dispatchHostProbeTaskTimeout bounds one task on the warm shell. It is the
	// longest one-shot probe budget in this file; a task that overruns kills the
	// shell and the caller falls back to a one-shot spawn.
	dispatchHostProbeTaskTimeout = 60 * time.Second
)

// dispatchHostProbeBootstrap is the loop the warm shell runs: read one
// base64-encoded script per line from stdin, execute it, then emit the
// sentinel. Owning the read loop (instead of feeding `powershell -Command -`)
// keeps banners and prompts off the wire, and base64 removes every quoting and
// newline hazard from the script we hand it. EOF on stdin ends the loop, so
// closing stdin is a clean, non-violent teardown of the console.
const dispatchHostProbeBootstrap = "$ErrorActionPreference='Continue'; " +
	"while ($true) { " +
	"$line = [Console]::In.ReadLine(); " +
	"if ($null -eq $line) { break }; " +
	"if ($line.Length -eq 0) { continue }; " +
	"try { $src = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($line)); & ([ScriptBlock]::Create($src)) } catch { }; " +
	"[Console]::Out.WriteLine('" + dispatchHostProbeSentinel + "'); " +
	"[Console]::Out.Flush() }"

var (
	dispatchHostProbeMu   sync.Mutex
	dispatchHostProbeRack *dispatchtick.ShellRack
	// dispatchHostProbeSpawner is the OS boundary the reuse spine sits on. A
	// test swaps it for an in-process fake so the probe seam can be driven end
	// to end without creating a single console.
	dispatchHostProbeSpawner dispatchtick.ShellSpawnFunc = spawnDispatchHostProbeShell
	// dispatchHostProbeOneShot is the historical path: one process per probe.
	// It is a var for the same reason -- so the before/after process-creation
	// count is measurable in-test rather than only in a captured run.
	dispatchHostProbeOneShot = runDispatchHostProbeOneShot
)

// dispatchHostProbeShellReuse arms the reuse spine. It is a package seam a
// caller WRITES, deliberately not an os.LookupEnv: internal/envconfiglint's
// CONFIG_NOT_ENV rule reserves the environment for secrets and puts behavioral
// settings on a config surface `--help` can name, and the dispatch tick already
// publishes its other settings to package seams the same way. The process-wide
// default stays false, so a caller that never declares anything (another verb
// sharing dispatchRunHostProbe, a helper, a test) keeps the historical one-shot
// spawn; the dispatch tick publishes its own resolved setting here on the way
// in, via dispatchArmHostProbeShellReuse.
var dispatchHostProbeShellReuse bool

// setDispatchHostProbeShellReuse arms or disarms the spine under the same mutex
// that guards the rack, so a caller flipping it cannot race a probe in flight.
func setDispatchHostProbeShellReuse(on bool) {
	dispatchHostProbeMu.Lock()
	dispatchHostProbeShellReuse = on
	dispatchHostProbeMu.Unlock()
}

// dispatchHostProbeShellEnabled reports whether the reuse spine is switched on.
func dispatchHostProbeShellEnabled() bool {
	dispatchHostProbeMu.Lock()
	defer dispatchHostProbeMu.Unlock()
	return dispatchHostProbeShellReuse
}

// dispatchHostProbeShellReuseForOS is the spine's default posture for a GOOS,
// split out from runtime.GOOS so the rule itself is testable on any host.
// Windows is the only GOOS whose probes go through PowerShell at all, and the
// only one that pays the ConPTY/conhost cost per spawn (#3153), so it is the
// only one where reuse has anything to collapse. Everywhere else the probes are
// `ps` one-shots and the rack would add a moving part for zero dividend.
func dispatchHostProbeShellReuseForOS(goos string) bool { return goos == "windows" }

// dispatchHostProbeShellReuseDefault is that posture for THIS host. It is what
// the tick's --host-probe-shell-reuse flag defaults to, so `--help` names the
// setting and prints the default the operator will actually get.
func dispatchHostProbeShellReuseDefault() bool {
	return dispatchHostProbeShellReuseForOS(runtime.GOOS)
}

// dispatchResolveHostProbeShellReuse folds a tick's DECLARED setting against the
// host default. nil means the caller declared nothing -- a programmatic tick
// (wave / sweep / garden) that never fills the field -- and takes the host
// default, so the fleet paths that never parse a flag still get the dividend.
// A non-nil declaration always wins, which is what makes
// `--host-probe-shell-reuse=false` a real off switch on Windows.
func dispatchResolveHostProbeShellReuse(declared *bool) bool {
	if declared != nil {
		return *declared
	}
	return dispatchHostProbeShellReuseDefault()
}

// dispatchArmHostProbeShellReuse publishes one tick's resolved setting into the
// seam and hands back the teardown that tick must defer. Arming and teardown are
// one call on purpose: the spine's whole failure mode is a warm console that
// outlives the tick that opened it, and a caller that cannot take the arm
// without also being handed the closer cannot leak the rack by forgetting half.
func dispatchArmHostProbeShellReuse(declared *bool) func() {
	setDispatchHostProbeShellReuse(dispatchResolveHostProbeShellReuse(declared))
	return dispatchCloseHostProbeShells
}

// dispatchHostProbeRackHandle returns the process-wide probe rack, building it
// on first use. A rack that cannot be built (never, at these constants) yields
// nil so the caller takes the one-shot path.
func dispatchHostProbeRackHandle() *dispatchtick.ShellRack {
	dispatchHostProbeMu.Lock()
	defer dispatchHostProbeMu.Unlock()
	if dispatchHostProbeRack != nil {
		return dispatchHostProbeRack
	}
	rack, err := dispatchtick.NewShellRack(1, dispatchHostProbeMaxIdle, func(key string) (dispatchtick.WarmShell, error) {
		return dispatchHostProbeSpawner(key)
	})
	if err != nil {
		return nil
	}
	dispatchHostProbeRack = rack
	return rack
}

// dispatchHostProbeRackStats exposes the rack's traffic counters (nil rack =
// zero), so a tick payload or a test can read the reuse dividend directly.
func dispatchHostProbeRackStats() dispatchtick.ShellRackStats {
	dispatchHostProbeMu.Lock()
	rack := dispatchHostProbeRack
	dispatchHostProbeMu.Unlock()
	if rack == nil {
		return dispatchtick.ShellRackStats{}
	}
	return rack.Stats()
}

// dispatchCloseHostProbeShells tears the reuse spine down: every warm probe
// shell is closed and the rack is dropped, so the next probe starts clean. A
// long-lived process (the tick, the serve loop) calls this on shutdown so the
// warm console does not outlive the fleet.
func dispatchCloseHostProbeShells() {
	dispatchHostProbeMu.Lock()
	rack := dispatchHostProbeRack
	dispatchHostProbeRack = nil
	dispatchHostProbeMu.Unlock()
	if rack != nil {
		rack.Close()
	}
}

// dispatchRunHostProbe runs one PowerShell probe script and returns its stdout.
// With the reuse spine on, the script rides the warm racked shell -- the second
// and later probes of a tick cost zero process creations. Anything that goes
// wrong falls back to the historical one-shot spawn, so a probe result is never
// lost to the pooling. combineStderr applies to the one-shot path only: the
// warm shell reads stdout, and its bootstrap already swallows script errors.
var (
	dispatchRunHostProbeFunc        = dispatchRunHostProbe
	dispatchRunHostProbeOneShotFunc = runDispatchHostProbeOneShot
)

func dispatchRunHostProbe(script string, timeout time.Duration, combineStderr bool) ([]byte, error) {
	if dispatchHostProbeShellEnabled() {
		if rack := dispatchHostProbeRackHandle(); rack != nil {
			if out, err := rack.RunTask(dispatchHostProbeKey, script); err == nil {
				return out, nil
			}
		}
	}
	return dispatchHostProbeOneShot(script, timeout, combineStderr)
}

func runDispatchHostProbeOneShot(script string, timeout time.Duration, combineStderr bool) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", script)
	cmd.WaitDelay = 10 * time.Second
	if combineStderr {
		return cmd.CombinedOutput()
	}
	return cmd.Output()
}

// dispatchHostProbeShell is ONE PowerShell process kept alive across probe
// tasks: the warm shell the rack hands out. Every task is serialized on mu, so
// the request/sentinel protocol on the pipe can never interleave.
type dispatchHostProbeShell struct {
	mu     sync.Mutex
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	dead   bool
}

// spawnDispatchHostProbeShell cold-starts a warm probe shell. This is the ONLY
// console creation the reuse spine performs, however many probes run on it.
func spawnDispatchHostProbeShell(string) (dispatchtick.WarmShell, error) {
	cmd := windowgate.Command("powershell", "-NoProfile", "-NonInteractive", "-Command", dispatchHostProbeBootstrap)
	cmd.WaitDelay = 5 * time.Second
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, err
	}
	return &dispatchHostProbeShell{cmd: cmd, stdin: stdin, stdout: bufio.NewReaderSize(stdout, 1<<16)}, nil
}

// RunTask writes one base64 task line and reads back everything the shell wrote
// before the sentinel. Any wire failure or overrun marks the shell dead and
// kills it, so the rack retires it instead of reusing a wedged console.
func (s *dispatchHostProbeShell) RunTask(task string) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.dead {
		return nil, errors.New("host probe shell is dead")
	}
	line := base64.StdEncoding.EncodeToString([]byte(task)) + "\n"
	if _, err := io.WriteString(s.stdin, line); err != nil {
		s.teardownLocked()
		return nil, err
	}
	type readResult struct {
		out []byte
		err error
	}
	done := make(chan readResult, 1)
	go func() {
		var buf bytes.Buffer
		for {
			text, err := s.stdout.ReadString('\n')
			if strings.TrimRight(text, "\r\n") == dispatchHostProbeSentinel {
				done <- readResult{out: append([]byte(nil), buf.Bytes()...)}
				return
			}
			buf.WriteString(text)
			if err != nil {
				done <- readResult{err: fmt.Errorf("host probe shell output ended before its sentinel: %w", err)}
				return
			}
		}
	}()
	select {
	case res := <-done:
		if res.err != nil {
			s.teardownLocked()
			return nil, res.err
		}
		return res.out, nil
	case <-time.After(dispatchHostProbeTaskTimeout):
		// The reader goroutine unblocks when teardown closes the pipe.
		s.teardownLocked()
		return nil, fmt.Errorf("host probe shell task exceeded %s", dispatchHostProbeTaskTimeout)
	}
}

// Healthy reports whether the shell may serve another task. It goes false the
// moment a task breaks the pipe or overruns, which is what makes the rack's
// health-checked reuse meaningful rather than decorative.
func (s *dispatchHostProbeShell) Healthy() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return !s.dead
}

// Close tears the console down. Closing stdin ends the bootstrap loop so
// PowerShell exits on its own and its conhost goes with it; the kill is only
// the backstop for a wedged shell, and both waits are bounded so teardown can
// never hang a tick.
func (s *dispatchHostProbeShell) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.teardownLocked()
	return nil
}

func (s *dispatchHostProbeShell) teardownLocked() {
	if s.dead {
		return
	}
	s.dead = true
	_ = s.stdin.Close()
	exited := make(chan struct{})
	go func() { _ = s.cmd.Wait(); close(exited) }()
	select {
	case <-exited:
		return
	case <-time.After(2 * time.Second):
	}
	if s.cmd.Process != nil {
		_ = s.cmd.Process.Kill()
	}
	select {
	case <-exited:
	case <-time.After(2 * time.Second):
	}
}

func dispatchRunExternalJSONImpl(root string, timeout time.Duration, name string, args ...string) (map[string]any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, name, args...)
	cmd.Dir = root
	// Bound the post-deadline pipe wait: `dos` is a pip console-script shim
	// whose real work runs in a python.exe GRANDCHILD holding the inherited
	// stdout pipe. When the context fires, Go kills only the shim; without a
	// WaitDelay, CombinedOutput() then blocks unboundedly until the grandchild
	// exits -- the dispatch tick "hangs past its timeout" class.
	cmd.WaitDelay = 10 * time.Second
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if obj, perr := lastJSONObject(out); perr == nil {
		return obj, nil
	}
	if err != nil {
		return nil, err
	}
	return nil, errors.New("no JSON object in helper output")
}

func dispatchProbeProcessesNative() dispatchtick.ProcGuardInput {
	procs, err := dispatchScanProcesses()
	collectError := ""
	if err != nil {
		collectError = err.Error()
	}
	return dispatchtick.ProcGuardInput{
		Processes:     procs,
		CollectError:  collectError,
		Thresholds:    dispatchtick.DefaultProcGuardThresholds(),
		ProtectedPIDs: []int{os.Getpid(), os.Getppid()},
	}
}

func dispatchScanProcesses() ([]dispatchtick.ProcInfo, error) {
	if runtime.GOOS == "windows" {
		return dispatchScanProcessesWindows()
	}
	return dispatchScanProcessesPOSIX()
}

// dispatchProcScanScript enumerates every live process for the procguard input.
const dispatchProcScanScript = "Get-Process -ErrorAction SilentlyContinue | ForEach-Object { " +
	"try { [pscustomobject]@{ pid=$_.Id; name=$_.ProcessName; threads=$_.Threads.Count; handles=$_.HandleCount; ws_mb=[int64]($_.WorkingSet64 / 1MB) } } catch {} " +
	"} | ConvertTo-Json -Compress"

func dispatchScanProcessesWindows() ([]dispatchtick.ProcInfo, error) {
	out, err := dispatchRunHostProbeFunc(dispatchProcScanScript, 60*time.Second, true)
	if err != nil {
		return nil, err
	}
	rows, decodeErr := decodeDispatchProcessRows(out)
	if decodeErr == nil {
		return rows, nil
	}
	// A warm PowerShell shell can transiently return an empty/truncated frame while
	// another fleet probe is turning over. Retry once through the same fallback-aware
	// probe seam before typing the process dimension unreadable.
	out, retryErr := dispatchRunHostProbeOneShotFunc(dispatchProcScanScript, 60*time.Second, true)
	if retryErr != nil {
		return nil, retryErr
	}
	return decodeDispatchProcessRows(out)
}

func decodeDispatchProcessRows(out []byte) ([]dispatchtick.ProcInfo, error) {
	var rows []struct {
		PID     int    `json:"pid"`
		Name    string `json:"name"`
		Threads int    `json:"threads"`
		Handles int    `json:"handles"`
		WSMB    int    `json:"ws_mb"`
	}
	if uerr := json.Unmarshal(out, &rows); uerr != nil {
		var one struct {
			PID     int    `json:"pid"`
			Name    string `json:"name"`
			Threads int    `json:"threads"`
			Handles int    `json:"handles"`
			WSMB    int    `json:"ws_mb"`
		}
		if oerr := json.Unmarshal(out, &one); oerr != nil {
			return nil, uerr
		}
		rows = []struct {
			PID     int    `json:"pid"`
			Name    string `json:"name"`
			Threads int    `json:"threads"`
			Handles int    `json:"handles"`
			WSMB    int    `json:"ws_mb"`
		}{one}
	}
	procs := make([]dispatchtick.ProcInfo, 0, len(rows))
	for _, row := range rows {
		procs = append(procs, dispatchtick.ProcInfo{
			PID:          row.PID,
			Name:         row.Name,
			Threads:      dispatchtick.IntPtr(row.Threads),
			Handles:      dispatchtick.IntPtr(row.Handles),
			WorkingSetMB: dispatchtick.IntPtr(row.WSMB),
		})
	}
	return procs, nil
}

// dispatchScanProcessesPOSIX reads the preflight's process census through procguard rather
// than shelling out to `ps` here (#5537).
//
// What this used to be, and why both halves were wrong: one hard-coded
// `ps -eo pid=,nlwp=,rss=,comm=` for every POSIX host, read with .Output() and
// `return nil, err`. `nlwp` is a procps-ng extension; BSD `ps` has no thread-count keyword
// at all and rejects the whole invocation rather than dropping that column, so on macOS
// this probe returned an error every tick, dispatchProbeProcessesNative folded it into
// collect_error, and the procguard dimension was skipped on every dispatch. Discarding
// stdout on a non-zero exit is the second half: a `ps` that printed a usable table and then
// exited 1 read as total failure.
//
// procguard.CollectProcesses is the one enumeration implementation that already fixed both
// per GOOS (psCensusSpec / runTool / censusError, #5385). Keeping a second copy of the
// invocation here is what let this site rot behind the fix.
func dispatchScanProcessesPOSIX() ([]dispatchtick.ProcInfo, error) {
	procs, collectErr := procguard.CollectProcesses()
	if collectErr != "" {
		return nil, errors.New(collectErr)
	}
	return dispatchProcInfoFromProcguard(procs), nil
}

// dispatchProcInfoFromProcguard projects a procguard census onto the preflight's ProcInfo.
//
// The one rule with teeth here is that an ABSENT dimension stays nil. procguard types "this
// host's `ps` dialect has no keyword for that column" as psNoColumn and leaves the field
// nil; EvaluateProcGuard skips a nil dimension as unread, whereas a 0 would be a
// measurement — the claim that a process has zero threads, which would put every macOS
// process permanently under the thread ceiling and silently disable the dimension this
// guard exists for. So the pointers are carried across as they are and never defaulted.
func dispatchProcInfoFromProcguard(procs []procguard.Proc) []dispatchtick.ProcInfo {
	out := make([]dispatchtick.ProcInfo, 0, len(procs))
	for _, p := range procs {
		out = append(out, dispatchtick.ProcInfo{
			PID:          p.PID,
			Name:         p.Name,
			Threads:      p.Threads,
			Handles:      p.Handles,
			WorkingSetMB: p.WSMB,
		})
	}
	return out
}

var dispatchBuildHostResources = dispatchPreflightHostResourcesFromProcesses

func dispatchPreflightHostResources() dispatchtick.HostResources {
	return dispatchPreflightHostResourcesFromProcesses(dispatchProbeProcesses())
}

func dispatchPreflightHostResourcesFromProcesses(processes dispatchtick.ProcGuardInput) dispatchtick.HostResources {
	cores := runtime.NumCPU()
	freeRAM := dispatchFreeRAM()
	totalThreads := 0
	seenThreads := false
	for _, proc := range processes.Processes {
		if proc.Threads != nil {
			totalThreads += *proc.Threads
			seenThreads = true
		}
	}
	var threads *int
	if seenThreads {
		threads = &totalThreads
	}
	return dispatchtick.HostResources{Cores: &cores, FreeRAMMB: freeRAM, TotalThreads: threads}
}

// dispatchFreeRAMScript reads free physical memory (KB) for the host-resource probe.
const dispatchFreeRAMScript = "$os=Get-CimInstance Win32_OperatingSystem; [int64]$os.FreePhysicalMemory"

func dispatchFreeRAM() *int {
	if runtime.GOOS != "windows" {
		return dispatchFreeRAMPOSIX()
	}
	out, err := dispatchRunHostProbe(dispatchFreeRAMScript, 20*time.Second, false)
	if err != nil {
		return nil
	}
	kb, err := strconv.ParseInt(strings.TrimSpace(string(out)), 10, 64)
	if err != nil {
		return nil
	}
	mb := int(kb / 1024)
	return &mb
}

// dispatchFreeRAMPOSIX reads free physical memory (MB) on a POSIX host.
//
// It used to be dispatchRAMAndThreadsPOSIX and also ran `ps -eo nlwp=`, summing the column
// into a host-wide thread total. That second scan is gone, for two independent reasons
// either of which is sufficient (#5537):
//
//   - `nlwp` is a procps-ng extension. BSD `ps` has no thread-count keyword whatsoever
//     (see psCensusSpec in internal/procguard), so on macOS the invocation could only ever
//     fail — there is no darwin argv that answers this question, which is exactly why
//     procguard types the BSD thread column as absent rather than guessing a substitute.
//   - The total was already dead on arrival. The sole caller took `free, _ :=` and dropped
//     it; dispatchPreflightHostResourcesFromProcesses derives TotalThreads by summing the
//     per-process census it is handed, which now carries the per-GOOS-correct thread column
//     and leaves it nil where the dialect has no keyword.
//
// So a POSIX tick paid for a second `ps` spawn whose answer nobody read and which could not
// be answered on half the supported hosts. Removing it changes no reported value.
//
// Note what this function still does NOT do: /proc/meminfo is Linux-only, so free RAM is
// nil on darwin. That is a pre-existing gap, reported as unknown rather than as zero.
func dispatchFreeRAMPOSIX() *int {
	var freeRAM *int
	if b, err := os.ReadFile("/proc/meminfo"); err == nil {
		for _, line := range strings.Split(string(b), "\n") {
			if strings.HasPrefix(line, "MemAvailable:") {
				fields := strings.Fields(line)
				if len(fields) >= 2 {
					if kb, err := strconv.Atoi(fields[1]); err == nil {
						mb := kb / 1024
						freeRAM = &mb
					}
				}
				break
			}
		}
	}
	return freeRAM
}

func dispatchProductWorkerCount(root, product string) int {
	return len(dispatchProductWorkerPIDs(root, product))
}

// dispatchProductWorkerPIDs is the identity behind dispatchProductWorkerCount: the set of
// live worker PIDs for a product -- lease-tracked resolve/repair pidfiles, goal-run
// breadcrumbs, cmdline-marked workers (`resolve GitHub issue #` / `dos-dispatch-loop`),
// plus codex ambient sessions. The count is len() of this set; exposing the set lets the
// #3109 self-heal name the exact orphan PIDs preflight counts as unattributed_live.
func dispatchProductWorkerPIDs(root, product string) map[int]bool {
	pids := dispatchLiveResolveWorkerPIDs(filepath.Join(root, dispatchtick.RunsDirName), product)
	// Snapshot the host worker-process table ONCE per call and share it: both the
	// goal-breadcrumb and cmdline-marker passes below classify the same Win32_Process
	// rows, yet each used to spawn its OWN dispatchProbeWorkerProcessRows() -- a cold
	// PowerShell start + full-table Get-CimInstance enumeration (~0.3-1.5s on a busy
	// box) -- so a claude/unscoped preflight paid the identical scan twice every tick.
	// One scan serves both; a scan error yields nil rows, which folds each pass to the
	// same empty result the old per-caller `if err != nil { return out }` produced.
	rows, _ := dispatchProbeWorkerProcessRows()
	for pid := range dispatchLiveGoalWorkerPIDs(filepath.Join(root, dispatchGoalRunsDirName), product, rows) {
		pids[pid] = true
	}
	for pid := range dispatchCmdlineWorkerPIDs(product, rows) {
		pids[pid] = true
	}
	if product == "codex" {
		for pid := range dispatchAmbientCodexPIDsExcludingSidecarParents(pids) {
			pids[pid] = true
		}
	}
	return pids
}

// dispatchLeasedWorkerPIDs is the set of worker PIDs that hold a LIVE seat lease -- the
// resolve/repair pidfiles under the runs dir whose PID is still alive. It is the "carries
// a live lease" half of the unattributed_live predicate: a PID in the worker set but NOT
// in this set is an orphan with no seat attribution, the exact thing preflight depletes
// the pool on (#3109). Reads the same leases dispatchPreflightSeat feeds to BuildSeatPool.
func dispatchLeasedWorkerPIDs(root string) map[int]bool {
	out := map[int]bool{}
	for _, lease := range dispatchLiveSeatLeases(filepath.Join(root, dispatchtick.RunsDirName)) {
		if lease.PID > 0 {
			out[lease.PID] = true
		}
	}
	return out
}

// dispatchUnattributedWorklist is the conservative reap worklist for #3109: the sorted
// PIDs that carry the dispatch-worker marker (they are in workerPIDs) AND hold no live
// seat lease (they are absent from leasedPIDs) -- exactly the set preflight counts as
// unattributed_live. A leased worker or an unrelated (non-marker) process can never
// appear here, so the janitor can never sweep something it should not. Pure; no I/O.
func dispatchUnattributedWorklist(workerPIDs, leasedPIDs map[int]bool) []int {
	out := make([]int, 0, len(workerPIDs))
	for pid := range workerPIDs {
		if pid > 0 && !leasedPIDs[pid] {
			out = append(out, pid)
		}
	}
	sort.Ints(out)
	return out
}

// dispatchReapPID is the destructive TREE reaper the #3109 self-heal routes each orphan
// PID through. It defaults to procguard.KillPID -- a process-tree kill (native job
// termination / taskkill /T on Windows, process-group/descendant SIGKILL on POSIX) -- so
// an orphan's own descendants (the node runtime + MCP/tool subprocesses a `claude`
// spawns) are reaped too; a bare kill would leave that subtree behind and re-poison the
// count. Injectable for tests. Mirrors fleetKillPID (fleet.go) / guardChildTreeKill.
var dispatchReapPID = procguard.KillPID

// dispatchReapOutcome records the result of tree-reaping one orphan PID from the janitor
// worklist -- surfaced on the refused dispatch-tick payload as an audit trail.
type dispatchReapOutcome struct {
	PID    int    `json:"pid"`
	OK     bool   `json:"ok"`
	Detail string `json:"detail,omitempty"`
}

// dispatchReapWorklist tree-reaps every PID in a #3109 janitor worklist through
// dispatchReapPID and returns the per-PID outcome. The dispatch tick calls this only on a
// LIVE tick after preflight has already refused (next-tick recovery), never inside the
// hot admission path -- so a mis-attributed kill under lease TOCTOU is impossible.
func dispatchReapWorklist(worklist []int) []dispatchReapOutcome {
	out := make([]dispatchReapOutcome, 0, len(worklist))
	for _, pid := range worklist {
		if pid <= 0 {
			continue
		}
		ok, detail := dispatchReapPID(pid)
		out = append(out, dispatchReapOutcome{PID: pid, OK: ok, Detail: detail})
	}
	return out
}

const dispatchGoalRunsDirName = ".goal-runs"

type dispatchCodexProcessRow struct {
	PID     int    `json:"pid"`
	PPID    int    `json:"ppid"`
	Name    string `json:"name"`
	Cmdline string `json:"cmdline"`
}

func dispatchAmbientCodexProcessCount() int {
	return len(dispatchAmbientCodexPIDs())
}

func dispatchAmbientCodexPIDs() map[int]bool {
	rows, err := dispatchProbeCodexProcessRows()
	if err != nil {
		return map[int]bool{}
	}
	return dispatchCodexProcessPIDs(rows)
}

func dispatchCodexProcessPIDs(rows []dispatchCodexProcessRow) map[int]bool {
	return dispatchCodexProcessPIDsExcludingParents(rows, nil)
}

func dispatchAmbientCodexPIDsExcludingSidecarParents(sidecarPIDs map[int]bool) map[int]bool {
	rows, err := dispatchProbeCodexProcessRows()
	if err != nil {
		return map[int]bool{}
	}
	return dispatchCodexProcessPIDsExcludingParents(rows, sidecarPIDs)
}

func dispatchCodexProcessPIDsExcludingParents(rows []dispatchCodexProcessRow, excludedParents map[int]bool) map[int]bool {
	native := map[int]bool{}
	wrappers := map[int]bool{}
	parent := map[int]int{}
	for _, row := range rows {
		if row.PID <= 0 {
			continue
		}
		parent[row.PID] = row.PPID
		switch {
		case dispatchIsCodexNativeImage(row.Name):
			native[row.PID] = true
		case dispatchIsCodexNodeWrapper(row.Name, row.Cmdline):
			wrappers[row.PID] = true
		}
	}
	wrappersWithNativeChild := map[int]bool{}
	for pid := range native {
		if ppid := parent[pid]; ppid > 0 {
			wrappersWithNativeChild[ppid] = true
		}
	}
	out := map[int]bool{}
	for pid := range native {
		if excludedParents != nil && dispatchPIDHasAncestor(pid, parent, excludedParents) {
			continue
		}
		out[pid] = true
	}
	for pid := range wrappers {
		if !wrappersWithNativeChild[pid] {
			if excludedParents != nil && dispatchPIDHasAncestor(pid, parent, excludedParents) {
				continue
			}
			out[pid] = true
		}
	}
	return out
}

func dispatchPIDHasAncestor(pid int, parents map[int]int, ancestors map[int]bool) bool {
	seen := map[int]bool{}
	for pid > 0 && !seen[pid] {
		seen[pid] = true
		parent := parents[pid]
		if ancestors[parent] {
			return true
		}
		pid = parent
	}
	return false
}

const (
	dispatchWorkerCmdMarker       = "dos-dispatch-loop"
	dispatchIssueResolveCmdMarker = "resolve GitHub issue #"
)

// dispatchCmdlineWorkerPIDs classifies a caller-supplied host process table (one
// Win32_Process snapshot shared across the preflight's worker-PID passes, see
// dispatchProductWorkerPIDs) into the marker-cmdline workers for a product. Nil rows
// (the scan errored, or nothing ran) yield an empty set.
func dispatchCmdlineWorkerPIDs(product string, rows []dispatchCodexProcessRow) map[int]bool {
	out := map[int]bool{}
	for _, row := range rows {
		if row.PID <= 0 || !dispatchIsWorkerCmdline(row.Cmdline) {
			continue
		}
		if product != "" && !dispatchProcessImageMatchesProduct(row.Name, product) {
			continue
		}
		out[row.PID] = true
	}
	return out
}

func dispatchIsWorkerCmdline(cmdline string) bool {
	low := strings.ToLower(cmdline)
	return strings.Contains(low, dispatchWorkerCmdMarker) ||
		strings.Contains(low, strings.ToLower(dispatchIssueResolveCmdMarker))
}

func dispatchProcessImageMatchesProduct(name, product string) bool {
	stem := dispatchProcessNameStem(name)
	if stem == "" {
		return false
	}
	for _, backend := range dispatchProductBackends(product) {
		backend = strings.TrimSpace(backend)
		if backend != "" && (stem == backend || strings.HasPrefix(stem, backend)) {
			return true
		}
	}
	return false
}

func dispatchIsCodexNativeImage(name string) bool {
	return dispatchProcessNameStem(name) == "codex"
}

func dispatchIsCodexNodeWrapper(name, cmdline string) bool {
	if dispatchProcessNameStem(name) != "node" {
		return false
	}
	low := strings.ToLower(strings.ReplaceAll(cmdline, "\\", "/"))
	return strings.Contains(low, "@openai/codex") || strings.Contains(low, "codex/bin/codex.js")
}

func dispatchProcessNameStem(name string) string {
	base := strings.ToLower(strings.Trim(strings.TrimSpace(name), `"`))
	base = strings.ReplaceAll(base, "\\", "/")
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	for _, ext := range []string{".exe", ".cmd", ".bat"} {
		if strings.HasSuffix(base, ext) {
			base = strings.TrimSuffix(base, ext)
			break
		}
	}
	return base
}

func dispatchScanCodexProcessRowsNative() ([]dispatchCodexProcessRow, error) {
	if runtime.GOOS == "windows" {
		return dispatchScanCodexProcessRowsWindows()
	}
	return dispatchScanCodexProcessRowsPOSIX()
}

func dispatchScanWorkerProcessRowsNative() ([]dispatchCodexProcessRow, error) {
	if runtime.GOOS == "windows" {
		return dispatchScanWorkerProcessRowsWindows()
	}
	return dispatchScanWorkerProcessRowsPOSIX()
}

// dispatchWorkerRowScanScript lists every agent-worker-shaped process row.
const dispatchWorkerRowScanScript = "$rows = @(Get-CimInstance Win32_Process " +
	"-Filter \"Name = 'claude.exe' OR Name = 'opencode.exe' OR Name = 'codex.exe' OR Name = 'node.exe'\" | " +
	"Select-Object @{n='pid';e={$_.ProcessId}},@{n='ppid';e={$_.ParentProcessId}},@{n='name';e={$_.Name}},@{n='cmdline';e={$_.CommandLine}}); " +
	"$rows | ConvertTo-Json -Compress"

func dispatchScanWorkerProcessRowsWindows() ([]dispatchCodexProcessRow, error) {
	out, err := dispatchRunHostProbe(dispatchWorkerRowScanScript, 10*time.Second, false)
	if err != nil {
		return nil, err
	}
	return decodeDispatchCodexProcessRows(out)
}

func dispatchScanWorkerProcessRowsPOSIX() ([]dispatchCodexProcessRow, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "ps", "-eo", "pid=,ppid=,comm=,args=")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	rows := []dispatchCodexProcessRow{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		pid, perr := strconv.Atoi(fields[0])
		ppid, pperr := strconv.Atoi(fields[1])
		if perr != nil || pperr != nil {
			continue
		}
		name := fields[2]
		cmdline := name
		if len(fields) > 3 {
			cmdline = strings.Join(fields[3:], " ")
		}
		rows = append(rows, dispatchCodexProcessRow{PID: pid, PPID: ppid, Name: name, Cmdline: cmdline})
	}
	return rows, nil
}

// dispatchCodexRowScanScript lists the codex/node process rows the ambient-session
// attribution walks.
const dispatchCodexRowScanScript = "$rows = @(Get-CimInstance Win32_Process " +
	"-Filter \"Name = 'codex.exe' OR Name = 'node.exe'\" | " +
	"Select-Object @{n='pid';e={$_.ProcessId}},@{n='ppid';e={$_.ParentProcessId}},@{n='name';e={$_.Name}},@{n='cmdline';e={$_.CommandLine}}); " +
	"$rows | ConvertTo-Json -Compress"

func dispatchScanCodexProcessRowsWindows() ([]dispatchCodexProcessRow, error) {
	out, err := dispatchRunHostProbe(dispatchCodexRowScanScript, 10*time.Second, false)
	if err != nil {
		return nil, err
	}
	return decodeDispatchCodexProcessRows(out)
}

func dispatchScanCodexProcessRowsPOSIX() ([]dispatchCodexProcessRow, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "ps", "-eo", "pid=,ppid=,comm=,args=")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	rows := []dispatchCodexProcessRow{}
	for _, line := range strings.Split(string(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		pid, perr := strconv.Atoi(fields[0])
		ppid, pperr := strconv.Atoi(fields[1])
		if perr != nil || pperr != nil {
			continue
		}
		name := fields[2]
		cmdline := name
		if len(fields) > 3 {
			cmdline = strings.Join(fields[3:], " ")
		}
		if dispatchIsCodexNativeImage(name) || dispatchIsCodexNodeWrapper(name, cmdline) {
			rows = append(rows, dispatchCodexProcessRow{PID: pid, PPID: ppid, Name: name, Cmdline: cmdline})
		}
	}
	return rows, nil
}

func decodeDispatchCodexProcessRows(out []byte) ([]dispatchCodexProcessRow, error) {
	text := strings.TrimSpace(string(out))
	if text == "" {
		return nil, nil
	}
	var rows []dispatchCodexProcessRow
	if err := json.Unmarshal([]byte(text), &rows); err == nil {
		return rows, nil
	}
	var one dispatchCodexProcessRow
	if err := json.Unmarshal([]byte(text), &one); err != nil {
		return nil, err
	}
	return []dispatchCodexProcessRow{one}, nil
}

func dispatchLiveResolveWorkerPIDs(runsDir, product string) map[int]bool {
	out := map[int]bool{}
	if st, err := os.Stat(runsDir); err != nil || !st.IsDir() {
		return out
	}
	for _, pidFile := range dispatchWorkerPIDFiles(runsDir) {
		if !dispatchResolvePIDRE.MatchString(filepath.Base(pidFile)) {
			continue
		}
		if product != "" && !dispatchBackendInProduct(dispatchReadBackendSidecar(pidFile), product) {
			continue
		}
		pid, ok := readPID(pidFile)
		if ok && dispatchPIDAlive(pid) {
			out[pid] = true
		}
	}
	return out
}

func dispatchWorkerPIDFiles(runsDir string) []string {
	matches := []string{}
	for _, pattern := range []string{"resolve-*.pid", "repair-*.pid"} {
		got, _ := filepath.Glob(filepath.Join(runsDir, pattern))
		matches = append(matches, got...)
	}
	sort.Strings(matches)
	return matches
}

// dispatchLiveGoalWorkerPIDs maps live goal-run breadcrumbs to worker PIDs, verifying
// each against the caller-supplied host process table (the single Win32_Process
// snapshot shared across the preflight's worker-PID passes, see
// dispatchProductWorkerPIDs). Nil rows (scan errored, or nothing ran) yield an empty
// set -- the same result the old per-caller scan-error early-return produced.
func dispatchLiveGoalWorkerPIDs(goalRunsDir, product string, rows []dispatchCodexProcessRow) map[int]bool {
	out := map[int]bool{}
	if st, err := os.Stat(goalRunsDir); err != nil || !st.IsDir() {
		return out
	}
	// tools/launch_goal_detached.ps1 is a Claude launcher; its breadcrumbs have
	// no backend sidecar, so a product-scoped count can only assign them to the
	// Claude pool. Empty product is the unscoped/global fold.
	if product != "" && product != "claude" {
		return out
	}
	byPID := map[int]dispatchCodexProcessRow{}
	for _, row := range rows {
		if row.PID > 0 {
			byPID[row.PID] = row
		}
	}
	matches, _ := filepath.Glob(filepath.Join(goalRunsDir, "*.pid"))
	sort.Strings(matches)
	for _, pidFile := range matches {
		if !dispatchGoalPIDRE.MatchString(filepath.Base(pidFile)) {
			continue
		}
		pid, ok := readPID(pidFile)
		if !ok || !dispatchPIDAlive(pid) {
			continue
		}
		row, ok := byPID[pid]
		if !ok || !dispatchProcessImageMatchesProduct(row.Name, "claude") {
			continue
		}
		// A stale breadcrumb reused by an unrelated system process must not
		// consume a worker slot. The launcher starts Claude, so require the
		// current PID to resolve to a Claude worker image before counting it.
		out[pid] = true
	}
	return out
}

func dispatchReadBackendSidecar(pidFile string) string {
	b, err := os.ReadFile(strings.TrimSuffix(pidFile, filepath.Ext(pidFile)) + ".backend")
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func dispatchBackendInProduct(backend, product string) bool {
	backend = strings.TrimSpace(backend)
	for _, candidate := range dispatchProductBackends(product) {
		if backend == candidate {
			return true
		}
	}
	return false
}

func dispatchProductBackends(product string) []string {
	switch product {
	case "claude":
		return []string{"claude"}
	case "opencode":
		return []string{"opencode"}
	case "codex":
		return []string{"codex"}
	default:
		return []string{product}
	}
}

const dispatchTreeBuildSuccessTTL = 2 * time.Minute

type dispatchTreeBuildSuccess struct {
	head string
	at   time.Time
}

var dispatchTreeBuildSuccesses struct {
	sync.Mutex
	byRoot map[string]dispatchTreeBuildSuccess
}

var dispatchTreeBuildHead = func(root string) string {
	cmd := exec.Command("git", "-C", root, "rev-parse", "HEAD")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func dispatchTreeBuildKey(root string) (string, string) {
	head := dispatchTreeBuildHead(root)
	if head == "" {
		return "", ""
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return "", ""
	}
	return absRoot, head
}

func dispatchTreeBuildSucceededRecently(root, head string, now time.Time) bool {
	if root == "" || head == "" {
		return false
	}
	dispatchTreeBuildSuccesses.Lock()
	defer dispatchTreeBuildSuccesses.Unlock()
	hit, ok := dispatchTreeBuildSuccesses.byRoot[root]
	return ok && hit.head == head && now.Sub(hit.at) <= dispatchTreeBuildSuccessTTL
}

func dispatchRecordTreeBuildSuccess(root, head string, now time.Time) {
	if root == "" || head == "" {
		return
	}
	dispatchTreeBuildSuccesses.Lock()
	defer dispatchTreeBuildSuccesses.Unlock()
	if dispatchTreeBuildSuccesses.byRoot == nil {
		dispatchTreeBuildSuccesses.byRoot = map[string]dispatchTreeBuildSuccess{}
	}
	for workspaceRoot, hit := range dispatchTreeBuildSuccesses.byRoot {
		if now.Sub(hit.at) > dispatchTreeBuildSuccessTTL {
			delete(dispatchTreeBuildSuccesses.byRoot, workspaceRoot)
		}
	}
	dispatchTreeBuildSuccesses.byRoot[root] = dispatchTreeBuildSuccess{head: head, at: now}
}

var dispatchTreeBuildCommand = func(root string) (string, error) {
	builds, output, err := trunkbuildprobe.BuildCommittedTarget(root, "./cmd/fak", 90*time.Second)
	if err != nil {
		return output, err
	}
	if !builds {
		return output, errors.New("committed tree build failed")
	}
	return output, nil
}

func dispatchProbeTreeBuild(root string) dispatchtick.TreeCheck {
	now := time.Now()
	cacheRoot, head := dispatchTreeBuildKey(root)
	if dispatchTreeBuildSucceededRecently(cacheRoot, head, now) || committedbuildwitness.Fresh(cacheRoot, head, now) {
		return dispatchtick.TreeCheck{}
	}
	out, err := dispatchTreeBuildCommand(root)
	if err == nil {
		// Record the HEAD observed before the build. If HEAD moved during the
		// probe, the next lookup sees a mismatch and rebuilds rather than
		// attributing the old build to the new commit.
		completedAt := time.Now()
		dispatchRecordTreeBuildSuccess(cacheRoot, head, completedAt)
		committedbuildwitness.Record(cacheRoot, head, "dispatch-preflight", completedAt)
		return dispatchtick.TreeCheck{}
	}
	// Missing toolchain/probe infrastructure fails open; a real compiler diagnostic
	// names a package/file and is the poison witness. A probe root without a Go
	// module (a bare temp dir, a misconfigured root) is infrastructure-missing, not
	// a red tree, so it must fail open too -- otherwise the #3583 poison gate freezes
	// the fleet over a moduleless probe root, and every dispatch test that ticks a
	// temp workspace refuses with TREE_POISONED. `go build` names the missing module
	// two ways: "go.mod file not found ..." in a bare dir, and "cannot find main
	// module, but found .git/config ..." once the root is git-init'd (as the tick
	// test harness leaves it). Both land in combined output, not err.Error().
	//
	// A build that timed out or was killed (context.DeadlineExceeded from the 90s
	// cap, or a SIGKILL "signal: killed" from an OOM reap under fleet load)
	// produced NO compiler diagnostic -- it is infrastructure, not a poison
	// witness -- so it fails open too. Otherwise a slow host quietly converts a
	// healthy tree into a fleet-wide freeze.
	lowered := strings.ToLower(err.Error() + "\n" + out)
	if errors.Is(err, exec.ErrNotFound) ||
		errors.Is(err, context.DeadlineExceeded) ||
		strings.Contains(lowered, "executable file not found") ||
		strings.Contains(lowered, "go.mod file not found") ||
		strings.Contains(lowered, "cannot find main module") ||
		strings.Contains(lowered, "not a valid object name: head") ||
		strings.Contains(lowered, "does not have any commits yet") ||
		strings.Contains(lowered, "signal: killed") {
		detail := strings.TrimSpace(out)
		if detail == "" {
			detail = err.Error()
		}
		return dispatchtick.TreeCheck{Error: detail}
	}
	line := ""
	for _, candidate := range strings.Split(strings.TrimSpace(out), "\n") {
		candidate = strings.TrimSpace(candidate)
		if candidate != "" {
			line = candidate
			break
		}
	}
	if line == "" {
		line = err.Error()
	}
	return dispatchtick.TreeCheck{Poisoned: true, Package: line, Error: err.Error()}
}
