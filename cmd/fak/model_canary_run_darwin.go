//go:build darwin

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/processstart"
	"golang.org/x/sys/unix"
)

type darwinModelCanaryRuntime struct {
	mu            sync.Mutex
	tools         map[string]string
	executableSHA map[string]string
}

type darwinModelCanaryLease struct{ lease *gpulease.Lease }

func (l *darwinModelCanaryLease) Release() error {
	if l == nil || l.lease == nil {
		return nil
	}
	l.lease.Release()
	l.lease = nil
	return nil
}

type darwinModelCanaryProcess struct {
	cmd       *exec.Cmd
	done      chan struct{}
	signalMu  sync.Mutex // serializes nonblocking reap with identity-check-plus-TERM
	mu        sync.Mutex
	err       error
	exitCode  int
	completed time.Time
	stdout    *modelCanaryBoundedBuffer
	stderr    *modelCanaryBoundedBuffer
}

const modelCanaryRequestOutputLimit = 16 << 20

type modelCanaryBoundedBuffer struct {
	mu       sync.Mutex
	buffer   bytes.Buffer
	limit    int
	overflow bool
}

func (b *modelCanaryBoundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	remaining := b.limit - b.buffer.Len()
	if remaining > 0 {
		chunk := p
		if len(chunk) > remaining {
			chunk = chunk[:remaining]
		}
		_, _ = b.buffer.Write(chunk)
	}
	if len(p) > remaining {
		b.overflow = true
	}
	return len(p), nil
}

func (b *modelCanaryBoundedBuffer) snapshot() ([]byte, bool) {
	if b == nil {
		return nil, false
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.buffer.Bytes()...), b.overflow
}

func modelCanaryLiveDependencies() (modelCanaryRunDeps, error) {
	if runtime.GOARCH != "arm64" {
		return modelCanaryRunDeps{}, fmt.Errorf("model canary-run live execution requires darwin/arm64; this host is %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	live := &darwinModelCanaryRuntime{}
	return modelCanaryRunDeps{
		Platform:           runtime.GOOS,
		Architecture:       runtime.GOARCH,
		Now:                time.Now,
		Preflight:          live.preflight,
		AcquireLease:       live.acquireLease,
		VerifyIncumbent:    live.verifyIncumbent,
		BootoutIncumbent:   live.bootoutIncumbent,
		StartCandidate:     live.startCandidate,
		WaitCandidateReady: live.waitCandidateReady,
		StartRequest:       live.startRequest,
		PollRequest:        live.pollRequest,
		RequestEvidence:    live.requestEvidence,
		StopRequest:        live.stopRequest,
		Sample:             live.sample,
		Sleep:              sleepModelCanaryContext,
		TermCandidate:      live.termCandidate,
		RestoreIncumbent:   live.restoreIncumbent,
		EndpointsStable:    live.endpointsStable,
	}, nil
}

func (d *darwinModelCanaryRuntime) preflight(ctx context.Context, cfg modelCanaryRunConfig) (modelCanaryPreflight, error) {
	tools := make(map[string]string)
	for _, name := range []string{"lsof", "ps", "footprint", "sysctl", "memory_pressure", "launchctl"} {
		path, err := exec.LookPath(name)
		if err != nil {
			return modelCanaryPreflight{}, fmt.Errorf("required Darwin executable %s: %w", name, err)
		}
		absolute, err := filepath.Abs(path)
		if err != nil {
			return modelCanaryPreflight{}, fmt.Errorf("resolve Darwin executable %s: %w", name, err)
		}
		tools[name] = filepath.Clean(absolute)
	}
	for name, argv := range map[string][]string{
		"candidate": cfg.Candidate.Command,
		"request":   cfg.Request.Command,
		"restore":   cfg.Incumbent.RestoreCommand,
	} {
		path, err := exec.LookPath(argv[0])
		if err != nil {
			return modelCanaryPreflight{}, fmt.Errorf("%s executable %q: %w", name, argv[0], err)
		}
		absolute, err := filepath.Abs(path)
		if err != nil {
			return modelCanaryPreflight{}, fmt.Errorf("resolve %s executable: %w", name, err)
		}
		tools[name] = filepath.Clean(absolute)
	}
	executableSHA := make(map[string]string, len(tools))
	for name, path := range tools {
		raw, err := os.ReadFile(path)
		if err != nil {
			return modelCanaryPreflight{}, fmt.Errorf("hash preflighted executable %s: %w", name, err)
		}
		executableSHA[name] = digestBytes(raw)
	}
	if err := validateDarwinModelCanaryRestore(cfg, tools); err != nil {
		return modelCanaryPreflight{}, err
	}
	plistRaw, err := os.ReadFile(cfg.Incumbent.RestorePlist)
	if err != nil {
		return modelCanaryPreflight{}, fmt.Errorf("read declared restore plist: %w", err)
	}
	plistSHA := digestBytes(plistRaw)
	if !sameModelCanarySHA256(plistSHA, cfg.Incumbent.RestorePlistSHA256) {
		return modelCanaryPreflight{}, errors.New("declared restore plist SHA256 does not match its bytes")
	}

	d.setTools(tools, executableSHA)
	incumbent, err := d.resolveListenerIdentity(ctx, cfg.Incumbent.ListenerPort)
	if err != nil {
		return modelCanaryPreflight{}, fmt.Errorf("resolve incumbent listener: %w", err)
	}
	if !sameModelCanarySHA256(incumbent.ArgvSHA256, cfg.Incumbent.ExpectedArgvSHA256) {
		return modelCanaryPreflight{}, errors.New("incumbent listener argv hash does not match incumbent.expected_argv_sha256")
	}
	launchdPID, launchdPlist, err := d.readLaunchdService(ctx, cfg.Incumbent.LaunchdTarget)
	if err != nil {
		return modelCanaryPreflight{}, fmt.Errorf("verify declared launchd target: %w", err)
	}
	if launchdPID != incumbent.PID {
		return modelCanaryPreflight{}, fmt.Errorf("declared launchd target PID %d does not own listener PID %d", launchdPID, incumbent.PID)
	}
	if filepath.Clean(launchdPlist) != filepath.Clean(cfg.Incumbent.RestorePlist) {
		return modelCanaryPreflight{}, errors.New("declared launchd target plist does not match incumbent.restore_plist")
	}

	// Exercise every live, read-only parser before acquiring launch cardinality or mutating
	// launchd. A missing source is a refusal, never an implicit zero.
	probe, err := d.sampleProcess(ctx, modelCanaryProcess{Identity: incumbent}, 0)
	if err != nil {
		return modelCanaryPreflight{}, fmt.Errorf("read-only Darwin parser preflight: %w", err)
	}
	return modelCanaryPreflight{
		Incumbent: incumbent, BaselineSwapBytes: probe.SwapUsedBytes,
		Tools: tools, ExecutableSHA256: executableSHA, RestorePlistSHA256: plistSHA,
	}, nil
}

func (d *darwinModelCanaryRuntime) acquireLease(_ context.Context, cfg modelCanaryLeaseConfig, timeout time.Duration) (modelCanaryLease, error) {
	lease, err := gpulease.Acquire(gpulease.Options{Path: cfg.Path, Timeout: timeout})
	if err != nil {
		return nil, err
	}
	return &darwinModelCanaryLease{lease: lease}, nil
}

func (d *darwinModelCanaryRuntime) verifyIncumbent(ctx context.Context, cfg modelCanaryRunConfig, expected modelCanaryProcessIdentity) (modelCanaryProcessIdentity, error) {
	if err := d.verifyRestoreInputs(cfg); err != nil {
		return modelCanaryProcessIdentity{}, fmt.Errorf("restore identity changed after preflight: %w", err)
	}
	current, err := d.resolveListenerIdentity(ctx, cfg.Incumbent.ListenerPort)
	if err != nil {
		return modelCanaryProcessIdentity{}, err
	}
	if !current.equal(expected) {
		return modelCanaryProcessIdentity{}, &modelCanaryRefusal{Reason: modelCanaryReasonIncumbentIdentityMismatch, Phase: modelCanaryPhaseIncumbentVerified, Detail: "listener PID/start/argv identity changed"}
	}
	launchdPID, launchdPlist, err := d.readLaunchdService(ctx, cfg.Incumbent.LaunchdTarget)
	if err != nil {
		return modelCanaryProcessIdentity{}, err
	}
	if launchdPID != current.PID || filepath.Clean(launchdPlist) != filepath.Clean(cfg.Incumbent.RestorePlist) {
		return modelCanaryProcessIdentity{}, &modelCanaryRefusal{Reason: modelCanaryReasonIncumbentIdentityMismatch, Phase: modelCanaryPhaseIncumbentVerified, Detail: "declared launchd service no longer matches the exact listener"}
	}
	return current, nil
}

func (d *darwinModelCanaryRuntime) bootoutIncumbent(ctx context.Context, cfg modelCanaryRunConfig, expected modelCanaryProcessIdentity, timeout time.Duration) error {
	if _, err := d.verifyIncumbent(ctx, cfg, expected); err != nil {
		return err
	}
	if _, err := d.runTool(ctx, "launchctl", "bootout", cfg.Incumbent.LaunchdTarget); err != nil {
		return fmt.Errorf("launchctl bootout exact target: %w", err)
	}
	deadline := time.Now().Add(timeout)
	for {
		if !d.processMatches(ctx, expected) {
			_, present, err := d.queryListenerIdentity(ctx, cfg.Incumbent.ListenerPort)
			if err != nil {
				return fmt.Errorf("observe listener removal after bootout: %w", err)
			}
			if !present {
				return nil
			}
		}
		if !time.Now().Before(deadline) {
			return errors.New("incumbent remained live or retained its listener after launchctl bootout")
		}
		if err := sleepModelCanaryContext(ctx, 25*time.Millisecond); err != nil {
			return err
		}
	}
}

func (d *darwinModelCanaryRuntime) startCandidate(_ context.Context, cfg modelCanaryCandidateConfig) (modelCanaryProcess, error) {
	return d.startOwnedProcess("candidate", cfg.Command, cfg.Environment, false)
}

func (d *darwinModelCanaryRuntime) waitCandidateReady(ctx context.Context, cfg modelCanaryRunConfig, candidate modelCanaryProcess, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		if !d.processMatches(ctx, candidate.Identity) {
			return errors.New("candidate exited or changed identity before readiness")
		}
		listener, err := d.resolveListenerIdentity(ctx, cfg.Candidate.ListenerPort)
		if err == nil && listener.equal(candidate.Identity) {
			if lastErr = probeModelCanaryEndpoints(ctx, cfg.Candidate.ReadinessEndpoints); lastErr == nil {
				return nil
			}
		} else if err != nil {
			lastErr = err
		} else {
			lastErr = errors.New("candidate readiness listener belongs to a different PID/start/argv identity")
		}
		if err := sleepModelCanaryContext(ctx, 100*time.Millisecond); err != nil {
			return err
		}
	}
	return fmt.Errorf("candidate readiness timeout: %w", lastErr)
}

func (d *darwinModelCanaryRuntime) startRequest(_ context.Context, cfg modelCanaryRequestConfig) (modelCanaryProcess, error) {
	return d.startOwnedProcess("request", cfg.Command, cfg.Environment, true)
}

func (d *darwinModelCanaryRuntime) pollRequest(process modelCanaryProcess) (bool, int, error) {
	handle, ok := process.Handle.(*darwinModelCanaryProcess)
	if !ok || handle == nil {
		return true, -1, errors.New("request process handle is invalid")
	}
	select {
	case <-handle.done:
		handle.mu.Lock()
		err, exitCode := handle.err, handle.exitCode
		handle.mu.Unlock()
		return true, exitCode, err
	default:
		return false, 0, nil
	}
}

func (d *darwinModelCanaryRuntime) requestEvidence(process modelCanaryProcess) (modelCanaryRequestEvidence, error) {
	handle, ok := process.Handle.(*darwinModelCanaryProcess)
	if !ok || handle == nil {
		return modelCanaryRequestEvidence{}, errors.New("request process handle is invalid")
	}
	select {
	case <-handle.done:
	default:
		return modelCanaryRequestEvidence{}, errors.New("request evidence requested before terminal response")
	}
	stdout, stdoutOverflow := handle.stdout.snapshot()
	stderr, stderrOverflow := handle.stderr.snapshot()
	if stdoutOverflow || stderrOverflow {
		return modelCanaryRequestEvidence{}, fmt.Errorf("request output exceeded the %d-byte per-stream evidence limit", modelCanaryRequestOutputLimit)
	}
	handle.mu.Lock()
	completed := handle.completed
	handle.mu.Unlock()
	return modelCanaryRequestEvidence{
		CompletedAt: completed.UTC().Format(time.RFC3339Nano),
		Stdout:      string(stdout), StdoutBytes: len(stdout), StdoutSHA256: digestBytes(stdout),
		Stderr: string(stderr), StderrBytes: len(stderr), StderrSHA256: digestBytes(stderr),
	}, nil
}

func (d *darwinModelCanaryRuntime) stopRequest(ctx context.Context, process modelCanaryProcess, timeout time.Duration) error {
	return d.termOwnedProcess(ctx, process, timeout, "request")
}

func (d *darwinModelCanaryRuntime) sample(ctx context.Context, candidate modelCanaryProcess, baselineSwap int64) (modelCanarySample, error) {
	return d.sampleProcess(ctx, candidate, baselineSwap)
}

func (d *darwinModelCanaryRuntime) termCandidate(ctx context.Context, candidate modelCanaryProcess, timeout time.Duration) error {
	return d.termOwnedProcess(ctx, candidate, timeout, "candidate")
}

func (d *darwinModelCanaryRuntime) restoreIncumbent(ctx context.Context, cfg modelCanaryRunConfig, timeout time.Duration) error {
	if err := d.verifyRestoreInputs(cfg); err != nil {
		return fmt.Errorf("restore identity recheck: %w", err)
	}
	tools := d.getTools()
	command := append([]string(nil), cfg.Incumbent.RestoreCommand...)
	command[0] = tools["launchctl"]
	restoreCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(restoreCtx, command[0], command[1:]...)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("declared restore command: %w: %s", err, strings.TrimSpace(string(output)))
	}
	return nil
}

func (d *darwinModelCanaryRuntime) endpointsStable(ctx context.Context, cfg modelCanaryRunConfig, stability, interval time.Duration) (modelCanaryProcessIdentity, error) {
	var stableSince time.Time
	var stableIdentity modelCanaryProcessIdentity
	var lastErr error
	for {
		if err := ctx.Err(); err != nil {
			return modelCanaryProcessIdentity{}, fmt.Errorf("restoration stability context: %w; last observation: %v", err, lastErr)
		}
		identity, err := d.resolveListenerIdentity(ctx, cfg.Incumbent.ListenerPort)
		if err == nil && !sameModelCanarySHA256(identity.ArgvSHA256, cfg.Incumbent.ExpectedArgvSHA256) {
			err = errors.New("restored listener argv hash differs from declared incumbent")
		}
		if err == nil {
			launchdPID, launchdPlist, launchdErr := d.readLaunchdService(ctx, cfg.Incumbent.LaunchdTarget)
			if launchdErr != nil {
				err = launchdErr
			} else if launchdPID != identity.PID || filepath.Clean(launchdPlist) != filepath.Clean(cfg.Incumbent.RestorePlist) {
				err = errors.New("restored listener does not match the declared launchd target/plist")
			}
		}
		if err == nil {
			err = probeModelCanaryEndpoints(ctx, cfg.Incumbent.StableEndpoints)
		}
		now := time.Now()
		if err == nil {
			if stableSince.IsZero() || !identity.equal(stableIdentity) {
				stableSince, stableIdentity = now, identity
			}
			if now.Sub(stableSince) >= stability {
				return stableIdentity, nil
			}
		} else {
			lastErr = err
			stableSince = time.Time{}
			stableIdentity = modelCanaryProcessIdentity{}
		}
		if err := sleepModelCanaryContext(ctx, interval); err != nil {
			return modelCanaryProcessIdentity{}, fmt.Errorf("restoration stability wait: %w; last observation: %v", err, lastErr)
		}
	}
}

func (d *darwinModelCanaryRuntime) startOwnedProcess(binding string, argv []string, environment map[string]string, captureOutput bool) (modelCanaryProcess, error) {
	path, err := d.verifyTool(binding)
	if err != nil {
		return modelCanaryProcess{}, fmt.Errorf("%s executable identity recheck: %w", binding, err)
	}
	actual := append([]string{path}, argv[1:]...)
	cmd := exec.Command(actual[0], actual[1:]...)
	// The declared map is the complete environment, not an overlay. Inheriting the
	// runner's ambient model/runtime variables would make actual execution diverge from
	// the command binding recorded in the receipt.
	cmd.Env = modelCanaryEnvironmentRows(environment)
	var stdout, stderr *modelCanaryBoundedBuffer
	if captureOutput {
		stdout = &modelCanaryBoundedBuffer{limit: modelCanaryRequestOutputLimit}
		stderr = &modelCanaryBoundedBuffer{limit: modelCanaryRequestOutputLimit}
		cmd.Stdout, cmd.Stderr = stdout, stderr
	} else {
		cmd.Stdout, cmd.Stderr = io.Discard, io.Discard
	}
	if err := cmd.Start(); err != nil {
		return modelCanaryProcess{}, err
	}
	handle := &darwinModelCanaryProcess{cmd: cmd, done: make(chan struct{}), exitCode: -1, stdout: stdout, stderr: stderr}
	go handle.reapWithoutPIDReuse()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		identity, err := d.readProcessIdentity(context.Background(), cmd.Process.Pid)
		if err == nil {
			return modelCanaryProcess{Identity: identity, Handle: handle}, nil
		}
		select {
		case <-handle.done:
			return modelCanaryProcess{}, errors.New("process exited before its exact start identity could be bound")
		case <-time.After(5 * time.Millisecond):
		}
	}
	bindErr := errors.New("timed out binding launched process PID/start/argv identity")
	if cleanupErr := handle.termUnbound(2 * time.Second); cleanupErr != nil {
		return modelCanaryProcess{}, fmt.Errorf("%w; exact launch-handle TERM cleanup: %v", bindErr, cleanupErr)
	}
	return modelCanaryProcess{}, fmt.Errorf("%w; exact launch handle was TERM-cleaned", bindErr)
}

func (d *darwinModelCanaryRuntime) termOwnedProcess(ctx context.Context, process modelCanaryProcess, timeout time.Duration, label string) error {
	handle, ok := process.Handle.(*darwinModelCanaryProcess)
	if !ok || handle == nil || handle.cmd == nil || handle.cmd.Process == nil {
		return fmt.Errorf("%s process handle is invalid", label)
	}
	handle.signalMu.Lock()
	select {
	case <-handle.done:
		handle.signalMu.Unlock()
		return nil
	default:
	}
	current, err := d.readProcessIdentity(ctx, process.Identity.PID)
	if err != nil {
		handle.signalMu.Unlock()
		return &modelCanaryRefusal{Reason: modelCanaryReasonCandidateIdentityMismatch, Phase: modelCanaryPhaseCandidateTerminated, Detail: label + " identity unavailable before TERM; no signal sent"}
	}
	if !current.equal(process.Identity) {
		handle.signalMu.Unlock()
		return &modelCanaryRefusal{Reason: modelCanaryReasonCandidateIdentityMismatch, Phase: modelCanaryPhaseCandidateTerminated, Detail: label + " PID/start/argv identity changed before TERM; no signal sent"}
	}
	if err := handle.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		handle.signalMu.Unlock()
		if errors.Is(err, os.ErrProcessDone) || errors.Is(err, syscall.ESRCH) {
			return handle.waitDone(ctx, timeout, label)
		}
		return fmt.Errorf("TERM exact %s: %w", label, err)
	}
	handle.signalMu.Unlock()
	return handle.waitDone(ctx, timeout, label)
}

// reapWithoutPIDReuse leaves an exited child unreaped until it owns signalMu. TERM owns
// the same lock from its final identity observation through Signal. Therefore the PID cannot
// be recycled in the only interval where a signal could otherwise race a concurrent Wait.
func (h *darwinModelCanaryProcess) reapWithoutPIDReuse() {
	pid := h.cmd.Process.Pid
	for {
		h.signalMu.Lock()
		var status unix.WaitStatus
		waitedPID, err := unix.Wait4(pid, &status, unix.WNOHANG, nil)
		if err == nil && waitedPID == 0 {
			h.signalMu.Unlock()
			time.Sleep(5 * time.Millisecond)
			continue
		}
		if err == nil && waitedPID == pid {
			// Wait4 already reaped the process. Cmd.Wait still drains/joins any
			// stdout/stderr copy goroutines; its expected ECHILD is not exit evidence.
			_ = h.cmd.Wait()
			exitCode := -1
			var waitErr error
			switch {
			case status.Exited():
				exitCode = status.ExitStatus()
				if exitCode != 0 {
					waitErr = fmt.Errorf("process exited with status %d", exitCode)
				}
			case status.Signaled():
				exitCode = 128 + int(status.Signal())
				waitErr = fmt.Errorf("process exited from signal %s", status.Signal())
			default:
				waitErr = errors.New("process reached an unrecognized wait status")
			}
			h.finish(exitCode, waitErr)
			h.signalMu.Unlock()
			return
		}
		h.finish(-1, fmt.Errorf("wait for owned process %d: %w", pid, err))
		h.signalMu.Unlock()
		return
	}
}

func (h *darwinModelCanaryProcess) finish(exitCode int, err error) {
	h.mu.Lock()
	h.exitCode = exitCode
	h.err = err
	h.completed = time.Now().UTC()
	h.mu.Unlock()
	close(h.done)
}

func (h *darwinModelCanaryProcess) termUnbound(timeout time.Duration) error {
	h.signalMu.Lock()
	select {
	case <-h.done:
		h.signalMu.Unlock()
		return nil
	default:
	}
	err := h.cmd.Process.Signal(syscall.SIGTERM)
	h.signalMu.Unlock()
	if err != nil && !errors.Is(err, os.ErrProcessDone) && !errors.Is(err, syscall.ESRCH) {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return h.waitDone(ctx, timeout, "unbound owned process")
}

func (h *darwinModelCanaryProcess) waitDone(ctx context.Context, timeout time.Duration, label string) error {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-h.done:
		return nil
	case <-timer.C:
		return fmt.Errorf("%s did not exit after TERM within %s; SIGKILL is forbidden", label, timeout)
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (d *darwinModelCanaryRuntime) sampleProcess(ctx context.Context, process modelCanaryProcess, baselineSwap int64) (modelCanarySample, error) {
	before, err := d.readProcessIdentity(ctx, process.Identity.PID)
	if err != nil || !before.equal(process.Identity) {
		return modelCanarySample{}, errors.New("candidate identity unavailable or changed before sample")
	}
	psRaw, err := d.runTool(ctx, "ps", "-p", strconv.Itoa(process.Identity.PID), "-o", "pid=,rss=")
	if err != nil {
		return modelCanarySample{}, err
	}
	rss, err := parseModelCanaryRSS(psRaw, process.Identity.PID)
	if err != nil {
		return modelCanarySample{}, err
	}
	footprintRaw, err := d.runTool(ctx, "footprint", "-p", strconv.Itoa(process.Identity.PID))
	if err != nil {
		return modelCanarySample{}, err
	}
	footprint, err := parseModelCanaryFootprint(footprintRaw)
	if err != nil {
		return modelCanarySample{}, err
	}
	swapRaw, err := d.runTool(ctx, "sysctl", "vm.swapusage")
	if err != nil {
		return modelCanarySample{}, err
	}
	swap, err := parseModelCanarySwap(swapRaw)
	if err != nil {
		return modelCanarySample{}, err
	}
	pressureRaw, err := d.runTool(ctx, "memory_pressure", "-Q")
	if err != nil {
		return modelCanarySample{}, err
	}
	free, err := parseModelCanaryMemoryPressure(pressureRaw)
	if err != nil {
		return modelCanarySample{}, err
	}
	memorystatusRaw, err := d.runTool(ctx, "sysctl", "-n", "kern.memorystatus_level")
	if err != nil {
		return modelCanarySample{}, err
	}
	memorystatus, err := parseModelCanaryMemorystatus(memorystatusRaw)
	if err != nil {
		return modelCanarySample{}, err
	}
	after, err := d.readProcessIdentity(ctx, process.Identity.PID)
	if err != nil || !after.equal(process.Identity) {
		return modelCanarySample{}, errors.New("candidate identity unavailable or changed during sample")
	}
	raw := map[string]string{
		"ps": string(psRaw), "footprint": string(footprintRaw), "swap": string(swapRaw),
		"memory_pressure": string(pressureRaw), "memorystatus": string(memorystatusRaw),
	}
	rawSHA := make(map[string]string, len(raw))
	for source, body := range raw {
		rawSHA[source] = digestBytes([]byte(body))
	}
	return modelCanarySample{
		ObservedAt: time.Now().UTC().Format(time.RFC3339Nano), Candidate: after,
		RSSBytes: rss, FootprintBytes: footprint, SwapUsedBytes: swap,
		SwapGrowthBytes: swap - baselineSwap, SystemFreePercent: free,
		MemorystatusPercent: memorystatus, Raw: raw, RawSHA256: rawSHA,
	}, nil
}

func (d *darwinModelCanaryRuntime) resolveListenerIdentity(ctx context.Context, port int) (modelCanaryProcessIdentity, error) {
	identity, present, err := d.queryListenerIdentity(ctx, port)
	if err != nil {
		return modelCanaryProcessIdentity{}, err
	}
	if !present {
		return modelCanaryProcessIdentity{}, fmt.Errorf("listener port %d has no owner", port)
	}
	return identity, nil
}

func (d *darwinModelCanaryRuntime) queryListenerIdentity(ctx context.Context, port int) (modelCanaryProcessIdentity, bool, error) {
	path, err := d.verifyTool("lsof")
	if err != nil {
		return modelCanaryProcessIdentity{}, false, err
	}
	args := []string{"-nP", "-iTCP:" + strconv.Itoa(port), "-sTCP:LISTEN", "-Fpcftn"}
	cmd := exec.CommandContext(ctx, path, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	runErr := cmd.Run()
	exitCode := 0
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			exitCode = -1
		}
	}
	present, classifyErr := classifyModelCanaryLsofExit(stdout.Bytes(), stderr.Bytes(), exitCode)
	if classifyErr != nil {
		if runErr != nil {
			return modelCanaryProcessIdentity{}, false, fmt.Errorf("%s %s: %w: %v", path, strings.Join(args, " "), runErr, classifyErr)
		}
		return modelCanaryProcessIdentity{}, false, classifyErr
	}
	if !present {
		return modelCanaryProcessIdentity{}, false, nil
	}
	raw := stdout.Bytes()
	owner, err := parseModelCanaryLsof(raw, port)
	if err != nil {
		return modelCanaryProcessIdentity{}, false, err
	}
	identity, err := d.readProcessIdentity(ctx, owner.PID)
	if err != nil {
		return modelCanaryProcessIdentity{}, false, err
	}
	return identity, true, nil
}

func (d *darwinModelCanaryRuntime) readProcessIdentity(ctx context.Context, pid int) (modelCanaryProcessIdentity, error) {
	started, ok := processstart.Start(pid)
	if !ok {
		return modelCanaryProcessIdentity{}, fmt.Errorf("kernel start identity unavailable for PID %d", pid)
	}
	raw, err := d.runTool(ctx, "ps", "-ww", "-p", strconv.Itoa(pid), "-o", "pid=,lstart=,command=")
	if err != nil {
		return modelCanaryProcessIdentity{}, err
	}
	parsed, _, err := parseModelCanaryPS(raw, time.Local)
	if err != nil {
		return modelCanaryProcessIdentity{}, err
	}
	if parsed.PID != pid {
		return modelCanaryProcessIdentity{}, errors.New("BSD ps returned the wrong PID")
	}
	parsedStart, _ := time.Parse(time.RFC3339Nano, parsed.StartedAt)
	if !parsedStart.Equal(started.Truncate(time.Second)) {
		return modelCanaryProcessIdentity{}, errors.New("BSD ps and kernel process-start identities disagree")
	}
	parsed.StartedAt = started.UTC().Format(time.RFC3339Nano)
	return parsed, nil
}

func (d *darwinModelCanaryRuntime) processMatches(ctx context.Context, expected modelCanaryProcessIdentity) bool {
	current, err := d.readProcessIdentity(ctx, expected.PID)
	return err == nil && current.equal(expected)
}

func (d *darwinModelCanaryRuntime) readLaunchdService(ctx context.Context, target string) (int, string, error) {
	raw, err := d.runTool(ctx, "launchctl", "print", target)
	if err != nil {
		return 0, "", err
	}
	return parseDarwinModelCanaryLaunchctl(raw, target)
}

func validateDarwinModelCanaryRestore(cfg modelCanaryRunConfig, tools map[string]string) error {
	argv := cfg.Incumbent.RestoreCommand
	if len(argv) != 4 || filepath.Base(argv[0]) != "launchctl" || argv[1] != "bootstrap" {
		return errors.New("incumbent.restore_command must be exactly launchctl bootstrap <domain> <restore_plist>")
	}
	wantDomain, _, hasSeparator := strings.Cut(cfg.Incumbent.LaunchdTarget, "/")
	if !hasSeparator || wantDomain == "" {
		return errors.New("incumbent.launchd_target must include a launchd domain and service label")
	}
	// A gui/user target domain contains two components (for example gui/501); the service
	// target contains one more. Keep all components except the final service label.
	if index := strings.LastIndex(cfg.Incumbent.LaunchdTarget, "/"); index > 0 {
		wantDomain = cfg.Incumbent.LaunchdTarget[:index]
	}
	if argv[2] != wantDomain || filepath.Clean(argv[3]) != filepath.Clean(cfg.Incumbent.RestorePlist) {
		return errors.New("incumbent.restore_command domain/plist does not match launchd_target and restore_plist")
	}
	resolved := tools["restore"]
	if resolved == "" {
		resolved = tools["launchctl"]
	}
	if filepath.Clean(resolved) != filepath.Clean(tools["launchctl"]) {
		return errors.New("incumbent.restore_command does not resolve to the preflighted launchctl executable")
	}
	return nil
}

func (d *darwinModelCanaryRuntime) verifyRestoreInputs(cfg modelCanaryRunConfig) error {
	tools := d.getTools()
	if err := validateDarwinModelCanaryRestore(cfg, tools); err != nil {
		return err
	}
	for _, name := range []string{"launchctl", "restore"} {
		if _, err := d.verifyTool(name); err != nil {
			return err
		}
	}
	plist, err := os.ReadFile(cfg.Incumbent.RestorePlist)
	if err != nil {
		return fmt.Errorf("read declared restore plist: %w", err)
	}
	if !sameModelCanarySHA256(digestBytes(plist), cfg.Incumbent.RestorePlistSHA256) {
		return errors.New("declared restore plist SHA256 changed after preflight")
	}
	return nil
}

func (d *darwinModelCanaryRuntime) verifyTool(name string) (string, error) {
	d.mu.Lock()
	path := d.tools[name]
	want := d.executableSHA[name]
	d.mu.Unlock()
	if path == "" || !validSHA256(want) {
		return "", fmt.Errorf("Darwin tool %s was not preflighted with an executable hash", name)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("rehash preflighted executable %s: %w", name, err)
	}
	if !sameModelCanarySHA256(digestBytes(raw), want) {
		return "", fmt.Errorf("preflighted executable %s changed before use", name)
	}
	return path, nil
}

func (d *darwinModelCanaryRuntime) runTool(ctx context.Context, name string, args ...string) ([]byte, error) {
	path, err := d.verifyTool(name)
	if err != nil {
		return nil, err
	}
	cmd := exec.CommandContext(ctx, path, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%s %s: %w: %s", path, strings.Join(args, " "), err, strings.TrimSpace(stderr.String()))
	}
	return stdout.Bytes(), nil
}

func (d *darwinModelCanaryRuntime) setTools(tools, executableSHA map[string]string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.tools = make(map[string]string, len(tools))
	for name, path := range tools {
		d.tools[name] = path
	}
	d.executableSHA = make(map[string]string, len(executableSHA))
	for name, digest := range executableSHA {
		d.executableSHA[name] = digest
	}
}

func (d *darwinModelCanaryRuntime) getTools() map[string]string {
	d.mu.Lock()
	defer d.mu.Unlock()
	copy := make(map[string]string, len(d.tools))
	for name, path := range d.tools {
		copy[name] = path
	}
	return copy
}

func probeModelCanaryEndpoints(ctx context.Context, endpoints []string) error {
	client := &http.Client{
		Timeout:       5 * time.Second,
		CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return http.ErrUseLastResponse },
	}
	for _, endpoint := range endpoints {
		request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
		if err != nil {
			return err
		}
		response, err := client.Do(request)
		if err != nil {
			return fmt.Errorf("probe %s: %w", endpoint, err)
		}
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
		closeErr := response.Body.Close()
		if response.StatusCode < 200 || response.StatusCode >= 300 {
			return fmt.Errorf("probe %s returned HTTP %d", endpoint, response.StatusCode)
		}
		if closeErr != nil {
			return fmt.Errorf("close probe %s: %w", endpoint, closeErr)
		}
	}
	return nil
}

func sleepModelCanaryContext(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func sameModelCanarySHA256(left, right string) bool {
	left = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(left)), "sha256:")
	right = strings.TrimPrefix(strings.ToLower(strings.TrimSpace(right)), "sha256:")
	return left != "" && left == right
}
